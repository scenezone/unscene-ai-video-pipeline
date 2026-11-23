"""Default production entry point: outbound-only Unscene AI worker."""
import logging
import math
import os
import random
import signal
import socket
import threading
from dataclasses import dataclass, field

import requests
from pydantic import ValidationError

from schemas.unscene import ClaimedJob
from services.unscene_client import PlatformError, UnsceneClient
from services.unscene_errors import LeaseLost, ProcessingFailure, SubmissionUncertain, WorkerStopped
from services.unscene_lease import LeaseGuard, STALE_CODES

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerSettings:
    base_url: str
    client_id: str
    client_secret: str = field(repr=False)
    worker_id: str
    steps: tuple = ("DETECT_TYPES", "MATCH")
    heartbeat_interval: float = 45
    poll_interval: float = 3
    target_fps: float = 1
    similarity_threshold: float = .55
    single_candidate_threshold: float = .70
    min_margin: float = .05
    min_mask_area_ratio: float = .20
    detector_version: str = "groundingdino-types-1"
    matcher_version: str = "dino-sam-clip-1"
    work_dir: str = "data/unscene"
    output_dir: str = "outputs/unscene"
    max_video_bytes: int = 20 * 1024**3
    max_image_bytes: int = 30 * 1024**2
    max_manifest_bytes: int = 20 * 1024**2

    def __post_init__(self):
        if not self.client_id or not self.client_secret or not self.worker_id:
            raise ValueError("Set UNSCENE_CLIENT_ID, UNSCENE_CLIENT_SECRET and a worker ID")
        if not self.steps or set(self.steps) - {"DETECT_TYPES", "MATCH"}:
            raise ValueError("UNSCENE_STEPS must contain DETECT_TYPES and/or MATCH")
        if not 30 <= self.heartbeat_interval <= 60 or not 2 <= self.poll_interval <= 5:
            raise ValueError("Heartbeat must be 30-60 seconds and idle poll 2-5 seconds")
        if not math.isfinite(self.target_fps) or self.target_fps <= 0:
            raise ValueError("Target FPS must be positive and finite")
        if not 0 <= self.similarity_threshold <= self.single_candidate_threshold <= 1:
            raise ValueError("Require 0 <= similarity threshold <= singleton threshold <= 1")
        if not 0 <= self.min_margin <= 2 or not 0 <= self.min_mask_area_ratio <= 1:
            raise ValueError("Invalid matching margin or mask-area ratio")
        if min(self.max_video_bytes, self.max_image_bytes, self.max_manifest_bytes) <= 0:
            raise ValueError("Asset byte limits must be positive")
        if not self.detector_version or not self.matcher_version:
            raise ValueError("Model versions cannot be empty")

    @classmethod
    def from_env(cls):
        return cls(
            base_url=os.getenv("UNSCENE_BASE_URL", "http://localhost:4380/ai/v1"),
            client_id=os.getenv("UNSCENE_CLIENT_ID", ""), client_secret=os.getenv("UNSCENE_CLIENT_SECRET", ""),
            worker_id=os.getenv("UNSCENE_WORKER_ID", f"{socket.gethostname()}-{os.getpid()}"),
            steps=tuple(s.strip() for s in os.getenv("UNSCENE_STEPS", "DETECT_TYPES,MATCH").split(",") if s.strip()),
            heartbeat_interval=float(os.getenv("UNSCENE_HEARTBEAT_SECONDS", "45")),
            poll_interval=float(os.getenv("UNSCENE_POLL_SECONDS", "3")),
            target_fps=float(os.getenv("AI_TARGET_FPS", "1")),
            similarity_threshold=float(os.getenv("AI_SIMILARITY_THRESHOLD", ".55")),
            single_candidate_threshold=float(os.getenv("AI_SINGLE_CANDIDATE_THRESHOLD", ".70")),
            min_margin=float(os.getenv("AI_MIN_MARGIN", ".05")),
            min_mask_area_ratio=float(os.getenv("AI_MIN_MASK_AREA_RATIO", ".20")),
            detector_version=os.getenv("AI_DETECTOR_VERSION", "groundingdino-types-1"),
            matcher_version=os.getenv("AI_MATCHER_VERSION", "dino-sam-clip-1"),
            work_dir=os.getenv("AI_WORK_DIR", "data/unscene"),
            output_dir=os.getenv("AI_OUTPUT_DIR", "outputs/unscene"),
            max_video_bytes=int(os.getenv("AI_MAX_VIDEO_BYTES", str(20 * 1024**3))),
            max_image_bytes=int(os.getenv("AI_MAX_IMAGE_BYTES", str(30 * 1024**2))),
            max_manifest_bytes=int(os.getenv("AI_MAX_MANIFEST_BYTES", str(20 * 1024**2))),
        )


def create_service(settings):
    from services.unscene_assets import AssetTransfer
    from services.unscene_service import UnsceneJobService

    registry = None

    def pipeline(step):
        nonlocal registry
        from services.model_registry import ModelRegistry
        from services.pipeline_factory import PipelineFactory

        if registry is None:
            registry = ModelRegistry()
        registry.load_for_step(step)
        return PipelineFactory(registry).create_platform(
            target_fps=settings.target_fps, similarity_threshold=settings.similarity_threshold,
            single_candidate_threshold=settings.single_candidate_threshold, min_margin=settings.min_margin,
            min_mask_area_ratio=settings.min_mask_area_ratio, detector_version=settings.detector_version,
            matcher_version=settings.matcher_version)

    return UnsceneJobService(pipeline, AssetTransfer(
        allow_http=settings.base_url.startswith("http://"), max_video_bytes=settings.max_video_bytes,
        max_image_bytes=settings.max_image_bytes, max_manifest_bytes=settings.max_manifest_bytes),
        settings.work_dir, settings.output_dir)


class UnsceneWorker:
    def __init__(self, settings, client=None, service_factory=None, stop_event=None):
        self.settings = settings
        self.client = client or UnsceneClient(settings.base_url, settings.client_id, settings.client_secret)
        self.service_factory = service_factory or (lambda: create_service(settings))
        self.stop_event = stop_event or threading.Event()
        self._service = None

    def stop(self):
        self.stop_event.set()

    def _report(self, guard, failure):
        guard.check()
        guard.submit("fail", dict(code=failure.code, message=failure.message, retryable=failure.retryable))

    def run_once(self):
        raw_job = self.client.claim(self.settings.worker_id, self.settings.steps)
        if raw_job is None:
            return False
        job = ClaimedJob.model_validate(raw_job)
        logger.info("Claimed jobId=%s step=%s", job.jobId, job.step)
        try:
            with LeaseGuard(self.client, job, self.stop_event, self.settings.heartbeat_interval) as guard:
                try:
                    if job.step not in self.settings.steps:
                        raise ProcessingFailure("unsupported_step", "Worker does not support the claimed step", True)
                    if self._service is None:
                        self._service = self.service_factory()
                    action, body = self._service.analyze(guard)
                    acknowledgement = guard.submit(action, body)
                    logger.info("Submitted jobId=%s step=%s nextState=%s nextStep=%s", job.jobId, job.step,
                                acknowledgement.get("state"), acknowledgement.get("step"))
                except ProcessingFailure as failure:
                    logger.warning("Step failed jobId=%s step=%s code=%s retryable=%s",
                                   job.jobId, job.step, failure.code, failure.retryable)
                    self._report(guard, failure)
                except (ValidationError, ValueError):
                    self._report(guard, ProcessingFailure("validation_failed", "AI input/output failed schema validation"))
                except PlatformError as error:
                    if error.code in STALE_CODES:
                        raise LeaseLost("Platform job changed") from None
                    if error.status == 422:
                        self._report(guard, ProcessingFailure("validation_failed", "Platform rejected step result schema"))
                    else:
                        # Auth/scope/protocol errors require operator correction, not inference retries.
                        raise
                except (LeaseLost, WorkerStopped, SubmissionUncertain):
                    raise
                except Exception as error:
                    # Include no raw exception text: URLs, tokens or vendor data can appear there.
                    logger.error("Processing error jobId=%s step=%s exceptionType=%s",
                                 job.jobId, job.step, type(error).__name__)
                    oom = isinstance(error, MemoryError) or "outofmemory" in type(error).__name__.lower()
                    if isinstance(error, RuntimeError) and "out of memory" in str(error).lower():
                        oom = True
                    self._report(guard, ProcessingFailure(
                        "out_of_memory" if oom else "processing_error",
                        "GPU/host memory unavailable" if oom else "AI processing failed; inspect worker diagnostics",
                        retryable=oom or isinstance(error, (requests.RequestException, OSError))))
        except (LeaseLost, WorkerStopped):
            logger.warning("Abandoned work jobId=%s step=%s: lease lost or worker stopping", job.jobId, job.step)
        except SubmissionUncertain:
            logger.warning("Submission uncertain jobId=%s step=%s; awaiting platform recovery", job.jobId, job.step)
        return True

    def run(self):
        failures = 0
        while not self.stop_event.is_set():
            try:
                worked = self.run_once()
                failures = 0
                if not worked:
                    self.stop_event.wait(self.settings.poll_interval)
            except Exception as error:
                failures += 1
                code = error.code if isinstance(error, PlatformError) else type(error).__name__
                logger.error("Worker control call failed code=%s", code)
                # Avoid token rate-limit storms and busy loops during platform outages.
                self.stop_event.wait(min(60, 2 ** min(failures, 6)) + random.random())


def main():
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        settings = WorkerSettings.from_env()
        worker = UnsceneWorker(settings)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    for name in ("SIGINT", "SIGTERM"):
        signal.signal(getattr(signal, name), lambda *_: worker.stop())
    logger.info("Starting Unscene worker workerId=%s steps=%s", settings.worker_id, ",".join(settings.steps))
    worker.run()


if __name__ == "__main__":
    main()

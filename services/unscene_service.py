"""Execute a claimed step; leave authoritative queue/leases/retries to Unscene."""
import hashlib
import logging
import tempfile
import uuid
from pathlib import Path

from services.unscene_client import json_bytes

logger = logging.getLogger(__name__)


class UnsceneJobService:
    def __init__(self, pipeline_factory, assets, work_dir="data/unscene", output_dir="outputs/unscene"):
        self.pipeline_factory, self.assets = pipeline_factory, assets
        self.work_dir, self.output_dir = Path(work_dir), Path(output_dir)

    def _save(self, job, raw):
        # Platform identifiers are never interpolated into filesystem paths.
        folder = self.output_dir / hashlib.sha256(job.jobId.encode()).hexdigest()[:24]
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{job.step.lower()}-{uuid.uuid4().hex}.json"
        partial = path.with_suffix(".part")
        partial.write_bytes(raw)
        partial.replace(path)
        return path

    def analyze(self, guard):
        job = guard.snapshot()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="step-", dir=self.work_dir) as directory:
            guard.check()
            # The guard runs before model initialization, which may take minutes.
            pipeline = self.pipeline_factory(job.step)
            guard.check()
            if job.step == "DETECT_TYPES":
                video = self.assets.video(guard, directory) if job.allowedTypes else None
                requirements = pipeline.detect_types(video, job, guard.check)
                allowed = {kind.code for kind in job.allowedTypes}
                if any(kind.typeCode not in allowed for kind in requirements.types):
                    raise ValueError("Detector returned a type outside allowedTypes")
                raw = json_bytes(requirements.model_dump(mode="json", exclude_none=True))
                self._save(job, raw)
                guard.check()
                return "requirements", requirements.model_dump(mode="json", exclude_none=True)

            manifest = self.assets.manifest(guard, directory)
            catalogs = self.assets.catalogs(manifest, directory, guard)
            video = self.assets.video(guard, directory) if catalogs else None
            cues = pipeline.match(video, job, manifest, catalogs, guard.check)
            cues.validate_context(job, manifest)
            raw = json_bytes(cues.model_dump(mode="json", exclude_none=True))
            self._save(job, raw)
            guard.check()
            result_url = self.assets.upload(raw, guard)
            return "cues", dict(schemaVersion="1.0", modelVersion=cues.modelVersion,
                                cueFileUrl=result_url, sha256=hashlib.sha256(raw).hexdigest(),
                                cueCount=len(cues.cues))

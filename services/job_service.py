import logging
import os
from pathlib import Path
from threading import RLock

from services.callback_service import CallbackService
from services.downloader import Downloader
from postprocessing.range_builder import RangeBuilder
from postprocessing.response_builder import ResponseBuilder

logger = logging.getLogger(__name__)

GENERIC_CATEGORIES = {"", "non-clothing", "object", "objects", "other", "product"}


def describe_items_enabled():
    return os.getenv("AI_DESCRIBE_ITEMS", "true").strip().lower() not in ("0", "false", "no", "off")


def default_describer_factory():
    from models.qwen_vl_loader import QwenVLLoader
    from services.description.item_describer import ItemDescriber

    loader = QwenVLLoader()
    return ItemDescriber(loader.load(), loader.model_name)


class JobService:
    # Protect shared models, including SAM image state, for direct callers too.
    _inference_lock = RLock()

    def __init__(self, pipeline_factory, registry, output_dir="outputs/jobs", prompt_factory=None,
                 describer_factory=None):
        self.pipeline_factory = pipeline_factory
        self.registry = registry
        self.output_dir = Path(output_dir)
        self.prompt_factory = prompt_factory
        self.describer_factory = describer_factory
        self.downloader = Downloader()
        self.range_builder = RangeBuilder()
        self.response_builder = ResponseBuilder()
        self.callback_service = CallbackService()

    def _load_describer(self):
        """Descriptions are optional: an unavailable model never fails the job."""
        if not describe_items_enabled():
            return None
        try:
            return (self.describer_factory or default_describer_factory)()
        except Exception as error:
            logger.error("Item descriptions skipped for this job: %s", error)
            return None

    def _describe(self, describer, video_path, output, object_labels, episode_dir):
        import json
        from services.description.video_item_describer import describe_detections

        if str(self.registry.device).startswith("cuda"):
            import torch
            # Return cached, unused GPU blocks so the Ollama vision model has more room.
            torch.cuda.empty_cache()
        try:
            describe_detections(video_path, output, describer, object_labels)
            episode_dir.mkdir(parents=True, exist_ok=True)
            (episode_dir / "video_timestamps_described.json").write_text(
                json.dumps(output, indent=4, ensure_ascii=False), encoding="utf-8")
        except Exception:
            logger.exception("Item descriptions failed for %s; continuing without them", video_path)

    def analyze(self, payload):
        """Compute a result; a callback retry must never repeat inference."""
        from schemas.request import AnalyzeRequest

        payload = AnalyzeRequest.model_validate(payload).model_dump()
        with self._inference_lock:
            assets = self.downloader.prepare_assets(payload)
            prompt_factory = self.prompt_factory
            if prompt_factory is None:
                from services.prompt_generation.prompt_generator import PromptGenerator
                prompt_factory = PromptGenerator
            generator = prompt_factory(clothing_dir=assets["clothing_dir"],
                                       object_dir=assets["objects_dir"], device=self.registry.device)
            prompts = generator.generate_prompt()
            del generator
            clothing_prompt = prompts.get("clothing_prompt", "")
            object_prompt = prompts.get("object_prompt", "")
            if not any(p["category"] != "clothing" for p in payload["products"]):
                object_prompt = ""
            if not (clothing_prompt.strip() or object_prompt.strip()):
                raise ValueError("No detection prompts generated from the supplied products")
            # Matched object keys are productIds; the product category is the item hint.
            object_labels = {p["productId"]: (p["category"] if p["category"].strip().lower()
                                              not in GENERIC_CATEGORIES else "product")
                             for p in payload["products"] if p["category"] != "clothing"}
            describer = self._load_describer()
            try:
                results = []
                for episode in assets["episodes"]:
                    episode_dir = self.output_dir / payload["jobId"] / episode["episodeId"]
                    pipeline = self.pipeline_factory.create(
                        clothing_prompt=clothing_prompt, object_prompt=object_prompt,
                        objects_folder=assets["objects_catalog"],
                        output_dir=str(episode_dir))
                    output = pipeline.process_video(episode["video_path"])
                    if describer is not None:
                        self._describe(describer, episode["video_path"], output, object_labels, episode_dir)
                    results.append(dict(episodeId=episode["episodeId"],
                                        ranges=self.range_builder.build_episode(output)))
            finally:
                if describer is not None:
                    describer.release()
            return self.response_builder.build(payload["jobId"], payload["seriesId"], results, assets["products"])

    @staticmethod
    def failure_payload(payload):
        return dict(jobId=payload["jobId"], seriesId=payload["seriesId"], status="failed",
                    error=dict(code="PROCESSING_FAILED", message="Job processing failed; consult the AI service logs."))

    def process_job(self, payload):
        """Local compatibility entry point; the HTTP worker persists before delivery."""
        try:
            result = self.analyze(payload)
        except Exception:
            self.callback_service.send(payload["callbackUrl"], self.failure_payload(payload))
            raise
        self.callback_service.send(payload["callbackUrl"], result)
        return result

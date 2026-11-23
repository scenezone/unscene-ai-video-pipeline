from services.detection.detector import GroundingDINODetector
from services.segmentation.segmenter import SAMSegmenter
from services.matching.product_catalog_matcher import ProductCatalogMatcher

from pipelines.video_pipeline import DEFAULT_MIN_MASK_AREA_RATIO, VideoPipeline

class PipelineFactory:

    def __init__(self, registry):
        self.registry = registry

    def create_platform(self, **settings):
        """Both platform stages use the same detector; clothing is product-matched too."""
        from pipelines.unscene_pipeline import UnsceneVideoPipeline

        return UnsceneVideoPipeline(
            detector=GroundingDINODetector(self.registry.dino_model, self.registry.device,
                                          box_threshold=.35, text_threshold=.30),
            segmenter=SAMSegmenter(self.registry.sam_predictor),
            predictor=self.registry.sam_predictor,
            matcher_factory=lambda: ProductCatalogMatcher(
                self.registry.clip_model, self.registry.preprocess, self.registry.device),
            **settings,
        )

    def create(
        self,
        clothing_prompt,
        object_prompt,
        objects_folder,
        crop_mode="hybrid",
        min_margin=0.05,
        min_mask_area_ratio=DEFAULT_MIN_MASK_AREA_RATIO,
        output_dir="outputs/video",
    ):

        detector = GroundingDINODetector(
            self.registry.dino_model,
            self.registry.device,
            box_threshold=0.35,
            text_threshold=0.30,
        )

        segmenter = SAMSegmenter(
            self.registry.sam_predictor
        )

        matcher = ProductCatalogMatcher(
            self.registry.clip_model,
            self.registry.preprocess,
            self.registry.device,
            min_margin=min_margin,
        )

        if object_prompt and object_prompt.strip():
            matcher.load_product_catalog(objects_folder)

        pipeline = VideoPipeline(
            detector=detector,
            segmenter=segmenter,
            matcher=matcher,
            sam_predictor=self.registry.sam_predictor,
            clothing_prompt=clothing_prompt,
            object_prompt=object_prompt,
            max_duration=None,
            crop_mode=crop_mode,
            min_mask_area_ratio=min_mask_area_ratio,
            output_dir=output_dir,
        )

        return pipeline

"""Test-only corrections; production loaders and pipeline remain untouched."""


def load_clip(device, corrected=True):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", force_quick_gelu=corrected,
    )
    return model.to(device).eval(), preprocess


class BGRPredictor:
    """Adapt the pipeline's OpenCV frames to SAM's explicit color contract."""

    def __init__(self, predictor):
        self.predictor = predictor

    def set_image(self, frame):
        return self.predictor.set_image(frame, image_format="BGR")

    def predict(self, **kwargs):
        return self.predictor.predict(**kwargs)


class BoxSegmenter:
    def set_image(self, frame):
        # Satisfy VideoPipeline's predictor interface without loading SAM.
        pass

    def segment(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        return frame[y1:y2, x1:x2].copy()

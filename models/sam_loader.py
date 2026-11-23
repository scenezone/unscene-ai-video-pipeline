import torch
from segment_anything import sam_model_registry, SamPredictor


class SamLoader:

    def __init__(self, checkpoint_path, model_type="vit_h", device="cpu"):
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.device = device

    def load(self):

        print("Loading SAM model...")

        sam = sam_model_registry[self.model_type](
            checkpoint=self.checkpoint_path
        )

        sam.to(device=self.device)

        predictor = SamPredictor(sam)

        return predictor
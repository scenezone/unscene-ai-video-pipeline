import torch
import open_clip


class CLIPLoader:

    def __init__(self, device="cpu"):
        self.device = device

    def load(self):

        print("Loading CLIP model...")

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14",
            pretrained="openai",
            force_quick_gelu=False,  # Explicit pipeline_gelu experiment configuration.
        )

        model = model.to(self.device)
        model.eval()

        return model, preprocess

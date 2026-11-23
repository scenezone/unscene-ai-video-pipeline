import torch
from groundingdino.util.inference import load_model


class DinoLoader:

    def __init__(self, config_path: str, weights_path: str, device: str = "cpu"):
        self.config_path = config_path
        self.weights_path = weights_path
        self.device = device
        self.model = None

    def load(self):
        print("Loading GroundingDINO model...")
        self.model = load_model(self.config_path, self.weights_path, device=self.device)
        self.model = self.model.to(self.device)
        self.model.eval()
        return self.model

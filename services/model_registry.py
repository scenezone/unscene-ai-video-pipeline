import torch
from models.dino_loader import DinoLoader
from models.sam_loader import SamLoader
from models.clip_loader import CLIPLoader

class ModelRegistry:

    def __init__(self):

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.dino_model = None
        self.sam_predictor = None
        self.clip_model = None
        self.preprocess = None

    def load_models(self):
        self.load_for_step("MATCH")

    def load_for_step(self, step):
        if step not in ("DETECT_TYPES", "MATCH"):
            raise ValueError("Unknown platform step")
        if self.dino_model is None:
            self._load_detector()
        if step == "MATCH":
            if self.sam_predictor is None:
                self.sam_predictor = SamLoader(
                    "weights/sam-weight/sam_vit_h_4b8939.pth", "vit_h", self.device).load()
            if self.clip_model is None:
                self.clip_model, self.preprocess = CLIPLoader(self.device).load()

    def _load_detector(self):

        print("\nLoading GroundingDINO model...")
    
        self.dino_model = DinoLoader(
            "weights/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "weights/GroundingDINO/weights/groundingdino_swint_ogc.pth",
            self.device
        ).load()

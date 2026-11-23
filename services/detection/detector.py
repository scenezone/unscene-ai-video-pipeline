import cv2
import torch
from typing import List
from PIL import Image

from groundingdino.util.inference import predict
from groundingdino.datasets.transforms import Compose, RandomResize, ToTensor, Normalize

from .schemas import Detection


class GroundingDINODetector:

    def __init__(
        self,
        model,
        device: str = "cpu",
        box_threshold: float = 0.35,
        text_threshold: float = 0.30,
    ):
        self.model = model
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # GroundingDINO image transform
        self.transform = Compose([
            RandomResize([800], max_size=1333),
            ToTensor(),
            Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            ),
        ])

    def detect(self, frame, prompt: str) -> List[Detection]:

        # Convert OpenCV BGR → RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL
        image_pil = Image.fromarray(image_rgb)

        # Apply GroundingDINO transform
        image_transformed, _ = self.transform(image_pil, None)

        boxes, logits, phrases = predict(
            model=self.model,
            image=image_transformed,
            caption=prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
        )

        detections = []

        for box, logit, phrase in zip(boxes, logits, phrases):
            detections.append(
                Detection(
                    bbox=box.tolist(),
                    score=float(logit),
                    label=phrase
                )
            )

        return detections
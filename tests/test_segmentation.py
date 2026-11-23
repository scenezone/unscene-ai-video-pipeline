import sys
import cv2
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.dino_loader import DinoLoader
from models.sam_loader import SamLoader

from services.detection.detector import GroundingDINODetector
from services.segmentation.segmenter import SAMSegmenter


# ----------------
# CONFIG
# ----------------

CONFIG_PATH = r"weights\GroundingDINO\groundingdino\config\GroundingDINO_SwinT_OGC.py"
DINO_WEIGHTS = r"weights\GroundingDINO\weights\groundingdino_swint_ogc.pth"

SAM_WEIGHTS = r"weights\sam-weight\sam_vit_h_4b8939.pth"

IMAGE_PATH = r"outputs\detection\frame_420.jpg"

DEVICE = "cpu"

PROMPT = "handbags"


OUTPUT_DIR = "outputs/segmentation"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------
# LOAD MODELS
# ----------------

dino_loader = DinoLoader(CONFIG_PATH, DINO_WEIGHTS, DEVICE)
dino_model = dino_loader.load()

sam_loader = SamLoader(SAM_WEIGHTS, "vit_h", DEVICE)
sam_predictor = sam_loader.load()

detector = GroundingDINODetector(dino_model, DEVICE)
segmenter = SAMSegmenter(sam_predictor)


# ----------------
# LOAD IMAGE
# ----------------

frame = cv2.imread(IMAGE_PATH)

if frame is None:
    raise RuntimeError("Image not found")


# ----------------
# DETECT
# ----------------

detections = detector.detect(frame, PROMPT)

h, w = frame.shape[:2]

sam_predictor.set_image(frame)


for i, det in enumerate(detections):

    cx, cy, bw, bh = det.bbox

    x1 = int((cx - bw/2)*w)
    y1 = int((cy - bh/2)*h)
    x2 = int((cx + bw/2)*w)
    y2 = int((cy + bh/2)*h)

    bbox = (x1, y1, x2, y2)

    segmented = segmenter.segment(frame, bbox)

    output_path = os.path.join(OUTPUT_DIR, f"seg_{i}.png")

    cv2.imwrite(output_path, segmented)

    print("Saved:", output_path)
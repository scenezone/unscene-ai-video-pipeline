import sys
import cv2
import os
from pathlib import Path

# Allow imports from project root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.dino_loader import DinoLoader
from services.detection.detector import GroundingDINODetector


# -------------------------
# CONFIG
# -------------------------

CONFIG_PATH = r"weights\GroundingDINO\groundingdino\config\GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = r"weights\GroundingDINO\weights\groundingdino_swint_ogc.pth"

VIDEO_PATH = r"data\testdata\Episode 2.mp4"

OUTPUT_DIR = r"outputs\detection"
CROP_DIR = r"outputs\crops"

DEVICE = "cpu"
PROMPT = "earrings. handbag."

TARGET_FPS = 1   # frames per second to process


# -------------------------
# CREATE OUTPUT DIRS
# -------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)


# -------------------------
# LOAD MODEL
# -------------------------

loader = DinoLoader(CONFIG_PATH, WEIGHTS_PATH, DEVICE)
model = loader.load()

detector = GroundingDINODetector(model, DEVICE)


# -------------------------
# LOAD VIDEO
# -------------------------

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Failed to open video")


video_fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = max(1, int(video_fps / TARGET_FPS))

print("Video FPS:", video_fps)
print("Processing every", frame_interval, "frames")


frame_idx = 0
crop_counter = 0


# -------------------------
# PROCESS VIDEO
# -------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    if frame_idx % frame_interval != 0:
        frame_idx += 1
        continue

    print(f"\nProcessing frame {frame_idx}")

    detections = detector.detect(frame, PROMPT)

    if len(detections) == 0:
        frame_idx += 1
        continue

    h, w = frame.shape[:2]

    annotated_frame = frame.copy()

    for i, det in enumerate(detections):

        cx, cy, bw, bh = det.bbox

        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1,y1),(x2,y2),(0,255,0),2)

        cv2.putText(
            annotated_frame,
            f"{det.label} {det.score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

        # Save crop
        crop = frame[y1:y2, x1:x2]

        if crop.size != 0:

            crop_path = os.path.join(
                CROP_DIR,
                f"frame_{frame_idx}_crop_{crop_counter}.jpg"
            )

            cv2.imwrite(crop_path, crop)

            print(f"Saved crop → {crop_path}")

            crop_counter += 1


    # Save annotated frame
    output_frame = os.path.join(
        OUTPUT_DIR,
        f"frame_{frame_idx}.jpg"
    )

    cv2.imwrite(output_frame, annotated_frame)

    print(f"Saved frame → {output_frame}")

    frame_idx += 1


cap.release()

print("\nVideo processing complete.")
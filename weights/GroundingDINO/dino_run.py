import torch
import cv2
from groundingdino.util.inference import load_model, load_image, predict, annotate

# -----------------------------
# CONFIG
# -----------------------------

CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = "GroundingDINO/weights/groundingdino_swint_ogc.pth"
IMAGE_PATH = "GroundingDINO/test.jpg"
OUTPUT_PATH = "GroundingDINO/outputs/result.jpg"
TEXT_PROMPT = "earring. handbag."

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

# -----------------------------
# CHECK GPU
# -----------------------------

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# -----------------------------
# LOAD MODEL
# -----------------------------

print("\nLoading GroundingDINO model...")

model = load_model(CONFIG_PATH, WEIGHTS_PATH)

# move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print("Model loaded on:", device)

# -----------------------------
# LOAD IMAGE
# -----------------------------

print("\nLoading image...")

image_source, image = load_image(IMAGE_PATH)

# -----------------------------
# RUN DETECTION
# -----------------------------

print("\nRunning object detection...")

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD
)

print("\nDetected objects:")
for phrase, score in zip(phrases, logits):
    print(f"{phrase} : {score:.2f}")

# -----------------------------
# DRAW BOXES
# -----------------------------

annotated_frame = annotate(
    image_source=image_source,
    boxes=boxes,
    logits=logits,
    phrases=phrases
)

cv2.imwrite(OUTPUT_PATH, annotated_frame)

print("\nResult saved to:", OUTPUT_PATH)
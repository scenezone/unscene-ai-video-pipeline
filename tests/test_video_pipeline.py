import sys
from pathlib import Path
import json
import os
import torch

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.dino_loader import DinoLoader
from models.sam_loader import SamLoader
from models.clip_loader import CLIPLoader

from services.detection.detector import GroundingDINODetector
from services.segmentation.segmenter import SAMSegmenter
from services.matching.product_catalog_matcher import ProductCatalogMatcher
from services.prompt_generation.prompt_generator import PromptGenerator

from pipelines.video_pipeline import VideoPipeline


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

VIDEO_PATH = str(PROJECT_ROOT / "data/testdata/Episode 2.mp4")

CLOTHING_FOLDER = str(PROJECT_ROOT / "data/productimages/clothing")
OBJECT_FOLDER = str(PROJECT_ROOT / "data/productimages/objects")

PROMPT_CACHE_FILE = str(PROJECT_ROOT / "config/prompt_cache.json")


def weight_path(relative_path):
    # This temp checkout shares the large checkpoints with the main checkout.
    roots = (PROJECT_ROOT, PROJECT_ROOT.parent.parent / "unscene-ai-video-pipeline")
    for root in roots:
        candidate = root / "weights" / relative_path
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        f"Missing model file {relative_path}. Checked: "
        + ", ".join(str(root / "weights" / relative_path) for root in roots)
    )


DINO_CONFIG = weight_path("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
DINO_WEIGHTS = weight_path("GroundingDINO/weights/groundingdino_swint_ogc.pth")
SAM_WEIGHTS = weight_path("sam-weight/sam_vit_h_4b8939.pth")

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# --------------------------------------------------
# LOAD OR GENERATE PROMPTS
# --------------------------------------------------

print("\nPreparing prompts...\n")

os.makedirs(Path(PROMPT_CACHE_FILE).parent, exist_ok=True)

if os.path.exists(PROMPT_CACHE_FILE):

    print("Loading cached prompts")

    with open(PROMPT_CACHE_FILE, "r") as f:
        prompt_data = json.load(f)

else:

    print("Generating prompts using Florence + Gemini")

    generator = PromptGenerator(
        clothing_dir=CLOTHING_FOLDER,
        object_dir=OBJECT_FOLDER,
        device=DEVICE
    )

    prompt_data = generator.generate_prompt()

    with open(PROMPT_CACHE_FILE, "w") as f:
        json.dump(prompt_data, f, indent=4)

    print("Prompt cache saved →", PROMPT_CACHE_FILE)


# --------------------------------------------------
# BUILD PROMPTS
# --------------------------------------------------

clothing_prompt = prompt_data.get("clothing_prompt", "")
object_prompt = prompt_data.get("object_prompt", "")

print("\nGenerated Prompts")
print("------------------")
print("Clothing Prompt:", clothing_prompt)
print("Object Prompt:", object_prompt)

if clothing_prompt == "" and object_prompt == "":
    raise RuntimeError("No prompts generated. Check product images.")


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

print("\nLoading GroundingDINO model...")

dino_model = DinoLoader(
    DINO_CONFIG,
    DINO_WEIGHTS,
    DEVICE
).load()


print("\nLoading SAM model...")

sam_predictor = SamLoader(
    SAM_WEIGHTS,
    "vit_h",
    DEVICE
).load()


print("\nLoading CLIP model...")

clip_model, preprocess = CLIPLoader(DEVICE).load()


# --------------------------------------------------
# INITIALIZE SERVICES
# --------------------------------------------------

print("\nInitializing services...\n")

detector = GroundingDINODetector(dino_model, DEVICE)

segmenter = SAMSegmenter(sam_predictor)

matcher = ProductCatalogMatcher(clip_model, preprocess, DEVICE, min_margin=0.05)

# IMPORTANT: load only object catalog for matching
matcher.load_product_catalog(OBJECT_FOLDER)


# --------------------------------------------------
# INITIALIZE PIPELINE
# --------------------------------------------------

print("Initializing video pipeline...\n")

pipeline = VideoPipeline(
    detector=detector,
    segmenter=segmenter,
    matcher=matcher,
    sam_predictor=sam_predictor,
    clothing_prompt=clothing_prompt,
    object_prompt=object_prompt,
    max_duration=None
)


# --------------------------------------------------
# RUN PIPELINE
# --------------------------------------------------

print("Starting video processing...\n")

pipeline.process_video(VIDEO_PATH)

print("\nPipeline finished successfully.")


# --------------------------------------------------
# TEST ONLY: DESCRIBE ITEMS WITH QWEN3-VL
# --------------------------------------------------
# Adds a colour/pattern/material description to every detection in a
# separate file (video_timestamps_described.json). Set to False to skip.

DESCRIBE_ITEMS = True

if DESCRIBE_ITEMS:
    from models.qwen_vl_loader import QwenVLLoader
    from services.description.item_describer import ItemDescriber
    from describe_video_items import describe_video_items

    product_names = {ident: meta["name"] for ident, meta in getattr(matcher, "product_metadata", {}).items()}
    output_dir = pipeline.output_dir

    # Free this process's GPU memory so Ollama can keep Qwen3-VL fully on the GPU.
    del pipeline, matcher, segmenter, detector, sam_predictor, clip_model, dino_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nDescribing detected items...\n")
    qwen_loader = QwenVLLoader()
    describer = ItemDescriber(qwen_loader.load(), qwen_loader.model_name)
    try:
        describe_video_items(VIDEO_PATH, output_dir, describer, product_names)
    finally:
        describer.release()

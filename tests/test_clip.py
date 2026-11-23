"""Run CLIP matching against a supplied object crop."""
import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PRODUCT_FOLDER = PROJECT_ROOT / "data/productimages/objects"
TEST_SEGMENT = PROJECT_ROOT / "outputs/segmentation/seg_0.png"
DEVICE = "cpu"


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_crop(image_path):
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Crop image not found: {image_path}. "
            "Pass --image with an existing object crop or generate segmentation "
            "output first (filenames start at seg_0.png)."
        )
    crop = cv2.imread(str(image_path))
    if crop is None or crop.size == 0:
        raise ValueError(f"Cannot decode crop image: {image_path}")
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=project_path, default=TEST_SEGMENT,
                        help="Object crop; relative paths resolve from the project root")
    parser.add_argument("--products", type=project_path, default=PRODUCT_FOLDER,
                        help="Folder containing catalog images directly")
    parser.add_argument("--device", default=DEVICE, choices=("cpu", "cuda"))
    args = parser.parse_args(argv)

    try:
        crop_rgb = load_crop(args.image)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    if not args.products.is_dir():
        parser.error(f"Product folder not found: {args.products}")
    if not any(path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
               for path in args.products.iterdir()):
        parser.error(f"No catalog images directly inside: {args.products}")

    # Validate input before importing and loading the expensive model.
    from models.clip_loader import CLIPLoader
    from services.matching.clip_matcher import CLIPMatcher

    clip_model, preprocess = CLIPLoader(args.device).load()
    matcher = CLIPMatcher(clip_model, preprocess, args.device)
    matcher.load_product_catalog(args.products)
    if not matcher.product_names:
        parser.error(f"No readable catalog images in: {args.products}")

    matched, product, score = matcher.match(crop_rgb)
    print("\nResult")
    print("Matched:", matched)
    print("Product:", product)
    print("Similarity:", score)


if __name__ == "__main__":
    main()

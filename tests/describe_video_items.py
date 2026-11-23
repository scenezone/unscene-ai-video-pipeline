"""TEST ONLY: add a Qwen3-VL description to every detection of a test pipeline run.

Reads video_timestamps.json written by VideoPipeline and writes
video_timestamps_described.json with exactly the same structure, plus a
"description" (colour, pattern, material...) on every object and clothing entry:

    {"objects":  {"OBJ005": [{"time": 19.0, "bbox": [...], "description": {...}}]},
     "clothing": {"pants":  [{"time": 11.0, "bbox": [...], "description": {...}}]}}

Each description is made from that detection's own box crop in that frame. The
pipeline output files are not changed. The main API uses the same code
(services/description/video_item_describer.py).

Run after tests/test_video_pipeline.py (which also calls it at the end), or alone:
    python tests/describe_video_items.py
    python tests/describe_video_items.py --video "data/testdata/Episode 1.mp4" --object-folder data/productimages/objects1
"""
import argparse
import copy
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.qwen_vl_loader import QwenVLLoader
from services.description.item_describer import ItemDescriber
from services.description.video_item_describer import describe_detections


def product_names_from_folder(object_folder):
    folder = Path(object_folder)
    folder = folder / "by_product" if (folder / "by_product").is_dir() else folder
    names = {}
    if folder.is_dir():
        for directory in folder.iterdir():
            match = re.fullmatch(r"(OBJ\d+)_(.+)", directory.name)
            if directory.is_dir() and match:
                names[match.group(1)] = match.group(2).replace("_", " ")
    return names


def describe_video_items(video_path, output_dir, describer, product_names=None):
    output_dir = Path(output_dir)
    timestamps = json.loads((output_dir / "video_timestamps.json").read_text(encoding="utf-8"))
    result = copy.deepcopy(timestamps)
    describe_detections(video_path, result, describer, product_names, crop_dir=output_dir / "described_crops")
    target = output_dir / "video_timestamps_described.json"
    target.write_text(json.dumps(result, indent=4, ensure_ascii=False), encoding="utf-8")
    print("Description JSON saved:", target)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default=str(PROJECT_ROOT / "data/testdata/Episode 2.mp4"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs/video"),
                        help="Folder holding video_timestamps.json from the test pipeline run")
    parser.add_argument("--object-folder", default=str(PROJECT_ROOT / "data/productimages/objects"),
                        help="Catalog folder used to turn OBJnnn IDs into readable names")
    parser.add_argument("--model", default=None, help="Ollama model (default AI_DESCRIBE_MODEL or qwen3-vl:4b-instruct)")
    args = parser.parse_args()

    loader = QwenVLLoader(args.model)
    describer = ItemDescriber(loader.load(), loader.model_name)
    try:
        describe_video_items(args.video, args.output_dir, describer, product_names_from_folder(args.object_folder))
    finally:
        describer.release()


if __name__ == "__main__":
    main()

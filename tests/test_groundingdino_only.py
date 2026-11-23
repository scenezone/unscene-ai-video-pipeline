"""Detect clothing and objects from prompt JSON using only GroundingDINO."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_prompts(path):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("Prompt file must contain a JSON object")
    prompts = {}
    for category, key in (("clothing", "clothing_prompt"), ("objects", "object_prompt")):
        value = data.get(key, "")
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        if value.strip():
            prompts[category] = value
    if not prompts:
        raise ValueError("Both clothing_prompt and object_prompt are empty")
    return prompts


def annotate(frame, detections):
    image = frame.copy()
    height, width = frame.shape[:2]
    results = []
    for det in detections:
        if not all(math.isfinite(v) for v in (*det.bbox, det.score)):
            continue
        cx, cy, bw, bh = det.bbox
        x1, x2 = [max(0, min(width, int(v * width))) for v in (cx - bw / 2, cx + bw / 2)]
        y1, y2 = [max(0, min(height, int(v * height))) for v in (cy - bh / 2, cy + bh / 2)]
        if x2 <= x1 or y2 <= y1:
            continue
        results.append(dict(label=det.label, confidence=det.score, bbox=[x1, y1, x2, y2]))
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), 2)
        # Preserve the returned label in JSON; explicitly mark an empty phrase visually.
        text = f"{det.label or '[no text label]'} {det.score:.3f}"
        cv2.putText(image, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    .6, (0, 220, 0), 2)
    if not results:
        cv2.putText(image, "No detections above threshold", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 200, 255), 2)
    return image, results


def run(args, detector, prompts):
    cap = cv2.VideoCapture(str(args.video))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Cannot read video: {args.video}")
        destination = args.output / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        for category in prompts:
            (destination / category).mkdir(parents=True)
        report = dict(status="running", model="GroundingDINO", video=str(args.video),
                      prompt_file=str(args.prompts), prompts=prompts, device=args.device,
                      box_threshold=args.box_threshold, text_threshold=args.text_threshold,
                      source_fps=fps, requested_fps=args.fps, max_frames=args.max_frames,
                      note="Full prompt per category; labels and confidence are returned by GroundingDINO. Confidence is not accuracy.",
                      frames=[])

        def save():
            temp = destination / "detections.json.part"
            temp.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            temp.replace(destination / "detections.json")

        index = sampled = 0
        save()
        while args.max_frames is None or sampled < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = index / fps
            if timestamp + 1e-9 >= sampled / min(args.fps, fps):
                entry = dict(frame=index, timestamp=timestamp, categories={})
                for category, prompt in prompts.items():
                    image, results = annotate(frame, detector.detect(frame, prompt))
                    relative = f"{category}/frame_{index:08d}.jpg"
                    if not cv2.imwrite(str(destination / relative), image):
                        raise OSError(f"Could not save {relative}")
                    entry["categories"][category] = dict(frame_path=relative, detections=results)
                report["frames"].append(entry)
                sampled += 1
                save()
                print(f"Saved frame {index} at {timestamp:.2f}s ({sampled} sampled)", flush=True)
            index += 1
        if not sampled:
            raise ValueError("No readable video frames")
        report["status"] = "complete"
        save()
        print(f"GroundingDINO results: {destination}", flush=True)
        return destination
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=project_path, default=ROOT / "data/testdata/Episode 2.mp4")
    parser.add_argument("--prompts", type=project_path, default=ROOT / "config/prompt_cache.json")
    parser.add_argument("--output", type=project_path, default=ROOT / "outputs/groundingdino_only")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--box-threshold", type=float, default=.35)
    parser.add_argument("--text-threshold", type=float, default=.30)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be positive and finite")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    if not all(math.isfinite(t) and 0 <= t <= 1 for t in (args.box_threshold, args.text_threshold)):
        parser.error("Thresholds must be between 0 and 1")
    try:
        prompts = read_prompts(args.prompts)
        if not args.video.is_file():
            raise ValueError(f"Missing video: {args.video}")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"GroundingDINO only; input prompts: {prompts}", flush=True)
    import torch
    from models.dino_loader import DinoLoader
    from services.detection.detector import GroundingDINODetector
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DinoLoader(str(ROOT / "weights/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"),
                       str(ROOT / "weights/GroundingDINO/weights/groundingdino_swint_ogc.pth"), args.device).load()
    detector = GroundingDINODetector(model, args.device, args.box_threshold, args.text_threshold)
    run(args, detector, prompts)


if __name__ == "__main__":
    main()

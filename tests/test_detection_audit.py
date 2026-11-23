"""Audit GroundingDINO confidence for clothing and objects, independently of the pipeline."""
import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIELDS = ["frame", "timestamp", "category", "prompt_label", "detected_label", "status",
          "confidence", "band", "x1", "y1", "x2", "y2", "frame_path"]
COLORS = {"low": (0, 0, 255), "medium": (0, 165, 255), "high": (0, 200, 0)}


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path



def parse_prompts(data):
    if not isinstance(data, dict):
        raise ValueError("Prompt JSON must be an object")
    prompts = []
    for category, key in (("clothing", "clothing_prompt"), ("objects", "object_prompt")):
        value = data.get(key, "")
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        labels = dict.fromkeys(part.strip() for part in value.split(".") if part.strip())
        prompts.extend((category, label) for label in labels)
    if not prompts:
        raise ValueError("clothing_prompt and object_prompt are both empty or missing")
    return prompts


def confidence_band(score, medium, high):
    return "high" if score >= high else "medium" if score >= medium else "low"


def audit_frame(frame, detector, prompts, medium, high):
    """Query each clothing or object label separately so missing predictions are recorded explicitly."""
    annotated = frame.copy()
    height, width = frame.shape[:2]
    rows = []
    for category, name in prompts:
        found = False
        for detection in detector.detect(frame, name + " ."):
            cx, cy, bw, bh = detection.bbox
            if not all(math.isfinite(v) for v in (*detection.bbox, detection.score)):
                continue
            x1 = max(0, min(width, int((cx - bw / 2) * width)))
            y1 = max(0, min(height, int((cy - bh / 2) * height)))
            x2 = max(0, min(width, int((cx + bw / 2) * width)))
            y2 = max(0, min(height, int((cy + bh / 2) * height)))
            if x2 <= x1 or y2 <= y1:
                continue
            found = True
            band = confidence_band(detection.score, medium, high)
            rows.append(dict(category=category, prompt_label=name, detected_label=detection.label,
                             status="detected", confidence=detection.score, band=band,
                             x1=x1, y1=y1, x2=x2, y2=y2))
            color = COLORS[band]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{category}/{name}: {detection.score:.3f} {band.upper()}",
                        (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        if not found:
            rows.append(dict(category=category, prompt_label=name, status="no_detection"))
    missing = [f"{row['category']}/{row['prompt_label']}" for row in rows if row["status"] == "no_detection"]
    # A footer preserves the frame pixels and lists prompts without predictions.
    footer = ["Model confidence (not measured accuracy)"]
    footer.extend(f"No detection above threshold: {name}" for name in missing)
    annotated = cv2.copyMakeBorder(annotated, 0, 26 * len(footer) + 8, 0, 0,
                                   cv2.BORDER_CONSTANT, value=(25, 25, 25))
    for index, label in enumerate(footer):
        cv2.putText(annotated, label, (8, height + 22 + 26 * index),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return annotated, rows


def run_audit(args, detector, prompts):
    cap = cv2.VideoCapture(str(args.video))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {args.video}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video has no valid frame rate")
        # A fresh directory for each run avoids mixing results with older reports.
        run_dir = args.output / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        frames_dir = run_dir / "frames"
        frames_dir.mkdir(parents=True)
        stats = {category: {} for category, _ in prompts}
        for category, name in prompts:
            stats[category][name] = dict(low=0, medium=0, high=0, frames_with_detection=0,
                                         frames_without_detection=0)
        sampled = index = 0
        with (run_dir / "detections.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            while args.max_frames is None or sampled < args.max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp = index / fps
                if timestamp + 1e-9 >= sampled / min(args.fps, fps):
                    annotated, rows = audit_frame(frame, detector, prompts,
                                                  args.medium, args.high)
                    relative_frame = f"frames/frame_{index:08d}.jpg"
                    if not cv2.imwrite(str(run_dir / relative_frame), annotated):
                        raise OSError(f"Could not save {relative_frame}")
                    detected_names = set()
                    for row in rows:
                        writer.writerow(dict(frame=index, timestamp=round(timestamp, 6),
                                             frame_path=relative_frame, **row))
                        if row["status"] == "detected":
                            stats[row["category"]][row["prompt_label"]][row["band"]] += 1
                            detected_names.add((row["category"], row["prompt_label"]))
                    for category, name in prompts:
                        key = "frames_with_detection" if (category, name) in detected_names else "frames_without_detection"
                        stats[category][name][key] += 1
                    handle.flush()
                    sampled += 1
                    print(f"Saved frame {index} at {timestamp:.2f}s ({sampled} sampled)", flush=True)
                index += 1
        if not sampled:
            raise ValueError("Video contains no readable frames")
        summary = dict(video=str(args.video), prompt_file=str(args.prompts), prompts={category: list(labels) for category, labels in stats.items()},
                       mode="Each clothing or object label is queried separately; scores may differ from the pipeline's combined prompt.",
                       interpretation="Confidence is not measured accuracy. No detection does not establish absence.",
                       box_threshold=args.box_threshold, text_threshold=args.text_threshold,
                       bands=dict(low=f"< {args.medium}", medium=f">= {args.medium} and < {args.high}",
                                  high=f">= {args.high}"),
                       requested_fps=args.fps, source_fps=fps, sampled_frames=sampled,
                       max_frames=args.max_frames, device=args.device, counts=stats)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Audit saved to: {run_dir}")
        return run_dir
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=project_path, default=ROOT / "data/testdata/Episode 2.mp4")
    parser.add_argument("--prompts", type=project_path, default=ROOT / "config/prompt_cache.json")
    parser.add_argument("--output", type=project_path, default=ROOT / "outputs/detection_audit")
    parser.add_argument("--fps", type=float, default=1.0, help="Sampled frames per second")
    parser.add_argument("--max-frames", type=int, help="Limit sampled frames for a quick check")
    parser.add_argument("--box-threshold", type=float, default=0.10)
    parser.add_argument("--text-threshold", type=float, default=0.30)
    parser.add_argument("--medium", type=float, default=0.30)
    parser.add_argument("--high", type=float, default=0.60)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be positive and finite")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    if not 0 <= args.box_threshold < args.medium < args.high <= 1:
        parser.error("Require 0 <= box-threshold < medium < high <= 1")
    if not 0 <= args.text_threshold <= 1:
        parser.error("--text-threshold must be between 0 and 1")
    try:
        data = json.loads(args.prompts.read_text(encoding="utf-8-sig"))
        prompts = parse_prompts(data)
        if not args.video.is_file():
            raise ValueError(f"Video not found: {args.video}")
    except (OSError, ValueError) as error:
        parser.error(str(error))

    import torch
    from models.dino_loader import DinoLoader
    from services.detection.detector import GroundingDINODetector

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Auditing {prompts} separately on {args.device}; scores are confidence, not accuracy.", flush=True)
    model = DinoLoader(str(ROOT / "weights/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"),
                       str(ROOT / "weights/GroundingDINO/weights/groundingdino_swint_ogc.pth"),
                       args.device).load()
    detector = GroundingDINODetector(model, args.device, args.box_threshold, args.text_threshold)
    run_audit(args, detector, prompts)


if __name__ == "__main__":
    main()

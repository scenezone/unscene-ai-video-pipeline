"""Compare box thresholds on identical video frames without changing the pipeline."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path

import cv2

from test_detection_audit import ROOT, audit_frame, parse_prompts, project_path


class CachedDetector:
    def __init__(self, predictions, threshold):
        self.predictions = predictions
        self.threshold = threshold

    def detect(self, frame, prompt):
        # GroundingDINO uses strict >, not >=, for its box score filter.
        return [item for item in self.predictions[prompt] if item.score > self.threshold]


def summarize(frames, prompts):
    counts = {category: {} for category, _ in prompts}
    total = 0
    for category, label in prompts:
        detections = [row for frame in frames for row in frame["results"]
                      if row["category"] == category and row["prompt_label"] == label
                      and row["status"] == "detected"]
        scores = [row["confidence"] for row in detections]
        present = sum(any(row["category"] == category and row["prompt_label"] == label
                          and row["status"] == "detected" for row in frame["results"])
                      for frame in frames)
        counts[category][label] = dict(
            detections=len(scores), frames_with_detection=present,
            frames_without_detection=len(frames) - present,
            frame_coverage=present / len(frames) if frames else 0,
            mean_confidence=sum(scores) / len(scores) if scores else None,
            min_confidence=min(scores) if scores else None,
            max_confidence=max(scores) if scores else None,
            bands={band: sum(row["band"] == band for row in detections)
                   for band in ("low", "medium", "high")})
        total += len(scores)
    return dict(total_detections=total, by_category=counts)


def experiment(args, detector, prompts):
    cap = cv2.VideoCapture(str(args.video))
    try:
        count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or count <= 0 or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Cannot read valid video metadata: {args.video}")
        samples = min(args.samples, count)
        indices = [round(i * (count - 1) / (samples - 1)) for i in range(samples)] if samples > 1 else [0]
        run_dir = args.output / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir.mkdir(parents=True)
        report = dict(
            status="running", video=str(args.video), prompt_file=str(args.prompts),
            prompts=[dict(category=category, label=label) for category, label in prompts],
            device=args.device, source_fps=fps, source_frames=count,
            sampling="Evenly spaced frames across the entire video, including first and last",
            requested_samples=args.samples, planned_frame_indices=indices,
            completed_samples=0, text_threshold=args.text_threshold,
            confidence_bands={"low": "<0.30", "medium": ">=0.30 and <0.60", "high": ">=0.60"},
            method="Query each label separately once per frame at the lowest box threshold; filter the same predictions with score > each threshold. Text threshold is fixed.",
            limitations=["Confidence and detection coverage are not measured accuracy.",
                         "No detection does not prove an item is absent.",
                         "Higher mean confidence at higher thresholds is expected from filtering, not evidence of higher accuracy.",
                         "Duplicate boxes and false positives may be included; no extra NMS is applied.",
                         "Separate label queries may differ from the pipeline's combined prompts."],
            best_threshold=None,
            best_threshold_reason="Requires manually labeled boxes to compare precision, recall and false positives.",
            thresholds=[dict(box_threshold=t, frames=[]) for t in args.thresholds])

        def save_report():
            for result in report["thresholds"]:
                result["summary"] = summarize(result["frames"], prompts)
            temp = run_dir / "comparison.json.part"
            temp.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            temp.replace(run_dir / "comparison.json")

        save_report()
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Cannot decode sampled frame {index}")
            predictions = {label + " .": detector.detect(frame, label + " .")
                           for label in dict.fromkeys(label for _, label in prompts)}
            for number, result in enumerate(report["thresholds"]):
                annotated, rows = audit_frame(frame, CachedDetector(predictions, result["box_threshold"]),
                                              prompts, .30, .60)
                relative = f"threshold_{number}_{result['box_threshold']:g}/frame_{index:08d}.jpg"
                target = run_dir / relative
                target.parent.mkdir(exist_ok=True)
                cv2.putText(annotated, f"BOX > {result['box_threshold']:g} | text {args.text_threshold:g}",
                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
                if not cv2.imwrite(str(target), annotated):
                    raise OSError(f"Cannot save {target}")
                result["frames"].append(dict(frame=index, timestamp=index / fps,
                                             frame_path=relative, results=rows))
            report["completed_samples"] += 1
            save_report()
            print(f"Compared frame {index} ({index/fps:.2f}s): {report['completed_samples']}/{samples}", flush=True)
        report["status"] = "complete"
        save_report()
        print(f"Comparison JSON: {run_dir / 'comparison.json'}", flush=True)
        return run_dir
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=project_path, default=ROOT / "data/testdata/Episode 2.mp4")
    parser.add_argument("--prompts", type=project_path, default=ROOT / "config/prompt_cache.json")
    parser.add_argument("--output", type=project_path, default=ROOT / "outputs/threshold_experiments")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[.1, .2, .3, .4, .5, .55, .6, .7])
    parser.add_argument("--text-threshold", type=float, default=.30)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    if not all(math.isfinite(t) and 0 <= t <= 1 for t in args.thresholds + [args.text_threshold]):
        parser.error("Thresholds must be finite numbers from 0 to 1")
    args.thresholds = sorted(set(args.thresholds))
    try:
        prompts = parse_prompts(json.loads(args.prompts.read_text(encoding="utf-8-sig")))
        if not args.video.is_file():
            raise ValueError(f"Missing video: {args.video}")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Comparing {args.thresholds} on {args.samples} evenly spaced frames", flush=True)
    import torch
    from models.dino_loader import DinoLoader
    from services.detection.detector import GroundingDINODetector
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DinoLoader(str(ROOT / "weights/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"),
                       str(ROOT / "weights/GroundingDINO/weights/groundingdino_swint_ogc.pth"), args.device).load()
    detector = GroundingDINODetector(model, args.device, min(args.thresholds), args.text_threshold)
    experiment(args, detector, prompts)


if __name__ == "__main__":
    main()

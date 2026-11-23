"""Compare CLIP activation and crop handling on identical video detections."""
import argparse
from collections import Counter
from datetime import datetime
import gc
import html
import json
import math
from pathlib import Path
import shutil
import tempfile
import time

import cv2
import numpy as np

from test_clip_matching_audit import ROOT, project_path
from clip_matching_variants import load_clip
from matching_decision import decide, category_counts, decision_category, CATEGORY_LABELS

VARIANTS = ("box", "mask_pipeline", "mask_rgb")


def save_json(path, data):
    payload = json.dumps(data, indent=2, allow_nan=False)
    # A unique, closed file avoids collisions and Windows open-handle conflicts.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".part", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError as error:
            if attempt == 7:
                raise PermissionError(f"Cannot replace {path}; latest checkpoint preserved at {temporary}") from error
            time.sleep(min(.1 * 2 ** attempt, 1.0))


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Cannot save {path}")


def extract(args, detector, run_dir, report):
    """Run each detection once, keeping identical boxes for all six comparisons."""
    cap = cv2.VideoCapture(str(args.video))
    try:
        fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or not math.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError(f"Invalid video metadata: {args.video}")
        end = count if args.max_duration is None else min(count, int(args.max_duration * fps) + 1)
        interval = max(1, int(fps / args.target_fps))
        indices = list(range(0, end, interval))
        if args.samples and len(indices) > args.samples:
            indices = ([indices[round(i * (len(indices) - 1) / (args.samples - 1))]
                        for i in range(args.samples)] if args.samples > 1 else [indices[0]])
        report.update(source_fps=fps, planned_frames=indices, completed_frames=0, detections=[])
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Cannot decode frame {index}")
            height, width = frame.shape[:2]
            rows = []
            for detection in detector.detect(frame, args.prompt):
                cx, cy, bw, bh = detection.bbox
                bbox = [max(0, int((cx - bw / 2) * width)), max(0, int((cy - bh / 2) * height)),
                        min(width, int((cx + bw / 2) * width)), min(height, int((cy + bh / 2) * height))]
                x1, y1, x2, y2 = bbox
                if x2 <= x1 or y2 <= y1:
                    continue
                ident = f"frame_{index:06d}_det_{len(rows):03d}"
                row = dict(id=ident, frame=index, timestamp=index / fps, bbox=bbox,
                           detector_label=detection.label, detector_score=float(detection.score),
                           crops={"box": f"crops/{ident}_box.png"}, results={})
                save_image(run_dir / row["crops"]["box"], frame[y1:y2, x1:x2])
                rows.append(row)
            if rows:
                # Lossless frames allow SAM to run later, after releasing DINO.
                path = f"frames/frame_{index:06d}.png"
                save_image(run_dir / path, frame)
                for row in rows:
                    row["source_frame"] = path
            report["detections"].extend(rows)
            report["completed_frames"] += 1
            save_json(run_dir / "report.json", report)
            print(f"Detection {report['completed_frames']}/{len(indices)}: frame {index}, {len(rows)} detections", flush=True)
    finally:
        cap.release()


def segment(sam, run_dir, report, variants=(("mask_pipeline", "RGB"), ("mask_rgb", "BGR"))):
    frames = {}
    for row in report["detections"]:
        frames.setdefault(row["source_frame"], []).append(row)
    report["segmented_frames"] = 0
    for source, rows in frames.items():
        frame = cv2.imread(str(run_dir / source))
        if frame is None:
            raise ValueError(f"Cannot read saved frame {source}")
        for variant, color in variants:
            sam.set_image(frame, image_format=color)
            for row in rows:
                x1, y1, x2, y2 = row["bbox"]
                masks, _, _ = sam.predict(box=np.array(row["bbox"]), multimask_output=False)
                mask = masks[0][y1:y2, x1:x2]
                crop = frame[y1:y2, x1:x2].copy()
                crop[~mask] = 0
                # Count mask membership, not nonblack image pixels.
                row.setdefault("mask_area_ratios", {})[variant] = float(np.count_nonzero(mask) / mask.size)
                path = f"crops/{row['id']}_{variant}.png"
                save_image(run_dir / path, crop)
                row["crops"][variant] = path
        report["segmented_frames"] += 1
        save_json(run_dir / "report.json", report)
        print(f"SAM {report['segmented_frames']}/{len(frames)}: {source}", flush=True)


def rank(matcher, crop_rgb, threshold, min_margin=.05):
    embedding = matcher.get_embedding(crop_rgb)
    ranking = matcher.rank_products(embedding) if hasattr(matcher, "rank_products") else sorted([dict(product=name, similarity=float((embedding @ ref.T).item()))
                      for name, ref in matcher.product_embeddings.items()],
                     key=lambda item: item["similarity"], reverse=True)
    if not ranking:
        raise ValueError("No readable catalog images")
    best = ranking[0]
    return dict(product=best["product"], similarity=best["similarity"],
                threshold=threshold, ranking=ranking, **decide(ranking, threshold, min_margin))


def score(args, run_dir, report):
    import torch
    from services.matching.clip_matcher import CLIPMatcher
    from product_catalog_matcher import ProductCatalogMatcher
    from test_clip_matching_audit import resolve_catalog_source
    grouped, flat = resolve_catalog_source(products=args.products)

    for corrected in (False, True):
        activation = "quickgelu" if corrected else "pipeline_gelu"
        print(f"Scoring {activation}", flush=True)
        model, preprocess = load_clip(args.device, corrected)
        matcher = (ProductCatalogMatcher if grouped else CLIPMatcher)(model, preprocess, args.device)
        matcher.load_product_catalog(grouped or flat)
        if not matcher.product_names:
            raise ValueError("No readable catalog images")
        report["catalog"] = matcher.product_names
        report.setdefault("catalog_images", {})
        for number, name in enumerate(matcher.product_names):
            source = grouped / matcher.view_paths[name][0] if grouped else flat / name
            relative = f"catalog/{number}{source.suffix}"
            (run_dir / "catalog").mkdir(exist_ok=True)
            shutil.copyfile(source, run_dir / relative)
            report["catalog_images"][name] = relative
        for number, row in enumerate(report["detections"]):
            for variant in VARIANTS:
                crop = cv2.imread(str(run_dir / row["crops"][variant]))
                if crop is None:
                    raise ValueError(f"Cannot read crop {row['id']} / {variant}")
                result = rank(matcher, cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), args.similarity_threshold, args.min_margin)
                row["results"][f"{activation}/{variant}"] = result
            save_json(run_dir / "report.json", report)
            print(f"{activation}: {number + 1}/{len(report['detections'])}", flush=True)
        del matcher, model, preprocess
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def summarize(report):
    baseline = "pipeline_gelu/mask_pipeline"
    summary = {}
    for activation in ("pipeline_gelu", "quickgelu"):
        for variant in VARIANTS:
            key = f"{activation}/{variant}"
            rows = report["detections"]
            results = [row["results"][key] for row in rows]
            summary[key] = dict(
                categories=category_counts(results),
                count=len(results), accepted=sum(item["accepted"] for item in results),
                similarity_only_accepted=sum(item.get("similarity_only_matched", item["accepted"]) for item in results),
                winners=dict(Counter(item["product"] for item in results)),
                mean_similarity=sum(item["similarity"] for item in results) / len(results) if results else None,
                changed_winner=sum(row["results"][key]["product"] != row["results"][baseline]["product"] for row in rows),
                changed_acceptance=sum(row["results"][key]["accepted"] != row["results"][baseline]["accepted"] for row in rows),
            )
    return summary


def write_review(run_dir, report, filename="review.html"):
    escape = html.escape
    parts = ['<!doctype html><meta charset="utf-8"><title>CLIP comparison</title>',
             '<style>body{font:16px system-ui;margin:24px;background:#eee}img{max-width:240px;max-height:240px;object-fit:contain}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #bbb;padding:12px;vertical-align:top}section{margin:32px 0}small{display:block} .catalog{display:flex;gap:24px}</style>',
             '<h1>CLIP comparison</h1><p>Same detection boxes; six scoring variants. Scores are cosine similarities, not probabilities. Changes in acceptance are not measured accuracy.</p><div class="catalog">']
    for name, path in report.get("catalog_images", {}).items():
        parts.append(f'<div><img src="{escape(path, quote=True)}"><p>{escape(name)}</p></div>')
    parts.append('</div><h2>Decision categories</h2><p>Similarity is checked first. Matched means both configured checks passed, not verified product accuracy. Counts are per detection before track filtering.</p><table><tr><th>Configuration</th>')
    parts.extend(f'<th>{escape(label)}</th>' for label in CATEGORY_LABELS.values())
    parts.append('</tr>')
    for key, summary in report["summary"].items():
        parts.append(f'<tr><td>{escape(key)}</td>')
        parts.extend(f'<td>{summary["categories"][category]}</td>' for category in CATEGORY_LABELS)
        parts.append('</tr>')
    parts.append('</table><details><summary>Additional statistics</summary><pre>' + escape(json.dumps(report["summary"], indent=2)) + '</pre></details>')
    for row in report["detections"]:
        parts.append(f'<section><h2>{escape(row["id"])} — {row["timestamp"]:.2f}s</h2>'
                     f'<p>Detection: {escape(row["detector_label"])} ({row["detector_score"]:.3f}); box {row["bbox"]}</p>'
                     f'<a href="{escape(row["source_frame"], quote=True)}">Original frame</a><table><tr>')
        for variant in VARIANTS:
            parts.append(f'<th>{variant}</th>')
        parts.append('</tr><tr>')
        for variant in VARIANTS:
            parts.append(f'<td><img src="{escape(row["crops"][variant], quote=True)}">')
            for activation in ("pipeline_gelu", "quickgelu"):
                result = row["results"][f"{activation}/{variant}"]
                label = CATEGORY_LABELS[decision_category(result)]
                label += f"; margin={result.get('margin')}; {result.get('decision_reason', '')}"
                ranking = '; '.join(f'{item["product"]}: {item["similarity"]:.4f}' for item in result["ranking"])
                parts.append(f'<p><b>{activation}: {escape(result["product"])}</b><br>{result["similarity"]:.4f} — {label}<small>{escape(ranking)}</small></p>')
            parts.append('</td>')
        parts.append('</tr></table></section>')
    (run_dir / filename).write_text('\n'.join(parts), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=project_path, default=ROOT / "data/testdata/Episode 2.mp4")
    parser.add_argument("--products", type=project_path, default=ROOT / "data/productimages/objects")
    parser.add_argument("--prompts", type=project_path, default=ROOT / "config/prompt_cache.json")
    parser.add_argument("--output", type=project_path, default=ROOT / "outputs/clip_matching_comparison")
    parser.add_argument("--max-duration", type=float, help="Optional time limit; default is the complete video")
    parser.add_argument("--samples", type=int, help="Optional evenly spaced subset of sampled frames")
    parser.add_argument("--target-fps", type=float, default=1)
    parser.add_argument("--box-threshold", type=float, default=.35)
    parser.add_argument("--text-threshold", type=float, default=.30)
    parser.add_argument("--similarity-threshold", type=float, default=.55)
    parser.add_argument("--min-margin", type=float, default=.05, help="Product score gap; 0 disables")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if not math.isfinite(args.min_margin) or not 0 <= args.min_margin <= 2:
        parser.error("--min-margin must be finite and between 0 and 2")
    for name in ("max_duration", "target_fps"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"{name} must be finite and positive")
    if args.samples is not None and args.samples <= 0:
        parser.error("samples must be positive")
    for name in ("box_threshold", "text_threshold", "similarity_threshold"):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0 <= value <= 1:
            parser.error(f"{name} must be between 0 and 1")
    try:
        args.prompt = json.loads(args.prompts.read_text(encoding="utf-8-sig"))["object_prompt"]
        if not isinstance(args.prompt, str) or not args.prompt.strip():
            raise ValueError("object_prompt must be a nonempty string")
        if not args.video.is_file() or not args.products.is_dir():
            raise ValueError("Video and catalog folder must exist")
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    import torch

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = args.output / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True)
    report = dict(status="extracting", settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  model="ViT-L-14", pretrained="openai", detections=[])
    save_json(run_dir / "report.json", report)
    try:
        print("Loading detection model", flush=True)
        from models.dino_loader import DinoLoader
        from models.sam_loader import SamLoader
        from services.detection.detector import GroundingDINODetector

        dino = DinoLoader(str(ROOT / "weights/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"),
                          str(ROOT / "weights/GroundingDINO/weights/groundingdino_swint_ogc.pth"), args.device).load()
        detector = GroundingDINODetector(dino, args.device, args.box_threshold, args.text_threshold)
        extract(args, detector, run_dir, report)
        del detector, dino
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        report["status"] = "segmenting"
        save_json(run_dir / "report.json", report)
        sam = SamLoader(str(ROOT / "weights/sam-weight/sam_vit_h_4b8939.pth"), "vit_h", args.device).load()
        segment(sam, run_dir, report)
        del sam
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        report["status"] = "scoring"
        save_json(run_dir / "report.json", report)
        score(args, run_dir, report)
        report["summary"] = summarize(report)
        write_review(run_dir, report)
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save_json(run_dir / "report.json", report)
        print(f"Comparison: {run_dir}", flush=True)


if __name__ == "__main__":
    main()

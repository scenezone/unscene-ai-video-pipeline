"""Test-only corrected SAM matching, with box fallback and a box-only baseline."""
import argparse
from collections import Counter
from datetime import datetime
import gc
import html
import json
import math
import sys
from pathlib import Path

import cv2

from test_clip_matching_audit import ROOT, project_path, resolve_catalog_source
from test_clip_matching_comparison import extract, segment, rank, save_json
from clip_matching_variants import load_clip
from product_catalog_matcher import ProductCatalogMatcher


DEFAULT_MIN_MASK_AREA_RATIO = .20


def select_result(mask, box, mask_area_ratio=None, min_mask_area_ratio=DEFAULT_MIN_MASK_AREA_RATIO):
    """Select box for small masks; otherwise retain mask-first rejection fallback."""
    if not math.isfinite(min_mask_area_ratio) or not 0 <= min_mask_area_ratio <= 1:
        raise ValueError("min_mask_area_ratio must be finite and between 0 and 1")
    if mask_area_ratio is not None and (not math.isfinite(mask_area_ratio) or not 0 <= mask_area_ratio <= 1):
        raise ValueError("mask_area_ratio must be finite and between 0 and 1")
    if mask_area_ratio is not None and mask_area_ratio < min_mask_area_ratio:
        return dict(outcome="matched_box_small_mask" if box["accepted"] else "rejected_box_small_mask",
                    accepted=box["accepted"], product=box["product"] if box["accepted"] else None,
                    fallback_attempted=False, mask_attempt=None, box_attempt=box, box_only_baseline=box,
                    selected_crop="box", crop_selection_reason="mask_area_below_threshold",
                    mask_area_ratio=mask_area_ratio, min_mask_area_ratio=min_mask_area_ratio)
    fallback = not mask["accepted"]
    selected = box if fallback else mask
    accepted = selected["accepted"]
    return dict(outcome=("matched_mask" if not fallback else
                         "matched_box_fallback" if accepted else "rejected_both"),
                accepted=accepted, product=selected["product"] if accepted else None,
                fallback_attempted=fallback, mask_attempt=mask,
                box_attempt=box if fallback else None, box_only_baseline=box,
                selected_crop="box" if fallback else "mask", crop_selection_reason="mask_first",
                mask_area_ratio=mask_area_ratio, min_mask_area_ratio=min_mask_area_ratio)


def score_detection(matcher, run_dir, row, args):
    area_ratio = row.get("mask_area_ratios", {}).get("mask_rgb")
    small_mask = area_ratio is not None and area_ratio < args.min_mask_area_ratio
    results = {}
    for variant in (("box",) if small_mask else ("mask_rgb", "box")):
        crop = cv2.imread(str(run_dir / row["crops"][variant]))
        if crop is None:
            raise ValueError(f"Unreadable crop: {row['id']} {variant}")
        results[variant] = rank(matcher, cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                                args.similarity_threshold, args.min_margin)
    return select_result(results.get("mask_rgb"), results["box"], area_ratio, args.min_mask_area_ratio)


def summarize(rows):
    counts = dict.fromkeys(("matched_mask", "matched_box_fallback", "rejected_both"), 0)
    counts.update(Counter(row["matching"]["outcome"] for row in rows))
    return dict(total=len(rows), outcomes=counts,
                mask_scored=sum(row["matching"]["mask_attempt"] is not None for row in rows),
                mask_only_accepted=sum(bool(row["matching"]["mask_attempt"] and
                                           row["matching"]["mask_attempt"]["accepted"]) for row in rows),
                box_only_accepted=sum(row["matching"]["box_only_baseline"]["accepted"] for row in rows),
                hybrid_accepted=sum(row["matching"]["accepted"] for row in rows),
                accepted_product_disagreements=sum(
                    bool(row["matching"]["mask_attempt"]) and row["matching"]["mask_attempt"]["accepted"] and
                    row["matching"]["box_only_baseline"]["accepted"] and
                    row["matching"]["mask_attempt"]["product"] !=
                    row["matching"]["box_only_baseline"]["product"] for row in rows))


def write_review(run_dir, report):
    esc = html.escape
    parts = ['<!doctype html><meta charset="utf-8"><title>Mask-first CLIP matching</title>',
             '<style>body{font:16px system-ui;margin:24px}img{max-width:260px;max-height:220px}td,th{border:1px solid #ccc;padding:12px;vertical-align:top}table{border-collapse:collapse}pre{white-space:pre-wrap}</style>',
             '<h1>Mask-first matching with box fallback</h1>',
             '<p>Small masks select the box before matching. Other masks keep the existing box fallback. '
             'Mask-only counts and disagreements cover scored masks only. Box-only scores are an independent baseline. '
             'Acceptance counts do not measure accuracy; review the fallback matches against catalog images.</p>',
             '<pre>' + esc(json.dumps(report["summary"], indent=2)) + '</pre>']
    for product in report.get("product_metadata", {}).values():
        parts.append('<h3>' + esc(product["name"]) + '</h3>')
        for path in product.get("review_images", []):
            parts.append(f'<img src="{esc(path, quote=True)}">')
    for row in report["detections"]:
        decision = row["matching"]
        parts.append(f'<h2>{esc(row["id"])} at {row["timestamp"]:.2f}s: {esc(decision["outcome"])}</h2>')
        parts.append('<p>' + esc(f"Mask/box area: {decision.get('mask_area_ratio')}; "
                                f"minimum: {decision.get('min_mask_area_ratio')}; "
                                f"selection: {decision.get('crop_selection_reason', 'mask_first')}") + '</p>')
        parts.append(f'<a href="{esc(row["source_frame"], quote=True)}">Source frame</a><table><tr>')
        for variant, result, label in (("mask_rgb", decision["mask_attempt"], "Mask"),
                                       ("box", decision["box_only_baseline"],
                                        "Box selected: small mask" if decision.get("crop_selection_reason") == "mask_area_below_threshold" else
                                        "Box fallback" if decision["fallback_attempted"] else "Box-only baseline (fallback not used)")):
            parts.append(f'<td><h3>{label}</h3><img src="{esc(row["crops"][variant], quote=True)}">'
                         '<pre>' + esc(json.dumps(result, indent=2) if result is not None else
                                       'Mask matching skipped: area below threshold') + '</pre></td>')
        parts.append('</tr></table>')
    (run_dir / "review.html").write_text('\n'.join(parts), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=project_path, default=ROOT / "data/testdata/Episode 2.mp4")
    parser.add_argument("--products", type=project_path, default=ROOT / "data/productimages/objects")
    parser.add_argument("--output", type=project_path, default=ROOT / "outputs/clip_mask_fallback")
    parser.add_argument("--activation", choices=("pipeline_gelu", "quickgelu"), default="pipeline_gelu")
    parser.add_argument("--prompt", default="accessories")
    parser.add_argument("--target-fps", type=float, default=1)
    parser.add_argument("--max-duration", type=float)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--box-threshold", type=float, default=.35)
    parser.add_argument("--text-threshold", type=float, default=.30)
    parser.add_argument("--similarity-threshold", type=float, default=.55)
    parser.add_argument("--min-margin", type=float, default=.05)
    parser.add_argument("--min-mask-area-ratio", type=float, default=DEFAULT_MIN_MASK_AREA_RATIO,
                        help="Use the box directly when SAM mask coverage is below this fraction (default: 0.20)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", type=project_path, help="Resume scoring from a saved report.json or preserved .part checkpoint")
    args = parser.parse_args(argv)
    resumed = args.resume is not None
    if resumed:
        report = json.loads(args.resume.read_text(encoding="utf-8"))
        run_dir = args.resume.parent
        if report.get("completed_frames") != len(report.get("planned_frames", [])) or "product_metadata" not in report:
            parser.error("Resume requires completed detection, segmentation, and a saved catalog snapshot")
        for row in report["detections"]:
            for variant in ("mask_rgb", "box"):
                if variant not in row["crops"] or not (run_dir / row["crops"][variant]).is_file():
                    parser.error(f"Missing saved crop for {row['id']}: {variant}")
        # Reuse the run's settings, never silently mix new thresholds or models.
        for key, value in report["settings"].items():
            setattr(args, key, project_path(value) if key in ("video", "products", "output") else value)
        # Old checkpoints retain their original mask-first policy, including unscored rows.
        args.min_mask_area_ratio = report["settings"].get("min_mask_area_ratio", 0.)
        if args.min_mask_area_ratio > 0 and any(
                "mask_rgb" not in row.get("mask_area_ratios", {}) for row in report["detections"]):
            parser.error("Area-aware resume requires saved SAM mask area ratios; start a new run")
    for key in ("box_threshold", "text_threshold", "similarity_threshold", "min_margin", "min_mask_area_ratio"):
        value = getattr(args, key)
        if not math.isfinite(value) or not 0 <= value <= (2 if key == "min_margin" else 1):
            parser.error(f"Invalid {key}")
    for key in ("target_fps", "max_duration", "samples"):
        value = getattr(args, key)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"{key} must be positive")
    grouped = None
    if not resumed:
        grouped, _ = resolve_catalog_source(products=args.products)
        if not args.video.is_file() or grouped is None or not args.prompt.strip():
            parser.error("A video, grouped product catalog, and nonempty prompt are required")
    import torch
    import shutil

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if not resumed:
        run_dir = args.output / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir.mkdir(parents=True)
        report = dict(status="extracting", settings={k: str(v) if isinstance(v, Path) else v
                                                   for k, v in vars(args).items() if k != "resume"}, detections=[])
    def release_memory():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    try:
        if not resumed:
            from models.dino_loader import DinoLoader
            from models.sam_loader import SamLoader
            from services.detection.detector import GroundingDINODetector
            dino = DinoLoader(str(ROOT / "weights/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"),
                              str(ROOT / "weights/GroundingDINO/weights/groundingdino_swint_ogc.pth"), args.device).load()
            detector = GroundingDINODetector(dino, args.device, args.box_threshold, args.text_threshold)
            extract(args, detector, run_dir, report)
            del detector, dino
            release_memory()
            report["status"] = "segmenting"
            sam = SamLoader(str(ROOT / "weights/sam-weight/sam_vit_h_4b8939.pth"), "vit_h", args.device).load()
            segment(sam, run_dir, report, variants=(("mask_rgb", "BGR"),))
            del sam
            release_memory()
        else:
            # Use the copied reference images, not a possibly changed live catalog.
            products = []
            for ident, metadata in report["product_metadata"].items():
                products.append(dict(product_id=ident, name=metadata["name"], category=metadata["category"],
                                     views=[dict(path=p) for p in metadata["review_images"]]))
            grouped = run_dir / "resume_catalog.json"
            save_json(grouped, dict(schema_version=1, products=products))
            print(f"Resuming: {sum('matching' in row for row in report['detections'])}/{len(report['detections'])} already scored", flush=True)
        report["status"] = "scoring"
        model, preprocess = load_clip(args.device, corrected=args.activation == "quickgelu")
        matcher = ProductCatalogMatcher(model, preprocess, args.device)
        matcher.load_product_catalog(grouped)
        if resumed:
            matcher.view_paths = {ident: metadata["views"] for ident, metadata in report["product_metadata"].items()}
        else:
            report["product_metadata"] = matcher.product_metadata
        for ident, paths in matcher.view_paths.items():
            if resumed:
                continue
            images = []
            for index, path in enumerate(paths):
                target = Path("catalog") / f"{ident}_{index}{Path(path).suffix}"
                (run_dir / target).parent.mkdir(exist_ok=True)
                shutil.copyfile(grouped / path, run_dir / target)
                images.append(target.as_posix())
            report["product_metadata"][ident]["review_images"] = images
        for row in report["detections"]:
            if "matching" in row:
                continue
            row["matching"] = score_detection(matcher, run_dir, row, args)
            save_json(run_dir / "report.json", report)
            print(row["id"], row["matching"]["outcome"], flush=True)
        report["summary"] = summarize(report["detections"])
        write_review(run_dir, report)
        report["status"] = "complete"
        report.pop("error", None)
        print(json.dumps(report["summary"], indent=2))
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        active_error = sys.exc_info()[0] is not None
        try:
            save_json(run_dir / "report.json", report)
        except PermissionError as error:
            if not active_error:
                raise
            print(f"Final checkpoint also failed: {error}", flush=True)
        print(f"Results: {run_dir}", flush=True)


if __name__ == "__main__":
    main()

"""Audit segmented CLIP catalog matching with similarity and product-margin checks."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys

import cv2
from matching_decision import decide, comparison_counts, category_counts, CATEGORY_LABELS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def resolve_catalog_source(products=None, catalog=None):
    if catalog is not None:
        return catalog, None
    if products is None:
        return ROOT / "data/productimages/objects/by_product", None
    if (products / "by_product").is_dir():
        return products / "by_product", None
    if products.is_dir() and any(p.is_dir() and p.name.startswith("OBJ") for p in products.iterdir()):
        return products, None
    return None, products


class AuditedMatcher:
    """Delegate the decision to production and record the same embedding's ranking."""

    def __init__(self, matcher, output, min_margin=.05):
        self.matcher = matcher
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.rows = []
        self.min_margin = min_margin

    def match(self, crop_rgb, threshold=.55):
        # Capture the embedding produced by the actual match call, avoiding a
        # second inference or a separate implementation of its decision rule.
        embeddings = []
        original = self.matcher.get_embedding

        def capture(image):
            embedding = original(image)
            embeddings.append(embedding)
            return embedding

        had_override = "get_embedding" in self.matcher.__dict__
        previous = self.matcher.__dict__.get("get_embedding")
        self.matcher.get_embedding = capture
        try:
            matched, product, score = self.matcher.match(crop_rgb, threshold)
        finally:
            if had_override:
                self.matcher.get_embedding = previous
            else:
                del self.matcher.get_embedding
        ranking = self.matcher.rank_products(embeddings[0]) if hasattr(self.matcher, "rank_products") else sorted(
            [dict(product=name, similarity=float((embeddings[0] @ ref.T).item()))
             for name, ref in self.matcher.product_embeddings.items()],
            key=lambda row: row["similarity"], reverse=True,
        )
        decision = decide(ranking, threshold, self.min_margin)
        matched = decision["accepted"]
        crop_name = f"crop_{len(self.rows):05d}.png"
        if not cv2.imwrite(str(self.output / crop_name), cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)):
            raise OSError(f"Cannot write {crop_name}")
        metadata = getattr(self.matcher, "product_metadata", {}).get(product, {})
        product_name = metadata.get("name", Path(product).stem if product else None)
        row = dict(crop=crop_name, matched=bool(matched), best_product=product,
                   product_name=product_name, category=metadata.get("category"),
                   similarity=float(score), threshold=threshold,
                   ranking=ranking, **decision)
        self.rows.append(row)
        self.last_decision = row
        with (self.output / "matches.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(f"CLIP {crop_name}: {product} ({product_name}), cosine={score:.4f}, "
              f"margin={decision['margin']}, accepted={matched}, reason={decision['decision_reason']}", flush=True)
        return matched, product, score


def main(argv=None):
    # Production prints Unicode arrows; legacy Windows consoles must not turn
    # an otherwise completed video run into an encoding exception.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=project_path, default=ROOT / "data/testdata/Episode 2.mp4")
    parser.add_argument("--image", type=project_path, help="Match a saved crop instead of running video detection/SAM")
    catalogs = parser.add_mutually_exclusive_group()
    catalogs.add_argument("--products", type=project_path, help="Custom flat or grouped product folder")
    catalogs.add_argument("--catalog", type=project_path,
                          help="Product-ID JSON or folder (default: data/productimages/objects/by_product)")
    parser.add_argument("--validate-catalog", action="store_true",
                        help="Check reference paths and names without loading models or processing video")
    parser.add_argument("--prompts", type=project_path, default=ROOT / "config/prompt_cache.json")
    parser.add_argument("--output", type=project_path, default=ROOT / "outputs/clip_matching_audit")
    parser.add_argument("--box-threshold", type=float, default=.35)
    parser.add_argument("--text-threshold", type=float, default=.30)
    parser.add_argument("--similarity-threshold", type=float, default=.55)
    parser.add_argument("--min-margin", type=float, default=.05,
                        help="Required gap between different product IDs (default .05; 0 disables)")
    parser.add_argument("--max-duration", type=float, default=None,
                        help="Optional limit in video seconds from start (default: complete video)")
    parser.add_argument("--target-fps", type=float, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--baseline", action="store_true",
                        help="Use the original CLIP activation and SAM color behavior")
    parser.add_argument("--crop-mode", choices=("masked", "box"), default="masked",
                        help="Default: masked (SAM segmentation before CLIP); box skips SAM")
    args = parser.parse_args(argv)
    if not math.isfinite(args.min_margin) or not 0 <= args.min_margin <= 2:
        parser.error("--min-margin must be finite and between 0 and 2")
    args.catalog, args.products = resolve_catalog_source(args.products, args.catalog)
    for name in ("box_threshold", "text_threshold", "similarity_threshold"):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0 <= value <= 1:
            parser.error(f"{name} must be finite and between 0 and 1")
    for name in ("max_duration", "target_fps"):
        value = getattr(args, name)
        if name == "max_duration" and value is None:
            continue
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name} must be finite and positive")
    if args.catalog:
        from product_catalog_matcher import read_catalog
        try:
            catalog_data = read_catalog(args.catalog)
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(str(error))
    elif not args.products.is_dir() or not any(
        p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        for p in args.products.iterdir()
    ):
        parser.error(f"No catalog images directly inside: {args.products}")
    print(f"Catalog source: {args.catalog or args.products}", flush=True)
    if args.catalog:
        base = args.catalog if args.catalog.is_dir() else args.catalog.parent
        for product in catalog_data["products"]:
            views = [v for v in product["views"] if v.get("enabled", True)]
            print(f"  {product['product_id']}: {product['name']} ({len(views)} views)", flush=True)
            if args.validate_catalog:
                for view in views:
                    if cv2.imread(str(base / view["path"])) is None:
                        parser.error(f"Cannot decode reference: {base / view['path']}")
    elif args.validate_catalog:
        for path in args.products.iterdir():
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                if cv2.imread(str(path)) is None:
                    parser.error(f"Cannot decode reference: {path}")
                print(f"  {path.name}")
    if args.validate_catalog:
        print("Catalog validation passed.")
        return
    source = args.image or args.video
    if not source.is_file():
        parser.error(f"Missing input: {source}")
    if not args.image:
        print("Matching flow: GroundingDINO -> "
              + ("SAM-masked crop" if args.crop_mode == "masked" else "box crop")
              + " -> CLIP product catalog", flush=True)
    crop = None
    if args.image:
        crop = cv2.imread(str(args.image))
        if crop is None:
            parser.error(f"Cannot decode {args.image}")
    else:
        try:
            prompt = json.loads(args.prompts.read_text(encoding="utf-8-sig"))["object_prompt"]
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("object_prompt must be a nonempty string")
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(str(error))

    import torch
    from clip_matching_variants import BGRPredictor, BoxSegmenter, load_clip
    from services.matching.clip_matcher import CLIPMatcher

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = args.output / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True)
    report = dict(status="running", settings={key: str(value) if isinstance(value, Path) else value
                                              for key, value in vars(args).items()},
                  model="ViT-L-14", pretrained="openai", matches=[],
                  quick_gelu=not args.baseline,
                  sam_image_format=(None if args.image or args.crop_mode == "box" else
                                    "RGB (BGR input)" if args.baseline else "BGR"))
    try:
        model, preprocess = load_clip(args.device, corrected=not args.baseline)
        if args.catalog:
            from product_catalog_matcher import ProductCatalogMatcher
            matcher = ProductCatalogMatcher(model, preprocess, args.device)
            matcher.load_product_catalog(args.catalog)
            report["product_metadata"] = matcher.product_metadata
            report["aggregation"] = "max_similarity_across_enabled_views"
        else:
            matcher = CLIPMatcher(model, preprocess, args.device)
            matcher.load_product_catalog(args.products)
        if not matcher.product_names:
            raise ValueError("Catalog contains no readable images")
        report["catalog"] = matcher.product_names
        audited = AuditedMatcher(matcher, run_dir / "crops", args.min_margin)
        report["matches"] = audited.rows
        if crop is not None:
            audited.match(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), args.similarity_threshold)
        else:
            from models.dino_loader import DinoLoader
            from services.detection.detector import GroundingDINODetector
            from services.segmentation.segmenter import SAMSegmenter
            from pipelines.video_pipeline import VideoPipeline

            dino = DinoLoader(str(ROOT / "weights/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"),
                              str(ROOT / "weights/GroundingDINO/weights/groundingdino_swint_ogc.pth"), args.device).load()
            if args.crop_mode == "box":
                sam = segmenter = BoxSegmenter()
            else:
                from models.sam_loader import SamLoader

                sam = SamLoader(str(ROOT / "weights/sam-weight/sam_vit_h_4b8939.pth"), "vit_h", args.device).load()
                if not args.baseline:
                    sam = BGRPredictor(sam)
                segmenter = SAMSegmenter(sam)
            pipeline = VideoPipeline(
                detector=GroundingDINODetector(dino, args.device, args.box_threshold, args.text_threshold),
                segmenter=segmenter, matcher=audited, sam_predictor=sam,
                object_prompt=prompt, clothing_prompt=None, output_dir=str(run_dir / "pipeline"),
                similarity_threshold=args.similarity_threshold, target_fps=args.target_fps,
                max_duration=args.max_duration,
                crop_mode=args.crop_mode,
                sam_image_format=None,  # Test adapter preserves corrected/original comparison.
            )
            report["final_timestamps"] = pipeline.process_video(str(args.video))
            report["before_track_filter"] = pipeline.product_timestamps
        report["accepted_crops"] = sum(row["matched"] for row in audited.rows)
        report["decision_categories"] = category_counts(audited.rows)
        for category, count in report["decision_categories"].items():
            print(f"{CATEGORY_LABELS[category]}: {count}", flush=True)
        report["similarity_only_accepted_crops"] = sum(row["similarity_only_matched"] for row in audited.rows)
        report["accepted_by_min_margin"] = comparison_counts(audited.rows, args.min_margin)
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (run_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Audit report: {run_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()

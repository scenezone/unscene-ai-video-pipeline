"""CPU checks for promoted catalog matching and video crop modes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tempfile
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch

from services.matching.product_catalog_matcher import ProductCatalogMatcher, read_catalog
from services.segmentation.segmenter import SAMSegmenter
from pipelines.video_pipeline import VideoPipeline
from services.matching.matching_decision import decide


class ProductionMatchingChecks(unittest.TestCase):
    def test_legacy_image_extensions_survive_pipeline_and_response_mapping(self):
        from postprocessing.response_builder import ResponseBuilder
        from postprocessing.range_builder import RangeBuilder
        for extension in (".jpg", ".jpeg", ".png", ".webp", ".JPG"):
            with self.subTest(extension=extension), tempfile.TemporaryDirectory() as directory:
                detector, matcher, cap = Mock(), Mock(), Mock()
                detector.detect.return_value = [SimpleNamespace(bbox=(.5, .5, 1., 1.), label="bag")]
                matcher.match.return_value = (True, "bag" + extension, .8)
                matcher.product_metadata = {}
                cap.isOpened.return_value = True
                cap.get.return_value = 30
                cap.read.side_effect = [(True, np.zeros((4, 4, 3), np.uint8)), (False, None)]
                pipeline = VideoPipeline(detector, None, matcher, None, object_prompt="bag",
                                         output_dir=directory, crop_mode="box")
                with patch("pipelines.video_pipeline.cv2.VideoCapture", return_value=cap):
                    output = pipeline.process_video("video")
                self.assertEqual(list(output["objects"]), ["bag"])
                result = ResponseBuilder().build("job", "series", [dict(episodeId="episode",
                    ranges=RangeBuilder().build_episode(output))], [dict(productId="bag", imageKey="key",
                    category="non-clothing", purchaseLink="shop", imageUrl="image",
                    local_image_path="objects/bag" + extension)])
                self.assertEqual(result["episodes"][0]["objects"][0]["seriesProductId"], "bag")

    def test_clothing_only_factory_skips_empty_object_catalog(self):
        # YAPF otherwise tries to write its generated grammar into site-packages.
        # The factory check does not need a persistent formatter cache.
        with patch("yapf_third_party._ylib2to3.pgen2.grammar.Grammar.dump"):
            from services.pipeline_factory import PipelineFactory
        registry = SimpleNamespace(dino_model=None, device="cpu", sam_predictor=Mock(), clip_model=None, preprocess=None)
        with tempfile.TemporaryDirectory() as directory, \
             patch("services.pipeline_factory.GroundingDINODetector"), \
             patch.object(ProductCatalogMatcher, "load_product_catalog") as load:
            pipeline = PipelineFactory(registry).create("shirt", "", "missing-catalog", output_dir=directory)
            load.assert_not_called()
            self.assertFalse(pipeline.enable_objects)
            self.assertTrue(pipeline.enable_clothing)

    def test_hybrid_each_detection_fallback_and_independent_attempt_reports(self):
        frame = np.full((4, 6, 3), [5, 20, 250], np.uint8)
        mask = frame.copy()
        mask[:, :3] = 0
        matcher = ProductCatalogMatcher(None, None)
        matcher.product_metadata = {}
        # Three detections: mask pass; mask low margin/box pass; both rejected.
        scores = iter([("OBJ004", .8, .6), ("OBJ004", .7, .69),
                       ("OBJ005", .8, .6), ("OBJ007", .5, .4), ("OBJ005", .7, .69)])
        seen = []
        matcher.last_decision = {}
        def match(crop, threshold):
            seen.append(crop.copy())
            product, first, second = next(scores)
            ranking = [dict(product=product, similarity=first), dict(product="OTHER", similarity=second)]
            decision = decide(ranking, threshold, matcher.min_margin)
            # Deliberately reuse a mutable object to verify previous attempts are snapshots.
            matcher.last_decision.clear()
            matcher.last_decision.update(ranking=ranking, **decision)
            return decision["accepted"], product, first
        matcher.match = match
        detector, predictor, segmenter, cap = Mock(), Mock(), Mock(), Mock()
        detector.detect.return_value = [SimpleNamespace(bbox=(.5, .5, 1., 1.),
                                                       label="accessories", score=.8)] * 3
        segmenter.segment.return_value = mask
        cap.isOpened.return_value = True
        cap.get.return_value = 30
        cap.read.side_effect = [(True, frame), (False, None)]
        with tempfile.TemporaryDirectory() as directory:
            pipeline = VideoPipeline(detector, segmenter, matcher, predictor,
                                     object_prompt="accessories", output_dir=directory)
            with patch("pipelines.video_pipeline.cv2.VideoCapture", return_value=cap):
                output = pipeline.process_video("fake.mp4")
            self.assertEqual(set(output["objects"]), {"OBJ004", "OBJ005"})
            self.assertEqual(len(seen), 5)  # No box inference after accepted mask.
            for index in (0, 1, 3):
                np.testing.assert_array_equal(seen[index], mask[:, :, ::-1])
            for index in (2, 4):
                np.testing.assert_array_equal(seen[index], frame[:, :, ::-1])
            predictor.set_image.assert_called_once()
            self.assertEqual(predictor.set_image.call_args.kwargs, {"image_format": "BGR"})
            self.assertEqual(segmenter.segment.call_count, 3)
            report = json.loads((Path(directory) / "object_matching_report.json").read_text())
            self.assertEqual(report["outcomes"], dict(matched_mask=1, matched_box_fallback=1, rejected_both=1))
            self.assertEqual(report["raw_top_pick_comparison"]["comparable"], 2)
            self.assertEqual(report["raw_top_pick_comparison"]["disagreements"], 2)
            rows = report["detections"]
            self.assertIsNone(rows[0]["box_attempt"])
            self.assertEqual(rows[1]["mask_attempt"]["best_product"], "OBJ004")
            self.assertEqual(rows[1]["box_attempt"]["best_product"], "OBJ005")
            self.assertEqual(rows[2]["mask_attempt"]["decision_reason"], "below_similarity_threshold")
            self.assertEqual(rows[2]["box_attempt"]["decision_reason"], "below_min_margin")
            for row in rows:
                for key in ("mask_attempt", "box_attempt"):
                    if row[key]:
                        self.assertTrue((Path(directory) / row[key]["crop"]).is_file())

    def test_all_detections_single_frame_duplicates_and_rejections(self):
        frame = np.full((4, 6, 3), 100, np.uint8)
        matcher = ProductCatalogMatcher(None, None)
        matcher.product_metadata = {}
        pairs = iter([(.8, .6), (.8, .6), (.7, .69), (.5, .4)])

        def match(crop, threshold):
            first, second = next(pairs)
            ranking = [dict(product="OBJ004", similarity=first),
                       dict(product="OBJ007", similarity=second)]
            decision = decide(ranking, threshold, .05)
            matcher.last_decision = dict(ranking=ranking, **decision)
            return decision["accepted"], "OBJ004", first

        matcher.match = match
        detector = Mock()
        detector.detect.return_value = [SimpleNamespace(bbox=(.5, .5, 1., 1.),
                                                        label="accessories", score=.8)] * 4
        cap = Mock()
        cap.isOpened.return_value = True
        cap.get.return_value = 30
        cap.read.side_effect = [(True, frame), (False, None)]
        with tempfile.TemporaryDirectory() as directory:
            pipeline = VideoPipeline(detector, None, matcher, None, object_prompt="accessories",
                                     output_dir=directory, crop_mode="box")
            with patch("pipelines.video_pipeline.cv2.VideoCapture", return_value=cap):
                output = pipeline.process_video("single-frame.mp4")
            self.assertEqual(len(output["objects"]["OBJ004"]), 2)
            self.assertNotIn("OBJ007", output["objects"])
            report = json.loads((Path(directory) / "object_matching_report.json").read_text())
            self.assertEqual(report["total_detections"], 4)
            self.assertEqual(report["decision_categories"], dict(matched=2, rejected_low_margin=1,
                             rejected_low_similarity=1, rejected_unavailable=0))
            self.assertTrue(all((Path(directory) / row["crop"]).is_file() for row in report["detections"]))

    def test_loader_explicitly_selects_pipeline_gelu(self):
        factory = Mock(return_value=(Mock(), None, Mock()))
        with patch.dict("sys.modules", {"open_clip": SimpleNamespace(create_model_and_transforms=factory)}):
            # Load an isolated module so other tests' imports remain unaffected.
            import runpy
            loader = runpy.run_path(str(Path(__file__).resolve().parents[1] / "models/clip_loader.py"))["CLIPLoader"]
            loader("cpu").load()
        self.assertIs(factory.call_args.kwargs["force_quick_gelu"], False)

    def test_grouped_catalog_views_margin_and_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for ident, colors in [("OBJ001_bag", [10, 11]), ("OBJ002_glasses", [20])]:
                folder = root / "by_product" / ident
                folder.mkdir(parents=True)
                for index, color in enumerate(colors):
                    cv2.imwrite(str(folder / f"{index}.png"), np.full((2, 2, 3), color, np.uint8))
            matcher = ProductCatalogMatcher(None, None)
            vectors = {10: [.70, .3], 11: [.699, .4], 20: [.699, .5]}
            matcher.get_embedding = lambda crop: torch.tensor([vectors[int(crop[0, 0, 0])]])
            matcher.load_product_catalog(root)
            self.assertEqual(matcher.product_embeddings["OBJ001"].shape[0], 2)
            self.assertEqual(set(matcher.product_names), {"OBJ001", "OBJ002"})
            matcher.get_embedding = lambda crop: torch.tensor([[1., 0.]])
            self.assertFalse(matcher.match(None)[0])
            self.assertEqual(matcher.last_decision["decision_reason"], "below_min_margin")
            matcher.product_embeddings["OBJ002"] = torch.tensor([[.60, .5]])
            self.assertTrue(matcher.match(None)[0])
            self.assertEqual(matcher.match(None)[1], "OBJ001")
            matcher.product_embeddings["OBJ001"] = torch.tensor([[.50, .3], [.49, .4]])
            matcher.product_embeddings["OBJ002"] = torch.tensor([[.40, .5]])
            self.assertFalse(matcher.match(None)[0])
            self.assertEqual(matcher.last_decision["decision_reason"], "below_similarity_threshold")
            flat = root / "flat"
            flat.mkdir()
            cv2.imwrite(str(flat / "phone.png"), np.zeros((2, 2, 3), np.uint8))
            matcher.load_product_catalog(flat)
            self.assertEqual(matcher.product_names, ["phone.png"])

    def test_box_and_masked_video_paths(self):
        frame = np.full((4, 6, 3), [5, 20, 250], np.uint8)
        for mode in ("box", "masked"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                cap = Mock()
                cap.isOpened.return_value = True
                cap.get.return_value = 1
                cap.read.side_effect = [(True, frame.copy()), (True, frame.copy()), (False, None)]
                detector = Mock()
                detector.detect.return_value = [SimpleNamespace(bbox=(.5, .5, 1., 1.), label="accessories")]
                predictor, segmenter, matcher = Mock(), Mock(), Mock()
                matcher.product_metadata = {"OBJ001": {"name": "Bag"}}
                matcher.match.return_value = (True, "OBJ001", .7)
                segmenter.segment.return_value = frame.copy()
                pipeline = VideoPipeline(detector, segmenter, matcher, predictor,
                                         object_prompt="accessories", output_dir=directory, crop_mode=mode)
                with patch("pipelines.video_pipeline.cv2.VideoCapture", return_value=cap):
                    result = pipeline.process_video("fake.mp4")
                self.assertEqual(len(result["objects"]["OBJ001"]), 2)
                np.testing.assert_array_equal(matcher.match.call_args.args[0], frame[:, :, ::-1])
                if mode == "box":
                    predictor.set_image.assert_not_called()
                    segmenter.segment.assert_not_called()
                else:
                    self.assertEqual(predictor.set_image.call_args.kwargs, {"image_format": "BGR"})
                    self.assertEqual(segmenter.segment.call_count, 2)

    def test_hybrid_routes_small_sam_mask_to_box_and_reports_area(self):
        frame = np.full((4, 6, 3), [5, 20, 250], np.uint8)
        mask = np.zeros((4, 6), dtype=bool)
        mask[1, 1:3] = True
        predictor = Mock()
        predictor.predict.return_value = (np.array([mask]), None, None)
        detector = Mock()
        detector.detect.return_value = [SimpleNamespace(
            bbox=(.5, .5, 1., 1.), label="accessories", score=.8)]
        matcher = Mock()
        matcher.product_metadata = {"OBJ001": {"name": "Bag"}}
        matcher.match.return_value = (True, "OBJ001", .7)
        cap = Mock()
        cap.isOpened.return_value = True
        cap.get.return_value = 30
        cap.read.side_effect = [(True, frame), (False, None)]
        with tempfile.TemporaryDirectory() as directory:
            pipeline = VideoPipeline(
                detector, SAMSegmenter(predictor), matcher, predictor,
                object_prompt="accessories", output_dir=directory)
            with patch("pipelines.video_pipeline.cv2.VideoCapture", return_value=cap):
                pipeline.process_video("small-mask.mp4")
            matcher.match.assert_called_once()
            np.testing.assert_array_equal(matcher.match.call_args.args[0], frame[:, :, ::-1])
            predictor.predict.assert_called_once()
            report = json.loads((Path(directory) / "object_matching_report.json").read_text())
            row = report["detections"][0]
            self.assertEqual(row["mask_area_ratio"], 2 / 24)
            self.assertEqual(row["min_mask_area_ratio"], .2)
            self.assertEqual(row["outcome"], "matched_box_small_mask")
            self.assertIsNone(row["mask_attempt"])
            self.assertFalse(row["fallback_attempted"])
            self.assertEqual(row["crop_selection_reason"], "mask_area_below_threshold")

    def test_hybrid_scores_mask_at_minimum_area_boundary(self):
        frame = np.full((2, 5, 3), 100, np.uint8)
        mask = np.zeros((2, 5), dtype=bool)
        mask[0, :2] = True
        predictor = Mock()
        predictor.predict.return_value = (np.array([mask]), None, None)
        detector = Mock()
        detector.detect.return_value = [SimpleNamespace(
            bbox=(.5, .5, 1., 1.), label="accessories", score=.8)]
        matcher = Mock()
        matcher.product_metadata = {"OBJ001": {"name": "Bag"}}
        matcher.match.return_value = (True, "OBJ001", .7)
        cap = Mock()
        cap.isOpened.return_value = True
        cap.get.return_value = 30
        cap.read.side_effect = [(True, frame), (False, None)]
        with tempfile.TemporaryDirectory() as directory:
            pipeline = VideoPipeline(
                detector, SAMSegmenter(predictor), matcher, predictor,
                object_prompt="accessories", output_dir=directory,
                min_mask_area_ratio=.2)
            with patch("pipelines.video_pipeline.cv2.VideoCapture", return_value=cap):
                pipeline.process_video("boundary-mask.mp4")
            matcher.match.assert_called_once()
            report = json.loads((Path(directory) / "object_matching_report.json").read_text())
            row = report["detections"][0]
            self.assertEqual(row["mask_area_ratio"], .2)
            self.assertEqual(row["outcome"], "matched_mask")
            self.assertEqual(row["selected_crop"], "mask")

    def test_minimum_mask_area_ratio_must_be_finite_fraction(self):
        detector, segmenter, matcher, predictor = Mock(), Mock(), Mock(), Mock()
        for ratio in (-.1, 1.1, float("nan"), float("inf")):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                VideoPipeline(detector, segmenter, matcher, predictor,
                              min_mask_area_ratio=ratio)

    def test_current_data_catalog(self):
        root = Path(__file__).resolve().parents[1] / "data/productimages/objects/by_product"
        data = read_catalog(root)
        self.assertGreater(len(data["products"]), 1)
        self.assertTrue(any(len(product["views"]) > 1 for product in data["products"]))


if __name__ == "__main__":
    unittest.main()

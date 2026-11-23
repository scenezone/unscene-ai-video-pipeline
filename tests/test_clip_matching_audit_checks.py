"""Small deterministic checks of production matching and audit transparency."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from test_clip_matching_audit import AuditedMatcher, ROOT
from services.matching.clip_matcher import CLIPMatcher
from clip_matching_variants import BGRPredictor, BoxSegmenter, load_clip
from test_clip_matching_comparison import extract, segment, rank, summarize, write_review, VARIANTS


class MatchingChecks(unittest.TestCase):
    def test_threshold_ranking_rgb_and_rejected_best_candidate(self):
        class Model:
            def encode_image(self, image):
                return torch.tensor([[3., 4.]])

        seen = []

        def preprocess(image):
            seen.append(np.array(image))
            return torch.zeros(3, 2, 2)

        matcher = CLIPMatcher(Model(), preprocess)
        matcher.product_embeddings = {"first.png": torch.tensor([[1., 0.]]),
                                      "winner.png": torch.tensor([[0., 1.]])}
        crop = np.full((2, 2, 3), [250, 20, 5], dtype=np.uint8)
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            audited = AuditedMatcher(matcher, directory)
            accepted = audited.match(crop, .8)
            rejected = audited.match(crop, .81)
            self.assertTrue(accepted[0])
            self.assertFalse(rejected[0])
            self.assertEqual(rejected[1], "winner.png")
            self.assertAlmostEqual(accepted[2], .8)
            self.assertEqual(len(seen), 2)  # one encode per match, including audit
            np.testing.assert_array_equal(seen[0], crop)
            self.assertNotIn("get_embedding", matcher.__dict__)
            row = audited.rows[0]
            self.assertAlmostEqual(row["margin"], .2)
            self.assertEqual([item["product"] for item in row["ranking"]], ["winner.png", "first.png"])
            self.assertTrue((Path(directory) / row["crop"]).is_file())
            self.assertEqual(len((Path(directory) / "matches.jsonl").read_text().splitlines()), 2)
            json.dumps(audited.rows, allow_nan=False)

    def test_embedding_method_restored_on_failure(self):
        matcher = CLIPMatcher(None, None)
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            with self.assertRaises(TypeError):
                AuditedMatcher(matcher, directory).match(np.zeros((2, 2, 3), dtype=np.uint8))
        self.assertNotIn("get_embedding", matcher.__dict__)

    def test_test_only_adapters_and_activation_selection(self):
        predictor = Mock()
        frame = np.full((4, 6, 3), [5, 20, 250], dtype=np.uint8)
        adapter = BGRPredictor(predictor)
        adapter.set_image(frame)
        predictor.set_image.assert_called_once_with(frame, image_format="BGR")
        adapter.predict(box=(0, 0, 1, 1), multimask_output=False)
        predictor.predict.assert_called_once_with(box=(0, 0, 1, 1), multimask_output=False)
        np.testing.assert_array_equal(BoxSegmenter().segment(frame, (1, 1, 4, 3)), frame[1:3, 1:4])
        factory = Mock(return_value=(Mock(), None, Mock()))
        with patch.dict("sys.modules", {"open_clip": SimpleNamespace(create_model_and_transforms=factory)}):
            load_clip("cpu")
            self.assertTrue(factory.call_args.kwargs["force_quick_gelu"])
            load_clip("cpu", corrected=False)
            self.assertFalse(factory.call_args.kwargs["force_quick_gelu"])

    def test_identical_boxes_color_variants_full_video_and_review(self):
        frame = np.full((4, 6, 3), [5, 20, 250], dtype=np.uint8)
        cap = Mock()
        cap.isOpened.return_value = True
        cap.get.side_effect = [1., 3]
        cap.read.return_value = (True, frame)
        detector = Mock()
        detector.detect.return_value = [SimpleNamespace(bbox=(.5, .5, 1., 1.), label="<bag>", score=.8)]

        class Predictor:
            def __init__(self):
                self.colors = []

            def set_image(self, image, image_format):
                np.testing.assert_array_equal(image, frame)
                self.colors.append(image_format)

            def predict(self, **kwargs):
                mask = np.zeros((4, 6), dtype=bool)
                mask[:, :3 if self.colors[-1] == "RGB" else 6] = True
                return np.array([mask]), None, None

        predictor = Predictor()
        args = SimpleNamespace(video=Path("fake.mp4"), max_duration=None,
                               target_fps=1, samples=None, prompt="accessories")
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            output = Path(directory)
            report = {}
            with patch("test_clip_matching_comparison.cv2.VideoCapture", return_value=cap):
                extract(args, detector, output, report)
            segment(predictor, output, report)
            self.assertEqual(report["planned_frames"], [0, 1, 2])
            self.assertEqual(report["completed_frames"], 3)
            self.assertEqual(detector.detect.call_count, 3)
            self.assertEqual(predictor.colors, ["RGB", "BGR"] * 3)
            self.assertEqual([r["timestamp"] for r in report["detections"]], [0., 1., 2.])
            import cv2
            for row in report["detections"]:
                self.assertEqual(row["bbox"], [0, 0, 6, 4])
                np.testing.assert_array_equal(cv2.imread(str(output / row["crops"]["box"])), frame)
                np.testing.assert_array_equal(cv2.imread(str(output / row["crops"]["mask_rgb"])), frame)
                self.assertEqual(cv2.imread(str(output / row["crops"]["mask_pipeline"]))[:, 3:].sum(), 0)
                for activation in ("pipeline_gelu", "quickgelu"):
                    for variant in VARIANTS:
                        fixed = activation == "quickgelu"
                        row["results"][f"{activation}/{variant}"] = dict(
                            product="bag" if fixed else "glasses", accepted=not fixed,
                            decision_reason="below_similarity_threshold" if fixed else "accepted",
                            similarity=.4 if fixed else .6, margin=.1, ranking=[])
            report["summary"] = summarize(report)
            self.assertEqual(report["summary"]["quickgelu/box"]["changed_winner"], 3)
            self.assertEqual(report["summary"]["quickgelu/box"]["changed_acceptance"], 3)
            self.assertEqual(report["summary"]["quickgelu/box"]["categories"]["rejected_low_similarity"], 3)
            write_review(output, report)
            page = (output / "review.html").read_text(encoding="utf-8")
            self.assertIn("&lt;bag&gt;", page)
            self.assertIn("Rejected - low margin", page)
            self.assertIn("Rejected - low similarity", page)
            self.assertNotIn("<bag>", page)
        cap.release.assert_called_once()

    def test_comparison_rank_matches_production_including_ties(self):
        matcher = CLIPMatcher(None, None)
        matcher.get_embedding = lambda _: torch.tensor([[1., 0.]])
        matcher.product_embeddings = {"first": torch.tensor([[1., 0.]]), "second": torch.tensor([[1., 0.]])}
        result = rank(matcher, None, 1., min_margin=0)
        matched, product, score = matcher.match(None, 1.)
        self.assertEqual((result["accepted"], result["product"], result["similarity"]), (matched, product, score))
        self.assertEqual(result["margin"], 0.)


if __name__ == "__main__":
    unittest.main()

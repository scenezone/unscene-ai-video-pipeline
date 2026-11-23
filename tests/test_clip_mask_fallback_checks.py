import tempfile
import unittest
from pathlib import Path
import json
from unittest.mock import patch
from types import SimpleNamespace

import cv2
import numpy as np

from matching_decision import decide
from test_clip_mask_fallback import select_result, summarize, write_review, score_detection
from test_clip_matching_comparison import save_json, segment


def result(first, second, product="OBJ001"):
    ranking = [dict(product=product, similarity=first), dict(product="OBJ002", similarity=second)]
    return dict(product=product, similarity=first, ranking=ranking, **decide(ranking, .55, .05))


class FallbackChecks(unittest.TestCase):
    def test_small_mask_box_still_requires_similarity_and_margin(self):
        for box, accepted in ((result(.8, .6), True), (result(.5, .3), False), (result(.7, .69), False)):
            decision = select_result(None, box, .19)
            self.assertEqual(decision["accepted"], accepted)
            self.assertIsNone(decision["mask_attempt"])
            self.assertIs(decision["box_attempt"], box)
            self.assertFalse(decision["fallback_attempted"])
            self.assertEqual(decision["product"], box["product"] if accepted else None)

    def test_area_boundary_empty_full_and_disabled(self):
        mask, box = result(.8, .6), result(.5, .3)
        for ratio in (.20, 1.):
            self.assertEqual(select_result(mask, box, ratio)["outcome"], "matched_mask")
        self.assertEqual(select_result(None, box, 0.)["outcome"], "rejected_box_small_mask")
        self.assertEqual(select_result(mask, box, 0., 0.)["outcome"], "matched_mask")
        self.assertEqual(select_result(result(.7, .69), mask, .20)["outcome"], "matched_box_fallback")
        for invalid in (-.1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                select_result(mask, box, invalid)
            with self.assertRaises(ValueError):
                select_result(mask, box, .5, invalid)

    def test_small_mask_skips_mask_inference_and_renders_review(self):
        args = SimpleNamespace(min_mask_area_ratio=.2, similarity_threshold=.55, min_margin=.05)
        row = dict(id="small", timestamp=0., source_frame="frame.png",
                   crops=dict(box="box.png", mask_rgb="missing_mask.png"), mask_area_ratios=dict(mask_rgb=.1))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            crop = np.full((10, 10, 3), [5, 20, 250], np.uint8)
            cv2.imwrite(str(root / "box.png"), crop)
            with patch("test_clip_mask_fallback.rank", return_value=result(.7, .69)) as rank_mock:
                row["matching"] = score_detection(None, root, row, args)
            rank_mock.assert_called_once()
            np.testing.assert_array_equal(rank_mock.call_args.args[1], crop[:, :, ::-1])
            self.assertEqual(rank_mock.call_args.args[2:], (.55, .05))
            summary = summarize([row])
            self.assertEqual(summary["mask_scored"], 0)
            self.assertEqual(summary["hybrid_accepted"], 0)
            write_review(root, dict(summary=summary, detections=[row]))
            page = (root / "review.html").read_text(encoding="utf-8")
            self.assertIn("Mask matching skipped", page)
            self.assertIn("below_min_margin", page)

    def test_mask_area_counts_black_foreground_inside_box_only(self):
        from unittest.mock import Mock
        frame = np.zeros((6, 6, 3), np.uint8)
        mask = np.ones((6, 6), dtype=bool)
        mask[1:5, 1:5] = False
        mask[1, 1:3] = True
        predictor = Mock()
        predictor.predict.return_value = (np.array([mask]), None, None)
        row = dict(id="black", source_frame="frame.png", bbox=[1, 1, 5, 5], crops={})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cv2.imwrite(str(root / "frame.png"), frame)
            segment(predictor, root, dict(detections=[row]), variants=(("mask_rgb", "BGR"),))
            self.assertEqual(row["mask_area_ratios"]["mask_rgb"], 2 / 16)
            predictor.predict.assert_called_once()

    def test_checkpoint_retries_transient_permission_failure(self):
        original_replace = Path.replace
        calls = []
        def replace(source, target):
            calls.append(source)
            if len(calls) < 3:
                raise PermissionError("Simulated Windows file lock")
            return original_replace(source, target)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "report.json"
            target.write_text('{"old": true}')
            with patch.object(Path, "replace", replace), patch("test_clip_matching_comparison.time.sleep"):
                save_json(target, {"new": True})
            self.assertEqual(json.loads(target.read_text()), {"new": True})
            self.assertEqual(len(calls), 3)
            self.assertEqual(list(Path(directory).glob("*.part")), [])

    def test_checkpoint_preserves_old_and_new_data_on_persistent_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "report.json"
            target.write_text('{"old": true}')
            with patch.object(Path, "replace", side_effect=PermissionError("locked")), patch("test_clip_matching_comparison.time.sleep"):
                with self.assertRaisesRegex(PermissionError, "checkpoint preserved"):
                    save_json(target, {"new": True})
            self.assertEqual(json.loads(target.read_text()), {"old": True})
            pending = list(Path(directory).glob("*.part"))
            self.assertEqual(len(pending), 1)
            self.assertEqual(json.loads(pending[0].read_text()), {"new": True})

    def test_mask_acceptance_is_never_overridden(self):
        decision = select_result(result(.8, .6), result(.9, .7, "OBJ003"))
        self.assertEqual(decision["outcome"], "matched_mask")
        self.assertEqual(decision["product"], "OBJ001")
        self.assertFalse(decision["fallback_attempted"])
        self.assertIsNone(decision["box_attempt"])

    def test_either_mask_rejection_can_trigger_fallback(self):
        for mask in (result(.5, .3), result(.7, .69)):
            decision = select_result(mask, result(.8, .6))
            self.assertEqual(decision["outcome"], "matched_box_fallback")
            self.assertTrue(decision["fallback_attempted"])
            self.assertIs(decision["mask_attempt"], mask)

    def test_box_must_pass_both_checks(self):
        for box in (result(.5, .3), result(.7, .69)):
            decision = select_result(result(.7, .69), box)
            self.assertEqual(decision["outcome"], "rejected_both")
            self.assertFalse(decision["accepted"])
            self.assertIsNone(decision["product"])
            self.assertIs(decision["box_attempt"], box)

    def test_counts_and_review(self):
        pairs = [(result(.8, .6), result(.9, .7, "OBJ003")),
                 (result(.7, .69), result(.8, .6)),
                 (result(.5, .3), result(.5, .4))]
        rows = [dict(id=f"crop_{index}", timestamp=index, source_frame="frames/a.png",
                     crops=dict(box="box.png", mask_rgb="mask.png"), matching=select_result(*pair))
                for index, pair in enumerate(pairs)]
        counts = summarize(rows)
        self.assertEqual(counts["outcomes"], dict(matched_mask=1, matched_box_fallback=1, rejected_both=1))
        self.assertEqual(counts["accepted_product_disagreements"], 1)
        self.assertEqual(counts["hybrid_accepted"], 2)
        self.assertEqual(counts["mask_only_accepted"], 1)
        self.assertEqual(counts["box_only_accepted"], 2)
        with tempfile.TemporaryDirectory() as directory:
            write_review(Path(directory), dict(summary=counts, detections=rows))
            page = (Path(directory) / "review.html").read_text(encoding="utf-8")
            self.assertIn("fallback not used", page)
            self.assertIn("below_min_margin", page)
            self.assertIn("below_similarity_threshold", page)


if __name__ == "__main__":
    unittest.main()

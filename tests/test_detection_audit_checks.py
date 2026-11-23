"""Check audit reporting without model weights."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

spec = importlib.util.spec_from_file_location("audit", Path(__file__).with_name("test_detection_audit.py"))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class AuditChecks(unittest.TestCase):
    def test_both_prompt_categories_and_optional_empty_category(self):
        self.assertEqual(audit.parse_prompts({"clothing_prompt": "shirt . shirt . bag",
                                             "object_prompt": "bag . glasses"}),
                         [("clothing", "shirt"), ("clothing", "bag"),
                          ("objects", "bag"), ("objects", "glasses")])
        self.assertEqual(audit.parse_prompts({"object_prompt": "bag"}), [("objects", "bag")])
        with self.assertRaises(ValueError):
            audit.parse_prompts({"clothing_prompt": " . ", "object_prompt": ""})
        with self.assertRaises(ValueError):
            audit.parse_prompts({"clothing_prompt": None, "object_prompt": "bag"})

    def test_band_boundaries(self):
        self.assertEqual([audit.confidence_band(v, .3, .6) for v in (.1, .3, .6)],
                         ["low", "medium", "high"])

    def test_saved_frames_and_missing_object_rows(self):
        frame = np.zeros((100, 150, 3), dtype=np.uint8)
        detector = Mock()
        detector.detect.side_effect = lambda image, prompt: [
            SimpleNamespace(bbox=(.5, .5, .5, .5), score=score, label="bag")
            for score in (.2, .4, .8)
        ] if prompt == "bag ." else []
        cap = Mock()
        cap.isOpened.return_value = True
        cap.get.return_value = 2.0
        cap.read.side_effect = [(True, frame)] * 3 + [(False, None)]
        with tempfile.TemporaryDirectory(dir=audit.ROOT) as directory:
            args = SimpleNamespace(video=Path("example.mp4"), output=Path(directory),
                                   prompts=Path("prompts.json"), fps=1, max_frames=None,
                                   medium=.3, high=.6, box_threshold=.1, text_threshold=.1,
                                   device="cpu")
            with patch.object(audit.cv2, "VideoCapture", return_value=cap):
                result = audit.run_audit(args, detector,
                                         [("clothing", "bag"), ("objects", "bag"),
                                          ("clothing", "shirt"), ("objects", "glasses")])
            with (result / "detections.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 16)
            self.assertEqual({row["category"] for row in rows}, {"clothing", "objects"})
            self.assertEqual({row["frame"] for row in rows}, {"0", "2"})
            self.assertTrue(all((result / row["frame_path"]).is_file() for row in rows))
            missing = [row for row in rows if row["status"] == "no_detection"]
            self.assertEqual(len(missing), 4)
            self.assertTrue(all(row["confidence"] == "" for row in missing))
            summary = json.loads((result / "summary.json").read_text())
            self.assertEqual(summary["sampled_frames"], 2)
            self.assertEqual(summary["counts"]["clothing"]["bag"]["low"], 2)
            self.assertEqual(summary["counts"]["objects"]["bag"]["low"], 2)
            self.assertEqual(summary["counts"]["clothing"]["shirt"]["frames_without_detection"], 2)
            self.assertEqual(summary["counts"]["objects"]["glasses"]["frames_without_detection"], 2)
        cap.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()

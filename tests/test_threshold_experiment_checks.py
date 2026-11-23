import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from test_threshold_experiment import ROOT, experiment


class ThresholdChecks(unittest.TestCase):
    def test_same_predictions_strict_thresholds_and_saved_reports(self):
        frame = np.zeros((100, 150, 3), dtype=np.uint8)
        cap = Mock()
        cap.isOpened.return_value = True
        cap.get.side_effect = [5, 2.0]
        cap.read.return_value = (True, frame)
        detector = Mock()
        detector.detect.return_value = [SimpleNamespace(bbox=(.5, .5, .5, .5), score=s, label="shirt")
                                        for s in (.2, .5)]
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            args = SimpleNamespace(video=Path("fake.mp4"), prompts=Path("fake.json"),
                                   samples=3, output=Path(directory), thresholds=[.1, .2, .5],
                                   text_threshold=.1, device="cpu")
            with patch("test_threshold_experiment.cv2.VideoCapture", return_value=cap):
                result = experiment(args, detector, [("clothing", "shirt"), ("objects", "shirt")])
            report = json.loads((result / "comparison.json").read_text())
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["planned_frame_indices"], [0, 2, 4])
            self.assertEqual(detector.detect.call_count, 3)
            self.assertEqual([t["summary"]["total_detections"] for t in report["thresholds"]], [12, 6, 0])
            for threshold in report["thresholds"]:
                self.assertEqual([f["frame"] for f in threshold["frames"]], [0, 2, 4])
                self.assertTrue(all((result / f["frame_path"]).is_file() for f in threshold["frames"]))
            empty = report["thresholds"][-1]["summary"]["by_category"]["clothing"]["shirt"]
            self.assertIsNone(empty["mean_confidence"])
            self.assertEqual(empty["frames_without_detection"], 3)
            self.assertIsNone(report["best_threshold"])
        cap.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()

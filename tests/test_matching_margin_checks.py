import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from test_clip_matching_audit import ROOT, AuditedMatcher
from product_catalog_matcher import ProductCatalogMatcher
from test_clip_matching_comparison import rank
from matching_decision import decide, comparison_counts, category_counts


def ranking(first, second):
    return [dict(product="BAG", similarity=first), dict(product="GLASSES", similarity=second)]


class MarginChecks(unittest.TestCase):
    def test_categories_are_exclusive_and_similarity_failure_takes_priority(self):
        decisions = [decide(ranking(.70, .60)), decide(ranking(.70, .69)),
                     decide(ranking(.50, .49)), decide(ranking(.50, .30))]
        self.assertEqual(decisions[2]["decision_category"], "rejected_low_similarity")
        counts = category_counts(decisions)
        self.assertEqual(counts, dict(matched=1, rejected_low_margin=1,
                                     rejected_low_similarity=2, rejected_unavailable=0))
        self.assertEqual(sum(counts.values()), len(decisions))
        self.assertEqual(decide([])["decision_category"], "rejected_unavailable")
        self.assertEqual(decide([dict(product="BAG", similarity=.9)])["decision_category"],
                         "rejected_unavailable")

    def test_rejection_threshold_margin_boundary_and_disable(self):
        for first, second, margin, accepted, reason in [
            (.65, .649, .05, False, "below_min_margin"),
            (.65, .62, .05, False, "below_min_margin"),
            (.65, .58, .05, True, "accepted"),
            (.53, .40, .05, False, "below_similarity_threshold"),
            (.60, .55, .05, True, "accepted"),
            (.60, .5501, .05, False, "below_min_margin"),
            (.65, .65, 0, True, "accepted"),
            (.65, .65, .05, False, "below_min_margin"),
            (.65, .58, .10, False, "below_min_margin"),
        ]:
            with self.subTest(first=first, second=second, margin=margin):
                result = decide(ranking(first, second), .55, margin)
                self.assertEqual(result["accepted"], accepted)
                self.assertEqual(result["decision_reason"], reason)

    def test_single_product_empty_and_invalid(self):
        one = [dict(product="BAG", similarity=.9)]
        self.assertEqual(decide(one)["decision_reason"], "no_second_product")
        self.assertTrue(decide(one, min_margin=0)["accepted"])
        self.assertFalse(decide([])["accepted"])
        for value in (-.1, float("nan"), float("inf"), 3):
            with self.assertRaises(ValueError):
                decide(one, min_margin=value)

    def test_two_views_of_same_product_do_not_reduce_margin(self):
        rows = [dict(product="BAG", similarity=.8), dict(product="BAG", similarity=.799),
                dict(product="GLASSES", similarity=.6)]
        self.assertTrue(decide(rows)["accepted"])
        self.assertAlmostEqual(decide(rows)["margin"], .2)

    def test_rejected_decision_reaches_pipeline_caller_and_reports_old_rule(self):
        matcher = ProductCatalogMatcher(None, None)
        matcher.get_embedding = lambda _: torch.tensor([[1., 0.]])
        matcher.product_embeddings = {"BAG": torch.tensor([[.65, .76], [.6495, .76]]),
                                      "GLASSES": torch.tensor([[.649, .76]])}
        matcher.view_paths = {"BAG": ["front", "side"], "GLASSES": ["front"]}
        crop = np.zeros((3, 3, 3), dtype=np.uint8)
        self.assertFalse(matcher.match(crop)[0])
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            audit = AuditedMatcher(matcher, Path(directory))
            self.assertFalse(audit.match(crop)[0])
            row = audit.rows[0]
            self.assertTrue(row["similarity_only_matched"])
            self.assertEqual(row["second_product"], "GLASSES")
            self.assertFalse(row["matched"])
            self.assertFalse(rank(matcher, crop, .55)["accepted"])
            self.assertEqual(comparison_counts(audit.rows, .05), {"0": 1, "0.05": 0, "0.1": 0})
            audit.min_margin = 0
            self.assertTrue(audit.match(crop)[0])


if __name__ == "__main__":
    unittest.main()

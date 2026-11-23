import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from test_clip_matching_audit import ROOT, AuditedMatcher, resolve_catalog_source
from product_catalog_matcher import ProductCatalogMatcher, read_catalog


class ProductCatalogChecks(unittest.TestCase):
    def test_folder_catalog_covers_current_reference_images(self):
        base = ROOT / "data/productimages/objects/by_product"
        data = read_catalog(base)
        self.assertGreater(len(data["products"]), 1)
        actual = {p.relative_to(base).as_posix() for p in base.glob("OBJ*/*")
                  if p.is_file() and p.suffix.lower() in {".png", ".jpeg", ".jpg", ".webp"}}
        listed = [v["path"] for p in data["products"] for v in p["views"]]
        self.assertEqual(actual, set(listed))
        self.assertEqual(len(listed), len(set(listed)))
        self.assertTrue(any(len(p["views"]) > 1 for p in data["products"]))
        self.assertEqual(data["unassigned"], [])

    def test_default_and_parent_paths_select_current_grouped_catalog(self):
        base = ROOT / "data/productimages/objects/by_product"
        self.assertEqual(resolve_catalog_source(), (base, None))
        self.assertEqual(resolve_catalog_source(products=base.parent), (base, None))
        self.assertEqual(resolve_catalog_source(products=base), (base, None))
        self.assertEqual(resolve_catalog_source(catalog=base / "catalog.json"), (base / "catalog.json", None))

    def test_folder_catalog_works_without_manifest_and_uses_folder_name(self):
        import cv2
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            root = Path(directory)
            group = root / "OBJ006_flower_keychain"
            group.mkdir()
            cv2.imwrite(str(group / "reference.png"), np.zeros((4, 4, 3), dtype=np.uint8))
            data = read_catalog(root)
            self.assertEqual(data["products"][0]["product_id"], "OBJ006")
            self.assertEqual(data["products"][0]["name"], "Flower Keychain")
            matcher = ProductCatalogMatcher(None, None)
            matcher.get_embedding = lambda _: torch.tensor([[1., 0.]])
            matcher.load_product_catalog(root)
            audit = AuditedMatcher(matcher, root / "results")
            audit.match(np.zeros((4, 4, 3), dtype=np.uint8))
            self.assertEqual(audit.rows[0]["product_name"], "Flower Keychain")
            self.assertEqual(audit.rows[0]["category"], "keychain")

    def test_alternate_views_return_one_product_and_margin_between_products(self):
        matcher = ProductCatalogMatcher(None, None)
        matcher.get_embedding = lambda image: torch.tensor([[1., 0.]])
        matcher.product_embeddings = {"BAG": torch.tensor([[0., 1.], [1., 0.]]),
                                      "GLASSES": torch.tensor([[.8, .6]])}
        matcher.view_paths = {"BAG": ["front.png", "side.png"], "GLASSES": ["glasses.png"]}
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            audit = AuditedMatcher(matcher, directory)
            self.assertEqual(audit.match(np.zeros((4, 4, 3), dtype=np.uint8), .9), (True, "BAG", 1.))
            row = audit.rows[0]
            self.assertEqual(len(row["ranking"]), 2)
            self.assertEqual(row["ranking"][0]["best_view"], "side.png")
            self.assertAlmostEqual(row["margin"], .2)

    def test_loader_excludes_unassigned_and_replaces_old_catalog(self):
        matcher = ProductCatalogMatcher(None, None)
        matcher.get_embedding = lambda image: torch.tensor([[1., 0.]])
        matcher.product_embeddings = {"OLD": torch.zeros(1, 2)}
        base = ROOT / "data/productimages/objects/by_product"
        expected = read_catalog(base)
        matcher.load_product_catalog(base)
        self.assertNotIn("OLD", matcher.product_embeddings)
        self.assertEqual({key: tensor.shape[0] for key, tensor in matcher.product_embeddings.items()},
                         {p["product_id"]: len(p["views"]) for p in expected["products"]})

    def test_duplicate_ids_fail_before_model_loading(self):
        data = dict(schema_version=1, products=[dict(product_id="BAD/ID", views=[])])
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            path = Path(directory) / "catalog.json"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "Invalid or duplicate"):
                read_catalog(path)


if __name__ == "__main__":
    unittest.main()

"""Multi-view product identity matching, isolated from the production matcher."""
import hashlib
import json
from pathlib import Path
import re

import cv2
import torch

from services.matching.clip_matcher import CLIPMatcher
from matching_decision import decide


def catalog_from_folders(folder):
    """Read current ID folders directly, without moving images or needing JSON."""
    folder = Path(folder).resolve()
    products = []
    for directory in sorted(folder.iterdir()):
        if not directory.is_dir() or directory.name == "review_required":
            continue
        match = re.fullmatch(r"(OBJ\d+)_(.+)", directory.name)
        if not match:
            raise ValueError(f"Expected an OBJnnn_name product folder: {directory}")
        ident, slug = match.groups()
        views = []
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            views.append(dict(path=path.relative_to(folder).as_posix(), source_filename=path.name,
                              view=re.sub(r"^OBJ\d+_\d+_", "", path.stem), enabled=True,
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        if not views:
            raise ValueError(f"No reference images in {directory}")
        category = next((word for word in ("earrings", "eyeglasses", "phone", "keychain") if word in slug), "handbag" if "bag" in slug else "accessory")
        products.append(dict(product_id=ident, name=slug.replace("_", " ").title(),
                             category=category, views=views))
    return dict(schema_version=1, identity_basis="Current user-maintained folder names and membership",
                aggregation="max_similarity_across_enabled_views", products=products, unassigned=[])


def read_catalog(manifest):
    manifest = Path(manifest).resolve()
    base = manifest if manifest.is_dir() else manifest.parent
    data = catalog_from_folders(manifest) if manifest.is_dir() else json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported catalog schema")
    products = data.get("products", [])
    ids, paths = set(), set()
    if not products:
        raise ValueError("Catalog has no products")
    for product in products:
        ident = product["product_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", ident) or ident in ids:
            raise ValueError(f"Invalid or duplicate product ID: {ident}")
        ids.add(ident)
        enabled = [view for view in product["views"] if view.get("enabled", True)]
        if not enabled:
            raise ValueError(f"No enabled views for {ident}")
        for view in enabled:
            path = (base / view["path"]).resolve()
            if not path.is_relative_to(base) or not path.is_file():
                raise ValueError(f"Missing or out-of-catalog image: {view['path']}")
            if path in paths:
                raise ValueError(f"Reference assigned more than once: {view['path']}")
            paths.add(path)
            if view.get("sha256") and hashlib.sha256(path.read_bytes()).hexdigest() != view["sha256"]:
                raise ValueError(f"Reference differs from reviewed image: {view['path']}")
    return data


class ProductCatalogMatcher(CLIPMatcher):
    def load_product_catalog(self, manifest):
        data = read_catalog(manifest)
        location = Path(manifest).resolve()
        base = location if location.is_dir() else location.parent
        embeddings, view_paths, metadata = {}, {}, {}
        for product in data["products"]:
            ident = product["product_id"]
            tensors, names = [], []
            for view in product["views"]:
                if not view.get("enabled", True):
                    continue
                image = cv2.imread(str(base / view["path"]))
                if image is None:
                    raise ValueError(f"Cannot decode catalog reference: {view['path']}")
                tensors.append(self.get_embedding(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
                names.append(view["path"])
            embeddings[ident] = torch.cat(tensors, dim=0)
            view_paths[ident] = names
            metadata[ident] = dict(name=product["name"], category=product["category"], views=names)
        # Replace the catalog atomically; reloading never retains old products.
        self.product_embeddings = embeddings
        self.product_names = list(embeddings)
        self.view_paths = view_paths
        self.product_metadata = metadata
        print(f"Loaded {sum(len(v) for v in view_paths.values())} views for {len(embeddings)} product IDs")

    def rank_products(self, embedding):
        ranking = []
        for ident, references in self.product_embeddings.items():
            scores = (embedding @ references.T).flatten().tolist()
            best = max(range(len(scores)), key=scores.__getitem__)
            ranking.append(dict(product=ident, product_name=getattr(self, "product_metadata", {}).get(ident, {}).get("name", ident), similarity=scores[best],
                                best_view=self.view_paths[ident][best],
                                view_scores=[dict(path=path, similarity=score)
                                             for path, score in zip(self.view_paths[ident], scores)]))
        return sorted(ranking, key=lambda row: row["similarity"], reverse=True)

    def match(self, crop_np, threshold=.55, min_margin=.05):
        if not self.product_embeddings:
            raise ValueError("Load a product catalog before matching")
        ranking = self.rank_products(self.get_embedding(crop_np))
        best = ranking[0]
        return decide(ranking, threshold, min_margin)["accepted"], best["product"], best["similarity"]

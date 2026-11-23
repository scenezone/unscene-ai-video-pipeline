"""Normalize names within current user groups; never regroup or import originals."""
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/productimages/objects/by_product"
SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
GROUPS = {
    "OBJ002": ("tote_bag", "Beige tote bag", "handbag"),
    "OBJ003": ("jhumka_earrings", "Silver lotus jhumka earrings", "earrings"),
    "OBJ004": ("black_eyeglasses", "Black rectangular eyeglasses", "eyeglasses"),
    "OBJ005": ("vivo_phone", "Dark gray Vivo phone", "phone"),
    "OBJ006": ("flower_keychain", "Gold keyring with white flower charms", "keychain"),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_path(path):
    resolved = path.resolve()
    if not resolved.is_relative_to(BASE.resolve()) or resolved == BASE.resolve():
        raise ValueError(f"Path outside catalog: {path}")
    return resolved


def main():
    manifest = BASE / "catalog.json"
    old = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {"products": [], "unassigned": []}
    previous = {v["sha256"]: v for p in old["products"] for v in p["views"]}
    previous.update({v["sha256"]: v for v in old.get("unassigned", [])})
    folders = {}
    for directory in sorted(BASE.iterdir()):
        if not directory.is_dir() or directory.name == "review_required":
            continue
        match = re.match(r"^(OBJ\d+)(?:_|$)", directory.name)
        if not match or match[1] not in GROUPS or match[1] in folders:
            raise ValueError(f"Review unexpected or duplicate group: {directory}")
        folders[match[1]] = safe_path(directory)
    if set(folders) != set(GROUPS):
        raise ValueError("Expected the five current user-maintained product groups")
    catalog = dict(schema_version=1, reviewed_on="2026-10-05",
                   identity_basis="User-maintained folder membership; visually reviewed names. Local IDs, not verified retail SKUs.",
                   aggregation="max_similarity_across_enabled_views", products=[], unassigned=[])
    plan, snapshots = [], {}
    for ident, directory in folders.items():
        slug, name, category = GROUPS[ident]
        target_dir = safe_path(BASE / f"{ident}_{slug}")
        if directory != target_dir and target_dir.exists():
            raise ValueError(f"Destination exists: {target_dir}")
        files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES)
        if not files:
            raise ValueError(f"No images in {directory}")
        snapshots[ident] = Counter(digest(p) for p in files)
        product = dict(product_id=ident, name=name, category=category,
                       notes="Membership preserved exactly as arranged by the user.", views=[])
        for number, path in enumerate(files, 1):
            checksum = digest(path)
            metadata = previous.get(checksum, {})
            view = metadata.get("view", re.sub(r"^(?:OBJ\d+_)?\d+_", "", path.stem))
            if ident == "OBJ002" and view == "three_quarter":
                view = "three_quarter_dark"
            view = re.sub(r"[^a-z0-9_]+", "_", view.lower()).strip("_")
            filename = f"{ident}_{number:02d}_{view}{path.suffix.lower()}"
            destination = safe_path(directory / filename)
            if destination != path and destination.exists():
                raise ValueError(f"File rename would overwrite: {destination}")
            plan.append((safe_path(path), destination))
            product["views"].append(dict(metadata, source_filename=metadata.get("source_filename", path.name), path=f"{target_dir.name}/{filename}", view=view,
                                         sha256=checksum, grouping_confidence="user_assigned",
                                         enabled=metadata.get("enabled", True)))
        if ident == "OBJ002":
            product["notes"] += " All five bag photos remain together despite visible logo/color differences. OBJ006 is the user's current keychain ID."
        catalog["products"].append(product)
        plan.append((directory, target_dir))
    review = BASE / "review_required"
    for path in sorted(review.iterdir()) if review.exists() else []:
        if path.is_file() and path.suffix.lower() in SUFFIXES:
            checksum = digest(path)
            catalog["unassigned"].append(dict(previous.get(checksum, {}), path=path.relative_to(BASE).as_posix(),
                                               sha256=checksum, enabled=False))
    # Validate every source/destination within BASE before renaming directories.
    for source, destination in plan:
        if source != destination:
            source.rename(destination)
    for product in catalog["products"]:
        actual = Counter(digest(BASE / view["path"]) for view in product["views"])
        if actual != snapshots[product["product_id"]]:
            raise ValueError("Image content or group membership changed unexpectedly")
    temporary = BASE / "catalog.json.part"
    temporary.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    temporary.replace(manifest)
    parts = ['<!doctype html><meta charset="utf-8"><title>Object catalog</title>',
             '<style>body{font:16px system-ui;margin:30px}.views{display:flex;flex-wrap:wrap;gap:16px}figure{width:230px;margin:0}img{width:230px;height:250px;object-fit:contain}figcaption{overflow-wrap:anywhere;font-size:13px}section{border-top:1px solid #aaa;padding:20px 0}</style>',
             '<h1>Object catalog</h1><p>Current user-assigned groups. Image contents and folder memberships are preserved.</p>']
    for product in catalog["products"]:
        parts.append(f'<section><h2>{product["product_id"]}: {html.escape(product["name"])}</h2><p>{html.escape(product["notes"])}</p><div class="views">')
        for view in product["views"]:
            parts.append(f'<figure><img src="{view["path"]}"><figcaption>{html.escape(view["path"])}<br>Original: {html.escape(view.get("source_filename", ""))}</figcaption></figure>')
        parts.append('</div></section>')
    for view in catalog["unassigned"]:
        parts.append(f'<section><h2>Review required - excluded</h2><img src="{view["path"]}"><p>{html.escape(view.get("reason", "Unassigned reference"))}</p></section>')
    (BASE / "index.html").write_text('\n'.join(parts), encoding="utf-8")
    print(f"Updated {len(catalog['products'])} IDs; preserved {sum(sum(c.values()) for c in snapshots.values())} grouped images and {len(catalog['unassigned'])} review images.")


if __name__ == "__main__":
    main()

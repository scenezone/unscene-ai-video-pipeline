"""Create two main-API requests from the episode-specific files in data/."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import quote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
EPISODES = {
    1: ("testdata/Episode 1.mp4", "productimages/clothing1", "productimages/objects1/by_product"),
    2: ("testdata/Episode 2.mp4", "productimages/clothing", "productimages/objects/by_product"),
}


def image_files(directory):
    return sorted((p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
                  key=lambda p: p.relative_to(directory).as_posix().casefold())


def data_key(path):
    resolved = path.resolve()
    if not resolved.is_relative_to(DATA.resolve()) or not resolved.is_file():
        raise ValueError(f"Missing file or file outside data/: {path}")
    return path.relative_to(DATA).as_posix()


def checked_base(value):
    parsed = urlsplit(value)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment):
        raise ValueError("Media base URL must be absolute HTTP(S) without credentials, query or fragment")
    return value.rstrip("/")


def media_url(base, path):
    return base + "/" + quote(data_key(path), safe="/")


def build_payload(number, base, callback_url=None, run_tag="local-v1"):
    video_key, clothing_key, object_key = EPISODES[number]
    video, clothing_dir, object_dir = (DATA / key for key in (video_key, clothing_key, object_key))
    data_key(video)
    if not clothing_dir.is_dir() or not object_dir.is_dir():
        raise ValueError(f"Missing episode {number} reference directories")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_tag):
        raise ValueError("Run tag must contain 1-80 ASCII letters, digits, hyphens or underscores")
    products, mapping, used_paths = [], [], set()

    def add_product(ident, category, paths):
        if not paths:
            raise ValueError(f"Product {ident} has no image references")
        urls = [media_url(base, path) for path in paths]
        product = dict(productId=ident, imageUrl=urls[0], category=category,
                       # No purchase URLs were supplied for local assets. Preview is an explicit test value.
                       purchaseLink=urls[0], imageKey=data_key(paths[0]), additionalImageUrls=urls[1:])
        products.append(product)
        for index, (path, url) in enumerate(zip(paths, urls)):
            key = data_key(path)
            if key in used_paths:
                raise ValueError(f"Reference assigned more than once: {key}")
            used_paths.add(key)
            mapping.append(dict(episode=f"episode-{number}", productId=ident, category=category,
                                role="primary" if index == 0 else "additional",
                                imageKey=key, imageUrl=url))

    for image in image_files(clothing_dir):
        # Flat clothing files provide no verified product grouping. Keep each reference separate.
        suffix = hashlib.sha256(data_key(image).encode("utf-8")).hexdigest()[:12]
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", image.stem).strip("_")[:60]
        add_product(f"EP{number}_CLOTHING_{stem}_{suffix}", "clothing", [image])

    for folder in sorted(object_dir.iterdir(), key=lambda p: p.name.casefold()):
        if not folder.is_dir():
            continue
        match = re.fullmatch(r"(OBJ\d+)_(.+)", folder.name)
        if not match:
            raise ValueError(f"Unrecognized product directory: {folder}")
        # EP prefix prevents OBJ002 (handbag in EP1 / glasses in EP2) identity collisions.
        add_product(f"EP{number}_{match.group(1)}", "non-clothing", image_files(folder))

    expected = {data_key(p) for directory in (clothing_dir, object_dir) for p in image_files(directory)}
    if used_paths != expected:
        raise ValueError(f"Incomplete asset mapping for episode {number}")
    if sum(p["category"] != "clothing" for p in products) < 2:
        raise ValueError("The main API's object margin rule needs at least two distinct products")
    if len({p["productId"] for p in products}) != len(products):
        raise ValueError("Duplicate product identities")
    return dict(jobId=f"episode-{number}-{run_tag}", seriesId="local-data-series",
                callbackUrl=callback_url or base + "/callback",
                episodes=[dict(episodeId=f"episode-{number}", title=f"Episode {number}",
                               videoUrl=media_url(base, video))], products=products), mapping


def allowed_assets():
    """Exact input-file allowlist for the local server; excludes databases and archives."""
    assets = {}
    for video_key, clothing_key, object_key in EPISODES.values():
        video = DATA / video_key
        assets[data_key(video)] = video
        for directory in (DATA / clothing_key, DATA / object_key):
            for image in image_files(directory):
                assets[data_key(image)] = image
    return assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media-base-url", default="http://127.0.0.1:8765",
                        help="Address the AI server can reach; use host.docker.internal for Docker Desktop")
    parser.add_argument("--callback-url", help="Optional backend receiver; default is the local media server /callback")
    parser.add_argument("--run-tag", default="local-v1", help="Change to submit fresh inference instead of reusing a job")
    args = parser.parse_args()
    base = checked_base(args.media_base_url)
    if args.callback_url:
        parsed = urlsplit(args.callback_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            parser.error("callback-url must be an absolute HTTP(S) URL")
    all_rows = []
    for number in EPISODES:
        payload, rows = build_payload(number, base, args.callback_url, args.run_tag)
        target = ROOT / "tests" / f"payload_episode_{number}.json"
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        clothing = sum(p["category"] == "clothing" for p in payload["products"])
        print(f"{target.name}: {clothing} clothing references, {len(payload['products']) - clothing} object products, "
              f"{len(rows)} total reference images")
        all_rows.extend(rows)
    mapping = ROOT / "tests/local_payload_mapping.csv"
    with mapping.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["episode", "productId", "category", "role", "imageKey", "imageUrl"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Full image mapping: {mapping}")


if __name__ == "__main__":
    main()

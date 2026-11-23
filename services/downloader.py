import json
from pathlib import Path

import requests
from PIL import Image


class Downloader:
    def __init__(self, base_dir="data/jobs"):
        self.base_dir = base_dir

    def _download_file(self, url: str, save_path: str):
        # Keep URL signatures out of local filenames.
        with requests.get(url, stream=True, timeout=300) as response:
            response.raise_for_status()
            with open(save_path, "wb") as destination:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        destination.write(chunk)

    def prepare_assets(self, payload: dict):
        from schemas.request import AnalyzeRequest

        payload = AnalyzeRequest.model_validate(payload).model_dump()
        job_dir = Path(self.base_dir) / payload["jobId"]
        videos_dir, clothing_dir, objects_dir = [job_dir / name for name in ("videos", "clothing", "objects")]
        for folder in (videos_dir, clothing_dir, objects_dir):
            folder.mkdir(parents=True, exist_ok=True)

        episodes = []
        for index, episode in enumerate(payload["episodes"]):
            video_path = videos_dir / f"episode_{index}.mp4"
            self._download_file(episode["videoUrl"], str(video_path))
            episodes.append(dict(episodeId=episode["episodeId"], title=episode["title"],
                                 video_path=str(video_path)))

        products, catalog = [], []
        for index, product in enumerate(payload["products"]):
            folder = clothing_dir if product["category"] == "clothing" else objects_dir
            views = []
            for view_index, url in enumerate([product["imageUrl"], *product["additionalImageUrls"]]):
                # Decode bytes and normalize losslessly; signed and extensionless URLs work.
                raw_path = folder / f"product_{index}_view_{view_index}.download"
                image_path = raw_path.with_suffix(".png")
                try:
                    self._download_file(url, str(raw_path))
                    with Image.open(raw_path) as image:
                        image.convert("RGB").save(image_path, format="PNG")
                finally:
                    raw_path.unlink(missing_ok=True)
                views.append(dict(path=image_path.name, enabled=True))
            products.append(dict(product, local_image_path=str(folder / views[0]["path"])))
            if product["category"] != "clothing":
                catalog.append(dict(product_id=product["productId"], name=product["productId"],
                                    category=product["category"], views=views))

        manifest = objects_dir / "catalog.json"
        manifest.write_text(json.dumps(dict(schema_version=1, products=catalog)), encoding="utf-8")
        return dict(job_dir=str(job_dir), videos_dir=str(videos_dir), clothing_dir=str(clothing_dir),
                    objects_dir=str(objects_dir), objects_catalog=str(manifest),
                    episodes=episodes, products=products)

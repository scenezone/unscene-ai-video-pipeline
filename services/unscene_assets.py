"""Bounded, cancellable signed-link transfers, isolated from platform credentials."""
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import requests

from schemas.unscene import CandidateManifest, media_url
from services.unscene_errors import ProcessingFailure


class AssetTransfer:
    def __init__(self, allow_http=False, max_video_bytes=20 * 1024**3,
                 max_image_bytes=30 * 1024**2, max_manifest_bytes=20 * 1024**2,
                 session=None):
        self.allow_http = allow_http
        self.max_video_bytes = max_video_bytes
        self.max_image_bytes = max_image_bytes
        self.max_manifest_bytes = max_manifest_bytes
        self.session = session or requests.Session()

    def _url(self, url):
        media_url(url)
        if not self.allow_http and urlsplit(url).scheme != "https":
            raise ProcessingFailure("invalid_media_url", "Production media links must use HTTPS")
        return url

    def download(self, source, target, limit, guard, expected_sha=None, expected_size=None):
        """source is a getter so link renewal cannot leave a stale copied URL."""
        target = Path(target)
        partial = target.with_suffix(target.suffix + ".part")
        target.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(2):
            guard.check()
            url = self._url(source())
            try:
                with self.session.get(url, stream=True, timeout=(10, 30), allow_redirects=False) as response:
                    if response.status_code in (401, 403) and attempt == 0:
                        guard.renew_links()
                        continue
                    if not 200 <= response.status_code < 300:
                        raise ProcessingFailure("download_failed", "Signed asset download failed",
                                                response.status_code in (401, 403, 408, 429) or response.status_code >= 500)
                    header = response.headers.get("Content-Length")
                    if header is not None and int(header) > limit:
                        raise ProcessingFailure("asset_too_large", "Asset exceeds the configured byte limit")
                    digest, size = hashlib.sha256(), 0
                    with partial.open("wb") as output:
                        for chunk in response.iter_content(1024 * 1024):
                            guard.check()
                            if not chunk:
                                continue
                            size += len(chunk)
                            if size > limit:
                                raise ProcessingFailure("asset_too_large", "Asset exceeds the configured byte limit")
                            output.write(chunk)
                            digest.update(chunk)
                    if expected_sha and digest.hexdigest() != expected_sha:
                        raise ProcessingFailure("checksum_mismatch", "Candidate manifest checksum does not match")
                    if expected_size is not None and size != expected_size:
                        raise ProcessingFailure("media_size_mismatch", "Downloaded video size differs from claim", True)
                    guard.check()
                    partial.replace(target)
                    return target
            except requests.RequestException:
                raise ProcessingFailure("download_failed", "Asset transfer timed out or disconnected", True) from None
            finally:
                partial.unlink(missing_ok=True)
        raise ProcessingFailure("expired_link", "Signed asset link could not be renewed", True)

    def video(self, guard, folder):
        video = guard.snapshot().video
        if video.sizeBytes > self.max_video_bytes:
            raise ProcessingFailure("asset_too_large", "Video exceeds the configured byte limit")
        return self.download(lambda: guard.snapshot().video.url, Path(folder) / "video.mp4",
                             self.max_video_bytes, guard, expected_size=video.sizeBytes)

    def manifest(self, guard, folder):
        job = guard.snapshot()
        path = self.download(lambda: guard.snapshot().candidates.manifestUrl,
                             Path(folder) / "candidates.json", self.max_manifest_bytes, guard,
                             expected_sha=job.candidates.sha256)
        manifest = CandidateManifest.model_validate_json(path.read_bytes())
        if (manifest.jobId, manifest.episodeId) != (job.jobId, job.episode.episodeId):
            raise ProcessingFailure("invalid_manifest", "Candidate manifest identifies a different job/episode")
        count = sum(len(group.products) for group in manifest.types)
        if count != job.candidates.productCount:
            raise ProcessingFailure("invalid_manifest", "Candidate count differs from claim")
        return manifest

    def catalogs(self, manifest, folder, guard):
        from PIL import Image, UnidentifiedImageError
        import json

        catalogs = {}
        for group_index, group in enumerate(manifest.types):
            guard.check()
            if not group.products or not group.segments:
                continue
            base = Path(folder) / f"type_{group_index}"
            base.mkdir(parents=True, exist_ok=True)
            products = []
            for product_index, product in enumerate(group.products):
                views = []
                for image_index, image in enumerate(product.images):
                    raw = base / f"product_{product_index}_view_{image_index}.download"
                    self.download(lambda image=image: image.url, raw, self.max_image_bytes, guard)
                    image_path = raw.with_suffix(".png")
                    try:
                        with Image.open(raw) as decoded:
                            decoded.convert("RGB").save(image_path, format="PNG")
                    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
                        raise ProcessingFailure("invalid_image", "Cannot decode candidate reference image") from None
                    finally:
                        raw.unlink(missing_ok=True)
                    guard.check()
                    views.append(dict(path=image_path.name, enabled=True,
                                      image_id=image.imageId, role=image.role))
                products.append(dict(product_id=product.productId, name=product.title,
                                     category=group.typeCode, views=views))
            catalog = base / "catalog.json"
            catalog.write_text(json.dumps(dict(schema_version=1, products=products)), encoding="utf-8")
            catalogs[group.typeCode] = catalog
        return catalogs

    def upload(self, raw, guard):
        guard.check()
        upload = guard.snapshot().cueUpload
        if upload.expiresAt.tzinfo is None:
            raise ProcessingFailure("invalid_upload", "Upload expiry needs a timezone")
        if upload.expiresAt <= datetime.now(timezone.utc):
            # Heartbeats may return cueUpload even though the PDF only illustrates GET renewal.
            guard.renew_links()
            upload = guard.snapshot().cueUpload
            if upload.expiresAt <= datetime.now(timezone.utc):
                raise ProcessingFailure("expired_upload", "Cue upload link expired; requeue MATCH", True)
        if len(raw) > min(upload.maxBytes, 10 * 1024**2):
            raise ProcessingFailure("cue_file_too_large", "Cue file exceeds upload byte limit")
        try:
            with self.session.put(self._url(upload.url), data=raw,
                                  headers={"Content-Type": upload.contentType},
                                  timeout=(10, 60), allow_redirects=False) as response:
                if not 200 <= response.status_code < 300:
                    raise ProcessingFailure("upload_failed", "Cue file upload failed",
                                            response.status_code in (401, 403, 408, 429) or response.status_code >= 500)
        except requests.RequestException:
            raise ProcessingFailure("upload_failed", "Cue file upload timed out or disconnected", True) from None
        guard.check()
        return upload.resultUrl

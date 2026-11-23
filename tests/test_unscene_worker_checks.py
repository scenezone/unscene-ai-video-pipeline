"""Offline protocol and pipeline integration. No GPU weights or network required.

Run: python -m unittest discover -s tests -p test_unscene_worker_checks.py -v
"""
import copy
import hashlib
import hmac
import io
import json
import os
import runpy
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np
import requests
from PIL import Image
from pydantic import ValidationError
from requests.adapters import BaseAdapter

from pipelines.unscene_pipeline import UnsceneVideoPipeline
from postprocessing.cue_builder import CueBuilder
from schemas.unscene import CandidateManifest, ClaimedJob, CueFile, Requirements
from services.unscene_assets import AssetTransfer
from services.unscene_client import PlatformError, UnsceneClient, json_bytes
from services.unscene_errors import LeaseLost, ProcessingFailure, SubmissionUncertain, WorkerStopped
from services.unscene_lease import LeaseGuard
from services.unscene_service import UnsceneJobService
from services.unscene_worker import UnsceneWorker, WorkerSettings


def future(seconds=600):
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


def claim(step="DETECT_TYPES"):
    result = dict(jobId="job-1", step=step, leaseExpiresAt=future(),
                  episode=dict(episodeId="ep-1", durationSeconds=1.25, width=32, height=24, fps=4),
                  video=dict(url="http://storage.test/video", expiresAt=future(21600),
                             contentType="video/mp4", sizeBytes=123))
    if step == "DETECT_TYPES":
        result.update(taxonomyVersion="2026.10", allowedTypes=[dict(code="clothing.shirts", name="Shirts")])
    else:
        result.update(candidates=dict(manifestUrl="http://storage.test/manifest", expiresAt=future(21600),
                                      sha256="a" * 64, productCount=1),
                      cueUpload=dict(url="http://storage.test/upload", method="PUT", contentType="application/json",
                                     maxBytes=10485760, expiresAt=future(21600), resultUrl="http://storage.test/result"))
    return result


def manifest_data():
    return dict(schemaVersion="1.0", jobId="job-1", episodeId="ep-1", generatedAt=future(0), types=[dict(
        typeCode="clothing.shirts", segments=[dict(startSeconds=0, endSeconds=1.25)], products=[dict(
            productId="shirt-1", title="Blue shirt", brand="Demo", attributes={"colour": "blue"},
            images=[dict(imageId="front", role="main", url="http://storage.test/front"),
                    dict(imageId="side", role="on_model", url="http://storage.test/side")])])])


class FakePlatform(BaseAdapter):
    """Actual PreparedRequests traverse OAuth, HMAC verification and signed storage."""
    def __init__(self, video=b"video"):
        super().__init__()
        self.video = video
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), "blue").save(buffer, format="PNG")
        self.image = buffer.getvalue()
        self.manifest = json_bytes(manifest_data())
        self.current_step = "DETECT_TYPES"
        self.calls, self.requirements, self.cue_metadata = [], [], []
        self.uploaded, self.token_count, self.heartbeat_count = None, 0, 0
        self.rejected_token_once = False
        self.expire_next = False
        self.failures = []
        self.extra_heartbeat = {}
        self.on_heartbeat = None
        self._lock = threading.RLock()

    def send(self, request, **kwargs):
        with self._lock:
            return self._send(request)

    def _send(self, request):
        parsed = urlsplit(request.url)
        body = request.body or b""
        raw = body.encode() if isinstance(body, str) else body
        self.calls.append((request.method, request.url, copy.deepcopy(dict(request.headers)), raw))
        response = requests.Response()
        response.request, response.url = request, request.url
        response.status_code = 200

        def reply(value, status=200):
            response.status_code, response._content = status, json_bytes(value)
            response.headers["Content-Type"] = "application/json"
            return response

        if parsed.hostname == "storage.test":
            assert not any(name.lower().startswith(("authorization", "x-unscene")) for name in request.headers)
            if request.method == "PUT":
                assert request.headers["Content-Type"] == "application/json"
                self.uploaded = raw
                response._content = b""
            else:
                response._content = {"/video": self.video, "/manifest": self.manifest,
                                     "/front": self.image, "/side": self.image,
                                     "/renewed": self.video}.get(parsed.path, b"")
            response.headers["Content-Length"] = str(len(response._content))
            response.raw = io.BytesIO(response._content)
            return response

        if parsed.path == "/ai/v1/oauth/token":
            form = parse_qs(raw.decode())
            assert form == dict(grant_type=["client_credentials"], client_id=["client"], client_secret=["secret"])
            assert "X-Unscene-Signature" not in request.headers
            self.token_count += 1
            return reply(dict(access_token=f"token-{self.token_count}", expires_in=600, token_type="Bearer"))

        timestamp = request.headers["X-Unscene-Timestamp"]
        message = f"{timestamp}.POST.{request.path_url}.{hashlib.sha256(raw).hexdigest()}"
        signature = "v1=" + hmac.new(b"secret", message.encode(), hashlib.sha256).hexdigest()
        assert request.headers["X-Unscene-Signature"] == signature
        assert request.headers["X-Unscene-Key-Id"] == "client"
        assert request.headers["Authorization"] == f"Bearer token-{self.token_count}"
        if self.expire_next:
            self.expire_next = False
            return reply({"error": {"code": "invalid_token"}}, 401)
        data = json.loads(raw)
        if parsed.path.endswith("/claim"):
            if self.current_step is None:
                return reply({"job": None})
            job = claim(self.current_step)
            job["video"]["sizeBytes"] = len(self.video)
            if job["step"] == "MATCH":
                job["candidates"]["sha256"] = hashlib.sha256(self.manifest).hexdigest()
            return reply({"job": job})
        if parsed.path.endswith("/heartbeat"):
            self.heartbeat_count += 1
            if self.on_heartbeat:
                return self.on_heartbeat(reply)
            return reply(dict(leaseExpiresAt=future(), **self.extra_heartbeat))
        if parsed.path.endswith("/requirements"):
            self.requirements.append(data)
            self.current_step = "MATCH"
            return reply(dict(state="queued", step="MATCH"), 202)
        if parsed.path.endswith("/cues"):
            assert data["cueFileUrl"] == "http://storage.test/result"
            assert data["sha256"] == hashlib.sha256(self.uploaded).hexdigest()
            assert data["cueCount"] == len(json.loads(self.uploaded)["cues"])
            self.cue_metadata.append(data)
            self.current_step = None
            return reply(dict(state="waiting", step="INGEST"), 202)
        if parsed.path.endswith("/fail"):
            self.failures.append(data)
            self.current_step = None
            return reply(dict(state="failed"), 202)
        return reply({"error": {"code": "not_found"}}, 404)

    def close(self):
        pass


def client_for(platform):
    def session():
        result = requests.Session()
        result.mount("http://", platform)
        return result
    return UnsceneClient("http://localhost:4380/ai/v1", "client", "secret", session_factory=session)


def storage_for(platform, **kwargs):
    session = requests.Session()
    session.mount("http://", platform)
    return AssetTransfer(allow_http=True, session=session, **kwargs)


def settings(**kwargs):
    return WorkerSettings(base_url="http://localhost:4380/ai/v1", client_id="client", client_secret="secret",
                          worker_id="test-worker", **kwargs)


class ContractChecks(unittest.TestCase):
    def test_step_specific_contracts_and_timezone(self):
        ClaimedJob.model_validate(claim())
        ClaimedJob.model_validate(claim("MATCH"))
        for change in (dict(allowedTypes=None), dict(taxonomyVersion=None),
                       dict(leaseExpiresAt="2026-10-20T00:00:00")):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                ClaimedJob.model_validate(dict(claim(), **change))

    def test_duplicate_candidates_and_checksum_schema(self):
        data = manifest_data()
        data["types"][0]["products"] *= 2
        with self.assertRaises(ValidationError):
            CandidateManifest.model_validate(data)
        data = claim("MATCH")
        data["candidates"]["sha256"] = "wrong"
        with self.assertRaises(ValidationError):
            ClaimedJob.model_validate(data)

    def test_cue_context_rejects_unknown_products_types_and_duration(self):
        job, manifest = ClaimedJob.model_validate(claim("MATCH")), CandidateManifest.model_validate(manifest_data())
        valid = dict(cueId="c1", productId="shirt-1", typeCode="clothing.shirts", matchType="similar",
                     confidence=.8, startSeconds=0, endSeconds=1)
        for mutation in (dict(productId="not-offered"), dict(typeCode="bags.handbags"), dict(endSeconds=2)):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                CueFile(jobId=job.jobId, episodeId=job.episode.episodeId, modelVersion="test",
                        cues=[dict(valid, **mutation)]).validate_context(job, manifest)
        with self.assertRaises(ValueError):
            CueFile(jobId=job.jobId, episodeId=job.episode.episodeId, modelVersion="test",
                    cues=[valid, valid]).validate_context(job, manifest)

    def test_cue_boxes_and_limits(self):
        base = dict(jobId="job-1", episodeId="ep-1", modelVersion="test", cues=[dict(
            cueId="c", productId="shirt-1", typeCode="clothing.shirts", matchType="similar", confidence=.8,
            startSeconds=0, endSeconds=1, boxes=[dict(atSeconds=.5, x=.1, y=.2, w=.3, h=.4)])])
        for mutation in (dict(x=.9), dict(atSeconds=1), dict(w=0)):
            data = copy.deepcopy(base)
            data["cues"][0]["boxes"][0].update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                CueFile.model_validate(data)
        base["cues"][0]["boxes"] *= 201
        with self.assertRaises(ValueError):
            CueFile.model_validate(base)

    def test_invalid_settings_fail_before_work(self):
        for values in (dict(steps=("WRONG",)), dict(heartbeat_interval=1), dict(poll_interval=0),
                       dict(target_fps=float("nan")), dict(single_candidate_threshold=.4), dict(max_video_bytes=0)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                settings(**values)


class ClientChecks(unittest.TestCase):
    def test_oauth_form_signatures_unicode_raw_bytes_and_cached_token(self):
        platform = FakePlatform()
        client = client_for(platform)
        client.claim("test", ["DETECT_TYPES"])
        client.job_post("job-1", "requirements", {"label": "sac à main", "types": []})
        self.assertEqual(platform.token_count, 1)
        self.assertIn("à".encode(), platform.calls[-1][3])

    def test_expired_token_refreshes_once_and_resigns(self):
        platform = FakePlatform()
        platform.expire_next = True
        client_for(platform).claim("test", ["MATCH"])
        self.assertEqual(platform.token_count, 2)
        claims = [call for call in platform.calls if call[1].endswith("/claim")]
        self.assertEqual(len(claims), 2)
        self.assertEqual(claims[0][3], claims[1][3])

    def test_expired_cached_token_renews_before_next_call(self):
        platform = FakePlatform()
        client = client_for(platform)
        client.claim("test", ["MATCH"])
        client._expires = 0
        client.claim("test", ["MATCH"])
        self.assertEqual(platform.token_count, 2)

    def test_ambiguous_claim_is_never_automatically_replayed(self):
        session = Mock()
        session.post.side_effect = requests.Timeout("sensitive URL")
        client = UnsceneClient("http://localhost:4380/ai/v1", "id", "secret", session_factory=lambda: session)
        client._token, client._expires = "token", time.monotonic() + 600
        with self.assertRaises(requests.Timeout):
            client.claim("worker", ["MATCH"])
        self.assertEqual(session.post.call_count, 1)

    def test_base_url_and_credentials_are_validated(self):
        for url in ("http://production.test/ai/v1", "https://host/other", "https://u:p@host/ai/v1",
                    "https://host/ai/v1?query=1"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                UnsceneClient(url, "id", "secret")


class LeaseChecks(unittest.TestCase):
    def test_successful_submission_cannot_be_submitted_twice_or_heartbeat_after_release(self):
        platform = FakePlatform()
        with LeaseGuard(client_for(platform), ClaimedJob.model_validate(claim()), threading.Event()) as guard:
            guard.submit("requirements", {"types": []})
            guard.renew_links()
            with self.assertRaises(LeaseLost):
                guard.submit("requirements", {"types": []})
        self.assertEqual(len(platform.requirements), 1)
        self.assertEqual(platform.heartbeat_count, 0)

    def test_heartbeats_run_independently_and_apply_partial_link_renewals(self):
        platform = FakePlatform()
        platform.extra_heartbeat = dict(video={"url": "http://storage.test/renewed", "expiresAt": future(21600)})
        job = ClaimedJob.model_validate(claim())
        with LeaseGuard(client_for(platform), job, threading.Event(), .01) as guard:
            deadline = time.monotonic() + 2
            while platform.heartbeat_count == 0 and time.monotonic() < deadline:
                time.sleep(.01)
            guard.check()
            self.assertGreater(platform.heartbeat_count, 0)
            self.assertEqual(guard.snapshot().video.url, "http://storage.test/renewed")
            self.assertEqual(guard.snapshot().video.sizeBytes, job.video.sizeBytes)

    def test_lease_loss_stops_work(self):
        platform = FakePlatform()
        platform.on_heartbeat = lambda reply: reply({"error": {"code": "lease_lost"}}, 409)
        with LeaseGuard(client_for(platform), ClaimedJob.model_validate(claim()), threading.Event(), 45) as guard:
            with self.assertRaises(LeaseLost):
                guard.renew_links()
            with self.assertRaises(LeaseLost):
                guard.submit("requirements", {})
        self.assertEqual(platform.requirements, [])

    def test_expired_lease_and_shutdown_are_cancelled(self):
        job = ClaimedJob.model_validate(dict(claim(), leaseExpiresAt=future(-1)))
        guard = LeaseGuard(Mock(), job, threading.Event())
        with self.assertRaises(LeaseLost):
            guard.check()
        stopped = threading.Event()
        stopped.set()
        with self.assertRaises(WorkerStopped):
            LeaseGuard(Mock(), ClaimedJob.model_validate(claim()), stopped).check()

    def test_changed_manifest_invalidates_work(self):
        platform = FakePlatform()
        platform.extra_heartbeat = dict(candidates={"sha256": "b" * 64})
        with LeaseGuard(client_for(platform), ClaimedJob.model_validate(claim("MATCH")), threading.Event()) as guard:
            with self.assertRaises(LeaseLost):
                guard.renew_links()

    def test_ambiguous_submission_is_not_replayed(self):
        client = Mock(timeout=1)
        client.job_post.side_effect = requests.Timeout()
        with LeaseGuard(client, ClaimedJob.model_validate(claim()), threading.Event()) as guard:
            with self.assertRaises(SubmissionUncertain):
                guard.submit("requirements", {"types": []})
        self.assertEqual(client.job_post.call_count, 1)


class AssetChecks(unittest.TestCase):
    def guard(self, platform, step="MATCH"):
        data = claim(step)
        data["video"]["sizeBytes"] = len(platform.video)
        if step == "MATCH":
            data["candidates"]["sha256"] = hashlib.sha256(platform.manifest).hexdigest()
        return LeaseGuard(client_for(platform), ClaimedJob.model_validate(data), threading.Event())

    def test_manifest_integrity_and_multiple_reference_views(self):
        platform = FakePlatform()
        transfer = storage_for(platform)
        with tempfile.TemporaryDirectory() as directory, self.guard(platform) as guard:
            manifest = transfer.manifest(guard, directory)
            catalogs = transfer.catalogs(manifest, directory, guard)
            catalog = json.loads(catalogs["clothing.shirts"].read_text())
            self.assertEqual(catalog["products"][0]["product_id"], "shirt-1")
            self.assertEqual(len(catalog["products"][0]["views"]), 2)
            self.assertEqual([v["role"] for v in catalog["products"][0]["views"]], ["main", "on_model"])
            self.assertFalse(list(Path(directory).rglob("*.part")))

    def test_wrong_manifest_checksum_rejected_before_parsing(self):
        platform = FakePlatform()
        with tempfile.TemporaryDirectory() as directory, self.guard(platform) as guard:
            platform.manifest += b" "
            with self.assertRaises(ProcessingFailure) as caught:
                storage_for(platform).manifest(guard, directory)
            self.assertEqual(caught.exception.code, "checksum_mismatch")
            self.assertFalse(list(Path(directory).rglob("*.part")))

    def test_download_byte_limit_and_video_size(self):
        platform = FakePlatform()
        with tempfile.TemporaryDirectory() as directory, self.guard(platform) as guard:
            with self.assertRaises(ProcessingFailure) as caught:
                storage_for(platform, max_video_bytes=1).video(guard, directory)
            self.assertEqual(caught.exception.code, "asset_too_large")
            with self.assertRaises(ProcessingFailure) as caught:
                storage_for(platform).download(lambda: "http://storage.test/video", Path(directory) / "v",
                                               100, guard, expected_size=10)
            self.assertEqual(caught.exception.code, "media_size_mismatch")

    def test_upload_limits_expiry_and_no_credentials_leak(self):
        platform = FakePlatform()
        with self.guard(platform) as guard:
            result = storage_for(platform).upload(b"{}", guard)
            self.assertEqual(result, "http://storage.test/result")
            self.assertEqual(platform.uploaded, b"{}")
            with self.assertRaises(ProcessingFailure):
                storage_for(platform).upload(b"x" * (10485760 + 1), guard)
            guard._job.cueUpload.expiresAt = datetime.now(timezone.utc) - timedelta(seconds=1)
            with self.assertRaises(ProcessingFailure) as caught:
                storage_for(platform).upload(b"{}", guard)
            self.assertEqual(caught.exception.code, "expired_upload")

    def test_production_requires_https_assets(self):
        with self.assertRaises(ProcessingFailure):
            AssetTransfer()._url("http://storage.test/file")

    def test_expired_download_refreshes_and_uses_new_link(self):
        platform = FakePlatform()
        platform.extra_heartbeat = dict(video={"url": "http://storage.test/renewed", "expiresAt": future(21600)})
        original = platform._send
        def send(request):
            response = original(request)
            if request.url == "http://storage.test/video":
                response.status_code = 403
            return response
        platform._send = send
        with tempfile.TemporaryDirectory() as directory, self.guard(platform) as guard:
            target = storage_for(platform).video(guard, directory)
            self.assertEqual(target.read_bytes(), platform.video)
        self.assertEqual(platform.heartbeat_count, 1)

    def test_wrong_job_identity_and_count_are_rejected(self):
        for mutate in (lambda d: d.update(jobId="someone-elses-job"),
                       lambda d: d["types"][0]["products"].clear()):
            platform = FakePlatform()
            data = manifest_data()
            mutate(data)
            platform.manifest = json_bytes(data)
            with tempfile.TemporaryDirectory() as directory, self.guard(platform) as guard:
                with self.assertRaises(ProcessingFailure) as caught:
                    storage_for(platform).manifest(guard, directory)
                self.assertEqual(caught.exception.code, "invalid_manifest")

    def test_transfer_cancellation_cleans_partial_file(self):
        platform = FakePlatform()
        with tempfile.TemporaryDirectory() as directory, self.guard(platform) as guard:
            original = guard.check
            count = 0
            def cancelled():
                nonlocal count
                count += 1
                if count > 1:
                    raise LeaseLost()
                original()
            guard.check = cancelled
            with self.assertRaises(LeaseLost):
                storage_for(platform).video(guard, directory)
            self.assertFalse(list(Path(directory).rglob("*.part")))


class MatchingPolicyChecks(unittest.TestCase):
    def test_production_matcher_singleton_is_explicit_and_uses_higher_threshold(self):
        # Only tensor inference is replaced: execute the production match method.
        class Base:
            def __init__(self, *_):
                self.product_embeddings = {"shirt-1": object()}

        with patch.dict(sys.modules, {"torch": SimpleNamespace(),
                        "services.matching.clip_matcher": SimpleNamespace(CLIPMatcher=Base)}):
            matcher_type = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                                             "services/matching/product_catalog_matcher.py"))["ProductCatalogMatcher"]
        matcher = matcher_type(None, None)
        matcher.get_embedding = lambda crop: None
        matcher.rank_products = lambda _: [dict(product="shirt-1", similarity=.69)]
        self.assertFalse(matcher.match(None, single_candidate_threshold=.70)[0])
        matcher.rank_products = lambda _: [dict(product="shirt-1", similarity=.75)]
        self.assertTrue(matcher.match(None, single_candidate_threshold=.70)[0])
        self.assertFalse(matcher.match(None)[0])  # compatibility policy still requires a competitor
        matcher.rank_products = lambda _: [dict(product="shirt-1", similarity=.75),
                                          dict(product="shirt-2", similarity=.74)]
        self.assertFalse(matcher.match(None, single_candidate_threshold=.70)[0])


class CueBuilderChecks(unittest.TestCase):
    def test_moving_boxes_do_not_fragment_cues_fractional_times_are_preserved(self):
        job = ClaimedJob.model_validate(claim("MATCH"))
        manifest = CandidateManifest.model_validate(manifest_data())
        rows = [dict(productId="shirt-1", typeCode="clothing.shirts", matchType="similar", time=t,
                     confidence=.8, box=dict(x=.1 + t / 10, y=.2, w=.3, h=.4)) for t in (.25, .5, .75, 1.)]
        result = CueBuilder().cues(rows, job, manifest, .25, "test")
        self.assertEqual(len(result.cues), 1)
        self.assertEqual((result.cues[0].startSeconds, result.cues[0].endSeconds), (.25, 1.25))
        self.assertEqual(len(result.cues[0].boxes), 4)

    def test_boxes_are_sampled_across_full_interval_and_capped(self):
        job = ClaimedJob.model_validate(claim("MATCH"))
        job.episode.durationSeconds = 1000
        manifest = CandidateManifest.model_validate(manifest_data())
        rows = [dict(productId="shirt-1", typeCode="clothing.shirts", matchType="similar", time=float(t),
                     confidence=.8, box=dict(x=.1, y=.2, w=.3, h=.4)) for t in range(1000)]
        result = CueBuilder().cues(rows, job, manifest, 1, "test")
        self.assertEqual(len(result.cues[0].boxes), 200)
        self.assertEqual(result.cues[0].boxes[-1].atSeconds, 999)

    def test_known_segment_gap_is_not_bridged(self):
        job = ClaimedJob.model_validate(claim("MATCH"))
        manifest = CandidateManifest.model_validate(manifest_data())
        base = dict(productId="shirt-1", typeCode="clothing.shirts", matchType="similar", confidence=.8,
                    box=dict(x=.1, y=.2, w=.3, h=.4))
        result = CueBuilder().cues([dict(base, time=0., endLimit=.1), dict(base, time=.25, endLimit=1.)],
                                    job, manifest, .25, "test")
        self.assertEqual(len(result.cues), 2)
        self.assertEqual(result.cues[0].endSeconds, .1)


class PipelineChecks(unittest.TestCase):
    def pipeline(self, **kwargs):
        detector = Mock()
        detector.detect.return_value = [SimpleNamespace(bbox=(.5, .5, .5, .5), score=.9, label="shirt")]
        predictor, segmenter = Mock(), Mock()
        segmenter.segment_with_area_ratio.side_effect = lambda frame, bbox: (frame.copy(), .8)
        matcher = Mock()
        matcher.match.return_value = (True, "shirt-1", .8)
        return UnsceneVideoPipeline(detector, segmenter, predictor, lambda: matcher, target_fps=4, **kwargs)

    def frames(self):
        return ((t, np.full((24, 32, 3), 100, np.uint8)) for t in (0., .25, .5, .75, 1.))

    def test_detection_uses_allowed_names_without_catalogue_images(self):
        pipeline = self.pipeline()
        job = ClaimedJob.model_validate(claim())
        pipeline.frames = lambda *_: self.frames()
        result = pipeline.detect_types("video", job, lambda: None)
        self.assertEqual(result.types[0].typeCode, "clothing.shirts")
        self.assertEqual(result.types[0].segments[0].endSeconds, 1.25)
        self.assertTrue(all(c.args[1] == "Shirts ." for c in pipeline.detector.detect.call_args_list))
        pipeline.matcher_factory().match.assert_not_called()

    def test_clothing_product_matching_boxes_and_singleton_policy(self):
        pipeline = self.pipeline()
        pipeline.frames = lambda *_: self.frames()
        result = pipeline.match("video", ClaimedJob.model_validate(claim("MATCH")),
                                CandidateManifest.model_validate(manifest_data()), {"clothing.shirts": "catalog"}, lambda: None)
        cue = result.cues[0]
        self.assertEqual((cue.productId, cue.matchType), ("shirt-1", "similar"))
        self.assertEqual(cue.boxes[0].x, .25)
        self.assertEqual(cue.boxes[0].w, .5)
        self.assertEqual(cue.confidence, .8)
        self.assertEqual(pipeline.matcher_factory().match.call_args.kwargs["single_candidate_threshold"], .70)

    def test_matching_respects_candidate_segments_and_box_fallback(self):
        pipeline = self.pipeline()
        matcher = pipeline.matcher_factory()
        matcher.match.side_effect = [(False, "shirt-1", .5), (True, "shirt-1", .8)]
        pipeline.frames = lambda *_: self.frames()
        manifest = CandidateManifest.model_validate(manifest_data())
        manifest.types[0].segments[0].startSeconds = .5
        manifest.types[0].segments[0].endSeconds = .75
        result = pipeline.match("video", ClaimedJob.model_validate(claim("MATCH")), manifest,
                                {"clothing.shirts": "catalog"}, lambda: None)
        self.assertEqual(matcher.match.call_count, 2)
        self.assertEqual((result.cues[0].startSeconds, result.cues[0].endSeconds), (.5, .75))

    def test_no_candidate_products_produces_empty_cue_file(self):
        pipeline = self.pipeline()
        result = pipeline.match(None, ClaimedJob.model_validate(claim("MATCH")),
                                CandidateManifest.model_validate(manifest_data()), {}, lambda: None)
        self.assertEqual(result.cues, [])
        pipeline.detector.detect.assert_not_called()

    def test_exact_requires_independent_verifier(self):
        pipeline = self.pipeline(exact_verifier=lambda crop, product: product.productId == "shirt-1")
        pipeline.frames = lambda *_: self.frames()
        result = pipeline.match("video", ClaimedJob.model_validate(claim("MATCH")),
                                CandidateManifest.model_validate(manifest_data()), {"clothing.shirts": "catalog"}, lambda: None)
        self.assertEqual(result.cues[0].matchType, "exact")

    def test_frame_capture_released_on_cancellation(self):
        pipeline = self.pipeline()
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.return_value = 4
        def cancelled():
            raise LeaseLost()
        with patch("pipelines.unscene_pipeline.cv2.VideoCapture", return_value=capture):
            with self.assertRaises(LeaseLost):
                list(pipeline.frames("video", ClaimedJob.model_validate(claim()), cancelled))
        capture.release.assert_called_once()


class WorkerIntegrationChecks(unittest.TestCase):
    def test_main_defaults_to_worker_and_checks_credentials_before_loading_models(self):
        environment = {"UNSCENE_CLIENT_ID": "", "UNSCENE_CLIENT_SECRET": ""}
        with patch.dict(os.environ, environment), \
             patch.dict(sys.modules, {"dotenv": SimpleNamespace(load_dotenv=lambda: None)}):
            with self.assertRaises(SystemExit) as caught:
                runpy.run_path(str(Path(__file__).resolve().parents[1] / "main.py"), run_name="__main__")
        self.assertIn("UNSCENE_CLIENT_ID", str(caught.exception))

    def test_model_registry_loads_detector_first_and_reuses_models_for_later_steps(self):
        dino, sam, clip = Mock(), Mock(), Mock()
        dino.return_value.load.return_value = object()
        sam.return_value.load.return_value = object()
        clip.return_value.load.return_value = (object(), object())
        with patch.dict(sys.modules, {
            "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
            "models.dino_loader": SimpleNamespace(DinoLoader=dino),
            "models.sam_loader": SimpleNamespace(SamLoader=sam),
            "models.clip_loader": SimpleNamespace(CLIPLoader=clip),
        }):
            registry_type = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                                               "services/model_registry.py"))["ModelRegistry"]
        registry = registry_type()
        registry.load_for_step("DETECT_TYPES")
        dino.return_value.load.assert_called_once()
        sam.assert_not_called()
        clip.assert_not_called()
        registry.load_for_step("MATCH")
        registry.load_for_step("MATCH")
        sam.return_value.load.assert_called_once()
        clip.return_value.load.assert_called_once()
        dino.return_value.load.assert_called_once()

    def test_complete_two_stage_flow_real_video_download_manifest_upload_and_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "source.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 4, (32, 24))
            self.assertTrue(writer.isOpened())
            for _ in range(5):
                writer.write(np.full((24, 32, 3), 100, np.uint8))
            writer.release()
            platform = FakePlatform(video.read_bytes())
            pipeline = PipelineChecks().pipeline()
            service = UnsceneJobService(lambda step: pipeline, storage_for(platform), root / "work", root / "output")
            worker = UnsceneWorker(settings(), client=client_for(platform), service_factory=lambda: service)
            self.assertTrue(worker.run_once())
            self.assertTrue(worker.run_once())
            self.assertFalse(worker.run_once())
            self.assertEqual(platform.requirements[0]["types"][0]["typeCode"], "clothing.shirts")
            cues = CueFile.model_validate_json(platform.uploaded)
            self.assertEqual(cues.cues[0].productId, "shirt-1")
            self.assertEqual(cues.cues[0].endSeconds, 1.25)
            self.assertEqual(platform.cue_metadata[0]["cueCount"], 1)
            self.assertEqual(platform.token_count, 1)
            self.assertFalse(platform.failures)
            self.assertEqual(len(list((root / "output").rglob("*.json"))), 2)
            self.assertEqual(list((root / "work").iterdir()), [])

    def test_failed_processing_is_reported_without_callback(self):
        platform = FakePlatform()
        service = Mock()
        service.analyze.side_effect = ProcessingFailure("decode_error", "Could not decode video", False)
        worker = UnsceneWorker(settings(), client_for(platform), lambda: service)
        worker.run_once()
        self.assertEqual(platform.failures, [dict(code="decode_error", message="Could not decode video", retryable=False)])

    def test_out_of_memory_is_retryable(self):
        platform = FakePlatform()
        service = Mock()
        service.analyze.side_effect = RuntimeError("CUDA out of memory")
        UnsceneWorker(settings(), client_for(platform), lambda: service).run_once()
        self.assertEqual(platform.failures[0]["code"], "out_of_memory")
        self.assertTrue(platform.failures[0]["retryable"])

    def test_lease_lost_during_processing_never_submits_result_or_fail(self):
        platform = FakePlatform()
        service = Mock()
        service.analyze.side_effect = LeaseLost()
        UnsceneWorker(settings(), client_for(platform), lambda: service).run_once()
        self.assertFalse(platform.requirements)
        self.assertFalse(platform.failures)


if __name__ == "__main__":
    unittest.main()

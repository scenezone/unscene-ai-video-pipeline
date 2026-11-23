"""Offline regression checks for the real API, queue, downloads, and result mapping."""
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from schemas.request import AnalyzeRequest
from services.callback_service import CallbackService
from services.downloader import Downloader
from services.job_service import JobService
from services.job_store import JobConflict, JobStore
from services.job_worker import JobWorker
from postprocessing.response_builder import ResponseBuilder


def payload(job_id="job-1"):
    return dict(jobId=job_id, seriesId="series-1", callbackUrl="https://backend.example/results",
                episodes=[dict(episodeId="ep-1", title="Episode", videoUrl="https://media.example/video?sig=test")],
                products=[dict(productId=ident, category="non-clothing", imageUrl=url,
                               imageKey=ident, purchaseLink="https://shop.example/" + ident)
                          for ident, url in (("bag", "https://media.example/bag.PNG?signature=test"),
                                             ("glasses", "https://media.example/image?id=2"))])


def completed(request):
    return dict(jobId=request["jobId"], seriesId=request["seriesId"], status="completed", episodes=[])


def wait_until(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.02)
    raise AssertionError("Timed out waiting for worker")


class BackendChecks(unittest.TestCase):
    def test_validation_clothing_single_object_duplicates_and_paths(self):
        valid = payload()
        self.assertEqual(AnalyzeRequest.model_validate(valid).products[0].additionalImageUrls, [])
        clothing = payload()
        clothing["products"] = [dict(clothing["products"][0], category="clothing")]
        AnalyzeRequest.model_validate(clothing)
        invalid = []
        for field in ("jobId",):
            item = payload(); item[field] = "../escape"; invalid.append(item)
        item = payload(); item["episodes"][0]["episodeId"] = "../../escape"; invalid.append(item)
        item = payload(); item["products"][0]["productId"] = "../escape"; invalid.append(item)
        item = payload(); item["products"] = item["products"][:1]; invalid.append(item)
        item = payload(); item["products"][1]["productId"] = "bag"; invalid.append(item)
        item = payload(); item["episodes"] *= 2; invalid.append(item)
        item = payload(); item["episodes"] = []; invalid.append(item)
        item = payload(); item["callbackUrl"] = "file:///tmp/result"; invalid.append(item)
        item = payload(); item["products"][0]["additionalImageUrls"] = ["invalid"]; invalid.append(item)
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(ValidationError):
                AnalyzeRequest.model_validate(item)

    def test_signed_extensionless_images_and_multiple_views_preserve_ids(self):
        request = payload()
        request["products"][0]["additionalImageUrls"] = ["https://media.example/side.webp?token=test"]
        seen = []
        def download(url, target):
            seen.append(url)
            if "video?" in url:
                Path(target).write_bytes(b"mock video")
            else:
                Image.new("RGB", (3, 3), (10, 20, 30)).save(target, format="PNG")
        with tempfile.TemporaryDirectory() as directory:
            downloader = Downloader(directory)
            downloader._download_file = download
            assets = downloader.prepare_assets(request)
            catalog = json.loads(Path(assets["objects_catalog"]).read_text())
            self.assertEqual([p["product_id"] for p in catalog["products"]], ["bag", "glasses"])
            self.assertEqual([len(p["views"]) for p in catalog["products"]], [2, 1])
            paths = [Path(assets["objects_dir"]) / v["path"] for p in catalog["products"] for v in p["views"]]
            self.assertEqual(len(set(paths)), 3)
            for path in paths:
                with Image.open(path) as image:
                    self.assertEqual(image.getpixel((0, 0)), (10, 20, 30))
            self.assertFalse(list(Path(directory).rglob("*.download")))
        self.assertIn(request["products"][0]["imageUrl"], seen)
        self.assertIn(request["products"][1]["imageUrl"], seen)

    def test_download_http_errors_close_response(self):
        response = Mock()
        response.raise_for_status.side_effect = requests.HTTPError("404")
        manager = Mock(__enter__=Mock(return_value=response), __exit__=Mock(return_value=False))
        with patch("services.downloader.requests.get", return_value=manager):
            with self.assertRaises(requests.HTTPError):
                Downloader()._download_file("https://media.example/missing", "unused")
        manager.__exit__.assert_called_once()

    def test_response_maps_ids_stems_and_all_image_extensions(self):
        product = payload()["products"][0]
        for extension in (".jpg", ".jpeg", ".png", ".webp", ".JPG"):
            for key in ("bag", "bag" + extension):
                with self.subTest(extension=extension, key=key):
                    result = ResponseBuilder().build("job", "series", [dict(episodeId="ep", ranges=dict(
                        objects={key: [dict(start=1, end=2, bbox=[1, 2, 3, 4])]}))],
                        [dict(product, local_image_path="objects/bag" + extension)])
                    self.assertEqual(result["episodes"][0]["objects"][0]["seriesProductId"], "bag")
        with self.assertRaisesRegex(ValueError, "no request metadata"):
            ResponseBuilder().build("job", "series", [dict(episodeId="ep", ranges=dict(objects={"missing": []}))], [])

    def test_job_service_isolates_episode_outputs_and_supports_clothing_only(self):
        request = payload()
        request["products"] = [dict(request["products"][0], category="clothing")]
        request["episodes"].append(dict(request["episodes"][0], episodeId="ep-2"))
        with tempfile.TemporaryDirectory() as directory:
            factory = Mock()
            factory.create.return_value.process_video.return_value = dict(objects={}, clothing={})
            prompts = Mock(return_value=SimpleNamespace(generate_prompt=lambda: dict(clothing_prompt="shirt", object_prompt="")))
            service = JobService(factory, SimpleNamespace(device="cpu"), directory, prompts)
            service.downloader = Mock()
            service.downloader.prepare_assets.return_value = dict(clothing_dir="clothes", objects_dir="objects",
                objects_catalog="objects/catalog.json", products=[], episodes=[dict(episodeId=e["episodeId"], video_path="video") for e in request["episodes"]])
            result = service.analyze(request)
            self.assertEqual(len(result["episodes"]), 2)
            calls = factory.create.call_args_list
            self.assertEqual([Path(c.kwargs["output_dir"]).parts[-2:] for c in calls], [("job-1", "ep-1"), ("job-1", "ep-2")])
            self.assertTrue(all(c.kwargs["object_prompt"] == "" for c in calls))
            request["jobId"] = "job-2"
            service.analyze(request)
            self.assertEqual(Path(factory.create.call_args.kwargs["output_dir"]).parts[-2:], ("job-2", "ep-2"))

    def test_callback_retries_transient_errors_and_requires_2xx(self):
        response = Mock(status_code=204)
        manager = Mock(__enter__=Mock(return_value=response), __exit__=Mock(return_value=False))
        with patch("services.callback_service.requests.post", side_effect=[requests.Timeout(), manager]) as post, \
             patch("services.callback_service.time.sleep") as sleep:
            CallbackService().send("https://backend.example/results", {"status": "completed"})
            self.assertEqual(post.call_count, 2)
            sleep.assert_called_once_with(1)
            self.assertFalse(post.call_args.kwargs["allow_redirects"])
        for status, attempts in ((500, 3), (429, 3), (400, 1), (302, 1)):
            response.status_code = status
            with self.subTest(status=status), patch("services.callback_service.requests.post", return_value=manager) as post, \
                 patch("services.callback_service.time.sleep"), self.assertRaises(RuntimeError):
                CallbackService().send("https://backend.example/results", {})
            self.assertEqual(post.call_count, attempts)

    def test_persistent_queue_deduplication_and_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "jobs.sqlite3"
            store = JobStore(path)
            self.assertTrue(store.submit(payload()))
            self.assertFalse(store.submit(payload()))
            changed = payload(); changed["seriesId"] = "different"
            with self.assertRaises(JobConflict):
                store.submit(changed)
            store.mark_running("job-1")
            restored = JobStore(path)
            restored.recover()
            self.assertEqual(restored.get("job-1")["status"], "queued")
            restored.save_result("job-1", completed(payload()))
            restarted = JobStore(path)
            restarted.recover()
            self.assertEqual(restarted.next_work()["result"]["status"], "completed")

    def test_failure_callback_and_retry_do_not_rerun_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory) / "jobs.sqlite3")
            store.submit(payload())
            service = Mock()
            service.analyze.side_effect = RuntimeError("model failed")
            service.callback_service.send.side_effect = RuntimeError("backend down")
            worker = JobWorker(store, lambda: service)
            with self.assertLogs("services.job_worker", level="ERROR"):
                worker.process_next(service)
            state = store.get("job-1")
            self.assertEqual((state["status"], state["callbackStatus"]), ("failed", "failed"))
            self.assertEqual(state["result"]["error"]["code"], "PROCESSING_FAILED")
            store.retry_callback("job-1")
            service.callback_service.send.side_effect = None
            worker.process_next(service)
            self.assertEqual(store.get("job-1")["callbackStatus"], "delivered")
            service.analyze.assert_called_once()
            self.assertEqual(service.callback_service.send.call_args.args[1]["status"], "failed")

    def test_success_is_preserved_when_callback_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory) / "jobs.sqlite3")
            store.submit(payload())
            service = Mock()
            service.analyze.side_effect = completed
            service.callback_service.send.side_effect = RuntimeError("offline")
            worker = JobWorker(store, lambda: service)
            with self.assertLogs("services.job_worker", level="ERROR"):
                worker.process_next(service)
            state = store.get("job-1")
            self.assertEqual(state["status"], "completed")
            self.assertEqual(state["callbackStatus"], "failed")
            store.retry_callback("job-1")
            service.callback_service.send.side_effect = None
            worker.process_next(service)
            service.analyze.assert_called_once()

    def test_api_validation_deduplication_polling_and_callback_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            # Import without loading the model stack or running startup.
            with patch.dict("os.environ", {"AI_JOB_DB": str(Path(directory) / "default.sqlite3")}):
                from main import create_app
            store = JobStore(Path(directory) / "api.sqlite3")
            app = create_app(lambda: None, store)
            client = TestClient(app)
            self.assertEqual(client.get("/healthy").json(), {"status": "healthy"})
            self.assertEqual(client.post("/v1/analyze/jobs", json=payload()).json(), dict(status="accepted", jobId="job-1"))
            self.assertEqual(client.post("/v1/analyze/jobs", json=payload()).status_code, 200)
            different = payload(); different["seriesId"] = "different"
            self.assertEqual(client.post("/v1/analyze/jobs", json=different).status_code, 409)
            bad = payload(); bad["products"] = bad["products"][:1]
            self.assertEqual(client.post("/v1/analyze/jobs", json=bad).status_code, 422)
            self.assertEqual(client.get("/v1/analyze/jobs/job-1").json()["status"], "queued")
            self.assertEqual(client.get("/v1/analyze/jobs/unknown").status_code, 404)
            retry = "/v1/analyze/jobs/job-1/callback"
            self.assertEqual(client.post(retry).status_code, 409)
            store.save_result("job-1", completed(payload()))
            store.callback_finished("job-1", "offline")
            self.assertEqual(client.post(retry).status_code, 200)
            self.assertEqual(client.get("/v1/analyze/jobs/job-1").json()["callbackStatus"], "pending")

    def test_two_workers_have_one_model_owner_and_recover_on_takeover(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory) / "jobs.sqlite3")
            service = Mock()
            started, release = threading.Event(), threading.Event()
            active = 0
            maximum = 0
            def analyze(request):
                nonlocal active, maximum
                active += 1
                maximum = max(maximum, active)
                started.set()
                if not release.wait(5):
                    raise RuntimeError("test timeout")
                active -= 1
                return completed(request)
            service.analyze.side_effect = analyze
            factory = Mock(return_value=service)
            workers = [JobWorker(store, factory), JobWorker(JobStore(store.path), factory)]
            store.submit(payload("job-1")); store.submit(payload("job-2"))
            for worker in workers:
                worker.start()
            try:
                self.assertTrue(started.wait(5))
                self.assertEqual(factory.call_count, 1)
                release.set()
                wait_until(lambda: store.get("job-2")["callbackStatus"] == "delivered")
                self.assertEqual(maximum, 1)
                self.assertEqual(service.analyze.call_count, 2)
            finally:
                release.set()
                for worker in workers:
                    worker.stop()
            # A new owner resumes a job interrupted before its result was saved.
            store.submit(payload("job-3")); store.mark_running("job-3")
            replacement = JobWorker(store, factory)
            replacement.start()
            try:
                wait_until(lambda: store.get("job-3")["callbackStatus"] == "delivered")
            finally:
                replacement.stop()

    def test_http_submission_runs_real_job_service_and_delivers_mapped_results(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("os.environ", {"AI_JOB_DB": str(Path(directory) / "default.sqlite3")}):
                from main import create_app
            store = JobStore(Path(directory) / "jobs.sqlite3")
            factory = Mock()
            factory.create.return_value.process_video.return_value = dict(clothing={}, objects={
                "bag": [dict(time=5, bbox=[10, 20, 30, 40])],
                "glasses": [dict(time=8, bbox=[40, 50, 60, 70])]})
            prompts = Mock(return_value=SimpleNamespace(generate_prompt=lambda: dict(object_prompt="accessories")))
            service = JobService(factory, SimpleNamespace(device="cpu"), Path(directory) / "outputs", prompts)
            service.downloader = Downloader(Path(directory) / "assets")
            def download(url, target):
                if "video?" in url:
                    Path(target).write_bytes(b"mock video")
                else:
                    Image.new("RGB", (3, 3), "red").save(target, format="PNG")
            service.downloader._download_file = download
            service.callback_service = Mock()
            with TestClient(create_app(lambda: service, store)) as client:
                self.assertEqual(client.post("/v1/analyze/jobs", json=payload()).status_code, 200)
                wait_until(lambda: store.get("job-1")["callbackStatus"] == "delivered")
                state = client.get("/v1/analyze/jobs/job-1").json()
                self.assertEqual(state["status"], "completed")
                objects = state["result"]["episodes"][0]["objects"]
                self.assertEqual([item["seriesProductId"] for item in objects], ["bag", "glasses"])
                self.assertEqual(objects[0]["ranges"], [dict(start=5, end=6, bbox=[10, 20, 30, 40])])
                service.callback_service.send.assert_called_once_with(payload()["callbackUrl"], state["result"])

    def test_worker_initialization_failure_still_delivers_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory) / "jobs.sqlite3")
            store.submit(payload())
            worker = JobWorker(store, Mock(side_effect=RuntimeError("weights missing")))
            worker.callback_service = Mock()
            with self.assertLogs("services.job_worker", level="ERROR"):
                worker.start()
                try:
                    wait_until(lambda: store.get("job-1")["callbackStatus"] == "delivered")
                finally:
                    worker.stop()
            self.assertEqual(store.get("job-1")["status"], "failed")
            worker.callback_service.send.assert_called_once()


if __name__ == "__main__":
    unittest.main()

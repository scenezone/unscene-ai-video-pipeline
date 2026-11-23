"""Independent lease maintenance; inference/downloads check cancellation cooperatively."""
import logging
import threading
import time
from datetime import datetime, timezone

import requests

from schemas.unscene import ClaimedJob
from services.unscene_client import PlatformError
from services.unscene_errors import LeaseLost, SubmissionUncertain, WorkerStopped

logger = logging.getLogger(__name__)
STALE_CODES = {"lease_lost", "wrong_step", "job_changed", "not_found"}


class LeaseGuard:
    def __init__(self, client, job, stop_event, heartbeat_interval=45):
        self.client, self.stop_event = client, stop_event
        self.interval = heartbeat_interval
        self._job = job.model_copy(deep=True)
        self._state_lock = threading.RLock()
        self._write_lock = threading.RLock()
        self._done = threading.Event()
        self._lost = threading.Event()
        self._deadline = self._deadline_for(job.leaseExpiresAt)
        self._thread = None

    @staticmethod
    def _deadline_for(expiry):
        return time.monotonic() + (expiry - datetime.now(timezone.utc)).total_seconds()

    def snapshot(self):
        with self._state_lock:
            return self._job.model_copy(deep=True)

    def check(self):
        if self.stop_event.is_set():
            raise WorkerStopped("Worker is shutting down")
        if self._lost.is_set() or time.monotonic() >= self._deadline:
            self._lost.set()
            raise LeaseLost("Lease no longer valid")

    def renew_links(self):
        with self._write_lock:
            if self._done.is_set():
                return
            self.check()
            job = self.snapshot()
            try:
                body = self.client.job_post(job.jobId, "heartbeat", {})
            except PlatformError as error:
                if error.code in STALE_CODES or error.status in (401, 403):
                    self._lost.set()
                    raise LeaseLost("Platform rejected the lease") from None
                raise
            with self._state_lock:
                # Link renewals may omit fields such as candidate sha256/productCount.
                data = self._job.model_dump(mode="json")
                for name in ("video", "candidates", "cueUpload"):
                    if body.get(name):
                        data[name] = {**(data.get(name) or {}), **body[name]}
                data["leaseExpiresAt"] = body["leaseExpiresAt"]
                updated = ClaimedJob.model_validate(data)
                if (updated.candidates and job.candidates
                        and updated.candidates.sha256 != job.candidates.sha256):
                    self._lost.set()
                    raise LeaseLost("Candidate manifest changed during lease")
                self._job = updated
                self._deadline = self._deadline_for(updated.leaseExpiresAt)
            self.check()

    def _run(self):
        while not self._done.wait(min(self.interval, max(.05, (self._deadline - time.monotonic()) / 2))):
            try:
                self.renew_links()
            except (LeaseLost, WorkerStopped):
                return
            except Exception:
                # Transport failures do not immediately invalidate an unexpired lease.
                # Never log requests exceptions: their text can include signed links.
                logger.warning("Heartbeat failed jobId=%s step=%s", self._job.jobId, self._job.step)
                if time.monotonic() >= self._deadline:
                    self._lost.set()
                    return
                if self._done.wait(min(5, max(.05, self._deadline - time.monotonic()))):
                    return

    def submit(self, action, body):
        with self._write_lock:
            if self._done.is_set():
                raise LeaseLost("Lease was already released")
            self.check()
            try:
                result = self.client.job_post(self._job.jobId, action, body)
            except requests.RequestException:
                self._done.set()
                raise SubmissionUncertain("Submission response was lost; do not replay") from None
            except PlatformError as error:
                if error.code in STALE_CODES:
                    self._lost.set()
                    raise LeaseLost("Job state changed before submission") from None
                if error.status >= 500:
                    self._done.set()
                    raise SubmissionUncertain("Submission outcome is unknown; do not replay") from None
                raise
            self._done.set()
            return result

    def __enter__(self):
        self.check()
        self._thread = threading.Thread(target=self._run, name="unscene-heartbeat", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._done.set()
        if self._thread:
            self._thread.join(timeout=self.client.timeout + 1)

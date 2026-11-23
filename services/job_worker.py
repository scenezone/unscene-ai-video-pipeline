"""One model worker per local queue, with an OS lock released automatically on exit."""
import logging
import os
from threading import Event, Thread

from services.job_service import JobService
from services.callback_service import CallbackService

logger = logging.getLogger(__name__)


class WorkerLock:
    def __init__(self, path):
        self.path = path
        self.handle = None

    def acquire(self):
        handle = open(self.path, "a+b")
        try:
            if os.name == "nt":
                import msvcrt
                if os.fstat(handle.fileno()).st_size == 0:
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return False
        self.handle = handle
        return True

    def release(self):
        if self.handle is not None:
            if os.name == "nt":
                import msvcrt
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


class JobWorker:
    def __init__(self, store, service_factory):
        self.store = store
        self.service_factory = service_factory
        self.stop_event = Event()
        self.thread = None
        self.callback_service = CallbackService()

    def start(self):
        self.thread = Thread(target=self.run, name="ai-job-worker", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)

    def process_next(self, service):
        work = self.store.next_work()
        if work is None:
            return False
        payload, result = work["payload"], work["result"]
        if result is None:
            self.store.mark_running(payload["jobId"])
            try:
                result = service.analyze(payload)
            except Exception:
                logger.exception("Processing job %s failed", payload["jobId"])
                result = JobService.failure_payload(payload)
            # Persist before delivery. A crash after this point retries only the callback.
            self.store.save_result(payload["jobId"], result)
        try:
            sender = service.callback_service if service is not None else self.callback_service
            sender.send(payload["callbackUrl"], result)
        except Exception:
            logger.exception("Callback for job %s failed", payload["jobId"])
            self.store.callback_finished(payload["jobId"], "Callback delivery failed; retry via the callback endpoint.")
        else:
            self.store.callback_finished(payload["jobId"])
        return True

    def run(self):
        ownership = WorkerLock(str(self.store.path) + ".worker.lock")
        while not self.stop_event.is_set():
            if ownership.acquire():
                break
            self.stop_event.wait(1)
        else:
            return
        try:
            self.store.recover()
            # Only the owner loads models. Other API workers can still accept/poll jobs.
            service = None
            while not self.stop_event.is_set():
                try:
                    work = self.store.next_work()
                    if work is None:
                        self.stop_event.wait(1)
                        continue
                    if work["result"] is not None:
                        self.process_next(service)
                        continue
                    if service is None:
                        service = self.service_factory()
                    self.process_next(service)
                except Exception:
                    # A model-initialization failure is terminal for this job, not invisible.
                    logger.exception("Job worker error")
                    if service is None:
                        work = self.store.next_work()
                        if work and work["result"] is None:
                            self.store.save_result(work["payload"]["jobId"], JobService.failure_payload(work["payload"]))
                    self.stop_event.wait(1)
        finally:
            ownership.release()

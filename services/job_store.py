"""Durable local job queue and results. All database transactions are short."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


class JobConflict(ValueError):
    pass


class JobStore:
    def __init__(self, path="data/jobs/jobs.sqlite3"):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY, payload TEXT NOT NULL,
                status TEXT NOT NULL, result TEXT,
                callback_status TEXT NOT NULL DEFAULT 'not_ready',
                callback_error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                owner_client_id TEXT
            )""")
            columns = {row["name"] for row in db.execute("PRAGMA table_info(jobs)")}
            if "owner_client_id" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN owner_client_id TEXT")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def now():
        return datetime.now(timezone.utc).isoformat()

    def submit(self, payload, owner_client_id=None):
        encoded = json.dumps(payload, sort_keys=True)
        now = self.now()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT payload,owner_client_id FROM jobs WHERE job_id=?",
                                  (payload["jobId"],)).fetchone()
            if existing:
                if existing["payload"] != encoded or existing["owner_client_id"] != owner_client_id:
                    raise JobConflict("jobId already exists with a different request")
                return False
            db.execute("""INSERT INTO jobs
                (job_id,payload,status,created_at,updated_at,owner_client_id) VALUES (?,?,?,?,?,?)""",
                       (payload["jobId"], encoded, "queued", now, now, owner_client_id))
        return True

    def get(self, job_id, owner_client_id=None):
        with self.connect() as db:
            if owner_client_id is None:
                row = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            else:
                row = db.execute("SELECT * FROM jobs WHERE job_id=? AND owner_client_id=?",
                                 (job_id, owner_client_id)).fetchone()
        if row is None:
            return None
        return dict(jobId=row["job_id"], status=row["status"],
                    callbackStatus=row["callback_status"], callbackError=row["callback_error"],
                    result=json.loads(row["result"]) if row["result"] else None,
                    createdAt=row["created_at"], updatedAt=row["updated_at"])

    def recover(self):
        # Called only while holding exclusive worker ownership, including across processes.
        with self.connect() as db:
            db.execute("UPDATE jobs SET status='queued', updated_at=? WHERE status='running'", (self.now(),))

    def next_work(self):
        with self.connect() as db:
            row = db.execute("""SELECT * FROM jobs
                WHERE status='queued' OR callback_status='pending'
                ORDER BY CASE WHEN callback_status='pending' THEN 0 ELSE 1 END, created_at
                LIMIT 1""").fetchone()
        if row is None:
            return None
        return dict(payload=json.loads(row["payload"]),
                    result=json.loads(row["result"]) if row["result"] else None)

    def mark_running(self, job_id):
        with self.connect() as db:
            db.execute("UPDATE jobs SET status='running', updated_at=? WHERE job_id=?", (self.now(), job_id))

    def save_result(self, job_id, result):
        with self.connect() as db:
            db.execute("""UPDATE jobs SET status=?, result=?, callback_status='pending',
                callback_error=NULL, updated_at=? WHERE job_id=?""",
                       (result["status"], json.dumps(result), self.now(), job_id))

    def callback_finished(self, job_id, error=None):
        with self.connect() as db:
            db.execute("UPDATE jobs SET callback_status=?, callback_error=?, updated_at=? WHERE job_id=?",
                       ("failed" if error else "delivered", error, self.now(), job_id))

    def retry_callback(self, job_id, owner_client_id=None):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if owner_client_id is None:
                row = db.execute("SELECT result, callback_status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            else:
                row = db.execute("SELECT result, callback_status FROM jobs WHERE job_id=? AND owner_client_id=?",
                                 (job_id, owner_client_id)).fetchone()
            if row is None:
                raise KeyError(job_id)
            if row["result"] is None:
                raise JobConflict("The job has no result yet")
            if row["callback_status"] != "failed":
                raise JobConflict("Only failed callback deliveries can be retried")
            db.execute("UPDATE jobs SET callback_status='pending', callback_error=NULL, updated_at=? WHERE job_id=?",
                       (self.now(), job_id))

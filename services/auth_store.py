"""SQLite-backed third-party client credentials and short-lived bearer tokens."""
import hashlib
import json
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_SCOPES = ("jobs:read", "jobs:write", "callbacks:retry")


def _now():
    return datetime.now(timezone.utc)


def _iso(value):
    return value.isoformat()


def _digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class AuthStore:
    def __init__(self, path="data/auth/auth.sqlite3"):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS api_clients (
                client_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                secret_hash TEXT NOT NULL,
                scopes TEXT NOT NULL,
                created_at TEXT NOT NULL,
                revoked_at TEXT
            )""")

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
    def normalize_scopes(scopes):
        if isinstance(scopes, str):
            scopes = scopes.replace(",", " ").split()
        allowed = set(DEFAULT_SCOPES)
        normalized = tuple(sorted(set(scopes)))
        if not normalized or set(normalized) - allowed:
            raise ValueError("Scopes must be selected from: " + " ".join(DEFAULT_SCOPES))
        return normalized

    def create_client(self, name, scopes=DEFAULT_SCOPES):
        name = name.strip()
        if not name:
            raise ValueError("Client name cannot be empty")
        scopes = self.normalize_scopes(scopes)
        client_id = "cli_" + secrets.token_urlsafe(12)
        client_secret = secrets.token_urlsafe(32)
        with self.connect() as db:
            db.execute("""INSERT INTO api_clients
                (client_id,name,secret_hash,scopes,created_at) VALUES (?,?,?,?,?)""",
                       (client_id, name, _digest(client_secret), json.dumps(scopes), _iso(_now())))
        return {"client_id": client_id, "client_secret": client_secret,
                "name": name, "scopes": scopes}

    def ensure_bootstrap_client(self):
        """Create and return one initial client if there are no active clients."""
        name = "Initial local API client"
        scopes = self.normalize_scopes(DEFAULT_SCOPES)
        client_id = "cli_" + secrets.token_urlsafe(12)
        client_secret = secrets.token_urlsafe(32)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            active = db.execute("SELECT 1 FROM api_clients WHERE revoked_at IS NULL LIMIT 1").fetchone()
            if active:
                return None
            db.execute("""INSERT INTO api_clients
                (client_id,name,secret_hash,scopes,created_at) VALUES (?,?,?,?,?)""",
                       (client_id, name, _digest(client_secret), json.dumps(scopes), _iso(_now())))
        return {"client_id": client_id, "client_secret": client_secret, "scopes": scopes}

    def rotate_client(self, client_id):
        secret = secrets.token_urlsafe(32)
        with self.connect() as db:
            cur = db.execute("UPDATE api_clients SET secret_hash=? WHERE client_id=? AND revoked_at IS NULL",
                             (_digest(secret), client_id))
            if not cur.rowcount:
                raise KeyError(client_id)
        return secret

    def revoke_client(self, client_id):
        now = _iso(_now())
        with self.connect() as db:
            cur = db.execute("UPDATE api_clients SET revoked_at=? WHERE client_id=? AND revoked_at IS NULL",
                             (now, client_id))
            if not cur.rowcount:
                raise KeyError(client_id)

    def authenticate_client(self, client_id, client_secret):
        if not client_id or not client_secret:
            return None
        with self.connect() as db:
            client = db.execute("SELECT * FROM api_clients WHERE client_id=?", (client_id,)).fetchone()
            if client is None or client["revoked_at"] is not None:
                return None
            if not secrets.compare_digest(client["secret_hash"], _digest(client_secret)):
                return None
        return {"client_id": client["client_id"], "scopes": set(json.loads(client["scopes"]))}

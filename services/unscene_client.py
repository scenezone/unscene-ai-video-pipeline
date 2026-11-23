"""Outbound OAuth/HMAC client. Never replay ambiguous state-changing requests."""
import hashlib
import hmac
import json
import math
import threading
import time
from urllib.parse import quote, urlsplit

import requests


def json_bytes(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


class PlatformError(RuntimeError):
    def __init__(self, status, code):
        # Do not include response bodies, secrets or signed links in exceptions/logs.
        self.status = status
        self.code = code
        super().__init__(f"Unscene API HTTP {status}: {code}")


class UnsceneClient:
    def __init__(self, base_url, client_id, client_secret, timeout=30, session_factory=requests.Session):
        parsed = urlsplit(base_url)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.query
                or parsed.fragment or parsed.username or parsed.password
                or parsed.path.rstrip("/") != "/ai/v1"):
            raise ValueError("Base URL must be http(s)://host/ai/v1 without credentials/query")
        if parsed.scheme != "https" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("Unscene requires HTTPS outside local development")
        if not client_id or not client_secret:
            raise ValueError("UNSCENE_CLIENT_ID and UNSCENE_CLIENT_SECRET are required")
        self.base_url = base_url.rstrip("/")
        self.client_id, self._secret = client_id, client_secret
        self.timeout = timeout
        self._session_factory = session_factory
        self._local = threading.local()
        self._token_lock = threading.RLock()
        self._token = None
        self._expires = 0

    @property
    def session(self):
        if not hasattr(self._local, "session"):
            self._local.session = self._session_factory()
        return self._local.session

    @staticmethod
    def _read(response):
        try:
            body = response.json()
        except ValueError:
            raise PlatformError(response.status_code, "invalid_response") from None
        if not 200 <= response.status_code < 300:
            error = body.get("error", "unknown_error") if isinstance(body, dict) else "unknown_error"
            code = error.get("code", "unknown_error") if isinstance(error, dict) else error
            # Server strings are untrusted; expose only bounded error codes.
            code = str(code)
            if not code.replace("_", "").isalnum() or len(code) > 80:
                code = "unknown_error"
            raise PlatformError(response.status_code, code)
        if not isinstance(body, dict):
            raise PlatformError(response.status_code, "invalid_response")
        return body

    def token(self, invalid_token=None):
        with self._token_lock:
            if invalid_token is not None and self._token == invalid_token:
                self._expires = 0
            if self._token and time.monotonic() < self._expires:
                return self._token
            started = time.monotonic()
            with self.session.post(self.base_url + "/oauth/token", data={
                "grant_type": "client_credentials", "client_id": self.client_id,
                "client_secret": self._secret}, timeout=self.timeout, allow_redirects=False) as response:
                body = self._read(response)
            token = body.get("access_token")
            try:
                lifetime = float(body["expires_in"])
            except (KeyError, TypeError, ValueError):
                raise PlatformError(200, "invalid_token_response") from None
            if not math.isfinite(lifetime) or lifetime <= 0 or not isinstance(token, str) or not token:
                raise PlatformError(200, "invalid_token_response")
            self._token = token
            self._expires = started + max(0, lifetime - min(30, lifetime / 10))
            return self._token

    def post(self, endpoint, body):
        raw = json_bytes(body)
        # Sign the actual encoded/normalized request path, including its query.
        path = requests.Request("POST", self.base_url + endpoint).prepare().path_url
        token = self.token()
        for attempt in range(2):
            timestamp = str(int(time.time()))
            message = f"{timestamp}.POST.{path}.{hashlib.sha256(raw).hexdigest()}"
            signature = hmac.new(self._secret.encode(), message.encode(), hashlib.sha256).hexdigest()
            try:
                with self.session.post(self.base_url + endpoint, data=raw, headers={
                    "Authorization": f"Bearer {token}", "Content-Type": "application/json",
                    "X-Unscene-Key-Id": self.client_id, "X-Unscene-Timestamp": timestamp,
                    "X-Unscene-Signature": "v1=" + signature,
                }, timeout=self.timeout, allow_redirects=False) as response:
                    return self._read(response)
            except PlatformError as error:
                # invalid_token is an explicit rejection; refreshing/re-signing is safe.
                if error.status != 401 or error.code != "invalid_token" or attempt:
                    raise
                token = self.token(invalid_token=token)

    def claim(self, worker_id, steps):
        body = self.post("/jobs/claim", {"workerId": worker_id, "steps": list(steps)})
        if "job" not in body:
            raise PlatformError(200, "invalid_claim_response")
        return body["job"]

    def job_post(self, job_id, action, body):
        if action not in ("heartbeat", "requirements", "cues", "fail"):
            raise ValueError("Unsupported job action")
        return self.post(f"/jobs/{quote(job_id, safe='')}/{action}", body)

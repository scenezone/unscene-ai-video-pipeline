"""Serve only the two episode input sets, and save local main-API callbacks."""
import argparse
import hashlib
import json
import re
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from prepare_local_payloads import DATA, ROOT, allowed_assets


class LocalAssetHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DATA), **kwargs)

    def send_head(self):
        key = unquote(urlsplit(self.path).path).lstrip("/")
        target = self.server.assets.get(key)
        if (target is None or not target.is_file()
                or not target.resolve().is_relative_to(DATA.resolve())):
            self.send_error(404, "Not an episode input asset")
            return None
        # The standard server supplies content types, length and streamed file bytes.
        return super().send_head()

    def do_POST(self):
        if urlsplit(self.path).path != "/callback":
            self.send_error(404, "Unknown callback path")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(400, "Invalid Content-Length")
            return
        if self.headers.get("Transfer-Encoding") or not 0 < length <= 10 * 1024 * 1024:
            self.send_error(413, "Callback must have a body of at most 10 MiB")
            return
        self.connection.settimeout(30)
        try:
            raw = self.rfile.read(length)
            payload = json.loads(raw)
            if not isinstance(payload, dict) or not isinstance(payload.get("jobId"), str):
                raise ValueError("Expected callback object and jobId")
            ident = payload["jobId"]
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", ident):
                ident = hashlib.sha256(ident.encode()).hexdigest()
            self.server.callback_dir.mkdir(parents=True, exist_ok=True)
            destination = self.server.callback_dir / f"{ident}.json"
            partial = destination.with_suffix(f".{uuid.uuid4().hex}.part")
            partial.write_bytes(raw)
            partial.replace(destination)
        except (ValueError, UnicodeDecodeError):
            self.send_error(400, "Invalid callback JSON")
            return
        self.send_response(204)
        self.end_headers()
        print(f"Saved callback: {destination}", flush=True)

    def log_message(self, format, *args):
        # Avoid printing request headers/credentials; this helper receives local test assets/results only.
        print(f"[local-assets] {format % args}", flush=True)


def create_server(bind="127.0.0.1", port=8765, callback_dir=None):
    server = ThreadingHTTPServer((bind, port), LocalAssetHandler)
    server.assets = allowed_assets()
    server.callback_dir = Path(callback_dir) if callback_dir else ROOT / "outputs/local_callbacks"
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    with create_server(args.bind, args.port) as server:
        print(f"Local input server: http://{args.bind}:{args.port}")
        print(f"Serving {len(server.assets)} selected files; callbacks saved under {server.callback_dir}")
        print("Keep this terminal open while the main API processes both episodes. Ctrl+C stops it.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()

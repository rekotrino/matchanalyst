"""Local dashboard server (standard library only).

  python -m nba_analyst serve             # http://127.0.0.1:8765
  python -m nba_analyst export-html out.html   # standalone page with data embedded

Routes:  /                 the dashboard page
         /api/dashboard    JSON payload (cached 60s; ?recompute=1 forces a rebuild)
"""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from .config import SETTINGS
from .payload import build_payload
from .storage import Store

UI_PATH = os.path.join(os.path.dirname(__file__), "ui", "dashboard.html")
SKELETON = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '{head}</head>\n<body>\n{body}\n</body>\n</html>\n')


def page_html(embed=None, ui_path=None):
    """Full HTML document. embed: payload dict to inline (standalone copy)."""
    with open(ui_path or UI_PATH, encoding="utf-8") as f:
        frag = f.read()
    head = ""
    if embed is not None:
        data = json.dumps(embed).replace("</", "<\\/")
        head = f"<script>window.__DATA__ = {data};</script>\n"
        label = embed.get("sport_label") if not embed.get("sports") else None
        if label:
            frag = frag.replace("<title>Sports Analyst Board</title>", f"<title>{label} Analyst Board</title>", 1)
    return SKELETON.format(head=head, body=frag)


def export_html(store, path, days_ahead=None, builder=build_payload):
    with open(path, "w", encoding="utf-8") as f:
        f.write(page_html(builder(store, days_ahead)))
    return path


class _Cache:
    """sources: {sport_key: (db_path, builder, store_cls)}. One sport -> plain payload;
    several -> {"sports": {key: payload}} so the page shows a sport switcher."""
    def __init__(self, sources):
        self.sources, self.data, self.ts, self.lock = sources, None, 0.0, threading.Lock()

    def get(self, force=False):
        with self.lock:
            if force or self.data is None or time.time() - self.ts > 60:
                built = {}
                for key, (db, builder, store_cls) in self.sources.items():
                    if os.path.exists(db):
                        built[key] = builder(store_cls(db))
                self.data = next(iter(built.values())) if len(built) == 1 else {"sports": built}
                self.ts = time.time()
            return self.data


def combined_payload(sources, days_ahead=None):
    out = {}
    for key, (db, builder, store_cls) in sources.items():
        if os.path.exists(db):
            out[key] = builder(store_cls(db), days_ahead)
    return {"sports": out}


def make_handler(cache):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ("/", "/index.html"):
                body = page_html().encode("utf-8")
                self._send(200, "text/html; charset=utf-8", body)
            elif u.path == "/api/dashboard":
                force = parse_qs(u.query).get("recompute", ["0"])[0] == "1"
                try:
                    body = json.dumps(cache.get(force)).encode("utf-8")
                    self._send(200, "application/json", body)
                except Exception as e:  # noqa: BLE001
                    self._send(500, "application/json", json.dumps({"error": str(e)}).encode())
            else:
                self._send(404, "text/plain", b"not found")

        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):  # quieter console
            if "api" in fmt % args or " 200 " not in fmt % args:
                return
    return Handler


def serve(db_path=None, host="127.0.0.1", port=8765, builder=build_payload, store_cls=Store, label="NBA", extra_sources=None):
    db_path = db_path or SETTINGS.db_path
    sources = {label.lower(): (db_path, builder, store_cls)}
    sources.update(extra_sources or {})
    if not any(os.path.exists(d) for d, _, _ in sources.values()):
        raise SystemExit(f"No database at {db_path}. Run the pipeline's `run` command first.")
    cache = _Cache(sources)
    httpd = ThreadingHTTPServer((host, port), make_handler(cache))
    print(f"{' + '.join(k.upper() for k in sources)} Analyst dashboard: http://{host}:{port}/  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass

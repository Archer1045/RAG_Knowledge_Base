"""本地 RAG 检索 HTTP 服务。"""
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from retrieve import load_index, retrieve

INDEX = load_index()


class Handler(BaseHTTPRequestHandler):
    server_version = "RiderRAG/1.0"

    def _send(self, status: int, payload) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        if self.path in ("/", "/health"):
            self._send(
                200,
                {
                    "ok": True,
                    "name": "骑手招聘助手 RAG 知识库",
                    "stats": INDEX.get("counts"),
                    "endpoints": {"retrieve": "POST /retrieve"},
                },
            )
            return
        self._send(404, {"error": "Not Found"})

    def do_POST(self):  # noqa: N802
        if self.path != "/retrieve":
            self._send(404, {"error": "Not Found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
            results = retrieve(
                payload.get("query", ""),
                top_k=int(payload.get("top_k", payload.get("topK", 5))),
                district=payload.get("district"),
                lat=payload.get("lat"),
                lon=payload.get("lon"),
                type_filter=payload.get("type"),
                mode=payload.get("mode", "hybrid"),
            )
            self._send(
                200,
                {
                    "query": payload.get("query", ""),
                    "count": len(results),
                    "results": results,
                    "stats": INDEX.get("counts"),
                },
            )
        except Exception as exc:  # noqa: BLE001
            self._send(400, {"error": str(exc)})

    def log_message(self, format, *args):  # noqa: A002
        print(f"[{self.log_date_time_string()}] {format % args}")


def main() -> None:
    port = int(os.environ.get("PORT", "8787"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"RAG retrieval server listening on http://localhost:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()

"""Local HTTP server for the web interface, built on the standard library only.

Routes (all JSON unless noted):
    GET  /                    the page
    GET  /api/state           what the page needs to draw itself
    GET  /api/history         the messages of the current conversation
    GET  /api/sessions        saved conversations
    GET  /api/quiz/download   the last quiz as a file (?answers=0|1&format=txt|json)
    GET  /api/artifact/download  the living document as Markdown
    POST /api/chat            a message; answered as a stream of JSON lines
    POST /api/confirm         the answer to a confirmation the stream asked for
    POST /api/command         a slash command, such as /rag on
    POST /api/quiz            the quiz form
    POST /api/upload          a document (raw body, name in ?name=)
    POST /api/profile         the profile form
    POST /api/new             start a new conversation

The server listens on 127.0.0.1 only. Two checks protect it from other web pages open in
the same browser: the Host header must be this machine, and a POST must be JSON or raw
bytes, which a foreign page cannot send without a CORS preflight that this server refuses.
"""

from __future__ import annotations

import json
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from web.app import WebApp

STATIC_DIR = Path(__file__).resolve().parent / "static"
UPLOAD_DIR = Path("data") / "uploads"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_JSON_BYTES = 1024 * 1024
HOST = "127.0.0.1"


class RequestRefused(Exception):
    """The request is invalid. The message and status are sent back to the page."""

    def __init__(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.status = status


def make_handler(app: WebApp, port: int) -> type[BaseHTTPRequestHandler]:
    """Build the request handler class bound to one WebApp."""

    allowed_hosts = {f"localhost:{port}", f"127.0.0.1:{port}"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # keep the terminal quiet
            pass

        # ------------------------------------------------------------ plumbing

        def _send(self, status: HTTPStatus, body: bytes, content_type: str, headers: dict[str, str] | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

        def _read_body(self, limit: int) -> bytes:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                raise RequestRefused("Invalid Content-Length.") from None
            if length > limit:
                raise RequestRefused("The request is too large.", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return self.rfile.read(length)

        def _read_json(self) -> dict[str, Any]:
            if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                raise RequestRefused("Content-Type must be application/json.", HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
            try:
                data = json.loads(self._read_body(MAX_JSON_BYTES) or b"{}")
            except ValueError:
                raise RequestRefused("The request is not valid JSON.") from None
            if not isinstance(data, dict):
                raise RequestRefused("The request must be a JSON object.")
            return data

        def _check_host(self) -> None:
            if self.headers.get("Host", "") not in allowed_hosts:
                raise RequestRefused("Unknown host.", HTTPStatus.FORBIDDEN)

        def _dispatch(self, routes: dict[str, Any]) -> None:
            try:
                self._check_host()
                parsed = urlparse(self.path)
                handler = routes.get(parsed.path)
                if handler is None:
                    raise RequestRefused("Not found.", HTTPStatus.NOT_FOUND)
                handler(parse_qs(parsed.query))
            except RequestRefused as error:
                self._send_json({"error": str(error)}, error.status)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the tab was closed mid-answer

        # --------------------------------------------------------------- routes

        def do_GET(self) -> None:
            self._dispatch(
                {
                    "/": self._page,
                    "/api/state": lambda _q: self._send_json({**app.state(), "model_problem": app.check_model()}),
                    "/api/history": lambda _q: self._send_json({"messages": app.history()}),
                    "/api/sessions": lambda _q: self._send_json({"sessions": app.saved_sessions()}),
                    "/api/quiz/download": self._quiz_download,
                    "/api/artifact/download": self._artifact_download,
                }
            )

        def do_POST(self) -> None:
            self._dispatch(
                {
                    "/api/chat": self._chat,
                    "/api/confirm": self._confirm,
                    "/api/command": self._command,
                    "/api/quiz": self._quiz,
                    "/api/upload": self._upload,
                    "/api/profile": self._profile,
                    "/api/new": self._new,
                }
            )

        def _page(self, _query: dict[str, list[str]]) -> None:
            self._send(HTTPStatus.OK, (STATIC_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")

        def _chat(self, _query: dict[str, list[str]]) -> None:
            message = str(self._read_json().get("message", "")).strip()
            if not message:
                raise RequestRefused("The message is empty.")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            for event in app.stream_chat(message):
                self.wfile.write(json.dumps(event).encode("utf-8") + b"\n")
                self.wfile.flush()

        def _confirm(self, _query: dict[str, list[str]]) -> None:
            body = self._read_json()
            found = app.resolve_confirmation(str(body.get("id", "")), body.get("approved") is True)
            self._send_json({"ok": found})

        def _command(self, _query: dict[str, list[str]]) -> None:
            body = self._read_json()
            command = str(body.get("command", "")).strip()
            if not command.startswith("/"):
                raise RequestRefused("A command starts with '/'.")
            confirmed = body.get("confirmed") is True
            self._send_json({"output": app.run_command(command, confirmed=confirmed), "state": app.state()})

        def _quiz(self, _query: dict[str, list[str]]) -> None:
            body = self._read_json()
            try:
                questions = int(body.get("questions", 5))
            except (TypeError, ValueError):
                raise RequestRefused("The number of questions must be a number.") from None
            output = app.quiz_command(str(body.get("topic", "")), questions, str(body.get("language", "")))
            self._send_json({"output": output, "state": app.state()})

        def _quiz_download(self, query: dict[str, list[str]]) -> None:
            as_json = query.get("format", ["txt"])[0] == "json"
            answers = query.get("answers", ["1"])[0] != "0"
            rendered = app.render_last_quiz(answers, as_json)
            if rendered is None:
                raise RequestRefused("No quiz has been generated yet.", HTTPStatus.NOT_FOUND)
            name, content = rendered
            self._send(
                HTTPStatus.OK,
                content.encode("utf-8"),
                "application/json" if as_json else "text/plain; charset=utf-8",
                {"Content-Disposition": f'attachment; filename="{name}"'},
            )

        def _artifact_download(self, _query: dict[str, list[str]]) -> None:
            rendered = app.artifact_markdown()
            if rendered is None:
                raise RequestRefused("There is no document yet.", HTTPStatus.NOT_FOUND)
            name, content = rendered
            self._send(
                HTTPStatus.OK,
                content.encode("utf-8"),
                "text/markdown; charset=utf-8",
                {"Content-Disposition": f'attachment; filename="{name}"'},
            )

        def _upload(self, query: dict[str, list[str]]) -> None:
            if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/octet-stream":
                raise RequestRefused("Content-Type must be application/octet-stream.", HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
            name = query.get("name", [""])[0]
            if not name:
                raise RequestRefused("The file name is missing.")
            data = self._read_body(MAX_UPLOAD_BYTES)
            self._send_json({"output": app.ingest_file(name, data, UPLOAD_DIR), "state": app.state()})

        def _profile(self, _query: dict[str, list[str]]) -> None:
            app.update_profile(self._read_json())
            self._send_json({"state": app.state()})

        def _new(self, _query: dict[str, list[str]]) -> None:
            self._read_json()
            app.new_conversation()
            self._send_json({"state": app.state()})

    return Handler


def serve(app: WebApp, port: int, open_browser: bool = True) -> None:
    """Serve the interface until Ctrl+C."""
    server = ThreadingHTTPServer((HOST, port), make_handler(app, port))
    url = f"http://localhost:{port}"
    print(f"SensAI is running at {url}  (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if app.config.features.memory_enabled and app.conversation_session.messages:
            app.conversation_session.save()
        print("SensAI stopped.")

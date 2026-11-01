import http.client
import json
import os
import socket
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

from cli.cli import CliDependencies
from cli.model_factory import ModelRuntime
from config import AppConfig
from mcp.manager import MCPManager
from rag.context import PreparedContext
from tools.base import Tool
from tools.registry import ToolRegistry
from utilities.ollama_client import OllamaConnectionError
from web.app import WebApp
from web.server import make_handler


def ndjson(*messages, **extra):
    """Streamed Ollama lines: one chunk per message, the last one carries the counters."""
    lines = [json.dumps({"message": message, "done": False}).encode() for message in messages]
    lines.append(json.dumps({"message": {"content": ""}, "done": True, **extra}).encode())
    return lines


def text(content):
    return {"role": "assistant", "content": content}


class SendTool(Tool):
    """A tool that asks before it runs, like web_search."""

    name = "send"
    description = "Send something outside"
    parameters = {"type": "object", "properties": {}}

    def __init__(self):
        self.ran = False

    def confirmation_prompt(self, arguments):
        return "Send this?"

    def execute(self, arguments):
        self.ran = True
        return "sent"


class WebAppCase(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        previous = os.getcwd()
        os.chdir(self.root)  # sessions/ and profile.json are relative to the current directory
        self.addCleanup(os.chdir, previous)

        self.client = MagicMock()
        self.client.stream_message.side_effect = lambda *a, **k: iter(ndjson(text("Hello teacher")))
        self.provider = MagicMock()
        self.registry = ToolRegistry()
        self.send_tool = SendTool()
        self.registry.register(self.send_tool)
        runtime = ModelRuntime(self.client, MagicMock(), self.provider)
        runtime.rag_indexer.vector_store.list_sources.return_value = [("course.md", 3)]
        patcher = patch("web.app.create_model_runtime", return_value=runtime)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.app = WebApp(
            AppConfig(),
            profile_path=str(self.root / "profile.json"),
            mcp_manager=MCPManager([]),
            dependencies=CliDependencies(registry=self.registry),
        )
        self.app._ollama = MagicMock()
        self.app._ollama.list_models.return_value = ["llama3.2:latest"]

    def chat(self, message, on_confirm=None):
        events = []
        for event in self.app.stream_chat(message):
            events.append(event)
            if event["type"] == "confirm" and on_confirm is not None:
                self.app.resolve_confirmation(event["id"], on_confirm)
        return events

    def types(self, events):
        return [event["type"] for event in events]


class ChatTests(WebAppCase):
    def test_answer_is_streamed_and_kept_in_history(self):
        events = self.chat("Hi")

        self.assertEqual("".join(e["text"] for e in events if e["type"] == "token"), "Hello teacher")
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(
            self.app.history(),
            [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello teacher"}],
        )

    def test_conversation_is_saved_after_each_turn(self):
        self.chat("Hi")

        self.assertEqual([s["preview"] for s in self.app.saved_sessions()], ["Hi"])

    def test_failed_turn_reports_an_error_and_removes_the_pending_message(self):
        self.client.stream_message.side_effect = OllamaConnectionError("down")

        events = self.chat("Hi")

        self.assertIn("error", self.types(events))
        self.assertIn("Connection error", next(e["text"] for e in events if e["type"] == "error"))
        self.assertEqual(self.app.history(), [])

    def test_slash_command_runs_through_the_cli_handlers(self):
        events = self.chat("/rag on")

        self.assertIn("RAG mode enabled", events[0]["text"])
        self.assertTrue(events[-1]["state"]["rag_enabled"])
        self.client.stream_message.assert_not_called()

    def test_unknown_command_is_explained_not_sent_to_the_model(self):
        events = self.chat("/nope")

        self.assertIn("Unknown command", events[0]["text"])
        self.client.stream_message.assert_not_called()

    def test_rag_context_is_used_and_sources_are_reported(self):
        self.app.session.rag_enabled = True
        result = MagicMock(source_filename="course.md", chunk_id="c1", chunk_index=0)
        self.provider.prepare.return_value = PreparedContext("Context", [result])

        with patch("web.app.format_source_attribution", return_value="Sources: course.md"):
            events = self.chat("What is a fraction?")

        self.provider.prepare.assert_called_once_with("What is a fraction?")
        self.assertIn("Sources: course.md", [e["text"] for e in events if e["type"] == "sources"])


class ToolConfirmationTests(WebAppCase):
    def script_tool_call(self):
        call = {"function": {"name": "send", "arguments": {}}}
        streams = [ndjson({"role": "assistant", "content": "", "tool_calls": [call]}), ndjson(text("Done"))]
        self.client.stream_message.side_effect = lambda *a, **k: iter(streams.pop(0))

    def test_approved_tool_runs(self):
        self.script_tool_call()

        events = self.chat("send it", on_confirm=True)

        self.assertIn("confirm", self.types(events))
        self.assertTrue(self.send_tool.ran)

    def test_refused_tool_does_not_run(self):
        self.script_tool_call()

        events = self.chat("send it", on_confirm=False)

        self.assertFalse(self.send_tool.ran)
        self.assertTrue(any("declined" in e["text"] for e in events if e["type"] == "tool"))

    def test_unknown_confirmation_is_ignored(self):
        self.assertFalse(self.app.resolve_confirmation("missing", True))


class ManagementTests(WebAppCase):
    def test_profile_update_changes_the_system_prompt(self):
        self.app.update_profile({"user_name": "Mme Martin", "constraints": ["Short answers", " "]})

        self.assertIn("Mme Martin", self.app.messages[0]["content"])
        self.assertIn("- Short answers", self.app.messages[0]["content"])
        self.assertEqual(self.app.state()["profile"]["constraints"], ["Short answers"])

    def test_new_conversation_saves_the_old_one_and_starts_empty(self):
        self.chat("Hi")
        old_id = self.app.conversation_session.session_id

        self.app.new_conversation()

        self.assertEqual(self.app.history(), [])
        self.assertNotEqual(self.app.conversation_session.session_id, old_id)
        self.assertEqual([s["id"] for s in self.app.saved_sessions()], [old_id])

    def test_loading_and_forgetting_use_the_cli_commands(self):
        self.chat("Hi")
        old_id = self.app.conversation_session.session_id
        self.app.new_conversation()

        self.app.run_command(f"/load {old_id}")
        self.assertEqual(len(self.app.history()), 2)

        self.app.run_command("/forget")
        self.assertEqual(self.app.history(), [])
        self.assertEqual(self.app.saved_sessions(), [])

    def test_forget_all_needs_the_browser_confirmation(self):
        self.chat("Hi")
        self.assertIn("Nothing deleted", self.app.run_command("/forget all"))
        self.assertEqual(len(self.app.saved_sessions()), 1)

        self.assertIn("Deleted 1", self.app.forget_all())
        self.assertEqual(self.app.saved_sessions(), [])

    def test_state_lists_documents_tools_and_tokens(self):
        state = self.app.state()

        self.assertEqual(state["documents"], [{"name": "course.md", "chunks": 3}])
        self.assertEqual([t["name"] for t in state["tools"]], ["send"])
        self.assertEqual(state["tokens"]["window"], AppConfig().model.context_window)

    def test_upload_is_indexed_then_deleted(self):
        seen = {}

        def index(path):
            seen["content"] = Path(path).read_bytes()
            seen["name"] = Path(path).name
            return MagicMock(document=MagicMock(source_filename="a.md"), chunk_count=1, inserted_chunk_count=1)

        self.app.runtime.rag_indexer.index_document.side_effect = index
        self.app.context.rag_indexer.index_document.side_effect = index
        upload_dir = self.root / "uploads"

        output = self.app.ingest_file("../a.md", b"# Title", upload_dir)

        self.assertIn("Document indexed", output)
        self.assertEqual((seen["name"], seen["content"]), ("a.md", b"# Title"))
        self.assertEqual(list(upload_dir.iterdir()), [])

    def test_upload_with_an_invalid_name_is_refused(self):
        self.assertIn("not valid", self.app.ingest_file("...", b"x", self.root / "uploads"))

    def test_quiz_form_validates_input(self):
        self.assertIn("topic is required", self.app.quiz_command("  ", 5, ""))
        self.assertIn("between 1 and", self.app.quiz_command("Volcanoes", 99, ""))
        self.assertIsNone(self.app.render_last_quiz(True, False))

    def test_last_quiz_can_be_rendered_without_answers(self):
        self.app.context.quiz_store.quiz = {
            "title": "Q", "questions": [{"question": "1+1?", "choices": ["1", "2"], "answer_index": 1}],
        }

        name, content = self.app.render_last_quiz(False, False)

        self.assertEqual(name, "quiz.txt")
        self.assertNotIn("Answer key", content)
        self.assertIn("Answer key", self.app.render_last_quiz(True, False)[1])
        self.assertIn('"answer_index"', self.app.render_last_quiz(True, True)[1])


class ArtifactAndModelTests(WebAppCase):
    def test_no_document_by_default(self):
        self.assertIsNone(self.app.state()["artifact"])
        self.assertIsNone(self.app.artifact_markdown())

    def test_living_document_is_exposed_and_saved_with_the_conversation(self):
        self.app.artifact_manager.init_document("Fractions", "")
        self.app.artifact_manager.update_section("Goals", "Understand halves")

        state = self.app.state()["artifact"]
        name, content = self.app.artifact_markdown()

        self.assertEqual((state["title"], state["sections"]), ("Fractions", ["Goals"]))
        self.assertEqual(name, "document.md")
        self.assertIn("Understand halves", content)
        self.chat("Hi")
        session_id = self.app.conversation_session.session_id
        self.assertTrue((self.root / "sessions" / f"{session_id}_artifact.md").exists())

    def test_new_conversation_discards_the_document(self):
        self.app.artifact_manager.init_document("Fractions", "")

        self.app.new_conversation()

        self.assertIsNone(self.app.state()["artifact"])

    def test_exit_command_does_not_stop_anything(self):
        events = self.chat("/exit")

        self.assertIn("stays open", events[0]["text"])
        self.assertFalse(self.app.session.exit_requested)

    def test_model_fallback_is_announced_in_the_chat(self):
        def stream(*args, **kwargs):
            self.app._fallback_notices.append("Model 'a' is unavailable. Retrying with fallback 'b'.")
            return iter(ndjson(text("ok")))

        self.client.stream_message.side_effect = stream

        events = self.chat("Hi")

        self.assertIn("fallback 'b'", next(e["text"] for e in events if e["type"] == "notice"))

    def test_model_check_accepts_the_fallback_model(self):
        self.assertIsNone(self.app.check_model())

    def test_model_check_reports_a_missing_model_and_a_stopped_ollama(self):
        self.app._ollama.list_models.return_value = ["other:1b"]
        self.assertIn("ollama pull", self.app.check_model())

        self.app._ollama.list_models.side_effect = OllamaConnectionError("down")
        self.assertIn("Connection error", self.app.check_model())


class ServerTests(WebAppCase):
    def setUp(self):
        super().setUp()
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), make_handler(self.app, self.port))
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def request(self, method, path, body=None, headers=None, host=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        merged = {"Host": host or f"localhost:{self.port}", **(headers or {})}
        connection.request(method, path, body=body, headers=merged)
        response = connection.getresponse()
        data = response.read()
        connection.close()
        return response.status, data

    def post_json(self, path, payload, **kwargs):
        headers = {"Content-Type": "application/json"}
        return self.request("POST", path, json.dumps(payload).encode(), headers, **kwargs)

    def test_page_and_state_are_served(self):
        status, page = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"SensAI", page)

        status, data = self.request("GET", "/api/state")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)["persona"], "teacher-assistant")

    def test_foreign_host_is_refused(self):
        self.assertEqual(self.request("GET", "/api/state", host="evil.example")[0], 403)
        self.assertEqual(self.post_json("/api/command", {"command": "/tokens"}, host="evil.example")[0], 403)

    def test_post_must_be_json_or_raw_bytes(self):
        status, _ = self.request("POST", "/api/command", b'{"command": "/tokens"}', {"Content-Type": "text/plain"})
        self.assertEqual(status, 415)
        status, _ = self.request("POST", "/api/upload?name=a.md", b"x", {"Content-Type": "text/plain"})
        self.assertEqual(status, 415)

    def test_chat_streams_json_lines(self):
        status, data = self.post_json("/api/chat", {"message": "Hi"})

        self.assertEqual(status, 200)
        events = [json.loads(line) for line in data.splitlines()]
        self.assertEqual(events[0], {"type": "token", "text": "Hello teacher"})
        self.assertEqual(events[-1]["type"], "done")

    def test_empty_message_and_non_command_are_refused(self):
        self.assertEqual(self.post_json("/api/chat", {"message": "  "})[0], 400)
        self.assertEqual(self.post_json("/api/command", {"command": "hello"})[0], 400)

    def test_command_and_history_endpoints(self):
        status, data = self.post_json("/api/command", {"command": "/sources off"})
        self.assertEqual(status, 200)
        self.assertFalse(json.loads(data)["state"]["show_sources"])
        self.assertEqual(json.loads(self.request("GET", "/api/history")[1]), {"messages": []})

    def test_artifact_download(self):
        self.assertEqual(self.request("GET", "/api/artifact/download")[0], 404)
        self.app.artifact_manager.init_document("Fractions", "")

        status, data = self.request("GET", "/api/artifact/download")

        self.assertEqual(status, 200)
        self.assertIn(b"# Fractions", data)

    def test_quiz_download_without_quiz_is_404(self):
        self.assertEqual(self.request("GET", "/api/quiz/download")[0], 404)

    def test_unknown_route_is_404(self):
        self.assertEqual(self.request("GET", "/nope")[0], 404)


if __name__ == "__main__":
    unittest.main()

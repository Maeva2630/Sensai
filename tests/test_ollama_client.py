import json
import unittest
from unittest.mock import patch, MagicMock
import requests
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

from utilities.ollama_client import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    OllamaError,
    OllamaHTTPClient,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaResponseError,
    OllamaTimeoutError,
    build_tool_message,
    parse_tool_calls,
)


class TestOllamaHTTPClient(unittest.TestCase):
    """Unit tests with mocks for OllamaHTTPClient."""

    def setUp(self):
        self.client = OllamaHTTPClient(base_url="http://localhost:11434", model="llama3.2")

    @patch("requests.get")
    def test_list_models_success(self, mock_get):
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.2:latest"},
                {"name": "qwen2.5:0.5b"},
            ]
        }
        mock_get.return_value = mock_response

        models = self.client.list_models()

        mock_get.assert_called_once_with("http://localhost:11434/api/tags", timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS)
        self.assertEqual(models, ["llama3.2:latest", "qwen2.5:0.5b"])

    @patch("requests.get")
    def test_list_models_connection_error(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        with self.assertRaises(OllamaConnectionError) as ctx:
            self.client.list_models()

        self.assertIn("Failed to connect to Ollama", str(ctx.exception))

    @patch("requests.get")
    def test_list_models_http_error(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": "Internal Server Error"}
        mock_get.return_value = mock_response

        with self.assertRaises(OllamaResponseError) as ctx:
            self.client.list_models()
        self.assertEqual(ctx.exception.status_code, 500)

    @patch("requests.post")
    def test_get_embedding_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_post.return_value = mock_response

        res = self.client.get_embedding("Hello world")

        mock_post.assert_called_once_with(
            "http://localhost:11434/api/embed",
            json={"model": "llama3.2", "input": "Hello world"},
            timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        )
        self.assertEqual(res, {"embedding": [0.1, 0.2, 0.3]})

    @patch("requests.post")
    def test_get_embedding_connection_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.RequestException("Network error")

        with self.assertRaises(OllamaConnectionError):
            self.client.get_embedding("Test text")

    @patch("requests.post")
    def test_get_embedding_model_not_found(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"error": "model 'llama3.2' not found"}
        mock_post.return_value = mock_response

        with self.assertRaises(OllamaModelNotFoundError):
            self.client.get_embedding("Test text")

    @patch("requests.post")
    def test_complete_message_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "Hello!"}]
        res = self.client.complete_message(messages)

        mock_post.assert_called_once_with(
            "http://localhost:11434/api/chat",
            json={"model": "llama3.2", "messages": messages, "stream": False},
            timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        )
        self.assertEqual(res.status_code, 200)

    @patch("requests.post")
    def test_complete_message_connection_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

        with self.assertRaises(OllamaConnectionError):
            self.client.complete_message([{"role": "user", "content": "Hi"}])

    @patch("requests.post")
    def test_slow_answer_is_a_timeout_not_a_connection_error(self, mock_post):
        # Ollama is running but the model is too slow: telling the user to start
        # Ollama would be wrong advice.
        mock_post.side_effect = requests.exceptions.ReadTimeout("Read timed out")

        with self.assertRaises(OllamaTimeoutError) as raised:
            self.client.complete_message([{"role": "user", "content": "Hi"}])

        self.assertNotIsInstance(raised.exception, OllamaConnectionError)
        self.assertEqual(raised.exception.timeout, DEFAULT_REQUEST_TIMEOUT_SECONDS)

    @patch("requests.post")
    def test_connect_timeout_means_ollama_is_unreachable(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectTimeout("connect timed out")

        with self.assertRaises(OllamaConnectionError):
            self.client.complete_message([{"role": "user", "content": "Hi"}])

    @patch("requests.post")
    def test_invalid_json_answer_is_a_response_error(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = requests.exceptions.JSONDecodeError("Expecting value", "<html>", 0)
        mock_post.return_value = mock_response

        with self.assertRaises(OllamaResponseError):
            self.client.get_embedding("Hello")

    def test_request_timeout_is_configurable(self):
        client = OllamaHTTPClient(model="llama3.2", request_timeout=300)

        self.assertEqual(client.request_timeout, 300)
        with self.assertRaises(ValueError):
            OllamaHTTPClient(model="llama3.2", request_timeout=0)

    def test_update_options_changes_the_next_request(self):
        client = OllamaHTTPClient(model="llama3.2", options={"temperature": 0.2, "num_ctx": 4096})

        client.update_options({"temperature": 0.9})

        payload = client._chat_payload([], False, None)
        self.assertEqual(payload["options"], {"temperature": 0.9, "num_ctx": 4096})

    @patch("requests.post")
    def test_complete_message_model_not_found(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"error": "model 'unknown:latest' not found, try pulling it first"}
        mock_post.return_value = mock_response

        with self.assertRaises(OllamaModelNotFoundError) as raised:
            self.client.complete_message([{"role": "user", "content": "Hi"}])
        self.assertEqual(raised.exception.model, "llama3.2")

    @patch("requests.post")
    def test_complete_message_response_error(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": "Out of memory"}
        mock_post.return_value = mock_response

        with self.assertRaises(OllamaResponseError) as ctx:
            self.client.complete_message([{"role": "user", "content": "Hi"}])
        self.assertEqual(ctx.exception.status_code, 500)

    @patch("requests.post")
    def test_stream_message_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            b'{"message": {"content": "Hello"}}',
            b'{"message": {"content": " World"}}',
        ]
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "Tell me a story"}]
        chunks = list(self.client.stream_message(messages))

        mock_post.assert_called_once_with(
            "http://localhost:11434/api/chat",
            json={"model": "llama3.2", "messages": messages, "stream": True},
            stream=True,
            timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        )
        self.assertEqual(chunks, [
            b'{"message": {"content": "Hello"}}',
            b'{"message": {"content": " World"}}',
        ])

    def test_chat_payload_includes_configured_ollama_options(self):
        client = OllamaHTTPClient(model="llama3.2", options={"temperature": 0.2, "num_ctx": 4096})

        payload = client._chat_payload([{"role": "user", "content": "Hello"}], True, None)

        self.assertEqual(payload["options"], {"temperature": 0.2, "num_ctx": 4096})

    @patch("requests.post")
    def test_stream_message_connection_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection lost")

        with self.assertRaises(OllamaConnectionError):
            list(self.client.stream_message([{"role": "user", "content": "Hi"}]))

    @patch("requests.post")
    def test_stream_message_model_not_found(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"error": "model 'llama3.2' not found"}
        mock_post.return_value = mock_response

        with self.assertRaises(OllamaModelNotFoundError):
            list(self.client.stream_message([{"role": "user", "content": "Hi"}]))

    TOOLS = [{
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time",
            "parameters": {"type": "object", "properties": {}},
        },
    }]

    @patch("requests.post")
    def test_complete_message_with_tools(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "What time is it?"}]
        self.client.complete_message(messages, tools=self.TOOLS)

        mock_post.assert_called_once_with(
            "http://localhost:11434/api/chat",
            json={"model": "llama3.2", "messages": messages, "stream": False, "tools": self.TOOLS},
            timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        )

    @patch("requests.post")
    def test_stream_message_with_tools(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = []
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "What time is it?"}]
        list(self.client.stream_message(messages, tools=self.TOOLS))

        mock_post.assert_called_once_with(
            "http://localhost:11434/api/chat",
            json={"model": "llama3.2", "messages": messages, "stream": True, "tools": self.TOOLS},
            stream=True,
            timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        )

    @patch("requests.post")
    def test_empty_tools_are_omitted_from_payload(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        self.client.complete_message([], tools=[])

        self.assertNotIn("tools", mock_post.call_args.kwargs["json"])

    def test_rejects_invalid_model_list_response(self):
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"models": "not a list"}
            mock_get.return_value = mock_response

            with self.assertRaisesRegex(OllamaResponseError, "invalid model-list"):
                self.client.list_models()

    def test_rejects_invalid_embedding_response(self):
        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = ["not an object"]
            mock_post.return_value = mock_response

            with self.assertRaisesRegex(OllamaResponseError, "invalid embedding"):
                self.client.get_embedding("Hello")

    def test_rejects_non_positive_request_timeout(self):
        with self.assertRaisesRegex(ValueError, "request_timeout"):
            OllamaHTTPClient(request_timeout=0)

    @patch("requests.post")
    def test_complete_message_with_output_format(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        schema = {"type": "object", "properties": {"title": {"type": "string"}}}

        self.client.complete_message([], output_format=schema)

        self.assertEqual(mock_post.call_args.kwargs["json"]["format"], schema)

    @patch("requests.post")
    def test_output_format_is_omitted_by_default(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        self.client.complete_message([])

        self.assertNotIn("format", mock_post.call_args.kwargs["json"])

    def test_parse_tool_calls_from_message(self):
        message = {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "add", "arguments": {"a": 1, "b": 2}}},
            {"function": {"name": "get_time", "arguments": {}}},
        ]}
        self.assertEqual(parse_tool_calls(message), [
            {"name": "add", "arguments": {"a": 1, "b": 2}},
            {"name": "get_time", "arguments": {}},
        ])

    def test_parse_tool_calls_from_streamed_chunk(self):
        chunk = json.loads(
            b'{"message": {"role": "assistant", "content": "", '
            b'"tool_calls": [{"function": {"name": "add", "arguments": {"a": 1}}}]}, "done": false}'
        )
        self.assertEqual(parse_tool_calls(chunk["message"]), [{"name": "add", "arguments": {"a": 1}}])

    def test_parse_tool_calls_decodes_string_arguments(self):
        message = {"tool_calls": [{"function": {"name": "add", "arguments": '{"a": 1}'}}]}
        self.assertEqual(parse_tool_calls(message), [{"name": "add", "arguments": {"a": 1}}])

    def test_parse_tool_calls_invalid_string_arguments(self):
        message = {"tool_calls": [{"function": {"name": "add", "arguments": "not json"}}]}
        self.assertEqual(parse_tool_calls(message), [{"name": "add", "arguments": {}}])

    def test_parse_tool_calls_without_tool_calls(self):
        self.assertEqual(parse_tool_calls({"role": "assistant", "content": "Hi"}), [])
        self.assertEqual(parse_tool_calls({}), [])
        self.assertEqual(parse_tool_calls(None), [])

    def test_parse_tool_calls_ignores_malformed_calls(self):
        message = {
            "tool_calls": [
                None,
                {"function": None},
                {"function": {"name": ""}},
                {"function": {"name": "add", "arguments": [1, 2]}},
                {"function": {"name": "get_time", "arguments": {}}},
            ]
        }

        self.assertEqual(
            parse_tool_calls(message),
            [{"name": "add", "arguments": {}}, {"name": "get_time", "arguments": {}}],
        )

    def test_parse_tool_calls_ignores_a_non_list_tool_calls_value(self):
        self.assertEqual(parse_tool_calls({"tool_calls": {"unexpected": "object"}}), [])

    def test_build_tool_message(self):
        self.assertEqual(
            build_tool_message("get_time", "12:00"),
            {"role": "tool", "tool_name": "get_time", "content": "12:00"},
        )


def run_integration_test() -> int:
    """Live check against the Ollama server and models configured in config.json.

    Each step prints PASS or FAIL and the run ends with one verdict; the exit code is 0
    only when every step passed, so the result can be read without knowing the code.
    """
    from config import load_config

    config = load_config()
    chat = OllamaHTTPClient(
        base_url=config.model.base_url,
        model=config.model.chat_model,
        options={"temperature": 0, "num_ctx": config.model.context_window},
        request_timeout=config.model.request_timeout_seconds,
    )
    embedder = OllamaHTTPClient(base_url=config.model.base_url, model=config.model.embedding_model)
    results: list[tuple[str, bool, str]] = []

    def step(name, check):
        try:
            passed, detail = check()
        except OllamaError as error:
            passed, detail = False, f"{type(error).__name__}: {error}"
        results.append((name, passed, detail))
        print(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}")

    def models_installed():
        installed = chat.list_models()
        wanted = [config.model.chat_model, config.model.embedding_model]
        missing = [model for model in wanted if not any(name.split(":")[0] == model.split(":")[0] for name in installed)]
        return not missing, f"missing {missing}" if missing else f"{', '.join(wanted)} installed"

    def complete():
        content = chat.complete_message([{"role": "user", "content": "Réponds seulement : OK"}]).json()["message"]["content"]
        return "ok" in content.lower(), repr(content.strip()[:60])

    def stream():
        chunks, text, stats = 0, "", {}
        for line in chat.stream_message([{"role": "user", "content": "Compte de 1 à 5."}]):
            if line:
                payload = json.loads(line)
                chunks += 1
                text += payload.get("message", {}).get("content", "")
                if payload.get("done"):
                    stats = payload
        speed = stats.get("eval_count", 0) / max(stats.get("eval_duration", 1) / 1e9, 1e-9)
        return chunks > 1 and "5" in text, f"{chunks} chunks streamed, {speed:.1f} tokens/s"

    def embed():
        response = embedder.get_embedding("Les fractions équivalentes")
        vectors = response.get("embeddings") or [response.get("embedding")]
        dimension = len(vectors[0] or [])
        return dimension > 0, f"{config.model.embedding_model}: {dimension} dimensions"

    print(f"Live Ollama check: {config.model.base_url}")
    step("installed models", models_installed)
    step("complete_message", complete)
    step("stream_message", stream)
    step("get_embedding", embed)
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"VERDICT: {'PASS' if passed == len(results) else 'FAIL'} ({passed}/{len(results)} steps)")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    if "--integration" in sys.argv or "-i" in sys.argv:
        sys.exit(run_integration_test())
    else:
        unittest.main()

import unittest
from unittest.mock import patch, MagicMock
import requests
import sys
import os

# Ensure `source` directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

# pyrefly: ignore [missing-import]
from utilities.ollama_client import (
    OllamaHTTPClient,
    OllamaError,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaResponseError,
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

        mock_get.assert_called_once_with("http://localhost:11434/api/tags")
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
        )
        self.assertEqual(res.status_code, 200)

    @patch("requests.post")
    def test_complete_message_connection_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout("Timeout error")

        with self.assertRaises(OllamaConnectionError):
            self.client.complete_message([{"role": "user", "content": "Hi"}])

    @patch("requests.post")
    def test_complete_message_model_not_found(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"error": "model 'unknown:latest' not found, try pulling it first"}
        mock_post.return_value = mock_response

        with self.assertRaises(OllamaModelNotFoundError):
            self.client.complete_message([{"role": "user", "content": "Hi"}])

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
        )
        self.assertEqual(chunks, [
            b'{"message": {"content": "Hello"}}',
            b'{"message": {"content": " World"}}',
        ])

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


def run_integration_test():
    """Interactive integration test against a live Ollama server."""
    print("=" * 60)
    print("Running Interactive Integration Test against Live Ollama Server")
    print("=" * 60)

    client = OllamaHTTPClient()
    print(f"Connecting to: {client.base_url} (model: {client.model})")

    # 1. Test list_models
    try:
        print("\n1. Testing list_models()...")
        models = client.list_models()
        print(f"[OK] Available models ({len(models)}): {models}")
    except Exception as e:
        print(f"[ERROR] Failed to list models: {e}")
        return

    # Select model if default model isn't listed
    if client.model not in models and models:
        print(f"[WARNING] Model '{client.model}' not found in installed models. Switching to '{models[0]}'")
        client.model = models[0]

    # 2. Test complete_message
    try:
        print(f"\n2. Testing complete_message() with model '{client.model}'...")
        messages = [{"role": "user", "content": "Say 'Test passed!' in one sentence."}]
        res = client.complete_message(messages)
        if res.status_code == 200:
            content = res.json().get("message", {}).get("content", "")
            print(f"[OK] Response: {content.strip()}")
        else:
            print(f"[WARNING] HTTP {res.status_code}: {res.text}")
    except Exception as e:
        print(f"[ERROR] complete_message failed: {e}")

    # 3. Test stream_message
    try:
        print(f"\n3. Testing stream_message() with model '{client.model}'...")
        messages = [{"role": "user", "content": "Count from 1 to 5."}]
        print("Output stream: ", end="", flush=True)
        import json
        for line in client.stream_message(messages):
            if line:
                data = json.loads(line)
                tok = data.get("message", {}).get("content", "")
                print(tok, end="", flush=True)
        print("\n[OK] Streaming complete.")
    except Exception as e:
        print(f"\n[ERROR] stream_message failed: {e}")

    # 4. Test get_embedding
    try:
        print(f"\n4. Testing get_embedding() with model '{client.model}'...")
        emb = client.get_embedding("Hello test")
        if "embedding" in emb:
            print(f"[OK] Embedding vector generated (dim: {len(emb['embedding'])})")
        elif "embeddings" in emb:
            print(f"[OK] Embedding vector generated (dim: {len(emb['embeddings'][0])})")
        else:
            print(f"[INFO] Response: {emb}")
    except Exception as e:
        print(f"[ERROR] get_embedding failed: {e}")

    print("\n=" * 60)
    print("Integration Test Completed!")
    print("=" * 60)


if __name__ == "__main__":
    if "--integration" in sys.argv or "-i" in sys.argv:
        run_integration_test()
    else:
        unittest.main()

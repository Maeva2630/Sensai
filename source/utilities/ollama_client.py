import requests
import json
from typing import List, Dict, Any, Generator, Optional


DEFAULT_REQUEST_TIMEOUT_SECONDS = 30


class OllamaError(Exception):
    """Base exception for all Ollama client errors."""
    pass


class OllamaConnectionError(OllamaError):
    """Raised when unable to connect to the Ollama server."""
    pass


class OllamaModelNotFoundError(OllamaError):
    """Raised when the requested model is not found on Ollama."""
    pass


class OllamaResponseError(OllamaError):
    """Raised when Ollama returns an HTTP error status code."""

    def __init__(self, message: str, status_code: Optional[int] = None, response_text: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text


def build_tool_message(tool_name: str, content: str) -> Dict[str, str]:
    """Build a `tool` role message carrying a tool result back to the model."""
    return {"role": "tool", "tool_name": tool_name, "content": content}


def parse_tool_calls(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract tool calls from a chat response message (full or streamed chunk).

    Returns a list of {"name": str, "arguments": dict}. Arguments sent as a JSON
    string are decoded; undecodable arguments become an empty dict.
    """
    if not isinstance(message, dict):
        return []

    raw_calls = message.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        return []

    calls = []
    for call in raw_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function", {})
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        calls.append({"name": name, "arguments": arguments})
    return calls


class OllamaHTTPClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        options: Optional[Dict[str, Any]] = None,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ):
        if isinstance(request_timeout, bool) or request_timeout <= 0:
            raise ValueError("request_timeout must be greater than zero.")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.options = dict(options or {})
        self.request_timeout = request_timeout

    def _handle_error_response(self, response: requests.Response) -> None:
        """Parse error message from response and raise appropriate exception."""
        error_msg = ""
        try:
            data = response.json()
            error_msg = data.get("error", response.text)
        except Exception:
            error_msg = response.text or f"HTTP {response.status_code}"

        if response.status_code == 404 or "not found" in error_msg.lower():
            raise OllamaModelNotFoundError(f"Model '{self.model}' not found on Ollama server: {error_msg}")
        else:
            raise OllamaResponseError(
                f"Ollama API error ({response.status_code}): {error_msg}",
                status_code=response.status_code,
                response_text=response.text,
            )

    def list_models(self) -> List[str]:
        """Fetch list of available models from Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.request_timeout)
            if response.status_code != 200:
                self._handle_error_response(response)
            
            data = response.json()
            if not isinstance(data, dict):
                raise OllamaResponseError("Ollama returned an invalid model-list response.")
            models_list = data.get("models", [])
            if not isinstance(models_list, list):
                raise OllamaResponseError("Ollama returned an invalid model-list response.")
            return [
                model["name"]
                for model in models_list
                if isinstance(model, dict) and isinstance(model.get("name"), str)
            ]
        except (OllamaModelNotFoundError, OllamaResponseError):
            raise
        except requests.exceptions.RequestException as e:
            raise OllamaConnectionError(f"Failed to connect to Ollama: {e}") from e
        except Exception as e:
            raise OllamaConnectionError(f"Unexpected error while listing models: {e}") from e

    def get_embedding(self, text: str) -> Dict[str, Any]:
        """Generate embedding vector for a given text input."""
        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": text},
                timeout=self.request_timeout,
            )
            if response.status_code != 200:
                self._handle_error_response(response)
            data = response.json()
            if not isinstance(data, dict):
                raise OllamaResponseError("Ollama returned an invalid embedding response.")
            return data
        except (OllamaModelNotFoundError, OllamaResponseError):
            raise
        except requests.exceptions.RequestException as e:
            raise OllamaConnectionError(f"Failed to connect to Ollama: {e}") from e
        except Exception as e:
            raise OllamaConnectionError(f"Unexpected error while getting embedding: {e}") from e

    def _chat_payload(self, messages: List[Dict[str, Any]], stream: bool,
                      tools: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        payload = {"model": self.model, "messages": messages, "stream": stream}
        if self.options:
            payload["options"] = self.options
        if tools:
            payload["tools"] = tools
        return payload

    def complete_message(self, messages: List[Dict[str, Any]],
                         tools: Optional[List[Dict[str, Any]]] = None) -> requests.Response:
        """Send a non-streaming chat completion request, optionally offering tools."""
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=self._chat_payload(messages, False, tools),
                timeout=self.request_timeout,
            )
            if response.status_code != 200:
                self._handle_error_response(response)
            return response
        except (OllamaModelNotFoundError, OllamaResponseError):
            raise
        except requests.exceptions.RequestException as e:
            raise OllamaConnectionError(f"Failed to connect to Ollama: {e}") from e
        except Exception as e:
            raise OllamaConnectionError(f"Unexpected error during completion: {e}") from e

    def stream_message(self, messages: List[Dict[str, Any]],
                       tools: Optional[List[Dict[str, Any]]] = None) -> Generator[bytes, None, None]:
        """Send a streaming chat completion request and yield response chunks."""
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=self._chat_payload(messages, True, tools),
                stream=True,
                timeout=self.request_timeout,
            )
            if response.status_code != 200:
                self._handle_error_response(response)
            for line in response.iter_lines():
                yield line
        except (OllamaModelNotFoundError, OllamaResponseError):
            raise
        except requests.exceptions.RequestException as e:
            raise OllamaConnectionError(f"Failed to connect to Ollama: {e}") from e
        except Exception as e:
            raise OllamaConnectionError(f"Unexpected error during streaming: {e}") from e

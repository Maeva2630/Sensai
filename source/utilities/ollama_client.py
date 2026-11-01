import requests
import json
from typing import List, Dict, Any, Generator, Optional


# Longest wait for the next piece of a response. A 3B model on a laptop needs more than
# 30 s for a 20-question quiz (2,400 tokens at about 56 tokens/s), so the default is higher
# and config.json can change it (model.request_timeout_seconds).
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120


class OllamaError(Exception):
    """Base exception for all Ollama client errors."""
    pass


class OllamaConnectionError(OllamaError):
    """Raised when unable to connect to the Ollama server."""
    pass


class OllamaModelNotFoundError(OllamaError):
    """Raised when the requested model is not found on Ollama."""

    def __init__(self, message: str, model: str = ""):
        super().__init__(message)
        self.model = model


class OllamaTimeoutError(OllamaError):
    """Raised when Ollama is reachable but does not answer within the request timeout."""

    def __init__(self, message: str, timeout: float = 0.0):
        super().__init__(message)
        self.timeout = timeout


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
        model: str = "qwen2.5:3b",
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
            error_msg = data.get("error", response.text) if isinstance(data, dict) else response.text
        except ValueError:
            error_msg = response.text or f"HTTP {response.status_code}"

        if response.status_code == 404 or "not found" in str(error_msg).lower():
            raise OllamaModelNotFoundError(
                f"Model '{self.model}' not found on Ollama server: {error_msg}", model=self.model
            )
        else:
            raise OllamaResponseError(
                f"Ollama API error ({response.status_code}): {error_msg}",
                status_code=response.status_code,
                response_text=response.text,
            )

    def _domain_error(self, error: Exception, action: str) -> OllamaError:
        """Translate a low-level failure into the error the user can act on.

        A refused connection, a slow model and a malformed answer need different
        advice, so they must not all be reported as "Ollama is unreachable".
        """
        if isinstance(error, requests.exceptions.ConnectionError):
            # Includes ConnectTimeout: the server could not be reached at all.
            return OllamaConnectionError(f"Failed to connect to Ollama at {self.base_url}: {error}")
        if isinstance(error, requests.exceptions.Timeout):
            return OllamaTimeoutError(
                f"Ollama did not answer within {self.request_timeout:g} s {action}.",
                timeout=self.request_timeout,
            )
        if isinstance(error, ValueError):
            # Includes requests' JSONDecodeError: the server answered, but not with JSON.
            return OllamaResponseError(f"Ollama returned an invalid response {action}: {error}")
        if isinstance(error, requests.exceptions.RequestException):
            return OllamaConnectionError(f"Failed to connect to Ollama: {error}")
        return OllamaResponseError(f"Unexpected error {action}: {error}")

    def update_options(self, options: Dict[str, Any]) -> None:
        """Change generation options (temperature, num_ctx...) for the next requests."""
        self.options.update(options)

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
        except OllamaError:
            raise
        except Exception as e:
            raise self._domain_error(e, "while listing models") from e

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
        except OllamaError:
            raise
        except Exception as e:
            raise self._domain_error(e, "while getting embedding") from e

    def _chat_payload(self, messages: List[Dict[str, Any]], stream: bool,
                      tools: Optional[List[Dict[str, Any]]],
                      output_format: Any = None,
                      options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {"model": self.model, "messages": messages, "stream": stream}
        merged_options = {**self.options, **(options or {})}
        if merged_options:
            payload["options"] = merged_options
        if tools:
            payload["tools"] = tools
        if output_format:
            payload["format"] = output_format
        return payload

    def complete_message(self, messages: List[Dict[str, Any]],
                         tools: Optional[List[Dict[str, Any]]] = None,
                         output_format: Any = None) -> requests.Response:
        """Send a non-streaming chat completion request.

        `tools` offers tools to the model. `output_format` is `"json"` or a JSON schema:
        Ollama then constrains the answer to valid JSON (matching the schema).
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=self._chat_payload(messages, False, tools, output_format),
                timeout=self.request_timeout,
            )
            if response.status_code != 200:
                self._handle_error_response(response)
            return response
        except OllamaError:
            raise
        except Exception as e:
            raise self._domain_error(e, "during completion") from e

    def stream_message(self, messages: List[Dict[str, Any]],
                       tools: Optional[List[Dict[str, Any]]] = None,
                       options: Optional[Dict[str, Any]] = None) -> Generator[bytes, None, None]:
        """Send a streaming chat completion request and yield response chunks.

        `options` are added to the client's generation options for this request only
        (for example `num_predict`, the longest answer allowed).
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=self._chat_payload(messages, True, tools, options=options),
                stream=True,
                timeout=self.request_timeout,
            )
            if response.status_code != 200:
                self._handle_error_response(response)
            for line in response.iter_lines():
                yield line
        except OllamaError:
            raise
        except Exception as e:
            raise self._domain_error(e, "during streaming") from e

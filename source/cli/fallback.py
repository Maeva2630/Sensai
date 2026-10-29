"""Controlled chat-model fallback for unavailable Ollama models."""

from __future__ import annotations

from collections.abc import Callable, Generator, Sequence
from typing import Any

from utilities.ollama_client import OllamaHTTPClient, OllamaModelNotFoundError


FallbackReporter = Callable[[str, str], None]


class FallbackChatClient:
    """Try configured chat models in order when Ollama cannot find one.

    A fallback is attempted only before a stream has produced content and only for a
    missing-model error. Connection failures and interrupted responses are surfaced to
    the user because retrying another local model would either fail for the same server
    outage or risk duplicating a partial answer.
    """

    def __init__(
        self,
        clients: Sequence[OllamaHTTPClient],
        on_fallback: FallbackReporter | None = None,
        model_names: Sequence[str] | None = None,
    ) -> None:
        if not clients:
            raise ValueError("FallbackChatClient requires at least one chat client.")
        if model_names is not None and len(model_names) != len(clients):
            raise ValueError("FallbackChatClient model names must match its clients.")
        self._clients = tuple(clients)
        self._model_names = tuple(model_names) if model_names is not None else tuple(client.model for client in clients)
        self._active_index = 0
        self._on_fallback = on_fallback

    @property
    def model(self) -> str:
        """Return the model that handled the latest successful request."""
        return self._model_names[self._active_index]

    def stream_message(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Generator[bytes, None, None]:
        """Stream with a fallback only if no response chunk has been yielded."""
        for index in range(self._active_index, len(self._clients)):
            client = self._clients[index]
            yielded_response = False
            try:
                for chunk in client.stream_message(messages, tools=tools):
                    yielded_response = True
                    yield chunk
            except OllamaModelNotFoundError:
                if yielded_response or index == len(self._clients) - 1:
                    raise
                self._notify_fallback(self._model_names[index], self._model_names[index + 1])
                continue
            else:
                self._active_index = index
                return

    def complete_message(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        output_format: Any = None,
    ) -> Any:
        """Complete a non-streaming request using the same missing-model policy."""
        for index in range(self._active_index, len(self._clients)):
            client = self._clients[index]
            try:
                response = client.complete_message(messages, tools=tools, output_format=output_format)
            except OllamaModelNotFoundError:
                if index == len(self._clients) - 1:
                    raise
                self._notify_fallback(self._model_names[index], self._model_names[index + 1])
                continue
            self._active_index = index
            return response
        raise RuntimeError("No configured chat model could complete the request.")

    def _notify_fallback(self, primary: str, fallback: str) -> None:
        if self._on_fallback is not None:
            self._on_fallback(primary, fallback)

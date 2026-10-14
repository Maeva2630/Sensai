import sys
import json
from typing import Any, Dict, List, Optional, Tuple
from utilities.ollama_client import (
    OllamaHTTPClient,
    OllamaConnectionError,
    OllamaError,
    build_tool_message,
    parse_tool_calls,
)
from tools.registry import ToolRegistry
from tools.time_tool import GetTimeTool

MAX_TOOL_ITERATIONS = 5


def default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(GetTimeTool())
    return registry


def stream_response(client: OllamaHTTPClient, messages: List[Dict[str, Any]],
                    tools: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """Stream one model response to the terminal.

    Returns the text printed and the raw tool calls requested by the model.
    """
    full_response = ""
    raw_tool_calls: List[Dict[str, Any]] = []
    printed_prefix = False

    for line in client.stream_message(messages, tools=tools):
        if not line:
            continue
        message = json.loads(line).get("message", {})
        content = message.get("content", "")
        if content:
            if not printed_prefix:
                print("Assistant: ", end="", flush=True)
                printed_prefix = True
            print(content, end="", flush=True)
            full_response += content
        raw_tool_calls.extend(message.get("tool_calls") or [])

    if printed_prefix:
        print() # Print '\n'
    return full_response, raw_tool_calls


def run_tool_calls(registry: ToolRegistry, raw_tool_calls: List[Dict[str, Any]],
                   messages: List[Dict[str, Any]]) -> None:
    """Execute each tool call and append its result to the history."""
    for call in parse_tool_calls({"tool_calls": raw_tool_calls}):
        print(f"[tool] {call['name']}({json.dumps(call['arguments'])})")
        result = registry.execute(call["name"], call["arguments"])
        label = "error" if result.is_error else "result"
        print(f"[tool] {label}: {result.content}")
        messages.append(build_tool_message(call["name"], result.content))


def cli(model: str = "llama3.2", registry: Optional[ToolRegistry] = None):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    client = OllamaHTTPClient(model=model)
    registry = registry if registry is not None else default_registry()
    tools = registry.to_ollama_tools()

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            if not user_input.strip():
                continue

            turn_start = len(messages)
            messages.append({"role": "user", "content": user_input})

            for _ in range(MAX_TOOL_ITERATIONS):
                full_response, raw_tool_calls = stream_response(client, messages, tools)

                assistant_message: Dict[str, Any] = {"role": "assistant", "content": full_response}
                if raw_tool_calls:
                    assistant_message["tool_calls"] = raw_tool_calls
                messages.append(assistant_message)

                if not raw_tool_calls:
                    break
                run_tool_calls(registry, raw_tool_calls, messages)
            else:
                print(f"\nWarning: stopped after {MAX_TOOL_ITERATIONS} tool calls without a final answer.")

        except KeyboardInterrupt:
            print("\nExiting.")
            sys.exit(0)

        except OllamaConnectionError as e:
            del messages[turn_start:]
            print(f"\nConnection Error: Unable to connect to Ollama server ({e}). Please ensure Ollama is running.")

        except OllamaError as e:
            del messages[turn_start:]
            print(f"\nWarning: {e}")

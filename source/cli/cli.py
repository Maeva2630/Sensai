import sys
import json
from utilities.ollama_client import OllamaHTTPClient, OllamaConnectionError, OllamaError
from memory.session_manager import SessionManager
from memory.profile_manager import ProfileManager

def cli(model: str = "llama3.2", session_id: str = None, profile_path: str = None):
    """Launch the CLI application with optional session resumption."""

    try:
        session = SessionManager(session_id=session_id)
        profile_manager = ProfileManager(profile_path=profile_path)
        messages = session.load(session_id) if session_id else [{"role": "system", "content": profile_manager.build_system_prompt()}]
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    client = OllamaHTTPClient(model=model)

    def cmd_exit(_args):
        session.save()
        sys.exit(0)

    def cmd_save(_args):
        session.save()

    def cmd_load(args):
        if not args:
            print("Usage: /load <session_id>")
            return
        try:
            session.load(args[0])
            print(f"Session {args[0]} loaded.")
        except FileNotFoundError as e:
            print(f"Error: {e}")

    COMMANDS = {
        "/save": cmd_save,
        "/load": cmd_load,
        "exit":  cmd_exit,
        "quit":  cmd_exit,
    }

    while True:
        try:
            user_input = input("You: ")

            if not user_input.strip():
                continue

            parts = user_input.strip().split()
            command = parts[0].lower()
            args = parts[1:]

            if command in COMMANDS:
                COMMANDS[command](args)
                continue

            messages.append({"role": "user", "content": user_input})
            response = client.stream_message(messages)

            full_response = ""
            print("Assistant: ", end="", flush=True)

            for line in response:
                if line:
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    print(content, end="", flush=True)
                    full_response += content

            print()
            messages.append({"role": "assistant", "content": full_response})
            session.add_message("user", user_input)
            session.add_message("assistant", full_response)

        except (KeyboardInterrupt, EOFError):
            session.save()
            print("Exiting.")
            sys.exit(0)

        except OllamaConnectionError as e:
            if messages and messages[-1].get("role") == "user":
                messages.pop()
            print(f"\nConnection Error: Unable to connect to Ollama server ({e}). Please ensure Ollama is running.")

        except OllamaError as e:
            if messages and messages[-1].get("role") == "user":
                messages.pop()
            print(f"\nWarning: {e}")

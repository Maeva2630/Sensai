import sys
import json
from utilities.ollama_client import OllamaHTTPClient, OllamaConnectionError, OllamaError

from memory.session_manager import SessionManager

def cli(model: str = "llama3.2"):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    client = OllamaHTTPClient(model=model)
    session = SessionManager()

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                saved_path = session.save()
                print(f"Session saved at {saved_path}")
                break
            if not user_input.strip():
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

            print() # Print '\n'
            messages.append({"role": "assistant", "content": full_response})

            session.add_message("user", user_input)
            session.add_message("assistant", full_response)


        except (KeyboardInterrupt, EOFError):
            saved_path = session.save()
            print(f"\nSession saved at {saved_path}")
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

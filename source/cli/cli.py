import sys
import json
from utilities.ollama_client import OllamaHTTPClient, OllamaConnectionError, OllamaError


def cli(model: str = "llama3.2"):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    client = OllamaHTTPClient(model=model)

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
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

        except KeyboardInterrupt:
            print("\nExiting.")
            sys.exit(0)

        except OllamaConnectionError as e:
            if messages and messages[-1].get("role") == "user":
                messages.pop()
            print(f"\nConnection Error: Unable to connect to Ollama server ({e}). Please ensure Ollama is running.")

        except OllamaError as e:
            if messages and messages[-1].get("role") == "user":
                messages.pop()
            print(f"\nWarning: {e}")

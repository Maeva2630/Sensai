import time
import json
from utilities.ollama_client import OllamaHTTPClient

def main():
    client = OllamaHTTPClient()

    messages = [{"role": "user", "content": "Tell me a really short story about sea shells"}]

    print("--- Streaming test ---")
    for line in client.stream_message(messages):
        data = json.loads(line)
        content = data.get("message", {}).get("content", "")
        
        print(content, end="", flush=True)

    print("\n\n--- End of the streaming test ---")

    print("\n--- Models list test ---")
    for model in client.list_models():
        print(f"- {model}")

    print("\n\n--- End of the models list test ---")

    print("\n\n--- Embedding test ---")
    # print(client.get_embedding("Hello !"))
    print("\n\n--- End of the embedding test ---")

if __name__ == "__main__":
    main()

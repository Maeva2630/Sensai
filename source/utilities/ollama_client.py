import requests

class OllamaHTTPClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2"):
        self.base_url = base_url
        self.model = model

    def chat(self, messages):
        try:
            response = requests.post(
                self.base_url + "/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                },
            )
            return response.json()
        except Exception as e:
            raise OllamaConnectionError(f"Failed to connect to Ollama: {e}")


class OllamaError(Exception):
    pass

class OllamaConnectionError(OllamaError):
    pass

class OllamaModelNotFoundError(OllamaError):
    pass

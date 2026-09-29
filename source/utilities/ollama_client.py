import requests
import json

class OllamaHTTPClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2"):
        self.base_url = base_url
        self.model = model

    def list_models(self):
        try:
            response = requests.get(self.base_url + "/api/tags")
            models = []
            for model in response.json()["models"]:
                models.append(model["name"])
            return models
        except Exception as e:
            raise OllamaConnectionError(f"Failed to connect to Ollama: {e}")

    def get_embedding(self, text: str):
        try:
            response = requests.post(self.base_url + "/api/embed", json={"model": self.model, "input": text})
            return response.json()
        except Exception as e:
            raise OllamaConnectionError(f"Failed to connect to Ollama: {e}")

    def complete_message(self, messages: str):
        try:
            response = requests.post(
                self.base_url + "/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                },
            )
            return response
        except Exception as e:
            raise OllamaConnectionError(f"Failed to connect to Ollama: {e}")

    def stream_message(self, messages: str):
        try:
            response = requests.post(
                self.base_url + "/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                },
                stream=True,
            )
            for line in response.iter_lines():
               yield line
        except Exception as e:
            raise OllamaConnectionError(f"Failed to connect to Ollama: {e}")

class OllamaError(Exception):
    pass

class OllamaConnectionError(OllamaError):
    pass

class OllamaModelNotFoundError(OllamaError):
    pass

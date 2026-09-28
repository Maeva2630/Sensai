import requests

class OllamaHTTPClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2"):
        self.base_url = base_url
        self.model = model

class OllamaError(Exception):
    pass

class OllamaConnectionError(OllamaError):
    pass

class OllamaModelNotFoundError(OllamaError):
    pass

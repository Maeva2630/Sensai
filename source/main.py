from config import ConfigurationError, load_config
from cli.cli import cli
from rag.storage import VectorStoreError

def main():
    try:
        app_config = load_config()
    except ConfigurationError as error:
        print(f"Configuration error: {error}")
        return
    if app_config.features.memory_enabled:
        print("Memory is enabled in config.json but is not implemented yet; it remains inactive.")
    try:
        cli(app_config=app_config)
    except VectorStoreError as error:
        print(f"RAG startup error: {error}")

if __name__ == "__main__":
    main()

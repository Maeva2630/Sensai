from config import ConfigurationError, load_config
from cli.cli import cli

def main():
    try:
        app_config = load_config()
    except ConfigurationError as error:
        print(f"Configuration error: {error}")
        return
    if app_config.features.memory_enabled:
        print("Memory is enabled in config.json but is not implemented yet; it remains inactive.")
    cli(app_config=app_config)

if __name__ == "__main__":
    main()

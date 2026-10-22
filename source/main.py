import argparse

from cli.cli import cli
from config import ConfigurationError, load_config


def main():
    parser = argparse.ArgumentParser(description="Sensai CLI")
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        metavar="SESSION_ID",
        help="Resume a previous session by its ID"
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="profile.json",
        help="Profile path to use (default: profile.json)"
    )

    args = parser.parse_args()

    try:
        app_config = load_config()
    except ConfigurationError as error:
        print(f"Configuration error: {error}")
        return
    if app_config.features.memory_enabled:
        print("Memory is enabled in config.json but is not implemented yet; it remains inactive.")
    cli(
        session_id=args.session,
        profile_path=args.profile,
        app_config=app_config,
    )

if __name__ == "__main__":
    main()

import argparse

from cli.cli import cli
from cli.error_handling import report_cli_error
from config import ConfigurationError, load_config
from observability.app_logging import configure_logging, get_logger

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
    try:
        configure_logging(app_config.logging)
    except OSError as error:
        print(f"Logging configuration error: {error}")
        return
    logger = get_logger("startup")
    logger.info(
        "application.started chat_model=%s embedding_model=%s rag_enabled=%s tools_enabled=%s memory_enabled=%s",
        app_config.model.chat_model,
        app_config.model.embedding_model,
        app_config.features.rag_enabled,
        app_config.features.tools_enabled,
        app_config.features.memory_enabled,
    )
    try:
        cli(
            session_id=args.session,
            profile_path=args.profile,
            app_config=app_config,
        )
    except Exception as error:
        report_cli_error(error, logger)

if __name__ == "__main__":
    main()

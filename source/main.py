import argparse
import errno
import os
import sys
from dataclasses import replace

from cli.cli import cli
from cli.error_handling import format_fatal_error
from config import DEFAULT_CONFIG_PATH, AppConfig, ConfigurationError, load_config
from observability.app_logging import configure_logging, get_logger


def _parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        metavar="PATH",
        help="Configuration file to use (default: config.json at the project root)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Chat model for this run, instead of model.chat_model in the configuration"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature (0 to 2) for this run, instead of model.temperature"
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=None,
        metavar="TOKENS",
        help="Context window (num_ctx, at least 512) for this run, instead of model.context_window"
    )

    parser.add_argument(
        "--web",
        action="store_true",
        help="Start the web interface instead of the terminal chat"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port of the web interface (default: 8000)"
    )
    return parser.parse_args(argv)


def apply_overrides(app_config: AppConfig, args: argparse.Namespace) -> AppConfig:
    """Apply command-line model settings on top of the configuration file, for one run."""
    model = app_config.model
    if args.model is not None:
        if not args.model.strip():
            raise ConfigurationError("--model must not be empty.")
        fallbacks = tuple(name for name in model.fallback_chat_models if name != args.model.strip())
        model = replace(model, chat_model=args.model.strip(), fallback_chat_models=fallbacks)
    if args.temperature is not None:
        if not 0.0 <= args.temperature <= 2.0:
            raise ConfigurationError("--temperature must be between 0 and 2.")
        model = replace(model, temperature=args.temperature)
    if args.context_window is not None:
        if args.context_window < 512:
            raise ConfigurationError("--context-window must be at least 512.")
        model = replace(model, context_window=args.context_window)
    return replace(app_config, model=model)


def main(argv: list[str] | None = None) -> int:
    args = _parse_arguments(argv)
    # Sessions, profile, artifacts, the RAG index, logs and exports can hold pupil data:
    # every file SensAI creates is readable by its owner only.
    os.umask(0o077)

    try:
        app_config = apply_overrides(load_config(args.config), args)
    except ConfigurationError as error:
        print(f"Configuration error: {error}")
        return 1
    try:
        configure_logging(app_config.logging)
    except OSError as error:
        print(f"Logging configuration error: {error}")
        return 1
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
        if args.web:
            from web.app import WebApp
            from web.server import serve

            serve(
                WebApp(app_config, session_id=args.session, profile_path=args.profile),
                args.port,
            )
        else:
            cli(
                session_id=args.session,
                profile_path=args.profile,
                app_config=app_config,
            )
    except KeyboardInterrupt:
        # Ctrl+C before the chat loop (for example while MCP servers start).
        print("\nExiting.")
        return 130
    except OSError as error:
        if args.web and error.errno == errno.EADDRINUSE:
            print(f"Port {args.port} is already in use. Start the web interface on another one with --port.")
            return 1
        print(f"\n{format_fatal_error(error, logger)}\nSensAI stopped. Details are in {app_config.logging.file_path}.")
        return 1
    except Exception as error:
        print(f"\n{format_fatal_error(error, logger)}\nSensAI stopped. Details are in {app_config.logging.file_path}.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

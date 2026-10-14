import argparse
from cli.cli import cli

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
        "--model",
        type=str,
        default="llama3.2",
        help="Ollama model to use (default: llama3.2)"
    )

    args = parser.parse_args()

    cli(model=args.model, session_id=args.session)

if __name__ == "__main__":
    main()

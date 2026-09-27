#!/usr/bin/env python3
"""Interactive setup for the Meridian Retail support-draft review tool.

Writes a project-local .env file. Never prints the API key.
"""

import getpass
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ENV_PATH = APP_DIR / ".env"
DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def write_env_file(path, api_key, model):
    model = (model or "").strip() or DEFAULT_MODEL
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"APP_ANTHROPIC_API_KEY={api_key}\n")
        f.write(f"APP_ANTHROPIC_MODEL={model}\n")
    return path


def main():
    print("Meridian Retail support-draft tool - environment setup")
    print(f"This will write a project-local .env file at: {ENV_PATH}")
    print()

    api_key = getpass.getpass("Enter your Anthropic API key (input hidden): ").strip()
    if not api_key:
        print("No API key entered. Aborting without writing .env.")
        return

    model = input(f"Enter model id [default: {DEFAULT_MODEL}]: ").strip()

    write_env_file(ENV_PATH, api_key, model)
    print(f"Wrote {ENV_PATH.name} (key not shown).")


if __name__ == "__main__":
    main()

"""Interactive setup for the project-only .env file.

Asks which local Ollama model to use and where the Ollama server lives, then
writes a project-local .env and checks that the server is reachable and the
model is installed.

There is no API key: the model runs on this machine, so there is no secret to
enter, store, or leak.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
DEFAULT_MODEL = "qwen3.5:9b"
DEFAULT_HOST = "http://localhost:11434"


def prompt_model(input_func=input):
    model = input_func(f"Enter APP_OLLAMA_MODEL [{DEFAULT_MODEL}]: ").strip()
    return model or DEFAULT_MODEL


def prompt_host(input_func=input):
    host = input_func(f"Enter APP_OLLAMA_HOST [{DEFAULT_HOST}]: ").strip()
    return host or DEFAULT_HOST


def write_env(model, host, env_path=ENV_PATH):
    """Write the project-only .env file. Returns the path written."""
    contents = (
        f"APP_OLLAMA_MODEL={model}\n"
        f"APP_OLLAMA_HOST={host}\n"
    )
    env_path.write_text(contents, encoding="utf-8")
    try:
        os.chmod(env_path, 0o600)
    except OSError:
        pass
    return env_path


def check_ollama(model, host):
    """Return a human-readable line about the local server and model.

    Never raises: a failed check is reported, not fatal, so setup still
    finishes and the app's own safe state handles a missing server later.
    """
    try:
        import ollama

        installed = [m.model for m in ollama.Client(host=host).list().models]
    except Exception as exc:
        return (
            f"  WARNING: could not reach Ollama at {host} "
            f"({type(exc).__name__}). Start it with: ollama serve"
        )
    if model in installed:
        return f"  OK: {model} is installed and {host} is reachable."
    return (
        f"  WARNING: {host} is reachable but {model!r} is not installed. "
        f"Install it with: ollama pull {model}"
    )


def run_setup(input_func=input, print_func=print, env_path=ENV_PATH, check_func=check_ollama):
    model = prompt_model(input_func)
    host = prompt_host(input_func)
    write_env(model, host, env_path)
    print_func(f"Wrote {env_path}")
    print_func(f"  APP_OLLAMA_MODEL = {model}")
    print_func(f"  APP_OLLAMA_HOST  = {host}")
    print_func(check_func(model, host))
    print_func("Done. No API key is needed; the model runs locally.")


if __name__ == "__main__":
    run_setup()

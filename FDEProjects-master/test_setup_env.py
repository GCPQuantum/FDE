"""Tests for setup_env.py. No external API and no Ollama server is involved.

Verifies that .env is written with the model and host, and that setup never
writes an API key (the local model needs none).
"""

import setup_env


def fake_check(model, host):
    return "  OK: fake check"


def test_write_env_creates_file(tmp_path):
    env_path = tmp_path / ".env"
    setup_env.write_env("some-model", "http://example:1234", env_path=env_path)
    contents = env_path.read_text(encoding="utf-8")
    assert "APP_OLLAMA_MODEL=some-model" in contents
    assert "APP_OLLAMA_HOST=http://example:1234" in contents


def test_run_setup_writes_defaults_and_no_key(tmp_path):
    env_path = tmp_path / ".env"
    printed = []

    def fake_input(prompt=""):
        return ""  # accept every default

    def fake_print(*args, **kwargs):
        printed.append(" ".join(str(a) for a in args))

    setup_env.run_setup(
        input_func=fake_input,
        print_func=fake_print,
        env_path=env_path,
        check_func=fake_check,
    )

    contents = env_path.read_text(encoding="utf-8")
    assert f"APP_OLLAMA_MODEL={setup_env.DEFAULT_MODEL}" in contents
    assert f"APP_OLLAMA_HOST={setup_env.DEFAULT_HOST}" in contents

    # No credential of any kind is written or printed.
    assert "API_KEY" not in contents
    assert "API_KEY" not in "\n".join(printed)


def test_run_setup_reports_a_failed_check_without_raising(tmp_path):
    printed = []

    def failing_check(model, host):
        return "  WARNING: could not reach Ollama"

    setup_env.run_setup(
        input_func=lambda _: "",
        print_func=lambda *a, **k: printed.append(" ".join(str(x) for x in a)),
        env_path=tmp_path / ".env",
        check_func=failing_check,
    )
    assert any("WARNING" in line for line in printed)


def test_check_ollama_never_raises_on_bad_host():
    line = setup_env.check_ollama("llama3:8b", "http://127.0.0.1:1")
    assert "WARNING" in line


def test_prompt_model_uses_default_when_blank():
    assert setup_env.prompt_model(input_func=lambda _: "") == setup_env.DEFAULT_MODEL


def test_prompt_model_accepts_override():
    assert setup_env.prompt_model(input_func=lambda _: "custom-model") == "custom-model"


def test_prompt_host_uses_default_when_blank():
    assert setup_env.prompt_host(input_func=lambda _: "") == setup_env.DEFAULT_HOST

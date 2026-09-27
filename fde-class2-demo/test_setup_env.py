import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import setup_env


def test_write_env_file_writes_expected_contents(tmp_path):
    env_path = tmp_path / ".env"
    setup_env.write_env_file(env_path, "sk-test-secret-value", "claude-haiku-4-5-20251001")

    contents = env_path.read_text(encoding="utf-8")
    assert "APP_ANTHROPIC_API_KEY=sk-test-secret-value" in contents
    assert "APP_ANTHROPIC_MODEL=claude-haiku-4-5-20251001" in contents


def test_write_env_file_never_prints_the_key(tmp_path, capsys):
    env_path = tmp_path / ".env"
    setup_env.write_env_file(env_path, "sk-super-secret-value", "claude-haiku-4-5-20251001")

    captured = capsys.readouterr()
    assert "sk-super-secret-value" not in captured.out
    assert "sk-super-secret-value" not in captured.err


def test_write_env_file_uses_default_model_when_blank(tmp_path):
    env_path = tmp_path / ".env"
    setup_env.write_env_file(env_path, "sk-test", "")

    contents = env_path.read_text(encoding="utf-8")
    assert f"APP_ANTHROPIC_MODEL={setup_env.DEFAULT_MODEL}" in contents

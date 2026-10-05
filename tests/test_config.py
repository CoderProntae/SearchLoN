from pathlib import Path

import pytest

from deepsearch.config import DEFAULT_OLLAMA_URL, AppConfig, ConfigError, load_config, save_config


def test_default_config_uses_requested_local_ollama_port() -> None:
    assert AppConfig().ollama_url == "http://localhost:11435"


def test_config_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "settings.json"
    expected = AppConfig(search_depth=4, web_search_enabled=False)

    save_config(expected, path)

    assert load_config(path) == expected
    assert path.exists()


def test_config_rejects_invalid_network_and_limits() -> None:
    with pytest.raises(ConfigError):
        AppConfig(ollama_url="file:///tmp/ollama").validate()
    with pytest.raises(ConfigError):
        AppConfig(search_depth=0).validate()
    with pytest.raises(ConfigError):
        AppConfig(ollama_url="http://user:pass@localhost:11435").validate()


def test_load_missing_settings_returns_defaults(tmp_path: Path) -> None:
    assert load_config(tmp_path / "missing.json").ollama_url == DEFAULT_OLLAMA_URL

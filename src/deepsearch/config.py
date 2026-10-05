"""Application settings and per-user paths."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

APP_DIRECTORY_NAME = "DeepSearchResearchAI"
DEFAULT_OLLAMA_URL = "http://localhost:11435"
DEFAULT_SEARCH_ENDPOINT = "https://html.duckduckgo.com/html/"


class ConfigError(ValueError):
    """Raised when persisted settings are invalid or cannot be read."""


def get_data_dir() -> Path:
    """Return a writable per-user data directory, with a test override."""
    override = os.environ.get("DEEPSEARCH_DATA_DIR")
    if override:
        return Path(override).expanduser()

    if os.name == "nt":
        root = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        base = Path(root) if root else Path.home() / "AppData" / "Local"
        return base / APP_DIRECTORY_NAME
    if os.uname().sysname == "Darwin":
        return Path.home() / "Library" / "Application Support" / APP_DIRECTORY_NAME
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "deepsearch-research-ai"


def _validate_http_url(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{label} URL metin olmalıdır.")
    value = value.strip().rstrip("/") if label == "Ollama" else value.strip()
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ConfigError(f"{label} adresi http:// veya https:// ile başlamalıdır.")
    if parts.username or parts.password:
        raise ConfigError(f"{label} adresine kullanıcı adı/şifre eklemeyin.")
    if any(char.isspace() for char in value):
        raise ConfigError(f"{label} adresi boşluk içeremez.")
    return value


@dataclass(frozen=True)
class AppConfig:
    ollama_url: str = DEFAULT_OLLAMA_URL
    web_search_enabled: bool = True
    search_endpoint: str = DEFAULT_SEARCH_ENDPOINT
    search_timeout_seconds: int = 12
    search_result_count: int = 6
    search_depth: int = 2
    request_timeout_seconds: int = 180
    max_output_tokens: int = 2048
    retry_count: int = 2

    def validate(self) -> AppConfig:
        _validate_http_url(self.ollama_url, "Ollama")
        _validate_http_url(self.search_endpoint, "Arama motoru")
        integer_ranges = (
            (self.search_timeout_seconds, 3, 60, "Arama zaman aşımı 3–60 saniye arasında olmalıdır."),
            (self.search_result_count, 1, 12, "Arama sonucu sayısı 1–12 arasında olmalıdır."),
            (self.search_depth, 1, 5, "Araştırma derinliği 1–5 arasında olmalıdır."),
            (self.request_timeout_seconds, 30, 900, "Ollama zaman aşımı 30–900 saniye arasında olmalıdır."),
            (self.max_output_tokens, 64, 8192, "Yanıt token üst sınırı 64–8192 arasında olmalıdır."),
            (self.retry_count, 0, 5, "Yeniden deneme sayısı 0–5 arasında olmalıdır."),
        )
        for value, minimum, maximum, message in integer_ranges:
            if type(value) is not int or not minimum <= value <= maximum:
                raise ConfigError(message)
        if not isinstance(self.web_search_enabled, bool):
            raise ConfigError("Web arama ayarı true/false olmalıdır.")
        return self

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AppConfig:
        allowed = set(cls.__dataclass_fields__)
        clean = {key: val for key, val in value.items() if key in allowed}
        try:
            result = cls(**clean)
        except TypeError as exc:
            raise ConfigError(f"Ayar dosyasındaki alanlar geçersiz: {exc}") from exc
        return result.validate()


def load_config(path: Path | None = None) -> AppConfig:
    path = path or get_data_dir() / "settings.json"
    if not path.exists():
        return AppConfig()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ConfigError("Ayar dosyasının kökü JSON nesnesi olmalıdır.")
        return AppConfig.from_dict(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Ayar dosyası okunamadı: {exc}") from exc


def save_config(config: AppConfig, path: Path | None = None) -> Path:
    config.validate()
    path = path or get_data_dir() / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(asdict(config), ensure_ascii=False, indent=2) + "\n"
    _atomic_write(path, content.encode("utf-8"), mode=0o600)
    return path


def _atomic_write(path: Path, content: bytes, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        if os.name != "nt":
            os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
        if os.name != "nt":
            path.chmod(mode)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        temp_path.unlink(missing_ok=True)
        raise

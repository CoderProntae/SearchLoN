"""Encrypted-at-rest usage limits shared by the chat app and the admin tool.

The Fernet key is created locally and restricted to the current user where the
platform supports POSIX permissions. This protects against casual disclosure or
accidental edits; it is not a remote licensing mechanism or a security boundary
against a person who controls the same OS account and executable.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

from .config import get_data_dir


class LimitStoreError(RuntimeError):
    """Base error for reading and writing limit settings."""


class LimitIntegrityError(LimitStoreError):
    """Encrypted limits are missing, corrupted, or cannot be authenticated."""


@dataclass(frozen=True)
class UsageLimits:
    daily_requests: int = 40
    daily_token_limit: int = 100_000
    daily_search_limit: int = 50
    developer_mode: bool = False

    def validate(self) -> UsageLimits:
        # Positive finite maxima are intentional: zero/unlimited is never allowed.
        bounds = (
            (self.daily_requests, 1, 100_000, "günlük istek"),
            (self.daily_token_limit, 1, 10_000_000, "günlük token"),
            (self.daily_search_limit, 1, 100_000, "günlük arama"),
        )
        for value, minimum, maximum, label in bounds:
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{label} limiti {minimum}–{maximum} arasında tam sayı olmalıdır.")
        if type(self.developer_mode) is not bool:
            raise ValueError("Geliştirici modu true/false olmalıdır.")
        return self

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> UsageLimits:
        try:
            result = cls(
                daily_requests=payload["daily_requests"],
                daily_token_limit=payload["daily_token_limit"],
                daily_search_limit=payload["daily_search_limit"],
                developer_mode=payload.get("developer_mode", False),
            )
        except (KeyError, TypeError) as exc:
            raise LimitStoreError(f"Limit yapılandırması eksik veya geçersiz: {exc}") from exc
        try:
            return result.validate()
        except ValueError as exc:
            raise LimitStoreError(str(exc)) from exc


class LimitStore:
    """Small authenticated-encryption store used by both executable entry points."""

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = directory or get_data_dir()
        self.key_path = self.directory / "limits.key"
        self.data_path = self.directory / "limits.enc"
        self._lock = threading.RLock()

    def load(self) -> UsageLimits:
        with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            key_exists = self.key_path.exists()
            data_exists = self.data_path.exists()
            if not key_exists and data_exists:
                raise LimitIntegrityError(
                    "Şifreli kota dosyası var ancak anahtarı bulunamadı. Güvenlik için uygulama kilitlendi."
                )
            key = self._load_or_create_key()
            if not self.data_path.exists():
                return self.save(UsageLimits(), _key=key)
            try:
                encrypted = self.data_path.read_bytes()
                cleartext = Fernet(key).decrypt(encrypted)
                payload = json.loads(cleartext.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise TypeError("JSON kökü nesne değil.")
                return UsageLimits.from_dict(payload)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, InvalidToken, ValueError, TypeError) as exc:
                raise LimitIntegrityError(
                    "Kota dosyası açılamadı veya değiştirilmiş. Kullanım güvenlik nedeniyle durduruldu."
                ) from exc
            except LimitStoreError as exc:
                raise LimitIntegrityError(
                    "Kota yapılandırması geçersiz. Kullanım güvenlik nedeniyle durduruldu."
                ) from exc

    def save(self, limits: UsageLimits, *, _key: bytes | None = None) -> UsageLimits:
        limits.validate()
        with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            if _key is None and not self.key_path.exists() and self.data_path.exists():
                raise LimitIntegrityError(
                    "Şifreli kota dosyası var ancak anahtarı bulunamadı; kayıt güvenlik nedeniyle durduruldu."
                )
            key = _key or self._load_or_create_key()
            cleartext = json.dumps(asdict(limits), ensure_ascii=False, sort_keys=True).encode("utf-8")
            encrypted = Fernet(key).encrypt(cleartext)
            self._atomic_write(self.data_path, encrypted, 0o600)
            return limits

    def _load_or_create_key(self) -> bytes:
        if self.key_path.exists():
            try:
                key = self.key_path.read_bytes().strip()
                # Validate key format before using it.
                Fernet(key)
                if os.name != "nt":
                    self.key_path.chmod(0o600)
                return key
            except (OSError, ValueError) as exc:
                raise LimitIntegrityError("Yerel kota şifreleme anahtarı okunamadı.") from exc

        key = Fernet.generate_key()
        try:
            fd = os.open(self.key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            # Another application instance initialized the key at the same time.
            return self._load_or_create_key()
        except OSError as exc:
            raise LimitStoreError(f"Kota anahtarı oluşturulamadı: {exc}") from exc
        with os.fdopen(fd, "wb") as stream:
            stream.write(key)
            stream.flush()
            os.fsync(stream.fileno())
        if os.name != "nt":
            self.key_path.chmod(0o600)
        return key

    @staticmethod
    def _atomic_write(path: Path, content: bytes, mode: int) -> None:
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

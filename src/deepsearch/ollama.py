"""Minimal streaming REST client for Ollama's local /api endpoints."""

from __future__ import annotations

import http.client
import json
import time
from collections.abc import Iterator
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaError(RuntimeError):
    """A connection, HTTP, or response-format error from Ollama."""


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11435",
        timeout_seconds: int = 180,
        retry_count: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.retry_count = retry_count

    def list_models(self) -> list[dict[str, Any]]:
        payload = self._get_json("/api/tags")
        models = payload.get("models", [])
        if not isinstance(models, list):
            raise OllamaError("Ollama /api/tags yanıt biçimi beklenenden farklı.")
        clean: list[dict[str, Any]] = []
        for item in models:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                clean.append(item)
        return sorted(clean, key=lambda item: item["name"].casefold())

    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        *,
        tools: list[dict[str, Any]] | None = None,
        num_predict: int = 2048,
    ) -> Iterator[dict[str, Any]]:
        if not model.strip():
            raise OllamaError("Önce Ollama modeli seçin.")
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"num_predict": max(1, int(num_predict))},
        }
        if tools:
            payload["tools"] = tools
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        attempt = 0
        while True:
            saw_chunk = False
            request = Request(
                f"{self.base_url}/api/chat",
                data=body,
                headers={"Content-Type": "application/json", "Accept": "application/x-ndjson"},
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    for raw_line in response:
                        line = raw_line.strip()
                        if not line:
                            continue
                        try:
                            event = json.loads(line.decode("utf-8"))
                        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                            raise OllamaError("Ollama akışından geçersiz JSON verisi geldi.") from exc
                        if not isinstance(event, dict):
                            continue
                        saw_chunk = True
                        if event.get("error"):
                            raise OllamaError(str(event["error"]))
                        yield event
                        if event.get("done") is True:
                            return
                return
            except HTTPError as exc:
                message = _http_error_message(exc)
                retryable = exc.code >= 500 or exc.code in {408, 425, 429}
                if retryable and attempt < self.retry_count and not saw_chunk:
                    attempt += 1
                    time.sleep(min(0.25 * (2**(attempt - 1)), 1.0))
                    continue
                raise OllamaError(f"Ollama HTTP {exc.code}: {message}") from exc
            except (URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
                if attempt < self.retry_count and not saw_chunk:
                    attempt += 1
                    time.sleep(min(0.25 * (2**(attempt - 1)), 1.0))
                    continue
                reason = getattr(exc, "reason", exc)
                raise OllamaError(
                    "Ollama sunucusuna bağlanılamadı veya akış kesildi. "
                    f"Adres: {self.base_url}. Ayrıntı: {reason}"
                ) from exc

    def _get_json(self, path: str) -> dict[str, Any]:
        request = Request(f"{self.base_url}{path}", headers={"Accept": "application/json"})
        attempt = 0
        while True:
            try:
                with urlopen(request, timeout=min(self.timeout_seconds, 20)) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(payload, dict):
                    raise OllamaError("Ollama JSON yanıtı beklenen biçimde değil.")
                return payload
            except HTTPError as exc:
                message = _http_error_message(exc)
                retryable = exc.code >= 500 or exc.code in {408, 425, 429}
                if retryable and attempt < self.retry_count:
                    attempt += 1
                    time.sleep(min(0.25 * (2**(attempt - 1)), 1.0))
                    continue
                raise OllamaError(f"Ollama HTTP {exc.code}: {message}") from exc
            except (URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
                if attempt < self.retry_count:
                    attempt += 1
                    time.sleep(min(0.25 * (2**(attempt - 1)), 1.0))
                    continue
                reason = getattr(exc, "reason", exc)
                raise OllamaError(
                    f"Ollama sunucusuna bağlanılamadı ({self.base_url}): {reason}"
                ) from exc
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise OllamaError("Ollama geçersiz JSON yanıtı döndürdü.") from exc


def _http_error_message(error: HTTPError) -> str:
    try:
        raw = error.read(4096).decode("utf-8", errors="replace")
        payload = json.loads(raw)
        if isinstance(payload, dict) and payload.get("error"):
            return str(payload["error"])
        return raw.strip() or error.reason
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return str(error.reason)


def estimate_tokens(text: str) -> int:
    """Conservative rough estimate for quota preflight (actual counts use Ollama stats)."""
    if not text:
        return 0
    # UTF-8 bytes cope better with Turkish and other non-ASCII scripts than len(text)/4.
    byte_count = len(text.encode("utf-8"))
    return max(1, (byte_count + 2) // 3)

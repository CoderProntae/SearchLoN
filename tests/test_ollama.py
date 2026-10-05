import json
from urllib.error import URLError

from deepsearch import ollama
from deepsearch.ollama import OllamaClient, estimate_tokens


class FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        return iter(self.payload.splitlines(keepends=True))

    def read(self, _count: int = -1) -> bytes:
        return self.payload


def test_streaming_client_yields_ollama_events(monkeypatch) -> None:
    events = [
        {"message": {"role": "assistant", "content": "Hello"}, "done": False},
        {"message": {"role": "assistant", "content": "!"}, "done": True, "eval_count": 2},
    ]
    payload = b"".join(json.dumps(event).encode() + b"\n" for event in events)
    monkeypatch.setattr(ollama, "urlopen", lambda *_args, **_kwargs: FakeResponse(payload))

    actual = list(OllamaClient(retry_count=0).stream_chat([{"role": "user", "content": "Hi"}], "qwen"))

    assert actual == events


def test_client_retries_connection_failure_before_stream(monkeypatch) -> None:
    calls = []

    def fake_open(*_args, **_kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise URLError("temporary")
        payload = json.dumps({"message": {"content": "ok"}, "done": True}).encode() + b"\n"
        return FakeResponse(payload)

    monkeypatch.setattr(ollama, "urlopen", fake_open)
    monkeypatch.setattr(ollama.time, "sleep", lambda _seconds: None)

    events = list(OllamaClient(retry_count=1).stream_chat([], "qwen"))

    assert len(calls) == 2
    assert events[0]["message"]["content"] == "ok"


def test_token_estimate_handles_unicode_and_empty_text() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("Merhaba dünya") > 0


def test_model_list_response(monkeypatch) -> None:
    body = json.dumps({"models": [{"name": "llama3"}, {"name": "qwen3"}]}).encode()
    monkeypatch.setattr(ollama, "urlopen", lambda *_args, **_kwargs: FakeResponse(body))

    names = [model["name"] for model in OllamaClient().list_models()]

    assert names == ["llama3", "qwen3"]

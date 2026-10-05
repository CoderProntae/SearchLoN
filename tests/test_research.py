from pathlib import Path

from deepsearch.config import AppConfig
from deepsearch.limits import UsageLimits
from deepsearch.ollama import OllamaError
from deepsearch.research import ResearchAgent
from deepsearch.storage import DataStore
from deepsearch.web import SearchResult, WebPage


class FakeOllama:
    def __init__(self) -> None:
        self.calls = 0
        self.num_predict_values = []

    def stream_chat(self, _messages, _model, **kwargs):
        self.calls += 1
        self.num_predict_values.append(kwargs["num_predict"])
        if self.calls == 1:
            yield {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "fetch_web_page",
                                "arguments": {"url": "https://93.184.216.34/source"},
                            }
                        }
                    ],
                },
                "done": True,
                "prompt_eval_count": 40,
                "eval_count": 2,
            }
        else:
            yield {
                "message": {"role": "assistant", "content": "Araştırılmış yanıt [1]."},
                "done": True,
                "prompt_eval_count": 55,
                "eval_count": 8,
            }


class FakeUnsupportedOllama:
    def __init__(self) -> None:
        self.calls = 0
        self.tool_options = []

    def stream_chat(self, _messages, _model, **kwargs):
        self.calls += 1
        self.tool_options.append(kwargs.get("tools"))
        if self.calls == 1:
            raise OllamaError("Ollama HTTP 400: model does not support tools")
        yield {
            "message": {"role": "assistant", "content": "Yanıt, arama kaynaklarıyla hazır."},
            "done": True,
            "prompt_eval_count": 55,
            "eval_count": 8,
        }


class FakeWeb:
    def __init__(self) -> None:
        self.searches = []
        self.fetches = []

    def search(self, query, result_count):
        self.searches.append((query, result_count))
        return [SearchResult("Example source", "https://example.org/source", "Search excerpt")]

    def fetch_page(self, url, max_chars=12_000):
        self.fetches.append((url, max_chars))
        return WebPage("Example source", url, "Readable source content.")


def _respond(agent: ResearchAgent, store: DataStore, session: str, events, deltas, resets, usage):
    return agent.respond(
        history=store.get_messages(session),
        model="qwen3",
        config=AppConfig(search_depth=1),
        limits=UsageLimits(),
        store=store,
        usage_day="2026-10-04",
        num_predict=512,
        remaining_token_budget=10_000,
        on_delta=deltas.append,
        on_reset=lambda: resets.append(True),
        on_event=events.append,
        on_usage=lambda incoming, outgoing: usage.append((incoming, outgoing)),
    )


def _session(tmp_path: Path):
    store = DataStore(tmp_path / "history.sqlite3")
    session = store.create_session()
    store.save_message(session, "user", "Find the current release and sources")
    return store, session


def test_agent_searches_reads_page_and_handles_model_tool_calls(tmp_path: Path) -> None:
    store, session = _session(tmp_path)
    ollama = FakeOllama()
    web = FakeWeb()
    agent = ResearchAgent(ollama, web)
    events, deltas, resets, usage = [], [], [], []

    outcome = _respond(agent, store, session, events, deltas, resets, usage)

    assert outcome.answer == "Araştırılmış yanıt [1]."
    assert (outcome.input_tokens, outcome.output_tokens) == (95, 10)
    assert len(web.searches) == 1
    assert web.fetches == [
        ("https://example.org/source", 4_500),
        ("https://93.184.216.34/source", 8_000),
    ]
    assert ollama.calls == 2
    assert resets == [True]
    assert deltas == ["Araştırılmış yanıt [1]."]
    assert usage == [(40, 2), (55, 8)]
    assert store.get_usage("2026-10-04").searches == 1
    assert any(event["kind"] == "fetched" for event in events)


def test_agent_falls_back_for_models_without_tool_support(tmp_path: Path) -> None:
    store, session = _session(tmp_path)
    ollama = FakeUnsupportedOllama()
    web = FakeWeb()
    events, deltas, resets, usage = [], [], [], []

    outcome = _respond(ResearchAgent(ollama, web), store, session, events, deltas, resets, usage)

    assert outcome.answer == "Yanıt, arama kaynaklarıyla hazır."
    assert ollama.calls == 2
    assert ollama.tool_options[0]
    assert ollama.tool_options[1] is None
    assert len(web.searches) == 1
    assert any("desteklemiyor" in event.get("title", "") for event in events)

from pathlib import Path

from deepsearch.limits import UsageLimits
from deepsearch.storage import DataStore


def test_history_persists_and_uses_first_user_message_as_title(tmp_path: Path) -> None:
    store = DataStore(tmp_path / "state.sqlite3")
    session = store.create_session()
    store.save_message(session, "user", "  Compare Qwen and Gemini today  ")
    store.save_message(session, "assistant", "## Comparison")

    assert [message["role"] for message in store.get_messages(session)] == ["user", "assistant"]
    assert store.list_sessions()[0]["title"] == "Compare Qwen and Gemini today"


def test_quota_reservations_and_daily_metrics(tmp_path: Path) -> None:
    store = DataStore(tmp_path / "state.sqlite3")
    limits = UsageLimits(daily_requests=2, daily_token_limit=1000, daily_search_limit=1)

    request = store.reserve_request(limits, estimated_input_tokens=50)
    assert request.allowed
    assert store.reserve_search(limits, request.day).allowed
    assert not store.reserve_search(limits, request.day).allowed

    store.record_tokens(80, 20, request.day)
    store.record_active_seconds(61.2, request.day)
    usage = store.get_usage(request.day)
    assert usage.requests == 1
    assert usage.total_tokens == 100
    assert usage.searches == 1
    assert usage.active_seconds == 61

    # A full search quota locks new requests rather than allowing unlimited chat.
    blocked = store.reserve_request(limits, estimated_input_tokens=20)
    assert not blocked.allowed
    assert "arama" in blocked.reason

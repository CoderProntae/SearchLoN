from pathlib import Path

import pytest

from deepsearch.limits import LimitIntegrityError, LimitStore, UsageLimits


def test_encrypted_limit_round_trip_and_finite_limits(tmp_path: Path) -> None:
    store = LimitStore(tmp_path)
    defaults = store.load()

    assert defaults == UsageLimits()
    assert store.data_path.exists()
    assert b"daily_requests" not in store.data_path.read_bytes()

    edited = UsageLimits(17, 45_000, 23, True)
    store.save(edited)
    assert store.load() == edited


def test_tampering_is_detected(tmp_path: Path) -> None:
    store = LimitStore(tmp_path)
    store.load()
    store.data_path.write_bytes(b"not authenticated ciphertext")

    with pytest.raises(LimitIntegrityError):
        store.load()


def test_lost_key_fails_closed(tmp_path: Path) -> None:
    store = LimitStore(tmp_path)
    store.load()
    store.key_path.unlink()

    with pytest.raises(LimitIntegrityError):
        store.load()
    with pytest.raises(LimitIntegrityError):
        store.save(UsageLimits())


def test_zero_and_unlimited_are_rejected() -> None:
    with pytest.raises(ValueError):
        UsageLimits(daily_requests=0).validate()
    with pytest.raises(ValueError):
        UsageLimits(daily_token_limit=10_000_001).validate()
    with pytest.raises(ValueError):
        UsageLimits(developer_mode=1).validate()

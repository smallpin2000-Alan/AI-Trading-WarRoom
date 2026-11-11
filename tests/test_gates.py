# tests/test_gates.py
"""管理密碼保護與五道可信度檢核閘門的單元測試"""
from datetime import datetime, timedelta

import polars as pl

from src.core.credibility_gate import MIN_BARS_FOR_CREDIBILITY, run_credibility_gate
from src.core.engine import ContractSpec
from src.schema.strategy import StrategySchema
from src.ui.admin_gate import FailedAttemptThrottle, verify_admin_password


def _strategy() -> StrategySchema:
    return StrategySchema.model_validate({
        "id": "gate_test_v1",
        "version": 1,
        "symbol": "TMF",
        "timeframe": "5m",
        "session": "day_and_night",
        "indicators": {
            "ma5": {"type": "SMA", "source": "close", "length": 5},
            "ma20": {"type": "SMA", "source": "close", "length": 20},
        },
        "entry_long": {"cross_above": ["ma5", "ma20"]},
        "entry_short": {"cross_below": ["ma5", "ma20"]},
        "exit": {
            "stop_loss": {"type": "fixed_points", "value": 45.0},
            "take_profit": {"type": "fixed_points", "value": 60.0},
            "trailing_stop": {"type": "points", "value": 15.0, "activation_points": 35.0},
        },
    })


def _contract() -> ContractSpec:
    return ContractSpec(
        symbol="TMF", name="微型臺指", exchange="TAIFEX", point_value=10.0, tick_size=1.0,
        tax_rate=0.00002, default_commission=12.0, initial_margin=16100.0, maintenance_margin=12400.0,
    )


def _flat_noise_bars(n: int) -> pl.DataFrame:
    """無趨勢的鋸齒盤：均線交叉策略在此應無法通過檢核"""
    start = datetime(2026, 11, 2, 9, 0)
    closes = [20000.0 + (15.0 if (i // 3) % 2 == 0 else -15.0) for i in range(n)]
    return pl.DataFrame({
        "timestamp": [start + timedelta(minutes=5 * i) for i in range(n)],
        "open": closes,
        "high": [c + 8 for c in closes],
        "low": [c - 8 for c in closes],
        "close": closes,
        "volume": [100] * n,
    })


def test_gate_rejects_when_bars_insufficient():
    res = run_credibility_gate(_strategy(), _contract(), 100_000.0, _flat_noise_bars(50))
    assert res.passed is False
    assert "不足" in res.reasons[0]


def test_gate_never_passes_by_default_on_noise():
    res = run_credibility_gate(_strategy(), _contract(), 100_000.0, _flat_noise_bars(MIN_BARS_FOR_CREDIBILITY + 100))
    assert res.passed is False
    assert res.reasons, "未通過時必須說明原因"


def test_gate_treats_validator_error_as_failure():
    bad_df = _flat_noise_bars(MIN_BARS_FOR_CREDIBILITY + 10).drop("close")
    res = run_credibility_gate(_strategy(), _contract(), 100_000.0, bad_df)
    assert res.passed is False


def test_verify_admin_password():
    assert verify_admin_password("s3cret", "s3cret")
    assert not verify_admin_password("wrong", "s3cret")
    assert not verify_admin_password("", ""), "未設定管理密碼時不可用空字串解鎖"
    assert not verify_admin_password(None, "s3cret")


def test_failed_attempt_throttle_locks_and_recovers():
    now = [1000.0]
    throttle = FailedAttemptThrottle(max_attempts=3, window_sec=60.0, clock=lambda: now[0])
    for _ in range(3):
        assert not throttle.is_locked()
        throttle.record_failure()
    assert throttle.is_locked()
    assert throttle.seconds_until_unlock() == 60
    now[0] += 61.0
    assert not throttle.is_locked()

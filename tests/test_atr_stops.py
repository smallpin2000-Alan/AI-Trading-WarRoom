# tests/test_atr_stops.py
"""停損改用 ATR 倍數：即時路徑必須能算出 ATR，預設策略與舊存檔轉換皆以 ATR 為準"""
from datetime import datetime, timedelta

import polars as pl
import pytest

from src.core.engine import ContractSpec
from src.core.exits import compute_exit_levels
from src.core.paper_trader import (
    DEFAULT_ATR_STOP_MULT,
    PaperTradingEngine,
    RiskConfig,
    default_ai_strategy,
    ensure_atr_exits,
)
from src.schema.strategy import StrategySchema

BAR_RANGE = 100.0  # 每根 K 棒高低差固定 100 點 → ATR(14) ≈ 100


def _contract() -> ContractSpec:
    return ContractSpec(
        symbol="TMF", name="微型臺指", exchange="TAIFEX", point_value=10.0, tick_size=1.0,
        tax_rate=0.00002, default_commission=12.0, initial_margin=16100.0, maintenance_margin=12400.0,
    )


def _bars(n: int = 40, base: float = 40000.0) -> pl.DataFrame:
    start = datetime(2026, 11, 10, 9, 0)
    closes = [base] * n
    return pl.DataFrame({
        "timestamp": [start + timedelta(minutes=5 * i) for i in range(n)],
        "open": closes,
        "high": [c + BAR_RANGE / 2 for c in closes],
        "low": [c - BAR_RANGE / 2 for c in closes],
        "close": closes,
        "volume": [100] * n,
    })


def _engine() -> PaperTradingEngine:
    eng = PaperTradingEngine(
        default_ai_strategy("TMF"), _contract(), initial_capital=100_000.0,
        risk_config=RiskConfig(max_daily_loss=50_000.0, max_contracts=1),
    )
    ok, _ = eng.start_session(credibility_passed=True)
    assert ok
    return eng


def test_default_strategy_uses_atr_exits():
    ex = default_ai_strategy("TMF").exit
    assert ex.stop_loss.type == "atr_multiple"
    assert ex.take_profit.type == "atr_multiple"
    assert ex.trailing_stop.type == "atr_multiple"
    assert ex.trailing_stop.activation_atr_multiple is not None
    # 停利距離必須大於停損距離（損益比 > 1）
    assert ex.take_profit.value > ex.stop_loss.value


def test_ai_path_opens_with_atr_stop_from_raw_bars():
    """AI 即時路徑傳入的是沒有 ATR 欄位的原始 K 棒，仍必須能算出 ATR 停損並開倉"""
    eng = _engine()
    eng._last_reset_ts = 0.0
    res = eng.sync_with_ai_advice("BUY_LONG", 90.0, 40000.0, _bars())
    assert eng.position is not None, res
    dist = eng.position.entry_price - eng.position.stop_loss_price
    assert dist == pytest.approx(DEFAULT_ATR_STOP_MULT * BAR_RANGE, rel=0.05)


def test_atr_stop_not_hit_by_normal_noise_but_hit_beyond():
    eng = _engine()
    eng._last_reset_ts = 0.0
    eng.sync_with_ai_advice("BUY_LONG", 90.0, 40000.0, _bars())
    entry = eng.position.entry_price
    # 跌 60 點（小於一根 K 棒的正常振幅）不可被停損
    assert eng.check_position_stops(entry - 60.0, _bars()) is None
    assert eng.position is not None
    # 跌超過 1.5 倍 ATR 必須停損
    eng.check_position_stops(entry - DEFAULT_ATR_STOP_MULT * BAR_RANGE - 5.0, _bars())
    assert eng.position is None


def test_trailing_activation_by_atr_multiple():
    strat = default_ai_strategy("TMF")
    ts = strat.exit.trailing_stop
    atr = {ts.atr_length: BAR_RANGE}
    entry = 40000.0
    act = ts.activation_atr_multiple * BAR_RANGE
    # 獲利未達啟動門檻：停損仍為初始 ATR 停損
    sl, _, reason = compute_exit_levels(strat, 1, entry, entry + act - 1, entry, atr, atr)
    assert reason == "stop_loss"
    # 達到門檻後改為移動停損：最高價 − 1 倍 ATR
    high = entry + act + 50
    sl, _, reason = compute_exit_levels(strat, 1, entry, high, entry, atr, atr)
    assert reason == "trailing_stop"
    assert sl == pytest.approx(high - ts.value * BAR_RANGE)


def test_ensure_atr_exits_converts_legacy_fixed_points_strategy():
    legacy = StrategySchema.model_validate({
        "id": "ai_copilot_auto", "version": 1, "symbol": "TMF", "timeframe": "5m", "session": "day_and_night",
        "indicators": {"ma5": {"type": "SMA", "source": "close", "length": 5},
                       "ma20": {"type": "SMA", "source": "close", "length": 20}},
        "entry_long": {"cross_above": ["ma5", "ma20"]},
        "entry_short": {"cross_below": ["ma5", "ma20"]},
        "exit": {"stop_loss": {"type": "fixed_points", "value": 45.0},
                 "take_profit": {"type": "fixed_points", "value": 60.0},
                 "trailing_stop": {"type": "points", "value": 15.0, "activation_points": 35.0}},
    })
    converted, changed = ensure_atr_exits(legacy)
    assert changed
    assert converted.exit.stop_loss.type == "atr_multiple"
    assert legacy.exit.stop_loss.type == "fixed_points", "不可修改原物件"

    again, changed_again = ensure_atr_exits(converted)
    assert not changed_again
    assert again.exit == converted.exit

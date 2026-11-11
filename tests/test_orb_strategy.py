# tests/test_orb_strategy.py
"""開盤區間突破（ORB）規則：1 分 K 時間標在分鐘結束（08:46 代表 08:45:00~08:45:59）"""
from datetime import datetime, timedelta

import polars as pl
import pytest

from src.core.orb_strategy import OrbParams, backtest_orb, compute_orb_state

DAY = datetime(2026, 11, 10)


def _bars(path: list[float], start: datetime = DAY.replace(hour=8, minute=46), spread: float = 2.0) -> pl.DataFrame:
    """以收盤價序列產生連續 1 分 K（每根高低 = 收盤 ± spread）"""
    ts = [start + timedelta(minutes=i) for i in range(len(path))]
    return pl.DataFrame({
        "timestamp": ts,
        "open": path,
        "high": [p + spread for p in path],
        "low": [p - spread for p in path],
        "close": path,
        "volume": [10] * len(path),
    })


def _day_path(after_range: list[float]) -> list[float]:
    """前 30 根（08:46~09:15）在 40000 ± 20 內震盪，之後接 after_range"""
    opening = [40000 + (20 if i % 2 else -20) for i in range(30)]
    return opening + after_range


def test_no_signal_before_range_complete():
    st = compute_orb_state(_bars(_day_path([])[:20]), now=DAY.replace(hour=9, minute=5))
    assert not st.range_ready
    assert st.action == "WAIT"


def test_range_levels_use_first_30_minutes():
    st = compute_orb_state(_bars(_day_path([40000] * 5)), now=DAY.replace(hour=9, minute=20))
    assert st.range_ready
    assert st.range_high == 40022 and st.range_low == 39978
    assert st.action == "WAIT"


def test_long_breakout_on_close_above_range():
    st = compute_orb_state(_bars(_day_path([40010, 40030, 40040])), now=DAY.replace(hour=9, minute=19))
    assert st.action == "BUY_LONG"
    assert st.breakout_price == 40030


def test_short_breakout_on_close_below_range():
    st = compute_orb_state(_bars(_day_path([39990, 39970])), now=DAY.replace(hour=9, minute=18))
    assert st.action == "SELL_SHORT"


def test_only_first_breakout_counts():
    """先突破上緣後又跌破下緣：當日方向仍以第一次突破為準"""
    st = compute_orb_state(_bars(_day_path([40030, 40000, 39960])), now=DAY.replace(hour=9, minute=19))
    assert st.action == "BUY_LONG"


def test_breakout_after_last_entry_time_is_ignored():
    flat = [40000] * (int((12 * 60 + 40 - (9 * 60 + 15))) )  # 09:16 ~ 12:40 都在區間內
    bars = _bars(_day_path(flat + [40100]))
    st = compute_orb_state(bars, now=DAY.replace(hour=12, minute=42))
    assert st.action == "WAIT"


def test_flatten_flag_at_close_and_outside_day_session():
    bars = _bars(_day_path([40030]))
    assert compute_orb_state(bars, now=DAY.replace(hour=13, minute=30)).flatten
    assert not compute_orb_state(bars, now=DAY.replace(hour=10, minute=0)).flatten
    assert compute_orb_state(bars, now=DAY.replace(hour=20, minute=0)).flatten  # 夜盤不持倉


def test_ignores_previous_days_bars():
    prev = _bars([45000] * 30, start=(DAY - timedelta(days=1)).replace(hour=8, minute=46))
    today = _bars(_day_path([40030]))
    st = compute_orb_state(pl.concat([prev, today]), now=DAY.replace(hour=9, minute=17))
    assert st.range_high == 40022
    assert st.action == "BUY_LONG"


def test_backtest_one_trade_per_day_and_flat_by_close():
    up_day = _day_path([40030 + 3 * i for i in range(240)])     # 突破後一路上漲
    res = backtest_orb(_bars(up_day), OrbParams(), atr_value=50.0)
    assert len(res) == 1
    tr = res[0]
    assert tr["direction"] == 1
    assert tr["exit_time"] <= DAY.replace(hour=13, minute=30)
    assert tr["pnl_points"] > 0


def test_backtest_stop_loss_hit():
    down_after_long = _day_path([40030] + [40030 - 10 * i for i in range(1, 60)])
    res = backtest_orb(_bars(down_after_long), OrbParams(), atr_value=50.0)
    assert len(res) == 1 and res[0]["exit_reason"] == "stop_loss"
    assert res[0]["pnl_points"] == pytest.approx(-1.5 * 50.0 - OrbParams().cost_points, abs=12)


# ---------------------------------------------------------------------------
# 模擬倉依規則訊號進出場
# ---------------------------------------------------------------------------
from src.core import paper_trader as pt_module  # noqa: E402
from src.core.engine import ContractSpec  # noqa: E402
from src.core.paper_trader import PaperTradingEngine, RiskConfig, default_ai_strategy  # noqa: E402


def _engine(monkeypatch, day="2026-11-10"):
    monkeypatch.setattr(pt_module, "current_trading_day", lambda: day)
    c = ContractSpec(symbol="TMF", name="微型臺指", exchange="TAIFEX", point_value=10.0, tick_size=1.0,
                     tax_rate=0.00002, default_commission=12.0, initial_margin=16100.0, maintenance_margin=12400.0)
    eng = PaperTradingEngine(default_ai_strategy("TMF"), c, initial_capital=100_000.0,
                             risk_config=RiskConfig(max_daily_loss=50_000.0, max_contracts=1))
    assert eng.start_session(credibility_passed=True)[0]
    eng._last_reset_ts = 0.0
    return eng


def _m5(n=40, base=40000.0):
    start = DAY.replace(hour=8, minute=45) - timedelta(minutes=5 * n)
    return pl.DataFrame({
        "timestamp": [start + timedelta(minutes=5 * i) for i in range(n)],
        "open": [base] * n, "high": [base + 25] * n, "low": [base - 25] * n,
        "close": [base] * n, "volume": [10] * n,
    })


def test_rule_signal_opens_once_per_day(monkeypatch):
    eng = _engine(monkeypatch)
    res = eng.sync_with_rule_signal("BUY_LONG", 40030.0, _m5(), flatten=False)
    assert eng.position is not None and eng.position.direction == 1, res
    eng._close_current_position(40030.0, exit_idx=-1, reason="test")
    # 同一交易日：即使訊號仍為做多也不再進場
    assert eng.sync_with_rule_signal("BUY_LONG", 40030.0, _m5(), flatten=False) is None
    assert eng.position is None
    # 換交易日後可以再進場
    monkeypatch.setattr(pt_module, "current_trading_day", lambda: "2026-11-11")
    eng.sync_with_rule_signal("BUY_LONG", 40030.0, _m5(), flatten=False)
    assert eng.position is not None


def test_rule_signal_never_reverses(monkeypatch):
    eng = _engine(monkeypatch)
    eng.sync_with_rule_signal("BUY_LONG", 40030.0, _m5(), flatten=False)
    eng.sync_with_rule_signal("SELL_SHORT", 40020.0, _m5(), flatten=False)
    assert eng.position is not None and eng.position.direction == 1
    assert len(eng.closed_trades) == 0


def test_rule_signal_flatten_closes_position(monkeypatch):
    eng = _engine(monkeypatch)
    eng.sync_with_rule_signal("BUY_LONG", 40030.0, _m5(), flatten=False)
    res = eng.sync_with_rule_signal("WAIT", 40040.0, _m5(), flatten=True)
    assert eng.position is None
    assert eng.closed_trades[-1].exit_reason == "orb_time_exit"
    assert "平倉" in res


def test_rule_signal_flatten_even_when_paused(monkeypatch):
    eng = _engine(monkeypatch)
    eng.sync_with_rule_signal("BUY_LONG", 40030.0, _m5(), flatten=False)
    eng.is_active = False
    eng.sync_with_rule_signal("WAIT", 40040.0, _m5(), flatten=True)
    assert eng.position is None


def test_last_entry_day_persisted(monkeypatch, tmp_path):
    eng = _engine(monkeypatch)
    eng.sync_with_rule_signal("BUY_LONG", 40030.0, _m5(), flatten=False)
    eng.save_state_to_disk(tmp_path / "s.json")
    restored = _engine(monkeypatch)
    restored.load_state_from_disk(tmp_path / "s.json")
    restored.position = None
    assert restored.sync_with_rule_signal("BUY_LONG", 40030.0, _m5(), flatten=False) is None

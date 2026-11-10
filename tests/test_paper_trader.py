# tests/test_paper_trader.py
from datetime import datetime, timedelta
import polars as pl
import pytest

from src.schema.strategy import StrategySchema
from src.core.engine import ContractSpec
from src.core.paper_trader import PaperTradingEngine, RiskConfig


def _make_sample_strategy() -> StrategySchema:
    return StrategySchema.model_validate({
        "id": "paper_test_v1",
        "version": 1,
        "symbol": "TMF",
        "timeframe": "1m",
        "session": "day_and_night",
        "indicators": {
            "fast_ma": {"type": "SMA", "source": "close", "length": 2},
            "slow_ma": {"type": "SMA", "source": "close", "length": 3},
        },
        "entry_long": {"cross_above": ["fast_ma", "slow_ma"]},
        "entry_short": {"cross_below": ["fast_ma", "slow_ma"]},
        "exit": {
            "stop_loss": {"type": "fixed_points", "value": 30},
            "take_profit": {"type": "fixed_points", "value": 60},
            "trailing_stop": None,
        },
        "execution": {"signal_on": "bar_close", "fill_at": "next_bar_open", "order_type": "range_market", "slippage_points": 1.0},
        "position_sizing": {"type": "fixed", "contracts": 1},
    })


def _make_contract() -> ContractSpec:
    return ContractSpec(
        symbol="TMF",
        name="微型臺指",
        exchange="TAIFEX",
        point_value=10.0,
        tick_size=1.0,
        tax_rate=0.00002,
        default_commission=12.0,
        initial_margin=16100.0,
        maintenance_margin=12400.0,
    )


def test_paper_trading_blocks_unverified_strategy():
    """測試未通過五道可信度驗證時，嚴格阻擋啟動紙上交易"""
    strat = _make_sample_strategy()
    spec = _make_contract()
    pt = PaperTradingEngine(strat, spec, initial_capital=100_000.0)

    ok, msg = pt.start_session(credibility_passed=False)
    assert not ok
    assert not pt.is_active
    assert "未通過五道可信度驗證" in msg


def test_paper_trading_signal_fill_and_kill_switch():
    """測試紙上交易通過驗證啟動、收盤訊號次根開盤成交與 Kill Switch 一鍵平倉"""
    strat = _make_sample_strategy()
    spec = _make_contract()
    pt = PaperTradingEngine(strat, spec, initial_capital=100_000.0)

    ok, _ = pt.start_session(credibility_passed=True)
    assert ok
    assert pt.is_active

    t0 = datetime(2026, 10, 22, 9, 0)
    # 前 3 根讓 fast_ma <= slow_ma，第 4 根收盤大漲使 fast_ma 上穿 slow_ma
    closes = [20000.0, 19990.0, 19980.0, 20050.0]
    df1 = pl.DataFrame({
        "timestamp": [t0 + timedelta(minutes=i) for i in range(4)],
        "open": closes,
        "high": [p + 5 for p in closes],
        "low": [p - 5 for p in closes],
        "close": closes,
        "volume": [100] * 4,
    })
    pt.sync_with_market(df1, latest_price=20050.0)
    # 收盤產生多單訊號，排入 pending_signal
    assert pt.pending_signal is not None
    assert pt.pending_signal[0] == 1
    assert pt.position is None

    # 第 5 根新 K 棒到達，於開盤價 20055.0 (+1 點滑價 = 20056.0) 成交建倉
    closes.append(20060.0)
    opens = [20000.0, 19990.0, 19980.0, 20050.0, 20055.0]
    df2 = pl.DataFrame({
        "timestamp": [t0 + timedelta(minutes=i) for i in range(5)],
        "open": opens,
        "high": [p + 10 for p in opens],
        "low": [p - 5 for p in opens],
        "close": closes,
        "volume": [100] * 5,
    })
    pt.sync_with_market(df2, latest_price=20060.0)
    assert pt.position is not None
    assert pt.position.direction == 1
    assert pt.position.entry_price == 20056.0

    # 觸發一鍵緊急平倉 Kill Switch
    ks_msg = pt.trigger_kill_switch(current_price=20070.0)
    assert "Kill Switch" in ks_msg
    assert pt.position is None
    assert pt.kill_switch_locked
    assert not pt.is_active
    assert len(pt.closed_trades) == 1


def test_paper_trading_daily_max_loss_lock():
    """測試單日最大虧損鎖定：當未實現虧損超過上限時自動強制平倉並鎖定"""
    strat = _make_sample_strategy()
    spec = _make_contract()
    risk_cfg = RiskConfig(max_daily_loss=500.0, max_contracts=1)
    pt = PaperTradingEngine(strat, spec, initial_capital=100_000.0, risk_config=risk_cfg)
    pt.start_session(credibility_passed=True)

    # 直接建立一筆多單部位 @ 20000
    pt._open_new_position(direction=1, raw_price=19999.0, bar_time_str="2026-10-22 09:05", reason="entry_long")
    assert pt.position is not None
    assert pt.position.entry_price == 20000.0

    # 價格急跌 60 點 -> 未實現虧損 = -60 * 10 = -600 元 <= -500 元上限
    locked = pt.check_daily_loss_limit(current_price=19940.0)
    assert locked
    assert pt.daily_loss_locked
    assert not pt.is_active
    assert pt.position is None
    assert len(pt.closed_trades) == 1

    # 測試全部模擬重置：清空所有歷史成交、風控鎖定，本金恢復 100,000 元 (save_to_disk=False 避免單元測試污染正式存檔)
    pt.reset_all_simulation(initial_capital=100_000.0, save_to_disk=False)
    assert pt.initial_capital == 100_000.0
    assert pt.realized_pnl == 0.0
    assert pt.get_total_equity() == 100_000.0
    assert pt.position is None
    assert len(pt.closed_trades) == 0
    assert not pt.daily_loss_locked
    assert not pt.kill_switch_locked
    assert not pt.order_safety_locked

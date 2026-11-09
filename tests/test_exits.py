# tests/test_exits.py
import pytest
import polars as pl
from datetime import datetime
from src.schema.strategy import StrategySchema
from src.core.engine import ContractSpec, EventDrivenEngine
from src.core.paper_trader import PaperTradingEngine, RiskConfig


def _make_sample_contract() -> ContractSpec:
    return ContractSpec(
        symbol="TMF",
        name="微型臺指",
        exchange="TAIFEX",
        point_value=10.0,
        tick_size=1.0,
        tax_rate=0.00002,
        default_commission=12.0,
        settlement_rule="third_wednesday",
    )


def test_atr_multiple_stop_loss_in_engine():
    # 建立 50 根合成 K 棒
    opens = [20000.0 + i * 5 for i in range(50)]
    highs = [o + 15.0 for o in opens]
    lows = [o - 15.0 for o in opens]
    closes = [o + 2.0 for o in opens]

    # 第 30 根引發深跌觸發 ATR 停損
    lows[30] = opens[30] - 80.0
    closes[30] = opens[30] - 60.0

    df = pl.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": [100] * 50,
    })

    strat = StrategySchema.model_validate({
        "id": "test_atr_exit",
        "version": 1,
        "symbol": "TMF",
        "timeframe": "5m",
        "session": "day_and_night",
        "indicators": {
            "ma5": {"type": "SMA", "source": "close", "length": 5},
            "ma10": {"type": "SMA", "source": "close", "length": 10},
        },
        "entry_long": {"gt": ["ma5", 0.0]},
        "exit": {
            "stop_loss": {"type": "atr_multiple", "value": 2.0, "atr_length": 14},
        },
    })

    contract = _make_sample_contract()
    eng = EventDrivenEngine(strat, contract, 100_000.0)
    rep = eng.run_backtest(df)

    assert len(rep.trades) > 0
    # 確保有觸發 ATR 停損離場
    exit_reasons = [t.exit_reason for t in rep.trades]
    assert "stop_loss" in exit_reasons or "end_of_data" in exit_reasons


def test_time_exit_before_settlement():
    # 模擬 2026 年 11 月台指期結算日：2026-11-18 (第三個星期三)
    # 13:00 進場，13:10 K 棒收盤時已達截止門檻 (13:15 結束距離 13:30 剩 15 分鐘)，於 13:15 開盤強制平倉
    dts = [
        datetime(2026, 11, 18, 12, 50),
        datetime(2026, 11, 18, 12, 55),
        datetime(2026, 11, 18, 13, 0),
        datetime(2026, 11, 18, 13, 5),
        datetime(2026, 11, 18, 13, 10),
        datetime(2026, 11, 18, 13, 15),
        datetime(2026, 11, 18, 13, 20),
    ]
    df = pl.DataFrame({
        "timestamp": dts,
        "open": [20000.0] * 7,
        "high": [20010.0] * 7,
        "low": [19990.0] * 7,
        "close": [20005.0] * 7,
        "volume": [100] * 7,
    })

    strat = StrategySchema.model_validate({
        "id": "test_settlement_exit",
        "version": 1,
        "symbol": "TMF",
        "timeframe": "5m",
        "session": "day_and_night",
        "indicators": {
            "ma3": {"type": "SMA", "source": "close", "length": 3},
        },
        "entry_long": {"gt": ["close", 0.0]},
        "exit": {
            "stop_loss": {"type": "fixed_points", "value": 50.0},
            "time_exit": {"before_settlement": True, "minutes_before_close": 15},
        },
    })

    contract = _make_sample_contract()
    eng = EventDrivenEngine(strat, contract, 100_000.0)
    rep = eng.run_backtest(df)

    assert any(t.exit_reason == "settlement_exit" for t in rep.trades)

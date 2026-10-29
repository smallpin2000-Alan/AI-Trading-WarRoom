# tests/test_engine.py
import polars as pl
import pytest
from src.schema.strategy import StrategySchema
from src.core.engine import ContractSpec, EventDrivenEngine
from src.core.indicators import IndicatorEngine


def _make_tmf_contract() -> ContractSpec:
    return ContractSpec(
        symbol="TMF",
        name="微型臺指期貨",
        exchange="TAIFEX",
        point_value=10.0,
        tick_size=1.0,
        tax_rate=0.00002,
        default_commission=12.0,
    )


def test_indicators_sma_and_atr():
    df = pl.DataFrame({
        "open": [20000.0, 20010.0, 20020.0, 20030.0, 20040.0],
        "high": [20020.0, 20030.0, 20040.0, 20050.0, 20060.0],
        "low":  [19990.0, 20000.0, 20010.0, 20020.0, 20030.0],
        "close":[20010.0, 20020.0, 20030.0, 20040.0, 20050.0],
        "volume":[100, 120, 130, 140, 150],
    })
    out = IndicatorEngine.compute_indicators(
        df,
        {
            "ma3": {"type": "SMA", "source": "close", "length": 3},
            "atr3": {"type": "ATR", "length": 3},
        },
    )
    assert out["ma3"][2] == 20020.0
    assert out["ma3"][4] == 20040.0
    assert out["atr3"][3] is not None


def test_gap_execution_rule_on_stop_loss():
    """
    驗證跳空穿價真實模擬 (Gap Execution Rule)：
    - 多單在 20,000 點進場，固定停損設 80 點（理想停損價 19,920）。
    - 隔根 K 棒直接跳空低開在 19,850（低於 19,920）。
    - 回測引擎必須以 min(19,920, 19,850) - 滑價(1點) = 19,849 成交，絕不能用 19,920 成交！
    """
    strategy_dict = {
        "id": "gap_test_v1",
        "version": 1,
        "symbol": "TMF",
        "timeframe": "60m",
        "session": "day_and_night",
        "indicators": {
            "ma2": {"type": "SMA", "source": "close", "length": 2},
        },
        "entry_long": {
            "gt": ["close", "ma2"]
        },
        "entry_short": None,
        "exit": {
            "stop_loss": {"type": "fixed_points", "value": 80},
            "take_profit": None,
            "trailing_stop": None,
        },
        "execution": {
            "signal_on": "bar_close",
            "fill_at": "next_bar_open",
            "order_type": "range_market",
            "slippage_points": 1.0,
        },
        "position_sizing": {"type": "fixed", "contracts": 1},
    }
    strategy = StrategySchema.model_validate(strategy_dict)
    contract = _make_tmf_contract()
    engine = EventDrivenEngine(strategy, contract, initial_capital=100_000.0)

    # Bar 0, 1: 醞釀均線，Bar 1 收盤 20050 > ma2(20025) 觸發多單訊號
    # Bar 2: 開盤 20000 進場 (+1點滑價 -> 成交在 20001，停損價為 20001 - 80 = 19921)
    # Bar 3: 夜盤/隔日開盤直接跳空暴跌開在 19850 (穿過 19921 停損價)
    df = pl.DataFrame({
        "open":  [20000.0, 20000.0, 20000.0, 19850.0],
        "high":  [20010.0, 20060.0, 20020.0, 19860.0],
        "low":   [19990.0, 19995.0, 19950.0, 19800.0],
        "close": [20000.0, 20050.0, 19980.0, 19820.0],
        "volume":[100, 100, 100, 100],
    })

    report = engine.run_backtest(df)
    assert report.total_trades == 1
    trade = report.trades[0]

    # 驗證進場價 = Bar 2 Open (20000) + 滑價 (1) = 20001
    assert trade.entry_price == 20001.0
    # 驗證跳空穿價出場價 = Bar 3 Open (19850) - 滑價 (1) = 19849 (而非 19921 - 1)
    assert trade.exit_price == 19849.0
    assert trade.exit_reason == "stop_loss"
    # 毛損益 = (19849 - 20001) * 10元/點 = -1520 元
    assert trade.gross_pnl == -1520.0

# tests/test_validator_and_bars.py
from datetime import datetime
import polars as pl
from src.core.bar_builder import TickData, SharedBarBuilder
from src.core.validator import CredibilityValidator
from src.core.engine import ContractSpec
from src.schema.strategy import StrategySchema


def test_shared_bar_builder_day_and_night():
    """測試共用 K 棒合成器：過濾非交易時段並正確合成日盤與夜盤 K 棒"""
    ticks = [
        TickData(datetime(2026, 10, 22, 8, 30, 0), 20000.0, 50),   # 盤前試撮 (應被過濾)
        TickData(datetime(2026, 10, 22, 8, 45, 1), 20010.0, 10),   # 日盤開盤
        TickData(datetime(2026, 10, 22, 8, 50, 0), 20050.0, 20),
        TickData(datetime(2026, 10, 22, 9, 10, 0), 19990.0, 15),
        TickData(datetime(2026, 10, 22, 14, 10, 0), 20100.0, 30),  # 下午休息時段 (應被過濾)
        TickData(datetime(2026, 10, 22, 15, 0, 5), 20020.0, 12),   # 夜盤開盤
    ]

    bars_day_only = SharedBarBuilder.ticks_to_bars(ticks, timeframe="60m", session="day_only")
    assert bars_day_only.height == 2
    assert bars_day_only["open"][0] == 20010.0
    assert bars_day_only["high"][0] == 20050.0

    bars_both = SharedBarBuilder.ticks_to_bars(ticks, timeframe="60m", session="day_and_night")
    assert bars_both.height == 3  # 多一根 15:00 夜盤 K 棒


def test_credibility_validator_pipeline():
    """測試階段二：IS/OOS 切分、雙倍成本壓力測試與參數高原熱力圖"""
    strategy_dict = {
        "id": "trend_follow_v1",
        "version": 1,
        "symbol": "TMF",
        "timeframe": "60m",
        "session": "day_and_night",
        "indicators": {
            "fast_ma": {"type": "SMA", "source": "close", "length": 2},
            "slow_ma": {"type": "SMA", "source": "close", "length": 4},
        },
        "entry_long": {"cross_above": ["fast_ma", "slow_ma"]},
        "entry_short": None,
        "exit": {
            "stop_loss": {"type": "fixed_points", "value": 50},
            "take_profit": {"type": "fixed_points", "value": 150},
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
    contract = ContractSpec(
        symbol="TMF",
        name="微型臺指期貨",
        exchange="TAIFEX",
        point_value=10.0,
        tick_size=1.0,
        tax_rate=0.00002,
        default_commission=12.0,
    )

    # 構造包含兩段明顯上漲趨勢 (IS 與 OOS 各一段) 的模擬行情
    closes = [
        20000.0, 19980.0, 19960.0, 19950.0, 19990.0, 20040.0, 20120.0, 20200.0, 20250.0, 20230.0,
        20200.0, 20180.0, 20160.0, 20150.0, 20190.0, 20260.0, 20350.0, 20420.0, 20480.0, 20500.0,
    ]
    df = pl.DataFrame({
        "open": [c - 5.0 for c in closes],
        "high": [c + 25.0 for c in closes],
        "low":  [c - 15.0 for c in closes],
        "close": closes,
        "volume": [100] * len(closes),
    })

    validator = CredibilityValidator(strategy, contract)
    res = validator.evaluate_all(df, ind_x="fast_ma", x_values=[2, 3], ind_y="slow_ma", y_values=[4, 5], mc_iterations=200)

    assert res.is_report.net_profit > 0
    assert res.oos_report.net_profit > 0
    assert res.oos_passed is True
    assert res.stress_passed is True
    assert res.heatmap_plateau_passed is True
    assert res.monte_carlo_passed is True
    assert res.monte_carlo_report is not None
    assert res.monte_carlo_report.sample_trades_count >= 1
    assert res.overall_passed is True


def test_monte_carlo_simulation_standalone():
    """獨立測試蒙地卡羅隨機交易重抽與 MDD/淨利分位數計算"""
    from src.core.engine import TradeRecord

    trades = [
        TradeRecord(1, 1, 0, 1, 20000, 20100, 1000, 24, 8, 968, "take_profit"),
        TradeRecord(1, 1, 2, 3, 20100, 20050, -500, 24, 8, -532, "stop_loss"),
        TradeRecord(1, 1, 4, 5, 20050, 20250, 2000, 24, 8, 1968, "take_profit"),
        TradeRecord(1, 1, 6, 7, 20250, 20200, -500, 24, 8, -532, "stop_loss"),
    ]
    contract = ContractSpec("TMF", "微台", "TAIFEX", 10.0, 1.0, 0.00002, 12.0)
    validator = CredibilityValidator(None, contract, initial_capital=100000.0)
    mc = validator.run_monte_carlo(trades, iterations=500, seed=123)

    assert mc.iterations == 500
    assert mc.sample_trades_count == 4
    assert mc.median_mdd >= 0.0
    assert mc.mdd_95th >= mc.median_mdd
    assert len(mc.mdd_distribution) == 500
    assert len(mc.profit_distribution) == 500


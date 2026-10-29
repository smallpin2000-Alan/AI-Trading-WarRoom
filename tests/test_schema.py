import pytest
from pydantic import ValidationError
from src.schema.strategy import StrategySchema

VALID_STRATEGY_JSON = {
    "id": "ma_cross_v1",
    "version": 4,
    "symbol": "TMF",
    "timeframe": "60m",
    "session": "day_and_night",
    "indicators": {
        "ma5": {"type": "SMA", "source": "close", "length": 5, "timeframe": "60m"},
        "ma10": {"type": "SMA", "source": "close", "length": 10, "timeframe": "60m"},
        "ma20": {"type": "SMA", "source": "close", "length": 20, "timeframe": "60m"},
        "daily_ma20": {"type": "SMA", "source": "close", "length": 20, "timeframe": "1D"}
    },
    "entry_long": {
        "all": [
            {"cross_above": ["ma5", "ma10"]},
            {"gt": ["ma10", "ma20"]},
            {"gt": ["close", "daily_ma20"]},
            {"lte": [{"bars_since": {"cross_above": ["ma5", "ma10"]}}, 3]}
        ]
    },
    "entry_short": {
        "all": [
            {"cross_below": ["ma5", "ma10"]},
            {"lt": ["ma10", "ma20"]},
            {"lt": ["close", "daily_ma20"]}
        ]
    },
    "exit": {
        "stop_loss": {
            "type": "indicator_cross",
            "long": {"cross_below": ["close", "ma20"]},
            "short": {"cross_above": ["close", "ma20"]}
        },
        "take_profit": None,
        "trailing_stop": {"type": "points", "value": 80},
        "time_exit": {"before_settlement": True, "minutes_before_close": 15}
    },
    "execution": {
        "signal_on": "bar_close",
        "fill_at": "next_bar_open",
        "order_type": "range_market",
        "slippage_points": 1
    },
    "position_sizing": {"type": "fixed", "contracts": 1}
}


def test_valid_strategy_schema():
    strategy = StrategySchema.model_validate(VALID_STRATEGY_JSON)
    assert strategy.symbol == "TMF"
    assert strategy.indicators["daily_ma20"].timeframe == "1D"
    assert strategy.exit.trailing_stop.value == 80


def test_reject_missing_stop_loss():
    bad_json = VALID_STRATEGY_JSON.copy()
    bad_json["exit"] = {"stop_loss": None, "take_profit": None, "trailing_stop": None}
    with pytest.raises(ValidationError, match="安全攔截：策略必須至少設定 stop_loss 或 trailing_stop"):
        StrategySchema.model_validate(bad_json)


def test_reject_undefined_indicator():
    bad_json = VALID_STRATEGY_JSON.copy()
    bad_json["entry_long"] = {"all": [{"cross_above": ["ma5", "ma60"]}]}
    with pytest.raises(ValidationError, match="未定義的指標或欄位名稱: 'ma60'"):
        StrategySchema.model_validate(bad_json)

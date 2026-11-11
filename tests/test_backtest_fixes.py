# tests/test_backtest_fixes.py
"""
回測引擎 / 出場 / 驗證器 / 指標 / Schema / 集成 的安全性與正確性回歸測試。
每個測試對應一項已確認的缺陷，避免日後又被改回去。
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from src.ai.confidence_engine import ConfidenceEngine
from src.ai import ensemble as ensemble_module
from src.ai.ensemble import MultiModelEnsemble
from src.core import exits as exits_module
from src.core import validator as validator_module
from src.core.engine import BacktestReport, ContractSpec, EventDrivenEngine, TradeRecord
from src.core.exits import (
    can_open_position,
    compute_contracts,
    compute_exit_levels,
    required_atr_lengths,
    settlement_cutoff_state,
)
from src.core.indicators import IndicatorEngine
from src.core.validator import CredibilityValidator
from src.schema.strategy import StrategySchema

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 共用工具
# ---------------------------------------------------------------------------
def _contract(initial_margin: float = 16100.0) -> ContractSpec:
    return ContractSpec(
        symbol="TMF",
        name="微型臺指期貨",
        exchange="TAIFEX",
        point_value=10.0,
        tick_size=1.0,
        tax_rate=0.00002,
        default_commission=12.0,
        initial_margin=initial_margin,
    )


def _strategy(**overrides) -> StrategySchema:
    base = {
        "id": "fix_test",
        "version": 1,
        "symbol": "TMF",
        "timeframe": "5m",
        "session": "day_and_night",
        "indicators": {
            "fast_ma": {"type": "SMA", "source": "close", "length": 2},
            "slow_ma": {"type": "SMA", "source": "close", "length": 4},
        },
        "entry_long": {"cross_above": ["fast_ma", "slow_ma"]},
        "entry_short": None,
        "exit": {"stop_loss": {"type": "fixed_points", "value": 50.0}},
    }
    base.update(overrides)
    return StrategySchema.model_validate(base)


def _bars(closes, start: datetime | None = None, minutes: int = 5, spread: float = 5.0) -> pl.DataFrame:
    data = {
        "open": [float(c) for c in closes],
        "high": [float(c) + spread for c in closes],
        "low": [float(c) - spread for c in closes],
        "close": [float(c) for c in closes],
        "volume": [100] * len(closes),
    }
    if start is not None:
        data = {"timestamp": [start + timedelta(minutes=minutes * i) for i in range(len(closes))], **data}
    return pl.DataFrame(data)


def _cyclic_trend_closes(cycles: int = 40) -> list[float]:
    """鋸齒狀多頭行情：每個循環先回檔 4 根、再上漲 8 根，讓順勢策略有足夠多筆交易。"""
    closes: list[float] = []
    price = 20000.0
    for c in range(cycles):
        for _ in range(4):
            price -= 20.0 + (c % 3) * 2.0
            closes.append(price)
        for _ in range(8):
            price += 45.0 + (c % 5) * 3.0
            closes.append(price)
    return closes


# ---------------------------------------------------------------------------
# 1. 純移動停損：啟動前也必須有硬停損
# ---------------------------------------------------------------------------
def test_pure_trailing_stop_has_initial_hard_stop_before_activation():
    strat = _strategy(exit={"trailing_stop": {"type": "points", "value": 30.0, "activation_points": 50.0}})

    stop, tp, reason = compute_exit_levels(strat, 1, 20000.0, 20000.0, 20000.0, {}, {})
    assert stop == 19970.0
    assert tp is None
    assert reason == "stop_loss"

    stop_s, _, reason_s = compute_exit_levels(strat, -1, 20000.0, 20000.0, 20000.0, {}, {})
    assert stop_s == 20030.0
    assert reason_s == "stop_loss"

    # 啟動後改由移動停損接手
    stop_a, _, reason_a = compute_exit_levels(strat, 1, 20000.0, 20060.0, 20000.0, {}, {})
    assert stop_a == 20030.0
    assert reason_a == "trailing_stop"


def test_engine_pure_trailing_stop_exits_on_crash_before_activation():
    strat = _strategy(
        entry_long={"gt": ["close", "fast_ma"]},
        exit={"trailing_stop": {"type": "points", "value": 30.0, "activation_points": 500.0}},
        execution={"slippage_points": 0.0},
    )
    # Bar 2 收盤 > 均線產生訊號，Bar 3 開盤 20020 進場，Bar 4 崩跌
    df = pl.DataFrame({
        "open":  [20000.0, 20000.0, 20020.0, 20020.0, 20015.0, 19800.0],
        "high":  [20005.0, 20005.0, 20025.0, 20025.0, 20020.0, 19805.0],
        "low":   [19995.0, 19995.0, 20000.0, 20010.0, 19700.0, 19790.0],
        "close": [20000.0, 20000.0, 20020.0, 20020.0, 19750.0, 19800.0],
        "volume": [100] * 6,
    })
    rep = EventDrivenEngine(strat, _contract(), 100_000.0).run_backtest(df)
    assert rep.total_trades >= 1
    first = rep.trades[0]
    assert first.exit_reason == "stop_loss"
    assert first.exit_price == first.entry_price - 30.0


# ---------------------------------------------------------------------------
# 2. indicator_cross 停損的安全規則
# ---------------------------------------------------------------------------
def _indicator_cross_exit(**kw):
    ex = {
        "stop_loss": {
            "type": "indicator_cross",
            "long": {"cross_below": ["close", "slow_ma"]},
            "short": {"cross_above": ["close", "slow_ma"]},
        },
        "trailing_stop": {"type": "points", "value": 80.0},
    }
    ex.update(kw)
    return ex


def test_schema_rejects_indicator_cross_without_price_protection():
    with pytest.raises(ValidationError, match="indicator_cross"):
        _strategy(exit=_indicator_cross_exit(trailing_stop=None))


def test_schema_rejects_indicator_cross_missing_side():
    only_short = {"type": "indicator_cross", "short": {"cross_above": ["close", "slow_ma"]}}
    with pytest.raises(ValidationError, match="long 出場條件"):
        _strategy(exit=_indicator_cross_exit(stop_loss=only_short))

    only_long = {"type": "indicator_cross", "long": {"cross_below": ["close", "slow_ma"]}}
    with pytest.raises(ValidationError, match="short 出場條件"):
        _strategy(
            entry_long=None,
            entry_short={"cross_below": ["fast_ma", "slow_ma"]},
            exit=_indicator_cross_exit(stop_loss=only_long),
        )

    # 只做多 + 只有 long 出場條件 + 有移動停損 → 合法
    ok = _strategy(exit=_indicator_cross_exit(stop_loss=only_long))
    assert ok.exit.stop_loss.short is None


def test_can_open_position_direction_check_for_indicator_cross():
    strat = _strategy(
        entry_short={"cross_below": ["fast_ma", "slow_ma"]},
        exit=_indicator_cross_exit(),
    )
    assert can_open_position(strat, 20000.0, {}) == (True, "")
    assert can_open_position(strat, 20000.0, {}, direction=1)[0] is True

    # 進場後策略被改動（或繞過 Schema 建構）導致缺少該方向出場條件 → 拒絕該方向
    strat.exit.stop_loss.long = None
    ok_long, msg = can_open_position(strat, 20000.0, {}, direction=1)
    assert ok_long is False and "long" in msg
    assert can_open_position(strat, 20000.0, {}, direction=-1)[0] is True

    # 沒有可生效的價位保護 → 拒絕
    strat.exit.trailing_stop = None
    ok_none, msg_none = can_open_position(strat, 20000.0, {}, direction=-1)
    assert ok_none is False and "價位" in msg_none


def test_can_open_position_indicator_cross_refuses_when_trailing_atr_not_ready():
    strat = _strategy(exit=_indicator_cross_exit(trailing_stop={"type": "atr_multiple", "value": 2.0}))
    assert can_open_position(strat, 20000.0, {14: float("nan")}, direction=1)[0] is False
    assert can_open_position(strat, 20000.0, {14: 25.0}, direction=1)[0] is True


# ---------------------------------------------------------------------------
# 3. atr_length 與未知欄位
# ---------------------------------------------------------------------------
def test_atr_length_is_part_of_schema_and_changes_stop_distance():
    s5 = _strategy(exit={"stop_loss": {"type": "atr_multiple", "value": 2.0, "atr_length": 5}})
    s14 = _strategy(exit={"stop_loss": {"type": "atr_multiple", "value": 2.0}})
    assert s5.exit.stop_loss.atr_length == 5
    assert s14.exit.stop_loss.atr_length == 14
    assert required_atr_lengths(s5) == [5]
    assert required_atr_lengths(s14) == [14]

    atr = {5: 10.0, 14: 30.0}
    stop5, _, _ = compute_exit_levels(s5, 1, 20000.0, 20000.0, 20000.0, atr, atr)
    stop14, _, _ = compute_exit_levels(s14, 1, 20000.0, 20000.0, 20000.0, atr, atr)
    assert stop5 == 19980.0
    assert stop14 == 19940.0


def test_atr_length_changes_backtest_result():
    closes = [20000.0 + i * 5 for i in range(60)]
    df = _bars(closes, spread=15.0)
    # 早期波動小、近期波動放大：短週期 ATR 反應較快，停損距離不同
    df = df.with_columns(
        pl.when(pl.int_range(pl.len()) >= 20).then(pl.col("high") + 40.0).otherwise(pl.col("high")).alias("high")
    )
    reports = {}
    for ln in (5, 14):
        strat = _strategy(
            entry_long={"gt": ["close", 0.0]},
            exit={"stop_loss": {"type": "atr_multiple", "value": 1.0, "atr_length": ln}},
        )
        reports[ln] = EventDrivenEngine(strat, _contract(), 100_000.0).run_backtest(df)
    assert reports[5].total_trades > 0 and reports[14].total_trades > 0
    assert reports[5].trades[0].entry_idx != reports[14].trades[0].entry_idx


def test_schema_rejects_unknown_fields_in_exit_and_condition_nodes():
    with pytest.raises(ValidationError):
        _strategy(exit={"stop_loss": {"type": "atr_multiple", "value": 2.0, "atr_len": 5}})
    with pytest.raises(ValidationError):
        _strategy(exit={"trailing_stop": {"type": "points", "value": 30.0, "activation": 10}})
    with pytest.raises(ValidationError):
        _strategy(exit={"stop_loss": {"type": "fixed_points", "value": 50.0}, "stoploss": {"type": "percent", "value": 1}})
    with pytest.raises(ValidationError):
        _strategy(entry_long={"cross_above": ["fast_ma", "slow_ma"], "typo_op": ["close", 1]})


def test_existing_strategies_remain_valid_and_round_trip():
    for p in sorted((PROJECT_ROOT / "strategies").glob("*.json")):
        s = StrategySchema.model_validate(json.loads(p.read_text(encoding="utf-8")))
        assert StrategySchema.model_validate(s.model_dump()) == s
        assert StrategySchema.model_validate(json.loads(s.model_dump_json())) == s

    # dashboard.py 內建預設策略（也是 data/paper_trading_state.json 內存的結構）
    default = StrategySchema.model_validate({
        "id": "ai_copilot_auto",
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
    dumped = default.model_dump()
    # 舊版存檔沒有 atr_length 欄位，也必須能讀回
    for key in ("stop_loss", "take_profit", "trailing_stop"):
        dumped["exit"][key].pop("atr_length")
    assert StrategySchema.model_validate(dumped) == default


# ---------------------------------------------------------------------------
# 4. 蒙地卡羅：有放回重抽
# ---------------------------------------------------------------------------
def _trades_from_pnls(pnls) -> list[TradeRecord]:
    return [TradeRecord(1, 1, i, i + 1, 20000.0, 20000.0, p, 0.0, 0.0, p, "take_profit") for i, p in enumerate(pnls)]


def test_monte_carlo_bootstrap_profit_is_not_an_identity():
    # 總淨利為正，但勝負幾乎各半：舊版（只重排順序）每次淨利都相同而必過
    pnls = [1000.0] * 15 + [-900.0] * 15
    v = CredibilityValidator(None, _contract(), initial_capital=100_000.0)
    mc = v.run_monte_carlo(_trades_from_pnls(pnls), iterations=500, seed=7)
    assert len(set(round(p, 2) for p in mc.profit_distribution)) > 10
    assert mc.profit_5th < 0.0
    assert mc.passed is False


def test_monte_carlo_passes_only_with_enough_trades_and_bounded_drawdown():
    v = CredibilityValidator(None, _contract(), initial_capital=100_000.0)
    steady = [300.0] * 30 + [-100.0] * 10
    assert v.run_monte_carlo(_trades_from_pnls(steady), iterations=500).passed is True

    # 交易筆數不足
    few = [300.0] * (validator_module.MC_MIN_TRADES - 1)
    assert v.run_monte_carlo(_trades_from_pnls(few), iterations=500).passed is False

    # 獲利面通過，但 95% 分位 MDD 超過本金上限比例
    small = CredibilityValidator(None, _contract(), initial_capital=10_000.0)
    wild = [6000.0, -4000.0] * 100
    mc = small.run_monte_carlo(_trades_from_pnls(wild), iterations=500)
    assert mc.profit_5th > 0.0
    assert mc.mdd_95th > 10_000.0 * validator_module.MC_MAX_MDD_RATIO
    assert mc.passed is False


# ---------------------------------------------------------------------------
# 5. overall_passed 納入 Walk-Forward 與最少交易數
# ---------------------------------------------------------------------------
def test_overall_passed_requires_walk_forward(monkeypatch):
    strat = _strategy(
        exit={"stop_loss": {"type": "fixed_points", "value": 50.0}, "take_profit": {"type": "fixed_points", "value": 150.0}}
    )
    df = _bars(_cyclic_trend_closes(40), spread=10.0)
    v = CredibilityValidator(strat, _contract())
    res = v.evaluate_all(df, "fast_ma", [2, 3], "slow_ma", [4, 5], mc_iterations=200)
    assert res.walk_forward_passed is True
    assert res.overall_passed is True

    failing = validator_module.WalkForwardResult(folds=[], consistency_ratio=0.0, total_oos_profit=-1.0, passed=False)
    monkeypatch.setattr(CredibilityValidator, "run_walk_forward", lambda self, *a, **k: failing)
    res2 = v.evaluate_all(df, "fast_ma", [2, 3], "slow_ma", [4, 5], mc_iterations=200)
    assert res2.oos_passed and res2.stress_passed and res2.monte_carlo_passed and res2.heatmap_plateau_passed
    assert res2.walk_forward_passed is False
    assert res2.overall_passed is False
    assert any("Walk-Forward" in r for r in res2.reasons)


def test_too_few_trades_cannot_pass_validation():
    strat = _strategy(
        exit={"stop_loss": {"type": "fixed_points", "value": 50.0}, "take_profit": {"type": "fixed_points", "value": 150.0}}
    )
    closes = [
        20000.0, 19980.0, 19960.0, 19950.0, 19990.0, 20040.0, 20120.0, 20200.0, 20250.0, 20230.0,
        20200.0, 20180.0, 20160.0, 20150.0, 20190.0, 20260.0, 20350.0, 20420.0, 20480.0, 20500.0,
    ]
    df = pl.DataFrame({
        "open": [c - 5.0 for c in closes],
        "high": [c + 25.0 for c in closes],
        "low": [c - 15.0 for c in closes],
        "close": closes,
        "volume": [100] * len(closes),
    })
    res = CredibilityValidator(strat, _contract()).evaluate_all(df, "fast_ma", [2, 3], "slow_ma", [4, 5], mc_iterations=100)
    assert res.oos_report.net_profit > 0  # 有賺錢，但樣本太少
    assert res.oos_passed is False
    assert res.stress_passed is False
    assert res.monte_carlo_passed is False
    assert res.overall_passed is False
    assert any("交易筆數" in r for r in res.reasons)


def test_walk_forward_fold_with_zero_trades_does_not_pass():
    strat = _strategy()
    df = _bars([20000.0] * 90)
    wf = CredibilityValidator(strat, _contract()).run_walk_forward(df, n_splits=3)
    assert len(wf.folds) == 3
    assert all(f.test_trades == 0 and f.passed is False for f in wf.folds)
    assert wf.passed is False


# ---------------------------------------------------------------------------
# 6. 保證金檢查
# ---------------------------------------------------------------------------
def test_compute_contracts_capped_by_margin():
    strat = _strategy(position_sizing={"type": "fixed", "contracts": 5})
    assert compute_contracts(strat, 100_000.0, 10.0, 20000.0, {}) == 5  # 未提供保證金 → 維持原行為
    assert compute_contracts(strat, 40_000.0, 10.0, 20000.0, {}, initial_margin=16100.0) == 2
    assert compute_contracts(strat, 16_000.0, 10.0, 20000.0, {}, initial_margin=16100.0) == 0
    assert compute_contracts(strat, 0.0, 10.0, 20000.0, {}, initial_margin=16100.0) == 0
    assert compute_contracts(strat, -500.0, 10.0, 20000.0, {}, initial_margin=16100.0) == 0


def test_backtest_respects_initial_margin():
    strat = _strategy(position_sizing={"type": "fixed", "contracts": 3})
    df = _bars(_cyclic_trend_closes(4), spread=10.0)

    rep = EventDrivenEngine(strat, _contract(16100.0), initial_capital=20_000.0).run_backtest(df)
    assert rep.total_trades > 0
    assert rep.trades[0].contracts == 1  # 20,000 只夠 1 口

    rep_none = EventDrivenEngine(strat, _contract(16100.0), initial_capital=10_000.0).run_backtest(df)
    assert rep_none.total_trades == 0  # 連 1 口保證金都不夠


# ---------------------------------------------------------------------------
# 7. 參數精靈：點數掃描只套用在點數型停損
# ---------------------------------------------------------------------------
def test_parameter_wizard_does_not_apply_points_to_percent_stop():
    df = _bars(_cyclic_trend_closes(12), spread=10.0)
    kwargs = dict(
        ind_x="fast_ma", fast_range=[2], ind_y="slow_ma", slow_range=[4],
        sl_range=[40.0, 60.0], threshold_range=[0.0], top_k=10,
    )
    pct = _strategy(exit={"stop_loss": {"type": "percent", "value": 0.5}})
    cands_pct = CredibilityValidator(pct, _contract()).run_parameter_wizard(df, **kwargs)
    assert len(cands_pct) == 1  # 停損維度被略過，不會把 40/60 當成 40%/60%
    assert pct.exit.stop_loss.value == 0.5

    fixed = _strategy(exit={"stop_loss": {"type": "fixed_points", "value": 50.0}})
    cands_fixed = CredibilityValidator(fixed, _contract()).run_parameter_wizard(df, **kwargs)
    assert sorted(c.stop_loss_pts for c in cands_fixed) == [40.0, 60.0]


# ---------------------------------------------------------------------------
# 8. 多週期指標（無前視）
# ---------------------------------------------------------------------------
def _three_day_5m_bars() -> pl.DataFrame:
    rows = []
    price = 20000.0
    for day in (9, 10, 11):  # 2026-11-09 (一) ~ 11-11 (三)
        ts = datetime(2026, 11, day, 9, 0)
        for i in range(24):  # 09:00 ~ 10:55
            price += 3.0
            rows.append((ts + timedelta(minutes=5 * i), price))
    return pl.DataFrame({
        "timestamp": [r[0] for r in rows],
        "open": [r[1] for r in rows],
        "high": [r[1] + 2 for r in rows],
        "low": [r[1] - 2 for r in rows],
        "close": [r[1] for r in rows],
        "volume": [10] * len(rows),
    })


def test_higher_timeframe_indicator_uses_only_closed_bars():
    df = _three_day_5m_bars()
    spec = {"d_close": {"type": "SMA", "source": "close", "length": 1, "timeframe": "1D"}}
    out = IndicatorEngine.compute_indicators(df, spec, base_timeframe="5m")
    vals = out["d_close"].to_list()
    closes = df["close"].to_list()

    assert all(v is None for v in vals[:24])                 # 第一天：尚無已收盤日線
    assert all(v == closes[23] for v in vals[24:48])         # 第二天整天（含最後一根）只看得到第一天收盤
    assert all(v == closes[47] for v in vals[48:72])

    # 無前視：竄改第二天盤中 / 收盤價，第二天的日線值不可改變
    tampered = df.with_columns(
        pl.when(pl.int_range(pl.len()) >= 24).then(pl.col("close") + 500.0).otherwise(pl.col("close")).alias("close")
    )
    out2 = IndicatorEngine.compute_indicators(tampered, spec, base_timeframe="5m")
    assert out2["d_close"].to_list()[:48] == vals[:48]


def test_higher_timeframe_60m_alignment_and_same_timeframe_passthrough():
    df = _three_day_5m_bars()
    out = IndicatorEngine.compute_indicators(
        df,
        {
            "h_close": {"type": "SMA", "source": "close", "length": 1, "timeframe": "60m"},
            "same": {"type": "SMA", "source": "close", "length": 3, "timeframe": "5m"},
            "plain": {"type": "SMA", "source": "close", "length": 3},
        },
        base_timeframe="5m",
    )
    closes = df["close"].to_list()
    h = out["h_close"].to_list()
    assert all(v is None for v in h[:12])                    # 09:00 這根 60 分 K 尚未收盤
    assert all(v == closes[11] for v in h[12:24])            # 10:00~10:55 只能用 09:00 那根（收在 09:55）
    assert out["same"].to_list() == out["plain"].to_list()


def test_higher_timeframe_invalid_requests_raise():
    df = _three_day_5m_bars()
    with pytest.raises(ValueError, match="小於"):
        IndicatorEngine.compute_indicators(
            df, {"x": {"type": "SMA", "length": 3, "timeframe": "1m"}}, base_timeframe="5m"
        )
    with pytest.raises(ValueError, match="timestamp"):
        IndicatorEngine.compute_indicators(
            df.drop("timestamp"), {"x": {"type": "SMA", "length": 3, "timeframe": "60m"}}, base_timeframe="5m"
        )


def test_engine_runs_strategy_with_daily_indicator():
    strat = _strategy(
        indicators={
            "fast_ma": {"type": "SMA", "source": "close", "length": 2},
            "d_ma": {"type": "SMA", "source": "close", "length": 1, "timeframe": "1D"},
        },
        entry_long={"all": [{"gt": ["close", "fast_ma"]}, {"gt": ["close", "d_ma"]}]},
    )
    rep = EventDrivenEngine(strat, _contract(), 100_000.0).run_backtest(_three_day_5m_bars())
    assert rep.total_trades >= 1
    # 第一天沒有已收盤日線，不可能進場
    assert min(t.entry_idx for t in rep.trades) >= 24


# ---------------------------------------------------------------------------
# 9 / 10 / 19 / 21. 集成
# ---------------------------------------------------------------------------
def _trend_bars(n: int = 120) -> pl.DataFrame:
    base_dt = datetime(2026, 10, 20, 9, 0, 0)
    prices = [20000.0 + i * 6.0 + np.sin(i / 3.0) * 15.0 for i in range(n)]
    return pl.DataFrame({
        "timestamp": [base_dt + timedelta(minutes=5 * i) for i in range(n)],
        "open": [p - 3.0 for p in prices],
        "high": [p + 12.0 for p in prices],
        "low": [p - 10.0 for p in prices],
        "close": prices,
        "volume": [100 + (i % 7) * 30 for i in range(n)],
    })


def _stopped_out_then_flat_bars() -> pl.DataFrame:
    closes = [20000.0] * 30 + [20030.0, 20040.0, 19900.0, 19895.0, 19890.0]
    return _bars(closes, start=datetime(2026, 10, 20, 9, 0), spread=3.0)


def test_ensemble_does_not_treat_closed_trade_as_current_signal():
    strat = _strategy(id="stopped_model", exit={"stop_loss": {"type": "fixed_points", "value": 20.0}})
    df = _stopped_out_then_flat_bars()
    dec = MultiModelEnsemble.evaluate_ensemble(df, _contract(), custom_pool=[("停損後空手", strat)])
    m = dec.models[0]
    assert m.report.total_trades >= 1
    assert m.report.trades[-1].exit_reason == "stop_loss"
    assert m.report.trades[-1].exit_idx >= df.height - 3
    assert m.latest_signal == 0
    assert dec.bull_votes == 0 and dec.final_action == 0


def test_ensemble_uses_direction_of_position_still_open_at_end():
    strat = _strategy(id="holding_model", exit={"stop_loss": {"type": "fixed_points", "value": 500.0}})
    closes = [20000.0] * 30 + [20030.0, 20040.0, 20050.0, 20060.0, 20070.0]
    df = _bars(closes, start=datetime(2026, 10, 20, 9, 0), spread=3.0)
    dec = MultiModelEnsemble.evaluate_ensemble(df, _contract(), custom_pool=[("仍持多單", strat)])
    m = dec.models[0]
    assert m.report.trades[-1].exit_reason == "end_of_data"
    assert m.latest_signal == 1
    assert dec.bull_votes == 1


def test_ensemble_votes_count_only_real_signals_and_action_needs_one():
    never = _strategy(id="never_signals", entry_long={"lt": ["close", 0.0]})
    df = _trend_bars(90)
    for method in ("dynamic_weighted", "average", "voting"):
        dec = MultiModelEnsemble.evaluate_ensemble(
            df, _contract(), method=method, threshold=30.0, regime_adaptive=False,
            custom_pool=[("永不出手 A", never), ("永不出手 B", never)],
        )
        assert all(m.latest_signal == 0 for m in dec.models)
        assert dec.bull_votes == 0 and dec.bear_votes == 0
        assert dec.neutral_votes == dec.total_active_models == 2
        assert dec.final_action == 0


def test_ensemble_gives_minimum_weight_to_models_without_enough_trades():
    never = _strategy(id="never_trades", entry_long={"lt": ["close", 0.0]})
    active = _strategy(
        id="active_trader",
        exit={"stop_loss": {"type": "fixed_points", "value": 50.0}, "take_profit": {"type": "fixed_points", "value": 150.0}},
    )
    df = _bars(_cyclic_trend_closes(12), start=datetime(2026, 10, 20, 9, 0), spread=10.0)
    dec = MultiModelEnsemble.evaluate_ensemble(
        df, _contract(), custom_pool=[("零交易", never), ("有績效", active)]
    )
    by_id = {m.model_id: m for m in dec.models}
    assert by_id["active_trader"].report.total_trades >= ensemble_module.ENSEMBLE_MIN_TRADES_FOR_WEIGHT
    assert by_id["never_trades"].report.total_trades == 0
    assert by_id["never_trades"].dynamic_weight < by_id["active_trader"].dynamic_weight
    assert by_id["never_trades"].dynamic_weight == min(m.dynamic_weight for m in dec.models)


def test_ensemble_logs_warning_for_unloadable_strategy_file(tmp_path, monkeypatch, caplog):
    strat_dir = tmp_path / "strategies"
    strat_dir.mkdir()
    (strat_dir / "broken_strategy.json").write_text('{"id": "broken", "version": 1}', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with caplog.at_level(logging.WARNING):
        pool = MultiModelEnsemble.build_default_model_pool()
    assert all(s.id != "broken" for _, s in pool)
    assert any("broken_strategy.json" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# 11 / 20. 信心度引擎
# ---------------------------------------------------------------------------
def test_effective_threshold_respects_user_base_outside_50_90():
    df = _trend_bars(90)
    low = ConfidenceEngine.compute_confidence_series(df, base_threshold=30.0, regime_adaptive=True)
    assert low["effective_threshold"].max() <= 40.0
    assert low["effective_threshold"].min() >= 30.0

    flat = _bars([20000.0 + (i % 2) for i in range(90)], start=datetime(2026, 10, 20, 9, 0))
    high = ConfidenceEngine.compute_confidence_series(flat, base_threshold=95.0, regime_adaptive=True)
    assert high["effective_threshold"][-1] == 95.0
    assert high["effective_threshold"].max() <= 95.0


def test_latest_snapshot_regime_matches_series():
    # 前段高波動、後段低波動：全域中位數與滾動中位數會不同
    rng = np.random.default_rng(6)
    closes = [20000.0 + float(rng.normal(0, 60)) for _ in range(120)]
    closes += [20000.0 + float(rng.normal(0, 10)) for _ in range(60)]
    df = _bars(closes, start=datetime(2026, 10, 20, 9, 0))
    for end in (150, 162, 164, 166, 168, 172, 180):
        sub = df.head(end)
        conf_df = ConfidenceEngine.compute_confidence_series(sub, base_threshold=60.0, regime_adaptive=True)
        snap = ConfidenceEngine.get_latest_snapshot(sub, base_threshold=60.0, regime_adaptive=True)
        assert snap.regime.label_zh == conf_df["regime_label"][-1]
        assert snap.effective_threshold == round(60.0 + snap.regime.threshold_offset, 1)


# ---------------------------------------------------------------------------
# 12 / 13. 指標暖機與 RSI 持平
# ---------------------------------------------------------------------------
def test_indicator_warmup_is_masked():
    df = _trend_bars(60)
    ema = IndicatorEngine.calculate_ema(df, 5).to_list()
    assert all(v is None for v in ema[:4]) and ema[4] is not None

    rsi = IndicatorEngine.calculate_rsi(df, 14).to_list()
    assert all(v is None for v in rsi[:14]) and rsi[14] is not None

    adx = IndicatorEngine.calculate_adx(df, 14).to_list()
    assert all(v is None for v in adx[:14]) and adx[14] is not None


def test_confidence_series_has_no_nan_after_warmup_masking():
    for n in (3, 10, 40):
        conf_df = ConfidenceEngine.compute_confidence_series(_trend_bars(n))
        for col in ("long_confidence", "short_confidence", "effective_threshold"):
            vals = conf_df[col].to_list()
            assert all(v is not None and not math.isnan(v) for v in vals)


def test_rsi_is_50_when_price_is_flat():
    df = _bars([20000.0] * 40)
    rsi = IndicatorEngine.calculate_rsi(df, 14).to_list()
    assert all(v == 50.0 for v in rsi[14:])

    # 單邊上漲仍應接近 100
    up = IndicatorEngine.calculate_rsi(_bars([20000.0 + i for i in range(40)]), 14).to_list()
    assert up[-1] > 99.0


# ---------------------------------------------------------------------------
# 15. 結算日保護無法判斷時間時要警告
# ---------------------------------------------------------------------------
def test_settlement_guard_warns_when_timestamp_unusable(caplog):
    strat = _strategy()
    exits_module._SETTLEMENT_WARNED.clear()
    with caplog.at_level(logging.WARNING):
        assert settlement_cutoff_state(strat, "third_wednesday", None) == (False, False)
    assert any("結算" in r.getMessage() for r in caplog.records)

    # ISO 字串時間戳仍可判斷（2026-11-18 為第三個星期三）
    assert settlement_cutoff_state(strat, "third_wednesday", "2026-11-18 13:10:00") == (True, True)


# ---------------------------------------------------------------------------
# 16. 權益曲線與交易紀錄一致
# ---------------------------------------------------------------------------
def test_equity_curve_final_point_matches_trades_after_end_of_data():
    strat = _strategy(exit={"stop_loss": {"type": "fixed_points", "value": 500.0}})
    closes = [20000.0] * 10 + [20030.0, 20040.0, 20050.0, 20060.0, 20070.0]
    rep = EventDrivenEngine(strat, _contract(), 100_000.0).run_backtest(_bars(closes, spread=3.0))
    assert rep.trades[-1].exit_reason == "end_of_data"
    assert rep.equity_curve[-1] == pytest.approx(100_000.0 + sum(t.net_pnl for t in rep.trades))


# ---------------------------------------------------------------------------
# 17. OOS 暖機區段不可產生交易
# ---------------------------------------------------------------------------
def test_trade_start_idx_blocks_entries_during_warmup():
    strat = _strategy(entry_long={"gt": ["close", 0.0]})
    df = _bars([20000.0 + i for i in range(40)])
    eng = EventDrivenEngine(strat, _contract(), 100_000.0)
    assert min(t.entry_idx for t in eng.run_backtest(df).trades) < 10
    rep = eng.run_backtest(df, trade_start_idx=10)
    assert rep.total_trades >= 1
    assert min(t.entry_idx for t in rep.trades) >= 10
    assert len(rep.equity_curve) >= df.height


def test_oos_report_excludes_trades_opened_in_warmup_segment():
    strat = _strategy(
        indicators={
            "fast_ma": {"type": "SMA", "source": "close", "length": 2},
            "slow_ma": {"type": "SMA", "source": "close", "length": 20},
        },
        entry_long={"gt": ["close", 0.0]},
    )
    df = _bars([20000.0 + i for i in range(100)])
    _, oos = CredibilityValidator(strat, _contract()).run_is_oos_split(df, is_ratio=0.7)
    assert oos.total_trades >= 1
    # OOS 切片前 20 根為暖機緩衝，進場位置必須在緩衝之後
    assert min(t.entry_idx for t in oos.trades) >= 20


# ---------------------------------------------------------------------------
# 18. 熱力圖高原：零交易格也算在分母
# ---------------------------------------------------------------------------
def test_heatmap_plateau_counts_zero_trade_cells(monkeypatch):
    profits = {(2, 4): 5000.0}  # 只有一格賺錢，其餘皆零交易

    class _FakeEngine:
        def __init__(self, strategy, contract, initial_capital=100_000.0):
            self.key = (strategy.indicators["fast_ma"].length, strategy.indicators["slow_ma"].length)

        def run_backtest(self, df, cost_multiplier=1.0, **kwargs):
            net = profits.get(self.key, 0.0)
            return BacktestReport([], [], net, max(net, 0.0), 0.0, 0.0, 0.0, 0.0, 0.0, 1 if net else 0, net)

    monkeypatch.setattr(validator_module, "EventDrivenEngine", _FakeEngine)
    v = CredibilityValidator(_strategy(), _contract())
    matrix, plateau = v.run_parameter_heatmap(_bars([20000.0] * 10), "fast_ma", [2, 3], "slow_ma", [4, 5])
    assert matrix == [[5000.0, 0.0], [0.0, 0.0]]
    assert plateau is False  # 1/4 = 25% < 60%

    profits.update({(3, 4): 100.0, (2, 5): 100.0})
    _, plateau2 = v.run_parameter_heatmap(_bars([20000.0] * 10), "fast_ma", [2, 3], "slow_ma", [4, 5])
    assert plateau2 is True  # 3/4 = 75%

    # 因 x >= y 被跳過的格子不計入分母：有效格只有 (2,3)，且為正報酬
    profits.clear()
    profits[(2, 3)] = 100.0
    _, plateau3 = v.run_parameter_heatmap(_bars([20000.0] * 10), "fast_ma", [2, 3], "slow_ma", [3], )
    assert plateau3 is True

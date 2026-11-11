# tests/test_ai_confidence_and_ensemble.py
from __future__ import annotations
from datetime import datetime, timedelta
import numpy as np
import polars as pl
import pytest

from src.schema.strategy import StrategySchema
from src.core.engine import ContractSpec, EventDrivenEngine
from src.core.indicators import IndicatorEngine
from src.core.validator import CredibilityValidator
from src.core.paper_trader import PaperTradingEngine, RiskConfig
from src.ai.confidence_engine import ConfidenceEngine, MarketRegimeDetector
from src.ai.ensemble import MultiModelEnsemble
from src.ai.strategy_copilot import AIStrategyCopilot


def _make_trend_bars(n: int = 120) -> pl.DataFrame:
    base_dt = datetime(2026, 10, 20, 9, 0, 0)
    timestamps = [base_dt + timedelta(minutes=5 * i) for i in range(n)]
    prices = [20000.0 + i * 6.0 + np.sin(i / 3.0) * 15.0 for i in range(n)]
    return pl.DataFrame({
        "timestamp": timestamps,
        "open": [p - 3.0 for p in prices],
        "high": [p + 12.0 for p in prices],
        "low": [p - 10.0 for p in prices],
        "close": prices,
        "volume": [100 + (i % 7) * 30 for i in range(n)],
    })


def test_feature_engineering_and_confidence_engine():
    df = _make_trend_bars(80)
    feat_df = IndicatorEngine.compute_feature_matrix(df)
    assert "feat_roc_5" in feat_df.columns
    assert "feat_bias_20" in feat_df.columns
    assert "feat_slope_10" in feat_df.columns
    assert "feat_bb_width" in feat_df.columns
    assert "feat_vol_ratio" in feat_df.columns
    assert "feat_accel" in feat_df.columns

    conf_df = ConfidenceEngine.compute_confidence_series(df, base_threshold=60.0, regime_adaptive=True)
    assert "long_confidence" in conf_df.columns
    assert "short_confidence" in conf_df.columns
    assert "effective_threshold" in conf_df.columns
    assert "regime_label" in conf_df.columns

    snap = ConfidenceEngine.get_latest_snapshot(df, base_threshold=60.0, regime_adaptive=True)
    assert snap is not None
    assert 1.0 <= snap.long_confidence <= 99.0
    assert 1.0 <= snap.short_confidence <= 99.0
    assert len(snap.feature_contributions) >= 4


def test_regime_detector_offset():
    strong = MarketRegimeDetector.detect_regime_at(adx_val=32.0, bb_width_val=1.5, median_bb_width=0.8)
    assert strong.regime == "STRONG_TREND"
    assert strong.threshold_offset < 0

    ranging = MarketRegimeDetector.detect_regime_at(adx_val=14.0, bb_width_val=0.4, median_bb_width=0.8)
    assert ranging.regime == "RANGING"
    assert ranging.threshold_offset > 0


def test_multi_model_ensemble_and_threshold_scan():
    df = _make_trend_bars(90)
    contract = ContractSpec(
        symbol="TMF",
        name="微型臺指",
        exchange="TAIFEX",
        point_value=10.0,
        tick_size=1.0,
        tax_rate=0.00002,
        default_commission=12.0,
    )

    for method in ("dynamic_weighted", "average", "voting"):
        dec = MultiModelEnsemble.evaluate_ensemble(
            df, contract=contract, method=method, threshold=60.0, regime_adaptive=True
        )
        assert dec.total_active_models >= 6
        assert dec.final_action in (-1, 0, 1)
        assert 0.0 <= dec.ensemble_long_conf <= 100.0
        assert 0.0 <= dec.ensemble_short_conf <= 100.0
        # 票數只來自各子模型的實際訊號，且三類票數加總等於啟用模型數
        active = [m for m in dec.models if m.enabled]
        assert dec.bull_votes == sum(1 for m in active if m.latest_signal == 1)
        assert dec.bear_votes == sum(1 for m in active if m.latest_signal == -1)
        assert dec.bull_votes + dec.bear_votes + dec.neutral_votes == dec.total_active_models
        # 最終動作必須至少有一個子模型在該方向有實際訊號
        if dec.final_action == 1:
            assert dec.bull_votes >= 1
        elif dec.final_action == -1:
            assert dec.bear_votes >= 1
        assert sum(m.dynamic_weight for m in active) == pytest.approx(1.0, abs=0.01)

    # 使用會頻繁交叉的震盪資料，確保門檻掃描真的有在過濾（避免 0 <= 0 之類的空洞斷言）
    n = 200
    prices = [20000.0 + i * 2.0 + np.sin(i / 3.0) * 60.0 for i in range(n)]
    chop_df = pl.DataFrame({
        "timestamp": [datetime(2026, 10, 20, 9, 0) + timedelta(minutes=5 * i) for i in range(n)],
        "open": [p - 3.0 for p in prices],
        "high": [p + 12.0 for p in prices],
        "low": [p - 10.0 for p in prices],
        "close": prices,
        "volume": [100 + (i % 7) * 30 for i in range(n)],
    })
    strat = StrategySchema.model_validate({
        "id": "scan_test", "version": 1, "timeframe": "5m",
        "indicators": {
            "fast_ma": {"type": "SMA", "source": "close", "length": 3},
            "slow_ma": {"type": "SMA", "source": "close", "length": 8},
        },
        "entry_long": {"cross_above": ["fast_ma", "slow_ma"]},
        "entry_short": {"cross_below": ["fast_ma", "slow_ma"]},
        "exit": {"stop_loss": {"type": "fixed_points", "value": 40.0}},
    })
    thresholds = [1.0, 50.0, 65.0, 80.0, 99.0]
    scan_rows = MultiModelEnsemble.scan_threshold_tradeoff(chop_df, strat, contract, thresholds=thresholds)
    assert [r.threshold for r in scan_rows] == thresholds
    trades = [r.total_trades for r in scan_rows]
    # 門檻越高交易次數單調不增，且低門檻確實有交易、高門檻確實被濾掉
    assert all(a >= b for a, b in zip(trades, trades[1:]))
    assert trades[0] > 0
    assert trades[-1] < trades[0]


def test_walk_forward_and_parameter_wizard():
    df = _make_trend_bars(120)
    contract = ContractSpec(
        symbol="TMF",
        name="微型臺指",
        exchange="TAIFEX",
        point_value=10.0,
        tick_size=1.0,
        tax_rate=0.00002,
        default_commission=12.0,
    )
    strat = StrategySchema.model_validate({
        "id": "wf_test_strat",
        "version": 1,
        "symbol": "TMF",
        "timeframe": "5m",
        "session": "day_and_night",
        "indicators": {
            "fast_ma": {"type": "SMA", "source": "close", "length": 5},
            "slow_ma": {"type": "SMA", "source": "close", "length": 20},
        },
        "entry_long": {"cross_above": ["fast_ma", "slow_ma"]},
        "entry_short": {"cross_below": ["fast_ma", "slow_ma"]},
        "exit": {"stop_loss": {"type": "fixed_points", "value": 50.0}},
    })
    validator = CredibilityValidator(strat, contract, 100000.0)
    wf = validator.run_walk_forward(df, n_splits=3, train_ratio=0.7)
    assert len(wf.folds) == 3

    cands = validator.run_parameter_wizard(
        df,
        ind_x="fast_ma",
        fast_range=[3, 5],
        ind_y="slow_ma",
        slow_range=[15, 20],
        sl_range=[50.0],
        threshold_range=[0.0, 60.0],
        top_k=3,
    )
    assert len(cands) == 3


def test_paper_trader_profit_target_and_degradation_lock():
    contract = ContractSpec(
        symbol="TMF",
        name="微型臺指",
        exchange="TAIFEX",
        point_value=10.0,
        tick_size=1.0,
        tax_rate=0.00002,
        default_commission=12.0,
    )
    strat = StrategySchema.model_validate({
        "id": "pt_adv_test",
        "version": 1,
        "symbol": "TMF",
        "timeframe": "5m",
        "session": "day_and_night",
        "indicators": {
            "fast_ma": {"type": "SMA", "source": "close", "length": 3},
            "slow_ma": {"type": "SMA", "source": "close", "length": 8},
        },
        "entry_long": {"cross_above": ["fast_ma", "slow_ma"]},
        "entry_short": {"cross_below": ["fast_ma", "slow_ma"]},
        "exit": {"stop_loss": {"type": "fixed_points", "value": 40.0}},
    })
    rc = RiskConfig(max_daily_loss=5000.0, daily_profit_target=1000.0, max_consecutive_losses=2)
    pt = PaperTradingEngine(strat, contract, initial_capital=100000.0, risk_config=rc)
    ok, _ = pt.start_session(credibility_passed=True)
    assert ok

    # 測試獲利達標全平鎖定
    pt._open_new_position(1, 20000.0, "2026-10-29 09:05", "entry_long")
    triggered = pt.check_daily_profit_target(current_price=20150.0)
    assert triggered is True
    assert pt.daily_profit_locked is True
    assert pt.position is None

    # 解除鎖定並測試連續虧損退化暫停
    pt.reset_risk_locks()
    pt.start_session(credibility_passed=True)
    pt._open_new_position(1, 20000.0, "2026-10-29 09:10", "entry_long")
    pt._close_current_position(19980.0, exit_idx=1, reason="stop_loss")
    pt._open_new_position(1, 20000.0, "2026-10-29 09:15", "entry_long")
    pt._close_current_position(19980.0, exit_idx=2, reason="stop_loss")

    assert pt.consecutive_losses >= 2
    deg = pt.check_model_degradation()
    assert deg is True
    assert pt.degradation_locked is True


def test_copilot_parses_confidence_threshold():
    res = AIStrategyCopilot.parse_natural_language(
        "當 5MA 向上突破 20MA 且信心度大於 65% 時做多，停損 45 點"
    )
    assert res.strategy.confidence_filter.enabled is True
    assert res.strategy.confidence_filter.threshold == 65.0


def test_watchlist_add_and_remove(tmp_path):
    from src.ui.dashboard import load_watchlist, add_symbol_to_watchlist, remove_symbol_from_watchlist

    wl_file = tmp_path / "watchlist.json"
    init_list = load_watchlist(wl_file)
    assert len(init_list) >= 4

    ok_add, _, after_add = add_symbol_to_watchlist("2603", "長榮", path=wl_file)
    assert ok_add is True
    assert after_add[0] == "2603 (長榮)"

    # 重複新增同代碼應阻擋
    ok_dup, _, _ = add_symbol_to_watchlist("2603", "長榮海運", path=wl_file)
    assert ok_dup is False

    # 移除剛新增的商品
    ok_rm, _, after_rm = remove_symbol_from_watchlist("2603 (長榮)", path=wl_file)
    assert ok_rm is True
    assert "2603 (長榮)" not in after_rm


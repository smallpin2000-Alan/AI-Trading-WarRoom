# src/core/credibility_gate.py
"""
啟動模擬操盤前的「五道可信度檢核」閘門。

把 CredibilityValidator 的結果整理成介面可以直接顯示的摘要，
讓啟動流程以「實際檢核結果」為準，而不是在呼叫端寫死為通過。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Sequence

import polars as pl

from src.core.engine import ContractSpec
from src.core.validator import CredibilityValidator
from src.schema.strategy import StrategySchema

logger = logging.getLogger(__name__)

# 檢核至少需要的 K 棒數（太少時 IS/OOS、Walk-Forward 都沒有統計意義）
MIN_BARS_FOR_CREDIBILITY = 300
# 熱力圖掃描：以策略目前的指標週期為中心，向兩側各取一檔
HEATMAP_SCALE_FACTORS: Sequence[float] = (0.6, 1.0, 1.6)
# 蒙地卡羅重抽次數（啟動時即時執行，取較小值以免卡住畫面）
GATE_MC_ITERATIONS = 300


@dataclass(frozen=True)
class CredibilityGateResult:
    passed: bool
    reasons: List[str] = field(default_factory=list)
    checks: List[tuple[str, bool]] = field(default_factory=list)  # (檢核名稱, 是否通過)
    bars_used: int = 0


def _heatmap_values(center: int) -> List[int]:
    values = sorted({max(2, int(round(center * f))) for f in HEATMAP_SCALE_FACTORS})
    return values


def run_credibility_gate(
    strategy: StrategySchema,
    contract: ContractSpec,
    initial_capital: float,
    df_bars: pl.DataFrame,
) -> CredibilityGateResult:
    """
    對目前策略執行五道可信度檢核。
    任何無法完成檢核的情況（資料不足、指標不足、檢核過程出錯）一律視為「未通過」，絕不預設放行。
    """
    n_bars = df_bars.height if df_bars is not None else 0
    if n_bars < MIN_BARS_FOR_CREDIBILITY:
        return CredibilityGateResult(
            passed=False,
            reasons=[f"K 棒數量不足（{n_bars} 根，至少需要 {MIN_BARS_FOR_CREDIBILITY} 根），無法執行可信度檢核"],
            bars_used=n_bars,
        )

    indicator_names = list(strategy.indicators.keys())
    if len(indicator_names) < 2:
        return CredibilityGateResult(
            passed=False,
            reasons=["策略指標少於 2 個，無法執行雙參數高原檢核"],
            bars_used=n_bars,
        )
    ind_x, ind_y = indicator_names[0], indicator_names[1]

    try:
        validator = CredibilityValidator(strategy, contract, initial_capital)
        result = validator.evaluate_all(
            df_bars,
            ind_x,
            _heatmap_values(int(strategy.indicators[ind_x].length)),
            ind_y,
            _heatmap_values(int(strategy.indicators[ind_y].length)),
            mc_iterations=GATE_MC_ITERATIONS,
        )
    except Exception as ex:
        logger.exception("五道可信度檢核執行失敗")
        return CredibilityGateResult(
            passed=False,
            reasons=[f"可信度檢核執行失敗：{ex}"],
            bars_used=n_bars,
        )

    checks = [
        ("樣本外 (OOS) 獲利", bool(result.oos_passed)),
        ("雙倍成本壓力測試", bool(result.stress_passed)),
        ("參數高原（非孤立尖峰）", bool(result.heatmap_plateau_passed)),
        ("蒙地卡羅模擬", bool(result.monte_carlo_passed)),
        ("Walk-Forward 滾動驗證", bool(result.walk_forward_passed)),
    ]
    return CredibilityGateResult(
        passed=bool(result.overall_passed),
        reasons=list(result.reasons),
        checks=checks,
        bars_used=n_bars,
    )

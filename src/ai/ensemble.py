# src/ai/ensemble.py
from __future__ import annotations
import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Any
import numpy as np
import polars as pl

from src.schema.strategy import StrategySchema
from src.core.engine import ContractSpec, EventDrivenEngine, BacktestReport
from src.core.indicators import IndicatorEngine
from src.ai.confidence_engine import ConfidenceEngine

EnsembleMethod = Literal["voting", "average", "dynamic_weighted"]

logger = logging.getLogger(__name__)

# 動態權重：回測交易筆數低於此門檻的模型沒有足夠績效證據，權重評分一律降到最低
ENSEMBLE_MIN_TRADES_FOR_WEIGHT = 5
# 動態權重評分下限（績效極差或交易數不足的模型）
ENSEMBLE_MIN_PERF_SCORE = 0.1


@dataclass
class SubModelStatus:
    model_id: str
    display_name: str
    strategy: StrategySchema
    report: BacktestReport
    latest_signal: int          # 1: 看多, -1: 看空, 0: 觀望
    long_confidence: float      # 0~100%
    short_confidence: float     # 0~100%
    dynamic_weight: float       # 動態分配權重 (0.0 ~ 1.0)
    enabled: bool = True


@dataclass
class EnsembleDecision:
    method: EnsembleMethod
    final_action: int           # 1: 做多訊號, -1: 做空訊號, 0: 觀望持平
    ensemble_long_conf: float   # 融合後做多信心度 (0~100%)
    ensemble_short_conf: float  # 融合後做空信心度 (0~100%)
    threshold: float
    effective_threshold: float
    regime_label: str
    bull_votes: int
    bear_votes: int
    neutral_votes: int
    total_active_models: int
    models: List[SubModelStatus] = field(default_factory=list)


@dataclass
class ThresholdScanRow:
    threshold: float
    total_trades: int
    win_rate: float
    net_profit: float
    profit_factor: float
    max_drawdown: float
    expectancy: float
    suitability_label: str


class MultiModelEnsemble:
    """
    AITrade 風格多模型並行監控與融合決策引擎 (Multi-Model Matrix & Ensemble)
    支援：
    1. 自動載入 strategies/ 範本庫並衍生多組參數子模型並行運算。
    2. 三種融合模式：
       - 投票制 (voting)：過半或超過指定比例的子模型看多/看空才觸發。
       - 平均制 (average)：所有啟用子模型的信心度取平均，超過有效門檻才觸發。
       - 動態績效加權制 (dynamic_weighted)：依各模型近期回測淨利與獲利因子動態分配權重。
    3. 信心度閾值取捨掃描表 (Threshold Trade-off Table)：驗證 50% ~ 85% 各門檻之頻率與勝率。
    """

    @classmethod
    def build_default_model_pool(
        cls,
        symbol: str = "TMF",
        timeframe: str = "5m",
        session: str = "day_and_night",
    ) -> List[tuple[str, StrategySchema]]:
        pool: List[tuple[str, StrategySchema]] = []

        # 1. 先載入 strategies/ 目錄下的 JSON 策略
        strat_dir = Path("strategies")
        if strat_dir.exists():
            for p in sorted(strat_dir.glob("*.json")):
                try:
                    raw = json.loads(p.read_text(encoding="utf-8"))
                    raw["symbol"] = symbol if symbol in ("TMF", "MXF", "TXF") else "TMF"
                    raw["timeframe"] = timeframe
                    raw["session"] = session
                    s = StrategySchema.model_validate(raw)
                    pool.append((f"範本: {s.id}", s))
                except Exception as exc:
                    # 不可靜默略過：策略檔壞掉或不符合安全規則時，必須讓使用者知道該模型未被載入
                    logger.warning("集成模型池略過策略檔 %s：%s", p.name, exc)

        # 2. 自動衍生多組不同方法論的子模型 (趨勢均線、EMA動能、通道突破、布林通道、RSI/ADX濾網)
        presets = [
            (
                "AI_SMA_5_20_Trend",
                "雙均線順勢 (SMA 5/20)",
                {
                    "fast_ma": {"type": "SMA", "source": "close", "length": 5},
                    "slow_ma": {"type": "SMA", "source": "close", "length": 20},
                },
                {"cross_above": ["fast_ma", "slow_ma"]},
                {"cross_below": ["fast_ma", "slow_ma"]},
                {"stop_loss": {"type": "fixed_points", "value": 50.0}, "take_profit": {"type": "fixed_points", "value": 120.0}},
            ),
            (
                "AI_EMA_8_21_Momentum",
                "指數均線動能 (EMA 8/21)",
                {
                    "ema_f": {"type": "EMA", "source": "close", "length": 8},
                    "ema_s": {"type": "EMA", "source": "close", "length": 21},
                },
                {"cross_above": ["ema_f", "ema_s"]},
                {"cross_below": ["ema_f", "ema_s"]},
                {"stop_loss": {"type": "fixed_points", "value": 45.0}, "trailing_stop": {"type": "points", "value": 40.0, "activation_points": 50.0}},
            ),
            (
                "AI_Channel_15_Breakout",
                "唐奇安通道突破 (15期高低點)",
                {
                    "hh15": {"type": "HIGHEST", "source": "high", "length": 15},
                    "ll15": {"type": "LOWEST", "source": "low", "length": 15},
                },
                {"cross_above": ["close", "hh15"]},
                {"cross_below": ["close", "ll15"]},
                {"stop_loss": {"type": "fixed_points", "value": 55.0}, "trailing_stop": {"type": "points", "value": 45.0, "activation_points": 60.0}},
            ),
            (
                "AI_Bollinger_20_Break",
                "布林通道波動突破 (BB 20)",
                {
                    "bb_u": {"type": "BB_UPPER", "source": "close", "length": 20},
                    "bb_l": {"type": "BB_LOWER", "source": "close", "length": 20},
                },
                {"cross_above": ["close", "bb_u"]},
                {"cross_below": ["close", "bb_l"]},
                {"stop_loss": {"type": "fixed_points", "value": 50.0}, "take_profit": {"type": "fixed_points", "value": 130.0}},
            ),
            (
                "AI_MACD_RSI_Combo",
                "MACD + RSI 雙重動能確認",
                {
                    "macd": {"type": "MACD", "source": "close", "length": 12},
                    "rsi": {"type": "RSI", "source": "close", "length": 14},
                },
                {"all": [{"gt": ["macd", 0.0]}, {"cross_above": ["rsi", 52.0]}]},
                {"all": [{"lt": ["macd", 0.0]}, {"cross_below": ["rsi", 48.0]}]},
                {"stop_loss": {"type": "fixed_points", "value": 40.0}, "take_profit": {"type": "fixed_points", "value": 100.0}},
            ),
            (
                "AI_KD_ADX_Regime",
                "KD 金死叉 + ADX 趨勢強度濾網",
                {
                    "k": {"type": "KD_K", "source": "close", "length": 9},
                    "d": {"type": "KD_D", "source": "close", "length": 9},
                    "adx": {"type": "ADX", "source": "close", "length": 14},
                },
                {"all": [{"cross_above": ["k", "d"]}, {"gt": ["adx", 20.0]}]},
                {"all": [{"cross_below": ["k", "d"]}, {"gt": ["adx", 20.0]}]},
                {"stop_loss": {"type": "fixed_points", "value": 45.0}, "take_profit": {"type": "fixed_points", "value": 110.0}},
            ),
        ]

        existing_ids = {s.id for _, s in pool}
        for mid, label, inds, e_long, e_short, ex_dict in presets:
            if mid in existing_ids:
                continue
            s_obj = StrategySchema.model_validate({
                "id": mid,
                "version": 1,
                "symbol": symbol if symbol in ("TMF", "MXF", "TXF") else "TMF",
                "timeframe": timeframe,
                "session": session,
                "indicators": inds,
                "entry_long": e_long,
                "entry_short": e_short,
                "exit": ex_dict,
            })
            pool.append((label, s_obj))

        return pool

    @classmethod
    def evaluate_ensemble(
        cls,
        df: pl.DataFrame,
        contract: ContractSpec,
        initial_capital: float = 100_000.0,
        method: EnsembleMethod = "dynamic_weighted",
        threshold: float = 60.0,
        regime_adaptive: bool = True,
        enabled_model_ids: Optional[List[str]] = None,
        custom_pool: Optional[List[tuple[str, StrategySchema]]] = None,
    ) -> EnsembleDecision:
        if df.height == 0:
            return EnsembleDecision(
                method=method,
                final_action=0,
                ensemble_long_conf=50.0,
                ensemble_short_conf=50.0,
                threshold=threshold,
                effective_threshold=threshold,
                regime_label="資料載入中",
                bull_votes=0,
                bear_votes=0,
                neutral_votes=0,
                total_active_models=0,
                models=[],
            )

        pool = custom_pool or cls.build_default_model_pool(
            symbol=contract.symbol if contract.symbol in ("TMF", "MXF", "TXF") else "TMF"
        )
        conf_df = ConfidenceEngine.compute_confidence_series(
            df, base_threshold=threshold, regime_adaptive=regime_adaptive
        )
        last_idx = conf_df.height - 1
        base_long_c = float(conf_df["long_confidence"][last_idx])
        base_short_c = float(conf_df["short_confidence"][last_idx])
        eff_th = float(conf_df["effective_threshold"][last_idx])
        regime_lbl = str(conf_df["regime_label"][last_idx])

        sub_models: List[SubModelStatus] = []
        raw_scores: List[float] = []

        for disp_name, strat in pool:
            eng = EventDrivenEngine(strat, contract, initial_capital)
            try:
                rep = eng.run_backtest(df, cost_multiplier=1.0)

                ind_specs = {k: v.model_dump() for k, v in strat.indicators.items()}
                df_ind = IndicatorEngine.compute_indicators(df, ind_specs, base_timeframe=strat.timeframe)
            except ValueError as exc:
                # 例如多週期指標無法由目前資料聚合：略過該模型並留下紀錄，不讓單一模型拖垮整個集成
                logger.warning("集成略過子模型 %s（%s）：%s", strat.id, disp_name, exc)
                continue
            cols = {col: df_ind[col].to_list() for col in df_ind.columns}

            # 目前訊號：當根 K 棒的進場條件成立，或回測到資料結束時仍持有部位 (end_of_data)。
            # 已經出場（停損 / 停利 / 反手）的交易方向不代表目前看法，不可當成訊號。
            sig = 0
            if last_idx >= 1:
                if eng._eval_condition(strat.entry_long, cols, last_idx):
                    sig = 1
                elif eng._eval_condition(strat.entry_short, cols, last_idx):
                    sig = -1
                elif len(rep.trades) > 0 and rep.trades[-1].exit_reason == "end_of_data":
                    sig = rep.trades[-1].direction

            # 子模型個別信心度：融合特徵基礎信心度與該模型近期勝率/訊號方向
            win_bonus = (rep.win_rate - 50.0) * 0.25 if rep.total_trades > 0 else 0.0
            if sig == 1:
                m_long = float(np.clip(base_long_c + 12.0 + win_bonus, 5.0, 99.0))
                m_short = float(np.clip(100.0 - m_long, 1.0, 95.0))
            elif sig == -1:
                m_short = float(np.clip(base_short_c + 12.0 + win_bonus, 5.0, 99.0))
                m_long = float(np.clip(100.0 - m_short, 1.0, 95.0))
            else:
                m_long = float(np.clip(base_long_c + win_bonus * 0.5, 5.0, 95.0))
                m_short = float(np.clip(base_short_c - win_bonus * 0.5, 5.0, 95.0))

            is_en = True if enabled_model_ids is None else (strat.id in enabled_model_ids)
            # 績效評分供動態權重使用
            pf_cap = min(rep.profit_factor, 4.0) if rep.profit_factor != float("inf") else 3.0
            if rep.total_trades < ENSEMBLE_MIN_TRADES_FOR_WEIGHT:
                # 交易數不足（含零交易）：沒有績效證據，不可因「沒虧錢」而拿到中等權重
                perf_score = ENSEMBLE_MIN_PERF_SCORE
            else:
                perf_score = max(
                    ENSEMBLE_MIN_PERF_SCORE,
                    (1.0 + rep.net_profit / max(10000.0, initial_capital)) * (0.5 + 0.3 * pf_cap),
                )
            raw_scores.append(perf_score if is_en else 0.0)

            sub_models.append(
                SubModelStatus(
                    model_id=strat.id,
                    display_name=disp_name,
                    strategy=strat,
                    report=rep,
                    latest_signal=sig,
                    long_confidence=round(m_long, 1),
                    short_confidence=round(m_short, 1),
                    dynamic_weight=0.0,
                    enabled=is_en,
                )
            )

        # 正規化動態權重并依回測淨損益由高至低排序
        total_score = sum(raw_scores)
        for idx_m, m in enumerate(sub_models):
            if m.enabled and total_score > 0:
                m.dynamic_weight = round(raw_scores[idx_m] / total_score, 3)
            else:
                m.dynamic_weight = 0.0

        sub_models.sort(key=lambda x: x.report.net_profit, reverse=True)
        active_models = [m for m in sub_models if m.enabled]
        n_active = len(active_models)

        if n_active == 0:
            return EnsembleDecision(
                method=method,
                final_action=0,
                ensemble_long_conf=50.0,
                ensemble_short_conf=50.0,
                threshold=threshold,
                effective_threshold=eff_th,
                regime_label=regime_lbl,
                bull_votes=0,
                bear_votes=0,
                neutral_votes=0,
                total_active_models=0,
                models=sub_models,
            )

        # 投票只計各子模型「實際的」訊號。各模型的信心度都源自同一份基礎信心度，
        # 若以信心度達標計票，等於同一個意見被重複計算 N 次，並非獨立投票。
        bull_votes = sum(1 for m in active_models if m.latest_signal == 1)
        bear_votes = sum(1 for m in active_models if m.latest_signal == -1)
        neutral_votes = max(0, n_active - bull_votes - bear_votes)

        if method == "average":
            ens_long = float(np.mean([m.long_confidence for m in active_models]))
            ens_short = float(np.mean([m.short_confidence for m in active_models]))
        elif method == "voting":
            ens_long = (bull_votes / n_active) * 100.0
            ens_short = (bear_votes / n_active) * 100.0
        else:
            # dynamic_weighted
            w_sum = sum(m.dynamic_weight for m in active_models) or 1.0
            ens_long = sum(m.long_confidence * m.dynamic_weight for m in active_models) / w_sum
            ens_short = sum(m.short_confidence * m.dynamic_weight for m in active_models) / w_sum

        ens_long = round(float(ens_long), 1)
        ens_short = round(float(ens_short), 1)

        final_act = 0
        if method == "voting":
            if bull_votes > n_active / 2.0 and bull_votes > bear_votes:
                final_act = 1
            elif bear_votes > n_active / 2.0 and bear_votes > bull_votes:
                final_act = -1
        else:
            # 除了融合信心度達標，還必須至少有一個子模型在該方向有實際訊號
            if ens_long >= eff_th and ens_long > ens_short and bull_votes >= 1:
                final_act = 1
            elif ens_short >= eff_th and ens_short > ens_long and bear_votes >= 1:
                final_act = -1

        return EnsembleDecision(
            method=method,
            final_action=final_act,
            ensemble_long_conf=ens_long,
            ensemble_short_conf=ens_short,
            threshold=threshold,
            effective_threshold=eff_th,
            regime_label=regime_lbl,
            bull_votes=bull_votes,
            bear_votes=bear_votes,
            neutral_votes=neutral_votes,
            total_active_models=n_active,
            models=sub_models,
        )

    @classmethod
    def scan_threshold_tradeoff(
        cls,
        df: pl.DataFrame,
        strategy: StrategySchema,
        contract: ContractSpec,
        initial_capital: float = 100_000.0,
        thresholds: Optional[List[float]] = None,
        regime_adaptive: bool = True,
    ) -> List[ThresholdScanRow]:
        """
        對照 AITrade 文章「閾值高低的取捨表」：
        實測不同信心度門檻 (如 50%, 55%, 60%, 65%, 70%, 75%, 80%) 下的訊號頻率、單筆勝率與總淨利。
        """
        if thresholds is None:
            thresholds = [50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0]

        rows: List[ThresholdScanRow] = []
        for th in thresholds:
            s_copy = copy.deepcopy(strategy)
            s_copy.confidence_filter.enabled = True
            s_copy.confidence_filter.threshold = float(th)
            s_copy.confidence_filter.regime_adaptive = regime_adaptive

            eng = EventDrivenEngine(s_copy, contract, initial_capital)
            rep = eng.run_backtest(df, cost_multiplier=1.0)

            if th <= 55.0:
                suit = "低門檻 (高頻率、需低手續費)"
            elif th <= 65.0:
                suit = "中門檻 (主流設定、風險報酬平衡)"
            elif th <= 75.0:
                suit = "高門檻 (選擇性下單、傾向波段)"
            else:
                suit = "極高門檻 (機會少但精準、適合小資金)"

            rows.append(
                ThresholdScanRow(
                    threshold=th,
                    total_trades=rep.total_trades,
                    win_rate=rep.win_rate,
                    net_profit=rep.net_profit,
                    profit_factor=rep.profit_factor,
                    max_drawdown=rep.max_drawdown,
                    expectancy=rep.expectancy,
                    suitability_label=suit,
                )
            )
        return rows

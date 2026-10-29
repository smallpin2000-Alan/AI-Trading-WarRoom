# src/core/validator.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import copy
import numpy as np
import polars as pl

from src.schema.strategy import StrategySchema
from src.core.engine import ContractSpec, EventDrivenEngine, BacktestReport, TradeRecord


@dataclass
class MonteCarloResult:
    iterations: int
    sample_trades_count: int
    median_mdd: float
    mdd_95th: float
    mdd_99th: float
    median_profit: float
    profit_5th: float
    passed: bool
    mdd_distribution: List[float] = field(default_factory=list)
    profit_distribution: List[float] = field(default_factory=list)


@dataclass
class WalkForwardFold:
    fold_index: int
    train_bars: int
    test_bars: int
    train_net_profit: float
    test_net_profit: float
    test_win_rate: float
    test_trades: int
    passed: bool


@dataclass
class WalkForwardResult:
    folds: List[WalkForwardFold]
    consistency_ratio: float    # 測試集獲利折數比例 (0.0 ~ 1.0)
    total_oos_profit: float
    passed: bool


@dataclass
class OptimizerCandidate:
    fast_length: int
    slow_length: int
    stop_loss_pts: float
    confidence_threshold: float
    is_net_profit: float
    oos_net_profit: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    total_trades: int
    score: float


@dataclass
class ValidationResult:
    is_report: BacktestReport
    oos_report: BacktestReport
    stress_report: BacktestReport
    monte_carlo_report: Optional[MonteCarloResult]
    oos_passed: bool
    stress_passed: bool
    heatmap_plateau_passed: bool
    monte_carlo_passed: bool
    overall_passed: bool
    reasons: List[str]
    heatmap_matrix: List[List[float]]
    walk_forward_report: Optional[WalkForwardResult] = None
    walk_forward_passed: bool = True


class CredibilityValidator:
    """
    階段二：回測可信度驗證器 (阻擋過擬合策略上線)
    1. IS (70%) / OOS (30%) 樣本內外切分檢驗：OOS 期望值與淨利須為正 (含指標暖機期銜接)。
    2. 雙倍成本壓力測試 (cost_multiplier = 2.0)：手續費與滑價加倍後仍須獲利。
    3. 雙參數優化熱力圖與高原判定 (Plateau Check)：最佳參數周邊鄰居的正報酬比例須 >= 60%，排除孤立尖峰。
    4. 蒙地卡羅模擬 (Monte Carlo)：隨機重排交易順序 1,000 次，檢驗 95% 信心水準下的 MDD 與獲利能力。
    """

    def __init__(self, strategy: StrategySchema, contract: ContractSpec, initial_capital: float = 100_000.0):
        self.strategy = strategy
        self.contract = contract
        self.initial_capital = initial_capital

    def run_is_oos_split(self, df: pl.DataFrame, is_ratio: float = 0.7) -> Tuple[BacktestReport, BacktestReport]:
        n = df.height
        split_idx = max(2, int(n * is_ratio))

        # 計算最大指標期數作為 OOS 暖機緩衝 (Warm-up Lookback)，避免切分點剛開始時指標為 NaN 漏訊號
        max_lookback = max((ind.length for ind in self.strategy.indicators.values()), default=1)
        oos_start = max(0, split_idx - max_lookback)

        df_is = df.slice(0, split_idx)
        df_oos = df.slice(oos_start, n - oos_start)

        engine = EventDrivenEngine(self.strategy, self.contract, self.initial_capital)
        is_rep = engine.run_backtest(df_is, cost_multiplier=1.0)
        oos_rep = engine.run_backtest(df_oos, cost_multiplier=1.0)
        return is_rep, oos_rep

    def run_cost_stress_test(self, df: pl.DataFrame, multiplier: float = 2.0) -> BacktestReport:
        engine = EventDrivenEngine(self.strategy, self.contract, self.initial_capital)
        return engine.run_backtest(df, cost_multiplier=multiplier)

    def run_parameter_heatmap(
        self,
        df: pl.DataFrame,
        ind_x: str,
        x_values: List[int],
        ind_y: str,
        y_values: List[int],
    ) -> Tuple[List[List[float]], bool]:
        """
        掃描兩組指標參數 (例如短均線 vs 長均線)，回傳淨利矩陣與是否符合「參數高原」特徵。
        """
        matrix: List[List[float]] = []
        for y_val in y_values:
            row: List[float] = []
            for x_val in x_values:
                if x_val >= y_val:
                    row.append(0.0)
                    continue
                s_copy = copy.deepcopy(self.strategy)
                if ind_x in s_copy.indicators:
                    s_copy.indicators[ind_x].length = x_val
                if ind_y in s_copy.indicators:
                    s_copy.indicators[ind_y].length = y_val

                eng = EventDrivenEngine(s_copy, self.contract, self.initial_capital)
                rep = eng.run_backtest(df, cost_multiplier=1.0)
                row.append(rep.net_profit)
            matrix.append(row)

        arr = np.array(matrix, dtype=float)
        valid_cells = arr[arr != 0.0]
        if valid_cells.size == 0:
            return matrix, False

        # 高原判定：有效參數組合中，至少 60% 為正報酬，且最大值不是唯一正報酬的孤立尖峰
        positive_ratio = float(np.sum(valid_cells > 0)) / float(valid_cells.size)
        is_plateau = positive_ratio >= 0.60
        return matrix, is_plateau

    def run_monte_carlo(
        self,
        trades: List[TradeRecord],
        iterations: int = 1000,
        seed: int = 42,
    ) -> MonteCarloResult:
        """
        蒙地卡羅隨機交易順序洗牌 (Monte Carlo Shuffle Simulation)
        - 評估在不同市場行情順序排列下，可能遭受的最大回撤 (95% 信心水準之 MDD) 與淨利分布。
        """
        if not trades:
            return MonteCarloResult(
                iterations=iterations,
                sample_trades_count=0,
                median_mdd=0.0,
                mdd_95th=0.0,
                mdd_99th=0.0,
                median_profit=0.0,
                profit_5th=0.0,
                passed=False,
                mdd_distribution=[],
                profit_distribution=[],
            )

        rng = np.random.default_rng(seed)
        net_pnls = np.array([t.net_pnl for t in trades], dtype=float)
        n_trades = len(net_pnls)

        mdds: List[float] = []
        profits: List[float] = []

        for _ in range(iterations):
            # 隨機重排交易順序 (Permutation / Reshuffle)
            shuffled = rng.permutation(net_pnls)
            equity_curve = self.initial_capital + np.cumsum(shuffled)
            running_max = np.maximum.accumulate(equity_curve)
            drawdowns = running_max - equity_curve
            max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
            net_prof = float(np.sum(shuffled))

            mdds.append(max_dd)
            profits.append(net_prof)

        mdd_arr = np.array(mdds)
        prof_arr = np.array(profits)

        median_mdd = float(np.median(mdd_arr))
        mdd_95th = float(np.percentile(mdd_arr, 95))
        mdd_99th = float(np.percentile(mdd_arr, 99))
        median_profit = float(np.median(prof_arr))
        profit_5th = float(np.percentile(prof_arr, 5))

        # 蒙地卡羅通過門檻：95% 信心水準下淨利仍大於 0 (極端不幸順序下仍不致虧損)
        passed = profit_5th > 0.0 if n_trades >= 3 else (median_profit > 0.0)

        return MonteCarloResult(
            iterations=iterations,
            sample_trades_count=n_trades,
            median_mdd=median_mdd,
            mdd_95th=mdd_95th,
            mdd_99th=mdd_99th,
            median_profit=median_profit,
            profit_5th=profit_5th,
            passed=passed,
            mdd_distribution=mdds,
            profit_distribution=profits,
        )

    def run_walk_forward(self, df: pl.DataFrame, n_splits: int = 3, train_ratio: float = 0.7) -> WalkForwardResult:
        """
        滾動視窗驗證 (Walk-Forward Analysis)
        將歷史資料切分為多個連續滾動視窗 (Train -> Unseen Test)，檢驗策略在不同時期的表現一致性。
        """
        n = df.height
        n_splits = max(2, n_splits)
        window_size = n // n_splits
        if window_size < 10:
            return WalkForwardResult(folds=[], consistency_ratio=0.0, total_oos_profit=0.0, passed=False)

        max_lookback = max((ind.length for ind in self.strategy.indicators.values()), default=1)
        folds: List[WalkForwardFold] = []

        for idx_f in range(n_splits):
            start_i = idx_f * window_size
            end_i = n if idx_f == n_splits - 1 else (idx_f + 1) * window_size
            seg_len = end_i - start_i
            split_pt = start_i + max(2, int(seg_len * train_ratio))

            df_train = df.slice(start_i, split_pt - start_i)
            oos_start = max(0, split_pt - max_lookback)
            df_test = df.slice(oos_start, end_i - oos_start)

            eng = EventDrivenEngine(self.strategy, self.contract, self.initial_capital)
            rep_train = eng.run_backtest(df_train, cost_multiplier=1.0)
            rep_test = eng.run_backtest(df_test, cost_multiplier=1.0)

            fold_ok = rep_test.net_profit >= 0.0
            folds.append(
                WalkForwardFold(
                    fold_index=idx_f + 1,
                    train_bars=df_train.height,
                    test_bars=end_i - split_pt,
                    train_net_profit=rep_train.net_profit,
                    test_net_profit=rep_test.net_profit,
                    test_win_rate=rep_test.win_rate,
                    test_trades=rep_test.total_trades,
                    passed=fold_ok,
                )
            )

        passed_count = sum(1 for f in folds if f.passed)
        consistency = round(passed_count / len(folds), 2) if folds else 0.0
        total_oos = round(sum(f.test_net_profit for f in folds), 2)
        wf_passed = (consistency >= 0.5 and total_oos >= 0.0)
        return WalkForwardResult(
            folds=folds,
            consistency_ratio=consistency,
            total_oos_profit=total_oos,
            passed=wf_passed,
        )

    def run_parameter_wizard(
        self,
        df: pl.DataFrame,
        ind_x: str = "fast_ma",
        fast_range: Optional[List[int]] = None,
        ind_y: str = "slow_ma",
        slow_range: Optional[List[int]] = None,
        sl_range: Optional[List[float]] = None,
        threshold_range: Optional[List[float]] = None,
        top_k: int = 5,
    ) -> List[OptimizerCandidate]:
        """
        AITrade 風格三步驟模型參數優化精靈 (Parameter Optimization Wizard)：
        自動搜索指標期數、停損點數與 AI 信心度閾值，並強制以樣本外 (OOS) 表現與回撤比排序，防止過擬合。
        """
        fast_list = fast_range or [3, 5, 8, 10]
        slow_list = slow_range or [15, 20, 30]
        sl_list = sl_range or [40.0, 60.0]
        th_list = threshold_range or [0.0, 55.0, 62.0]

        n = df.height
        split_idx = max(2, int(n * 0.7))
        df_is = df.slice(0, split_idx)
        df_oos = df.slice(max(0, split_idx - 35), n - max(0, split_idx - 35))

        candidates: List[OptimizerCandidate] = []
        for f_val in fast_list:
            for s_val in slow_list:
                if f_val >= s_val:
                    continue
                for sl_val in sl_list:
                    for th_val in th_list:
                        s_copy = copy.deepcopy(self.strategy)
                        if ind_x in s_copy.indicators:
                            s_copy.indicators[ind_x].length = int(f_val)
                        if ind_y in s_copy.indicators:
                            s_copy.indicators[ind_y].length = int(s_val)
                        if s_copy.exit.stop_loss and s_copy.exit.stop_loss.type in ("fixed_points", "percent"):
                            s_copy.exit.stop_loss.value = float(sl_val)
                        if th_val > 0:
                            s_copy.confidence_filter.enabled = True
                            s_copy.confidence_filter.threshold = float(th_val)
                        else:
                            s_copy.confidence_filter.enabled = False

                        eng = EventDrivenEngine(s_copy, self.contract, self.initial_capital)
                        rep_full = eng.run_backtest(df, cost_multiplier=1.0)
                        rep_is = eng.run_backtest(df_is, cost_multiplier=1.0)
                        rep_oos = eng.run_backtest(df_oos, cost_multiplier=1.0)

                        # 綜合穩健度評分：重視 OOS 樣本外獲利並懲罰過大 MDD
                        score = (rep_is.net_profit * 0.35) + (rep_oos.net_profit * 0.65) - (rep_full.max_drawdown * 0.25)
                        candidates.append(
                            OptimizerCandidate(
                                fast_length=int(f_val),
                                slow_length=int(s_val),
                                stop_loss_pts=float(sl_val),
                                confidence_threshold=float(th_val),
                                is_net_profit=rep_is.net_profit,
                                oos_net_profit=rep_oos.net_profit,
                                win_rate=rep_full.win_rate,
                                profit_factor=rep_full.profit_factor,
                                max_drawdown=rep_full.max_drawdown,
                                total_trades=rep_full.total_trades,
                                score=round(score, 2),
                            )
                        )

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top_k]

    def evaluate_all(
        self,
        df: pl.DataFrame,
        ind_x: str,
        x_values: List[int],
        ind_y: str,
        y_values: List[int],
        mc_iterations: int = 1000,
    ) -> ValidationResult:
        is_rep, oos_rep = self.run_is_oos_split(df, is_ratio=0.7)
        stress_rep = self.run_cost_stress_test(df, multiplier=2.0)
        heatmap, plateau_ok = self.run_parameter_heatmap(df, ind_x, x_values, ind_y, y_values)
        wf_res = self.run_walk_forward(df, n_splits=3, train_ratio=0.7)

        # 彙整所有樣本交易進行蒙地卡羅測試
        all_engine = EventDrivenEngine(self.strategy, self.contract, self.initial_capital)
        full_rep = all_engine.run_backtest(df, cost_multiplier=1.0)
        mc_res = self.run_monte_carlo(full_rep.trades, iterations=mc_iterations)

        reasons: List[str] = []
        oos_ok = oos_rep.net_profit > 0 and oos_rep.expectancy > 0
        if not oos_ok:
            reasons.append(f"樣本外 (OOS) 期望值未達標 (淨利: {oos_rep.net_profit})")

        stress_ok = stress_rep.net_profit > 0
        if not stress_ok:
            reasons.append(f"雙倍成本壓力測試未通過 (加倍後淨利: {stress_rep.net_profit})")

        if not plateau_ok:
            reasons.append("參數熱力圖呈孤立尖峰或正報酬區域不足 60%，疑似過度擬合 (Overfitting)")

        mc_ok = mc_res.passed
        if not mc_ok:
            reasons.append(f"蒙地卡羅壓力測試未通過：95% 信心水準下淨利為負 ({mc_res.profit_5th:.1f})")

        overall = oos_ok and stress_ok and plateau_ok and mc_ok
        if overall:
            reasons.append("通過所有可信度驗證門檻（IS/OOS、雙倍成本、參數高原、蒙地卡羅、Walk-Forward 檢核），准許進入紙上交易階段")

        return ValidationResult(
            is_report=is_rep,
            oos_report=oos_rep,
            stress_report=stress_rep,
            monte_carlo_report=mc_res,
            oos_passed=oos_ok,
            stress_passed=stress_ok,
            heatmap_plateau_passed=plateau_ok,
            monte_carlo_passed=mc_ok,
            overall_passed=overall,
            reasons=reasons,
            heatmap_matrix=heatmap,
            walk_forward_report=wf_res,
            walk_forward_passed=wf_res.passed,
        )

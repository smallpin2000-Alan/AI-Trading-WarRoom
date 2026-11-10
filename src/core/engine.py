# src/core/engine.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json
from pathlib import Path
import polars as pl
import numpy as np

from src.schema.strategy import (
    StrategySchema,
    ConditionNode,
    LogicalGroup,
    LeafCondition,
    BarsSinceExpr,
)
from src.core.indicators import IndicatorEngine
from src.ai.confidence_engine import ConfidenceEngine
from src.core.exits import (
    add_exit_atr_columns,
    read_atr_values,
    can_open_position,
    compute_exit_levels,
    compute_contracts,
    settlement_cutoff_state,
)


@dataclass
class ContractSpec:
    symbol: str
    name: str
    exchange: str
    point_value: float
    tick_size: float
    tax_rate: float
    default_commission: float
    initial_margin: float = 16100.0
    maintenance_margin: float = 12400.0
    settlement_rule: str = "third_wednesday"

    @classmethod
    def from_json(cls, symbol: str, json_path: str | Path = "config/contracts.json") -> ContractSpec:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if symbol not in data:
            raise ValueError(f"商品規格檔中找不到代碼: {symbol}")
        spec = data[symbol]
        return cls(
            symbol=spec.get("symbol", symbol),
            name=spec["name"],
            exchange=spec["exchange"],
            point_value=float(spec["point_value"]),
            tick_size=float(spec["tick_size"]),
            tax_rate=float(spec["tax_rate"]),
            default_commission=float(spec["default_commission"]),
            initial_margin=float(spec.get("initial_margin", 16100)),
            maintenance_margin=float(spec.get("maintenance_margin", 12400)),
            settlement_rule=spec.get("settlement_rule", "third_wednesday"),
        )


@dataclass
class TradeRecord:
    direction: int          # 1: Long, -1: Short
    contracts: int
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    gross_pnl: float
    commission: float
    tax: float
    net_pnl: float
    exit_reason: str        # "stop_loss", "take_profit", "trailing_stop", "indicator_exit", "reverse", "end_of_data"
    entry_time: str = ""    # 格式 "%Y-%m-%d %H:%M"
    exit_time: str = ""     # 格式 "%Y-%m-%d %H:%M"


@dataclass
class BacktestReport:
    trades: List[TradeRecord]
    equity_curve: List[float]
    net_profit: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    win_rate: float
    max_drawdown: float
    max_drawdown_pct: float
    total_trades: int
    expectancy: float


class EventDrivenEngine:
    """
    單一事件驅動核心引擎 (回測 / 紙上交易 / 實盤共用評估邏輯)
    嚴格執行：
    1. K 棒收盤確認訊號 (signal_on = bar_close)，下一根開盤成交 (fill_at = next_bar_open)，杜絕未來函數。
    2. 跳空穿價真實模擬 (Gap Execution Rule)：
       - 多單停損觸發時，成交價 = min(停損價, 當根開盤價) - 滑價
       - 空單停損觸發時，成交價 = max(停損價, 當根開盤價) + 滑價
    3. 保守原則：同一根 K 棒若同時滿足停損與停利，優先判定停損。
    """

    def __init__(self, strategy: StrategySchema, contract: ContractSpec, initial_capital: float = 100_000.0):
        self.strategy = strategy
        self.contract = contract
        self.initial_capital = initial_capital

    @staticmethod
    def _get_val(row_dict: Dict[str, List[Any]], key: str | float | int, idx: int) -> float:
        if isinstance(key, (int, float)):
            return float(key)
        val = row_dict[key][idx]
        return float("nan") if val is None else float(val)

    def _eval_condition(self, node: Optional[ConditionNode], cols: Dict[str, List[Any]], idx: int) -> bool:
        if node is None or idx < 1:
            return False

        if isinstance(node, LogicalGroup):
            if node.all is not None:
                return all(self._eval_condition(child, cols, idx) for child in node.all)
            if node.any is not None:
                return any(self._eval_condition(child, cols, idx) for child in node.any)
            return False

        if isinstance(node, LeafCondition):
            if node.cross_above is not None:
                a_key, b_key = node.cross_above
                a_curr, a_prev = self._get_val(cols, a_key, idx), self._get_val(cols, a_key, idx - 1)
                b_curr, b_prev = self._get_val(cols, b_key, idx), self._get_val(cols, b_key, idx - 1)
                if np.isnan(a_curr) or np.isnan(a_prev) or np.isnan(b_curr) or np.isnan(b_prev):
                    return False
                return (a_prev <= b_prev) and (a_curr > b_curr)

            if node.cross_below is not None:
                a_key, b_key = node.cross_below
                a_curr, a_prev = self._get_val(cols, a_key, idx), self._get_val(cols, a_key, idx - 1)
                b_curr, b_prev = self._get_val(cols, b_key, idx), self._get_val(cols, b_key, idx - 1)
                if np.isnan(a_curr) or np.isnan(a_prev) or np.isnan(b_curr) or np.isnan(b_prev):
                    return False
                return (a_prev >= b_prev) and (a_curr < b_curr)

            for op_name, op_pair in [
                ("gt", node.gt),
                ("gte", node.gte),
                ("lt", node.lt),
                ("lte", node.lte),
            ]:
                if op_pair is not None:
                    lhs_raw, rhs_raw = op_pair
                    lhs = self._eval_operand(lhs_raw, cols, idx)
                    rhs = self._eval_operand(rhs_raw, cols, idx)
                    if np.isnan(lhs) or np.isnan(rhs):
                        return False
                    if op_name == "gt":
                        return lhs > rhs
                    if op_name == "gte":
                        return lhs >= rhs
                    if op_name == "lt":
                        return lhs < rhs
                    if op_name == "lte":
                        return lhs <= rhs
        return False

    def _eval_operand(self, op: Any, cols: Dict[str, List[Any]], idx: int) -> float:
        if isinstance(op, (int, float)):
            return float(op)
        if isinstance(op, str):
            return self._get_val(cols, op, idx)
        if isinstance(op, BarsSinceExpr):
            for back in range(0, idx + 1):
                if self._eval_condition(op.bars_since, cols, idx - back):
                    return float(back)
            return float("inf")
        return float("nan")

    def _calc_trade_Record(
        self,
        direction: int,
        contracts: int,
        entry_idx: int,
        exit_idx: int,
        entry_price: float,
        exit_price: float,
        reason: str,
        cost_multiplier: float = 1.0,
    ) -> TradeRecord:
        pv = self.contract.point_value
        gross_pnl = (exit_price - entry_price) * direction * pv * contracts
        comm = self.contract.default_commission * 2 * contracts * cost_multiplier
        tax_entry = round(entry_price * pv * contracts * self.contract.tax_rate)
        tax_exit = round(exit_price * pv * contracts * self.contract.tax_rate)
        tax = float(tax_entry + tax_exit)
        net_pnl = gross_pnl - comm - tax
        return TradeRecord(
            direction=direction,
            contracts=contracts,
            entry_idx=entry_idx,
            exit_idx=exit_idx,
            entry_price=entry_price,
            exit_price=exit_price,
            gross_pnl=gross_pnl,
            commission=comm,
            tax=tax,
            net_pnl=net_pnl,
            exit_reason=reason,
        )

    def run_backtest(self, df: pl.DataFrame, cost_multiplier: float = 1.0) -> BacktestReport:
        # 1. 預先計算指標與 AI 信心度序列
        ind_specs = {k: v.model_dump() for k, v in self.strategy.indicators.items()}
        df_ind = IndicatorEngine.compute_indicators(df, ind_specs)
        cf_spec = self.strategy.confidence_filter
        if cf_spec.enabled:
            df_ind = ConfidenceEngine.compute_confidence_series(
                df_ind, base_threshold=cf_spec.threshold, regime_adaptive=cf_spec.regime_adaptive
            )
        # 補齊 ATR 出場相關欄位
        df_ind = add_exit_atr_columns(df_ind, self.strategy)
        cols = {col: df_ind[col].to_list() for col in df_ind.columns}
        n_bars = df_ind.height

        slippage = self.strategy.execution.slippage_points * cost_multiplier
        default_contracts = self.strategy.position_sizing.contracts

        pos_dir = 0           # 0: 空手, 1: 多單, -1: 空單
        contracts = default_contracts
        entry_price = 0.0
        entry_idx = 0
        highest_since_entry = 0.0
        lowest_since_entry = float("inf")
        entry_atr: Dict[int, float] = {}

        pending_signal: Optional[tuple[int, str]] = None  # (target_dir, reason)
        trades: List[TradeRecord] = []
        equity_curve: List[float] = []
        realized_equity = self.initial_capital

        sl_spec = self.strategy.exit.stop_loss

        for i in range(n_bars):
            o = float(cols["open"][i])
            h = float(cols["high"][i])
            l = float(cols["low"][i])
            c = float(cols["close"][i])
            bar_ts = cols["timestamp"][i] if "timestamp" in cols else None

            # Step A: 在當根 K 棒開盤執行前一根收盤產生的訊號 (fill_at = next_bar_open)
            if pending_signal is not None:
                target_dir, sig_reason = pending_signal
                pending_signal = None

                # 先平掉原有部位
                if pos_dir != 0:
                    fill_exit = o - slippage if pos_dir == 1 else o + slippage
                    tr = self._calc_trade_Record(
                        pos_dir, contracts, entry_idx, i, entry_price, fill_exit, sig_reason, cost_multiplier
                    )
                    trades.append(tr)
                    realized_equity += tr.net_pnl
                    pos_dir = 0

                # 若目標為建新倉 (1 或 -1)
                if target_dir != 0:
                    cand_entry_price = o + slippage if target_dir == 1 else o - slippage
                    # 讀取進場前一根收盤 (或當根) 之 ATR
                    atr_now = read_atr_values(cols, self.strategy, max(0, i - 1))
                    can_open, err_msg = can_open_position(self.strategy, cand_entry_price, atr_now)
                    if can_open:
                        calc_contracts = compute_contracts(
                            self.strategy,
                            realized_equity,
                            self.contract.point_value,
                            cand_entry_price,
                            atr_now,
                        )
                        if calc_contracts > 0:
                            pos_dir = target_dir
                            contracts = calc_contracts
                            entry_idx = i
                            entry_price = cand_entry_price
                            highest_since_entry = entry_price
                            lowest_since_entry = entry_price
                            entry_atr = atr_now

            # Step B: 若持有部位，檢查當根 K 棒盤中是否觸發停損 / 移動停損 / 停利 (含跳空穿價規則)
            if pos_dir != 0:
                current_atr = read_atr_values(cols, self.strategy, max(0, i - 1))
                stop_price, tp_price, stop_reason = compute_exit_levels(
                    self.strategy,
                    pos_dir,
                    entry_price,
                    highest_since_entry,
                    lowest_since_entry,
                    entry_atr,
                    current_atr,
                )

                # 檢查是否觸發停損 (含 Gap Execution Rule 跳空穿價)
                stopped_out = False
                if stop_price is not None:
                    if pos_dir == 1 and l <= stop_price:
                        # 多單停損：若開盤直接跳空低於停損價，以 min(stop_price, open) - 滑價 成交
                        raw_exit = min(stop_price, o)
                        fill_exit = raw_exit - slippage
                        tr = self._calc_trade_Record(
                            pos_dir, contracts, entry_idx, i, entry_price, fill_exit, stop_reason, cost_multiplier
                        )
                        trades.append(tr)
                        realized_equity += tr.net_pnl
                        pos_dir = 0
                        stopped_out = True
                    elif pos_dir == -1 and h >= stop_price:
                        # 空單停損：若開盤直接跳空高於停損價，以 max(stop_price, open) + 滑價 成交
                        raw_exit = max(stop_price, o)
                        fill_exit = raw_exit + slippage
                        tr = self._calc_trade_Record(
                            pos_dir, contracts, entry_idx, i, entry_price, fill_exit, stop_reason, cost_multiplier
                        )
                        trades.append(tr)
                        realized_equity += tr.net_pnl
                        pos_dir = 0
                        stopped_out = True

                # 檢查固定點數/百分比/ATR 停利與更新極值
                if not stopped_out and pos_dir != 0:
                    if tp_price is not None:
                        if pos_dir == 1 and h >= tp_price:
                            raw_exit = max(tp_price, o)
                            fill_exit = raw_exit - slippage
                            tr = self._calc_trade_Record(
                                pos_dir, contracts, entry_idx, i, entry_price, fill_exit, "take_profit", cost_multiplier
                            )
                            trades.append(tr)
                            realized_equity += tr.net_pnl
                            pos_dir = 0
                        elif pos_dir == -1 and l <= tp_price:
                            raw_exit = min(tp_price, o)
                            fill_exit = raw_exit + slippage
                            tr = self._calc_trade_Record(
                                pos_dir, contracts, entry_idx, i, entry_price, fill_exit, "take_profit", cost_multiplier
                            )
                            trades.append(tr)
                            realized_equity += tr.net_pnl
                            pos_dir = 0

                    # 更新持倉期間最高/最低價供下一根移動停損使用
                    if pos_dir != 0:
                        highest_since_entry = max(highest_since_entry, h)
                        lowest_since_entry = min(lowest_since_entry, l)

            # Step C: K 棒收盤確認訊號 (signal_on = bar_close)，排入下一根開盤執行
            if i < n_bars - 1:
                # 0. 檢查結算日強制平倉 (time_exit)
                force_settle_exit, block_settle_entry = settlement_cutoff_state(
                    self.strategy, self.contract.settlement_rule, bar_ts
                )
                if force_settle_exit and pos_dir != 0:
                    pending_signal = (0, "settlement_exit")
                elif block_settle_entry:
                    # 結算前截止期不建新倉，若有部位且未平倉則平倉
                    if pos_dir != 0:
                        pending_signal = (0, "settlement_exit")
                else:
                    # 1. 檢查指標型出場 (indicator_cross)
                    if pos_dir == 1 and sl_spec and sl_spec.type == "indicator_cross" and sl_spec.long:
                        if self._eval_condition(sl_spec.long, cols, i):
                            pending_signal = (0, "indicator_exit")
                    elif pos_dir == -1 and sl_spec and sl_spec.type == "indicator_cross" and sl_spec.short:
                        if self._eval_condition(sl_spec.short, cols, i):
                            pending_signal = (0, "indicator_exit")

                    # 2. 檢查進場或反手訊號 (結合 AI 信心度閾值與市場狀態過濾)
                    long_sig = self._eval_condition(self.strategy.entry_long, cols, i)
                    short_sig = self._eval_condition(self.strategy.entry_short, cols, i)

                    if cf_spec.enabled and "long_confidence" in cols:
                        eff_th = float(cols["effective_threshold"][i])
                        if long_sig and float(cols["long_confidence"][i]) < eff_th:
                            long_sig = False
                        if short_sig and float(cols["short_confidence"][i]) < eff_th:
                            short_sig = False

                    if long_sig and pos_dir <= 0:
                        pending_signal = (1, "reverse" if pos_dir == -1 else "entry_long")
                    elif short_sig and pos_dir >= 0:
                        pending_signal = (-1, "reverse" if pos_dir == 1 else "entry_short")

            # 記錄當根收盤權益曲線 (含未實現損益)
            unrealized = 0.0
            if pos_dir != 0:
                unrealized = (c - entry_price) * pos_dir * self.contract.point_value * contracts
            equity_curve.append(realized_equity + unrealized)

        # 資料結束若仍持有部位，以最後一根收盤價強制結算一筆 end_of_data 交易，讓回測統計真實完整
        if pos_dir != 0 and n_bars > 0:
            last_c = float(cols["close"][-1])
            fill_exit = last_c - slippage if pos_dir == 1 else last_c + slippage
            tr = self._calc_trade_Record(
                pos_dir, contracts, entry_idx, n_bars - 1, entry_price, fill_exit, "end_of_data", cost_multiplier
            )
            trades.append(tr)

        return self._build_report(trades, equity_curve)

    def _build_report(self, trades: List[TradeRecord], equity_curve: List[float]) -> BacktestReport:
        total_trades = len(trades)
        if total_trades == 0:
            return BacktestReport(
                trades=[],
                equity_curve=equity_curve,
                net_profit=0.0,
                gross_profit=0.0,
                gross_loss=0.0,
                profit_factor=0.0,
                win_rate=0.0,
                max_drawdown=0.0,
                max_drawdown_pct=0.0,
                total_trades=0,
                expectancy=0.0,
            )

        gross_profit = sum(t.net_pnl for t in trades if t.net_pnl > 0)
        gross_loss = abs(sum(t.net_pnl for t in trades if t.net_pnl < 0))
        net_profit = gross_profit - gross_loss
        wins = sum(1 for t in trades if t.net_pnl > 0)
        win_rate = (wins / total_trades) * 100.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
        expectancy = net_profit / total_trades

        peak = self.initial_capital
        max_dd = 0.0
        max_dd_pct = 0.0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = (dd / peak) * 100.0 if peak > 0 else 0.0

        return BacktestReport(
            trades=trades,
            equity_curve=equity_curve,
            net_profit=round(net_profit, 2),
            gross_profit=round(gross_profit, 2),
            gross_loss=round(gross_loss, 2),
            profit_factor=round(profit_factor, 2),
            win_rate=round(win_rate, 2),
            max_drawdown=round(max_dd, 2),
            max_drawdown_pct=round(max_dd_pct, 2),
            total_trades=total_trades,
            expectancy=round(expectancy, 2),
        )

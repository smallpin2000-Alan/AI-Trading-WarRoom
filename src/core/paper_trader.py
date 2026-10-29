# src/core/paper_trader.py
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Literal
import polars as pl

from src.schema.strategy import StrategySchema
from src.core.engine import ContractSpec, EventDrivenEngine, TradeRecord
from src.core.indicators import IndicatorEngine
from src.ai.confidence_engine import ConfidenceEngine


@dataclass
class RiskConfig:
    """風控中心參數設定"""
    max_daily_loss: float = 5000.0          # 單日最大虧損上限 (TWD)
    daily_profit_target: float = 0.0        # 單日獲利達標全平上限 (0 代表不啟用，>0 代表達標自動全平鎖定)
    max_consecutive_losses: int = 4         # 連續虧損幾筆觸發模型退化自動暫停 (0 代表不限制)
    max_contracts: int = 2                  # 最大允許持倉口數
    check_margin: bool = True               # 是否強制檢查原始保證金


@dataclass
class AuditLogEntry:
    """稽核日誌項目"""
    timestamp: str
    event_type: Literal["SYSTEM", "SIGNAL", "ORDER_FILL", "STOP_OUT", "RISK_BREACH", "KILL_SWITCH"]
    message: str
    price: Optional[float] = None
    pnl: Optional[float] = None


@dataclass
class OpenPosition:
    """當前未平倉部位"""
    direction: int          # 1: 多單 (Long), -1: 空單 (Short)
    contracts: int
    entry_price: float
    entry_time: str
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    highest_since_entry: float = 0.0
    lowest_since_entry: float = float("inf")


class PaperTradingEngine:
    """
    紙上交易模擬撮合與風控中心引擎 (Paper Trading & Risk Center)
    嚴格遵守 AGENTS.md 交易安全性最高原則：
    1. 啟動前必須通過五道可信度檢核 (credibility_passed == True)。
    2. 策略停損為強制品，若無停損設定絕對不可執行。
    3. 與 EventDrivenEngine 共用同一套指標與條件評估核心。
    4. 內建單日最大虧損鎖定 (Daily Max Loss Limit) 與一鍵緊急平倉 (Kill Switch)。
    """

    VERSION = 2

    def __init__(
        self,
        strategy: StrategySchema,
        contract: ContractSpec,
        initial_capital: float = 100_000.0,
        risk_config: Optional[RiskConfig] = None,
    ):
        # 強制安全檢查：無停損絕對不可執行
        if strategy.exit.stop_loss is None and strategy.exit.trailing_stop is None:
            raise ValueError("安全攔截：策略未設定任何停損機制，絕對禁止載入紙上交易或下單引擎！")

        self.strategy = strategy
        self.contract = contract
        self.initial_capital = initial_capital
        self.risk_config = risk_config or RiskConfig()
        self.core_engine = EventDrivenEngine(strategy, contract, initial_capital)

        self.is_active: bool = False
        self.kill_switch_locked: bool = False
        self.daily_loss_locked: bool = False
        self.daily_profit_locked: bool = False
        self.degradation_locked: bool = False
        self.credibility_verified: bool = False

        self.position: Optional[OpenPosition] = None
        self.pending_signal: Optional[tuple[int, str]] = None  # (target_dir, reason)
        self.last_processed_bar_ts: Optional[datetime] = None
        self.last_price: float = 0.0

        self.closed_trades: List[TradeRecord] = []
        self.audit_logs: List[AuditLogEntry] = []

    def _log(
        self,
        event_type: Literal["SYSTEM", "SIGNAL", "ORDER_FILL", "STOP_OUT", "RISK_BREACH", "KILL_SWITCH"],
        message: str,
        price: Optional[float] = None,
        pnl: Optional[float] = None,
    ):
        entry = AuditLogEntry(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            event_type=event_type,
            message=message,
            price=price,
            pnl=pnl,
        )
        self.audit_logs.append(entry)
        if len(self.audit_logs) > 200:
            self.audit_logs.pop(0)

    def start_session(self, credibility_passed: bool) -> tuple[bool, str]:
        """
        啟動紙上交易會話。
        依據 AGENTS.md：下單執行前必須通過五道可信度檢核，且策略必須具備停損。
        """
        if self.strategy.exit.stop_loss is None and self.strategy.exit.trailing_stop is None:
            msg = "安全攔截：策略缺乏停損設定，禁止啟動！"
            self._log("RISK_BREACH", msg)
            return False, msg

        if not credibility_passed:
            msg = "安全攔截：策略未通過五道可信度驗證門檻，禁止啟動紙上交易執行！"
            self._log("RISK_BREACH", msg)
            return False, msg

        if self.kill_switch_locked:
            msg = "風控攔截：Kill Switch 鎖定中，請先手動解除風控鎖定。"
            self._log("RISK_BREACH", msg)
            return False, msg

        if self.daily_loss_locked:
            msg = "風控攔截：已達單日最大虧損上限，當日禁止重新啟動交易。"
            self._log("RISK_BREACH", msg)
            return False, msg

        if self.daily_profit_locked:
            msg = "風控攔截：已達單日獲利達標全平上限，當日已獲利了結休息，請先解除鎖定。"
            self._log("RISK_BREACH", msg)
            return False, msg

        if self.degradation_locked:
            msg = "風控攔截：策略觸發退化監控（連續虧損超標），請檢視市場結構或解除鎖定後再啟動。"
            self._log("RISK_BREACH", msg)
            return False, msg

        self.credibility_verified = True
        self.is_active = True
        msg = f"紙上交易已啟動 (策略: {self.strategy.id} v{self.strategy.version}, 標的: {self.contract.symbol})"
        self._log("SYSTEM", msg)
        return True, msg

    def stop_session(self):
        """暫停紙上交易訊號接收（保留當前部位）"""
        self.is_active = False
        self._log("SYSTEM", "紙上交易已暫停自動訊號執行。")

    def trigger_kill_switch(self, current_price: Optional[float] = None, reason: str = "使用者手動觸發一鍵緊急平倉") -> str:
        """
        一鍵緊急平倉並停止所有策略 (Kill Switch)
        """
        self.is_active = False
        self.kill_switch_locked = True
        self.pending_signal = None

        exec_price = current_price if (current_price and current_price > 0) else self.last_price
        if self.position is not None and exec_price > 0:
            tr = self._close_current_position(exec_price, exit_idx=-1, reason="kill_switch")
            msg = f"🚨 Kill Switch 觸發 ({reason})！已於 {tr.exit_price:,.1f} 強制平倉 {tr.contracts} 口，淨損益: {tr.net_pnl:,.0f} 元。系統已鎖定。"
            self._log("KILL_SWITCH", msg, price=tr.exit_price, pnl=tr.net_pnl)
            return msg

        msg = f"🚨 Kill Switch 觸發 ({reason})！目前無未平倉部位，所有策略與掛單已立即停止並鎖定。"
        self._log("KILL_SWITCH", msg)
        return msg

    def reset_risk_locks(self):
        """重置風控鎖定狀態（Kill Switch、單日虧損、獲利達標與模型退化鎖定）"""
        self.kill_switch_locked = False
        self.daily_loss_locked = False
        self.daily_profit_locked = False
        self.degradation_locked = False
        self._log("SYSTEM", "已手動解除 Kill Switch / 單日損益 / 模型退化風控鎖定。")

    @property
    def consecutive_losses(self) -> int:
        """計算目前連續虧損筆數"""
        cnt = 0
        for tr in reversed(self.closed_trades):
            if tr.net_pnl < 0:
                cnt += 1
            else:
                break
        return cnt

    @property
    def realized_pnl(self) -> float:
        return round(sum(t.net_pnl for t in self.closed_trades), 2)

    def get_unrealized_pnl(self, current_price: Optional[float] = None) -> float:
        if self.position is None:
            return 0.0
        p = current_price if (current_price and current_price > 0) else self.last_price
        if p <= 0:
            return 0.0
        gross = (p - self.position.entry_price) * self.position.direction * self.contract.point_value * self.position.contracts
        return round(gross, 2)

    def get_total_equity(self, current_price: Optional[float] = None) -> float:
        return round(self.initial_capital + self.realized_pnl + self.get_unrealized_pnl(current_price), 2)

    def _compute_stop_and_tp(self, pos: OpenPosition) -> tuple[Optional[float], Optional[float], str]:
        sl_spec = self.strategy.exit.stop_loss
        tp_spec = self.strategy.exit.take_profit
        ts_spec = self.strategy.exit.trailing_stop

        stop_price: Optional[float] = None
        stop_reason = "stop_loss"

        if sl_spec is not None and sl_spec.type in ("fixed_points", "percent") and sl_spec.value is not None:
            dist = sl_spec.value if sl_spec.type == "fixed_points" else pos.entry_price * (sl_spec.value / 100.0)
            stop_price = (pos.entry_price - dist) if pos.direction == 1 else (pos.entry_price + dist)

        if ts_spec is not None:
            ts_dist = ts_spec.value if ts_spec.type == "points" else pos.entry_price * (ts_spec.value / 100.0)
            if pos.direction == 1 and (pos.highest_since_entry - pos.entry_price) >= ts_spec.activation_points:
                ts_candidate = pos.highest_since_entry - ts_dist
                if stop_price is None or ts_candidate > stop_price:
                    stop_price = ts_candidate
                    stop_reason = "trailing_stop"
            elif pos.direction == -1 and (pos.entry_price - pos.lowest_since_entry) >= ts_spec.activation_points:
                ts_candidate = pos.lowest_since_entry + ts_dist
                if stop_price is None or ts_candidate < stop_price:
                    stop_price = ts_candidate
                    stop_reason = "trailing_stop"

        tp_price: Optional[float] = None
        if tp_spec is not None and tp_spec.type in ("fixed_points", "percent") and tp_spec.value is not None:
            tp_dist = tp_spec.value if tp_spec.type == "fixed_points" else pos.entry_price * (tp_spec.value / 100.0)
            tp_price = (pos.entry_price + tp_dist) if pos.direction == 1 else (pos.entry_price - tp_dist)

        return stop_price, tp_price, stop_reason

    def _open_new_position(self, direction: int, raw_price: float, bar_time_str: str, reason: str) -> bool:
        contracts = min(self.strategy.position_sizing.contracts, self.risk_config.max_contracts)
        if self.risk_config.check_margin:
            required_margin = self.contract.initial_margin * contracts
            avail_equity = self.get_total_equity(raw_price)
            if avail_equity < required_margin:
                self._log(
                    "RISK_BREACH",
                    f"保證金不足攔截：可用權益 {avail_equity:,.0f} 元 < 所需原始保證金 {required_margin:,.0f} 元",
                    price=raw_price,
                )
                return False

        slippage = self.strategy.execution.slippage_points
        fill_price = raw_price + slippage if direction == 1 else raw_price - slippage

        pos = OpenPosition(
            direction=direction,
            contracts=contracts,
            entry_price=fill_price,
            entry_time=bar_time_str,
            highest_since_entry=fill_price,
            lowest_since_entry=fill_price,
        )
        sl_p, tp_p, _ = self._compute_stop_and_tp(pos)
        pos.stop_loss_price = sl_p
        pos.take_profit_price = tp_p
        self.position = pos

        dir_label = "多單 (Long)" if direction == 1 else "空單 (Short)"
        self._log(
            "ORDER_FILL",
            f"新倉成交：{dir_label} {contracts} 口 @ {fill_price:,.1f} (停損價: {sl_p if sl_p else '指標出場'}, 原因: {reason})",
            price=fill_price,
        )
        return True

    def _close_current_position(self, raw_price: float, exit_idx: int, reason: str) -> TradeRecord:
        assert self.position is not None
        slippage = self.strategy.execution.slippage_points
        fill_exit = raw_price - slippage if self.position.direction == 1 else raw_price + slippage
        tr = self.core_engine._calc_trade_Record(
            direction=self.position.direction,
            contracts=self.position.contracts,
            entry_idx=0,
            exit_idx=exit_idx,
            entry_price=self.position.entry_price,
            exit_price=fill_exit,
            reason=reason,
            cost_multiplier=1.0,
        )
        self.closed_trades.append(tr)
        self.position = None
        return tr

    def check_daily_loss_limit(self, current_price: Optional[float] = None) -> bool:
        """
        檢查是否觸發單日最大虧損鎖定。
        若 (已實現損益 + 未實現損益) <= -max_daily_loss，立即強制平倉並鎖定。
        """
        p = current_price if (current_price and current_price > 0) else self.last_price
        if p > 0:
            self.last_price = p

        total_daily_pnl = self.realized_pnl + self.get_unrealized_pnl(p)
        if total_daily_pnl <= -abs(self.risk_config.max_daily_loss):
            self.is_active = False
            self.daily_loss_locked = True
            self.pending_signal = None
            if self.position is not None and p > 0:
                tr = self._close_current_position(p, exit_idx=-1, reason="daily_loss_limit")
                self._log(
                    "RISK_BREACH",
                    f"🛑 觸發單日最大虧損鎖定 (累積損益 {self.realized_pnl:,.0f} 元 <= -{self.risk_config.max_daily_loss:,.0f} 元)！已於 {tr.exit_price:,.1f} 自動強制平倉。",
                    price=tr.exit_price,
                    pnl=tr.net_pnl,
                )
            else:
                self._log(
                    "RISK_BREACH",
                    f"🛑 觸發單日最大虧損鎖定 (當日損益 {total_daily_pnl:,.0f} 元 <= -{self.risk_config.max_daily_loss:,.0f} 元)！已鎖定當日交易。",
                    price=p if p > 0 else None,
                    pnl=total_daily_pnl,
                )
            return True
        return False

    def check_daily_profit_target(self, current_price: Optional[float] = None) -> bool:
        """
        檢查是否觸發單日獲利達標全平 (AITrade 旗艦版功能：達標全平)。
        若 daily_profit_target > 0 且 (已實現損益 + 未實現損益) >= daily_profit_target，自動平倉入袋為安並鎖定。
        """
        if self.risk_config.daily_profit_target <= 0:
            return False

        p = current_price if (current_price and current_price > 0) else self.last_price
        if p > 0:
            self.last_price = p

        total_daily_pnl = self.realized_pnl + self.get_unrealized_pnl(p)
        if total_daily_pnl >= self.risk_config.daily_profit_target:
            self.is_active = False
            self.daily_profit_locked = True
            self.pending_signal = None
            if self.position is not None and p > 0:
                tr = self._close_current_position(p, exit_idx=-1, reason="daily_profit_target")
                self._log(
                    "ORDER_FILL",
                    f"🏆 觸發單日獲利達標全平 (累積損益 {self.realized_pnl:,.0f} 元 >= 目標 {self.risk_config.daily_profit_target:,.0f} 元)！已於 {tr.exit_price:,.1f} 自動平倉入袋為安。",
                    price=tr.exit_price,
                    pnl=tr.net_pnl,
                )
            else:
                self._log(
                    "SYSTEM",
                    f"🏆 觸發單日獲利達標鎖定 (當日損益 {total_daily_pnl:,.0f} 元 >= 目標 {self.risk_config.daily_profit_target:,.0f} 元)！當日停止新倉交易。",
                    price=p if p > 0 else None,
                    pnl=total_daily_pnl,
                )
            return True
        return False

    def check_model_degradation(self) -> bool:
        """
        檢查是否觸發模型退化自動暫停 (Model Degradation Monitor)。
        當連續虧損筆數 >= max_consecutive_losses 時，自動暫停策略並發出風控警示。
        """
        limit = self.risk_config.max_consecutive_losses
        if limit <= 0:
            return False

        consec = self.consecutive_losses
        if consec >= limit:
            self.is_active = False
            self.degradation_locked = True
            self.pending_signal = None
            self._log(
                "RISK_BREACH",
                f"⚠️ 觸發模型退化監控警示：策略已連續虧損 {consec} 筆 (達上限 {limit} 筆)，疑似當前市場結構與模型不符，已自動暫停下單！",
            )
            return True
        return False

    def sync_with_market(self, df_bars: pl.DataFrame, latest_price: Optional[float] = None):
        """
        接收台新 API 的最新 K 棒序列與即時報價，驅動紙上交易撮合與風控檢查。
        """
        if df_bars.height == 0:
            if latest_price and latest_price > 0:
                self.last_price = latest_price
                self.check_daily_loss_limit(latest_price)
                self.check_daily_profit_target(latest_price)
            return

        ind_specs = {k: v.model_dump() for k, v in self.strategy.indicators.items()}
        df_ind = IndicatorEngine.compute_indicators(df_bars, ind_specs)
        cf_spec = self.strategy.confidence_filter
        if cf_spec.enabled:
            df_ind = ConfidenceEngine.compute_confidence_series(
                df_ind, base_threshold=cf_spec.threshold, regime_adaptive=cf_spec.regime_adaptive
            )
        cols = {col: df_ind[col].to_list() for col in df_ind.columns}
        n_bars = df_ind.height

        bar_ts: datetime = cols["timestamp"][-1]
        o = float(cols["open"][-1])
        h = float(cols["high"][-1])
        l = float(cols["low"][-1])
        c = float(cols["close"][-1])
        curr_price = float(latest_price) if (latest_price and latest_price > 0) else c
        self.last_price = curr_price
        bar_time_str = bar_ts.strftime("%Y-%m-%d %H:%M")

        # 1. 先做即時風控：單日最大虧損、獲利達標全平、模型退化檢查
        if self.check_daily_loss_limit(curr_price) or self.check_daily_profit_target(curr_price):
            return

        if (
            not self.is_active
            or self.kill_switch_locked
            or self.daily_loss_locked
            or self.daily_profit_locked
            or self.degradation_locked
        ):
            return

        # 2. 若有前一根 K 棒收盤產生的待執行訊號，於新 K 棒開盤價執行 (fill_at = next_bar_open)
        if self.pending_signal is not None and (self.last_processed_bar_ts is None or bar_ts > self.last_processed_bar_ts):
            target_dir, sig_reason = self.pending_signal
            self.pending_signal = None

            if self.position is not None:
                tr = self._close_current_position(o, exit_idx=n_bars - 1, reason=sig_reason)
                self._log(
                    "ORDER_FILL",
                    f"平倉成交：{sig_reason} @ {tr.exit_price:,.1f}，淨損益: {tr.net_pnl:,.0f} 元",
                    price=tr.exit_price,
                    pnl=tr.net_pnl,
                )
                if (
                    self.check_daily_loss_limit(curr_price)
                    or self.check_daily_profit_target(curr_price)
                    or self.check_model_degradation()
                ):
                    return

            if target_dir != 0:
                self._open_new_position(target_dir, o, bar_time_str, sig_reason)

        # 3. 若持有部位，檢查盤中停損 / 移動停損 / 停利
        if self.position is not None:
            self.position.highest_since_entry = max(self.position.highest_since_entry, h, curr_price)
            self.position.lowest_since_entry = min(self.position.lowest_since_entry, l, curr_price)
            stop_price, tp_price, stop_reason = self._compute_stop_and_tp(self.position)
            self.position.stop_loss_price = stop_price
            self.position.take_profit_price = tp_price

            low_check = min(l, curr_price)
            high_check = max(h, curr_price)

            stopped = False
            if stop_price is not None:
                if self.position.direction == 1 and low_check <= stop_price:
                    raw_exit = min(stop_price, o)
                    tr = self._close_current_position(raw_exit, exit_idx=n_bars - 1, reason=stop_reason)
                    self._log(
                        "STOP_OUT",
                        f"🛡️ 多單停損觸發 ({stop_reason}) @ {tr.exit_price:,.1f}，淨損益: {tr.net_pnl:,.0f} 元",
                        price=tr.exit_price,
                        pnl=tr.net_pnl,
                    )
                    stopped = True
                elif self.position.direction == -1 and high_check >= stop_price:
                    raw_exit = max(stop_price, o)
                    tr = self._close_current_position(raw_exit, exit_idx=n_bars - 1, reason=stop_reason)
                    self._log(
                        "STOP_OUT",
                        f"🛡️ 空單停損觸發 ({stop_reason}) @ {tr.exit_price:,.1f}，淨損益: {tr.net_pnl:,.0f} 元",
                        price=tr.exit_price,
                        pnl=tr.net_pnl,
                    )
                    stopped = True

            if not stopped and self.position is not None and tp_price is not None:
                if self.position.direction == 1 and high_check >= tp_price:
                    raw_exit = max(tp_price, o)
                    tr = self._close_current_position(raw_exit, exit_idx=n_bars - 1, reason="take_profit")
                    self._log(
                        "ORDER_FILL",
                        f"🎯 多單停利成交 @ {tr.exit_price:,.1f}，淨損益: {tr.net_pnl:,.0f} 元",
                        price=tr.exit_price,
                        pnl=tr.net_pnl,
                    )
                elif self.position.direction == -1 and low_check <= tp_price:
                    raw_exit = min(tp_price, o)
                    tr = self._close_current_position(raw_exit, exit_idx=n_bars - 1, reason="take_profit")
                    self._log(
                        "ORDER_FILL",
                        f"🎯 空單停利成交 @ {tr.exit_price:,.1f}，淨損益: {tr.net_pnl:,.0f} 元",
                        price=tr.exit_price,
                        pnl=tr.net_pnl,
                    )

            if (
                self.check_daily_loss_limit(curr_price)
                or self.check_daily_profit_target(curr_price)
                or self.check_model_degradation()
            ):
                return

        # 4. 在新 K 棒評估收盤訊號 (排入下一根 K 棒開盤執行)
        if n_bars >= 2 and (self.last_processed_bar_ts is None or bar_ts > self.last_processed_bar_ts):
            idx = n_bars - 1
            sl_spec = self.strategy.exit.stop_loss

            if self.position is not None:
                if self.position.direction == 1 and sl_spec and sl_spec.type == "indicator_cross" and sl_spec.long:
                    if self.core_engine._eval_condition(sl_spec.long, cols, idx):
                        self.pending_signal = (0, "indicator_exit")
                        self._log("SIGNAL", f"產生指標平多訊號 (將於次根開盤執行)")
                elif self.position.direction == -1 and sl_spec and sl_spec.type == "indicator_cross" and sl_spec.short:
                    if self.core_engine._eval_condition(sl_spec.short, cols, idx):
                        self.pending_signal = (0, "indicator_exit")
                        self._log("SIGNAL", f"產生指標平空訊號 (將於次根開盤執行)")

            long_sig = self.core_engine._eval_condition(self.strategy.entry_long, cols, idx)
            short_sig = self.core_engine._eval_condition(self.strategy.entry_short, cols, idx)
            if cf_spec.enabled and "long_confidence" in cols:
                eff_th = float(cols["effective_threshold"][idx])
                if long_sig and float(cols["long_confidence"][idx]) < eff_th:
                    long_sig = False
                if short_sig and float(cols["short_confidence"][idx]) < eff_th:
                    short_sig = False

            pos_dir = self.position.direction if self.position else 0

            if long_sig and pos_dir <= 0:
                reason = "reverse" if pos_dir == -1 else "entry_long"
                self.pending_signal = (1, reason)
                self._log("SIGNAL", f"產生多方進場訊號 ({reason}，將於次根開盤執行)", price=c)
            elif short_sig and pos_dir >= 0:
                reason = "reverse" if pos_dir == -1 else "entry_short"
                self.pending_signal = (-1, reason)
                self._log("SIGNAL", f"產生空方進場訊號 ({reason}，將於次根開盤執行)", price=c)

            self.last_processed_bar_ts = bar_ts

# src/core/paper_trader.py
from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional, Literal, Dict, Any
import polars as pl

from src.schema.strategy import StrategySchema
from src.core.engine import ContractSpec, EventDrivenEngine, TradeRecord
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
    entry_atr: dict[int, float] = field(default_factory=dict)


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
        self._daily_loss_baseline_pnl: float = 0.0

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

    def stop_session(self, force_flat_price: Optional[float] = None) -> Optional[str]:
        """
        暫停紙上交易訊號接收。
        安全防禦機制（AGENTS.md 安全原則）：
        若當前持有部位，且提供了現價 (force_flat_price > 0)，
        自動強制市價平倉，絕不允許留未平倉部位在場中「無人防守裸漂」！
        """
        self.is_active = False
        p = force_flat_price if (force_flat_price and force_flat_price > 0) else self.last_price
        if self.position is not None and p > 0:
            # 只有當價格屬於同一品種範圍時才允許平倉，防止誤傳跨品種價格
            if (self.position.entry_price > 10000 and p > 10000) or (self.position.entry_price < 5000 and p < 5000):
                try:
                    tr = self._close_current_position(p, exit_idx=-1, reason="暫停操盤主動防守平倉")
                    msg = f"🛡️ 暫停操盤：安全強制平倉 {tr.contracts} 口 @ {tr.exit_price:,.1f}，淨損益: {tr.net_pnl:+,.0f} 元（防呆保護避免裸單漂流）。"
                    self._log("ORDER_FILL", msg, price=tr.exit_price, pnl=tr.net_pnl)
                    return msg
                except Exception as ex:
                    self._log("RISK_BREACH", f"暫停平倉攔截：{ex}")
            else:
                self._log("RISK_BREACH", f"暫停平倉價格防護：傳入價格 {p} 與部位進場價 {self.position.entry_price} 品種不符，拒絕撮合！")
        self._log("SYSTEM", "紙上交易已暫停自動訊號執行。")
        return None

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

    def reset_risk_locks(self, reset_daily_accounting: bool = True):
        """
        重置風控鎖定狀態（Kill Switch、單日虧損、獲利達標與模型退化鎖定）。
        當 reset_daily_accounting=True 時（例如使用者決定手動重啟）：
        將累計虧損歸檔或將重啟後的損益作為新回合計算，避免解鎖後又立刻被 check_daily_loss_limit 重新判定鎖定。
        """
        self.kill_switch_locked = False
        self.daily_loss_locked = False
        self.daily_profit_locked = False
        self.degradation_locked = False
        if reset_daily_accounting:
            # 記錄重置前的已實現損益基準，讓新的當日損益從重置點重新計算
            self._daily_loss_baseline_pnl = self.realized_pnl
        self._log("SYSTEM", "已手動解除 Kill Switch / 單日損益 / 模型退化風控鎖定，並重置風控門檻。")

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

    def _compute_stop_and_tp(
        self,
        pos: OpenPosition,
        current_atr: Optional[dict[int, float]] = None,
    ) -> tuple[Optional[float], Optional[float], str]:
        return compute_exit_levels(
            self.strategy,
            pos.direction,
            pos.entry_price,
            pos.highest_since_entry,
            pos.lowest_since_entry,
            pos.entry_atr,
            current_atr,
        )

    def _open_new_position(
        self,
        direction: int,
        raw_price: float,
        bar_time_str: str,
        reason: str,
        entry_atr: Optional[dict[int, float]] = None,
    ) -> bool:
        entry_atr_val = entry_atr or {}
        slippage = self.strategy.execution.slippage_points
        cand_entry_price = raw_price + slippage if direction == 1 else raw_price - slippage

        # 安全檢核：必須能算出有效停損距離
        can_open, err_msg = can_open_position(self.strategy, cand_entry_price, entry_atr_val)
        if not can_open:
            self._log("RISK_BREACH", f"停損防護攔截：{err_msg}", price=raw_price)
            return False

        avail_equity = self.get_total_equity(raw_price)
        contracts = compute_contracts(
            self.strategy,
            avail_equity,
            self.contract.point_value,
            cand_entry_price,
            entry_atr_val,
            max_contracts=self.risk_config.max_contracts,
        )
        if contracts <= 0:
            self._log("RISK_BREACH", "風控攔截：依風險比例計算之口數為 0，取消進場", price=raw_price)
            return False

        if self.risk_config.check_margin:
            required_margin = self.contract.initial_margin * contracts
            if avail_equity < required_margin:
                self._log(
                    "RISK_BREACH",
                    f"保證金不足攔截：可用權益 {avail_equity:,.0f} 元 < 所需原始保證金 {required_margin:,.0f} 元",
                    price=raw_price,
                )
                return False

        fill_price = cand_entry_price
        pos = OpenPosition(
            direction=direction,
            contracts=contracts,
            entry_price=fill_price,
            entry_time=bar_time_str,
            highest_since_entry=fill_price,
            lowest_since_entry=fill_price,
            entry_atr=entry_atr_val,
        )
        sl_p, tp_p, _ = self._compute_stop_and_tp(pos, current_atr=entry_atr_val)
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
        # 安全邊界防呆：出場價格若與進場價格出現超乎常理的跨品種落差（例如微台 39,000 點被股票 2,000 元撮合），嚴格拒絕撮合！
        if self.position.entry_price > 10000 and raw_price < 5000:
            raise ValueError(f"品種價格嚴重錯亂：期貨進場價 {self.position.entry_price} 絕不允許以個股現價 {raw_price} 撮合平倉！")
        if self.position.entry_price < 5000 and raw_price > 10000:
            raise ValueError(f"品種價格嚴重錯亂：個股進場價 {self.position.entry_price} 絕不允許以期貨現價 {raw_price} 撮合平倉！")

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
        若自重置後的新增虧損 (已實現損益 - baseline + 未實現損益) <= -max_daily_loss，立即強制平倉並鎖定。
        """
        p = current_price if (current_price and current_price > 0) else self.last_price
        if p > 0:
            self.last_price = p

        effective_daily_pnl = (self.realized_pnl - getattr(self, "_daily_loss_baseline_pnl", 0.0)) + self.get_unrealized_pnl(p)
        if effective_daily_pnl <= -abs(self.risk_config.max_daily_loss):
            self.is_active = False
            self.daily_loss_locked = True
            self.pending_signal = None
            if self.position is not None and p > 0:
                tr = self._close_current_position(p, exit_idx=-1, reason="daily_loss_limit")
                self._log(
                    "RISK_BREACH",
                    f"🛑 觸發單日最大虧損鎖定 (本回合損益 {effective_daily_pnl:,.0f} 元 <= -{self.risk_config.max_daily_loss:,.0f} 元)！已於 {tr.exit_price:,.1f} 自動強制平倉。",
                    price=tr.exit_price,
                    pnl=tr.net_pnl,
                )
            else:
                self._log(
                    "RISK_BREACH",
                    f"🛑 觸發單日最大虧損鎖定 (本回合損益 {effective_daily_pnl:,.0f} 元 <= -{self.risk_config.max_daily_loss:,.0f} 元)！已鎖定當日交易。",
                    price=p if p > 0 else None,
                    pnl=effective_daily_pnl,
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

    def check_position_stops(self, current_price: float, df_bars: pl.DataFrame) -> Optional[str]:
        """
        獨立即時風控檢核（保鏢機制）：
        不論 AI 大腦是否在冷靜期或出訊號，每當有新報價跳動時，
        立即檢核盤中未平倉部位是否觸發 30/45 點停損、停利或移動鎖利，觸發即秒級平倉！
        """
        if current_price <= 0 or self.position is None or df_bars.height == 0:
            return None
        self.last_price = current_price

        # 1. 每日損益熔斷檢核
        if self.check_daily_loss_limit(current_price) or self.check_daily_profit_target(current_price):
            return "風控鎖定"

        if not self.is_active or self.kill_switch_locked or self.daily_loss_locked or self.daily_profit_locked or self.degradation_locked:
            return "系統暫停或風控鎖定中"

        cols = {col: df_bars[col].to_list() for col in df_bars.columns}
        n_bars = df_bars.height
        l = float(cols["low"][-1])
        h = float(cols["high"][-1])
        o = float(cols["open"][-1])
        low_check = min(l, current_price)
        high_check = max(h, current_price)

        current_atr_val = read_atr_values(cols, self.strategy, max(0, n_bars - 2))
        stop_price, tp_price, stop_reason = self._compute_stop_and_tp(self.position, current_atr=current_atr_val)
        self.position.stop_loss_price = stop_price
        self.position.take_profit_price = tp_price
        self.position.highest_since_entry = max(self.position.highest_since_entry, high_check)
        self.position.lowest_since_entry = min(self.position.lowest_since_entry, low_check)

        # 停損
        if stop_price is not None:
            if self.position.direction == 1 and low_check <= stop_price:
                raw_exit = min(stop_price, o)
                tr = self._close_current_position(raw_exit, exit_idx=n_bars - 1, reason=stop_reason)
                self._log("STOP_OUT", f"🛡️ 多單停損觸發 ({stop_reason}) @ {tr.exit_price:,.1f}，淨損益: {tr.net_pnl:,.0f} 元", price=tr.exit_price, pnl=tr.net_pnl)
                self.save_state_to_disk()
                return f"多單停損 ({tr.net_pnl:+,.0f} 元)"
            elif self.position.direction == -1 and high_check >= stop_price:
                raw_exit = max(stop_price, o)
                tr = self._close_current_position(raw_exit, exit_idx=n_bars - 1, reason=stop_reason)
                self._log("STOP_OUT", f"🛡️ 空單停損觸發 ({stop_reason}) @ {tr.exit_price:,.1f}，淨損益: {tr.net_pnl:,.0f} 元", price=tr.exit_price, pnl=tr.net_pnl)
                self.save_state_to_disk()
                return f"空單停損 ({tr.net_pnl:+,.0f} 元)"

        # 停利
        if self.position is not None and tp_price is not None:
            if self.position.direction == 1 and high_check >= tp_price:
                raw_exit = max(tp_price, o)
                tr = self._close_current_position(raw_exit, exit_idx=n_bars - 1, reason="take_profit")
                self._log("ORDER_FILL", f"🎯 多單停利成交 @ {tr.exit_price:,.1f}，淨損益: {tr.net_pnl:,.0f} 元", price=tr.exit_price, pnl=tr.net_pnl)
                self.save_state_to_disk()
                return f"多單停利 ({tr.net_pnl:+,.0f} 元)"
            elif self.position.direction == -1 and low_check <= tp_price:
                raw_exit = min(tp_price, o)
                tr = self._close_current_position(raw_exit, exit_idx=n_bars - 1, reason="take_profit")
                self._log("ORDER_FILL", f"🎯 空單停利成交 @ {tr.exit_price:,.1f}，淨損益: {tr.net_pnl:,.0f} 元", price=tr.exit_price, pnl=tr.net_pnl)
                self.save_state_to_disk()
                return f"空單停利 ({tr.net_pnl:+,.0f} 元)"
        return None

    def sync_with_ai_advice(
        self,
        action: Literal["BUY_LONG", "SELL_SHORT", "WAIT"],
        confidence: float,
        current_price: float,
        df_bars: pl.DataFrame,
        source_note: str = "Gemini AI",
        min_confidence: float = 60.0,
    ) -> Optional[str]:
        """
        接收頂層 AI 決策顧問 (Gemini 3.5 / 量化矩陣) 產生的即時建議，直接驅動虛擬操盤交易。
        遵循紀律：
        1. 信心度 >= min_confidence 才准許進場。
        2. 當方向與現有部位相符時：續抱、不重複加碼。
        3. 當收到反向訊號時：自動「平倉原部位 ➔ 反向開新倉」。
        4. 當收到 WAIT 訊號時：不主動平倉，持倉由 ATR 停損/停利/移動停利守護。
        5. 每一筆撮合自動持久化至本地 JSON。
        """
        if current_price <= 0:
            return None
        self.last_price = current_price

        # 1. 先執行風控檢查
        if self.check_daily_loss_limit(current_price) or self.check_daily_profit_target(current_price):
            return "風控鎖定"

        if not self.is_active or self.kill_switch_locked or self.daily_loss_locked or self.daily_profit_locked or self.degradation_locked:
            return "系統暫停或風控鎖定中"

        # 2. 檢查盤中持倉是否觸發停損/停利 (呼叫專屬即時風控檢核)
        stop_res = self.check_position_stops(current_price, df_bars)
        if stop_res:
            return stop_res

        # 3. 處理 AI 進場或反手訊號
        target_dir = 1 if action == "BUY_LONG" else -1 if action == "SELL_SHORT" else 0
        if target_dir == 0 or confidence < min_confidence:
            return None  # 觀望或信心度不足門檻

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        pos_dir = self.position.direction if self.position else 0

        # 若方向一致，續抱
        if pos_dir == target_dir:
            return None

        # 若持有反向單，先平倉
        closed_msg = ""
        if pos_dir != 0:
            tr = self._close_current_position(current_price, exit_idx=-1, reason=f"AI反手平倉 ({source_note})")
            closed_msg = f"已平倉反向單 @ {tr.exit_price:,.1f} (損益 {tr.net_pnl:+,.0f} 元) ➔ "
            self._log("ORDER_FILL", f"AI反手平倉：{tr.exit_reason} @ {tr.exit_price:,.1f}，淨損益: {tr.net_pnl:,.0f} 元", price=tr.exit_price, pnl=tr.net_pnl)

        # 讀取 ATR
        atr_entry_val = {}
        if df_bars.height > 0:
            cols = {col: df_bars[col].to_list() for col in df_bars.columns}
            atr_entry_val = read_atr_values(cols, self.strategy, max(0, df_bars.height - 2))

        # 開新倉
        opened = self._open_new_position(target_dir, current_price, now_str, reason=f"AI訊號進場 ({source_note}, 信心度{confidence:.0f}%)", entry_atr=atr_entry_val)
        if opened:
            self.save_state_to_disk()
            dir_zh = "多單" if target_dir == 1 else "空單"
            return f"{closed_msg}成功開立 {dir_zh} 1 口 @ {current_price:,.1f}"
        return None

    def save_state_to_disk(self, file_path: str | Path = "data/paper_trading_state.json") -> None:
        """將虛擬操盤帳戶狀態（資金、部位、已平倉交易、風控日誌）持久化儲存至本地磁碟"""
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        pos_dict = None
        if self.position is not None:
            pos_dict = {
                "direction": self.position.direction,
                "contracts": self.position.contracts,
                "entry_price": self.position.entry_price,
                "entry_time": self.position.entry_time,
                "stop_loss_price": self.position.stop_loss_price,
                "take_profit_price": self.position.take_profit_price,
                "highest_since_entry": self.position.highest_since_entry,
                "lowest_since_entry": self.position.lowest_since_entry,
                "entry_atr": {str(k): v for k, v in self.position.entry_atr.items()},
            }
        trades_list = [asdict(t) for t in self.closed_trades[-100:]]
        logs_list = [asdict(l) for l in self.audit_logs[-100:]]

        payload = {
            "is_active": self.is_active,
            "kill_switch_locked": self.kill_switch_locked,
            "daily_loss_locked": self.daily_loss_locked,
            "daily_profit_locked": self.daily_profit_locked,
            "degradation_locked": self.degradation_locked,
            "credibility_verified": self.credibility_verified,
            "daily_loss_baseline_pnl": getattr(self, "_daily_loss_baseline_pnl", 0.0),
            "initial_capital": self.initial_capital,
            "realized_pnl": self.realized_pnl,
            "strategy": self.strategy.model_dump(),
            "risk_config": asdict(self.risk_config),
            "position": pos_dict,
            "closed_trades": trades_list,
            "audit_logs": logs_list,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_state_from_disk(self, file_path: str | Path = "data/paper_trading_state.json") -> bool:
        """從本地磁碟恢復虛擬操盤帳戶狀態"""
        p = Path(file_path)
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            self.is_active = data.get("is_active", False)
            self.kill_switch_locked = data.get("kill_switch_locked", False)
            self.daily_loss_locked = data.get("daily_loss_locked", False)
            self.daily_profit_locked = data.get("daily_profit_locked", False)
            self.degradation_locked = data.get("degradation_locked", False)
            self.credibility_verified = data.get("credibility_verified", True)
            self._daily_loss_baseline_pnl = float(data.get("daily_loss_baseline_pnl", 0.0))
            self.initial_capital = data.get("initial_capital", self.initial_capital)

            strat_data = data.get("strategy")
            if strat_data:
                try:
                    self.strategy = StrategySchema.model_validate(strat_data)
                    self.core_engine = EventDrivenEngine(self.strategy, self.contract, self.initial_capital)
                except Exception:
                    pass

            rc_data = data.get("risk_config")
            if rc_data:
                try:
                    self.risk_config = RiskConfig(**rc_data)
                except Exception:
                    pass

            pos_data = data.get("position")
            if pos_data:
                self.position = OpenPosition(
                    direction=pos_data["direction"],
                    contracts=pos_data["contracts"],
                    entry_price=pos_data["entry_price"],
                    entry_time=pos_data["entry_time"],
                    stop_loss_price=pos_data.get("stop_loss_price"),
                    take_profit_price=pos_data.get("take_profit_price"),
                    highest_since_entry=pos_data.get("highest_since_entry", pos_data["entry_price"]),
                    lowest_since_entry=pos_data.get("lowest_since_entry", pos_data["entry_price"]),
                    entry_atr={int(k): v for k, v in pos_data.get("entry_atr", {}).items()},
                )
            else:
                self.position = None

            raw_trades = data.get("closed_trades", [])
            self.closed_trades = [TradeRecord(**t) for t in raw_trades]

            raw_logs = data.get("audit_logs", [])
            self.audit_logs = [AuditLogEntry(**l) for l in raw_logs]
            return True
        except Exception:
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
        df_ind = add_exit_atr_columns(df_ind, self.strategy)
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
                # 讀取前一根已收盤 K 棒的 ATR 作為進場基準
                atr_entry_val = read_atr_values(cols, self.strategy, max(0, n_bars - 2))
                self._open_new_position(target_dir, o, bar_time_str, sig_reason, entry_atr=atr_entry_val)

        # 3. 若持有部位，檢查盤中停損 / 移動停損 / 停利
        if self.position is not None:
            self.position.highest_since_entry = max(self.position.highest_since_entry, h, curr_price)
            self.position.lowest_since_entry = min(self.position.lowest_since_entry, l, curr_price)
            current_atr_val = read_atr_values(cols, self.strategy, max(0, n_bars - 2))
            stop_price, tp_price, stop_reason = self._compute_stop_and_tp(self.position, current_atr=current_atr_val)
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
            # 0. 檢查結算日強制平倉 (time_exit)
            force_settle_exit, block_settle_entry = settlement_cutoff_state(
                self.strategy, self.contract.settlement_rule, bar_ts
            )
            if force_settle_exit and self.position is not None:
                self.pending_signal = (0, "settlement_exit")
                self._log("SIGNAL", "結算日前強制平倉訊號 (將於次根開盤執行)")
                self.last_processed_bar_ts = bar_ts
                return
            elif block_settle_entry:
                if self.position is not None:
                    self.pending_signal = (0, "settlement_exit")
                    self._log("SIGNAL", "結算日前平倉訊號 (進入截止時段)")
                self.last_processed_bar_ts = bar_ts
                return

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

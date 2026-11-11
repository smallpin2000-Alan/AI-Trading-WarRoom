# src/core/paper_trader.py
from __future__ import annotations
import json
import logging
import os
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Literal, Dict, Any
import polars as pl

from src.schema.strategy import StrategySchema
from src.core.engine import ContractSpec, EventDrivenEngine, TradeRecord
from src.core.indicators import IndicatorEngine
from src.ai.confidence_engine import ConfidenceEngine
from src.data import state_sync
from src.core.exits import (
    add_exit_atr_columns,
    read_atr_values,
    can_open_position,
    compute_exit_levels,
    compute_contracts,
    settlement_cutoff_state,
)

# 模擬倉狀態檔預設路徑（測試會以 monkeypatch 改寫，避免污染正式存檔）
DEFAULT_STATE_PATH: Path = Path("data/paper_trading_state.json")

logger = logging.getLogger(__name__)

# 平倉價格防護：成交價與進場價偏離超過此比例，視為誤傳其他商品的價格（例如以個股現價撮合期貨部位）
MAX_PRICE_DEVIATION_RATIO = 0.30
# 稽核日誌保留筆數上限
MAX_AUDIT_LOGS = 200
# 停損出場後，同方向再進場的冷靜期（秒）
STOP_OUT_COOLDOWN_SEC = 300.0
# 「全部模擬重置」後暫緩自動建倉的保護期（秒）
RESET_GRACE_SEC = 12.0
# 持倉中收到反向訊號時，允許反手所需的最低信心度（%）
REVERSE_MIN_CONFIDENCE = 68.0

# AI 自動操盤預設出場參數（以 ATR 倍數表示，隨市場波動自動調整停損距離）
# 背景：固定 45 點停損遠小於微台 5 分 K 振幅中位數（約 124 點），幾乎必被正常波動掃出場
DEFAULT_ATR_LENGTH = 14
DEFAULT_ATR_STOP_MULT = 1.5        # 停損 = 1.5 倍 ATR
DEFAULT_ATR_TP_MULT = 2.5          # 停利 = 2.5 倍 ATR（損益比約 1.67）
DEFAULT_ATR_TRAIL_MULT = 1.0       # 移動停損距離 = 1 倍 ATR
DEFAULT_ATR_TRAIL_ACTIVATION = 1.5  # 獲利達 1.5 倍 ATR 才啟動移動停損

_TW_TZ = timezone(timedelta(hours=8))
# 台指期夜盤 15:00 開盤即屬於「下一個交易日」：將台灣時間往後推 9 小時再取日期
_TRADING_DAY_SHIFT = timedelta(hours=9)


def trading_day_of(now_tw: datetime) -> str:
    """回傳某個台灣時間所屬的期貨交易日 (YYYY-MM-DD)；15:00 之後的夜盤歸屬次日。"""
    return (now_tw + _TRADING_DAY_SHIFT).strftime("%Y-%m-%d")


def current_trading_day() -> str:
    """目前所屬的期貨交易日（以台灣時間 UTC+8 計算，不受主機時區影響）。"""
    return trading_day_of(datetime.now(_TW_TZ).replace(tzinfo=None))


def default_atr_exit_spec() -> Dict[str, Any]:
    """AI 自動操盤預設的 ATR 出場設定（dict 形式，供 StrategySchema 驗證）。"""
    return {
        "stop_loss": {"type": "atr_multiple", "value": DEFAULT_ATR_STOP_MULT, "atr_length": DEFAULT_ATR_LENGTH},
        "take_profit": {"type": "atr_multiple", "value": DEFAULT_ATR_TP_MULT, "atr_length": DEFAULT_ATR_LENGTH},
        "trailing_stop": {
            "type": "atr_multiple",
            "value": DEFAULT_ATR_TRAIL_MULT,
            "activation_atr_multiple": DEFAULT_ATR_TRAIL_ACTIVATION,
            "atr_length": DEFAULT_ATR_LENGTH,
        },
    }


def default_ai_strategy(symbol: str = "TMF") -> StrategySchema:
    """戰情室 AI 自動模擬操盤使用的預設策略（進場由 AI 訊號驅動，出場以 ATR 倍數控管）。"""
    return StrategySchema.model_validate({
        "id": "ai_copilot_auto",
        "version": 2,
        "symbol": symbol,
        "timeframe": "5m",
        "session": "day_and_night",
        "indicators": {
            "ma5": {"type": "SMA", "source": "close", "length": 5},
            "ma20": {"type": "SMA", "source": "close", "length": 20},
        },
        "entry_long": {"cross_above": ["ma5", "ma20"]},
        "entry_short": {"cross_below": ["ma5", "ma20"]},
        "exit": default_atr_exit_spec(),
    })


def ensure_atr_exits(strategy: StrategySchema) -> tuple[StrategySchema, bool]:
    """
    若策略的停損不是 ATR 倍數（例如舊版存檔的固定 45 點），回傳改用預設 ATR 出場設定的新策略。
    不修改傳入的物件；回傳 (策略, 是否有轉換)。
    """
    sl = strategy.exit.stop_loss
    if sl is not None and sl.type == "atr_multiple":
        return strategy, False
    data = strategy.model_dump()
    data["exit"] = {**data["exit"], **default_atr_exit_spec()}
    return StrategySchema.model_validate(data), True


def _dt_to_str(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if isinstance(value, datetime) else None


def _dt_from_str(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


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

    VERSION = 8  # v8：出場改為 ATR 倍數停損

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
        self.order_safety_locked: bool = False   # 下單安全鎖 (True 時鎖定下單，不開新倉)
        self.custom_tp_points: float = 0.0       # 自訂自動停利點數 (點)

        self.position: Optional[OpenPosition] = None
        self.pending_signal: Optional[tuple[int, str]] = None  # (target_dir, reason)
        self.last_processed_bar_ts: Optional[datetime] = None
        self.last_price: float = 0.0

        self.closed_trades: List[TradeRecord] = []
        self.audit_logs: List[AuditLogEntry] = []
        self._daily_loss_baseline_pnl: float = 0.0
        self._trading_day: Optional[str] = None  # 目前計帳所屬的交易日（換日時自動歸零單日損益）
        self._last_reset_ts: float = 0.0
        self._last_stop_out_ts: float = 0.0
        self._last_stop_out_dir: int = 0
        self._last_stop_out_bar_ts: Optional[datetime] = None
        self._last_entry_bar_ts: Optional[datetime] = None
        self._last_entry_trading_day: Optional[str] = None  # 規則策略「一天只進場一次」的依據

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
        self.audit_logs = [*self.audit_logs, entry][-MAX_AUDIT_LOGS:]

    def start_session(self, credibility_passed: bool, simulation_override: bool = False) -> tuple[bool, str]:
        """
        啟動紙上交易會話。
        依據 AGENTS.md：下單執行前必須通過五道可信度檢核，且策略必須具備停損。

        simulation_override：僅限「模擬倉」使用。未通過檢核時，必須由使用者在介面上明確勾選確認後才可傳入 True；
        此時仍會啟動，但 credibility_verified 維持 False，並在稽核日誌留下 RISK_BREACH 紀錄（不是靜默放行）。
        """
        if self.strategy.exit.stop_loss is None and self.strategy.exit.trailing_stop is None:
            msg = "安全攔截：策略缺乏停損設定，禁止啟動！"
            self._log("RISK_BREACH", msg)
            return False, msg

        if not credibility_passed and not simulation_override:
            self.credibility_verified = False
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

        self.credibility_verified = bool(credibility_passed)
        self.is_active = True
        msg = f"紙上交易已啟動 (策略: {self.strategy.id} v{self.strategy.version}, 標的: {self.contract.symbol})"
        if credibility_passed:
            self._log("SYSTEM", msg)
        else:
            msg += "｜⚠️ 未通過五道可信度檢核，經使用者明確確認後以「僅限模擬倉」方式啟動"
            self._log("RISK_BREACH", msg)
        return True, msg

    def _price_matches_position(self, price: float) -> bool:
        """檢查傳入價格是否與目前部位屬於同一商品的合理價格範圍（防止以其他商品現價撮合）。"""
        if self.position is None or price <= 0:
            return False
        entry = self.position.entry_price
        if entry <= 0:
            return False
        return abs(price - entry) / entry <= MAX_PRICE_DEVIATION_RATIO

    def _roll_trading_day(self) -> None:
        """
        換交易日處理：將單日損益基準歸零重算，並解除「單日虧損 / 單日獲利達標」鎖定。
        安全考量：換日只解除單日鎖定，不會自動恢復 is_active，也不會解除 Kill Switch 與模型退化鎖定。
        """
        today = current_trading_day()
        if self._trading_day is None:
            self._trading_day = today
            return
        if today == self._trading_day:
            return
        prev_day = self._trading_day
        self._trading_day = today
        self._daily_loss_baseline_pnl = self.realized_pnl
        was_locked = self.daily_loss_locked or self.daily_profit_locked
        self.daily_loss_locked = False
        self.daily_profit_locked = False
        note = "，已解除單日損益鎖定（自動操盤維持暫停，需手動重新啟動）" if was_locked else ""
        self._log("SYSTEM", f"📅 交易日由 {prev_day} 換至 {today}：單日損益重新計帳{note}。")

    @property
    def daily_realized_pnl(self) -> float:
        """本交易日（或自上次手動重置風控後）的已實現損益。"""
        return round(self.realized_pnl - self._daily_loss_baseline_pnl, 2)

    def _bars_with_exit_atr(self, df_bars: pl.DataFrame) -> Dict[str, List[Any]]:
        """
        將 K 棒轉成欄位字典，並補上出場規則需要的 ATR 欄位。
        AI 即時路徑傳入的是原始 K 棒（沒有 ATR 欄位），若不補算，ATR 型停損會永遠「暖機中」而無法開倉或更新。
        """
        df_atr = add_exit_atr_columns(df_bars, self.strategy)
        return {col: df_atr[col].to_list() for col in df_atr.columns}

    @staticmethod
    def _entry_reason(target_dir: int, pos_dir: int) -> str:
        """進場原因標籤：持有反向部位時為 reverse，否則為 entry_long / entry_short。"""
        if pos_dir == -target_dir and pos_dir != 0:
            return "reverse"
        return "entry_long" if target_dir == 1 else "entry_short"

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
            if self._price_matches_position(p):
                tr = self._close_current_position(p, exit_idx=-1, reason="暫停操盤主動防守平倉")
                msg = f"🛡️ 暫停操盤：安全強制平倉 {tr.contracts} 口 @ {tr.exit_price:,.1f}，淨損益: {tr.net_pnl:+,.0f} 元（防呆保護避免裸單漂流）。"
                self._log("ORDER_FILL", msg, price=tr.exit_price, pnl=tr.net_pnl)
                return msg
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
        if self.position is not None and exec_price > 0 and not self._price_matches_position(exec_price):
            # 系統照樣鎖定，但絕不以明顯不屬於該商品的價格結算部位
            msg = (
                f"🚨 Kill Switch 觸發 ({reason})！系統已鎖定，但傳入價格 {exec_price:,.1f} 與部位進場價 "
                f"{self.position.entry_price:,.1f} 不屬於同一商品，已拒絕以此價格平倉；"
                "部位仍由停損機制守護，請以正確報價再次平倉。"
            )
            self._log("KILL_SWITCH", msg, price=exec_price)
            return msg
        if self.position is not None and exec_price > 0:
            tr = self._close_current_position(exec_price, exit_idx=-1, reason="kill_switch")
            msg = f"🚨 Kill Switch 觸發 ({reason})！已於 {tr.exit_price:,.1f} 強制平倉 {tr.contracts} 口，淨損益: {tr.net_pnl:,.0f} 元。系統已鎖定。"
            self._log("KILL_SWITCH", msg, price=tr.exit_price, pnl=tr.net_pnl)
            return msg
        if self.position is not None:
            msg = f"🚨 Kill Switch 觸發 ({reason})！系統已鎖定，但目前沒有有效報價可供平倉；部位仍由停損機制守護。"
            self._log("KILL_SWITCH", msg)
            return msg

        msg = f"🚨 Kill Switch 觸發 ({reason})！目前無未平倉部位，所有策略與掛單已立即停止並鎖定。"
        self._log("KILL_SWITCH", msg)
        return msg

    def close_position_manually(self, current_price: Optional[float] = None, reason: str = "手動平倉") -> tuple[bool, str]:
        """
        手動市價平倉（對話框「平倉」指令使用）。
        與 Kill Switch 不同：只平掉部位，不改變 is_active，也不會設定或解除任何風控鎖定。
        回傳 (是否成功平倉, 訊息)。
        """
        if self.position is None:
            return False, "目前帳戶為空手 (無未平倉部位)，無須執行平倉。"
        price = current_price if (current_price and current_price > 0) else self.last_price
        if price <= 0:
            return False, "平倉失敗：目前沒有有效的即時報價。"
        if not self._price_matches_position(price):
            msg = f"平倉失敗：價格 {price:,.1f} 與部位進場價 {self.position.entry_price:,.1f} 不屬於同一商品，已拒絕撮合。"
            self._log("RISK_BREACH", msg, price=price)
            return False, msg
        tr = self._close_current_position(price, exit_idx=-1, reason=reason)
        msg = f"已於 {tr.exit_price:,.1f} 平倉 {tr.contracts} 口，淨損益 {tr.net_pnl:+,.0f} 元。"
        self._log("ORDER_FILL", f"{reason}：{msg}", price=tr.exit_price, pnl=tr.net_pnl)
        self.check_model_degradation()
        return True, msg

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

    def reset_all_simulation(self, initial_capital: float = 100_000.0, save_to_disk: bool = True):
        """
        全部模擬重置：
        清空所有未平倉部位、歷史平倉交易紀錄、風控鎖定與稽核日誌，
        並將虛擬總資金恢復為初始設定（預設 100,000 元）。
        """
        self.initial_capital = float(initial_capital)
        self.position = None
        self.pending_signal = None
        self.last_processed_bar_ts = None
        self.closed_trades.clear()
        self.audit_logs.clear()
        self._daily_loss_baseline_pnl = 0.0
        self.kill_switch_locked = False
        self.daily_loss_locked = False
        self.daily_profit_locked = False
        self.degradation_locked = False
        self.order_safety_locked = False
        # 注意：重置不會改變 credibility_verified —— 可信度檢核結果不可因重置而被「順便」視為通過
        self._trading_day = current_trading_day()
        self._last_reset_ts = time.time()
        self._last_stop_out_ts = 0.0
        self._last_stop_out_dir = 0
        self._last_stop_out_bar_ts = None
        self._last_entry_bar_ts = None
        self._last_entry_trading_day = None
        self._log("SYSTEM", f"🔄 已執行全部模擬重置：虛擬本金恢復為 {self.initial_capital:,.0f} 元，所有持倉、歷史成交與風控紀錄已全數清空。")
        if save_to_disk:
            self.save_state_to_disk()

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
        sl_p, tp_p, reason = compute_exit_levels(
            self.strategy,
            pos.direction,
            pos.entry_price,
            pos.highest_since_entry,
            pos.lowest_since_entry,
            pos.entry_atr,
            current_atr,
        )
        if getattr(self, "custom_tp_points", 0.0) > 0:
            custom_tp = pos.entry_price + self.custom_tp_points if pos.direction == 1 else pos.entry_price - self.custom_tp_points
            tp_p = custom_tp
        return sl_p, tp_p, reason

    def _open_new_position(
        self,
        direction: int,
        raw_price: float,
        bar_time_str: str,
        reason: str,
        entry_atr: Optional[dict[int, float]] = None,
    ) -> bool:
        if getattr(self, "order_safety_locked", False):
            self._log("RISK_BREACH", f"下單安全鎖攔截：當前下單已鎖定，拒絕開立新倉 (欲進場: {reason})", price=raw_price)
            return False

        entry_atr_val = entry_atr or {}
        slippage = self.strategy.execution.slippage_points
        cand_entry_price = raw_price + slippage if direction == 1 else raw_price - slippage

        # 安全檢核：必須能算出有效停損距離
        can_open, err_msg = can_open_position(self.strategy, cand_entry_price, entry_atr_val, direction=direction)
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
            initial_margin=self.contract.initial_margin if self.risk_config.check_margin else None,
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
        if not self._price_matches_position(raw_price):
            raise ValueError(
                f"品種價格嚴重錯亂：進場價 {self.position.entry_price} 絕不允許以偏離超過 "
                f"{MAX_PRICE_DEVIATION_RATIO:.0%} 的價格 {raw_price} 撮合平倉！"
            )

        slippage = self.strategy.execution.slippage_points
        fill_exit = raw_price - slippage if self.position.direction == 1 else raw_price + slippage
        now_time_str = datetime.now().strftime("%Y-%m-%d %H:%M")
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
        tr.entry_time = getattr(self.position, "entry_time", "") or now_time_str
        tr.exit_time = now_time_str
        self.closed_trades.append(tr)
        self.position = None
        return tr

    def check_daily_loss_limit(self, current_price: Optional[float] = None) -> bool:
        """
        檢查是否觸發單日最大虧損鎖定。
        若自重置後的新增虧損 (已實現損益 - baseline + 未實現損益) <= -max_daily_loss，立即強制平倉並鎖定。
        """
        self._roll_trading_day()
        p = current_price if (current_price and current_price > 0) else self.last_price
        if p > 0:
            self.last_price = p

        effective_daily_pnl = self.daily_realized_pnl + self.get_unrealized_pnl(p)
        if effective_daily_pnl <= -abs(self.risk_config.max_daily_loss):
            self.is_active = False
            self.daily_loss_locked = True
            self.pending_signal = None
            if self.position is not None and p > 0 and self._price_matches_position(p):
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
        self._roll_trading_day()
        if self.risk_config.daily_profit_target <= 0:
            return False

        p = current_price if (current_price and current_price > 0) else self.last_price
        if p > 0:
            self.last_price = p

        # 「當日」以交易日為界，並自上次手動重置風控後重新計算（與單日虧損採用同一個基準）
        total_daily_pnl = self.daily_realized_pnl + self.get_unrealized_pnl(p)
        if total_daily_pnl >= self.risk_config.daily_profit_target:
            self.is_active = False
            self.daily_profit_locked = True
            self.pending_signal = None
            if self.position is not None and p > 0 and self._price_matches_position(p):
                tr = self._close_current_position(p, exit_idx=-1, reason="daily_profit_target")
                self._log(
                    "ORDER_FILL",
                    f"🏆 觸發單日獲利達標全平 (當日損益 {total_daily_pnl:,.0f} 元 >= 目標 {self.risk_config.daily_profit_target:,.0f} 元)！已於 {tr.exit_price:,.1f} 自動平倉入袋為安。",
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
        if current_price <= 0 or self.position is None:
            return None
        if not self._price_matches_position(current_price):
            # 報價明顯不屬於該商品（例如誤傳個股現價）：不更新、不撮合，留待下一筆正確報價
            logger.warning(
                "check_position_stops 略過異常報價 %s（部位進場價 %s）", current_price, self.position.entry_price
            )
            return None
        self.last_price = current_price

        # 1. 每日損益熔斷檢核
        if self.check_daily_loss_limit(current_price) or self.check_daily_profit_target(current_price):
            self.save_state_to_disk()
            return "風控鎖定"

        # 安全最高原則：只要還有未平倉部位，停損 / 停利 / 移動鎖利就必須持續守護，
        # 不因「暫停操盤」、Kill Switch、單日損益或模型退化鎖定而停止檢核（這些狀態只禁止開新倉）。
        has_bars = df_bars is not None and df_bars.height > 0
        cols = self._bars_with_exit_atr(df_bars) if has_bars else {}
        n_bars = df_bars.height if has_bars else 0
        latest_bar_ts = cols["timestamp"][-1] if "timestamp" in cols and n_bars > 0 else None

        # 關鍵修正：盤中即時保鏢檢核僅以「進場後真實即時報價 (current_price)」更新最高/最低價並觸發停損停利，
        # 絕不誤拿整根 5 分 K 在進場前發生的歷史 High/Low/Open 回溯掃損！
        self.position.highest_since_entry = max(self.position.highest_since_entry, current_price)
        self.position.lowest_since_entry = min(self.position.lowest_since_entry, current_price)

        current_atr_val = read_atr_values(cols, self.strategy, max(0, n_bars - 2)) if has_bars else {}
        stop_price, tp_price, stop_reason = self._compute_stop_and_tp(self.position, current_atr=current_atr_val)
        self.position.stop_loss_price = stop_price
        self.position.take_profit_price = tp_price

        direction = self.position.direction
        dir_zh = "多單" if direction == 1 else "空單"

        # 停損 (以真實即時價 current_price 判定與出場，並啟動停損後防雙巴冷靜期 Cooldown)
        stop_hit = stop_price is not None and (
            current_price <= stop_price if direction == 1 else current_price >= stop_price
        )
        if stop_hit:
            tr = self._close_current_position(current_price, exit_idx=n_bars - 1, reason=stop_reason)
            self._last_stop_out_ts = time.time()
            self._last_stop_out_dir = direction
            self._last_stop_out_bar_ts = latest_bar_ts
            self._log("STOP_OUT", f"🛡️ {dir_zh}停損觸發 ({stop_reason}) @ {tr.exit_price:,.1f}，淨損益: {tr.net_pnl:,.0f} 元 (啟動防雙巴冷靜期)", price=tr.exit_price, pnl=tr.net_pnl)
            # 即時路徑也必須執行「連續虧損休眠」檢核
            self.check_model_degradation()
            self.save_state_to_disk()
            return f"{dir_zh}停損 ({tr.net_pnl:+,.0f} 元)"

        # 停利
        tp_hit = tp_price is not None and (
            current_price >= tp_price if direction == 1 else current_price <= tp_price
        )
        if tp_hit:
            tr = self._close_current_position(current_price, exit_idx=n_bars - 1, reason="take_profit")
            self._log("ORDER_FILL", f"🎯 {dir_zh}停利成交 @ {tr.exit_price:,.1f}，淨損益: {tr.net_pnl:,.0f} 元", price=tr.exit_price, pnl=tr.net_pnl)
            self.check_model_degradation()
            self.save_state_to_disk()
            return f"{dir_zh}停利 ({tr.net_pnl:+,.0f} 元)"
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
            self.save_state_to_disk()
            return "風控鎖定"

        # 2. 既有持倉的停損/停利永遠優先檢核（不受暫停或風控鎖定影響）
        stop_res = self.check_position_stops(current_price, df_bars)
        if stop_res:
            return stop_res

        # 以下為「開新倉 / 反手」邏輯，暫停或任何風控鎖定時一律禁止
        if not self.is_active or self.kill_switch_locked or self.daily_loss_locked or self.daily_profit_locked or self.degradation_locked:
            return "系統暫停或風控鎖定中"

        # 剛執行「全部模擬重置」後的冷卻保護期內，暫緩立即自動建倉，確保畫面乾淨呈現初始資金與 0 筆紀錄
        if time.time() - self._last_reset_ts < RESET_GRACE_SEC:
            return None

        # 3. 處理 AI 進場或反手訊號
        target_dir = 1 if action == "BUY_LONG" else -1 if action == "SELL_SHORT" else 0
        if target_dir == 0 or confidence < min_confidence:
            return None  # 觀望或信心度不足門檻

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        pos_dir = self.position.direction if self.position else 0
        latest_bar_ts = df_bars["timestamp"][-1] if df_bars.height > 0 and "timestamp" in df_bars.columns else None

        # 若方向一致，續抱
        if pos_dir == target_dir:
            return None

        # 防雙巴鐵律 1：同根 5m K 棒不重複開倉，且停損出場後強制執行 5 分鐘 (300秒) 冷靜期 (Anti-Whipsaw Cooldown)
        if pos_dir == 0:
            if (
                self._last_entry_bar_ts is not None
                and latest_bar_ts is not None
                and latest_bar_ts <= self._last_entry_bar_ts
            ):
                return None
            if (
                self._last_stop_out_bar_ts is not None
                and latest_bar_ts is not None
                and latest_bar_ts <= self._last_stop_out_bar_ts
            ):
                return None
            if (
                time.time() - self._last_stop_out_ts < STOP_OUT_COOLDOWN_SEC
                and target_dir == self._last_stop_out_dir
            ):
                return None

        # 防雙巴鐵律 2：若已持有部位，必須反向訊號具備高確信度才允許反手，避免在盤整區被臨界訊號來回巴
        closed_msg = ""
        if pos_dir != 0:
            if confidence < REVERSE_MIN_CONFIDENCE:
                return None
            tr = self._close_current_position(current_price, exit_idx=-1, reason=f"AI反手平倉 ({source_note})")
            closed_msg = f"已平倉反向單 @ {tr.exit_price:,.1f} (損益 {tr.net_pnl:+,.0f} 元) ➔ "
            self._log("ORDER_FILL", f"AI反手平倉：{tr.exit_reason} @ {tr.exit_price:,.1f}，淨損益: {tr.net_pnl:,.0f} 元", price=tr.exit_price, pnl=tr.net_pnl)
            # 平倉已是既成事實：不論後續新倉是否開成，都必須立即落盤並重新檢核風控
            risk_locked = (
                self.check_daily_loss_limit(current_price)
                or self.check_daily_profit_target(current_price)
                or self.check_model_degradation()
            )
            self.save_state_to_disk()
            if risk_locked:
                return f"{closed_msg}風控鎖定，未開立新倉"

        # 讀取 ATR
        atr_entry_val = {}
        if df_bars.height > 0:
            cols = self._bars_with_exit_atr(df_bars)
            atr_entry_val = read_atr_values(cols, self.strategy, max(0, df_bars.height - 2))

        # 開新倉
        opened = self._open_new_position(target_dir, current_price, now_str, reason=f"AI訊號進場 ({source_note}, 信心度{confidence:.0f}%)", entry_atr=atr_entry_val)
        if opened:
            self._last_entry_bar_ts = latest_bar_ts
            self.save_state_to_disk()
            dir_zh = "多單" if target_dir == 1 else "空單"
            return f"{closed_msg}成功開立 {dir_zh} {self.position.contracts} 口 @ {self.position.entry_price:,.1f}"
        return f"{closed_msg}新倉未成立（詳見風控日誌）" if closed_msg else None

    def sync_with_rule_signal(
        self,
        action: Literal["BUY_LONG", "SELL_SHORT", "WAIT"],
        current_price: float,
        df_bars: pl.DataFrame,
        flatten: bool,
        source_note: str = "開盤區間突破",
    ) -> Optional[str]:
        """
        依「規則訊號」（如開盤區間突破）驅動模擬倉：
        1. 風控檢查與既有持倉的停損 / 停利永遠優先（不受暫停或鎖定影響）。
        2. flatten=True（收盤前或日盤以外時段）時強制平倉，且不開新倉。
        3. 一個交易日最多進場一次；持倉中收到反向訊號不反手。
        """
        if current_price <= 0:
            return None
        self.last_price = current_price

        if self.check_daily_loss_limit(current_price) or self.check_daily_profit_target(current_price):
            self.save_state_to_disk()
            return "風控鎖定"

        stop_res = self.check_position_stops(current_price, df_bars)
        if stop_res:
            return stop_res

        if flatten:
            if self.position is None:
                return None
            if not self._price_matches_position(current_price):
                return None
            tr = self._close_current_position(current_price, exit_idx=-1, reason="orb_time_exit")
            msg = f"⏰ 收盤前強制平倉 @ {tr.exit_price:,.1f}，淨損益 {tr.net_pnl:+,.0f} 元（{source_note}不留倉）"
            self._log("ORDER_FILL", msg, price=tr.exit_price, pnl=tr.net_pnl)
            self.check_model_degradation()
            self.save_state_to_disk()
            return msg

        if not self.is_active or self.kill_switch_locked or self.daily_loss_locked or self.daily_profit_locked or self.degradation_locked:
            return "系統暫停或風控鎖定中"
        if time.time() - self._last_reset_ts < RESET_GRACE_SEC:
            return None

        target_dir = 1 if action == "BUY_LONG" else -1 if action == "SELL_SHORT" else 0
        if target_dir == 0 or self.position is not None:
            return None  # 觀望；或已有持倉（同向續抱、反向不反手）
        today = current_trading_day()
        if self._last_entry_trading_day == today:
            return None  # 一天只進場一次

        atr_entry_val = (
            read_atr_values(self._bars_with_exit_atr(df_bars), self.strategy, max(0, df_bars.height - 2))
            if df_bars.height > 0 else {}
        )
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        opened = self._open_new_position(target_dir, current_price, now_str, reason=f"規則訊號進場 ({source_note})", entry_atr=atr_entry_val)
        if not opened:
            return None
        self._last_entry_trading_day = today
        self.save_state_to_disk()
        dir_zh = "多單" if target_dir == 1 else "空單"
        return f"{source_note}：開立 {dir_zh} {self.position.contracts} 口 @ {self.position.entry_price:,.1f}"

    def save_state_to_disk(self, file_path: Optional[str | Path] = None) -> None:
        """將虛擬操盤帳戶狀態（資金、部位、已平倉交易、風控日誌）持久化儲存至本地磁碟"""
        p = Path(file_path) if file_path is not None else DEFAULT_STATE_PATH
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
        # 已平倉交易必須完整保存：realized_pnl 與總權益是由全部交易加總而來，截斷會讓重啟後的損益跳動
        trades_list = [asdict(t) for t in self.closed_trades]
        logs_list = [asdict(l) for l in self.audit_logs[-100:]]

        payload = {
            "is_active": self.is_active,
            "kill_switch_locked": self.kill_switch_locked,
            "daily_loss_locked": self.daily_loss_locked,
            "daily_profit_locked": self.daily_profit_locked,
            "degradation_locked": self.degradation_locked,
            "credibility_verified": self.credibility_verified,
            "order_safety_locked": getattr(self, "order_safety_locked", False),
            "custom_tp_points": getattr(self, "custom_tp_points", 0.0),
            "daily_loss_baseline_pnl": self._daily_loss_baseline_pnl,
            "trading_day": self._trading_day,
            "last_stop_out_ts": self._last_stop_out_ts,
            "last_stop_out_dir": self._last_stop_out_dir,
            "last_stop_out_bar_ts": _dt_to_str(self._last_stop_out_bar_ts),
            "last_entry_bar_ts": _dt_to_str(self._last_entry_bar_ts),
            "last_entry_trading_day": self._last_entry_trading_day,
            "initial_capital": self.initial_capital,
            "realized_pnl": self.realized_pnl,
            "strategy": self.strategy.model_dump(),
            "risk_config": asdict(self.risk_config),
            "position": pos_dict,
            "closed_trades": trades_list,
            "audit_logs": logs_list,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        # 原子寫入：先寫暫存檔再整檔替換，避免寫到一半中斷而留下損毀的 JSON
        content = json.dumps(payload, ensure_ascii=False, indent=2)
        tmp_path = p.with_name(f"{p.name}.{os.getpid()}.tmp")
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, p)
        # Cloud Run 無狀態：同步一份到 GCS，避免縮容或重新部署後狀態遺失（本機執行時不動作）
        state_sync.upload_in_background(p, content)

    def load_state_from_disk(self, file_path: Optional[str | Path] = None) -> bool:
        """
        從本地磁碟恢復虛擬操盤帳戶狀態。
        檔案不存在回傳 False；檔案損毀時會將其改名為 *.corrupt-<時間> 保留並記錄錯誤後回傳 False。
        """
        p = Path(file_path) if file_path is not None else DEFAULT_STATE_PATH
        # 雲端以 GCS 上的最新狀態為準（本機執行時不動作）
        state_sync.download_if_available(p)
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("狀態檔最外層不是 JSON 物件")
        except (OSError, ValueError) as ex:
            backup = p.with_name(f"{p.name}.corrupt-{datetime.now().strftime('%Y%m%d%H%M%S')}")
            logger.error("模擬倉狀態檔 %s 無法解析（%s），已改名保留為 %s", p, ex, backup.name)
            try:
                os.replace(p, backup)
            except OSError as rename_ex:
                logger.error("損毀狀態檔改名失敗：%s", rename_ex)
            return False
        try:
            self.is_active = data.get("is_active", False)
            self.kill_switch_locked = data.get("kill_switch_locked", False)
            self.daily_loss_locked = data.get("daily_loss_locked", False)
            self.daily_profit_locked = data.get("daily_profit_locked", False)
            self.degradation_locked = data.get("degradation_locked", False)
            # 缺少欄位時一律視為「尚未通過可信度檢核」
            self.credibility_verified = bool(data.get("credibility_verified", False))
            self.order_safety_locked = data.get("order_safety_locked", False)
            self.custom_tp_points = float(data.get("custom_tp_points", 0.0))
            self._daily_loss_baseline_pnl = float(data.get("daily_loss_baseline_pnl", 0.0))
            self._trading_day = data.get("trading_day") or None
            self._last_stop_out_ts = float(data.get("last_stop_out_ts", 0.0) or 0.0)
            self._last_stop_out_dir = int(data.get("last_stop_out_dir", 0) or 0)
            self._last_stop_out_bar_ts = _dt_from_str(data.get("last_stop_out_bar_ts"))
            self._last_entry_bar_ts = _dt_from_str(data.get("last_entry_bar_ts"))
            self._last_entry_trading_day = data.get("last_entry_trading_day") or None
            self.initial_capital = data.get("initial_capital", self.initial_capital)

            strat_data = data.get("strategy")
            if strat_data:
                try:
                    restored_strategy = StrategySchema.model_validate(strat_data)
                    if restored_strategy.exit.stop_loss is None and restored_strategy.exit.trailing_stop is None:
                        raise ValueError("存檔策略沒有任何停損設定")
                    self.strategy = restored_strategy
                    self.core_engine = EventDrivenEngine(self.strategy, self.contract, self.initial_capital)
                except Exception as ex:  # 存檔策略不合法：保留建構時傳入的安全預設策略
                    logger.warning("存檔中的策略無法套用（%s），沿用目前的預設策略", ex)

            rc_data = data.get("risk_config")
            if rc_data:
                try:
                    self.risk_config = RiskConfig(**rc_data)
                except TypeError as ex:
                    logger.warning("存檔中的風控設定無法套用（%s），沿用目前的風控設定", ex)

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
        except Exception as ex:
            # 內容欄位不合預期（例如舊版格式）：原檔保留不動，記錄錯誤讓問題可被察覺
            logger.exception("模擬倉狀態檔 %s 內容無法還原：%s", p, ex)
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
        df_ind = IndicatorEngine.compute_indicators(df_bars, ind_specs, base_timeframe=self.strategy.timeframe)
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

        # 暫停或風控鎖定時只禁止「開新倉」；既有部位的平倉訊號、停損、停利仍須持續執行
        can_open_new = not (
            not self.is_active
            or self.kill_switch_locked
            or self.daily_loss_locked
            or self.daily_profit_locked
            or self.degradation_locked
        )
        if not can_open_new and self.position is None:
            return

        # 2. 若有前一根 K 棒收盤產生的待執行訊號，於新 K 棒開盤價執行 (fill_at = next_bar_open)
        if self.pending_signal is not None and (self.last_processed_bar_ts is None or bar_ts > self.last_processed_bar_ts):
            target_dir, sig_reason = self.pending_signal
            self.pending_signal = None
            if not can_open_new and target_dir != 0:
                # 鎖定期間不反手、不建倉：進場訊號直接作廢，部位改由停損/停利守護
                target_dir, sig_reason = None, ""

            if target_dir is not None and self.position is not None:
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

            if target_dir is not None and target_dir != 0:
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

            if not can_open_new:
                # 鎖定期間不產生任何進場訊號
                self.last_processed_bar_ts = bar_ts
                return

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
                reason = self._entry_reason(target_dir=1, pos_dir=pos_dir)
                self.pending_signal = (1, reason)
                self._log("SIGNAL", f"產生多方進場訊號 ({reason}，將於次根開盤執行)", price=c)
            elif short_sig and pos_dir >= 0:
                reason = self._entry_reason(target_dir=-1, pos_dir=pos_dir)
                self.pending_signal = (-1, reason)
                self._log("SIGNAL", f"產生空方進場訊號 ({reason}，將於次根開盤執行)", price=c)

            self.last_processed_bar_ts = bar_ts

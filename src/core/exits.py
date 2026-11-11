# src/core/exits.py
"""
共用出場價位計算核心（回測引擎 EventDrivenEngine 與紙上交易 PaperTradingEngine 共用）

設計原則：
1. 停損 / 移動停損 / 停利的價位計算只寫一份，避免回測與實盤邏輯分歧。
2. 支援 fixed_points / points / percent / atr_multiple。
3. 安全優先：若策略要求的停損距離無法計算（例如 ATR 尚在暖機期為 NaN），
   回傳 None，由呼叫端「拒絕進場」，絕不在無停損狀態下建倉。
4. 結算日強制平倉（time_exit.before_settlement）與風險比例口數計算。
"""
from __future__ import annotations

import logging
import math
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional, Tuple

import polars as pl

from src.core.indicators import IndicatorEngine
from src.schema.strategy import StrategySchema

# 台指期月契約最後交易日（第三個星期三）日盤收盤時間
SETTLEMENT_CLOSE_TIME = time(13, 30)
# 日盤結束後 15:00 開始的夜盤已屬於次月契約
NIGHT_SESSION_START = time(15, 0)

_TIMEFRAME_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "60m": 60, "1D": 1440}

logger = logging.getLogger(__name__)
# 結算日保護無法判斷時間時，每種時間戳型別只警告一次（避免逐根 K 棒洗版）
_SETTLEMENT_WARNED: set = set()


# ---------------------------------------------------------------------------
# ATR 欄位
# ---------------------------------------------------------------------------
def atr_column_name(length: int) -> str:
    return f"__exit_atr_{int(length)}"


def required_atr_lengths(strategy: StrategySchema) -> List[int]:
    """列出出場規則需要的 ATR 週期（去重）。"""
    lengths: List[int] = []
    ex = strategy.exit
    for spec in (ex.stop_loss, ex.take_profit, ex.trailing_stop):
        if spec is not None and spec.type == "atr_multiple":
            ln = int(getattr(spec, "atr_length", 14) or 14)
            if ln not in lengths:
                lengths.append(ln)
    return lengths


def add_exit_atr_columns(df: pl.DataFrame, strategy: StrategySchema) -> pl.DataFrame:
    """為 atr_multiple 出場規則補上內部 ATR 欄位（只用當根及以前資料，無未來函數）。"""
    lengths = required_atr_lengths(strategy)
    if not lengths or df.height == 0:
        return df
    cols = []
    for ln in lengths:
        name = atr_column_name(ln)
        if name not in df.columns:
            cols.append(IndicatorEngine.calculate_atr(df, ln).alias(name))
    return df.with_columns(cols) if cols else df


def read_atr_values(cols: Dict[str, List[Any]], strategy: StrategySchema, idx: int) -> Dict[int, float]:
    """讀取第 idx 根 K 棒的 ATR 值（idx < 0 或缺值時為 NaN）。"""
    out: Dict[int, float] = {}
    for ln in required_atr_lengths(strategy):
        name = atr_column_name(ln)
        val = float("nan")
        if idx >= 0 and name in cols and idx < len(cols[name]):
            raw = cols[name][idx]
            if raw is not None:
                val = float(raw)
        out[ln] = val
    return out


# ---------------------------------------------------------------------------
# 距離與價位計算
# ---------------------------------------------------------------------------
def _distance(spec: Any, entry_price: float, atr_values: Dict[int, float]) -> Optional[float]:
    """依規格回傳點數距離；無法計算時回傳 None。"""
    if spec is None:
        return None
    s_type = spec.type
    value = getattr(spec, "value", None)
    if value is None:
        return None
    if s_type in ("fixed_points", "points"):
        return float(value)
    if s_type == "percent":
        return entry_price * (float(value) / 100.0)
    if s_type == "atr_multiple":
        ln = int(getattr(spec, "atr_length", 14) or 14)
        atr = atr_values.get(ln, float("nan"))
        if atr is None or math.isnan(atr) or atr <= 0:
            return None
        return float(value) * atr
    return None  # indicator_cross 等非價位型


def initial_stop_distance(strategy: StrategySchema, entry_price: float, atr_values: Dict[int, float]) -> Optional[float]:
    """進場當下的固定停損距離（供風險比例口數計算）。"""
    return _distance(strategy.exit.stop_loss, entry_price, atr_values)


def can_open_position(
    strategy: StrategySchema,
    entry_price: float,
    atr_values: Dict[int, float],
    direction: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    安全檢核：確認進場時至少有一道可生效的「價位型」停損。
    - 價位型停損（fixed/percent/atr）必須能算出距離。
    - indicator_cross 停損只在 K 棒收盤才判斷，盤中沒有價位保護，
      因此必須同時有一道算得出距離的移動停損；若提供 direction（1 多 / -1 空），
      還必須有該方向的指標出場條件。
    - 只有移動停損時，移動停損距離必須能算出（啟動前以 entry ∓ 距離 作為初始硬停損）。
    """
    sl = strategy.exit.stop_loss
    ts = strategy.exit.trailing_stop
    if sl is not None:
        if sl.type == "indicator_cross":
            if direction == 1 and sl.long is None:
                return False, "indicator_cross 停損缺少 long 出場條件，拒絕建立多單"
            if direction == -1 and sl.short is None:
                return False, "indicator_cross 停損缺少 short 出場條件，拒絕建立空單"
            if ts is None:
                return False, "indicator_cross 停損沒有搭配價位型保護（trailing_stop），拒絕進場"
            if _distance(ts, entry_price, atr_values) is None:
                return False, f"indicator_cross 停損搭配的價位型保護 {ts.type} 目前無法計算距離（ATR 暖機中或數值無效），拒絕進場"
            return True, ""
        if _distance(sl, entry_price, atr_values) is not None:
            return True, ""
        return False, f"停損類型 {sl.type} 目前無法計算距離（ATR 暖機中或數值無效），拒絕進場"
    if ts is not None:
        if _distance(ts, entry_price, atr_values) is not None:
            return True, ""
        return False, f"移動停損類型 {ts.type} 目前無法計算距離（ATR 暖機中或數值無效），拒絕進場"
    return False, "策略沒有任何停損設定，拒絕進場"


def compute_exit_levels(
    strategy: StrategySchema,
    direction: int,
    entry_price: float,
    highest_since_entry: float,
    lowest_since_entry: float,
    entry_atr: Dict[int, float],
    current_atr: Optional[Dict[int, float]] = None,
) -> Tuple[Optional[float], Optional[float], str]:
    """
    計算目前的停損價、停利價與停損原因。
    - 固定停損 / 停利：使用進場當下的 ATR（entry_atr），進場後價位固定不變。
    - 移動停損：使用最近一根已收盤 K 棒的 ATR（current_atr），缺值時退回 entry_atr。
    - 沒有價位型停損（只有移動停損，或 stop_loss 為 indicator_cross）時，
      移動停損啟動前以 entry ∓ 移動停損距離 作為初始硬停損（原因標示為 stop_loss），
      確保任何時刻都有一道價位保護。
    """
    ex = strategy.exit
    sl_spec, tp_spec, ts_spec = ex.stop_loss, ex.take_profit, ex.trailing_stop

    stop_price: Optional[float] = None
    stop_reason = "stop_loss"

    sl_dist = _distance(sl_spec, entry_price, entry_atr)
    if sl_dist is not None:
        stop_price = (entry_price - sl_dist) if direction == 1 else (entry_price + sl_dist)

    if ts_spec is not None:
        ts_dist = _distance(ts_spec, entry_price, current_atr or {})
        if ts_dist is None:
            ts_dist = _distance(ts_spec, entry_price, entry_atr)
        if stop_price is None:
            # 啟動前的初始硬停損：優先用進場當下的 ATR，讓價位在啟動前固定不變
            init_dist = _distance(ts_spec, entry_price, entry_atr)
            if init_dist is None:
                init_dist = ts_dist
            if init_dist is not None:
                stop_price = (entry_price - init_dist) if direction == 1 else (entry_price + init_dist)
        if ts_dist is not None:
            if direction == 1 and (highest_since_entry - entry_price) >= ts_spec.activation_points:
                cand = highest_since_entry - ts_dist
                if stop_price is None or cand > stop_price:
                    stop_price, stop_reason = cand, "trailing_stop"
            elif direction == -1 and (entry_price - lowest_since_entry) >= ts_spec.activation_points:
                cand = lowest_since_entry + ts_dist
                if stop_price is None or cand < stop_price:
                    stop_price, stop_reason = cand, "trailing_stop"

    tp_price: Optional[float] = None
    tp_dist = _distance(tp_spec, entry_price, entry_atr)
    if tp_dist is not None:
        tp_price = (entry_price + tp_dist) if direction == 1 else (entry_price - tp_dist)

    return stop_price, tp_price, stop_reason


# ---------------------------------------------------------------------------
# 口數計算
# ---------------------------------------------------------------------------
def compute_contracts(
    strategy: StrategySchema,
    equity: float,
    point_value: float,
    entry_price: float,
    atr_values: Dict[int, float],
    max_contracts: Optional[int] = None,
    initial_margin: Optional[float] = None,
) -> int:
    """
    依 position_sizing 計算口數。
    - 若提供 initial_margin（每口原始保證金），口數不得超過 floor(權益 ÷ 保證金)；
      權益 ≤ 0 或連 1 口保證金都不夠時回傳 0（不進場）。
    - fixed：使用 contracts。
    - risk_percent：floor(權益 × risk_pct ÷ (停損點數 × 每點價值))，上限為 contracts 欄位（與 max_contracts）。
      若單口風險已超過允許金額，回傳 0（不進場）。
      若停損距離無法量化（例如 indicator_cross），保守使用 1 口。
    """
    ps = strategy.position_sizing
    cap = ps.contracts
    if max_contracts is not None:
        cap = min(cap, max_contracts)
    if initial_margin is not None and initial_margin > 0:
        if equity is None or math.isnan(equity) or equity <= 0:
            return 0
        cap = min(cap, int(math.floor(equity / initial_margin)))
    if cap <= 0:
        return 0

    if ps.type != "risk_percent" or not ps.risk_pct:
        return max(0, cap)

    dist = initial_stop_distance(strategy, entry_price, atr_values)
    if dist is None:
        ts = strategy.exit.trailing_stop
        dist = _distance(ts, entry_price, atr_values) if ts is not None else None
    if dist is None or dist <= 0:
        return min(1, cap)

    risk_budget = max(0.0, equity) * ps.risk_pct
    per_contract_risk = dist * point_value
    n = int(math.floor(risk_budget / per_contract_risk)) if per_contract_risk > 0 else 0
    return max(0, min(n, cap))


# ---------------------------------------------------------------------------
# 結算日（第三個星期三）
# ---------------------------------------------------------------------------
def third_wednesday(year: int, month: int) -> date:
    first = date(year, month, 1)
    offset = (2 - first.weekday()) % 7  # Monday=0, Wednesday=2
    return first + timedelta(days=offset + 14)


def is_settlement_day(d: date) -> bool:
    """
    是否為台指期月契約最後交易日（第三個星期三）。
    註：若遇國定假日順延，需另行維護假日表（目前未處理）。
    """
    return d == third_wednesday(d.year, d.month)


def timeframe_minutes(timeframe: str) -> int:
    return _TIMEFRAME_MINUTES.get(timeframe, 60)


def settlement_cutoff_state(
    strategy: StrategySchema,
    settlement_rule: str,
    bar_ts: Any,
) -> Tuple[bool, bool]:
    """
    回傳 (force_exit, block_entry)：
    - force_exit：此根 K 棒收盤時應強制平倉（結算日日盤收盤前 N 分鐘）。
    - block_entry：此根收盤後不得再排入新倉（避免持倉進入結算）。
    bar_ts 為 K 棒開始時間（group_by_dynamic label=left）。
    若已啟用結算日保護但時間戳無法判斷（None / 非 datetime），會記錄 warning 並回傳 (False, False)；
    不 raise，以免中斷模擬倉即時路徑。
    """
    te = strategy.exit.time_exit
    if not te.before_settlement or settlement_rule != "third_wednesday":
        return False, False
    if isinstance(bar_ts, str):
        try:
            bar_ts = datetime.fromisoformat(bar_ts.strip())
        except ValueError:
            pass
    if not isinstance(bar_ts, datetime):
        kind = type(bar_ts).__name__
        if kind not in _SETTLEMENT_WARNED:
            _SETTLEMENT_WARNED.add(kind)
            logger.warning(
                "結算日強制平倉保護已啟用 (before_settlement=True)，但 K 棒時間戳型別為 %s，無法判斷結算日；"
                "此保護目前未生效（策略 %s）。請提供 datetime 型別的 timestamp 欄位。",
                kind,
                getattr(strategy, "id", "?"),
            )
        return False, False
    if not is_settlement_day(bar_ts.date()):
        return False, False
    if bar_ts.time() >= NIGHT_SESSION_START or bar_ts.time() >= SETTLEMENT_CLOSE_TIME:
        # 13:30 之後（含夜盤）已是次月契約，不受限制
        return False, False

    cutoff = datetime.combine(bar_ts.date(), SETTLEMENT_CLOSE_TIME) - timedelta(minutes=te.minutes_before_close)
    tf_min = timeframe_minutes(strategy.timeframe)
    bar_end = bar_ts + timedelta(minutes=tf_min)
    next_bar_end = bar_end + timedelta(minutes=tf_min)

    force_exit = bar_end >= cutoff
    # 若下一根 K 棒結束時已超過截止時間，就不要在下一根開盤建新倉
    block_entry = next_bar_end >= cutoff
    return force_exit, block_entry


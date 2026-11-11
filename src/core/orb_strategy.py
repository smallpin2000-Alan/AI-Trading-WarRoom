# src/core/orb_strategy.py
"""
開盤區間突破（Opening Range Breakout, ORB）規則 —— 取代 AI 判斷多空方向。

規則（只做日盤）：
1. 開盤區間：日盤開盤後前 range_minutes 分鐘（預設 08:45~09:15）的最高價 / 最低價。
2. 進場：區間完成後，第一根「收盤價」突破上緣 → 做多；跌破下緣 → 做空。
   當日只認第一次突破，之後反向突破不反手；last_entry 之後的突破不進場。
3. 出場：沿用模擬倉的 ATR 停損 / 停利 / 移動鎖利；flatten_at（預設 13:30）強制平倉，不留倉過夜。

資料慣例：1 分 K 的時間標在「分鐘結束」（08:46 代表 08:45:00~08:45:59），
與期交所歷史資料（taifex_history）及台新 K 棒一致。

參考：Tsai et al. (2019) "Assessing the Profitability of Timely Opening Range Breakout on
Index Futures Markets", IEEE Access 7:32061-32071（台指期 2003~2013 實證）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Literal, Optional

import polars as pl

from src.core.exits import compute_exit_levels
from src.core.indicators import IndicatorEngine
from src.schema.strategy import StrategySchema

Action = Literal["BUY_LONG", "SELL_SHORT", "WAIT"]


@dataclass(frozen=True)
class OrbParams:
    session_open: time = time(8, 45)
    session_close: time = time(13, 45)
    range_minutes: int = 30
    last_entry: time = time(12, 30)
    flatten_at: time = time(13, 30)
    buffer_points: float = 0.0      # 突破需超過區間邊界的點數（0 = 收盤價超過即算）
    cost_points: float = 2.0        # 回測用：來回手續費 + 稅 + 滑價（點）
    atr_length: int = 14            # 回測用：以 5 分 K 計算 ATR


@dataclass(frozen=True)
class OrbState:
    trading_date: date
    range_ready: bool
    range_high: Optional[float]
    range_low: Optional[float]
    action: Action
    flatten: bool
    breakout_ts: Optional[datetime] = None
    breakout_price: Optional[float] = None
    reason: str = ""


def _session_bounds(d: date, p: OrbParams) -> tuple[datetime, datetime, datetime, datetime, datetime]:
    open_dt = datetime.combine(d, p.session_open)
    return (
        open_dt,
        open_dt + timedelta(minutes=p.range_minutes),
        datetime.combine(d, p.last_entry),
        datetime.combine(d, p.flatten_at),
        datetime.combine(d, p.session_close),
    )


def _day_bars(bars_1m: pl.DataFrame, d: date, p: OrbParams, until: Optional[datetime] = None) -> pl.DataFrame:
    open_dt, _, _, _, close_dt = _session_bounds(d, p)
    end = min(close_dt, until) if until else close_dt
    return bars_1m.filter((pl.col("timestamp") > open_dt) & (pl.col("timestamp") <= end)).sort("timestamp")


def _find_breakout(day: pl.DataFrame, d: date, p: OrbParams) -> tuple[bool, Optional[float], Optional[float], int, Optional[int]]:
    """回傳 (區間是否完成, 上緣, 下緣, 突破方向, 突破那根在 day 中的索引)"""
    _, range_end, last_entry_dt, _, _ = _session_bounds(d, p)
    ts = day["timestamp"].to_list()
    rng = day.filter(pl.col("timestamp") <= range_end)
    ready = any(t >= range_end for t in ts)
    if rng.height == 0 or not ready:
        return ready, None, None, 0, None
    hi, lo = float(rng["high"].max()), float(rng["low"].min())
    closes = day["close"].to_list()
    for i, t in enumerate(ts):
        if t <= range_end or t > last_entry_dt:
            continue
        if closes[i] > hi + p.buffer_points:
            return True, hi, lo, 1, i
        if closes[i] < lo - p.buffer_points:
            return True, hi, lo, -1, i
    return True, hi, lo, 0, None


def compute_orb_state(bars_1m: pl.DataFrame, now: datetime, params: OrbParams = OrbParams()) -> OrbState:
    """以「現在時間」之前已完成的 1 分 K 計算當日 ORB 狀態（不使用未來資料）。"""
    p = params
    d = now.date()
    open_dt, _, _, flatten_dt, close_dt = _session_bounds(d, p)
    in_day_session = open_dt <= now <= close_dt
    flatten = (not in_day_session) or now >= flatten_dt

    day = _day_bars(bars_1m, d, p, until=now)
    ready, hi, lo, direction, idx = _find_breakout(day, d, p)

    if not ready:
        return OrbState(d, False, hi, lo, "WAIT", flatten, reason="開盤區間尚未完成")
    if direction == 0:
        return OrbState(d, True, hi, lo, "WAIT", flatten, reason="價格仍在開盤區間內")
    row = day.row(idx, named=True)
    action: Action = "BUY_LONG" if direction == 1 else "SELL_SHORT"
    if flatten:
        action = "WAIT"
    return OrbState(
        d, True, hi, lo, action, flatten,
        breakout_ts=row["timestamp"], breakout_price=float(row["close"]),
        reason=("收盤突破開盤區間上緣" if direction == 1 else "收盤跌破開盤區間下緣"),
    )


def _atr_before(bars_1m: pl.DataFrame, before: datetime, length: int) -> float:
    """以 before 之前已完成的 5 分 K 計算 ATR（不含未來資料）。"""
    hist = bars_1m.filter(pl.col("timestamp") <= before).with_columns(
        (pl.col("timestamp") - pl.duration(minutes=1)).alias("start")
    )
    m5 = hist.group_by_dynamic("start", every="5m", label="left", closed="left").agg(
        pl.col("open").first(), pl.col("high").max(), pl.col("low").min(), pl.col("close").last()
    )
    if m5.height <= length:
        return float("nan")
    val = m5.select(IndicatorEngine.calculate_atr(m5, length)).to_series()[-1]
    return float(val) if val is not None else float("nan")


def backtest_orb(
    bars_1m: pl.DataFrame,
    params: OrbParams = OrbParams(),
    strategy: Optional[StrategySchema] = None,
    atr_value: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    逐日回測 ORB：突破那根收盤後，以下一根開盤價進場；出場用 strategy 的出場設定（預設為模擬倉的 ATR 設定），
    同一根 K 棒同時觸及停損與停利時以停損計（保守）；flatten_at 以該根收盤價平倉。
    atr_value 有給時固定使用（測試用），否則以進場前的 5 分 K ATR 計算。
    """
    if strategy is None:
        from src.core.paper_trader import default_ai_strategy  # 延遲匯入避免循環相依

        strategy = default_ai_strategy("TMF")
    p = params
    df = bars_1m.sort("timestamp")
    dates = sorted({t.date() for t in df["timestamp"].to_list()})
    trades: List[Dict[str, Any]] = []

    for d in dates:
        day = _day_bars(df, d, p)
        ready, hi, lo, direction, idx = _find_breakout(day, d, p)
        if not ready or direction == 0 or idx is None or idx + 1 >= day.height:
            continue
        rows = day.rows(named=True)
        entry_row = rows[idx + 1]
        entry = float(entry_row["open"])
        atr = atr_value if atr_value is not None else _atr_before(df, rows[idx]["timestamp"], p.atr_length)
        if atr is None or math.isnan(atr) or atr <= 0:
            continue
        atr_map = {ln: atr for ln in (strategy.exit.stop_loss.atr_length if strategy.exit.stop_loss else 14,
                                      strategy.exit.trailing_stop.atr_length if strategy.exit.trailing_stop else 14,
                                      strategy.exit.take_profit.atr_length if strategy.exit.take_profit else 14)}
        flatten_dt = datetime.combine(d, p.flatten_at)
        hi_since = lo_since = entry
        exit_price, exit_reason, exit_time = None, "", None

        for r in rows[idx + 1:]:
            stop, tp, stop_reason = compute_exit_levels(strategy, direction, entry, hi_since, lo_since, atr_map, atr_map)
            bar_open, bar_high, bar_low = float(r["open"]), float(r["high"]), float(r["low"])
            if stop is not None and ((direction == 1 and bar_low <= stop) or (direction == -1 and bar_high >= stop)):
                exit_price = min(stop, bar_open) if direction == 1 else max(stop, bar_open)
                exit_reason, exit_time = stop_reason, r["timestamp"]
                break
            if tp is not None and ((direction == 1 and bar_high >= tp) or (direction == -1 and bar_low <= tp)):
                exit_price = max(tp, bar_open) if direction == 1 else min(tp, bar_open)
                exit_reason, exit_time = "take_profit", r["timestamp"]
                break
            if r["timestamp"] >= flatten_dt:
                exit_price, exit_reason, exit_time = float(r["close"]), "time_exit", r["timestamp"]
                break
            hi_since, lo_since = max(hi_since, bar_high), min(lo_since, bar_low)
        if exit_price is None:  # 資料在收盤前中斷：以最後一根收盤平倉
            exit_price, exit_reason, exit_time = float(rows[-1]["close"]), "end_of_data", rows[-1]["timestamp"]

        trades.append({
            "date": d,
            "direction": direction,
            "range_high": hi,
            "range_low": lo,
            "entry_time": entry_row["timestamp"],
            "entry_price": entry,
            "exit_time": exit_time,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "atr": atr,
            "pnl_points": (exit_price - entry) * direction - p.cost_points,
        })
    return trades

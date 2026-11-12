# src/core/key_levels.py
"""
客觀關鍵價位（取代 AI 預估線）：全部由公式計算，同樣資料每次結果相同，且可用歷史資料驗證。

- 前一日日盤的最高 / 最低 / 收盤價：常見的支撐壓力參考。
- 當日日盤 VWAP（成交量加權平均價）：當天市場平均成本，常用來判斷多空強弱。
- 預期波動區間：今日開盤價 ± 平均日振幅（前 adr_days 個交易日日盤高低差平均）× adr_mult。
  這是「範圍」而非方向預測，可用 expected_range_hit_rate() 實測歷史上整天落在區間內的比例。

資料慣例：1 分 K 時間標在分鐘結束（08:46 代表 08:45:00~08:45:59）。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from typing import List, Optional, Tuple

import polars as pl

DEFAULT_ADR_DAYS = 10
DEFAULT_ADR_MULT = 1.0  # 9/24～11/11 實測：0.75 倍僅 25% 的日子整天落在區間內，1.0 倍為 60%


@dataclass(frozen=True)
class KeyLevels:
    today_open: Optional[float] = None
    prev_high: Optional[float] = None
    prev_low: Optional[float] = None
    prev_close: Optional[float] = None
    vwap: Optional[float] = None
    adr: Optional[float] = None
    exp_upper: Optional[float] = None
    exp_lower: Optional[float] = None


def _day_session(bars_1m: pl.DataFrame, open_t: time, close_t: time) -> pl.DataFrame:
    """只保留日盤 K 棒，並加上 date 欄位（時間在 (開盤, 收盤] 之間）"""
    t = pl.col("timestamp")
    secs = t.dt.hour().cast(pl.Int32) * 3600 + t.dt.minute().cast(pl.Int32) * 60 + t.dt.second().cast(pl.Int32)
    o = open_t.hour * 3600 + open_t.minute * 60
    c = close_t.hour * 3600 + close_t.minute * 60
    return bars_1m.filter((secs > o) & (secs <= c)).with_columns(t.dt.date().alias("date")).sort("timestamp")


def _daily_table(day_bars: pl.DataFrame) -> pl.DataFrame:
    return (
        day_bars.group_by("date", maintain_order=True)
        .agg(
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
        )
        .sort("date")
    )


def compute_key_levels(
    bars_1m: pl.DataFrame,
    now: datetime,
    open_t: time,
    close_t: time,
    adr_days: int = DEFAULT_ADR_DAYS,
    adr_mult: float = DEFAULT_ADR_MULT,
) -> KeyLevels:
    """以 now 之前已完成的 1 分 K 計算今日的關鍵價位（不使用未來資料）。缺資料的欄位為 None。"""
    if bars_1m is None or bars_1m.height == 0:
        return KeyLevels()
    day = _day_session(bars_1m.filter(pl.col("timestamp") <= now), open_t, close_t)
    if day.height == 0:
        return KeyLevels()
    today = now.date()
    daily = _daily_table(day)
    past = daily.filter(pl.col("date") < today)
    today_bars = day.filter(pl.col("date") == today)

    prev = past.row(-1, named=True) if past.height else None
    today_open = float(today_bars["open"][0]) if today_bars.height else None

    vwap = None
    if today_bars.height:
        vol = today_bars["volume"].cast(pl.Float64)
        if float(vol.sum()) > 0:
            typical = (today_bars["high"] + today_bars["low"] + today_bars["close"]) / 3.0
            vwap = float((typical * vol).sum() / vol.sum())

    adr = exp_upper = exp_lower = None
    if past.height >= adr_days:
        recent = past.tail(adr_days)
        adr = float((recent["high"] - recent["low"]).mean())
        if today_open is not None:
            exp_upper = today_open + adr * adr_mult
            exp_lower = today_open - adr * adr_mult

    return KeyLevels(
        today_open=today_open,
        prev_high=float(prev["high"]) if prev else None,
        prev_low=float(prev["low"]) if prev else None,
        prev_close=float(prev["close"]) if prev else None,
        vwap=vwap,
        adr=adr,
        exp_upper=exp_upper,
        exp_lower=exp_lower,
    )


def expected_range_hit_rate(
    bars_1m: pl.DataFrame,
    open_t: time,
    close_t: time,
    adr_days: int = DEFAULT_ADR_DAYS,
    adr_mult: float = DEFAULT_ADR_MULT,
) -> Tuple[Optional[float], int]:
    """歷史上「整個日盤都沒有超出預期波動區間」的天數比例；回傳 (比例, 可評估天數)。"""
    if bars_1m is None or bars_1m.height == 0:
        return None, 0
    daily = _daily_table(_day_session(bars_1m, open_t, close_t))
    ranges: List[float] = (daily["high"] - daily["low"]).to_list()
    opens, highs, lows = daily["open"].to_list(), daily["high"].to_list(), daily["low"].to_list()
    hits = total = 0
    for i in range(adr_days, daily.height):
        adr = sum(ranges[i - adr_days:i]) / adr_days
        upper, lower = opens[i] + adr * adr_mult, opens[i] - adr * adr_mult
        total += 1
        hits += int(highs[i] <= upper and lows[i] >= lower)
    return (hits / total if total else None), total

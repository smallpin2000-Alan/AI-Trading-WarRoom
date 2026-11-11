# src/core/bar_builder.py
from __future__ import annotations
import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import List, Optional, Literal
import polars as pl


@dataclass
class TickData:
    timestamp: datetime
    price: float
    volume: int
    side: int = 0          # 1: 外盤主動買進(紅), -1: 內盤主動賣出(綠), 0: 中性(白)
    bid_price: float = 0.0 # 當時委買一價
    ask_price: float = 0.0 # 當時委賣一價


class SharedBarBuilder:
    """
    共用 K 棒合成器 (回測與實盤共用同一套切分邏輯)
    - 支援台指期日盤 (08:45 ~ 13:45) 與夜盤 (15:00 ~ 隔日 05:00) 時段過濾。
    - 支援將逐筆 Tick 合成指定週期：分鐘 (Nm，如 1m/5m/15m/30m/60m)、小時 (Nh) 與日 K (1D)。
      其他週期字串一律 raise ValueError，不會默默產生對齊錯誤的 K 棒。
    - 分鐘/小時週期預設以整點對齊（與既有回測行為相同）；30m/60m 在日盤 08:45 開盤會因此多出一根
      08:45~09:00 殘棒，如需以開盤時間為錨點對齊（08:45、09:45…；夜盤仍以 15:00 整點對齊），
      請傳入 session_aligned=True。
    - 1D 於 day_and_night 時段以「交易日」歸屬：15:00 之後的夜盤（含跨午夜至 05:00）併入下一個交易日，
      週五夜盤與週六凌晨併入下週一；僅依週末推算，不含國定假日行事曆。
    """

    DAY_OPEN = time(8, 45)
    DAY_CLOSE = time(13, 45)
    NIGHT_OPEN = time(15, 0)
    NIGHT_CLOSE = time(5, 0)

    @classmethod
    def is_valid_session_time(
        cls, dt: datetime, session: Literal["day_only", "day_and_night"] = "day_and_night"
    ) -> bool:
        t = dt.time()
        in_day = cls.DAY_OPEN <= t <= cls.DAY_CLOSE
        if session == "day_only":
            return in_day
        in_night = (t >= cls.NIGHT_OPEN) or (t <= cls.NIGHT_CLOSE)
        return in_day or in_night

    _OHLCV_AGGS = [
        pl.col("price").first().alias("open"),
        pl.col("price").max().alias("high"),
        pl.col("price").min().alias("low"),
        pl.col("price").last().alias("close"),
        pl.col("volume").sum().alias("volume"),
    ]

    @staticmethod
    def _parse_timeframe(timeframe: str) -> tuple[str, int]:
        """解析週期字串，回傳 (種類, 分鐘數)；種類為 "intraday" 或 "day"。不支援者 raise ValueError。"""
        tf = str(timeframe or "").strip()
        if tf.lower() == "1d":
            return "day", 1440
        m = re.fullmatch(r"(\d+)([mh])", tf)
        if m:
            minutes = int(m.group(1)) * (60 if m.group(2) == "h" else 1)
            if 0 < minutes < 1440:
                return "intraday", minutes
        raise ValueError(f"不支援的 K 棒週期: {timeframe!r}（僅支援 Nm、Nh 與 1D）")

    @classmethod
    def trading_date_of(cls, dt: datetime) -> datetime:
        """回傳該時間所屬交易日 (當日 00:00)：15:00 起的夜盤歸下一交易日，週六/週日順延至週一。"""
        d = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        if dt.time() >= cls.NIGHT_OPEN:
            d += timedelta(days=1)
        if d.weekday() == 5:
            d += timedelta(days=2)
        elif d.weekday() == 6:
            d += timedelta(days=1)
        return d

    @classmethod
    def ticks_to_bars(
        cls,
        ticks: List[TickData],
        timeframe: str = "60m",
        session: Literal["day_only", "day_and_night"] = "day_and_night",
        session_aligned: bool = False,
        day_anchor: Optional[time] = None,
    ) -> pl.DataFrame:
        """
        將 Tick 序列合成為標準 OHLCV DataFrame，供回測與實盤無縫共用。
        - session_aligned=True：日盤 K 棒以 day_anchor（預設 08:45；個股請傳 09:00）為錨點對齊，夜盤以整點對齊。
        - timeframe 不支援時 raise ValueError。
        """
        kind, tf_minutes = cls._parse_timeframe(timeframe)
        filtered = [tk for tk in ticks if cls.is_valid_session_time(tk.timestamp, session)]
        if not filtered:
            return pl.DataFrame(
                schema={
                    "timestamp": pl.Datetime,
                    "open": pl.Float64,
                    "high": pl.Float64,
                    "low": pl.Float64,
                    "close": pl.Float64,
                    "volume": pl.Int64,
                }
            )

        df = pl.DataFrame(
            {
                "timestamp": [tk.timestamp for tk in filtered],
                "price": [float(tk.price) for tk in filtered],
                "volume": [int(tk.volume) for tk in filtered],
            }
        ).sort("timestamp")

        if kind == "day":
            # 日 K：依交易日歸屬分組（day_only 沒有夜盤，等同日曆日）
            day_keys = [
                cls.trading_date_of(ts) if session == "day_and_night" else ts.replace(hour=0, minute=0, second=0, microsecond=0)
                for ts in df["timestamp"].to_list()
            ]
            bars = (
                df.with_columns(pl.Series("_trade_date", day_keys, dtype=df.schema["timestamp"]))
                .group_by("_trade_date", maintain_order=True)
                .agg(cls._OHLCV_AGGS)
                .rename({"_trade_date": "timestamp"})
                .sort("timestamp")
                .filter(pl.col("volume") > 0)
            )
            return bars

        every = f"{tf_minutes}m"
        anchor = day_anchor or cls.DAY_OPEN
        day_offset_min = (anchor.hour * 60 + anchor.minute) % tf_minutes
        if not session_aligned or day_offset_min == 0:
            return (
                df.group_by_dynamic("timestamp", every=every, closed="left", label="left")
                .agg(cls._OHLCV_AGGS)
                .filter(pl.col("volume") > 0)
            )

        # 以開盤時間為錨點：日盤與夜盤分開切棒，日盤視窗位移 day_offset_min 分鐘
        in_day = [cls.DAY_OPEN <= ts.time() <= cls.DAY_CLOSE for ts in df["timestamp"].to_list()]
        df = df.with_columns(pl.Series("_in_day", in_day))
        parts = []
        for is_day, offset in ((True, f"{day_offset_min}m"), (False, "0m")):
            part = df.filter(pl.col("_in_day") == is_day).drop("_in_day")
            if part.height == 0:
                continue
            parts.append(
                part.group_by_dynamic("timestamp", every=every, offset=offset, closed="left", label="left")
                .agg(cls._OHLCV_AGGS)
            )
        return pl.concat(parts).sort("timestamp").filter(pl.col("volume") > 0)

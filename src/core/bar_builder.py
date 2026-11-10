# src/core/bar_builder.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, time
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
    - 支援將逐筆 Tick 或 1分K 合成指定週期 (5m, 15m, 30m, 60m, 1D)。
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

    @classmethod
    def ticks_to_bars(
        cls,
        ticks: List[TickData],
        timeframe: str = "60m",
        session: Literal["day_only", "day_and_night"] = "day_and_night",
    ) -> pl.DataFrame:
        """
        將 Tick 序列合成為標準 OHLCV DataFrame，供回測與實盤無縫共用。
        """
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

        bars = (
            df.group_by_dynamic("timestamp", every=timeframe, closed="left", label="left")
            .agg(
                [
                    pl.col("price").first().alias("open"),
                    pl.col("price").max().alias("high"),
                    pl.col("price").min().alias("low"),
                    pl.col("price").last().alias("close"),
                    pl.col("volume").sum().alias("volume"),
                ]
            )
            .filter(pl.col("volume") > 0)
        )
        return bars

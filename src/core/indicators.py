# src/core/indicators.py
from __future__ import annotations
import polars as pl
import numpy as np
from typing import Dict, Any


class IndicatorEngine:
    """
    技術指標計算與快取引擎
    使用 Polars 進行高效能向量化計算，完整支援 StrategySchema 定義之 12 種技術指標：
    SMA, EMA, RSI, KD_K, KD_D, MACD, ATR, ADX, BB_UPPER, BB_LOWER, HIGHEST, LOWEST
    """

    @staticmethod
    def calculate_sma(df: pl.DataFrame, length: int, source: str = "close") -> pl.Series:
        """簡單移動平均線 (SMA)"""
        return df[source].rolling_mean(window_size=length)

    @staticmethod
    def calculate_ema(df: pl.DataFrame, length: int, source: str = "close") -> pl.Series:
        """指數移動平均線 (EMA)"""
        return df[source].ewm_mean(span=length, adjust=False)

    @staticmethod
    def calculate_rsi(df: pl.DataFrame, length: int = 14, source: str = "close") -> pl.Series:
        """相對強弱指標 (RSI)"""
        _col = df[source]
        delta = _col.diff()
        gain = delta.clip_lower(0) if hasattr(delta, "clip_lower") else delta.clip(lower_bound=0)
        loss = -delta.clip_upper(0) if hasattr(delta, "clip_upper") else -delta.clip(upper_bound=0)

        avg_gain = gain.ewm_mean(com=max(1, length - 1), adjust=False)
        avg_loss = loss.ewm_mean(com=max(1, length - 1), adjust=False)

        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    @staticmethod
    def calculate_atr(df: pl.DataFrame, length: int = 14) -> pl.Series:
        """真實波幅 (ATR)"""
        high = df["high"]
        low = df["low"]
        close = df["close"].shift(1)

        tr = pl.max_horizontal([
            high - low,
            (high - close).abs(),
            (low - close).abs(),
        ])
        return tr.rolling_mean(window_size=length)

    @staticmethod
    def calculate_kd(df: pl.DataFrame, length: int = 9) -> tuple[pl.Series, pl.Series]:
        """隨機指標 (KD_K, KD_D)"""
        low_min = df["low"].rolling_min(window_size=length)
        high_max = df["high"].rolling_max(window_size=length)
        denom = (high_max - low_min) + 1e-9
        rsv = ((df["close"] - low_min) / denom * 100.0).fill_null(50.0)
        k_series = rsv.ewm_mean(alpha=1.0 / 3.0, adjust=False)
        d_series = k_series.ewm_mean(alpha=1.0 / 3.0, adjust=False)
        return k_series, d_series

    @staticmethod
    def calculate_macd(df: pl.DataFrame, length: int = 12, source: str = "close") -> pl.Series:
        """MACD 差離值 (DIF = Fast EMA - Slow EMA)"""
        fast_span = max(2, length)
        slow_span = max(fast_span + 1, int(round(length * 26 / 12)))
        ema_fast = df[source].ewm_mean(span=fast_span, adjust=False)
        ema_slow = df[source].ewm_mean(span=slow_span, adjust=False)
        return ema_fast - ema_slow

    @staticmethod
    def calculate_bollinger(df: pl.DataFrame, length: int = 20, source: str = "close", num_std: float = 2.0) -> tuple[pl.Series, pl.Series]:
        """布林通道上下軌 (BB_UPPER, BB_LOWER)"""
        ma = df[source].rolling_mean(window_size=length)
        std = df[source].rolling_std(window_size=length).fill_null(0.0)
        return ma + num_std * std, ma - num_std * std

    @staticmethod
    def calculate_adx(df: pl.DataFrame, length: int = 14) -> pl.Series:
        """平均趨向指標 (ADX)"""
        high = df["high"]
        low = df["low"]
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = df["close"].shift(1)

        up_move = high - prev_high
        down_move = prev_low - low

        plus_dm = pl.when((up_move > down_move) & (up_move > 0)).then(up_move).otherwise(0.0)
        minus_dm = pl.when((down_move > up_move) & (down_move > 0)).then(down_move).otherwise(0.0)

        tr = pl.max_horizontal([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ])
        tmp = df.with_columns([
            plus_dm.alias("_pdm"),
            minus_dm.alias("_mdm"),
            tr.alias("_tr"),
        ])
        atr = tmp["_tr"].ewm_mean(span=length, adjust=False) + 1e-9
        pdi = 100.0 * tmp["_pdm"].ewm_mean(span=length, adjust=False) / atr
        mdi = 100.0 * tmp["_mdm"].ewm_mean(span=length, adjust=False) / atr
        dx = ((pdi - mdi).abs() / (pdi + mdi + 1e-9)) * 100.0
        return dx.ewm_mean(span=length, adjust=False)

    @staticmethod
    def calculate_highest(df: pl.DataFrame, length: int = 20, source: str = "high") -> pl.Series:
        """過去 N 根最高價通道 (不含當根，供突破策略比較當根收盤與前 N 根高點)"""
        return df[source].shift(1).rolling_max(window_size=length)

    @staticmethod
    def calculate_lowest(df: pl.DataFrame, length: int = 20, source: str = "low") -> pl.Series:
        """過去 N 根最低價通道 (不含當根，供跌破策略比較當根收盤與前 N 根低點)"""
        return df[source].shift(1).rolling_min(window_size=length)

    @staticmethod
    def calculate_bb_width(df: pl.DataFrame, length: int = 20, source: str = "close", num_std: float = 2.0) -> pl.Series:
        """布林通道寬度百分比 (BB_WIDTH = (Upper - Lower) / Middle * 100)"""
        ma = df[source].rolling_mean(window_size=length)
        std = df[source].rolling_std(window_size=length).fill_null(0.0)
        width = (2.0 * num_std * std) / (ma.abs() + 1e-9) * 100.0
        return width

    @staticmethod
    def calculate_roc(df: pl.DataFrame, length: int = 10, source: str = "close") -> pl.Series:
        """價格變化率 / 動能 (ROC % = (Price - Price[t-n]) / Price[t-n] * 100)"""
        prev = df[source].shift(length)
        return ((df[source] - prev) / (prev.abs() + 1e-9)) * 100.0

    @staticmethod
    def calculate_bias(df: pl.DataFrame, length: int = 20, source: str = "close") -> pl.Series:
        """均線乖離率 (BIAS % = (Close - MA) / MA * 100)"""
        ma = df[source].rolling_mean(window_size=length)
        return ((df[source] - ma) / (ma.abs() + 1e-9)) * 100.0

    @staticmethod
    def calculate_ma_slope(df: pl.DataFrame, length: int = 20, source: str = "close", lookback: int = 3) -> pl.Series:
        """均線斜率變化百分比 (MA_SLOPE %)"""
        ma = df[source].rolling_mean(window_size=length)
        prev_ma = ma.shift(max(1, lookback))
        return ((ma - prev_ma) / (prev_ma.abs() + 1e-9)) * 100.0

    @staticmethod
    def calculate_vol_ratio(df: pl.DataFrame, length: int = 20) -> pl.Series:
        """相對成交量倍數 (VOL_RATIO = Volume / MA(Volume))"""
        vol_ma = df["volume"].cast(pl.Float64).rolling_mean(window_size=length)
        return df["volume"].cast(pl.Float64) / (vol_ma + 1e-9)

    @classmethod
    def compute_feature_matrix(cls, df: pl.DataFrame) -> pl.DataFrame:
        """
        AITrade 風格多維度特徵工程矩陣：
        自動從 K 線衍生動能、均線斜率/乖離、波動率、技術指標、量價與盤中時段特徵。
        """
        if df.height == 0:
            return df.clone()

        feat_df = df.clone()
        feat_df = feat_df.with_columns([
            cls.calculate_roc(feat_df, 5, "close").fill_null(0.0).alias("feat_roc_5"),
            cls.calculate_roc(feat_df, 10, "close").fill_null(0.0).alias("feat_roc_10"),
            cls.calculate_roc(feat_df, 20, "close").fill_null(0.0).alias("feat_roc_20"),
            cls.calculate_bias(feat_df, 10, "close").fill_null(0.0).alias("feat_bias_10"),
            cls.calculate_bias(feat_df, 20, "close").fill_null(0.0).alias("feat_bias_20"),
            cls.calculate_ma_slope(feat_df, 10, "close", 3).fill_null(0.0).alias("feat_slope_10"),
            cls.calculate_ma_slope(feat_df, 20, "close", 3).fill_null(0.0).alias("feat_slope_20"),
            cls.calculate_rsi(feat_df, 14, "close").fill_null(50.0).alias("feat_rsi_14"),
            cls.calculate_macd(feat_df, 12, "close").fill_null(0.0).alias("feat_macd"),
            cls.calculate_atr(feat_df, 14).fill_null(0.0).alias("feat_atr_14"),
            cls.calculate_adx(feat_df, 14).fill_null(20.0).alias("feat_adx_14"),
            cls.calculate_bb_width(feat_df, 20, "close").fill_null(1.0).alias("feat_bb_width"),
            cls.calculate_vol_ratio(feat_df, 20).fill_null(1.0).alias("feat_vol_ratio"),
        ])
        k_ser, d_ser = cls.calculate_kd(feat_df, 9)
        feat_df = feat_df.with_columns([
            k_ser.fill_null(50.0).alias("feat_kd_k"),
            d_ser.fill_null(50.0).alias("feat_kd_d"),
        ])
        # 動能加速度 (Acceleration = ROC_5 - prev ROC_5)
        feat_df = feat_df.with_columns(
            (feat_df["feat_roc_5"] - feat_df["feat_roc_5"].shift(1)).fill_null(0.0).alias("feat_accel")
        )
        return feat_df

    @classmethod
    def compute_indicators(cls, df: pl.DataFrame, indicators_spec: Dict[str, Dict[str, Any]]) -> pl.DataFrame:
        """
        根據 Strategy Schema 中的 indicators 定義，批次計算所有需要的指標
        """
        computed_df = df.clone()

        for name, spec in indicators_spec.items():
            ind_type = spec.get("type")
            length = int(spec.get("length", 14))
            source = spec.get("source", "close")

            if ind_type == "SMA":
                computed_df = computed_df.with_columns(
                    cls.calculate_sma(computed_df, length, source).alias(name)
                )
            elif ind_type == "EMA":
                computed_df = computed_df.with_columns(
                    cls.calculate_ema(computed_df, length, source).alias(name)
                )
            elif ind_type == "RSI":
                computed_df = computed_df.with_columns(
                    cls.calculate_rsi(computed_df, length, source).alias(name)
                )
            elif ind_type == "ATR":
                computed_df = computed_df.with_columns(
                    cls.calculate_atr(computed_df, length).alias(name)
                )
            elif ind_type == "KD_K":
                k_ser, _ = cls.calculate_kd(computed_df, length)
                computed_df = computed_df.with_columns(k_ser.alias(name))
            elif ind_type == "KD_D":
                _, d_ser = cls.calculate_kd(computed_df, length)
                computed_df = computed_df.with_columns(d_ser.alias(name))
            elif ind_type == "MACD":
                computed_df = computed_df.with_columns(
                    cls.calculate_macd(computed_df, length, source).alias(name)
                )
            elif ind_type == "BB_UPPER":
                upper, _ = cls.calculate_bollinger(computed_df, length, source)
                computed_df = computed_df.with_columns(upper.alias(name))
            elif ind_type == "BB_LOWER":
                _, lower = cls.calculate_bollinger(computed_df, length, source)
                computed_df = computed_df.with_columns(lower.alias(name))
            elif ind_type == "BB_WIDTH":
                computed_df = computed_df.with_columns(
                    cls.calculate_bb_width(computed_df, length, source).alias(name)
                )
            elif ind_type == "ADX":
                computed_df = computed_df.with_columns(
                    cls.calculate_adx(computed_df, length).alias(name)
                )
            elif ind_type == "HIGHEST":
                computed_df = computed_df.with_columns(
                    cls.calculate_highest(computed_df, length, source).alias(name)
                )
            elif ind_type == "LOWEST":
                computed_df = computed_df.with_columns(
                    cls.calculate_lowest(computed_df, length, source).alias(name)
                )
            elif ind_type == "ROC":
                computed_df = computed_df.with_columns(
                    cls.calculate_roc(computed_df, length, source).alias(name)
                )
            elif ind_type == "BIAS":
                computed_df = computed_df.with_columns(
                    cls.calculate_bias(computed_df, length, source).alias(name)
                )
            elif ind_type == "MA_SLOPE":
                computed_df = computed_df.with_columns(
                    cls.calculate_ma_slope(computed_df, length, source).alias(name)
                )
            elif ind_type == "VOL_RATIO":
                computed_df = computed_df.with_columns(
                    cls.calculate_vol_ratio(computed_df, length).alias(name)
                )

        return computed_df

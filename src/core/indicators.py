# src/core/indicators.py
from __future__ import annotations
import polars as pl
import numpy as np
from typing import Dict, Any, Optional

# 各週期對應的分鐘數（多週期指標重採樣用）
TIMEFRAME_MINUTES: Dict[str, int] = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "60m": 60, "1D": 1440}


class IndicatorEngine:
    """
    技術指標計算與快取引擎
    使用 Polars 進行高效能向量化計算，完整支援 StrategySchema 定義之 12 種技術指標：
    SMA, EMA, RSI, KD_K, KD_D, MACD, ATR, ADX, BB_UPPER, BB_LOWER, HIGHEST, LOWEST
    """

    @staticmethod
    def _mask_warmup(series: pl.Series, n_bars: int) -> pl.Series:
        """將前 n_bars 根（暖機期）輸出改為 null，避免遞迴型指標在資料不足時輸出看似有效的數值。"""
        n = min(max(0, int(n_bars)), series.len())
        if n == 0:
            return series
        keep = pl.Series([False] * n + [True] * (series.len() - n))
        return series.zip_with(keep, pl.Series([None] * series.len(), dtype=series.dtype))

    @staticmethod
    def calculate_sma(df: pl.DataFrame, length: int, source: str = "close") -> pl.Series:
        """簡單移動平均線 (SMA)"""
        return df[source].rolling_mean(window_size=length)

    @staticmethod
    def calculate_ema(df: pl.DataFrame, length: int, source: str = "close") -> pl.Series:
        """指數移動平均線 (EMA)；前 length-1 根為暖機期，輸出 null（與 SMA 一致）"""
        ema = df[source].ewm_mean(span=length, adjust=False)
        return IndicatorEngine._mask_warmup(ema, length - 1)

    @staticmethod
    def calculate_rsi(df: pl.DataFrame, length: int = 14, source: str = "close") -> pl.Series:
        """
        相對強弱指標 (RSI)
        - 前 length 根為暖機期，輸出 null。
        - 漲跌幅皆為 0（價格完全持平）時回傳中性值 50，而非 0。
        """
        _col = df[source]
        delta = _col.diff()
        gain = delta.clip_lower(0) if hasattr(delta, "clip_lower") else delta.clip(lower_bound=0)
        loss = -delta.clip_upper(0) if hasattr(delta, "clip_upper") else -delta.clip(upper_bound=0)

        avg_gain = gain.ewm_mean(com=max(1, length - 1), adjust=False)
        avg_loss = loss.ewm_mean(com=max(1, length - 1), adjust=False)

        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        # 價格持平：平均漲幅與平均跌幅皆為 0 → 中性 50
        not_flat = ((avg_gain + avg_loss) >= 1e-9).fill_null(True)
        rsi = rsi.zip_with(not_flat, pl.Series([50.0] * rsi.len(), dtype=pl.Float64))
        return IndicatorEngine._mask_warmup(rsi, length)

    @staticmethod
    def calculate_atr(df: pl.DataFrame, length: int = 14) -> pl.Series:
        """
        真實波幅 (ATR)
        注意：此為「TR 的簡單移動平均」，並非 Wilder 原版的平滑 (RMA) 算法，
        數值會與一般看盤軟體的 ATR 略有差異；既有停損 / 門檻皆以此版本校準，請勿任意更換算法。
        """
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
        """
        平均趨向指標 (ADX)
        注意：此為非標準算法——平滑採用 span=length 的 EMA (alpha = 2/(length+1))，
        而非 Wilder 原版的 RMA (alpha = 1/length)，數值反應較快、與一般看盤軟體的 ADX 不同；
        既有的 ADX 門檻 (20 / 25) 皆以此版本校準，請勿任意更換算法。
        前 length 根為暖機期，輸出 null。
        """
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
        adx = dx.ewm_mean(span=length, adjust=False)
        return IndicatorEngine._mask_warmup(adx, length)

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
    def _calc_single(cls, df: pl.DataFrame, ind_type: Optional[str], length: int, source: str) -> Optional[pl.Series]:
        """計算單一指標序列；不支援的類型回傳 None。"""
        if ind_type == "SMA":
            return cls.calculate_sma(df, length, source)
        if ind_type == "EMA":
            return cls.calculate_ema(df, length, source)
        if ind_type == "RSI":
            return cls.calculate_rsi(df, length, source)
        if ind_type == "ATR":
            return cls.calculate_atr(df, length)
        if ind_type == "KD_K":
            return cls.calculate_kd(df, length)[0]
        if ind_type == "KD_D":
            return cls.calculate_kd(df, length)[1]
        if ind_type == "MACD":
            return cls.calculate_macd(df, length, source)
        if ind_type == "BB_UPPER":
            return cls.calculate_bollinger(df, length, source)[0]
        if ind_type == "BB_LOWER":
            return cls.calculate_bollinger(df, length, source)[1]
        if ind_type == "BB_WIDTH":
            return cls.calculate_bb_width(df, length, source)
        if ind_type == "ADX":
            return cls.calculate_adx(df, length)
        if ind_type == "HIGHEST":
            return cls.calculate_highest(df, length, source)
        if ind_type == "LOWEST":
            return cls.calculate_lowest(df, length, source)
        if ind_type == "ROC":
            return cls.calculate_roc(df, length, source)
        if ind_type == "BIAS":
            return cls.calculate_bias(df, length, source)
        if ind_type == "MA_SLOPE":
            return cls.calculate_ma_slope(df, length, source)
        if ind_type == "VOL_RATIO":
            return cls.calculate_vol_ratio(df, length)
        return None

    @staticmethod
    def _infer_base_minutes(df: pl.DataFrame) -> Optional[int]:
        """未指定主週期時，由 timestamp 欄位的相鄰間隔中位數推斷（推斷不出來回傳 None）。"""
        if "timestamp" not in df.columns or df.height < 2 or not isinstance(df.schema["timestamp"], pl.Datetime):
            return None
        med = df["timestamp"].diff().dt.total_minutes().median()
        if med is None or med <= 0:
            return None
        return int(med)

    @classmethod
    def _calc_higher_timeframe(
        cls,
        df: pl.DataFrame,
        name: str,
        ind_type: Optional[str],
        length: int,
        source: str,
        target_tf: str,
        base_minutes: int,
    ) -> pl.Series:
        """
        多週期指標：將主週期 K 棒聚合成較大週期後計算指標，再對齊回主週期。
        無前視偏誤：每根主週期 K 棒只能看到「上一根已收盤」高週期 K 棒的指標值
        （例如當日所有 K 棒看到的日線值，都是前一個日曆日收盤後才確定的值）。
        聚合邊界與 SharedBarBuilder 一致（以時鐘時間切齊；1D 以日曆日切分，夜盤跨日的部分歸屬於次一日曆日）。
        """
        target_minutes = TIMEFRAME_MINUTES[target_tf]
        if target_minutes < base_minutes:
            raise ValueError(
                f"指標 '{name}' 宣告的週期 {target_tf} 小於主週期（{base_minutes} 分鐘），無法由主週期資料聚合"
            )
        if target_minutes % base_minutes != 0:
            raise ValueError(
                f"指標 '{name}' 宣告的週期 {target_tf} 不是主週期（{base_minutes} 分鐘）的整數倍，無法聚合"
            )
        if "timestamp" not in df.columns or not isinstance(df.schema["timestamp"], pl.Datetime):
            raise ValueError(
                f"指標 '{name}' 宣告了較大週期 {target_tf}，資料必須有 datetime 型別的 timestamp 欄位才能重採樣"
            )
        if df["timestamp"].null_count() > 0 or not df["timestamp"].is_sorted():
            raise ValueError(f"指標 '{name}' 需要重採樣為 {target_tf}，但 timestamp 欄位含缺值或未依時間排序")

        every = "1d" if target_tf == "1D" else f"{target_minutes}m"
        work = df.with_columns(pl.col("timestamp").dt.truncate(every).alias("__htf_bucket"))
        htf = work.group_by("__htf_bucket", maintain_order=True).agg([
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
        ])
        series = cls._calc_single(htf, ind_type, length, source)
        if series is None:
            raise ValueError(f"指標 '{name}' 的類型 {ind_type} 不支援多週期計算")
        # shift(1)：只使用上一根已收盤高週期 K 棒的值，杜絕前視
        htf = htf.select([pl.col("__htf_bucket"), series.shift(1).alias("__htf_value")])
        aligned = work.select("__htf_bucket").join(htf, on="__htf_bucket", how="left", maintain_order="left")
        return aligned["__htf_value"]

    @classmethod
    def compute_indicators(
        cls,
        df: pl.DataFrame,
        indicators_spec: Dict[str, Dict[str, Any]],
        base_timeframe: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        根據 Strategy Schema 中的 indicators 定義，批次計算所有需要的指標。
        - base_timeframe：df 的主週期（例如策略的 timeframe）。
        - 指標若宣告了與主週期不同的 timeframe，會重採樣到該較大週期計算，且只使用已收盤的上一根高週期 K 棒
          （詳見 _calc_higher_timeframe）；宣告的週期小於主週期或無法聚合時 raise ValueError。
        - 未提供 base_timeframe 時，由 timestamp 間隔推斷主週期；推斷不出來且指標有宣告 timeframe 時 raise ValueError，
          絕不靜默改在主週期計算。
        """
        computed_df = df.clone()
        base_minutes: Optional[int] = None
        if base_timeframe is not None:
            if base_timeframe not in TIMEFRAME_MINUTES:
                raise ValueError(f"不支援的主週期: {base_timeframe}")
            base_minutes = TIMEFRAME_MINUTES[base_timeframe]

        for name, spec in indicators_spec.items():
            ind_type = spec.get("type")
            length = int(spec.get("length", 14))
            source = spec.get("source", "close")
            ind_tf = spec.get("timeframe")

            if ind_tf is not None and df.height > 0:
                if ind_tf not in TIMEFRAME_MINUTES:
                    raise ValueError(f"指標 '{name}' 宣告了不支援的週期: {ind_tf}")
                if base_minutes is None:
                    base_minutes = cls._infer_base_minutes(df)
                    if base_minutes is None:
                        raise ValueError(
                            f"指標 '{name}' 宣告了週期 {ind_tf}，但未提供 base_timeframe 且無法由 timestamp 推斷主週期"
                        )
                if TIMEFRAME_MINUTES[ind_tf] != base_minutes:
                    # 以原始 OHLCV（df）重採樣，不受先前已算出的指標欄位影響
                    series = cls._calc_higher_timeframe(df, name, ind_type, length, source, ind_tf, base_minutes)
                    computed_df = computed_df.with_columns(series.alias(name))
                    continue

            series = cls._calc_single(computed_df, ind_type, length, source)
            if series is not None:
                computed_df = computed_df.with_columns(series.alias(name))

        return computed_df

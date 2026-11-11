# src/ai/confidence_engine.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Any
import numpy as np
import polars as pl

from src.core.indicators import IndicatorEngine

RegimeType = Literal["STRONG_TREND", "VOLATILE_CHOP", "RANGING"]


@dataclass
class RegimeSnapshot:
    regime: RegimeType
    label_zh: str
    adx_val: float
    bb_width_val: float
    threshold_offset: float  # 趨勢盤可適度調降門檻 (-5%)，盤整盤自動提高門檻 (+10%) 防雙巴


@dataclass
class ConfidenceSnapshot:
    bar_idx: int
    long_confidence: float      # 0.0 ~ 100.0 (%)
    short_confidence: float     # 0.0 ~ 100.0 (%)
    hold_preference: float      # 0.0 ~ 100.0 (%)
    effective_threshold: float  # 經市場狀態調節後之有效門檻 (%)
    regime: RegimeSnapshot
    feature_contributions: Dict[str, float] = field(default_factory=dict)


class MarketRegimeDetector:
    """
    市場狀態辨識器 (Regime Detection)
    解決 AITrade 洞察報告點出的關鍵問題：「信心度高 ≠ 賺錢機率高（趨勢盤 vs 震盪盤）」。
    結合 ADX 趨勢強度與布林通道寬度 (BB_WIDTH) 辨識三種市場狀態：
    1. STRONG_TREND (🔥 強趨勢盤)：ADX >= 25 → 門檻調降 -5%，積極參與趨勢。
    2. VOLATILE_CHOP (🌊 高波震盪盤)：ADX < 25 且 BB_WIDTH 偏高 → 門檻提高 +6%，嚴選訊號。
    3. RANGING (💤 狹幅盤整盤)：ADX < 20 且 BB_WIDTH 偏低 → 門檻提高 +10%，防止盤整雙巴。
    """

    @staticmethod
    def detect_regime_at(adx_val: float, bb_width_val: float, median_bb_width: float = 0.8) -> RegimeSnapshot:
        adx = 20.0 if np.isnan(adx_val) else float(adx_val)
        bbw = 0.8 if np.isnan(bb_width_val) else float(bb_width_val)
        med_bbw = max(0.2, float(median_bb_width))

        if adx >= 25.0:
            return RegimeSnapshot(
                regime="STRONG_TREND",
                label_zh="🔥 強趨勢盤 (Strong Trend)",
                adx_val=round(adx, 1),
                bb_width_val=round(bbw, 2),
                threshold_offset=-5.0,
            )
        elif adx < 20.0 and bbw <= med_bbw * 1.1:
            return RegimeSnapshot(
                regime="RANGING",
                label_zh="💤 狹幅盤整盤 (Ranging)",
                adx_val=round(adx, 1),
                bb_width_val=round(bbw, 2),
                threshold_offset=+10.0,
            )
        else:
            return RegimeSnapshot(
                regime="VOLATILE_CHOP",
                label_zh="🌊 震盪洗盤區 (Volatile Chop)",
                adx_val=round(adx, 1),
                bb_width_val=round(bbw, 2),
                threshold_offset=+5.0,
            )


class ConfidenceEngine:
    """
    AI 特徵工程與信心度推論引擎 (Confidence & Threshold Engine)
    對齊 AITrade 5 步驟量化決策流程：
    ① 資料收集 → ② 多維度特徵工程 → ③ 模型推論輸出 0~100% 信心度 → ④ 閾值判斷 + 市場狀態動態調節
    """

    @staticmethod
    def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))

    @staticmethod
    def _rolling_bbw_median(bbw: np.ndarray) -> np.ndarray:
        """布林帶寬的滾動中位數 (回看 30 根 K 棒，只用當根及過去資料)。"""
        return (
            pl.Series(bbw)
            .rolling_median(window_size=30, min_samples=1)
            .fill_null(strategy="backward")
            .fill_null(0.8)
            .to_numpy()
        )

    @classmethod
    def compute_confidence_series(
        cls,
        df: pl.DataFrame,
        base_threshold: float = 60.0,
        regime_adaptive: bool = True,
    ) -> pl.DataFrame:
        """
        對整份 K 線 DataFrame 批次運算多維度特徵向量，並推論每根 K 線的：
        - `long_confidence` (0~100%)
        - `short_confidence` (0~100%)
        - `effective_threshold` (依 Regime 動態調整後之門檻)
        - `regime_label` (市場狀態標籤)
        """
        if df.height == 0:
            return df.clone()

        feat_df = IndicatorEngine.compute_feature_matrix(df)
        n = feat_df.height

        roc5 = feat_df["feat_roc_5"].to_numpy()
        roc10 = feat_df["feat_roc_10"].to_numpy()
        accel = feat_df["feat_accel"].to_numpy()
        slope10 = feat_df["feat_slope_10"].to_numpy()
        slope20 = feat_df["feat_slope_20"].to_numpy()
        bias20 = feat_df["feat_bias_20"].to_numpy()
        rsi14 = feat_df["feat_rsi_14"].to_numpy()
        macd = feat_df["feat_macd"].to_numpy()
        kd_k = feat_df["feat_kd_k"].to_numpy()
        kd_d = feat_df["feat_kd_d"].to_numpy()
        adx14 = feat_df["feat_adx_14"].to_numpy()
        bbw = feat_df["feat_bb_width"].to_numpy()
        vol_ratio = feat_df["feat_vol_ratio"].to_numpy()

        # 正規化各項特徵因子得分 (-1.0 ~ +1.0)
        # 注意：杜絕前視偏差 (Look-ahead bias)，使用滾動式/擴展式 (expanding/rolling) 統計量，僅使用當根及過去 K 棒
        f_momentum = np.tanh(roc5 * 1.8 + roc10 * 1.2 + accel * 1.5)
        f_ma_trend = np.tanh(slope10 * 3.0 + slope20 * 2.5 + bias20 * 0.8)
        f_rsi = np.clip((rsi14 - 50.0) / 25.0, -1.5, 1.5)

        # 計算滾動式 MACD 標準差 (回看 30 根 K 棒，無未來資訊)
        s_macd = pl.Series(macd)
        macd_rolling_std = (
            s_macd.rolling_std(window_size=30, min_samples=1)
            .fill_null(strategy="backward")
            .fill_null(1.0)
            .to_numpy()
        )
        macd_rolling_std = np.where(macd_rolling_std < 1e-4, 1.0, macd_rolling_std)
        f_macd = np.tanh(macd / macd_rolling_std)
        f_kd = np.tanh((kd_k - kd_d) / 12.0)

        # 量價確認放大器 (出量時動能權重放大)
        vol_boost = np.clip(0.85 + 0.25 * np.tanh(vol_ratio - 1.0), 0.7, 1.35)

        # 綜合線性組合並透過溫度校準 Sigmoid 映射至 1% ~ 99% 信心度
        raw_score = (
            0.30 * f_momentum
            + 0.28 * f_ma_trend
            + 0.16 * f_rsi
            + 0.14 * f_macd
            + 0.12 * f_kd
        ) * vol_boost

        # 趨勢強度加乘：當 ADX 越高，方向性信心度越鮮明
        adx_factor = np.clip(0.85 + (adx14 - 20.0) / 50.0, 0.75, 1.35)
        calibrated_logit = raw_score * 2.4 * adx_factor

        long_prob = cls._sigmoid(calibrated_logit) * 100.0
        short_prob = cls._sigmoid(-calibrated_logit) * 100.0

        long_conf = np.clip(np.round(long_prob, 1), 1.0, 99.0)
        short_conf = np.clip(np.round(short_prob, 1), 1.0, 99.0)

        # 布林帶寬中位數亦採用滾動窗口 (回看 30 根 K 棒)，嚴格杜絕全局未來資訊偷看
        bbw_rolling_med = cls._rolling_bbw_median(bbw)

        eff_thresholds: List[float] = []
        regimes: List[str] = []

        for i in range(n):
            cur_med_bbw = float(bbw_rolling_med[i]) if i < len(bbw_rolling_med) else 0.8
            reg = MarketRegimeDetector.detect_regime_at(float(adx14[i]), float(bbw[i]), cur_med_bbw)
            regimes.append(reg.label_zh)
            if regime_adaptive:
                # 夾限範圍必須包含使用者設定的 base：設定 30 或 95 時不可被硬改寫成 50 / 90
                lo = min(50.0, float(base_threshold))
                hi = max(90.0, float(base_threshold))
                eff_th = float(np.clip(base_threshold + reg.threshold_offset, lo, hi))
            else:
                eff_th = float(base_threshold)
            eff_thresholds.append(round(eff_th, 1))

        return feat_df.with_columns([
            pl.Series("long_confidence", long_conf),
            pl.Series("short_confidence", short_conf),
            pl.Series("effective_threshold", eff_thresholds),
            pl.Series("regime_label", regimes),
        ])

    @classmethod
    def get_latest_snapshot(
        cls,
        df: pl.DataFrame,
        base_threshold: float = 60.0,
        regime_adaptive: bool = True,
    ) -> Optional[ConfidenceSnapshot]:
        """取得最新一根 K 棒的 AI 信心度推論與特徵貢獻拆解"""
        if df.height == 0:
            return None

        conf_df = cls.compute_confidence_series(df, base_threshold=base_threshold, regime_adaptive=regime_adaptive)
        idx = conf_df.height - 1

        adx_v = float(conf_df["feat_adx_14"][idx])
        bbw_v = float(conf_df["feat_bb_width"][idx])
        # 與 compute_confidence_series 使用同一個滾動中位數，確保快照的 regime 與序列的
        # regime_label / effective_threshold 完全一致（不可改用全域中位數）
        med_bbw = float(cls._rolling_bbw_median(conf_df["feat_bb_width"].to_numpy())[idx])
        reg = MarketRegimeDetector.detect_regime_at(adx_v, bbw_v, med_bbw)

        long_c = float(conf_df["long_confidence"][idx])
        short_c = float(conf_df["short_confidence"][idx])
        eff_th = float(conf_df["effective_threshold"][idx])
        hold_pref = max(0.0, round(100.0 - max(long_c, short_c), 1))

        contribs = {
            "動能變化率 (ROC 5/10 + 加速度)": round(float(np.tanh(conf_df["feat_roc_5"][idx] * 1.8 + conf_df["feat_accel"][idx])) * 100, 1),
            "均線斜率與乖離 (MA Slope & BIAS)": round(float(np.tanh(conf_df["feat_slope_10"][idx] * 3.0 + conf_df["feat_bias_20"][idx] * 0.8)) * 100, 1),
            "相對強弱 (RSI 14)": round(float(conf_df["feat_rsi_14"][idx] - 50.0) * 2.0, 1),
            "KD / MACD 動能差離": round(float(np.tanh((conf_df["feat_kd_k"][idx] - conf_df["feat_kd_d"][idx]) / 12.0)) * 100, 1),
            "量價爆發倍數 (Volume Ratio)": round(float(conf_df["feat_vol_ratio"][idx]), 2),
        }

        return ConfidenceSnapshot(
            bar_idx=idx,
            long_confidence=long_c,
            short_confidence=short_c,
            hold_preference=hold_pref,
            effective_threshold=eff_th,
            regime=reg,
            feature_contributions=contribs,
        )

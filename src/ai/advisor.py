# src/ai/advisor.py
from __future__ import annotations
import os
import json
import time
import requests
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Literal
import polars as pl
import numpy as np

from src.ai.confidence_engine import MarketRegimeDetector, RegimeSnapshot
from src.ai.cloud_intel import CloudIntelSnapshot, CloudIntelligenceHub
from src.core.indicators import IndicatorEngine


@dataclass
class FuturesAdvice:
    symbol: str
    timeframe: str
    current_price: float
    action: Literal["BUY_LONG", "SELL_SHORT", "WAIT"]  # 做多 / 做空 / 觀望
    action_label: str                                  # 繁體中文標籤
    confidence: float                                  # 信心度 (0 ~ 100%)
    long_probability: float                            # 做多機率 %
    short_probability: float                           # 做空機率 %
    regime_label: str                                  # 市場狀態 (強趨勢/盤整/震盪)
    entry_zone: str                                    # 建議參考進場區間
    stop_loss: str                                     # 防守停損點 (跌破/突破失效)
    target_price: str                                  # 短線目標/壓力/支撐
    rationale: str                                     # AI 白話分析核心理由
    generated_at: str                                  # 產生時間
    source: str                                        # "Gemini AI" 或 "本地量化多空模型"
    # 台新 Nova API (FugleSDK) 量化三要素與統計通道欄位
    indicator_summary: str = ""                        # 【指標】：均線成本、布林上下軌、MACD 柱體、ATR
    signal_summary: str = ""                           # 【訊號】：ORB 區間突破 / 布林順逆勢 / 動能轉折
    method_summary: str = ""                           # 【方法】：進場觸發條件、停損停利紀律
    backtest_summary: str = ""                         # 【近期回測驗證】：勝率、賺賠比、獲利因子 (Profit Factor)
    bb_upper: float = 0.0                              # 布林通道上軌 (壓力)
    bb_middle: float = 0.0                             # 布林通道中軸 (20MA 成本)
    bb_lower: float = 0.0                              # 布林通道下軌 (支撐)
    atr_val: float = 0.0                               # 近 14 根 K 棒平均真實波幅 (ATR)

    def extract_price_levels(self, fallback_price: float = 0.0) -> Dict[str, float]:
        """從文字欄位提取數值型進場價、停損價、目標價，並以 ATR 與布林通道進行合理邊界校準，杜絕誇張衝天炮預估"""
        import re
        base_p = self.current_price if self.current_price > 0 else fallback_price
        if base_p <= 0:
            return {"entry": 0.0, "stop_loss": 0.0, "target": 0.0}

        def _parse_valid_nums(text: str) -> List[float]:
            if not text:
                return []
            raw = re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", str(text))
            nums = []
            for r in raw:
                try:
                    val = float(r.replace(",", ""))
                    # 過濾合理短線價格區間 (基準價的 92% ~ 108%)，避免誤抓週期 (如 20MA 的 20) 或誇張異常值
                    if base_p * 0.92 <= val <= base_p * 1.08:
                        nums.append(val)
                except ValueError:
                    pass
            return nums

        entry_nums = _parse_valid_nums(self.entry_zone)
        sl_nums = _parse_valid_nums(self.stop_loss)
        tp_nums = _parse_valid_nums(self.target_price)

        is_fut = FuturesAIAdvisor.is_futures(self.symbol)
        eff_atr = self.atr_val if self.atr_val > 0 else (max(15.0, base_p * 0.0008) if is_fut else max(0.5, base_p * 0.003))
        default_sl_gap = round(eff_atr * 1.3, 1)
        default_tp_gap = round(eff_atr * 1.8, 1)
        max_tp_gap = max(eff_atr * 2.5, abs(self.bb_upper - self.bb_lower) * 0.85) if self.bb_upper > self.bb_lower else eff_atr * 2.5
        max_sl_gap = eff_atr * 2.0

        entry_val = float(np.mean(entry_nums[:2])) if entry_nums else base_p
        if abs(entry_val - base_p) > eff_atr * 1.2:
            entry_val = base_p

        if self.action == "BUY_LONG":
            sl_val = sl_nums[0] if sl_nums else round(base_p - default_sl_gap, 1)
            tp_val = tp_nums[0] if tp_nums else round(base_p + default_tp_gap, 1)
            # 確保停損在現價下方、目標在現價上方，且不超過短線合理 ATR 範圍
            if sl_val >= base_p or (base_p - sl_val) > max_sl_gap:
                sl_val = round(base_p - min(default_sl_gap, max_sl_gap), 1)
            if tp_val <= base_p or (tp_val - base_p) > max_tp_gap:
                tp_val = round(base_p + min(default_tp_gap, max_tp_gap), 1)
        elif self.action == "SELL_SHORT":
            sl_val = sl_nums[0] if sl_nums else round(base_p + default_sl_gap, 1)
            tp_val = tp_nums[0] if tp_nums else round(base_p - default_tp_gap, 1)
            if sl_val <= base_p or (sl_val - base_p) > max_sl_gap:
                sl_val = round(base_p + min(default_sl_gap, max_sl_gap), 1)
            if tp_val >= base_p or (base_p - tp_val) > max_tp_gap:
                tp_val = round(base_p - min(default_tp_gap, max_tp_gap), 1)
        else:
            sl_val = sl_nums[0] if sl_nums else round(self.bb_lower if self.bb_lower > 0 else base_p - default_sl_gap, 1)
            tp_val = tp_nums[0] if tp_nums else round(self.bb_upper if self.bb_upper > 0 else base_p + default_sl_gap, 1)
            if abs(sl_val - base_p) > max_tp_gap:
                sl_val = round(base_p - default_sl_gap, 1)
            if abs(tp_val - base_p) > max_tp_gap:
                tp_val = round(base_p + default_sl_gap, 1)

        return {
            "entry": round(entry_val, 1),
            "stop_loss": round(sl_val, 1),
            "target": round(tp_val, 1),
        }


class FuturesAIAdvisor:
    """
    期貨 / 個股 AI 多空決策顧問 (Futures & Stock AI Advisor)
    - 自動判別標的為「期貨 (TMF/TXF/MTX)」或「個股 (2330/2317 等)」。
    - 結合台新 API 即時 K 線形態、均線多空排列、ADX/布林市場狀態辨識。
    - 支援 Gemini 大模型深度盤勢解讀與關鍵點位預測。
    - 支援本地量化融合引擎自動回退（即使斷網或無 API Key 依然 100% 正常分析）。
    - 物理阻斷下單，僅提供方向建議與防守點位參考。
    """

    @staticmethod
    def is_futures(symbol: str) -> bool:
        s = (symbol or "").strip().upper()
        return any(k in s for k in ["TMF", "TXF", "MTX", "MXF", "TX", "MT"]) or "R1" in s or "R2" in s

    @classmethod
    def get_market_session_status(cls, symbol: str, now_dt: Optional[Any] = None) -> tuple[bool, str]:
        """
        判斷指定商品目前是否處於開盤交易時段 (台灣時間 UTC+8)：
        - 回傳: (is_open: bool, status_label: str)
        """
        import datetime as _dt
        if now_dt is None:
            now_dt = _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=8))).replace(tzinfo=None)
        wk = now_dt.weekday()
        hm = (now_dt.hour, now_dt.minute)
        is_fut = cls.is_futures(symbol)

        if not is_fut:
            # 台股個股 / ETF：週一至週五 09:00 ~ 13:30
            if wk < 5 and (9, 0) <= hm <= (13, 30):
                return True, "🟢 台股盤中交易中 (09:00~13:30)"
            if wk >= 5:
                return False, "🌙 台股週末休市中（週一 09:00 開盤）"
            if hm > (13, 30):
                return False, "🌙 台股今日已收盤 (13:30 停止交易)"
            return False, "🌙 台股尚未開盤 (09:00 開盤)"
        else:
            # 台指期貨：日盤 08:45 ~ 13:45，夜盤 15:00 ~ 隔日 05:00
            if wk < 5 and (8, 45) <= hm <= (13, 45):
                return True, "🟢 期貨日盤交易中 (08:45~13:45)"
            if wk < 5 and hm >= (15, 0):
                return True, "🟢 期貨夜盤交易中 (15:00~05:00)"
            if 1 <= wk <= 5 and hm <= (5, 0):
                return True, "🟢 期貨夜盤交易中 (15:00~05:00)"
            if wk < 5 and (13, 45) < hm < (15, 0):
                return False, "🌙 期貨日盤已收盤（15:00 夜盤開盤）"
            if (wk == 5 and hm > (5, 0)) or wk == 6 or (wk == 0 and hm < (8, 45)):
                return False, "🌙 期貨週末休市中（週一 08:45 開盤）"
            return False, "🌙 期貨夜盤已收盤（08:45 日盤開盤）"

    @classmethod
    def evaluate_recent_strategy_metrics(cls, df: pl.DataFrame, is_fut: bool = True) -> Dict[str, Any]:
        """
        實作台新證券 Nova API《策略績效度量的方法 (上/下)》四大回測指標驗證：
        對當前商品近期歷史 K 棒即時回測三大經典策略：
          1. 均線成本順勢策略 (5MA / 20MA 交叉)
          2. 布林通道順勢突破策略 (BBand 突破上軌買進 / 跌破中軌出場)
          3. 布林通道區間逆勢策略 (BBand 下軌支撐買進 / 上軌壓力出場)
        計算：勝率 (Win Rate)、賺賠比 (Odds)、獲利因子 (Profit Factor)，挑出最適應當前商品之策略！
        """
        closes = df["close"].to_list()
        ma5_list = df["ma5"].to_list()
        ma20_list = df["ma20"].to_list()
        bb_up_list = df["bb_up"].to_list()
        bb_low_list = df["bb_low"].to_list()
        n = len(closes)
        if n < 22:
            return {
                "best_strategy": "均線與布林複合策略",
                "win_rate": 55.0,
                "odds": 1.35,
                "profit_factor": 1.65,
                "trade_count": 0,
                "summary": "均線與布林複合模型（樣本累積中）",
            }

        # 手續費與滑價微幅折減（確保回測真實性）
        cost_per_rt = max(1.0, closes[-1] * 0.00015) if is_fut else max(0.05, closes[-1] * 0.0018)

        def _score_pnls(pnls: List[float], name: str) -> Dict[str, Any]:
            if not pnls:
                return {"name": name, "win_rate": 50.0, "odds": 1.0, "pf": 1.0, "count": 0}
            wins = [p for p in pnls if p > 0]
            losses = [abs(p) for p in pnls if p <= 0]
            win_rate = round((len(wins) / len(pnls)) * 100.0, 1)
            avg_win = (sum(wins) / len(wins)) if wins else 0.0
            avg_loss = (sum(losses) / len(losses)) if losses else 0.0
            odds = round(avg_win / max(1e-6, avg_loss), 2) if avg_loss > 0 else (2.5 if avg_win > 0 else 1.0)
            pf = round(sum(wins) / max(1e-6, sum(losses)), 2) if sum(losses) > 0 else (3.0 if sum(wins) > 0 else 1.0)
            return {"name": name, "win_rate": win_rate, "odds": odds, "pf": pf, "count": len(pnls)}

        def _is_bad(v: Any) -> bool:
            if v is None:
                return True
            try:
                return bool(np.isnan(float(v)))
            except Exception:
                return True

        # 1. 均線成本順勢 (5MA > 20MA 做多, < 20MA 做空，每 3 根結算一次區段損益以貼近當沖節奏)
        ma_pnls: List[float] = []
        step = 3
        start_idx = max(5, min(20, n - 10), n - 90)
        for i in range(start_idx, n - step, step):
            if _is_bad(ma5_list[i]) or _is_bad(ma20_list[i]):
                continue
            direction = 1.0 if float(ma5_list[i]) >= float(ma20_list[i]) else -1.0
            raw_pnl = (closes[i + step] - closes[i]) * direction - cost_per_rt
            ma_pnls.append(raw_pnl)

        # 2. 布林順勢突破策略 (接近或突破上軌/中軌順勢動能)
        bb_trend_pnls: List[float] = []
        for i in range(start_idx, n - step, step):
            if _is_bad(bb_up_list[i]) or _is_bad(bb_low_list[i]) or _is_bad(ma20_list[i]):
                continue
            mid = float(ma20_list[i])
            if closes[i] > mid and closes[i] > closes[i - 1]:
                bb_trend_pnls.append((closes[i + step] - closes[i]) - cost_per_rt)
            elif closes[i] < mid and closes[i] < closes[i - 1]:
                bb_trend_pnls.append((closes[i] - closes[i + step]) - cost_per_rt)

        # 3. 布林區間逆勢震盪策略 (觸及上軌附近回檔做空、觸及下軌附近反彈做多)
        bb_rev_pnls: List[float] = []
        for i in range(start_idx, n - step, step):
            if _is_bad(bb_up_list[i]) or _is_bad(bb_low_list[i]):
                continue
            bw = float(bb_up_list[i]) - float(bb_low_list[i])
            if bw <= 0:
                continue
            pos = (closes[i] - float(bb_low_list[i])) / bw
            if pos >= 0.75:
                bb_rev_pnls.append((closes[i] - closes[i + step]) - cost_per_rt)
            elif pos <= 0.25:
                bb_rev_pnls.append((closes[i + step] - closes[i]) - cost_per_rt)

        candidates = [
            _score_pnls(ma_pnls, "均線成本順勢策略"),
            _score_pnls(bb_trend_pnls, "布林通道順勢突破"),
            _score_pnls(bb_rev_pnls, "布林通道區間逆勢"),
        ]
        valid_cands = [c for c in candidates if c["count"] >= 2] or candidates
        best = max(valid_cands, key=lambda c: (c["pf"], c["win_rate"]))
        summary_str = (
            f"近期最適策略：【{best['name']}】｜"
            f"回測勝率 {best['win_rate']:.1f}% ｜ 賺賠比 {best['odds']:.2f} ｜ 獲利因子(PF) {best['pf']:.2f}"
        )
        return {
            "best_strategy": best["name"],
            "win_rate": best["win_rate"],
            "odds": best["odds"],
            "profit_factor": best["pf"],
            "trade_count": best["count"],
            "summary": summary_str,
        }

    @classmethod
    def analyze_market(
        cls,
        df_bars: pl.DataFrame,
        symbol: str = "TMF",
        timeframe: str = "5m",
        gemini_api_key: str = "",
        gemini_model: str = "gemini-2.5-flash",
        cloud_intel: Optional[CloudIntelSnapshot] = None,
        fast_local_only: bool = False,
    ) -> FuturesAdvice:
        import datetime as _dt
        now_str = _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
        is_fut = cls.is_futures(symbol)
        asset_label = "台指期貨" if is_fut else "個股股票"
        unit = "點" if is_fut else "元"

        if df_bars is None or df_bars.height < 10:
            return FuturesAdvice(
                symbol=symbol,
                timeframe=timeframe,
                current_price=0.0,
                action="WAIT",
                action_label="⚪ 數據回補中，建議觀望",
                confidence=50.0,
                long_probability=50.0,
                short_probability=50.0,
                regime_label="⏳ 數據載入中",
                entry_zone="--",
                stop_loss="--",
                target_price="--",
                rationale=f"目前 {symbol} K 線數量不足 (少於 10 根)，請點擊「🚀 連線行情伺服器」或等待台新 API 回補完整 K 線後再行分析。",
                generated_at=now_str,
                source="系統提示",
            )

        # 1. 提取台新 Nova API 教學之三大技術指標積木：
        #    (A) 均線成本排列 (5MA / 10MA / 20MA / 60MA)
        #    (B) 布林統計通道 (20MA ± 2倍標準差：上軌壓力、中軸成本、下軌支撐)
        #    (C) MACD 快慢線與柱狀體動能 (DIF / Signal / Hist) + ATR(14) 真實波幅
        w20 = min(20, max(5, df_bars.height - 1))
        w14 = min(14, max(5, df_bars.height - 1))
        ma5_s = IndicatorEngine.calculate_sma(df_bars, 5, "close")
        ma10_s = IndicatorEngine.calculate_sma(df_bars, min(10, w20), "close")
        ma20_s = IndicatorEngine.calculate_sma(df_bars, w20, "close")
        ma60_s = IndicatorEngine.calculate_sma(df_bars, min(60, df_bars.height - 1), "close")
        rsi_s = IndicatorEngine.calculate_rsi(df_bars, w14, "close")
        bb_up_s, bb_low_s = IndicatorEngine.calculate_bollinger(df_bars, w20, "close", 2.0)
        adx_s = IndicatorEngine.calculate_adx(df_bars, w14)
        atr_s = IndicatorEngine.calculate_atr(df_bars, w14)
        macd_dif_s = IndicatorEngine.calculate_macd(df_bars, 12, "close")
        macd_sig_s = macd_dif_s.ewm_mean(span=9, adjust=False)
        macd_hist_s = (macd_dif_s - macd_sig_s) * 2.0

        df = df_bars.with_columns([
            ma5_s.alias("ma5"),
            ma10_s.alias("ma10"),
            ma20_s.alias("ma20"),
            ma60_s.alias("ma60"),
            rsi_s.alias("rsi"),
            bb_up_s.alias("bb_up"),
            bb_low_s.alias("bb_low"),
            adx_s.alias("adx"),
            atr_s.alias("atr"),
            macd_dif_s.alias("macd_dif"),
            macd_sig_s.alias("macd_sig"),
            macd_hist_s.alias("macd_hist"),
        ])

        closes = df["close"].to_list()
        highs = df["high"].to_list()
        lows = df["low"].to_list()

        def _sf(val: Any, default: float) -> float:
            if val is None:
                return float(default)
            try:
                f = float(val)
                return float(default) if np.isnan(f) else f
            except Exception:
                return float(default)

        curr_p = float(closes[-1])
        ma5 = _sf(df["ma5"][-1], curr_p)
        ma10 = _sf(df["ma10"][-1], curr_p)
        ma20 = _sf(df["ma20"][-1], curr_p)
        ma60 = _sf(df["ma60"][-1], curr_p)
        rsi_val = _sf(df["rsi"][-1], 50.0)
        adx_val = _sf(df["adx"][-1], 20.0)
        fallback_atr = float(np.mean([h - l for h, l in zip(highs[-10:], lows[-10:])]))
        raw_atr = _sf(df["atr"][-1], fallback_atr)
        if raw_atr <= 0:
            raw_atr = fallback_atr
        atr_val = max(12.0, raw_atr) if is_fut else max(0.5, raw_atr)

        bb_up = _sf(df["bb_up"][-1], curr_p + atr_val * 2.0)
        bb_low = _sf(df["bb_low"][-1], curr_p - atr_val * 2.0)
        if bb_up <= bb_low:
            bb_up = curr_p + atr_val * 2.0
            bb_low = curr_p - atr_val * 2.0

        macd_dif = _sf(df["macd_dif"][-1], 0.0)
        macd_sig = _sf(df["macd_sig"][-1], 0.0)
        macd_hist = _sf(df["macd_hist"][-1], 0.0)
        prev_macd_hist = _sf(df["macd_hist"][-2], macd_hist) if len(closes) >= 2 else macd_hist

        # 計算布林位置 (0% = 下軌, 50% = 20MA 中軸, 100% = 上軌) 與布林帶寬
        bb_span = max(1e-6, bb_up - bb_low)
        bb_pos = (curr_p - bb_low) / bb_span
        bb_width = (bb_span / max(1.0, ma20)) * 100.0 if ma20 > 0 else 1.0
        regime = MarketRegimeDetector.detect_regime_at(adx_val, bb_width)

        # ORB (Opening Range / 近 15 根箱型高低區間) 與近 5 根短線高低點
        recent_n = min(30, len(closes))
        recent_high = float(max(highs[-recent_n:]))
        recent_low = float(min(lows[-recent_n:]))
        orb_n = min(15, len(closes))
        orb_high = float(max(highs[-orb_n:-1])) if len(highs) > 2 else recent_high
        orb_low = float(min(lows[-orb_n:-1])) if len(lows) > 2 else recent_low
        last5_high = float(max(highs[-min(5, len(highs)):]))
        last5_low = float(min(lows[-min(5, len(lows)):]))

        # 偵測「漲多/跌深後的近 5 根 K 棒橫盤鈍化 (Stalling / Consolidation)」
        # 條件：已進入布林上軌區 (bb_pos >= 0.70)，且近 5 根淨位移極小 (<= 0.40 倍 ATR) 或高點不再創高且區間收斂
        last5_net_move = abs(closes[-1] - closes[-min(5, len(closes))])
        last5_box_range = last5_high - last5_low
        making_higher_bars = closes[-1] > closes[-3] and highs[-1] >= max(highs[-5:-1])
        making_lower_bars = closes[-1] < closes[-3] and lows[-1] <= min(lows[-5:-1])

        is_high_stalling = (
            not making_higher_bars
            and bb_pos >= 0.70
            and (last5_net_move <= atr_val * 0.45 or last5_box_range <= atr_val * 1.35)
            and curr_p <= bb_up
        )
        is_low_stalling = (
            not making_lower_bars
            and bb_pos <= 0.30
            and (last5_net_move <= atr_val * 0.45 or last5_box_range <= atr_val * 1.35)
            and curr_p >= bb_low
        )

        # 即時計算台新《策略績效度量》四大回測指標 (勝率、賺賠比、獲利因子 PF)
        bt_metrics = cls.evaluate_recent_strategy_metrics(df, is_fut=is_fut)

        # 2. 台新三積木融合打分邏輯 (均線成本 + 布林通道順逆勢 + MACD 動能柱體)
        long_score = 0
        short_score = 0

        # (A) 均線平均成本判斷
        if ma5 > ma20:
            long_score += 22
            if curr_p > ma5:
                long_score += 10
        else:
            short_score += 22
            if curr_p < ma5:
                short_score += 10

        if ma20 > ma60:
            long_score += 12
        else:
            short_score += 12

        # (B) MACD 柱狀體與黃金/死亡交叉動能確認
        if macd_dif > macd_sig and macd_hist > 0:
            long_score += 18 if macd_hist >= prev_macd_hist else 8
        elif macd_dif < macd_sig and macd_hist < 0:
            short_score += 18 if macd_hist <= prev_macd_hist else 8

        # (C) 布林通道與 ORB 突破 / 高低檔鈍化修正
        if curr_p > orb_high and macd_hist >= prev_macd_hist:
            # 帶動能突破 ORB 箱型高點 -> 順勢做多加分
            long_score += 20
        elif curr_p < orb_low and macd_hist <= prev_macd_hist:
            # 帶動能跌破 ORB 箱型低點 -> 順勢做空加分
            short_score += 20
        elif is_high_stalling:
            # 已逼近布林上軌壓力且近 5 根橫盤 / MACD 縮腳 -> 抑制盲目追高！
            short_score += 18
        elif is_low_stalling:
            # 已逼近布林下軌支撐且近 5 根止跌 / MACD 收斂 -> 抑制盲目追空！
            long_score += 18

        # RSI 適度動能 (若 RSI > 75 超買且未突破上軌則扣減追多分數)
        if 53 <= rsi_val <= 72:
            long_score += 10
        elif rsi_val > 72 and curr_p <= bb_up:
            short_score += 10
        elif 28 <= rsi_val <= 47:
            short_score += 10
        elif rsi_val < 28 and curr_p >= bb_low:
            long_score += 10

        # 融合 Google 雲端四維度情報加權
        if cloud_intel is not None:
            if "偏多" in cloud_intel.global_bias_label:
                long_score += 8
            elif "偏空" in cloud_intel.global_bias_label:
                short_score += 8
            if "偏多" in cloud_intel.news_sentiment_label:
                long_score += 6
            elif "偏空" in cloud_intel.news_sentiment_label:
                short_score += 6
            if cloud_intel.is_in_morning_picks:
                long_score += 10
            if cloud_intel.is_in_cooling_off:
                short_score += 12

        total_s = max(1, long_score + short_score)
        base_long_prob = round((long_score / total_s) * 100.0, 1)
        base_short_prob = round(100.0 - base_long_prob, 1)

        # 组装台新 Nova API 標準【指標 / 訊號 / 方法】結構說明
        macd_state = (
            "紅柱放大(動能轉強)" if (macd_hist > 0 and macd_hist >= prev_macd_hist)
            else "紅柱縮腳(高檔動能減弱)" if (macd_hist > 0 and macd_hist < prev_macd_hist)
            else "綠柱放大(空方動能轉強)" if (macd_hist < 0 and macd_hist <= prev_macd_hist)
            else "綠柱收斂(跌勢趨緩)"
        )
        indicator_summary = (
            f"均線成本 5MA({ma5:,.1f}) / 20MA({ma20:,.1f}) ｜ "
            f"布林上軌({bb_up:,.1f}) / 下軌({bb_low:,.1f}) ｜ "
            f"MACD {macd_state} ｜ 單根波幅 ATR({atr_val:,.1f}{unit})"
        )

        # 依據【布林通道上下軌 + 20MA 成本支撐/解套壓力 + 真實 ATR】計算合理短線目標與停損（絕不超過 1.2~2.0 倍 ATR）
        intel_tail = ""
        if cloud_intel is not None:
            intel_tail = f" 結合雲端情報：{cloud_intel.global_bias_label}、{cloud_intel.news_sentiment_label}。"

        if base_long_prob >= 60.0 and not is_high_stalling:
            action = "BUY_LONG"
            action_label = "🟢 建議偏多買進 (BUY LONG)"
            conf = base_long_prob
            # 進場區間：回測 5MA 支撐或現價微幅區間
            ent_low = round(max(curr_p - atr_val * 0.45, min(curr_p, ma5)), 1)
            ent_high = round(curr_p + atr_val * 0.15, 1)
            entry = f"{ent_low:,.1f} ~ {ent_high:,.1f} {unit}"
            # 停損：守近 5 根低點或 20MA 成本支撐，控制在 1.0 ~ 1.5 倍 ATR
            raw_sl = min(last5_low, ma20)
            sl_gap = float(np.clip(curr_p - raw_sl, atr_val * 1.0, atr_val * 1.5))
            sl_pt = round(curr_p - sl_gap, 1)
            # 目標：優先對齊「布林通道上軌壓力 (bb_up)」或「ORB 上緣」，且控制在 1.2 ~ 2.0 倍 ATR 內！
            raw_tp = bb_up if bb_up > curr_p + atr_val * 0.6 else curr_p + atr_val * 1.5
            tp_gap = float(np.clip(raw_tp - curr_p, atr_val * 1.1, atr_val * 2.0))
            tp_pt = round(curr_p + tp_gap, 1)

            signal_summary = (
                f"股價站上 20MA({ma20:,.1f}) 平均成本支撐，MACD {macd_state}，"
                f"上方第一道統計壓力位於布林上軌 {bb_up:,.1f} {unit}"
            )
            method_summary = (
                f"逢回 {entry} 偏多佈局，短線目標看布林上軌區 {tp_pt:,.1f} {unit} "
                f"(+{tp_pt - curr_p:,.1f}{unit})，跌破 {sl_pt:,.1f} {unit} 嚴格停損"
            )
            rationale = (
                f"【指標】{symbol} 位於 20MA({ma20:,.1f}) 平均持倉成本之上，多數持倉者處於獲利支撐狀態，MACD 呈現{macd_state}。"
                f"【訊號】目前布林軌道位置在 {bb_pos*100:.0f}%，{bt_metrics['summary']}。{intel_tail}"
                f"【方法】短線第一目標對齊布林上軌壓力 {tp_pt:,.1f} {unit}（約 {tp_gap/atr_val:.1f} 倍 ATR 合理波幅），防守線設於 {sl_pt:,.1f} {unit}。"
            )
        elif base_short_prob >= 60.0 and not is_low_stalling:
            action = "SELL_SHORT"
            action_label = "🔴 建議偏空賣出 (SELL SHORT)"
            conf = base_short_prob
            ent_low = round(curr_p - atr_val * 0.15, 1)
            ent_high = round(min(curr_p + atr_val * 0.45, max(curr_p, ma5)), 1)
            entry = f"{ent_low:,.1f} ~ {ent_high:,.1f} {unit}"
            raw_sl = max(last5_high, ma20)
            sl_gap = float(np.clip(raw_sl - curr_p, atr_val * 1.0, atr_val * 1.5))
            sl_pt = round(curr_p + sl_gap, 1)
            raw_tp = bb_low if bb_low < curr_p - atr_val * 0.6 else curr_p - atr_val * 1.5
            tp_gap = float(np.clip(curr_p - raw_tp, atr_val * 1.1, atr_val * 2.0))
            tp_pt = round(curr_p - tp_gap, 1)

            signal_summary = (
                f"股價跌破 20MA({ma20:,.1f}) 形成上檔解套賣壓，MACD {macd_state}，"
                f"下方第一道統計支撐位於布林下軌 {bb_low:,.1f} {unit}"
            )
            method_summary = (
                f"反彈 {entry} 偏空操作，短線目標看布林下軌區 {tp_pt:,.1f} {unit} "
                f"(-{curr_p - tp_pt:,.1f}{unit})，突破 {sl_pt:,.1f} {unit} 嚴格停損"
            )
            rationale = (
                f"【指標】{symbol} 位於 20MA({ma20:,.1f}) 平均成本之下，上檔具備解套賣壓，MACD 呈現{macd_state}。"
                f"【訊號】目前布林軌道位置在 {bb_pos*100:.0f}%，{bt_metrics['summary']}。{intel_tail}"
                f"【方法】短線第一目標對齊布林下軌支撐 {tp_pt:,.1f} {unit}（約 {tp_gap/atr_val:.1f} 倍 ATR 合理波幅），防守線設於 {sl_pt:,.1f} {unit}。"
            )
        else:
            action = "WAIT"
            action_label = "⚪ 建議區間觀望 / 等待突破 (WAIT)"
            conf = max(base_long_prob, base_short_prob)
            # 標示清楚 ORB / 布林突破觸發邊界
            up_trigger = round(min(bb_up, max(last5_high, curr_p + atr_val * 0.8)), 1)
            dn_trigger = round(max(bb_low, min(last5_low, curr_p - atr_val * 0.8)), 1)
            entry = f"突破 {up_trigger:,.1f} {unit} / 跌破 {dn_trigger:,.1f} {unit}"
            sl_pt = dn_trigger
            tp_pt = up_trigger
            stall_note = (
                f"股價已逼近布林上軌壓力({bb_up:,.1f})且近 5 根 K 棒橫盤量縮、MACD 動能減弱"
                if is_high_stalling
                else f"股價已逼近布林下軌支撐({bb_low:,.1f})且跌勢收斂"
                if is_low_stalling
                else f"股價於 5MA({ma5:,.1f}) 與 20MA({ma20:,.1f}) 區間內狹幅整理"
            )
            signal_summary = f"{stall_note}，尚未出現有效 ORB 區間突破訊號"
            method_summary = (
                f"目前不追價；待帶量站上 {up_trigger:,.1f} {unit} 才順勢做多，"
                f"或回測 20MA({ma20:,.1f}) / 跌破 {dn_trigger:,.1f} {unit} 再行動作"
            )
            rationale = (
                f"【指標】{stall_note}（單根 ATR 波幅僅 {atr_val:,.1f} {unit}）。"
                f"【訊號】依台新 ORB 與布林通道法則，高檔橫盤或區間整理時追價極易遭雙巴，{bt_metrics['summary']}。{intel_tail}"
                f"【方法】建議觀望，以上檔壓力 {up_trigger:,.1f} {unit} 與下檔支撐 {dn_trigger:,.1f} {unit} 為突破觸發分水嶺。"
            )

        # 3. 嘗試呼叫 Gemini 進行專業分析 (若有 API Key 且非極速本地預覽模式)
        api_key = (gemini_api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        if api_key and not fast_local_only:
            advice = cls._call_gemini_advisor(
                api_key=api_key,
                model_name=gemini_model,
                symbol=symbol,
                timeframe=timeframe,
                curr_p=curr_p,
                ma5=ma5,
                ma20=ma20,
                ma60=ma60,
                bb_up=bb_up,
                bb_low=bb_low,
                atr_val=atr_val,
                macd_state=macd_state,
                rsi=rsi_val,
                adx=adx_val,
                regime=regime,
                recent_high=recent_high,
                recent_low=recent_low,
                bt_summary=bt_metrics["summary"],
                indicator_summary=indicator_summary,
                default_signal_summary=signal_summary,
                default_method_summary=method_summary,
                df_slice=df.tail(15),
                now_str=now_str,
                cloud_intel=cloud_intel,
            )
            if advice:
                is_open, _ = cls.get_market_session_status(symbol)
                if is_open:
                    levels = advice.extract_price_levels(fallback_price=curr_p)
                    CloudIntelligenceHub.record_prediction(
                        symbol=symbol,
                        timeframe=timeframe,
                        action=advice.action,
                        current_price=curr_p,
                        target_val=levels["target"],
                        stop_val=levels["stop_loss"],
                    )
                return advice

        fallback_advice = FuturesAdvice(
            symbol=symbol,
            timeframe=timeframe,
            current_price=curr_p,
            action=action,
            action_label=action_label,
            confidence=conf,
            long_probability=base_long_prob,
            short_probability=base_short_prob,
            regime_label=regime.label_zh,
            entry_zone=entry,
            stop_loss=f"{sl_pt:,.1f} {unit}",
            target_price=f"{tp_pt:,.1f} {unit}",
            rationale=rationale,
            generated_at=now_str,
            source="本地量化多空模型",
            indicator_summary=indicator_summary,
            signal_summary=signal_summary,
            method_summary=method_summary,
            backtest_summary=bt_metrics["summary"],
            bb_upper=round(bb_up, 1),
            bb_middle=round(ma20, 1),
            bb_lower=round(bb_low, 1),
            atr_val=round(atr_val, 1),
        )
        is_open, _ = cls.get_market_session_status(symbol)
        if is_open:
            CloudIntelligenceHub.record_prediction(
                symbol=symbol,
                timeframe=timeframe,
                action=action,
                current_price=curr_p,
                target_val=tp_pt,
                stop_val=sl_pt,
            )
        return fallback_advice

    @classmethod
    def _call_gemini_advisor(
        cls,
        api_key: str,
        model_name: str,
        symbol: str,
        timeframe: str,
        curr_p: float,
        ma5: float,
        ma20: float,
        ma60: float,
        bb_up: float,
        bb_low: float,
        atr_val: float,
        macd_state: str,
        rsi: float,
        adx: float,
        regime: RegimeSnapshot,
        recent_high: float,
        recent_low: float,
        bt_summary: str,
        indicator_summary: str,
        default_signal_summary: str,
        default_method_summary: str,
        df_slice: pl.DataFrame,
        now_str: str,
        cloud_intel: Optional[CloudIntelSnapshot] = None,
    ) -> Optional[FuturesAdvice]:
        """呼叫 Gemini 進行專業操盤顧問分析（採用台新 Nova API 指標/訊號/方法架構與布林/ATR 邊界約束）"""
        try:
            is_fut = cls.is_futures(symbol)
            asset_label = "台指期貨" if is_fut else "個股股票"
            unit = "點" if is_fut else "元"
            role_title = "資深的台指期貨量化操盤手" if is_fut else "資深的台股量化技術分析師"

            k_lines = []
            for row in df_slice.iter_rows(named=True):
                t_str = row["timestamp"].strftime("%H:%M") if hasattr(row["timestamp"], "strftime") else str(row["timestamp"])
                k_lines.append(f"{t_str} | O:{row['open']:.1f} H:{row['high']:.1f} L:{row['low']:.1f} C:{row['close']:.1f} V:{row['volume']}")
            k_summary = "\n".join(k_lines[-10:])

            intel_block = cloud_intel.to_prompt_context() if cloud_intel else "【🌐 Google 雲端情報庫】：目前僅依據技術指標分析"
            max_tp_move = round(atr_val * 2.0, 1)
            max_sl_move = round(atr_val * 1.5, 1)

            prompt = f"""你是一位{role_title}與首席量化策略顧問。
請根據台新證券 Nova API 量化交易實戰架構（「指標 ➔ 訊號 ➔ 方法」、均線成本、布林通道統計邊界、MACD 動能與 ORB 開盤區間突破），分析【{asset_label} - {symbol}】({timeframe} K線) 的下一階段趨勢。

【當前即時盤勢與統計指標】
- 分析標的: {symbol} ({asset_label}) | 最新成交價: {curr_p} {unit} | 週期: {timeframe}
- 均線平均成本: 5MA={ma5:.1f}, 20MA(中軸成本)={ma20:.1f}, 60MA={ma60:.1f}
- 布林通道 (20MA ± 2σ): 上軌壓力={bb_up:.1f} {unit} / 下軌支撐={bb_low:.1f} {unit}
- 單根 K 棒平均真實波幅 ATR(14): {atr_val:.1f} {unit}
- MACD 動能狀態: {macd_state} | RSI(14): {rsi:.1f} | ADX: {adx:.1f} ({regime.label_zh})
- 近期高低區間: 高點 {recent_high:.1f} / 低點 {recent_low:.1f}
- 近期歷史 K 棒回測績效: {bt_summary}

{intel_block}

【最近 10 根 K 棒走勢】
{k_summary}

【嚴格量化規範（極重要，禁止誇大預估）】
1. 這是 {timeframe} 短線 K 線，單根平均波幅 (ATR) 僅 {atr_val:.1f} {unit}！
2. 目標價 (target_price) 必須對齊「布林通道上軌({bb_up:.1f}) / 下軌({bb_low:.1f})」或近期實際支撐壓力，與現價({curr_p:.1f})的距離絕對不可超過 ±{max_tp_move} {unit} (約 2 倍 ATR)！
3. 停損價 (stop_loss) 必須對齊「20MA({ma20:.1f})」或近 5 根高低點，與現價距離不可超過 ±{max_sl_move} {unit}！
4. 若最後幾根 K 棒已在布林上軌附近橫盤縮小量或 MACD 動能減弱，應建議 "WAIT" (高檔區間震盪/等待突破)，切勿盲目追高。

請以嚴格合法的 JSON 格式輸出：
{{
  "action": "BUY_LONG" 或 "SELL_SHORT" 或 "WAIT",
  "confidence": 70,
  "long_probability": 70,
  "short_probability": 30,
  "entry_zone": "進場參考價位區間 (附單位 {unit})",
  "stop_loss": "防守停損價位 (附單位 {unit})",
  "target_price": "短線合理目標價位 (附單位 {unit}，須在布林通道與 2 倍 ATR 內)",
  "signal_summary": "一句話說明當前【訊號】狀態 (如布林位置、MACD動能與ORB突破與否)",
  "method_summary": "一句話說明當前【方法】操作紀律 (何處進場、何處停利停損)",
  "rationale": "依【指標】【訊號】【方法】三要素撰寫 110~150 字繁體中文專業分析"
}}
"""
            model_target = model_name or "gemini-2.5-flash"
            if "gemini-3.5" in model_target:
                model_target = "gemini-2.5-flash"

            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_target}:generateContent?key={api_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "temperature": 0.15
                }
            }
            res = requests.post(url, headers=headers, json=payload, timeout=15)
            if res.status_code == 200:
                data = res.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                parsed = json.loads(text)

                act = parsed.get("action", "WAIT")
                if act == "BUY_LONG":
                    act_label = "🟢 建議偏多買進 (BUY LONG)"
                elif act == "SELL_SHORT":
                    act_label = "🔴 建議偏空賣出 (SELL SHORT)"
                else:
                    act = "WAIT"
                    act_label = "⚪ 建議區間觀望 / 等待突破 (WAIT)"

                adv = FuturesAdvice(
                    symbol=symbol,
                    timeframe=timeframe,
                    current_price=curr_p,
                    action=act,
                    action_label=act_label,
                    confidence=float(parsed.get("confidence", 60)),
                    long_probability=float(parsed.get("long_probability", 50)),
                    short_probability=float(parsed.get("short_probability", 50)),
                    regime_label=regime.label_zh,
                    entry_zone=str(parsed.get("entry_zone", "--")),
                    stop_loss=str(parsed.get("stop_loss", "--")),
                    target_price=str(parsed.get("target_price", "--")),
                    rationale=str(parsed.get("rationale", "")),
                    generated_at=now_str,
                    source=f"Gemini AI ({model_target})",
                    indicator_summary=indicator_summary,
                    signal_summary=str(parsed.get("signal_summary", default_signal_summary)),
                    method_summary=str(parsed.get("method_summary", default_method_summary)),
                    backtest_summary=bt_summary,
                    bb_upper=round(bb_up, 1),
                    bb_middle=round(ma20, 1),
                    bb_lower=round(bb_low, 1),
                    atr_val=round(atr_val, 1),
                )
                # 強制透過 extract_price_levels 進行 ATR 與布林邊界防呆校準，確保顯示數值與圖表完全一致且合理
                checked = adv.extract_price_levels(fallback_price=curr_p)
                adv.stop_loss = f"{checked['stop_loss']:,.1f} {unit}"
                adv.target_price = f"{checked['target']:,.1f} {unit}"
                return adv
        except Exception as e:
            print(f"FuturesAIAdvisor: Gemini 呼叫例外，自動切換至本地量化回退: {e}")

        return None

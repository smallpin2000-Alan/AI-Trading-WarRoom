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
    source: str                                        # "Gemini AI" 或 "本地量化模型"

    def extract_price_levels(self, fallback_price: float = 0.0) -> Dict[str, float]:
        """從文字欄位提取數值型進場價、停損價、目標價，供 K 線圖視覺化繪製 AI 預估軌跡"""
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
                    # 過濾合理價格區間 (基準價的 70% ~ 130%)，避免誤抓週期或百分比數字
                    if base_p * 0.70 <= val <= base_p * 1.30:
                        nums.append(val)
                except ValueError:
                    pass
            return nums

        entry_nums = _parse_valid_nums(self.entry_zone)
        sl_nums = _parse_valid_nums(self.stop_loss)
        tp_nums = _parse_valid_nums(self.target_price)

        is_fut = FuturesAIAdvisor.is_futures(self.symbol)
        default_sl_gap = max(30.0, base_p * 0.003) if is_fut else max(1.0, base_p * 0.02)
        default_tp_gap = max(60.0, base_p * 0.006) if is_fut else max(2.0, base_p * 0.04)

        entry_val = float(np.mean(entry_nums[:2])) if entry_nums else base_p
        if self.action == "BUY_LONG":
            sl_val = sl_nums[0] if sl_nums else round(base_p - default_sl_gap, 1)
            tp_val = tp_nums[0] if tp_nums else round(base_p + default_tp_gap, 1)
        elif self.action == "SELL_SHORT":
            sl_val = sl_nums[0] if sl_nums else round(base_p + default_sl_gap, 1)
            tp_val = tp_nums[0] if tp_nums else round(base_p - default_tp_gap, 1)
        else:
            sl_val = sl_nums[0] if sl_nums else round(base_p - default_sl_gap * 0.8, 1)
            tp_val = tp_nums[0] if tp_nums else round(base_p + default_sl_gap * 0.8, 1)

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
    def analyze_market(
        cls,
        df_bars: pl.DataFrame,
        symbol: str = "TMF",
        timeframe: str = "5m",
        gemini_api_key: str = "",
        gemini_model: str = "gemini-2.5-flash",
        cloud_intel: Optional[CloudIntelSnapshot] = None,
    ) -> FuturesAdvice:
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
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

        # 1. 提取最新技術指標與價格
        ma5_s = IndicatorEngine.calculate_sma(df_bars, 5, "close")
        ma20_s = IndicatorEngine.calculate_sma(df_bars, 20, "close")
        ma60_s = IndicatorEngine.calculate_sma(df_bars, min(60, df_bars.height - 1), "close")
        rsi_s = IndicatorEngine.calculate_rsi(df_bars, 14, "close")
        bb_up_s, bb_low_s = IndicatorEngine.calculate_bollinger(df_bars, 20, "close", 2.0)
        adx_s = IndicatorEngine.calculate_adx(df_bars, 14)

        df = df_bars.with_columns([
            ma5_s.alias("ma5"),
            ma20_s.alias("ma20"),
            ma60_s.alias("ma60"),
            rsi_s.alias("rsi"),
            bb_up_s.alias("bb_up"),
            bb_low_s.alias("bb_low"),
            adx_s.alias("adx"),
        ])

        closes = df["close"].to_list()
        highs = df["high"].to_list()
        lows = df["low"].to_list()
        volumes = df["volume"].to_list()

        curr_p = float(closes[-1])
        ma5 = float(df["ma5"][-1]) if "ma5" in df.columns else curr_p
        ma20 = float(df["ma20"][-1]) if "ma20" in df.columns else curr_p
        ma60 = float(df["ma60"][-1]) if "ma60" in df.columns else curr_p
        rsi_val = float(df["rsi"][-1]) if "rsi" in df.columns and not np.isnan(df["rsi"][-1]) else 50.0
        adx_val = float(df["adx"][-1]) if "adx" in df.columns and not np.isnan(df["adx"][-1]) else 20.0
        bb_up = float(df["bb_up"][-1]) if "bb_up" in df.columns else curr_p * 1.01
        bb_low = float(df["bb_low"][-1]) if "bb_low" in df.columns else curr_p * 0.99

        # 計算布林帶寬度
        bb_width = ((bb_up - bb_low) / max(1.0, ma20)) * 100.0 if ma20 > 0 else 1.0
        regime = MarketRegimeDetector.detect_regime_at(adx_val, bb_width)

        # 近期支撐與壓力 (最近 30 根最高與最低)
        recent_n = min(30, len(closes))
        recent_high = max(highs[-recent_n:])
        recent_low = min(lows[-recent_n:])

        # 2. 本地量化打分邏輯 (Quantitative Scoring)
        long_score = 0
        short_score = 0

        # 均線排列判斷
        if ma5 > ma20:
            long_score += 25
            if curr_p > ma5:
                long_score += 15
        else:
            short_score += 25
            if curr_p < ma5:
                short_score += 15

        if ma20 > ma60:
            long_score += 15
        else:
            short_score += 15

        # RSI 動能
        if rsi_val > 55:
            long_score += 15
        elif rsi_val < 45:
            short_score += 15

        # 突破與布林位置
        if curr_p >= recent_high * 0.998:
            long_score += 20
        elif curr_p <= recent_low * 1.002:
            short_score += 20

        # 融合 Google 雲端四維度情報加權
        if cloud_intel is not None:
            if "偏多" in cloud_intel.global_bias_label:
                long_score += 10
            elif "偏空" in cloud_intel.global_bias_label:
                short_score += 10
            if "偏多" in cloud_intel.news_sentiment_label:
                long_score += 8
            elif "偏空" in cloud_intel.news_sentiment_label:
                short_score += 8
            if cloud_intel.is_in_morning_picks:
                long_score += 12
            if cloud_intel.is_in_cooling_off:
                short_score += 15

        # 正規化機率
        total_s = max(1, long_score + short_score)
        base_long_prob = round((long_score / total_s) * 100.0, 1)
        base_short_prob = round(100.0 - base_long_prob, 1)

        # 3. 嘗試呼叫 Gemini 進行專業分析 (若有 API Key)
        api_key = (gemini_api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        if api_key:
            advice = cls._call_gemini_advisor(
                api_key=api_key,
                model_name=gemini_model,
                symbol=symbol,
                timeframe=timeframe,
                curr_p=curr_p,
                ma5=ma5,
                ma20=ma20,
                ma60=ma60,
                rsi=rsi_val,
                adx=adx_val,
                regime=regime,
                recent_high=recent_high,
                recent_low=recent_low,
                df_slice=df.tail(15),
                now_str=now_str,
                cloud_intel=cloud_intel,
            )
            if advice:
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

        # 4. 本地量化回退分析 (當 Gemini 未設定或超時)
        # 自適應停損與進場區間 (期貨依點數，個股依百分比)
        if is_fut:
            sl_delta = max(30.0, curr_p * 0.003)
            tp_delta = max(60.0, curr_p * 0.006)
            entry_delta = 10.0
        else:
            sl_delta = max(1.0, curr_p * 0.02)   # 個股 2% 停損
            tp_delta = max(2.0, curr_p * 0.04)   # 個股 4% 目標
            entry_delta = max(0.5, curr_p * 0.005)

        intel_tail = ""
        if cloud_intel is not None:
            intel_tail = f" 結合雲端情報：{cloud_intel.global_bias_label}、{cloud_intel.news_sentiment_label}。"

        if base_long_prob >= 62.0:
            action = "BUY_LONG"
            action_label = "🟢 建議偏多買進 (BUY LONG)"
            conf = base_long_prob
            entry = f"{curr_p - entry_delta:,.1f} ~ {curr_p + entry_delta*0.3:,.1f} {unit}"
            sl_pt = round(curr_p - sl_delta, 1)
            tp_pt = round(curr_p + tp_delta, 1)
            rationale = (
                f"當前 {symbol} 5MA({ma5:,.1f}) 位於 20MA({ma20:,.1f}) 之上，呈現多頭排列；"
                f"RSI({rsi_val:.1f}) 動能偏多，市場狀態為【{regime.label_zh}】。{intel_tail}"
                f"下一階段建議偏多佈局，以跌破 {sl_pt} {unit} 為防守失效點。"
            )
        elif base_short_prob >= 62.0:
            action = "SELL_SHORT"
            action_label = "🔴 建議偏空賣出 (SELL SHORT)"
            conf = base_short_prob
            entry = f"{curr_p - entry_delta*0.3:,.1f} ~ {curr_p + entry_delta:,.1f} {unit}"
            sl_pt = round(curr_p + sl_delta, 1)
            tp_pt = round(curr_p - tp_delta, 1)
            rationale = (
                f"當前 {symbol} 5MA({ma5:,.1f}) 位於 20MA({ma20:,.1f}) 之下，空方掌控局勢；"
                f"RSI({rsi_val:.1f}) 偏弱，市場狀態為【{regime.label_zh}】。{intel_tail}"
                f"下一階段建議逢高偏空操作，以突破 {sl_pt} {unit} 為防守失效點。"
            )
        else:
            action = "WAIT"
            action_label = "⚪ 建議觀望休息 (WAIT)"
            conf = 50.0
            entry = "暫不建議進場"
            sl_pt = round(recent_low, 1)
            tp_pt = round(recent_high, 1)
            rationale = (
                f"目前 {symbol} 價格在 5MA 與 20MA 間震盪，多空拉鋸未分勝負 (多方機率 {base_long_prob}% vs 空方機率 {base_short_prob}%)。"
                f"市場屬於【{regime.label_zh}】。{intel_tail}此時進場容易遭雙巴，建議靜待方向明朗再行動作。"
            )

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
        )
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
        rsi: float,
        adx: float,
        regime: RegimeSnapshot,
        recent_high: float,
        recent_low: float,
        df_slice: pl.DataFrame,
        now_str: str,
        cloud_intel: Optional[CloudIntelSnapshot] = None,
    ) -> Optional[FuturesAdvice]:
        """呼叫 Gemini 進行專業操盤顧問分析（融合 Google 雲端四維度情報）"""
        try:
            is_fut = cls.is_futures(symbol)
            asset_label = "台指期貨" if is_fut else "個股股票"
            unit = "點" if is_fut else "元"
            role_title = "資深的台指期貨專業操盤手" if is_fut else "資深的台股專業技術分析師"

            # 整理近幾根 K 線數據
            k_lines = []
            for row in df_slice.iter_rows(named=True):
                t_str = row["timestamp"].strftime("%H:%M") if hasattr(row["timestamp"], "strftime") else str(row["timestamp"])
                k_lines.append(f"{t_str} | O:{row['open']:.1f} H:{row['high']:.1f} L:{row['low']:.1f} C:{row['close']:.1f} V:{row['volume']}")
            k_summary = "\n".join(k_lines[-10:])

            intel_block = cloud_intel.to_prompt_context() if cloud_intel else "【🌐 Google 雲端情報庫】：目前僅依據技術指標分析"

            prompt = f"""你是一位{role_title}與首席市場決策顧問。
現在請根據台新官方 API 回傳的最新【{asset_label} - {symbol}】({timeframe} K線) 真實市場數據，以及 Google 雲端收集的四維度跨市場情報，進行下一階段操作建議分析。

【當前即時盤勢數據】
- 分析標的: {symbol} ({asset_label})
- 最新成交價: {curr_p} {unit}
- K線週期: {timeframe}
- 移動平均線: 5MA={ma5:.1f}, 20MA={ma20:.1f}, 60MA={ma60:.1f}
- RSI (14): {rsi:.1f}
- ADX 趨勢強度: {adx:.1f}
- 市場形態: {regime.label_zh}
- 近期區間高低點: 高點 {recent_high:.1f} / 低點 {recent_low:.1f}

{intel_block}

【最近 10 根 K 棒走勢】
{k_summary}

【分析任務】
請綜合「技術面 K 線均線」、「國際跨市場連動 (那指期/台積ADR/費半)」、「最新財經新聞情緒」與「雲端法人籌碼/歷史預測勝率回顧」，評估下一階段【{symbol}】的走勢，並給出明確的交易方向建議 (偏多買進 / 偏空賣出 / 觀望休息)。
請以合適的價格單位（{unit}）設定進場區間、防守停損線與目標價。
必須以嚴格合法的 JSON 格式輸出，不得包含額外註解或 Markdown 標籤，格式如下：
{{
  "action": "BUY_LONG" 或 "SELL_SHORT" 或 "WAIT",
  "confidence": 75,
  "long_probability": 75,
  "short_probability": 25,
  "entry_zone": "建議進場區間",
  "stop_loss": "防守停損點 (跌破/突破防守點撤退)",
  "target_price": "第一道目標或壓力支撐點",
  "rationale": "繁體中文白話分析理由 (110~150字，同時點出技術面與雲端國際盤/消息面/籌碼面的綜合依據)"
}}
"""
            model_target = model_name or "gemini-2.5-flash"
            # 支援常見 Gemini 模型名稱相容
            if "gemini-3.5" in model_target:
                model_target = "gemini-2.5-flash"

            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_target}:generateContent?key={api_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "temperature": 0.2
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
                    act_label = "⚪ 建議觀望休息 (WAIT)"

                return FuturesAdvice(
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
                )
        except Exception as e:
            print(f"FuturesAIAdvisor: Gemini 呼叫例外，自動切換至本地量化回退: {e}")

        return None

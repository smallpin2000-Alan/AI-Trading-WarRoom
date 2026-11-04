# src/ai/cloud_intel.py
from __future__ import annotations
import os
import json
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import requests


@dataclass
class GlobalQuoteItem:
    symbol: str
    name: str
    price: float
    change_pct: float
    status_icon: str


@dataclass
class CloudIntelSnapshot:
    symbol: str
    fetched_at: str
    # 1. 全球跨市場即時指標
    global_quotes: List[GlobalQuoteItem] = field(default_factory=list)
    global_bias_label: str = "⚪ 國際盤中性"
    global_summary_text: str = "暫無國際盤即時數據"
    # 2. 即時新聞情緒
    news_headlines: List[str] = field(default_factory=list)
    news_sentiment_label: str = "⚪ 消息面中性"
    # 3. Google Cloud 法人籌碼與盤後覆盤連動
    gcs_connected: bool = False
    institutional_note: str = "尚未載入雲端日線籌碼資料"
    is_in_cooling_off: bool = False
    is_in_morning_picks: bool = False
    # 4. 雲端 AI 歷史預測勝率回顧
    ai_win_rate: Optional[float] = None
    ai_verified_count: int = 0
    ai_self_review_note: str = "累積預測樣本中"

    def to_prompt_context(self) -> str:
        """將四大雲端情報整合成結構化 Prompt 文字，供 Gemini AI 深度推理"""
        lines = ["【🌐 Google 雲端四維度即時情報庫】"]
        lines.append(f"1. 國際跨市場連動 ({self.global_bias_label})：{self.global_summary_text}")
        if self.news_headlines:
            top_news = "；".join(self.news_headlines[:3])
            lines.append(f"2. 最新即時財經新聞 ({self.news_sentiment_label})：{top_news}")
        else:
            lines.append("2. 最新即時財經新聞：目前無重大突發新聞")
        lines.append(f"3. 雲端法人籌碼與日線覆盤：{self.institutional_note}")
        if self.ai_win_rate is not None and self.ai_verified_count > 0:
            lines.append(
                f"4. AI 歷史預測自我回顧：過去已驗證 {self.ai_verified_count} 次預測，"
                f"方向準確率 {self.ai_win_rate:.0f}% ({self.ai_self_review_note})"
            )
        else:
            lines.append(f"4. AI 歷史預測自我回顧：{self.ai_self_review_note}")
        return "\n".join(lines)


class CloudIntelligenceHub:
    """
    Google Cloud 多維度情報收集與 AI 預測勝率追蹤中心
    整合：
      1. 全球跨市場指標 (NQ=F, TSM, ^SOX, TWD=X, ^VIX)
      2. Google News 即時新聞標題與多空情緒
      3. Google Cloud Storage (point-api-512405-stock-history) 法人籌碼與冷卻名單
      4. 雲端/本地雙向同步之 AI 歷史預測勝率追蹤器
    """

    _cache: Dict[str, tuple[float, Any]] = {}
    CACHE_TTL_SEC = 120.0  # 2 分鐘快取，確保看盤流暢不卡頓

    DEFAULT_GCS_BUCKET = os.getenv("GCS_BUCKET", "point-api-512405-stock-history").strip()
    LOCAL_PRED_PATH = Path("data/ai_predictions_log.json")
    LOCAL_INVESTMENT_HISTORY_DIR = Path(r"d:\一般資料\個人資料\Investment\history")

    GLOBAL_TARGETS = [
        ("NQ=F", "那指期"),
        ("TSM", "台積ADR"),
        ("^SOX", "費半指數"),
        ("TWD=X", "美元台幣"),
        ("^VIX", "VIX恐慌"),
    ]

    @classmethod
    def collect_all(cls, symbol: str, symbol_label: str = "", current_price: float = 0.0) -> CloudIntelSnapshot:
        now_ts = time.time()
        clean_sym = (symbol or "TMF").strip().upper()
        cache_key = f"intel_{clean_sym}"

        if cache_key in cls._cache:
            cached_time, cached_snap = cls._cache[cache_key]
            if now_ts - cached_time < cls.CACHE_TTL_SEC:
                # 即使使用快取，若有新價格仍即時更新預測驗證勝率
                if current_price > 0:
                    cls._update_and_evaluate_predictions(clean_sym, current_price, cached_snap)
                return cached_snap

        snap = CloudIntelSnapshot(
            symbol=clean_sym,
            fetched_at=time.strftime("%H:%M:%S"),
        )

        # 1. 抓取全球跨市場連動指標
        cls._populate_global_quotes(snap)

        # 2. 抓取 Google News 即時財經新聞
        cls._populate_realtime_news(snap, clean_sym, symbol_label)

        # 3. 讀取 Google Cloud Storage / 本地共享之盤前推薦與覆盤籌碼資料
        cls._populate_gcs_institutional(snap, clean_sym)

        # 4. 評估 AI 歷史預測準確率
        if current_price > 0:
            cls._update_and_evaluate_predictions(clean_sym, current_price, snap)
        else:
            cls._load_prediction_stats_only(clean_sym, snap)

        cls._cache[cache_key] = (now_ts, snap)
        return snap

    @classmethod
    def _populate_global_quotes(cls, snap: CloudIntelSnapshot) -> None:
        headers = {"User-Agent": "Mozilla/5.0"}
        quotes: List[GlobalQuoteItem] = []
        bull_pts = 0
        bear_pts = 0

        for ticker, name in cls.GLOBAL_TARGETS:
            try:
                encoded = urllib.parse.quote(ticker)
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}?interval=1d&range=2d"
                resp = requests.get(url, headers=headers, timeout=3.5)
                if resp.status_code == 200:
                    meta = resp.json().get("chart", {}).get("result", [{}])[0].get("meta", {})
                    price = float(meta.get("regularMarketPrice") or 0.0)
                    prev = float(meta.get("chartPreviousClose") or meta.get("previousClose") or price)
                    if price > 0 and prev > 0:
                        pct = ((price - prev) / prev) * 100.0
                        icon = "🔺" if pct > 0.05 else ("🔻" if pct < -0.05 else "➖")
                        quotes.append(GlobalQuoteItem(
                            symbol=ticker,
                            name=name,
                            price=price,
                            change_pct=round(pct, 2),
                            status_icon=icon,
                        ))
                        if ticker in ("NQ=F", "TSM", "^SOX"):
                            if pct >= 0.35:
                                bull_pts += 1
                            elif pct <= -0.35:
                                bear_pts += 1
                        elif ticker == "^VIX":
                            if pct >= 5.0:
                                bear_pts += 1
                            elif pct <= -3.0:
                                bull_pts += 1
            except Exception:
                continue

        snap.global_quotes = quotes
        if quotes:
            parts = [f"{q.name} {q.status_icon}{q.change_pct:+.2f}%" for q in quotes]
            snap.global_summary_text = " ｜ ".join(parts)
            if bull_pts > bear_pts:
                snap.global_bias_label = "🟢 國際盤偏多助漲"
            elif bear_pts > bull_pts:
                snap.global_bias_label = "🔴 國際盤偏空承壓"
            else:
                snap.global_bias_label = "⚪ 國際盤多空持平"

    @classmethod
    def _populate_realtime_news(cls, snap: CloudIntelSnapshot, clean_sym: str, symbol_label: str) -> None:
        is_fut = any(k in clean_sym for k in ["TMF", "TXF", "MTX", "MXF", "R1", "R2"])
        if is_fut:
            query = "台指期 OR 台股大盤 OR 外資期貨"
        else:
            name_part = symbol_label.replace(clean_sym, "").strip()
            query = f"{clean_sym} {name_part} 台股".strip()

        encoded_q = urllib.parse.quote(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_q}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        try:
            resp = requests.get(rss_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=4.0)
            if resp.status_code == 200 and resp.text:
                root = ET.fromstring(resp.text)
                titles: List[str] = []
                for item in root.findall(".//item")[:5]:
                    t_elem = item.find("title")
                    if t_elem is not None and t_elem.text:
                        clean_t = t_elem.text.split(" - ")[0].strip()
                        if clean_t and clean_t not in titles:
                            titles.append(clean_t)
                snap.news_headlines = titles

                pos_words = ["大漲", "創高", "買超", "利多", "看好", "成長", "突破", "強勢", "反彈", "雙升"]
                neg_words = ["大跌", "重挫", "賣超", "利空", "下修", "破底", "走弱", "砍價", "衰退", "跳水"]
                pos_cnt = sum(1 for t in titles for w in pos_words if w in t)
                neg_cnt = sum(1 for t in titles for w in neg_words if w in t)
                if pos_cnt > neg_cnt:
                    snap.news_sentiment_label = "🟢 消息面偏多"
                elif neg_cnt > pos_cnt:
                    snap.news_sentiment_label = "🔴 消息面偏空"
                else:
                    snap.news_sentiment_label = "⚪ 消息面中性"
        except Exception:
            pass

    @classmethod
    def _read_cloud_or_local_json(cls, filename: str) -> Optional[dict]:
        """優先從 Google Cloud Storage 儲存桶讀取，若本機無憑證則讀取本機共享目錄"""
        if cls.DEFAULT_GCS_BUCKET:
            try:
                from google.cloud import storage
                client = storage.Client()
                bucket = client.bucket(cls.DEFAULT_GCS_BUCKET)
                blob = bucket.blob(f"history/{filename}")
                if blob.exists():
                    return json.loads(blob.download_as_text(encoding="utf-8"))
            except Exception:
                pass

        local_file = cls.LOCAL_INVESTMENT_HISTORY_DIR / filename
        if local_file.exists():
            try:
                return json.loads(local_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return None

    @classmethod
    def _populate_gcs_institutional(cls, snap: CloudIntelSnapshot, clean_sym: str) -> None:
        notes: List[str] = []
        today_str = time.strftime("%Y-%m-%d")

        # 1. 檢查冷卻黑名單 (cooling_off_list.json)
        cooling_data = cls._read_cloud_or_local_json("cooling_off_list.json")
        if cooling_data is not None:
            snap.gcs_connected = True
            for item in cooling_data.get("cooling_stocks", []):
                if str(item.get("code", "")).strip() == clean_sym and item.get("cooldown_until", "") >= today_str:
                    snap.is_in_cooling_off = True
                    notes.append(f"⚠️ 雲端風控警示：{clean_sym} 目前處於破線冷卻觀察期 (至 {item.get('cooldown_until')})")
                    break

        # 2. 檢查今日或近期盤前嚴選清單 (picks_YYYY-MM-DD.json)
        picks_data = cls._read_cloud_or_local_json(f"picks_{today_str}.json")
        if picks_data and picks_data.get("picks"):
            snap.gcs_connected = True
            for p in picks_data.get("picks", []):
                if str(p.get("code", "")).strip() == clean_sym:
                    snap.is_in_morning_picks = True
                    notes.append(f"🌟 雲端早盤嚴選：{clean_sym} 列入今日 AI 法人籌碼多頭精選標的")
                    break
            if not snap.is_in_morning_picks:
                pick_names = [f"{p.get('code')}{p.get('name', '')}" for p in picks_data.get("picks", [])[:3]]
                if pick_names:
                    notes.append(f"今日雲端早盤焦點股：{'、'.join(pick_names)}")

        if notes:
            snap.institutional_note = "；".join(notes)
        else:
            snap.institutional_note = "雲端籌碼庫正常連線，該標的目前無特殊冷卻警示"

    @classmethod
    def record_prediction(
        cls,
        symbol: str,
        timeframe: str,
        action: str,
        current_price: float,
        target_val: float,
        stop_val: float,
    ) -> None:
        """將 AI 每次的多空預測記錄至本地與 Google Cloud，供後續驗證勝率"""
        if action not in ("BUY_LONG", "SELL_SHORT") or current_price <= 0:
            return
        try:
            records = cls._load_prediction_records()
            clean_sym = symbol.strip().upper()
            # 避免同一商品在 3 分鐘內重複寫入相同預測
            now_ts = time.time()
            for r in reversed(records):
                if r.get("symbol") == clean_sym and r.get("timeframe") == timeframe:
                    if now_ts - float(r.get("ts", 0)) < 180:
                        return
                    break

            records.append({
                "symbol": clean_sym,
                "timeframe": timeframe,
                "action": action,
                "entry_price": round(current_price, 2),
                "target_price": round(target_val, 2),
                "stop_loss": round(stop_val, 2),
                "ts": round(now_ts, 1),
                "time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "PENDING",  # PENDING / WIN / LOSS
            })
            # 僅保留最近 200 筆，輕量快速
            records = records[-200:]
            cls._save_prediction_records(records)
        except Exception:
            pass

    @classmethod
    def _update_and_evaluate_predictions(cls, clean_sym: str, current_price: float, snap: CloudIntelSnapshot) -> None:
        records = cls._load_prediction_records()
        changed = False
        now_ts = time.time()

        for r in records:
            if r.get("symbol") != clean_sym or r.get("status") != "PENDING":
                continue
            entry_p = float(r.get("entry_price", 0))
            target_p = float(r.get("target_price", 0))
            stop_p = float(r.get("stop_loss", 0))
            act = r.get("action", "")
            age_sec = now_ts - float(r.get("ts", now_ts))

            if entry_p <= 0:
                continue

            if act == "BUY_LONG":
                if target_p > entry_p and current_price >= target_p:
                    r["status"] = "WIN"
                    changed = True
                elif stop_p > 0 and current_price <= stop_p:
                    r["status"] = "LOSS"
                    changed = True
                elif age_sec >= 900:  # 超過 15 分鐘依實際順向漲跌結算
                    if current_price > entry_p * 1.0005:
                        r["status"] = "WIN"
                        changed = True
                    elif current_price < entry_p * 0.9995:
                        r["status"] = "LOSS"
                        changed = True
            elif act == "SELL_SHORT":
                if target_p > 0 and target_p < entry_p and current_price <= target_p:
                    r["status"] = "WIN"
                    changed = True
                elif stop_p > 0 and current_price >= stop_p:
                    r["status"] = "LOSS"
                    changed = True
                elif age_sec >= 900:
                    if current_price < entry_p * 0.9995:
                        r["status"] = "WIN"
                        changed = True
                    elif current_price > entry_p * 1.0005:
                        r["status"] = "LOSS"
                        changed = True

        if changed:
            cls._save_prediction_records(records)

        cls._compute_stats_from_records(records, clean_sym, snap)

    @classmethod
    def _load_prediction_stats_only(cls, clean_sym: str, snap: CloudIntelSnapshot) -> None:
        records = cls._load_prediction_records()
        cls._compute_stats_from_records(records, clean_sym, snap)

    @classmethod
    def _compute_stats_from_records(cls, records: List[dict], clean_sym: str, snap: CloudIntelSnapshot) -> None:
        sym_settled = [r for r in records if r.get("symbol") == clean_sym and r.get("status") in ("WIN", "LOSS")]
        if not sym_settled:
            sym_settled = [r for r in records if r.get("status") in ("WIN", "LOSS")]

        wins = sum(1 for r in sym_settled if r.get("status") == "WIN")
        total = len(sym_settled)
        snap.ai_verified_count = total
        if total > 0:
            snap.ai_win_rate = round((wins / total) * 100.0, 1)
            last_r = sym_settled[-1]
            outcome_zh = "✅ 命中目標" if last_r.get("status") == "WIN" else "❌ 觸及停損/反向"
            snap.ai_self_review_note = f"前次 ({last_r.get('time_str', '')[-8:]}) 建議 {last_r.get('action')} 結果：{outcome_zh}"
        else:
            pending_cnt = sum(1 for r in records if r.get("symbol") == clean_sym and r.get("status") == "PENDING")
            snap.ai_win_rate = None
            snap.ai_self_review_note = f"目前已記錄 {pending_cnt} 筆預測，等待後續行情驗證中"

    @classmethod
    def _load_prediction_records(cls) -> List[dict]:
        if cls.LOCAL_PRED_PATH.exists():
            try:
                return json.loads(cls.LOCAL_PRED_PATH.read_text(encoding="utf-8"))
            except Exception:
                pass
        # 若在 Cloud Run 環境且本機尚無快取檔，嘗試從 GCS 下載歷史預測紀錄
        if os.getenv("K_SERVICE") or os.getenv("GCS_BUCKET"):
            cloud_data = cls._read_cloud_or_local_json("ai_predictions_log.json")
            if isinstance(cloud_data, dict) and "records" in cloud_data:
                return cloud_data["records"]
        return []

    @classmethod
    def _save_prediction_records(cls, records: List[dict]) -> None:
        try:
            cls.LOCAL_PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
            cls.LOCAL_PRED_PATH.write_text(
                json.dumps(records, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            # 若在 Cloud Run 環境，同步備份至 Google Cloud Storage
            if (os.getenv("K_SERVICE") or os.getenv("GCS_BUCKET")) and cls.DEFAULT_GCS_BUCKET:
                try:
                    from google.cloud import storage
                    client = storage.Client()
                    bucket = client.bucket(cls.DEFAULT_GCS_BUCKET)
                    blob = bucket.blob("history/ai_predictions_log.json")
                    blob.upload_from_string(
                        json.dumps({"records": records}, ensure_ascii=False, indent=2),
                        content_type="application/json; charset=utf-8",
                    )
                except Exception:
                    pass
        except Exception:
            pass

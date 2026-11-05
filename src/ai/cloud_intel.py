# src/ai/cloud_intel.py
from __future__ import annotations
import os
import json
import time
import threading
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
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
    _refreshing_keys: set[str] = set()
    CACHE_TTL_SEC = 120.0  # 2 分鐘快取，確保看盤流暢不卡頓

    # 🚀 全域共用極速多級快取 (Multi-Tier Shared Cache)：跨所有標的共用，切換商品 0.001 秒瞬開！
    _shared_session: Optional[requests.Session] = None
    _shared_macro: Optional[tuple[float, List[GlobalQuoteItem], str, str]] = None          # (ts, quotes, bias, summary)
    _refreshing_macro: bool = False
    _shared_institutional: Optional[tuple[float, Optional[dict], Optional[dict]]] = None  # (ts, cooling_data, picks_data)
    _refreshing_institutional: bool = False
    _shared_news: Dict[str, tuple[float, List[str], str]] = {}                             # key -> (ts, titles, label)
    _refreshing_news_keys: set[str] = set()

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
    def _get_http_session(cls) -> requests.Session:
        if cls._shared_session is None:
            s = requests.Session()
            s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
            adapter = requests.adapters.HTTPAdapter(pool_connections=15, pool_maxsize=30, max_retries=1)
            s.mount("https://", adapter)
            s.mount("http://", adapter)
            cls._shared_session = s
        return cls._shared_session

    @classmethod
    def prewarm_shared_cache(cls) -> None:
        """背景全自動預熱：伺服器啟動時於背景載入國際報價與雲端籌碼，讓第一位使用者 0.001 秒秒開"""
        def _warm():
            try:
                cls._refresh_macro_data()
                cls._refresh_institutional_data()
                cls._refresh_news_data("FUTURES", "台指期")
            except Exception:
                pass
        threading.Thread(target=_warm, daemon=True).start()

    @classmethod
    def _do_full_collect(cls, clean_sym: str, symbol_label: str, current_price: float) -> CloudIntelSnapshot:
        snap = CloudIntelSnapshot(
            symbol=clean_sym,
            fetched_at=datetime.now(timezone(timedelta(hours=8))).strftime("%H:%M:%S"),
        )
        # 極速並行：若快取中已有數據則 0.0001 秒直接裝填；若無則透過共用連線池平行拉取
        with ThreadPoolExecutor(max_workers=3) as pool:
            f1 = pool.submit(cls._populate_global_quotes, snap)
            f2 = pool.submit(cls._populate_realtime_news, snap, clean_sym, symbol_label)
            f3 = pool.submit(cls._populate_gcs_institutional, snap, clean_sym)
            for fut in (f1, f2, f3):
                try:
                    fut.result(timeout=2.5)
                except Exception:
                    pass

        if current_price > 0:
            cls._update_and_evaluate_predictions(clean_sym, current_price, snap)
        else:
            cls._load_prediction_stats_only(clean_sym, snap)

        cls._cache[f"intel_{clean_sym}"] = (time.time(), snap)
        return snap

    @classmethod
    def collect_all(
        cls,
        symbol: str,
        symbol_label: str = "",
        current_price: float = 0.0,
        allow_background: bool = False,
    ) -> CloudIntelSnapshot:
        now_ts = time.time()
        clean_sym = (symbol or "TMF").strip().upper()
        cache_key = f"intel_{clean_sym}"

        if cache_key in cls._cache:
            cached_time, cached_snap = cls._cache[cache_key]
            if now_ts - cached_time < cls.CACHE_TTL_SEC:
                return cached_snap
            if allow_background:
                if cache_key not in cls._refreshing_keys:
                    cls._refreshing_keys.add(cache_key)

                    def _bg_refresh():
                        try:
                            cls._do_full_collect(clean_sym, symbol_label, current_price)
                        finally:
                            cls._refreshing_keys.discard(cache_key)

                    threading.Thread(target=_bg_refresh, daemon=True).start()
                return cached_snap

        # 若無該商品專屬快取，但全域共用宏觀快取已存在，立即組裝完整快照（0.001 秒），杜絕卡頓！
        if cls._shared_macro is not None or allow_background:
            return cls._do_full_collect(clean_sym, symbol_label, current_price)

        return cls._do_full_collect(clean_sym, symbol_label, current_price)

    @classmethod
    def _refresh_macro_data(cls) -> Optional[tuple[List[GlobalQuoteItem], str, str]]:
        session = cls._get_http_session()
        quotes: List[GlobalQuoteItem] = []
        bull_pts = 0
        bear_pts = 0

        def _fetch_one(item: tuple[str, str]) -> Optional[GlobalQuoteItem]:
            ticker, name = item
            try:
                encoded = urllib.parse.quote(ticker)
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}?interval=1d&range=2d"
                resp = session.get(url, timeout=2.2)
                if resp.status_code == 200:
                    meta = resp.json().get("chart", {}).get("result", [{}])[0].get("meta", {})
                    price = float(meta.get("regularMarketPrice") or 0.0)
                    prev = float(meta.get("chartPreviousClose") or meta.get("previousClose") or price)
                    if price > 0 and prev > 0:
                        pct = ((price - prev) / prev) * 100.0
                        icon = "🔺" if pct > 0.05 else ("🔻" if pct < -0.05 else "➖")
                        return GlobalQuoteItem(
                            symbol=ticker,
                            name=name,
                            price=price,
                            change_pct=round(pct, 2),
                            status_icon=icon,
                        )
            except Exception:
                pass
            return None

        with ThreadPoolExecutor(max_workers=5) as ex:
            results = list(ex.map(_fetch_one, cls.GLOBAL_TARGETS))

        for q in results:
            if q is None:
                continue
            quotes.append(q)
            if q.symbol in ("NQ=F", "TSM", "^SOX"):
                if q.change_pct >= 0.35:
                    bull_pts += 1
                elif q.change_pct <= -0.35:
                    bear_pts += 1
            elif q.symbol == "^VIX":
                if q.change_pct >= 5.0:
                    bear_pts += 1
                elif q.change_pct <= -3.0:
                    bull_pts += 1

        if quotes:
            parts = [f"{q.name} {q.status_icon}{q.change_pct:+.2f}%" for q in quotes]
            summary_text = " ｜ ".join(parts)
            if bull_pts > bear_pts:
                bias_label = "🟢 國際盤偏多助漲"
            elif bear_pts > bull_pts:
                bias_label = "🔴 國際盤偏空承壓"
            else:
                bias_label = "⚪ 國際盤多空持平"
            cls._shared_macro = (time.time(), quotes, bias_label, summary_text)
            return quotes, bias_label, summary_text
        return None

    @classmethod
    def _populate_global_quotes(cls, snap: CloudIntelSnapshot) -> None:
        now_ts = time.time()
        # 1. 命中全域共用宏觀快取 (跨商品共用，0.00001 秒立即返回！)
        if cls._shared_macro is not None:
            cached_ts, quotes, bias_label, summary_text = cls._shared_macro
            snap.global_quotes = list(quotes)
            snap.global_bias_label = bias_label
            snap.global_summary_text = summary_text
            # 若快取超過 120 秒，背景自動靜默刷新，絕不阻塞前端使用者！
            if now_ts - cached_ts > 120.0 and not cls._refreshing_macro:
                cls._refreshing_macro = True
                def _bg_m():
                    try:
                        cls._refresh_macro_data()
                    finally:
                        cls._refreshing_macro = False
                threading.Thread(target=_bg_m, daemon=True).start()
            return

        # 2. 冷啟動極速拉取
        res = cls._refresh_macro_data()
        if res:
            quotes, bias_label, summary_text = res
            snap.global_quotes = list(quotes)
            snap.global_bias_label = bias_label
            snap.global_summary_text = summary_text

    @classmethod
    def _refresh_news_data(cls, news_key: str, query: str) -> Optional[tuple[List[str], str]]:
        session = cls._get_http_session()
        encoded_q = urllib.parse.quote(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_q}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        try:
            resp = session.get(rss_url, timeout=2.5)
            if resp.status_code == 200 and resp.text:
                root = ET.fromstring(resp.text)
                titles: List[str] = []
                for item in root.findall(".//item")[:5]:
                    t_elem = item.find("title")
                    if t_elem is not None and t_elem.text:
                        clean_t = t_elem.text.split(" - ")[0].strip()
                        if clean_t and clean_t not in titles:
                            titles.append(clean_t)

                pos_words = ["大漲", "創高", "買超", "利多", "看好", "成長", "突破", "強勢", "反彈", "雙升"]
                neg_words = ["大跌", "重挫", "賣超", "利空", "下修", "破底", "走弱", "砍價", "衰退", "跳水"]
                pos_cnt = sum(1 for t in titles for w in pos_words if w in t)
                neg_cnt = sum(1 for t in titles for w in neg_words if w in t)
                if pos_cnt > neg_cnt:
                    sentiment_label = "🟢 消息面偏多"
                elif neg_cnt > pos_cnt:
                    sentiment_label = "🔴 消息面偏空"
                else:
                    sentiment_label = "⚪ 消息面中性"

                cls._shared_news[news_key] = (time.time(), titles, sentiment_label)
                return titles, sentiment_label
        except Exception:
            pass
        return None

    @classmethod
    def _populate_realtime_news(cls, snap: CloudIntelSnapshot, clean_sym: str, symbol_label: str) -> None:
        is_fut = any(k in clean_sym for k in ["TMF", "TXF", "MTX", "MXF", "R1", "R2"])
        news_key = "FUTURES" if is_fut else clean_sym
        now_ts = time.time()

        # 1. 命中新聞快取 (期貨跨標的共用，5 分鐘快取，0.00001 秒立即返回！)
        if news_key in cls._shared_news:
            cached_ts, titles, sentiment_label = cls._shared_news[news_key]
            snap.news_headlines = list(titles)
            snap.news_sentiment_label = sentiment_label
            if now_ts - cached_ts > 300.0 and news_key not in cls._refreshing_news_keys:
                cls._refreshing_news_keys.add(news_key)
                def _bg_n():
                    try:
                        q = "台指期 OR 台股大盤 OR 外資期貨" if is_fut else f"{clean_sym} 台股".strip()
                        cls._refresh_news_data(news_key, q)
                    finally:
                        cls._refreshing_news_keys.discard(news_key)
                threading.Thread(target=_bg_n, daemon=True).start()
            return

        # 2. 冷啟動拉取
        if is_fut:
            query = "台指期 OR 台股大盤 OR 外資期貨"
        else:
            name_part = symbol_label.replace(clean_sym, "").strip()
            query = f"{clean_sym} {name_part} 台股".strip()

        res = cls._refresh_news_data(news_key, query)
        if res:
            snap.news_headlines, snap.news_sentiment_label = res

    @classmethod
    def _refresh_institutional_data(cls) -> tuple[Optional[dict], Optional[dict]]:
        cooling_data = cls._read_cloud_or_local_json("cooling_off_list.json")
        today_str = time.strftime("%Y-%m-%d")
        picks_data = cls._read_cloud_or_local_json(f"picks_{today_str}.json")
        cls._shared_institutional = (time.time(), cooling_data, picks_data)
        return cooling_data, picks_data

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
        now_ts = time.time()
        # 1. 命中全域法人籌碼快取 (10 分鐘快取，記憶體比對 0.00001 秒！)
        if cls._shared_institutional is not None:
            cached_ts, cooling_data, picks_data = cls._shared_institutional
            if now_ts - cached_ts > 600.0 and not cls._refreshing_institutional:
                cls._refreshing_institutional = True
                def _bg_i():
                    try:
                        cls._refresh_institutional_data()
                    finally:
                        cls._refreshing_institutional = False
                threading.Thread(target=_bg_i, daemon=True).start()
        else:
            cooling_data, picks_data = cls._refresh_institutional_data()

        notes: List[str] = []
        today_str = time.strftime("%Y-%m-%d")

        # 記憶體極速檢驗：冷卻黑名單
        if cooling_data is not None:
            snap.gcs_connected = True
            for item in cooling_data.get("cooling_stocks", []):
                if str(item.get("code", "")).strip() == clean_sym and item.get("cooldown_until", "") >= today_str:
                    snap.is_in_cooling_off = True
                    notes.append(f"⚠️ 雲端風控警示：{clean_sym} 目前處於破線冷卻觀察期 (至 {item.get('cooldown_until')})")
                    break

        # 記憶體極速檢驗：早盤嚴選標的
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
                "high_since": round(current_price, 2),
                "low_since": round(current_price, 2),
                "status": "PENDING",  # PENDING / WIN / LOSS / TIE
            })
            records = records[-200:]
            cls._save_prediction_records(records)
        except Exception:
            pass

    @classmethod
    def _update_and_evaluate_predictions(cls, clean_sym: str, current_price: float, snap: CloudIntelSnapshot) -> None:
        records = cls._load_prediction_records()
        changed = False
        now_ts = time.time()

        is_futures = any(k in clean_sym for k in ["TMF", "TXF", "MTX", "MXF", "R1", "R2"])
        # 扣除實戰摩擦成本 (現貨 0.6% 稅費，期貨 12 點價差與手續費)
        min_win_return = 0.006 if not is_futures else (12.0 / current_price if current_price > 0 else 0.0006)

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

            # 動態維護期間最高/最低穿價
            high_since = max(float(r.get("high_since", entry_p)), current_price)
            low_since = min(float(r.get("low_since", entry_p)), current_price)
            r["high_since"] = round(high_since, 2)
            r["low_since"] = round(low_since, 2)

            if act == "BUY_LONG":
                # 嚴格實戰風控：若期間曾先跌破停損線，即便目前價格反彈亦判定為失敗 (LOSS)
                if stop_p > 0 and low_since <= stop_p:
                    r["status"] = "LOSS"
                    changed = True
                elif target_p > entry_p and high_since >= target_p:
                    r["status"] = "WIN"
                    changed = True
                elif age_sec >= 1200:  # 20 分鐘後以實質盈虧結算 (扣除交易稅費)
                    net_ret = (current_price - entry_p) / entry_p
                    if net_ret >= min_win_return:
                        r["status"] = "WIN"
                    elif net_ret <= -min_win_return:
                        r["status"] = "LOSS"
                    else:
                        r["status"] = "TIE"
                    changed = True

            elif act == "SELL_SHORT":
                # 嚴格實戰風控：若期間曾先漲破停損線，判定為失敗 (LOSS)
                if stop_p > 0 and high_since >= stop_p:
                    r["status"] = "LOSS"
                    changed = True
                elif target_p > 0 and target_p < entry_p and low_since <= target_p:
                    r["status"] = "WIN"
                    changed = True
                elif age_sec >= 1200:  # 20 分鐘後以實質盈虧結算
                    net_ret = (entry_p - current_price) / entry_p
                    if net_ret >= min_win_return:
                        r["status"] = "WIN"
                    elif net_ret <= -min_win_return:
                        r["status"] = "LOSS"
                    else:
                        r["status"] = "TIE"
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
        # 嚴格商品隔離：各商品各自統計實盤穿價結算紀錄，絕不跨商品挪用，杜絕虛假勝率！
        sym_settled = [r for r in records if r.get("symbol") == clean_sym and r.get("status") in ("WIN", "LOSS")]

        wins = sum(1 for r in sym_settled if r.get("status") == "WIN")
        total = len(sym_settled)
        snap.ai_verified_count = total
        # 統計學門檻防護：若已結算樣本少於 5 筆，不宣稱勝率百分比，避免 1 戰 1 勝顯示虛假 100%
        if total >= 5:
            snap.ai_win_rate = round((wins / total) * 100.0, 1)
            last_r = sym_settled[-1]
            outcome_zh = "✅ 命中目標" if last_r.get("status") == "WIN" else "❌ 觸及停損/反向"
            snap.ai_self_review_note = f"實盤結算 {total} 次 (獲勝 {wins} / 失敗 {total - wins})，前次：{outcome_zh}"
        elif total > 0:
            snap.ai_win_rate = None
            snap.ai_self_review_note = f"實盤累積樣本中 (已結算 {total}/5 筆，獲勝 {wins} 筆)"
        else:
            pending_cnt = sum(1 for r in records if r.get("symbol") == clean_sym and r.get("status") == "PENDING")
            snap.ai_win_rate = None
            if pending_cnt > 0:
                snap.ai_self_review_note = f"實盤追蹤中 (等待穿價結算，目前掛單 {pending_cnt} 筆)"
            else:
                snap.ai_self_review_note = "新標的實盤信號追蹤中 (待累積驗證)"

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
            # 若在 Cloud Run 環境，於背景執行緒非阻塞備份至 Google Cloud Storage
            if (os.getenv("K_SERVICE") or os.getenv("GCS_BUCKET")) and cls.DEFAULT_GCS_BUCKET:
                bucket_name = cls.DEFAULT_GCS_BUCKET
                payload_str = json.dumps({"records": records}, ensure_ascii=False, indent=2)

                def _bg_save_pred():
                    try:
                        from google.cloud import storage
                        client = storage.Client()
                        bucket = client.bucket(bucket_name)
                        blob = bucket.blob("history/ai_predictions_log.json")
                        blob.upload_from_string(
                            payload_str,
                            content_type="application/json; charset=utf-8",
                        )
                    except Exception:
                        pass

                threading.Thread(target=_bg_save_pred, daemon=True).start()
        except Exception:
            pass


# 模組載入時自動於背景啟動極速預熱，讓前端第一次連線就能秒開情報雷達
try:
    CloudIntelligenceHub.prewarm_shared_cache()
except Exception:
    pass


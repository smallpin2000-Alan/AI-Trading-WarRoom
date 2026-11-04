# src/ui/dashboard.py
from __future__ import annotations
import os
import gc
import json
import time
import threading
from pathlib import Path
import numpy as np
import polars as pl
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.ai.advisor import FuturesAIAdvisor, FuturesAdvice
from src.ai.cloud_intel import CloudIntelligenceHub, CloudIntelSnapshot
from src.broker.taishin_quote import TaishinQuoteFeed


def load_taishin_default_config() -> dict:
    cfg = {
        "symbol": os.getenv("SOL_SYMBOL", "2330"),
        "sol_user_id": os.getenv("SOL_USER_ID", ""),
        "sol_password": os.getenv("SOL_PASSWORD", ""),
        "gemini_api_key": os.getenv("GEMINI_API_KEY", ""),
        "gemini_model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    }
    paths = [
        Path("config/taishin_config.json"),
        Path(r"D:\一般資料\工具\taishin_dashboard\taishin_config.json"),
    ]
    for p in paths:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    file_cfg = json.load(f)
                    for k, v in file_cfg.items():
                        if not cfg.get(k):
                            cfg[k] = v
                    break
            except Exception:
                pass
    return cfg


def mask_id_number(raw_id: str) -> str:
    raw_id = (raw_id or "").strip()
    if len(raw_id) <= 3:
        return raw_id
    return raw_id[:3] + "*" * (len(raw_id) - 3)


st.set_page_config(
    page_title="AI 智慧看盤與多空決策戰情室",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 深色專業看盤主題 CSS
st.markdown("""
<style>
    /* 隱藏不必要的空白與側邊欄按鈕干擾 */
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 2rem !important;
    }
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] > div,
    [data-testid="stMetricValue"] * {
        font-size: clamp(0.95rem, 1.2vw, 1.35rem) !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        word-break: break-word !important;
        line-height: 1.35 !important;
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] > div,
    [data-testid="stMetricLabel"] p {
        font-size: 0.85rem !important;
        white-space: normal !important;
        line-height: 1.25 !important;
    }
    h1 {
        font-size: 1.65rem !important;
        margin-bottom: 0.2rem !important;
    }
    h2, h3 {
        font-size: 1.2rem !important;
    }
</style>
""", unsafe_allow_html=True)


DEFAULT_WATCHLIST = [
    "2330 (台積電)",
    "2317 (鴻海)",
    "2454 (聯發科)",
    "0050 (元大台灣50)",
    "TMFR1 (微台指近月)",
    "TXFR1 (大台指近月)",
    "MTXR1 (小台指近月)",
]


def load_watchlist(path: str | Path = "config/watchlist.json") -> list[str]:
    p = Path(path)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list) and len(data) > 0:
                return [str(x).strip() for x in data if str(x).strip()]
        except Exception:
            pass
    return list(DEFAULT_WATCHLIST)


def save_watchlist(items: list[str], path: str | Path = "config/watchlist.json") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def add_symbol_to_watchlist(code: str, name_note: str = "", path: str | Path = "config/watchlist.json") -> tuple[bool, str, list[str]]:
    clean_code = (code or "").strip().upper()
    if not clean_code:
        return False, "請輸入商品代碼（例如 2317、2603 或 TMFR1）", load_watchlist(path)
    items = load_watchlist(path)
    existing_codes = {item.split(" ")[0].upper() for item in items}
    if clean_code in existing_codes:
        return False, f"商品代碼 `{clean_code}` 已經在清單中了！", items

    note = (name_note or "").strip()
    entry = f"{clean_code} ({note})" if note else clean_code
    items.insert(0, entry)
    save_watchlist(items, path)
    return True, f"已成功新增 `{entry}` 至自選清單！", items


def remove_symbol_from_watchlist(target_entry: str, path: str | Path = "config/watchlist.json") -> tuple[bool, str, list[str]]:
    items = load_watchlist(path)
    if len(items) <= 1:
        return False, "自選清單至少需保留 1 項商品，無法全部刪除。", items
    if target_entry not in items:
        return False, f"找不到 `{target_entry}`", items
    items.remove(target_entry)
    save_watchlist(items, path)
    return True, f"已從自選清單移除 `{target_entry}`。", items


APP_VERSION = "v2.5.2"
_refreshing_advice_keys: set[str] = set()


@st.cache_resource
def get_global_taishin_feed() -> TaishinQuoteFeed:
    """跨瀏覽器分頁與 Rerun 共用唯一台新連線實例，徹底防止 Too Many Connections"""
    return TaishinQuoteFeed(auto_load_store=True)


@st.cache_resource
def get_global_advice_cache() -> dict:
    """跨所有訪客共用的 AI 決策快取 (key -> (timestamp, bar_count, FuturesAdvice))，讓新訪客 0.001 秒直接秒開"""
    return {}


def main():
    # 取得全域共用台新 API 連線單例與跨訪客 AI 快取
    feed = get_global_taishin_feed()
    global_advice_cache = get_global_advice_cache()
    if getattr(feed, "VERSION", None) != TaishinQuoteFeed.VERSION:
        feed.__class__ = TaishinQuoteFeed
        feed.VERSION = TaishinQuoteFeed.VERSION
    if "載入中" in feed.status_msg and len(feed.kbars_1m) > 0:
        feed.ta_rcv_done = True
        feed.status_msg = f"🟢 已就緒 ({feed.symbol}: {len(feed.kbars_1m)} 根 1分K)"
    st.session_state["taishin_feed"] = feed

    # 帳密僅在伺服器後端讀取，絕不顯示在前端網頁
    default_cfg = load_taishin_default_config()
    actual_user = default_cfg.get("sol_user_id", "")
    ts_pwd = default_cfg.get("sol_password", "")

    # 30 分鐘自動防呆休眠計時器 (避免忘記關網頁耗費雲端資源，確保 100% 0 元)
    if "last_active_time" not in st.session_state:
        st.session_state["last_active_time"] = time.time()

    elapsed_sec = time.time() - st.session_state["last_active_time"]
    is_idle_sleeping = elapsed_sec > 1800  # 30 分鐘

    # ==================== 頂部標題區（標示軟體版本號） ====================
    st.title(f"🎯 AI 智慧看盤與多空決策戰情室 `{APP_VERSION}`")
    st.caption(f"🏷️ **版本號：{APP_VERSION} (2026-11-04 極速雲端版)** ｜ 🛡️ 串接台新官方 API 與期交所 0 延遲 K 線 • Gemini AI 多空決策 • 純報價零下單風險")

    if is_idle_sleeping:
        st.warning("⏰【智慧防呆休眠中】您已連續看盤超過 30 分鐘。為保護雲端免費額度（確保 0 元），自動刷新已暫停。")
        if st.button("🔄 繼續看盤（重設 30 分鐘計時）", type="primary", key="btn_wake_idle"):
            st.session_state["last_active_time"] = time.time()
            st.rerun()

    # ==================== 1. 頂部統一控制列（全站唯一選商品處） ====================
    watchlist_items = load_watchlist()
    default_idx = 0
    if "selected_watchlist_item" in st.session_state and st.session_state["selected_watchlist_item"] in watchlist_items:
        default_idx = watchlist_items.index(st.session_state["selected_watchlist_item"])

    ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns([1.8, 1.0, 1.3, 1.1])
    with ctrl_c1:
        ts_symbol = st.selectbox(
            "📌 選擇分析商品（股票 / 期貨）",
            watchlist_items,
            index=default_idx,
            key="watchlist_selectbox",
        )
        st.session_state["selected_watchlist_item"] = ts_symbol
        clean_sym = ts_symbol.split(" ")[0]

    with ctrl_c2:
        tf_label = st.selectbox(
            "⏱️ K 線週期",
            ["5m (推薦)", "1m", "15m", "30m", "60m"],
            index=0,
            key="main_tf_select",
        )
        chart_tf = tf_label.split(" ")[0]

    with ctrl_c3:
        refresh_opt = st.selectbox(
            "🔄 畫面自動更新頻率 (離線自動休眠)",
            ["1 分鐘 (推薦)", "3 分鐘", "5 分鐘", "30 秒", "10 秒", "關閉自動刷新"],
            index=0,
            key="live_refresh_interval_opt",
        )
        if refresh_opt == "關閉自動刷新" or is_idle_sleeping:
            auto_refresh_sec = None
        else:
            opt_str = refresh_opt.split(" ")[0]
            auto_refresh_sec = int(opt_str) * 60 if "分鐘" in refresh_opt else int(opt_str)

    with ctrl_c4:
        st.write("")
        manual_trigger = st.button("🚀 連線 / 重新 AI 分析", type="primary", width="stretch")

    # 當使用者切換下拉選單商品時，自動熱切換訂閱標的並載入新商品 K 線
    if clean_sym and feed.symbol != clean_sym:
        st.session_state["last_active_time"] = time.time()
        feed.switch_symbol(clean_sym, actual_user, ts_pwd)
        if len(feed.kbars_1m) == 0:
            with st.spinner(f"正在從台新伺服器載入 {ts_symbol} 真實 K 線..."):
                for _ in range(12):
                    time.sleep(0.4)
                    if feed.ta_rcv_done or len(feed.kbars_1m) > 0 or "失敗" in feed.status_msg or "異常" in feed.status_msg:
                        break
        st.rerun()

    # 手動點擊「連線 / 重新 AI 分析」按鈕
    if manual_trigger:
        st.session_state["last_active_time"] = time.time()
        with st.spinner(f"正在同步 {ts_symbol} 今日最新 0 秒延遲 K 線與報價..."):
            feed._sync_realtime_kbars_and_quote(clean_sym)
            if not feed.connected and actual_user and ts_pwd:
                feed.start(actual_user, ts_pwd, clean_sym)
        # 清除舊分析快取與雲端情報快取，強制重抓最新國際盤/新聞並重算 AI 建議
        advice_cache_key = f"ai_advice_{clean_sym}_{chart_tf}"
        st.session_state.pop(advice_cache_key, None)
        global_advice_cache.pop(advice_cache_key, None)
        CloudIntelligenceHub._cache.pop(f"intel_{clean_sym}", None)
        st.rerun()

    # 自選商品增刪管理（預設收合，保持畫面乾淨）
    with st.expander("➕ / 🗑️ 管理自選商品清單（新增或移除股票 / 期貨代碼）"):
        wl_c1, wl_c2, wl_c3, wl_c4 = st.columns([1.2, 1.5, 1.0, 1.1])
        with wl_c1:
            new_sym_code = st.text_input("新增商品代碼", placeholder="例如: 2603 或 TMFR1", key="in_new_sym_code")
        with wl_c2:
            new_sym_note = st.text_input("商品名稱 (選填)", placeholder="例如: 長榮", key="in_new_sym_note")
        with wl_c3:
            st.write("")
            if st.button("➕ 新增商品", width="stretch", type="primary"):
                ok_add, msg_add, updated_list = add_symbol_to_watchlist(new_sym_code, new_sym_note)
                if ok_add:
                    st.session_state["selected_watchlist_item"] = updated_list[0]
                    st.toast(msg_add)
                    st.rerun()
                else:
                    st.warning(msg_add)
        with wl_c4:
            st.write("")
            if st.button(f"🗑️ 移除目前商品 ({clean_sym})", width="stretch"):
                ok_rm, msg_rm, updated_list = remove_symbol_from_watchlist(ts_symbol)
                if ok_rm:
                    st.session_state["selected_watchlist_item"] = updated_list[0]
                    st.toast(msg_rm)
                    st.rerun()
                else:
                    st.warning(msg_rm)

    st.markdown("---")

    # ==================== 2. 主戰情室區塊（支援自動定時刷新） ====================
    @st.fragment(run_every=auto_refresh_sec)
    def _render_unified_war_room():
        feed.ensure_fresh_data(clean_sym, actual_user, ts_pwd)
        sync_info = feed.get_server_sync_info(clean_sym)
        df_bars = feed.get_bars(timeframe=chart_tf, session="day_and_night")
        is_fut = FuturesAIAdvisor.is_futures(clean_sym)
        unit = "點" if is_fut else "元"

        latest_tk = feed.get_latest_tick()
        curr_price = latest_tk.price if latest_tk else (float(df_bars["close"][-1]) if df_bars.height > 0 else 0.0)
        curr_vol = latest_tk.volume if latest_tk else (int(df_bars["volume"][-1]) if df_bars.height > 0 else 0)
        last_bar_time_str = df_bars["timestamp"][-1].strftime("%Y-%m-%d %H:%M") if df_bars.height > 0 else "--"
        curr_time_str = (
            latest_tk.timestamp.strftime("%H:%M:%S")
            if latest_tk
            else last_bar_time_str
        )

        # 收集 Google 雲端四維度情報 (SWR 無阻塞背景更新架構，主執行緒 0.001 秒立即返回！)
        intel_snap = CloudIntelligenceHub.collect_all(
            symbol=clean_sym,
            symbol_label=ts_symbol,
            current_price=curr_price,
            allow_background=True,
        )

        # 計算或讀取 AI 決策建議（採用 SWR 零等待架構：先以 0.01 秒顯示現有快取或極速量化結果，背景執行緒非阻塞呼叫 Gemini AI！）
        advice_key = f"ai_advice_{clean_sym}_{chart_tf}"
        now_ts = time.time()
        bar_sig = f"{df_bars.height}_{last_bar_time_str}"
        cached_entry = global_advice_cache.get(advice_key)
        if (
            cached_entry is not None
            and (now_ts - cached_entry[0] < 90.0)
            and cached_entry[1] == bar_sig
        ):
            advice: FuturesAdvice = cached_entry[2]
        else:
            if cached_entry is not None:
                advice = cached_entry[2]
            else:
                # 首次載入此商品：0.01 秒極速算出本地量化多空決策，絕不讓畫面卡住乾等 5 秒 Gemini API！
                advice = FuturesAIAdvisor.analyze_market(
                    df_bars=df_bars,
                    symbol=clean_sym,
                    timeframe=chart_tf,
                    gemini_api_key=default_cfg.get("gemini_api_key", ""),
                    gemini_model=default_cfg.get("gemini_model", "gemini-2.5-flash"),
                    cloud_intel=intel_snap,
                    fast_local_only=True,
                )
                global_advice_cache[advice_key] = (0.0, bar_sig, advice)

            # 背景執行緒非阻塞呼叫 Gemini 2.5 Flash 深度分析，算完自動更新全域快取
            if advice_key not in _refreshing_advice_keys and df_bars.height >= 10:
                _refreshing_advice_keys.add(advice_key)
                _df_copy = df_bars.clone()
                _sym_copy = clean_sym
                _lbl_copy = ts_symbol
                _tf_copy = chart_tf
                _p_copy = curr_price
                _sig_copy = bar_sig
                _key_copy = default_cfg.get("gemini_api_key", "")
                _mdl_copy = default_cfg.get("gemini_model", "gemini-2.5-flash")

                def _bg_compute_gemini():
                    try:
                        full_intel = CloudIntelligenceHub.collect_all(
                            symbol=_sym_copy,
                            symbol_label=_lbl_copy,
                            current_price=_p_copy,
                            allow_background=False,
                        )
                        full_adv = FuturesAIAdvisor.analyze_market(
                            df_bars=_df_copy,
                            symbol=_sym_copy,
                            timeframe=_tf_copy,
                            gemini_api_key=_key_copy,
                            gemini_model=_mdl_copy,
                            cloud_intel=full_intel,
                            fast_local_only=False,
                        )
                        global_advice_cache[advice_key] = (time.time(), _sig_copy, full_adv)
                    except Exception:
                        pass
                    finally:
                        _refreshing_advice_keys.discard(advice_key)

                threading.Thread(target=_bg_compute_gemini, daemon=True).start()

        # 📡 伺服器資料拉取驗證狀態列（僅在確實從遠端伺服器拉到資料後才會顯示綠色與更新時間）
        if sync_info["is_synced"]:
            sync_border = "#089981"
            sync_bg = "rgba(8, 153, 129, 0.12)"
            sync_badge = f"🟢 已從伺服器拉取最新資料（{sync_info['seconds_ago']} 秒前驗證）"
        else:
            sync_border = "#f0b90b"
            sync_bg = "rgba(240, 185, 11, 0.12)"
            sync_badge = "🟡 尚未從伺服器拉取最新資料（目前顯示快取，請點擊上方「🚀 連線 / 重新 AI 分析」）"

        latest_1m_str = sync_info["bar_time_str"] if sync_info["bar_time_str"] != "--" else last_bar_time_str
        bar_display_label = latest_1m_str

        st.markdown(f"""
        <div style="background-color: {sync_bg}; border: 1px solid {sync_border}; border-radius: 8px; padding: 8px 16px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; font-size: 13.5px; color: #e0e3eb;">
            <div><b>📡 伺服器同步狀態：</b><span style="color: {sync_border}; font-weight: 700;">{sync_badge}</span></div>
            <div>🕒 <b>伺服器最後更新時間：</b><span style="color: #40c4ff; font-weight: 700;">{sync_info['pull_time_str']} (台灣時間)</span> &nbsp;|&nbsp; 📊 <b>最新 K 棒時間：</b><span style="color: #f0b90b; font-weight: 700;">{bar_display_label}</span> &nbsp;|&nbsp; 🔗 <b>來源：</b>{sync_info['source']}</div>
        </div>
        """, unsafe_allow_html=True)

        # 顏色主題設定
        if advice.action == "BUY_LONG":
            box_color = "#089981"
            bg_color = "rgba(8, 153, 129, 0.14)"
        elif advice.action == "SELL_SHORT":
            box_color = "#f23645"
            bg_color = "rgba(242, 54, 69, 0.14)"
        else:
            box_color = "#f0b90b"
            bg_color = "rgba(240, 185, 11, 0.14)"

        # AI 決策大橫幅卡片
        st.markdown(f"""
        <div style="background-color: {bg_color}; border: 2px solid {box_color}; border-radius: 12px; padding: 18px 24px; margin-bottom: 18px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; margin-bottom: 6px;">
                <span style="font-size: 15px; font-weight: 600; color: #b2b5be;">📌 分析標的：{ts_symbol} &nbsp;|&nbsp; 週期：{chart_tf} &nbsp;|&nbsp; 市場狀態：{advice.regime_label}</span>
                <span style="font-size: 13px; color: #848e9c;">分析引擎：{advice.source} (更新於 {advice.generated_at})</span>
            </div>
            <div style="font-size: 28px; font-weight: 800; color: {box_color}; margin: 8px 0 12px 0;">
                {advice.action_label}
            </div>
            <div style="font-size: 15.5px; color: #e0e3eb; line-height: 1.65;">
                💬 <b>AI 盤勢解讀與建議理由：</b>{advice.rationale}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 5 張關鍵作戰指標卡片
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("最新成交價", f"{curr_price:,.1f} {unit}", f"單量 {curr_vol} | {curr_time_str}")
        m2.metric("多空勝率評估", f"多 {advice.long_probability:.0f}% / 空 {advice.short_probability:.0f}%", delta=f"{advice.long_probability - advice.short_probability:+.0f}%")
        m3.metric("建議參考進場區間", advice.entry_zone)
        m4.metric("關鍵防守停損線", advice.stop_loss)
        m5.metric("短線目標 / 壓力點", advice.target_price)

        # 🌐 Google 雲端四維度綜合情報雷達（精簡一列展示 + 展開詳情）
        win_rate_display = (
            f"準確率 {intel_snap.ai_win_rate:.0f}% ({intel_snap.ai_verified_count}次驗證)"
            if intel_snap.ai_win_rate is not None and intel_snap.ai_verified_count > 0
            else intel_snap.ai_self_review_note
        )
        top_news_preview = intel_snap.news_headlines[0] if intel_snap.news_headlines else "暫無重大突發新聞"

        st.markdown(f"""
        <div style="background-color: rgba(30, 34, 45, 0.75); border: 1px solid #2b3139; border-radius: 10px; padding: 12px 18px; margin: 12px 0 6px 0; font-size: 13.5px; color: #d1d4dc;">
            <div style="font-weight: 700; color: #f0b90b; margin-bottom: 6px;">
                🌐 Google 雲端四維度綜合情報雷達（已納入 AI 下一階段趨勢預估）
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px;">
                <div>🌍 <b>國際連動：</b>{intel_snap.global_bias_label}<br/><span style="color:#848e9c; font-size:12.5px;">{intel_snap.global_summary_text}</span></div>
                <div>📰 <b>即時新聞：</b>{intel_snap.news_sentiment_label}<br/><span style="color:#848e9c; font-size:12.5px;">{top_news_preview[:42]}</span></div>
                <div>☁️ <b>雲端籌碼庫：</b>{"🟢 已連線" if intel_snap.gcs_connected else "⚪ 本地模式"}<br/><span style="color:#848e9c; font-size:12.5px;">{intel_snap.institutional_note}</span></div>
                <div>🎯 <b>AI 歷史勝率：</b><span style="color:#40c4ff;">{win_rate_display}</span><br/><span style="color:#848e9c; font-size:12.5px;">{intel_snap.ai_self_review_note}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📰 展開檢視：Google 雲端收集之最新財經新聞與國際報價明細"):
            ic1, ic2 = st.columns([1.1, 1.3])
            with ic1:
                st.markdown("**🌍 國際跨市場即時指標**")
                if intel_snap.global_quotes:
                    df_gq = pd.DataFrame([
                        {"指標名稱": q.name, "代號": q.symbol, "最新報價": f"{q.price:,.2f}", "漲跌幅": f"{q.status_icon} {q.change_pct:+.2f}%"}
                        for q in intel_snap.global_quotes
                    ])
                    st.dataframe(df_gq, width="stretch", hide_index=True)
                else:
                    st.caption("目前暫無國際盤即時數據")
            with ic2:
                st.markdown(f"**📰 `{ts_symbol}` 相關最新即時新聞標題**")
                if intel_snap.news_headlines:
                    for idx, title in enumerate(intel_snap.news_headlines, 1):
                        st.markdown(f"{idx}. {title}")
                else:
                    st.caption("目前無相關新聞標題")

        st.markdown("---")

        # ==================== 3. 即時 K 線與 AI 預估走勢圖表 ====================
        k_top1, k_top2, k_top3 = st.columns([2.3, 1.0, 0.9])
        with k_top1:
            st.subheader(f"📈 {ts_symbol} — {chart_tf} 即時 K 線與 AI 預估走勢圖")
        with k_top2:
            st.caption(
                f"📡 伺服器最後拉取：`{sync_info['pull_time_str']}`\n\n"
                f"📊 最新 K 棒：`{bar_display_label}`"
            )
        with k_top3:
            vis_bars_opt = st.selectbox(
                "畫面顯示 K 棒數量",
                ["60 根", "120 根 (推薦)", "250 根", "500 根", "全部顯示"],
                index=1,
                key="k_vis_bars_select",
            )

        if df_bars.height > 0:
            df_plot = df_bars.to_pandas()
            df_plot["time_label"] = df_plot["timestamp"].dt.strftime("%m-%d %H:%M")
            df_plot["ma5"] = df_plot["close"].rolling(5).mean()
            df_plot["ma20"] = df_plot["close"].rolling(20).mean()

            # 智慧傳輸瘦身：先算完完整均線後，若非選「全部顯示」，僅傳送最近 max(240, vis_count*2) 根至前端，減少 92% 網路傳輸量！
            if vis_bars_opt != "全部顯示":
                req_vis = int(vis_bars_opt.split(" ")[0])
                max_send = max(240, req_vis * 2)
                if len(df_plot) > max_send:
                    df_plot = df_plot.iloc[-max_send:].reset_index(drop=True)

            total_k = len(df_plot)
            vis_count = total_k if vis_bars_opt == "全部顯示" else min(total_k, int(vis_bars_opt.split(" ")[0]))
            init_start = max(0, total_k - vis_count)
            init_end = max(0, total_k - 1)

            last_close = float(df_plot["close"].iloc[-1])
            last_label = str(df_plot["time_label"].iloc[-1])

            # 提取 AI 預估關鍵價位 (進場、停損、目標)
            ai_levels = advice.extract_price_levels(fallback_price=last_close)
            entry_val = ai_levels["entry"]
            stop_val = ai_levels["stop_loss"]
            target_val = ai_levels["target"]

            # 產生未來 5 根 K 棒的 AI 預估走勢座標
            future_labels = [f"AI預估+{i}" for i in range(1, 6)]
            proj_x = [last_label] + future_labels

            if advice.action == "BUY_LONG":
                # 偏多預估路徑：先測試進場支撐後穩步推升至 AI 目標價
                step1 = min(last_close, entry_val) if entry_val > 0 else last_close
                proj_y = [
                    last_close,
                    round(last_close * 0.4 + step1 * 0.6, 1),
                    round(last_close + (target_val - last_close) * 0.30, 1),
                    round(last_close + (target_val - last_close) * 0.55, 1),
                    round(last_close + (target_val - last_close) * 0.80, 1),
                    round(target_val, 1),
                ]
                proj_color = "#00e676"
                proj_name = "🤖 AI 預估走勢 (偏多)"
            elif advice.action == "SELL_SHORT":
                # 偏空預估路徑：先小幅反彈測試進場壓力後下探至 AI 目標價
                step1 = max(last_close, entry_val) if entry_val > 0 else last_close
                proj_y = [
                    last_close,
                    round(last_close * 0.4 + step1 * 0.6, 1),
                    round(last_close + (target_val - last_close) * 0.30, 1),
                    round(last_close + (target_val - last_close) * 0.55, 1),
                    round(last_close + (target_val - last_close) * 0.80, 1),
                    round(target_val, 1),
                ]
                proj_color = "#ff5252"
                proj_name = "🤖 AI 預估走勢 (偏空)"
            else:
                # 觀望預估路徑：於區間內橫向震盪整理
                amp = max(abs(target_val - last_close) * 0.25, last_close * 0.0015)
                proj_y = [
                    last_close,
                    round(last_close + amp * 0.6, 1),
                    round(last_close - amp * 0.5, 1),
                    round(last_close + amp * 0.4, 1),
                    round(last_close - amp * 0.3, 1),
                    round(last_close, 1),
                ]
                proj_color = "#ffd54f"
                proj_name = "🤖 AI 預估走勢 (區間震盪)"

            # 計算 Y 軸顯示範圍，確保 K 線與 AI 預估目標/停損線完整呈現
            vis_slice = df_plot.iloc[init_start:]
            y_candidates = [
                float(vis_slice["low"].min()),
                float(vis_slice["high"].max()),
            ]
            if target_val > 0:
                y_candidates.append(target_val)
            if stop_val > 0:
                y_candidates.append(stop_val)
            y_min = min(y_candidates)
            y_max = max(y_candidates)
            y_pad = max(1.0, (y_max - y_min) * 0.10)

            # 水平參考線橫跨最近 35 根 K 線至未來預估區間，畫面清晰不雜亂
            ref_start_idx = max(init_start, total_k - 35)
            ref_x = list(df_plot["time_label"].iloc[ref_start_idx:]) + future_labels

            fig_k = go.Figure()
            fig_k.add_trace(go.Candlestick(
                x=df_plot["time_label"],
                open=df_plot["open"],
                high=df_plot["high"],
                low=df_plot["low"],
                close=df_plot["close"],
                increasing_line_color="#ef5350",
                decreasing_line_color="#26a69a",
                name="K線",
            ))
            fig_k.add_trace(go.Scatter(
                x=df_plot["time_label"],
                y=df_plot["ma5"],
                line=dict(color="#ffa726", width=1.6),
                name="5MA",
            ))
            fig_k.add_trace(go.Scatter(
                x=df_plot["time_label"],
                y=df_plot["ma20"],
                line=dict(color="#29b6f6", width=1.6),
                name="20MA",
            ))

            # 加入 AI 預估未來走勢軌跡線
            fig_k.add_trace(go.Scatter(
                x=proj_x,
                y=proj_y,
                mode="lines+markers",
                line=dict(color=proj_color, width=2.8, dash="dash"),
                marker=dict(size=6, color=proj_color, symbol="diamond"),
                name=proj_name,
            ))

            # 加入 AI 目標價、建議進場價、防守停損線
            if target_val > 0:
                fig_k.add_trace(go.Scatter(
                    x=ref_x,
                    y=[target_val] * len(ref_x),
                    mode="lines",
                    line=dict(color="#00e676", width=1.4, dash="dot"),
                    name=f"🏆 AI 目標 ({target_val:,.1f})",
                ))
            if advice.action != "WAIT" and entry_val > 0:
                fig_k.add_trace(go.Scatter(
                    x=ref_x,
                    y=[entry_val] * len(ref_x),
                    mode="lines",
                    line=dict(color="#40c4ff", width=1.3, dash="dashdot"),
                    name=f"🎯 AI 進場 ({entry_val:,.1f})",
                ))
            if stop_val > 0:
                fig_k.add_trace(go.Scatter(
                    x=ref_x,
                    y=[stop_val] * len(ref_x),
                    mode="lines",
                    line=dict(color="#ff5252", width=1.4, dash="dot"),
                    name=f"🛑 AI 停損 ({stop_val:,.1f})",
                ))

            # 在預估路徑終點加上直覺標籤
            fig_k.add_annotation(
                x=future_labels[-1],
                y=proj_y[-1],
                text=f"🤖 AI預估 {proj_y[-1]:,.1f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1.5,
                arrowcolor=proj_color,
                ax=0,
                ay=-28 if advice.action != "SELL_SHORT" else 28,
                font=dict(color="#ffffff", size=12),
                bgcolor="rgba(20, 24, 35, 0.85)",
                bordercolor=proj_color,
                borderwidth=1,
                borderpad=4,
            )

            # 由最後一根最新 K 棒 (total_k - 1) 往回取固定間距刻度，確保最新 K 棒時間 100% 顯示在 X 軸上絕不跳格！
            tick_step = max(1, vis_count // 18)
            tick_indices = list(range(total_k - 1, -1, -tick_step))[::-1]
            all_time_labels = df_plot["time_label"].tolist()
            explicit_tickvals = [all_time_labels[idx] for idx in tick_indices] + [future_labels[1], future_labels[4]]

            fig_k.update_layout(
                template="plotly_dark",
                height=500,
                margin=dict(l=15, r=25, t=55, b=20),
                xaxis_rangeslider_visible=False,
                xaxis=dict(
                    type="category",
                    tickmode="array",
                    tickvals=explicit_tickvals,
                    ticktext=explicit_tickvals,
                    range=[init_start - 0.5, total_k + len(future_labels) + 0.5],
                    minallowed=-0.5,
                    maxallowed=total_k + len(future_labels) + 1.5,
                ),
                yaxis=dict(
                    range=[y_min - y_pad, y_max + y_pad],
                    fixedrange=False,
                ),
                # 圖例改為靠左對齊 (x=0)，完全避開右上角的 Plotly 相機/縮放工具列
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0,
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(size=12),
                ),
                dragmode="pan",
                uirevision=f"unified_k_{clean_sym}_{chart_tf}_{vis_bars_opt}_{last_label}",
            )
            st.plotly_chart(
                fig_k,
                width="stretch",
                key=f"plotly_unified_{clean_sym}_{chart_tf}_{vis_bars_opt}",
                config={"scrollZoom": True, "displayModeBar": True},
            )
        else:
            st.info(f"💡 目前尚無 `{ts_symbol}` 的 K 線數據。請點擊右上方「🚀 連線 / 重新 AI 分析」按鈕載入台新真實 K 線。")

        # 最新逐筆成交明細（收合或精簡展示）
        recent_ticks = feed.get_recent_ticks()
        if recent_ticks:
            with st.expander(f"⚡ 檢視最新逐筆成交明細 (最近 {min(15, len(recent_ticks))} 筆)", expanded=False):
                tick_rows = [
                    {
                        "成交時間": tk.timestamp.strftime("%H:%M:%S"),
                        "成交價": f"{tk.price:,.1f}",
                        "單筆數量": tk.volume,
                    }
                    for tk in reversed(recent_ticks[-15:])
                ]
                st.dataframe(pd.DataFrame(tick_rows), width="stretch", height=180)

    _render_unified_war_room()

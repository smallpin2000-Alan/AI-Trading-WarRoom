# src/ui/dashboard.py
from __future__ import annotations
import os
import gc
import json
import logging
import time
import threading
from pathlib import Path
from typing import Optional
import numpy as np
import polars as pl
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.ai.advisor import FuturesAIAdvisor, FuturesAdvice
from src.ai.cloud_intel import CloudIntelligenceHub, CloudIntelSnapshot
from src.broker.taishin_quote import TaishinQuoteFeed
from src.data.symbol_lookup import get_symbol_name, format_symbol_display
from dataclasses import replace
from src.core.paper_trader import (
    PaperTradingEngine,
    RiskConfig,
    OpenPosition,
    default_ai_strategy,
    ensure_atr_exits,
    DEFAULT_ATR_STOP_MULT,
    DEFAULT_ATR_TP_MULT,
    DEFAULT_ATR_TRAIL_MULT,
    DEFAULT_ATR_TRAIL_ACTIVATION,
    DEFAULT_ATR_LENGTH,
)
from src.core.indicators import IndicatorEngine
from src.core.engine import ContractSpec, EventDrivenEngine

# AI 自動模擬操盤的預設單日最大虧損（元）
DEFAULT_MAX_DAILY_LOSS = 5000.0
from src.schema.strategy import StrategySchema
from src.core.credibility_gate import run_orb_credibility_gate
from src.data.taifex_history import DailyHistoryUpdater, history_path
from src.core.orb_strategy import OrbParams, compute_orb_state
from src.core.key_levels import compute_key_levels, expected_range_hit_rate
from datetime import datetime, timedelta, timezone, time as dt_time

# 模擬倉使用的開盤區間突破參數（預設 30 分鐘區間）
ORB_PARAMS = OrbParams()
from src.ui.admin_gate import (
    FailedAttemptThrottle,
    get_admin_password,
    is_protection_enabled,
    verify_admin_password,
)

logger = logging.getLogger(__name__)


def load_taishin_default_config() -> dict:
    cfg = {
        "symbol": os.getenv("SOL_SYMBOL", "TMFR1"),
        "sol_user_id": os.getenv("SOL_USER_ID", ""),
        "sol_password": os.getenv("SOL_PASSWORD", ""),
        "gemini_api_key": os.getenv("GEMINI_API_KEY", ""),
        "gemini_model": os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
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


@st.cache_resource
def _get_global_paper_trader_holder() -> dict:
    """跨瀏覽器分頁共用的紙上交易模擬引擎單例容器，避免不同分頁自動刷新時互相覆蓋重置狀態"""
    return {"instance": None}


def get_or_create_paper_trader(symbol: str) -> PaperTradingEngine:
    """初始化並獲取全域紙上交易模擬引擎實例（自動從磁碟恢復狀態，跨分頁同步）"""
    holder = _get_global_paper_trader_holder()
    pt = holder.get("instance")
    if pt is None or getattr(pt, "VERSION", None) != PaperTradingEngine.VERSION:
        # 預設採用微台 TMF 規格與預設順勢停損策略
        sym_key = symbol.split(" ")[0].upper()
        clean_key = "TMF" if "TMF" in sym_key else "MXF" if "MXF" in sym_key else "TXF" if "TXF" in sym_key else "TMF"
        c_spec = ContractSpec.from_json(clean_key)
        default_strat = default_ai_strategy(clean_key)
        pt = PaperTradingEngine(
            strategy=default_strat,
            contract=c_spec,
            initial_capital=100_000.0,
            # 單日虧損上限需容納數次 ATR 停損（1.5 倍 ATR 約 150~200 點 ≈ 1,500~2,000 元/口）
            risk_config=RiskConfig(max_daily_loss=DEFAULT_MAX_DAILY_LOSS, max_contracts=1),
        )
        pt.load_state_from_disk()
        # 舊版存檔的固定點數停損（如 45 點）遠小於正常波動，一律轉換為 ATR 倍數停損
        migrated, changed = ensure_atr_exits(pt.strategy)
        if changed:
            pt.strategy = migrated
            pt.core_engine = EventDrivenEngine(migrated, pt.contract, pt.initial_capital)
            pt.credibility_verified = False
            if pt.risk_config.max_daily_loss < DEFAULT_MAX_DAILY_LOSS:
                pt.risk_config = replace(pt.risk_config, max_daily_loss=DEFAULT_MAX_DAILY_LOSS)
            pt._log("SYSTEM", "出場設定已由固定點數改為 ATR 倍數停損（可信度檢核需重新執行）")
            pt.save_state_to_disk()
        holder["instance"] = pt
    st.session_state["paper_trader"] = pt
    return pt


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
    "TMFR1 (微台指近一)",
    "TXFR1 (大台指近一)",
    "MTXR1 (小台指近一)",
    "2330 (台積電)",
    "2317 (鴻海)",
    "2454 (聯發科)",
    "0050 (元大台灣50)",
]


def load_watchlist(path: str | Path = "config/watchlist.json") -> list[str]:
    p = Path(path)
    raw_list: list[str] = []
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list) and len(data) > 0:
                raw_list = [str(x).strip() for x in data if str(x).strip()]
        except Exception:
            pass
    if not raw_list:
        raw_list = list(DEFAULT_WATCHLIST)

    # 自動標準化為「代碼 (商品名稱)」格式，確保分析選單每個標的皆明確帶出中文名稱
    normalized: list[str] = []
    for item in raw_list:
        formatted = format_symbol_display(item)
        if formatted and formatted not in normalized:
            normalized.append(formatted)
    return normalized


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
    if not note:
        # 自動從內建字典或 TWSE 官方 API 帶出中文商品名稱
        detected = get_symbol_name(clean_code)
        if detected:
            note = detected

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


@st.cache_resource
def _get_admin_throttle() -> FailedAttemptThrottle:
    """全站共用的管理密碼錯誤次數節流器"""
    return FailedAttemptThrottle()


def render_admin_gate() -> bool:
    """
    顯示管理密碼解鎖區塊，回傳目前 session 是否可執行「寫入類」操作。
    未設定 WARROOM_ADMIN_PASSWORD 時不擋（維持舊行為），但顯示提醒。
    """
    if not is_protection_enabled():
        st.caption("⚠️ 尚未設定管理密碼（環境變數 WARROOM_ADMIN_PASSWORD），任何訪客都能操作模擬倉；公開部署時強烈建議設定。")
        return True
    if st.session_state.get("warroom_admin_ok"):
        if st.button("🔐 登出管理權限", key="btn_admin_logout"):
            st.session_state["warroom_admin_ok"] = False
            st.rerun()
        return True

    throttle = _get_admin_throttle()
    with st.expander("🔐 輸入管理密碼以操作模擬倉（目前為唯讀模式）"):
        if throttle.is_locked():
            st.error(f"密碼錯誤次數過多，請 {throttle.seconds_until_unlock()} 秒後再試。")
            return False
        pwd = st.text_input("管理密碼", type="password", key="in_admin_pwd")
        if st.button("解鎖", key="btn_admin_login"):
            if verify_admin_password(pwd, get_admin_password()):
                st.session_state["warroom_admin_ok"] = True
                st.rerun()
            else:
                throttle.record_failure()
                logger.warning("戰情室管理密碼驗證失敗")
                st.error("密碼錯誤")
    return False


def render_credibility_start(pt: PaperTradingEngine, bars: pl.DataFrame, unlock_first: bool, can_write: bool) -> None:
    """
    「啟動 / 解鎖並重新啟動」流程：
    先實際執行五道可信度檢核；通過即啟動。
    未通過時列出各項結果，使用者必須勾選「僅限模擬倉、已了解未通過」才可確認啟動（並寫入稽核日誌）。
    """
    label = "🔄 解鎖並重新啟動操盤" if unlock_first else "▶ 啟動 AI 自動操盤"
    if st.button(label, key="btn_start_ai_trade", type="primary", disabled=not can_write):
        with st.spinner("正在以歷史資料回測開盤區間突破規則..."):
            # 模擬倉進場由開盤區間突破規則決定：以期交所歷史 1 分 K 回測此規則作為檢核依據
            hist_path = history_path("TMF")
            hist = pl.read_parquet(hist_path) if hist_path.exists() else None
            gate = run_orb_credibility_gate(hist, ORB_PARAMS, strategy=pt.strategy)
        if gate.passed:
            if unlock_first:
                pt.reset_risk_locks(reset_daily_accounting=True)
            succ, msg = pt.start_session(credibility_passed=True)
            pt.save_state_to_disk()
            st.session_state.pop("cred_gate_result", None)
            (st.success if succ else st.error)(msg)
            st.rerun()
        st.session_state["cred_gate_result"] = gate

    gate = st.session_state.get("cred_gate_result")
    if gate is None:
        return
    st.warning("⚠️ 開盤區間突破規則未通過歷史資料可信度檢核：")
    for name, ok in gate.checks:
        st.caption(f"{'✅' if ok else '❌'} {name}")
    for reason in gate.reasons:
        st.caption(f"・{reason}")
    confirmed = st.checkbox("我了解此策略未通過檢核，僅限模擬倉啟動", key="chk_cred_override")
    c_ok, c_cancel = st.columns(2)
    with c_ok:
        if st.button("確認啟動", key="btn_cred_override", disabled=not (confirmed and can_write)):
            if unlock_first:
                pt.reset_risk_locks(reset_daily_accounting=True)
            succ, msg = pt.start_session(credibility_passed=False, simulation_override=True)
            pt.save_state_to_disk()
            st.session_state.pop("cred_gate_result", None)
            (st.warning if succ else st.error)(msg)
            st.rerun()
    with c_cancel:
        if st.button("取消", key="btn_cred_cancel"):
            st.session_state.pop("cred_gate_result", None)
            st.rerun()


APP_VERSION = "v2.9.1"
_refreshing_advice_keys: set[str] = set()


@st.cache_resource
def get_global_taishin_feed() -> TaishinQuoteFeed:
    """跨瀏覽器分頁與 Rerun 共用唯一台新連線實例，徹底防止 Too Many Connections"""
    return TaishinQuoteFeed(auto_load_store=True)


@st.cache_resource
def get_history_updater() -> Optional[DailyHistoryUpdater]:
    """
    啟動期交所歷史資料每日自動更新（全站只啟動一個背景執行緒）。
    Cloud Run 檔案系統為暫存且無人瀏覽時不執行，預設不啟用；設定 HISTORY_AUTO_UPDATE=1 可強制啟用。
    """
    on_cloud_run = bool(os.getenv("K_SERVICE"))
    if on_cloud_run and os.getenv("HISTORY_AUTO_UPDATE", "").strip() != "1":
        return None
    updater = DailyHistoryUpdater(product="TMF")
    updater.start()
    return updater


@st.cache_data(ttl=600, show_spinner=False)
def load_history_bars() -> Optional[pl.DataFrame]:
    """讀取期交所歷史微台 1 分 K（每 10 分鐘最多重讀一次；檔案不存在時回傳 None）"""
    path = history_path("TMF")
    if not path.exists():
        return None
    try:
        return pl.read_parquet(path)
    except (OSError, pl.exceptions.PolarsError):
        logger.exception("讀取歷史 K 棒失敗：%s", path)
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def expected_range_hit_rate_cached(symbol: str, open_t: dt_time, close_t: dt_time, _bars: pl.DataFrame) -> tuple[Optional[float], int]:
    """預期波動區間的歷史命中率（每商品每小時計算一次）"""
    return expected_range_hit_rate(_bars, open_t, close_t)


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
    st.caption(f"🏷️ **版本號：{APP_VERSION} (2026-11-12 規則操盤戰情版)** ｜ 🛡️ 串接台新官方 API 與期交所 0 延遲 K 線 • 開盤區間突破規則多空決策（AI 僅供參考） • 純報價零下單風險")
    history_updater = get_history_updater()
    if history_updater is not None:
        st.caption(f"📚 期交所歷史資料自動更新：{history_updater.last_result}")

    if is_idle_sleeping:
        st.warning("⏰【智慧防呆休眠中】您已連續看盤超過 30 分鐘。為保護雲端免費額度（確保 0 元），自動刷新已暫停。")
        if st.button("🔄 繼續看盤（重設 30 分鐘計時）", type="primary", key="btn_wake_idle"):
            st.session_state["last_active_time"] = time.time()
            st.rerun()

    # ==================== 1. 頂部統一控制列（全站唯一選商品處，開啟網頁預設直接鎖定微台 TMFR1） ====================
    watchlist_items = load_watchlist()
    default_idx = next(
        (i for i, item in enumerate(watchlist_items) if item.startswith("TMFR1")),
        0,
    )
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
            ["15 秒 (推薦極速風控)", "30 秒", "1 分鐘", "3 分鐘", "5 分鐘", "關閉自動刷新"],
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
        # 會觸發台新登入／重連，屬於需管理權限的操作（避免訪客反覆按造成登入風暴）
        manual_trigger = st.button(
            "🚀 連線 / 重新 AI 分析",
            type="primary",
            width="stretch",
            disabled=not st.session_state.get("warroom_can_write", not is_protection_enabled()),
        )

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
        advice_cache_key = f"ai_advice_{APP_VERSION}_{clean_sym}_{chart_tf}"
        st.session_state.pop(advice_cache_key, None)
        global_advice_cache.pop(advice_cache_key, None)
        st.session_state.pop(f"ai_advice_{clean_sym}_{chart_tf}", None)
        global_advice_cache.pop(f"ai_advice_{clean_sym}_{chart_tf}", None)
        CloudIntelligenceHub._cache.pop(f"intel_{clean_sym}", None)
        st.rerun()

    # 自選商品增刪管理（預設收合，保持畫面乾淨）
    with st.expander("➕ / 🗑️ 管理自選商品清單（新增或移除股票 / 期貨代碼）"):
        wl_c1, wl_c2, wl_c3, wl_c4 = st.columns([1.2, 1.5, 1.0, 1.1])
        with wl_c1:
            new_sym_code = st.text_input("新增商品代碼", placeholder="例如: 2603 或 TMFR1", key="in_new_sym_code")
            detected_name = get_symbol_name(new_sym_code) if new_sym_code else ""
            if detected_name:
                st.caption(f"✨ 系統已識別：**{detected_name}**")
        with wl_c2:
            clean_in_code = new_sym_code.strip().upper() if new_sym_code else ""
            new_sym_note = st.text_input(
                "商品名稱 (已自動帶出)",
                value=detected_name,
                placeholder=detected_name if detected_name else "例如: 長榮",
                key=f"in_new_sym_note_{clean_in_code}",
            )
        with wl_c3:
            st.write("")
            if st.button("➕ 新增商品", width="stretch", type="primary", disabled=not st.session_state.get("warroom_can_write", not is_protection_enabled())):
                effective_note = new_sym_note.strip() if new_sym_note.strip() else detected_name
                ok_add, msg_add, updated_list = add_symbol_to_watchlist(new_sym_code, effective_note)
                if ok_add:
                    st.session_state["selected_watchlist_item"] = updated_list[0]
                    st.toast(msg_add)
                    st.rerun()
                else:
                    st.warning(msg_add)
        with wl_c4:
            st.write("")
            if st.button(f"🗑️ 移除目前商品 ({clean_sym})", width="stretch", disabled=not st.session_state.get("warroom_can_write", not is_protection_enabled())):
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
        is_mkt_open, mkt_status_label = FuturesAIAdvisor.get_market_session_status(clean_sym)
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
        advice_key = f"ai_advice_{APP_VERSION}_{clean_sym}_{chart_tf}"
        now_ts = time.time()
        bar_sig = f"{df_bars.height}_{last_bar_time_str}"
        cached_entry = global_advice_cache.get(advice_key)
        # 節流防爆頻率：在開盤時段至少快取 60 秒（1 分鐘），確保盤中靈敏反應且仍在 Google 每日免費額度內
        ai_cache_ttl = float(default_cfg.get("predict_frequency", 60))
        if (
            cached_entry is not None
            and (now_ts - cached_entry[0] < ai_cache_ttl)
            and cached_entry[1] == bar_sig
        ):
            advice: FuturesAdvice = cached_entry[2]
        else:
            if cached_entry is not None and cached_entry[1] == bar_sig:
                advice = cached_entry[2]
            else:
                # 首次載入或 K 棒已推進：0.005 秒極速重算台新三積木（均線+布林+MACD+ATR）多空決策，絕不卡畫面！
                try:
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
                except Exception as ex:
                    # AI 分析失敗不可拖垮整個戰情室（後面的模擬倉停損檢核仍必須執行）
                    logger.exception("AI 決策分析失敗 (%s %s)", clean_sym, chart_tf)
                    st.error(f"⚠️ AI 決策分析暫時失敗：{ex}（模擬倉停損守護不受影響）")
                    advice = (
                        cached_entry[2]
                        if cached_entry is not None
                        # df_bars=None 會走「資料不足、建議觀望」的安全分支，不會再拋例外
                        else FuturesAIAdvisor.analyze_market(None, symbol=clean_sym, timeframe=chart_tf)
                    )

            # 背景執行緒非阻塞呼叫 Gemini 2.5 Flash 深度分析 (僅在開盤交易時段執行，收盤後不再重複呼叫預估)
            if is_mkt_open and advice_key not in _refreshing_advice_keys and df_bars.height >= 10:
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
        <div style="background-color: {sync_bg}; border: 1px solid {sync_border}; border-radius: 8px; padding: 8px 16px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; font-size: 13.5px; color: #e0e3eb;">
            <div><b>📡 伺服器同步狀態：</b><span style="color: {sync_border}; font-weight: 700;">{sync_badge}</span> &nbsp;|&nbsp; <b>{mkt_status_label}</b></div>
            <div>🕒 <b>伺服器最後更新時間：</b><span style="color: #40c4ff; font-weight: 700;">{sync_info['pull_time_str']} (台灣時間)</span> &nbsp;|&nbsp; 📊 <b>最新 K 棒時間：</b><span style="color: #f0b90b; font-weight: 700;">{bar_display_label}</span> &nbsp;|&nbsp; 🔗 <b>來源：</b>{sync_info['source']}</div>
        </div>
        """, unsafe_allow_html=True)

        # ⚡ 頂部完整當日盤況與 Tick 流速遙測橫幅列（對齊專業期貨交易介面規格，禁止 4 空白縮排）
        mkt_stats = feed.get_market_stats_summary()
        p_chg = mkt_stats["change_val"]
        p_pct = mkt_stats["change_pct"]
        chg_color = "#ef5350" if p_chg > 0 else ("#26a69a" if p_chg < 0 else "#cfd8dc")
        chg_sign = "+" if p_chg > 0 else ""
        chg_arrow = "▲" if p_chg > 0 else ("▼" if p_chg < 0 else "—")
        basis_v = mkt_stats.get("basis_val", 0.0)
        basis_sign = "▲+" if basis_v > 0 else ("▼" if basis_v < 0 else "")
        basis_col = "#ef5350" if basis_v > 0 else ("#26a69a" if basis_v < 0 else "#cfd8dc")

        top_stats_html = (
            '<div style="background: linear-gradient(90deg, #131722 0%, #1a233a 100%); border: 1px solid #2a2e39; border-radius: 8px; padding: 10px 16px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; color: #d1d4dc;">'
            '<div style="display: flex; align-items: baseline; gap: 10px;">'
            f'<span style="font-size: 16px; font-weight: 700; color: #ffffff;">{ts_symbol}</span>'
            f'<span style="font-size: 26px; font-weight: 800; color: {chg_color}; font-family: monospace;">{mkt_stats["current_price"]:,.1f}</span>'
            f'<span style="font-size: 14px; font-weight: 700; color: {chg_color};">{chg_arrow} {chg_sign}{p_chg:,.1f} ({chg_sign}{p_pct:.2f}%)</span>'
            '</div>'
            '<div style="display: flex; gap: 15px; font-size: 13px; font-family: monospace; flex-wrap: wrap;">'
            f'<div><span style="color: #787b86;">開</span> <b style="color: #e0e3eb;">{mkt_stats["open_p"]:,.1f}</b></div>'
            f'<div><span style="color: #787b86;">高</span> <b style="color: #ef5350;">{mkt_stats["high_p"]:,.1f}</b></div>'
            f'<div><span style="color: #787b86;">低</span> <b style="color: #26a69a;">{mkt_stats["low_p"]:,.1f}</b></div>'
            f'<div><span style="color: #787b86;">昨收</span> <b style="color: #cfd8dc;">{mkt_stats["prev_close"]:,.1f}</b></div>'
            f'<div><span style="color: #787b86;">總量</span> <b style="color: #f0b90b;">{mkt_stats["total_vol"]:,}</b></div>'
            f'<div><span style="color: #ef5350;">漲停</span> <b style="color: #ef5350;">{mkt_stats["limit_up"]:,.1f}</b></div>'
            f'<div><span style="color: #26a69a;">跌停</span> <b style="color: #26a69a;">{mkt_stats["limit_down"]:,.1f}</b></div>'
            f'<div><span style="color: #787b86;">價差(20MA)</span> <b style="color: {basis_col};">{basis_sign}{basis_v:,.1f}</b></div>'
            '</div>'
            '<div style="display: flex; gap: 10px; align-items: center; font-size: 12.5px;">'
            f'<div style="background: rgba(0,230,118,0.12); border: 1px solid #00e676; border-radius: 4px; padding: 2px 8px; color: #00e676;">⚡ 流速 <b>{mkt_stats["ticks_per_sec"]}</b> tick/秒</div>'
            f'<div style="background: rgba(41,182,246,0.12); border: 1px solid #29b6f6; border-radius: 4px; padding: 2px 8px; color: #29b6f6;">📶 外盤 <b style="color:#ff5252;">{mkt_stats["outer_ratio"]}%</b> / 內盤 <b style="color:#00e676;">{mkt_stats["inner_ratio"]}%</b></div>'
            '</div>'
            '</div>'
        )
        st.markdown(top_stats_html, unsafe_allow_html=True)

        # 顏色主題與收盤休市狀態設定
        if not is_mkt_open:
            box_color = "#64b5f6"
            bg_color = "rgba(100, 181, 246, 0.10)"
            banner_title = f"🌙 {mkt_status_label} — 已停止交易（暫停盤中 AI 分析）"
            banner_rationale = (
                f"目前 `{ts_symbol}` 已收盤停止交易，最後收盤價為 <b>{curr_price:,.1f} {unit}</b>（最後 K 棒時間：{bar_display_label}）。"
                f"收盤休市期間暫停盤中 AI 分析（AI 僅供參考，模擬倉依開盤區間突破規則下單）。"
            )
        else:
            if advice.action == "BUY_LONG":
                box_color = "#089981"
                bg_color = "rgba(8, 153, 129, 0.14)"
            elif advice.action == "SELL_SHORT":
                box_color = "#f23645"
                bg_color = "rgba(242, 54, 69, 0.14)"
            else:
                box_color = "#f0b90b"
                bg_color = "rgba(240, 185, 11, 0.14)"
            banner_title = f"{advice.action_label}（AI 分析僅供參考，不影響模擬倉下單）"
            banner_rationale = advice.rationale

        # 🤖 AI 分析收進預設收合的折疊區：僅供參考，不影響模擬倉下單（多空方向由下方規則訊號決定）
        ai_box = st.expander(f"🤖 AI 盤勢分析（僅供參考，不影響下單）｜目前 AI 看法：{advice.action_label}", expanded=False)
        # AI 決策大橫幅卡片（整合台新 Nova API 量化三要素：【指標】➔【訊號】➔【方法】＋ 近期回測獲利因子）
        # 注意：禁止任何開頭 4 空白縮排，避免被 Streamlit 誤判為 <pre><code> 程式碼區塊
        taishin_tri_block = ""
        if is_mkt_open and getattr(advice, "indicator_summary", ""):
            ens_block = ""
            if getattr(advice, "ensemble_votes_summary", ""):
                ens_block = f'<div style="color: #69f0ae; margin-top: 2px;">⚡ <b>【量化矩陣大腦】：</b>{advice.ensemble_votes_summary} ｜ 最適戰法：<b>{advice.ensemble_best_model}</b></div>'

            taishin_tri_block = (
                '<div style="margin-top: 10px; padding-top: 10px; border-top: 1px dashed rgba(255,255,255,0.16); font-size: 13.5px; color: #cfd8dc; line-height: 1.65;">'
                f'{ens_block}'
                f'<div>📐 <b>【指標 (Indicator)】：</b>{advice.indicator_summary}</div>'
                f'<div>🚦 <b>【訊號 (Signal)】：</b>{advice.signal_summary}</div>'
                f'<div>🎯 <b>【方法 (Method)】：</b>{advice.method_summary}</div>'
                f'<div style="color: #40c4ff; margin-top: 2px;">📊 <b>【近期 K 棒回測驗證】：</b>{advice.backtest_summary}</div>'
                '</div>'
            )

        ai_box.markdown(
            f'<div style="background-color: {bg_color}; border: 2px solid {box_color}; border-radius: 12px; padding: 18px 24px; margin-bottom: 18px;">'
            '<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; margin-bottom: 6px;">'
            f'<span style="font-size: 15px; font-weight: 600; color: #b2b5be;">📌 分析標的：{ts_symbol} &nbsp;|&nbsp; 週期：{chart_tf} &nbsp;|&nbsp; 市場狀態：{advice.regime_label if is_mkt_open else "🌙 已收盤休市"}</span>'
            f'<span style="font-size: 13px; color: #848e9c;">分析引擎：{advice.source if is_mkt_open else "收盤休市模式"} (更新於 {advice.generated_at})</span>'
            '</div>'
            f'<div style="font-size: 18px; font-weight: 700; color: {box_color}; margin: 4px 0 8px 0;">{banner_title}</div>'
            f'<div style="font-size: 15.5px; color: #e0e3eb; line-height: 1.65;">💬 <b>{"AI 盤勢解讀與建議理由" if is_mkt_open else "收盤狀態說明"}：</b>{banner_rationale}</div>'
            f'{taishin_tri_block}'
            '</div>',
            unsafe_allow_html=True,
        )

        # 5 張關鍵作戰指標卡片
        m1, m2, m3, m4, m5 = ai_box.columns(5)
        if is_mkt_open:
            m1.metric("最新成交價", f"{curr_price:,.1f} {unit}", f"單量 {curr_vol} | {curr_time_str}")
            m2.metric("多空勝率評估", f"多 {advice.long_probability:.0f}% / 空 {advice.short_probability:.0f}%", delta=f"{advice.long_probability - advice.short_probability:+.0f}%")
            m3.metric("建議參考進場區間", advice.entry_zone)
            m4.metric("關鍵防守停損線", advice.stop_loss)
            m5.metric("布林 / 短線目標點", advice.target_price)
        else:
            m1.metric("最新收盤價", f"{curr_price:,.1f} {unit}", f"收盤時間 {bar_display_label[-5:]}")
            m2.metric("收盤多空結構", f"多 {advice.long_probability:.0f}% / 空 {advice.short_probability:.0f}%", delta="已收盤")
            m3.metric("建議參考進場區間", "🌙 已收盤暫停")
            m4.metric("關鍵防守停損線", "🌙 已收盤暫停")
            m5.metric("布林 / 短線目標點", "🌙 已收盤暫停")

        # 🌐 Google 雲端四維度綜合情報雷達（精簡一列展示 + 展開詳情）
        # 🎯 整合台新 Nova API 回測四大指標（勝率、賺賠比、獲利因子 PF）與實盤穿價追蹤，杜絕虛假 100% 勝率！
        bt_wr = getattr(advice, "backtest_win_rate", 0.0)
        bt_odds = getattr(advice, "backtest_odds", 0.0)
        bt_pf = getattr(advice, "backtest_pf", 0.0)

        if bt_wr > 0 and bt_pf > 0:
            radar_strategy_display = f"回測勝率 {bt_wr:.1f}% ｜ PF {bt_pf:.2f} ｜ 賺賠比 {bt_odds:.2f}"
        elif getattr(advice, "backtest_summary", ""):
            radar_strategy_display = advice.backtest_summary.split("｜", 1)[-1].strip() if "｜" in advice.backtest_summary else advice.backtest_summary
        else:
            radar_strategy_display = "回測指標計算中"

        if intel_snap.ai_win_rate is not None and intel_snap.ai_verified_count >= 5:
            radar_live_subnote = f"實盤穿價準確率 {intel_snap.ai_win_rate:.0f}% ({intel_snap.ai_verified_count}次驗證)"
        else:
            radar_live_subnote = f"實盤追蹤：{intel_snap.ai_self_review_note}"

        top_news_preview = intel_snap.news_headlines[0] if intel_snap.news_headlines else "暫無重大突發新聞"

        ai_box.markdown(f"""
        <div style="background-color: rgba(30, 34, 45, 0.75); border: 1px solid #2b3139; border-radius: 10px; padding: 12px 18px; margin: 12px 0 6px 0; font-size: 13.5px; color: #d1d4dc;">
            <div style="font-weight: 700; color: #f0b90b; margin-bottom: 6px;">
                🌐 Google 雲端四維度綜合情報雷達（已納入 AI 下一階段趨勢預估）
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px;">
                <div>🌍 <b>國際連動：</b>{intel_snap.global_bias_label}<br/><span style="color:#848e9c; font-size:12.5px;">{intel_snap.global_summary_text}</span></div>
                <div>📰 <b>即時新聞：</b>{intel_snap.news_sentiment_label}<br/><span style="color:#848e9c; font-size:12.5px;">{top_news_preview[:42]}</span></div>
                <div>☁️ <b>雲端籌碼庫：</b>{"🟢 已連線" if intel_snap.gcs_connected else "⚪ 本地模式"}<br/><span style="color:#848e9c; font-size:12.5px;">{intel_snap.institutional_note}</span></div>
                <div>🎯 <b>策略勝率與實盤：</b><br/><span style="color:#40c4ff; font-weight:600;">{radar_strategy_display}</span><br/><span style="color:#848e9c; font-size:12px;">{radar_live_subnote}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 展開多策略矩陣戰法明細
        if getattr(advice, "ensemble_models_detail", None):
            with st.expander("⚡ 展開檢視：【量化多策略矩陣】6大戰法即時評分、動態權重與回測勝率"):
                st.caption("💡 矩陣大腦每根 K 棒並行回測 6 大經典戰法，自動分配高勝率戰法更大權重，過半數共振才發出明確多空訊號：")
                df_ens = pd.DataFrame(advice.ensemble_models_detail)
                df_ens.columns = ["策略戰法名稱", "即時訊號方向", "動態權重占比 (%)", "獲利因子 (PF)", "近期回測勝率 (%)"]
                st.dataframe(df_ens, width="stretch", hide_index=True)

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

        # ==================== 2.5 🤖 AI 自動模擬操盤帳戶面板 ====================
        pt = get_or_create_paper_trader("TMF")

        # 🎯 徹底解耦：無論使用者在上方看什麼商品，期貨模擬操盤帳戶永遠只看微台 (TMFR1) 的 K 線與價格
        feed_tmf_bars = df_bars
        tmf_bars_1m = feed.get_bars(timeframe="1m", session="day_and_night") if (is_fut and "TMF" in clean_sym) else None
        tmf_curr_p = curr_price
        if not is_fut or "TMF" not in clean_sym:
            # 若目前看板顯示的不是微台，獨立載入 TMFR1 專屬最新 K 線與價格
            store_inst = feed.store or KBarStore()
            tmf_dict = store_inst.load_kbars_dict("TMFR1")
            feed_tmf_bars = TaishinQuoteFeed._empty_bars_df()
            tmf_curr_p = 0.0
            if tmf_dict:
                tmf_feed_temp = TaishinQuoteFeed(auto_load_store=False)
                tmf_feed_temp.kbars_1m = tmf_dict
                tmf_feed_temp.symbol = "TMFR1"
                feed_tmf_bars = tmf_feed_temp.get_bars("5m")
                tmf_bars_1m = tmf_feed_temp.get_bars("1m")
                last_tmf_dt = max(tmf_dict.keys())
                tmf_curr_p = float(tmf_dict[last_tmf_dt]["close"])

        # 判斷台指期開盤狀態
        tmf_is_open, _ = FuturesAIAdvisor.get_market_session_status("TMF")

        # 📐 開盤區間突破 (ORB) 規則：模擬倉的多空方向完全由規則決定，AI 分析僅供參考
        now_tw = datetime.now(timezone(timedelta(hours=8))).replace(tzinfo=None)
        orb = compute_orb_state(
            tmf_bars_1m if tmf_bars_1m is not None else TaishinQuoteFeed._empty_bars_df(), now=now_tw, params=ORB_PARAMS
        )
        # 規則訊號大卡片（模擬倉的多空方向唯一依據）
        if orb.flatten:
            orb_title, orb_color, orb_bg = "⏰ 非進場時段：不開新倉，持倉強制平倉", "#848e9c", "rgba(132, 142, 156, 0.14)"
        elif orb.action == "BUY_LONG":
            orb_title, orb_color, orb_bg = "🟢 突破開盤區間上緣：做多", "#089981", "rgba(8, 153, 129, 0.16)"
        elif orb.action == "SELL_SHORT":
            orb_title, orb_color, orb_bg = "🔴 跌破開盤區間下緣：做空", "#f23645", "rgba(242, 54, 69, 0.16)"
        elif not orb.range_ready:
            orb_title, orb_color, orb_bg = "⏳ 開盤區間形成中：觀望", "#f0b90b", "rgba(240, 185, 11, 0.14)"
        else:
            orb_title, orb_color, orb_bg = "⚪ 價格仍在開盤區間內：觀望", "#f0b90b", "rgba(240, 185, 11, 0.14)"

        if orb.range_high is not None and orb.range_low is not None:
            range_html = (
                f"<div>📈 <b>區間上緣：</b><span style='font-size:20px;font-weight:700;color:#e0e3eb;'>{orb.range_high:,.0f}</span></div>"
                f"<div>📉 <b>區間下緣：</b><span style='font-size:20px;font-weight:700;color:#e0e3eb;'>{orb.range_low:,.0f}</span></div>"
                f"<div>📏 <b>區間寬度：</b>{orb.range_high - orb.range_low:,.0f} 點</div>"
            )
            if tmf_curr_p > 0:
                range_html += (
                    f"<div>💹 <b>微台現價：</b>{tmf_curr_p:,.0f}"
                    f"<br/><span style='font-size:12.5px;color:#848e9c;'>距上緣 {orb.range_high - tmf_curr_p:+,.0f} ／ 距下緣 {orb.range_low - tmf_curr_p:+,.0f} 點</span></div>"
                )
        else:
            range_html = f"<div>開盤區間將於 {ORB_PARAMS.session_open:%H:%M} 起 {ORB_PARAMS.range_minutes} 分鐘內形成</div>"
        breakout_html = (
            f"<div style='margin-top:8px;font-size:15px;color:#e0e3eb;'>⚡ {orb.reason}：{orb.breakout_ts:%H:%M} 收盤 {orb.breakout_price:,.0f}</div>"
            if orb.breakout_ts else f"<div style='margin-top:8px;font-size:15px;color:#cfd8dc;'>📝 {orb.reason}</div>"
        )
        st.markdown(
            f'<div style="background-color: {orb_bg}; border: 2px solid {orb_color}; border-radius: 12px; padding: 18px 24px; margin: 10px 0 14px 0;">'
            '<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">'
            f'<span style="font-size: 15px; font-weight: 600; color: #b2b5be;">📐 規則訊號：開盤區間突破（微台 TMF，{ORB_PARAMS.session_open:%H:%M} 起 {ORB_PARAMS.range_minutes} 分鐘區間）｜ 模擬倉依此下單</span>'
            f'<span style="font-size: 13px; color: #848e9c;">更新於 {now_tw:%H:%M:%S}</span>'
            '</div>'
            f'<div style="font-size: 30px; font-weight: 800; color: {orb_color}; margin: 10px 0 12px 0;">{orb_title}</div>'
            '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; font-size: 14.5px; color: #d1d4dc;">'
            f'{range_html}'
            '</div>'
            f'{breakout_html}'
            f'<div style="margin-top:10px;padding-top:8px;border-top:1px dashed rgba(255,255,255,0.16);font-size:13px;color:#9aa0a6;">'
            f'規則：區間完成後第一根 1 分 K 收盤突破即進場，一天最多一次、不反手；{ORB_PARAMS.last_entry:%H:%M} 後不開新倉，'
            f'{ORB_PARAMS.flatten_at:%H:%M} 強制平倉；停損停利沿用 ATR 設定。⚠️ 尚未經足夠歷史資料驗證。</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # 🛡️ 停損停利、收盤強制平倉與規則進場（停損與平倉不受暫停或風控鎖定影響；價格防護由引擎把關）
        if tmf_is_open and tmf_curr_p > 0 and (pt.position is not None or pt.is_active):
            trade_res = pt.sync_with_rule_signal(
                action=orb.action,
                current_price=tmf_curr_p,
                df_bars=feed_tmf_bars,
                flatten=orb.flatten,
                source_note="開盤區間突破",
            )
            if trade_res and trade_res != "系統暫停或風控鎖定中":
                st.toast(f"⚡ 微台模擬倉：{trade_res}", icon="📐")

        # 操盤狀態橫幅卡（永遠綁定微台專屬即時價格 tmf_curr_p）
        unrealized_pnl = pt.get_unrealized_pnl(tmf_curr_p)
        total_eq = pt.get_total_equity(tmf_curr_p)
        ret_pct = ((total_eq - pt.initial_capital) / pt.initial_capital) * 100.0

        pos_str = "空手等待訊號"
        pos_color = "#848e9c"
        if pt.position is not None:
            dir_zh = "🟢 多單" if pt.position.direction == 1 else "🔴 空單"
            sl_info = f" (停損: {pt.position.stop_loss_price:,.1f})" if pt.position.stop_loss_price else ""
            pos_str = f"{dir_zh} {pt.position.contracts} 口 @ {pt.position.entry_price:,.1f}{sl_info}"
            pos_color = "#00e676" if pt.position.direction == 1 else "#ff5252"

        status_badge = "🟢 自動操盤運作中" if pt.is_active else "⏸ 自動操盤已暫停"
        if pt.kill_switch_locked:
            status_badge = "🚨 Kill Switch 鎖定中"
        elif pt.daily_loss_locked:
            status_badge = "🛑 單日風控鎖定中"

        st.markdown(f"""
        <div style="background-color: rgba(22, 26, 37, 0.9); border: 2px solid #2962ff; border-radius: 12px; padding: 16px 20px; margin: 10px 0 16px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; margin-bottom: 8px;">
                <span style="font-size: 17px; font-weight: 700; color: #40c4ff;">🤖 AI 期貨自動模擬操盤中心（微台 TMF 1口模擬）</span>
                <span style="font-size: 13.5px; font-weight: 600; padding: 3px 10px; border-radius: 6px; background-color: rgba(41, 98, 255, 0.2); color: #82b1ff;">{status_badge}</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; font-size: 14px;">
                <div>💼 <b>虛擬總權益：</b><span style="color:#ffffff; font-weight:700; font-size:16px;">{total_eq:,.0f} 元</span> <span style="font-size:12px; color:{'#00e676' if ret_pct>=0 else '#ff5252'};">({ret_pct:+.2f}%)</span></div>
                <div>📊 <b>持倉部位：</b><span style="color:{pos_color}; font-weight:700;">{pos_str}</span></div>
                <div>💰 <b>未實現損益：</b><span style="color:{'#00e676' if unrealized_pnl>=0 else '#ff5252'}; font-weight:700;">{unrealized_pnl:+,.0f} 元</span></div>
                <div>🎯 <b>已實現損益：</b><span style="color:{'#00e676' if pt.realized_pnl>=0 else '#ff5252'}; font-weight:700;">{pt.realized_pnl:+,.0f} 元 ({len(pt.closed_trades)}筆)</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 操盤控制按鈕與緊急平倉
        can_write = render_admin_gate()
        st.session_state["warroom_can_write"] = can_write
        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4, ctrl_col5 = st.columns([1.2, 1.2, 1.1, 1.1, 1.6])
        with ctrl_col1:
            if not pt.is_active:
                unlock_first = pt.daily_loss_locked or pt.daily_profit_locked or pt.kill_switch_locked or pt.degradation_locked
                render_credibility_start(pt, feed_tmf_bars, unlock_first=unlock_first, can_write=can_write)
            else:
                if st.button("⏸ 暫停自動操盤", key="btn_stop_ai_trade", disabled=not can_write):
                    flat_msg = pt.stop_session(force_flat_price=tmf_curr_p)
                    pt.save_state_to_disk()
                    if flat_msg:
                        st.info(flat_msg)
                    st.rerun()
        with ctrl_col2:
            # 🔒 下單安全鎖 (對齊閃電下單安全鎖機制)
            is_locked = getattr(pt, "order_safety_locked", False)
            btn_lock_label = "🔒 下單已鎖定 (點此解鎖)" if is_locked else "⚡ 下單已解鎖 (點此鎖定)"
            btn_lock_type = "secondary" if is_locked else "primary"
            if st.button(btn_lock_label, key="btn_toggle_order_lock", type=btn_lock_type, disabled=not can_write, help="鎖定時僅能執行平倉與風控，嚴格禁止開立新倉"):
                pt.order_safety_locked = not is_locked
                pt.save_state_to_disk()
                st.rerun()
        with ctrl_col3:
            if st.button("🚨 一鍵緊急平倉 (Kill)", key="btn_kill_switch", disabled=not can_write, help="立即市價平掉所有部位並鎖定系統"):
                kill_msg = pt.trigger_kill_switch(current_price=tmf_curr_p)
                pt.save_state_to_disk()
                st.warning(kill_msg)
                st.rerun()
        with ctrl_col4:
            if st.button("🧹 全部模擬重置 (回10萬)", key="btn_reset_all_sim", disabled=not can_write, help="清空所有持倉、歷史成交紀錄與損益，將虛擬總權益恢復為初始 100,000 元"):
                pt.reset_all_simulation(initial_capital=100_000.0)
                st.toast("🔄 模擬帳戶已完全重置，本金恢復 100,000 元！", icon="🧹")
                st.rerun()
        with ctrl_col5:
            st.caption(f"🛡️ 風控：單日虧損上限 -{pt.risk_config.max_daily_loss:,.0f} 元 ｜ 連虧4筆休眠 ｜ 上限 {pt.risk_config.max_contracts} 口")

        with st.expander("⚙️ 點擊自訂個人操盤策略（停損／停利／移動鎖利以 ATR 倍數設定、持倉上限、單日虧損上限）"):
            ex_cur = pt.strategy.exit

            def _atr_mult(spec, default: float) -> float:
                return float(spec.value) if spec is not None and spec.type == "atr_multiple" and spec.value else default

            # 目前 ATR(14) 估計值，僅用於把倍數換算成點數提示
            atr_now = 0.0
            if feed_tmf_bars.height > DEFAULT_ATR_LENGTH:
                atr_last = feed_tmf_bars.select(IndicatorEngine.calculate_atr(feed_tmf_bars, DEFAULT_ATR_LENGTH)).to_series()[-1]
                atr_now = float(atr_last) if atr_last is not None and atr_last == atr_last else 0.0
            st.caption(
                f"📏 目前微台 5 分 K ATR({DEFAULT_ATR_LENGTH}) ≈ **{atr_now:,.0f} 點**（停損距離會隨市場波動自動調整）"
                if atr_now > 0 else "📏 ATR 尚在暖機中（K 棒不足），暫時無法換算點數"
            )
            sc1, sc2, sc3, sc4, sc5, sc6, sc7 = st.columns([1.0, 1.0, 1.0, 1.0, 0.8, 1.1, 0.8])
            with sc1:
                new_sl = st.number_input("停損 (ATR 倍數)", min_value=0.5, max_value=5.0, step=0.1,
                                         value=min(max(_atr_mult(ex_cur.stop_loss, DEFAULT_ATR_STOP_MULT), 0.5), 5.0),
                                         help="進場後反向虧損達「倍數 × 進場時 ATR」即停損")
                if atr_now > 0:
                    st.caption(f"≈ {new_sl * atr_now:,.0f} 點 / {new_sl * atr_now * pt.contract.point_value:,.0f} 元/口")
            with sc2:
                new_tp = st.number_input("停利 (ATR 倍數)", min_value=0.5, max_value=10.0, step=0.1,
                                         value=min(max(_atr_mult(ex_cur.take_profit, DEFAULT_ATR_TP_MULT), 0.5), 10.0),
                                         help="獲利達「倍數 × 進場時 ATR」即停利；建議大於停損倍數")
                if atr_now > 0:
                    st.caption(f"≈ {new_tp * atr_now:,.0f} 點")
            with sc3:
                new_ts = st.number_input("移動鎖利距離 (ATR 倍數)", min_value=0.3, max_value=5.0, step=0.1,
                                         value=min(max(_atr_mult(ex_cur.trailing_stop, DEFAULT_ATR_TRAIL_MULT), 0.3), 5.0),
                                         help="啟動後，自波段最高（低）點回檔超過「倍數 × ATR」即出場")
            with sc4:
                cur_act = (ex_cur.trailing_stop.activation_atr_multiple if ex_cur.trailing_stop else None) or DEFAULT_ATR_TRAIL_ACTIVATION
                new_act = st.number_input("移動鎖利啟動 (ATR 倍數)", min_value=0.3, max_value=10.0, step=0.1,
                                          value=min(max(float(cur_act), 0.3), 10.0),
                                          help="獲利達「倍數 × 進場時 ATR」後才開始移動鎖利")
            with sc5:
                new_max_c = st.number_input("持倉上限 (口)", min_value=1, max_value=10,
                                            value=min(max(int(pt.risk_config.max_contracts), 1), 10), step=1)
            with sc6:
                new_daily_loss = st.number_input("單日最大虧損上限 (元)", min_value=500.0, max_value=50000.0, step=500.0,
                                                 value=min(max(float(pt.risk_config.max_daily_loss), 500.0), 50000.0),
                                                 help="當日累積虧損達到此金額時，強制平倉並停止當日交易")
                if atr_now > 0 and new_daily_loss < new_sl * atr_now * pt.contract.point_value * new_max_c:
                    st.caption("⚠️ 上限小於一次停損金額，停損前就會先觸發單日鎖定")
            with sc7:
                st.write("")
                st.write("")
                if st.button("💾 儲存設定", key="btn_save_user_strategy", type="secondary", disabled=not can_write):
                    strat_data = pt.strategy.model_dump()
                    strat_data["exit"] = {
                        **strat_data["exit"],
                        "stop_loss": {"type": "atr_multiple", "value": float(new_sl), "atr_length": DEFAULT_ATR_LENGTH},
                        "take_profit": {"type": "atr_multiple", "value": float(new_tp), "atr_length": DEFAULT_ATR_LENGTH},
                        "trailing_stop": {"type": "atr_multiple", "value": float(new_ts),
                                          "activation_atr_multiple": float(new_act), "atr_length": DEFAULT_ATR_LENGTH},
                    }
                    try:
                        new_strategy = StrategySchema.model_validate(strat_data)
                    except ValueError as ex:
                        st.error(f"設定不合法，未儲存：{ex}")
                    else:
                        pt.strategy = new_strategy
                        pt.core_engine = EventDrivenEngine(new_strategy, pt.contract, pt.initial_capital)
                        pt.custom_tp_points = 0.0  # 停利改由 ATR 倍數決定，清除舊的固定點數覆寫
                        pt.risk_config = replace(pt.risk_config, max_contracts=int(new_max_c), max_daily_loss=float(new_daily_loss))
                        # 出場參數已變更：先前的可信度檢核結果不再適用於這組新參數
                        pt.credibility_verified = False
                        pt._log("SYSTEM", f"使用者修改策略參數（停損 {new_sl:.1f} / 停利 {new_tp:.1f} / 移動鎖利 {new_ts:.1f}，啟動 {new_act:.1f} 倍 ATR），可信度檢核需重新執行")
                        pt.save_state_to_disk()
                        st.success("✅ 策略參數已套用儲存！")
                        st.rerun()

        with st.expander("📋 展開檢視：AI 模擬操盤交易明細與風控稽核日誌"):
            log_c1, log_c2 = st.columns([1.2, 1.0])
            with log_c1:
                st.markdown("**📜 已平倉歷史交易紀錄**")
                if pt.closed_trades:
                    t_rows = []
                    for t in reversed(pt.closed_trades[-15:]):
                        t_rows.append({
                            "方向": "多" if t.direction == 1 else "空",
                            "口數": t.contracts,
                            "進場價": f"{t.entry_price:,.1f}",
                            "出場價": f"{t.exit_price:,.1f}",
                            "手續費+稅": f"{t.commission + t.tax:,.0f}",
                            "淨損益": f"{t.net_pnl:+,.0f} 元",
                            "出場原因": t.exit_reason,
                        })
                    st.dataframe(pd.DataFrame(t_rows), width="stretch", hide_index=True)
                else:
                    st.caption("目前尚無平倉交易紀錄")
            with log_c2:
                st.markdown("**🛡️ 即時風控與撮合稽核日誌**")
                if pt.audit_logs:
                    l_rows = [
                        {"時間": l.timestamp[-8:], "事件": l.event_type, "訊息": l.message}
                        for l in reversed(pt.audit_logs[-15:])
                    ]
                    st.dataframe(pd.DataFrame(l_rows), width="stretch", hide_index=True)
                else:
                    st.caption("目前尚無稽核日誌")

        # ==================== 2.8 💬 AI 戰情操盤顧問即時對話框 (獨立 Fragment，不受 15 秒刷新干擾，發送立即回覆) ====================
        @st.fragment
        def _render_ai_chat_box():
            if "war_room_chat_history" not in st.session_state:
                st.session_state["war_room_chat_history"] = [
                    {
                        "role": "assistant",
                        "content": (
                            "👋 您好！我是您的 **AI 戰情室即時操盤顧問**。\n"
                            "您可以直接在下方輸入問題或下達操作指令，例如：\n"
                            "- 📊 **盤勢提問**：「為什麼現在沒有交易？」、「目前支撐跟壓力在哪？」、「現在適合做多還是做空？」\n"
                            "- ⚡ **直接指令**：「幫我把停利改成 50 點」、「幫我平倉」、「幫我重置回 10 萬」、「鎖定下單」"
                        ),
                    }
                ]

            with st.expander("💬 AI 戰情操盤顧問即時對話框（可直接問盤勢、查部位損益、或下達「平倉 / 重置10萬 / 停利改50點」指令）", expanded=True):
                q_c1, q_c2, q_c3, q_c4 = st.columns([1.2, 1.2, 1.2, 0.8])
                quick_prompt = None
                with q_c1:
                    if st.button("❓ 為什麼現在沒交易 / 觀望？", key="btn_qa_why_wait", width="stretch"):
                        quick_prompt = "為什麼現在沒有交易？何時才會進場？"
                with q_c2:
                    if st.button("🎯 目前支撐與壓力點位在哪？", key="btn_qa_levels", width="stretch"):
                        quick_prompt = f"請幫我診斷目前 {clean_sym} 的支撐、壓力與目標停損點位"
                with q_c3:
                    if st.button("💼 診斷我目前的持倉與損益", key="btn_qa_pos", width="stretch"):
                        quick_prompt = "請幫我診斷目前的模擬帳戶持倉部位、損益與風控狀態"
                with q_c4:
                    if st.button("🧹 清空對話紀錄", key="btn_qa_clear", width="stretch"):
                        st.session_state["war_room_chat_history"] = st.session_state["war_room_chat_history"][:1]

                with st.form(key="war_room_chat_form", clear_on_submit=True):
                    in_col1, in_col2 = st.columns([4.5, 1.0])
                    with in_col1:
                        user_chat_input = st.text_input(
                            "輸入問題或操盤指令",
                            placeholder="直接跟我對話，例如：現在為什麼沒進場？ / 支撐壓力在哪？ / 停利改 50 點 / 幫我重置回 10 萬...",
                            label_visibility="collapsed",
                        )
                    with in_col2:
                        send_chat = st.form_submit_button("📨 發送對話", type="primary", width="stretch")

                active_msg = quick_prompt or (user_chat_input.strip() if (send_chat and user_chat_input) else None)
                if active_msg:
                    st.session_state["last_active_time"] = time.time()
                    st.session_state["war_room_chat_history"].append({"role": "user", "content": active_msg})
                    with st.spinner("🤖 AI 顧問正在分析回覆中..."):
                        reply_text = FuturesAIAdvisor.chat_with_advisor(
                            user_msg=active_msg,
                            symbol=clean_sym,
                            timeframe=chart_tf,
                            current_price=curr_price,
                            advice=advice,
                            paper_trader=pt,
                            chat_history=st.session_state["war_room_chat_history"][:-1],
                            gemini_api_key=default_cfg.get("gemini_api_key", ""),
                            gemini_model=default_cfg.get("gemini_model", "gemini-2.5-flash"),
                            trader_price=tmf_curr_p,
                            allow_commands=st.session_state.get("warroom_can_write", False),
                        )
                    st.session_state["war_room_chat_history"].append({"role": "assistant", "content": reply_text})
                    # 只有實際改變帳戶狀態（回覆開頭為 ✅/🔒/⚡/⚠️/⏸️/▶️/🎯）時才整頁刷新，讓上方帳戶面板同步
                    if reply_text.startswith(("✅", "🔒", "⚡", "⚠️", "⏸️", "▶️", "🎯")):
                        st.rerun()

                # 顯示最近 8 則對話訊息（最新對話立即呈現在最下方）
                for msg_item in st.session_state["war_room_chat_history"][-8:]:
                    role_label = "assistant" if msg_item.get("role") == "assistant" else "user"
                    with st.chat_message(role_label):
                        st.markdown(msg_item.get("content", ""))

        _render_ai_chat_box()

        st.markdown("---")
        k_top1, k_top2, k_top3 = st.columns([2.3, 1.0, 0.9])
        with k_top1:
            if is_mkt_open:
                st.subheader(f"📈 {ts_symbol} — {chart_tf} 即時 K 線與量價決策戰情圖")
            else:
                st.subheader(f"📈 {ts_symbol} — {chart_tf} 收盤 K 線圖（{mkt_status_label}）")
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
            if len(df_plot) > 3:
                med_p = float(df_plot["close"].median())
                if med_p > 0:
                    df_plot = df_plot[
                        (df_plot["close"] >= med_p * 0.65)
                        & (df_plot["close"] <= med_p * 1.35)
                        & (df_plot["low"] >= med_p * 0.65)
                    ].reset_index(drop=True)
            df_plot["time_label"] = df_plot["timestamp"].dt.strftime("%m-%d %H:%M")
            df_plot["ma5"] = df_plot["close"].rolling(5).mean()
            df_plot["ma20"] = df_plot["close"].rolling(20).mean()
            df_plot["ma60"] = df_plot["close"].rolling(60).mean()
            bb_std_s = df_plot["close"].rolling(20).std().fillna(0.0)
            df_plot["bb_up"] = df_plot["ma20"] + 2.0 * bb_std_s
            df_plot["bb_low"] = df_plot["ma20"] - 2.0 * bb_std_s

            # 智慧傳輸瘦身：先算完完整均線與布林通道後，若非選「全部顯示」，僅傳送最近 max(240, vis_count*2) 根至前端
            if vis_bars_opt != "全部顯示":
                req_vis = int(vis_bars_opt.split(" ")[0])
                max_send = max(240, req_vis * 2)
                if len(df_plot) > max_send:
                    df_plot = df_plot.iloc[-max_send:].reset_index(drop=True)

            total_k = len(df_plot)
            vis_count = total_k if vis_bars_opt == "全部顯示" else min(total_k, int(vis_bars_opt.split(" ")[0]))
            init_start = max(0, total_k - vis_count)

            last_close = float(df_plot["close"].iloc[-1])
            last_open = float(df_plot["open"].iloc[-1])
            last_high = float(df_plot["high"].iloc[-1])
            last_low = float(df_plot["low"].iloc[-1])
            last_vol = int(df_plot["volume"].iloc[-1])
            last_label = str(df_plot["time_label"].iloc[-1])
            last_ma5 = float(df_plot["ma5"].iloc[-1]) if pd.notna(df_plot["ma5"].iloc[-1]) else last_close
            last_ma20 = float(df_plot["ma20"].iloc[-1]) if pd.notna(df_plot["ma20"].iloc[-1]) else last_close
            last_ma60 = float(df_plot["ma60"].iloc[-1]) if pd.notna(df_plot["ma60"].iloc[-1]) else last_close
            recent_bar_atr = float((df_plot["high"].tail(14) - df_plot["low"].tail(14)).mean())
            eff_atr = (
                float(advice.atr_val)
                if getattr(advice, "atr_val", 0.0) > 0
                else max(12.0 if is_fut else 0.5, recent_bar_atr)
            )

            # 買賣膠囊與均線狀態帶 (對齊截圖左上角「買進 42,267 | 1 | 42,271 賣出」與「開高低收量 + MA5/20/60」，無前導縮排)
            bid_v = mkt_stats.get("bid_price", last_close)
            ask_v = mkt_stats.get("ask_price", last_close + 1.0)
            spd_v = mkt_stats.get("spread", 1.0)

            chart_header_html = (
                '<div style="background: rgba(19, 23, 34, 0.95); border: 1px solid #2a2e39; border-radius: 6px; padding: 6px 12px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; font-family: monospace; font-size: 13px;">'
                '<div style="display: flex; align-items: center; gap: 6px;">'
                f'<div style="background: #26a69a; color: #ffffff; padding: 2px 10px; border-radius: 4px; font-weight: 700;">買進 {bid_v:,.0f}</div>'
                f'<div style="background: #1e222d; color: #29b6f6; padding: 2px 8px; border-radius: 4px; border: 1px solid #2a2e39; font-weight: 700;">{spd_v:,.0f}</div>'
                f'<div style="background: #ef5350; color: #ffffff; padding: 2px 10px; border-radius: 4px; font-weight: 700;">賣出 {ask_v:,.0f}</div>'
                '</div>'
                '<div style="display: flex; gap: 10px; color: #d1d4dc; font-size: 12.5px; flex-wrap: wrap;">'
                f'<span>開 <b style="color:#ffffff;">{last_open:,.1f}</b></span>'
                f'<span>高 <b style="color:#ef5350;">{last_high:,.1f}</b></span>'
                f'<span>低 <b style="color:#26a69a;">{last_low:,.1f}</b></span>'
                f'<span>收 <b style="color:#ffffff;">{last_close:,.1f}</b></span>'
                f'<span>量 <b style="color:#f0b90b;">{last_vol:,}</b></span>'
                f'<span style="color: #ffa726;">MA5 <b>{last_ma5:,.1f}</b></span>'
                f'<span style="color: #29b6f6;">MA20 <b>{last_ma20:,.1f}</b></span>'
                f'<span style="color: #ba68c8;">MA60 <b>{last_ma60:,.1f}</b></span>'
                '</div>'
                '</div>'
            )
            st.markdown(chart_header_html, unsafe_allow_html=True)

            # 📏 客觀關鍵價位（取代 AI 預估線）：前一日高低收、當日 VWAP、預期波動區間、開盤區間上下緣
            future_labels: list[str] = []  # 不再於 X 軸附加任何「預估」K 棒
            session_open_t, session_close_t = (dt_time(8, 45), dt_time(13, 45)) if is_fut else (dt_time(9, 0), dt_time(13, 30))
            levels_1m = feed.get_bars(timeframe="1m", session="day_and_night")
            if is_fut and any(k in clean_sym for k in ("TMF", "TXF", "MTX", "MXF")):
                # 台指期家族報價點數相同：以期交所歷史微台 1 分 K 補足前幾日資料（計算平均日振幅用）
                hist_bars = load_history_bars()
                if hist_bars is not None and hist_bars.height:
                    levels_1m = pl.concat([hist_bars, levels_1m.select(hist_bars.columns)]).unique(
                        subset="timestamp", keep="last"
                    ).sort("timestamp")
            now_levels = datetime.now(timezone(timedelta(hours=8))).replace(tzinfo=None)
            key_lv = compute_key_levels(levels_1m, now=now_levels, open_t=session_open_t, close_t=session_close_t)
            exp_rate, exp_days = expected_range_hit_rate_cached(clean_sym, session_open_t, session_close_t, levels_1m)

            # 計算 Y 軸顯示範圍，確保 K 線、布林通道與合理目標/停損線完整呈現且不被壓扁
            vis_slice = df_plot.iloc[init_start:]
            y_candidates = [
                float(vis_slice["low"].min()),
                float(vis_slice["high"].max()),
            ]
            if not vis_slice["bb_up"].dropna().empty:
                y_candidates.append(float(vis_slice["bb_up"].dropna().max()))
                y_candidates.append(float(vis_slice["bb_low"].dropna().min()))
            for lv in (key_lv.prev_high, key_lv.prev_low, key_lv.prev_close, key_lv.vwap, key_lv.exp_upper, key_lv.exp_lower):
                if lv is not None and abs(lv - last_close) <= last_close * 0.03:  # 只納入現價 ±3% 內，避免 K 線被壓扁
                    y_candidates.append(lv)
            y_min = min(y_candidates)
            y_max = max(y_candidates)
            y_pad = max(1.0, (y_max - y_min) * 0.08)

            # 水平參考線橫跨最近 35 根 K 線至未來預估區間，畫面清晰不雜亂
            ref_start_idx = max(init_start, total_k - 35)
            ref_x = list(df_plot["time_label"].iloc[ref_start_idx:]) + future_labels

            fig_k = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.78, 0.22],
            )

            # 主圖：K 線圖
            fig_k.add_trace(go.Candlestick(
                x=df_plot["time_label"],
                open=df_plot["open"],
                high=df_plot["high"],
                low=df_plot["low"],
                close=df_plot["close"],
                increasing_line_color="#ef5350",
                decreasing_line_color="#26a69a",
                name="K線",
            ), row=1, col=1)
            fig_k.add_trace(go.Scatter(
                x=df_plot["time_label"],
                y=df_plot["ma5"],
                line=dict(color="#ffa726", width=1.6),
                name="5MA",
            ), row=1, col=1)
            fig_k.add_trace(go.Scatter(
                x=df_plot["time_label"],
                y=df_plot["ma20"],
                line=dict(color="#29b6f6", width=1.6),
                name="20MA(中軸成本)",
            ), row=1, col=1)
            fig_k.add_trace(go.Scatter(
                x=df_plot["time_label"],
                y=df_plot["ma60"],
                line=dict(color="#ba68c8", width=1.5),
                name="60MA(季線/趨勢)",
            ), row=1, col=1)
            fig_k.add_trace(go.Scatter(
                x=df_plot["time_label"],
                y=df_plot["bb_up"],
                line=dict(color="rgba(186, 104, 200, 0.65)", width=1.2, dash="dot"),
                name="布林上軌(2σ壓力)",
            ), row=1, col=1)
            fig_k.add_trace(go.Scatter(
                x=df_plot["time_label"],
                y=df_plot["bb_low"],
                line=dict(color="rgba(186, 104, 200, 0.65)", width=1.2, dash="dot"),
                name="布林下軌(2σ支撐)",
            ), row=1, col=1)

            # 水平關鍵價位線（全部為客觀公式計算，非預測）
            def _hline(y: Optional[float], name: str, color: str, dash: str = "dot", width: float = 1.4) -> None:
                if y is None:
                    return
                fig_k.add_trace(go.Scatter(
                    x=ref_x, y=[y] * len(ref_x), mode="lines",
                    line=dict(color=color, width=width, dash=dash), name=f"{name} ({y:,.0f})",
                ), row=1, col=1)

            _hline(key_lv.prev_high, "前日高", "#ef9a9a")
            _hline(key_lv.prev_low, "前日低", "#80cbc4")
            _hline(key_lv.prev_close, "前日收", "#b0bec5", dash="dash", width=1.1)
            _hline(key_lv.vwap, "今日 VWAP 均價", "#ffd54f", dash="solid", width=1.6)
            rate_txt = f"，歷史 {exp_days} 天中 {exp_rate * 100:.0f}% 整天落在區間內" if exp_rate is not None else ""
            _hline(key_lv.exp_upper, f"預期波動上緣（開盤＋平均日振幅{rate_txt}）", "#9575cd", dash="dashdot")
            _hline(key_lv.exp_lower, "預期波動下緣（開盤－平均日振幅）", "#9575cd", dash="dashdot")
            if is_fut and "TMF" in clean_sym and orb.range_high is not None:
                _hline(orb.range_high, "📐 開盤區間上緣（突破做多）", "#00e676", dash="dash", width=2.0)
                _hline(orb.range_low, "📐 開盤區間下緣（跌破做空）", "#ff5252", dash="dash", width=2.0)

            # 🎯 K 線買賣進出場標記點位（對齊截圖中的「▲ 進買1 @ 42,234」與「▼ 出賣1 @ 42,349」，支援 1m/5m/15m/30m/60m 自動吸附對應 K 棒）
            def _resolve_chart_x_label(raw_t: str) -> Optional[str]:
                if not raw_t:
                    return None
                lbl_11 = raw_t[-11:] if len(raw_t) >= 11 else raw_t
                if lbl_11 in set(df_plot["time_label"]):
                    return lbl_11
                try:
                    dt_parsed = pd.to_datetime(raw_t[:16])
                    matched = df_plot[df_plot["timestamp"] <= dt_parsed]
                    if not matched.empty:
                        last_match_dt = matched["timestamp"].iloc[-1]
                        if abs((dt_parsed - last_match_dt).total_seconds()) <= 3600:
                            return str(matched["time_label"].iloc[-1])
                except Exception:
                    pass
                return None

            if is_fut:
                # 1. 標註已平倉交易
                for tr in (pt.closed_trades[-12:] if pt.closed_trades else []):
                    e_lbl = _resolve_chart_x_label(tr.entry_time)
                    x_lbl = _resolve_chart_x_label(tr.exit_time)
                    if e_lbl:
                        arr_color = "#ef5350" if tr.direction == 1 else "#26a69a"
                        arr_text = f"▲ 進買{tr.contracts} @ {tr.entry_price:,.0f}" if tr.direction == 1 else f"▼ 進空{tr.contracts} @ {tr.entry_price:,.0f}"
                        fig_k.add_annotation(
                            x=e_lbl,
                            y=tr.entry_price,
                            text=arr_text,
                            showarrow=True,
                            arrowhead=2,
                            arrowcolor=arr_color,
                            ay=32 if tr.direction == 1 else -32,
                            font=dict(color="#ffffff", size=10.5),
                            bgcolor="rgba(19, 23, 34, 0.9)",
                            bordercolor=arr_color,
                            borderwidth=1,
                            row=1,
                            col=1,
                        )
                    if x_lbl:
                        x_color = "#00e676" if tr.net_pnl > 0 else "#ff5252"
                        x_text = f"▼ 出賣{tr.contracts} @ {tr.exit_price:,.0f}" if tr.direction == 1 else f"▲ 出買{tr.contracts} @ {tr.exit_price:,.0f}"
                        fig_k.add_annotation(
                            x=x_lbl,
                            y=tr.exit_price,
                            text=f"{x_text} ({tr.net_pnl:+,.0f})",
                            showarrow=True,
                            arrowhead=2,
                            arrowcolor=x_color,
                            ay=-32 if tr.direction == 1 else 32,
                            font=dict(color="#ffffff", size=10),
                            bgcolor="rgba(19, 23, 34, 0.9)",
                            bordercolor=x_color,
                            borderwidth=1,
                            row=1,
                            col=1,
                        )
                # 2. 標註當前持倉進場點
                if pt.position is not None:
                    pos_lbl = _resolve_chart_x_label(pt.position.entry_time) or last_label
                    p_color = "#ef5350" if pt.position.direction == 1 else "#26a69a"
                    p_text = f"▲ 進買{pt.position.contracts} @ {pt.position.entry_price:,.0f}" if pt.position.direction == 1 else f"▼ 進賣{pt.position.contracts} @ {pt.position.entry_price:,.0f}"
                    fig_k.add_annotation(
                        x=pos_lbl,
                        y=pt.position.entry_price,
                        text=p_text,
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor=p_color,
                        ay=35 if pt.position.direction == 1 else -35,
                        font=dict(color="#ffffff", size=11, family="Roboto Mono"),
                        bgcolor=p_color,
                        bordercolor="#ffffff",
                        borderwidth=1,
                        row=1,
                        col=1,
                    )

            # 副圖：紅綠成交量柱狀圖 (Row 2，補齊未來預估空白格以確保上下子圖 X 軸類別座標 100% 同步對齊)
            vol_x = list(df_plot["time_label"]) + (future_labels if is_mkt_open else [])
            vol_y = list(df_plot["volume"]) + ([0] * len(future_labels) if is_mkt_open else [])
            vol_colors = [
                "rgba(239, 83, 80, 0.75)" if c >= o else "rgba(38, 166, 154, 0.75)"
                for o, c in zip(df_plot["open"], df_plot["close"])
            ] + (["rgba(0,0,0,0)"] * len(future_labels) if is_mkt_open else [])
            fig_k.add_trace(go.Bar(
                x=vol_x,
                y=vol_y,
                marker_color=vol_colors,
                name="成交量",
                showlegend=False,
            ), row=2, col=1)

            # 由最後一根最新 K 棒 (total_k - 1) 往回取固定間距刻度，確保最新 K 棒時間 100% 顯示在 X 軸上絕不跳格！
            tick_step = max(1, vis_count // 16)
            tick_indices = list(range(total_k - 1, -1, -tick_step))[::-1]
            all_time_labels = df_plot["time_label"].tolist()
            explicit_tickvals = [all_time_labels[idx] for idx in tick_indices]
            x_max_range = total_k - 0.2

            shared_x_cfg = dict(
                type="category",
                tickmode="array",
                tickvals=explicit_tickvals,
                ticktext=explicit_tickvals,
                range=[init_start - 0.5, x_max_range],
                minallowed=-0.5,
                maxallowed=x_max_range + 1.2,
            )

            fig_k.update_layout(
                template="plotly_dark",
                height=580,
                margin=dict(l=15, r=60, t=80, b=25),
                xaxis_rangeslider_visible=False,
                xaxis=dict(**shared_x_cfg, showticklabels=False),
                xaxis2=dict(**shared_x_cfg, showticklabels=True),
                yaxis=dict(
                    range=[y_min - y_pad, y_max + y_pad],
                    fixedrange=False,
                ),
                yaxis2=dict(
                    title="量",
                    showgrid=False,
                    fixedrange=False,
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.03,
                    xanchor="left",
                    x=0,
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11),
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

        # ==================== 底部三合一專業資訊艙 (對齊專業閃電交易介面，無 4 空白縮排) ====================
        st.markdown("---")
        bot_col1, bot_col2, bot_col3 = st.columns([1.5, 1.2, 1.3])

        with bot_col1:
            st.markdown("##### 📑 委託 / 成交")
            filter_mode = st.radio(
                "委託篩選",
                ["全部", "已成交", "持倉中", "風控攔截"],
                index=0,
                horizontal=True,
                label_visibility="collapsed",
                key="order_filter_radio",
            )
            order_rows = []
            if filter_mode in ["全部", "持倉中"] and pt.position is not None:
                p_dir = "買" if pt.position.direction == 1 else "賣"
                order_rows.append({
                    "時間": pt.position.entry_time[-8:] if len(pt.position.entry_time) >= 8 else pt.position.entry_time,
                    "買賣": p_dir,
                    "口數": pt.position.contracts,
                    "成交": f"{pt.position.contracts} @ {pt.position.entry_price:,.0f}",
                    "條件": "IOC",
                    "狀態": "🟢 持倉中",
                })
                if pt.position.take_profit_price:
                    tp_dir = "賣" if pt.position.direction == 1 else "買"
                    order_rows.append({
                        "時間": pt.position.entry_time[-8:] if len(pt.position.entry_time) >= 8 else "--",
                        "買賣": tp_dir,
                        "口數": pt.position.contracts,
                        "成交": f"停利 @ {pt.position.take_profit_price:,.0f}",
                        "條件": "ROD",
                        "狀態": "⏳ 自動停利掛單",
                    })

            if filter_mode in ["全部", "已成交"]:
                for tr in reversed(pt.closed_trades[-12:] if pt.closed_trades else []):
                    exit_dir = "賣" if tr.direction == 1 else "買"
                    entry_dir = "買" if tr.direction == 1 else "賣"
                    order_rows.append({
                        "時間": tr.exit_time[-8:] if len(tr.exit_time) >= 8 else "--",
                        "買賣": exit_dir,
                        "口數": tr.contracts,
                        "成交": f"{tr.contracts} @ {tr.exit_price:,.0f} ({tr.net_pnl:+,.0f}元)",
                        "條件": "IOC",
                        "狀態": "✅ 全部成交",
                    })
                    order_rows.append({
                        "時間": tr.entry_time[-8:] if len(tr.entry_time) >= 8 else "--",
                        "買賣": entry_dir,
                        "口數": tr.contracts,
                        "成交": f"{tr.contracts} @ {tr.entry_price:,.0f}",
                        "條件": "IOC",
                        "狀態": "✅ 全部成交",
                    })

            if filter_mode in ["全部", "風控攔截"]:
                for l in reversed(pt.audit_logs[-15:] if pt.audit_logs else []):
                    if l.event_type in ["RISK_BREACH", "KILL_SWITCH"]:
                        order_rows.append({
                            "時間": l.timestamp[-8:],
                            "買賣": "系統",
                            "口數": 1,
                            "成交": f"{l.price:,.0f}" if l.price else "--",
                            "條件": "FOK",
                            "狀態": "🛑 風控攔截",
                        })

            if order_rows:
                st.dataframe(pd.DataFrame(order_rows), width="stretch", height=225, hide_index=True)
            else:
                st.caption(f"目前無『{filter_mode}』相關紀錄")

        with bot_col2:
            st.markdown("##### ⚡ 成交明細 (Time & Sales)")
            out_r = mkt_stats.get("outer_ratio", 50.0)
            in_r = mkt_stats.get("inner_ratio", 50.0)
            ratio_bar_html = (
                '<div style="margin-bottom: 6px; font-size: 11.5px; font-family: monospace;">'
                '<div style="display: flex; justify-content: space-between; margin-bottom: 2px;">'
                f'<span style="color: #ff5252; font-weight: 700;">外盤買進 {out_r:.0f}%</span>'
                f'<span style="color: #00e676; font-weight: 700;">內盤賣出 {in_r:.0f}%</span>'
                '</div>'
                '<div style="width: 100%; height: 6px; background: #00e676; border-radius: 3px; overflow: hidden;">'
                f'<div style="width: {out_r}%; height: 100%; background: #ff5252;"></div>'
                '</div>'
                '</div>'
            )
            recent_ticks = feed.get_recent_ticks()
            if recent_ticks:
                t_list_html = []
                for tk in reversed(recent_ticks[-15:]):
                    t_ms = tk.timestamp.strftime("%H:%M:%S") + f".{tk.timestamp.microsecond // 1000:03d}"
                    tk_side = getattr(tk, "side", 0)
                    t_color = "#ff5252" if tk_side == 1 else ("#00e676" if tk_side == -1 else "#ffffff")
                    side_tag = "外" if tk_side == 1 else ("內" if tk_side == -1 else "—")
                    t_list_html.append(
                        '<div style="display: flex; justify-content: space-between; padding: 2px 4px; font-family: monospace; font-size: 12.5px; border-bottom: 1px solid rgba(255,255,255,0.05);">'
                        f'<span style="color: #787b86;">{t_ms}</span>'
                        f'<span style="color: {t_color}; font-weight: 700;">{tk.price:,.1f} <small>({side_tag})</small></span>'
                        f'<span style="color: {t_color}; font-weight: 700;">{tk.volume}</span>'
                        '</div>'
                    )
                all_t_html = "".join(t_list_html)
                ts_box_html = (
                    f'{ratio_bar_html}'
                    f'<div style="background: #131722; border: 1px solid #2a2e39; border-radius: 6px; padding: 6px; height: 195px; overflow-y: auto;">{all_t_html}</div>'
                )
                st.markdown(ts_box_html, unsafe_allow_html=True)
            else:
                st.markdown(ratio_bar_html, unsafe_allow_html=True)
                st.caption("等待即時 Tick 成交中...")

        with bot_col3:
            st.markdown("##### 🖥️ 系統訊息 (Execution Log)")
            sys_logs = []
            if pt.position is not None and pt.position.take_profit_price:
                tp_pts = abs(pt.position.take_profit_price - pt.position.entry_price)
                tp_side = "賣" if pt.position.direction == 1 else "買"
                sys_logs.append(
                    '<div style="padding: 2px 0; border-bottom: 1px solid rgba(255,255,255,0.04); font-size: 12px; font-family: monospace;">'
                    f'<span style="color: #787b86;">{pt.position.entry_time[-8:]}</span> '
                    '<span style="color: #f0b90b; font-weight: 700;">[自動停利]</span> '
                    f'<span style="color: #cfd8dc;">掛出 {tp_side} {pt.position.contracts} 口 @ {pt.position.take_profit_price:,.0f} (成交價 +{tp_pts:.0f}點)</span>'
                    '</div>'
                )
            if pt.audit_logs:
                for l in reversed(pt.audit_logs[-20:]):
                    t_short = l.timestamp[-8:]
                    ev = l.event_type
                    ev_zh = "模擬成交" if ev == "ORDER_FILL" else ("風控停損" if ev == "STOP_OUT" else ("風控攔截" if ev == "RISK_BREACH" else ev))
                    ev_color = "#00e676" if ev == "ORDER_FILL" else ("#ff5252" if ev in ["RISK_BREACH", "STOP_OUT"] else "#40c4ff")
                    sys_logs.append(
                        '<div style="padding: 2px 0; border-bottom: 1px solid rgba(255,255,255,0.04); font-size: 12px; font-family: monospace;">'
                        f'<span style="color: #787b86;">{t_short}</span> '
                        f'<span style="color: {ev_color}; font-weight: 700;">[{ev_zh}]</span> '
                        f'<span style="color: #cfd8dc;">{l.message}</span>'
                        '</div>'
                    )
            else:
                sys_logs.append('<div style="color: #787b86; font-size: 12px; font-family: monospace;">[系統初始化完成，已載入台新與期交所即時 K 棒...]</div>')

            all_sys_html = "".join(sys_logs)
            sys_box_html = f'<div style="background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 8px; height: 225px; overflow-y: auto;">{all_sys_html}</div>'
            st.markdown(sys_box_html, unsafe_allow_html=True)

    _render_unified_war_room()

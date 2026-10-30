# src/ui/dashboard.py
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import polars as pl
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.schema.strategy import StrategySchema
from src.core.engine import ContractSpec, EventDrivenEngine, BacktestReport
from src.core.validator import CredibilityValidator, ValidationResult
from src.core.paper_trader import PaperTradingEngine, RiskConfig
from src.ai.strategy_copilot import AIStrategyCopilot
from src.ai.confidence_engine import ConfidenceEngine
from src.ai.ensemble import MultiModelEnsemble
from src.broker.taishin_quote import TaishinQuoteFeed


def load_taishin_default_config() -> dict:
    paths = [
        Path("config/taishin_config.json"),
        Path(r"D:\一般資料\工具\taishin_dashboard\taishin_config.json")
    ]
    for p in paths:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return {"symbol": "2330", "sol_user_id": "", "sol_password": ""}


def mask_id_number(raw_id: str) -> str:
    raw_id = (raw_id or "").strip()
    if len(raw_id) <= 3:
        return raw_id
    return raw_id[:3] + "*" * (len(raw_id) - 3)


st.set_page_config(
    page_title="AI 自動下單與回測系統 - 策略儀表板",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 套用深色專業量化交易主題 CSS (含指標卡片自動字體縮放與防截斷)
st.markdown("""
<style>
    .metric-card {
        background-color: #1e222d;
        border-radius: 8px;
        padding: 14px;
        border: 1px solid #2a2e39;
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 13px;
        color: #787b86;
        margin-bottom: 4px;
        font-weight: 500;
    }
    .metric-value {
        font-size: 18px;
        font-weight: 700;
        color: #d1d4dc;
    }
    .metric-profit {
        color: #089981 !important;
    }
    .metric-loss {
        color: #f23645 !important;
    }
    /* 解決 st.metric 字型太大導致文字被截斷 (...) 的問題，強制內層 div 允許換行與自适应字級 */
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] > div,
    [data-testid="stMetricValue"] * {
        font-size: clamp(0.92rem, 1.15vw, 1.28rem) !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        word-break: break-word !important;
        overflow-wrap: anywhere !important;
        line-height: 1.35 !important;
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] > div,
    [data-testid="stMetricLabel"] p {
        font-size: 0.84rem !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        line-height: 1.25 !important;
    }
    [data-testid="stMetricDelta"],
    [data-testid="stMetricDelta"] > div,
    [data-testid="stMetricDelta"] * {
        font-size: 0.76rem !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
    }
    h1 {
        font-size: 1.6rem !important;
        line-height: 1.3 !important;
    }
    h2, h3 {
        font-size: 1.22rem !important;
        line-height: 1.3 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        background-color: #131722;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_contracts() -> dict:
    contracts_file = Path("config/contracts.json")
    if contracts_file.exists():
        with open(contracts_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_strategy_from_file(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


DEFAULT_WATCHLIST = [
    "2330 (台積電 - 股票實機連線驗證)",
    "2317 (鴻海 - 股票實機連線驗證)",
    "2454 (聯發科 - 股票實機連線驗證)",
    "0050 (元大台灣50 - ETF實機連線驗證)",
    "TMFR1 (微台近月 - 期貨保留待開通)",
    "TXFR1 (大台近月 - 期貨保留待開通)",
    "MTXR1 (小台近月 - 期貨保留待開通)",
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
        return False, "請輸入商品代碼（例如 2317、2603 或 TXFR1）", load_watchlist(path)
    items = load_watchlist(path)
    existing_codes = {item.split(" ")[0].upper() for item in items}
    if clean_code in existing_codes:
        return False, f"商品代碼 `{clean_code}` 已經在清單中了！", items

    note = (name_note or "").strip()
    entry = f"{clean_code} ({note})" if note else clean_code
    items.insert(0, entry)
    save_watchlist(items, path)
    return True, f"已成功新增 `{entry}` 至訂閱清單！", items


def remove_symbol_from_watchlist(target_entry: str, path: str | Path = "config/watchlist.json") -> tuple[bool, str, list[str]]:
    items = load_watchlist(path)
    if len(items) <= 1:
        return False, "訂閱清單至少需保留 1 項商品，無法全部刪除。", items
    if target_entry not in items:
        return False, f"找不到 `{target_entry}`", items
    items.remove(target_entry)
    save_watchlist(items, path)
    return True, f"已從訂閱清單移除 `{target_entry}`。", items


@st.cache_resource
def get_global_taishin_feed() -> TaishinQuoteFeed:
    """跨瀏覽器分頁與 Rerun 共用唯一台新連線實例，徹底防止 Too Many Connections"""
    return TaishinQuoteFeed(auto_load_store=True)


def main():
    st.sidebar.title("⚙️ 策略與回測設定")

    # 取得全域共用台新 API 連線單例 (原位升級類別方法以保留現有連線，防止重連造成 Too Many Connections)
    feed = get_global_taishin_feed()
    if getattr(feed, "VERSION", None) != TaishinQuoteFeed.VERSION:
        feed.__class__ = TaishinQuoteFeed
        feed.VERSION = TaishinQuoteFeed.VERSION
    if "載入中" in feed.status_msg and len(feed.kbars_1m) > 0:
        feed.ta_rcv_done = True
        feed.status_msg = f"🟢 已載入完成 ({feed.symbol}: 共 {len(feed.kbars_1m)} 根 K 棒)"
    st.session_state["taishin_feed"] = feed

    default_cfg = load_taishin_default_config()
    if "actual_sol_user_id" not in st.session_state:
        st.session_state["actual_sol_user_id"] = default_cfg.get("sol_user_id", "")
    if "show_plain_id" not in st.session_state:
        st.session_state["show_plain_id"] = False

    # 若本地尚無 K 線快取且尚未連線，才自動觸發首次背景連線；若已有本地 K 線則直接秒開不阻塞畫面
    if not st.session_state.get("auto_connect_attempted", False):
        st.session_state["auto_connect_attempted"] = True
        uid = st.session_state["actual_sol_user_id"]
        pwd = default_cfg.get("sol_password", "")
        if uid and pwd and not feed.connected and not feed.is_running and len(feed.kbars_1m) == 0:
            feed.start(uid, pwd, "2330")

    # 1. 商品規格設定
    contracts = load_contracts()
    symbol_options = list(contracts.keys()) if contracts else ["TMF", "MTX", "TXF"]
    selected_symbol = st.sidebar.selectbox("交易標的", symbol_options, index=0)

    contract_info = contracts.get(selected_symbol, {
        "name": "微型臺指", "point_value": 10.0, "tick_size": 1.0,
        "tax_rate": 0.00002, "default_commission": 12.0
    })
    contract = ContractSpec(
        symbol=selected_symbol,
        name=contract_info.get("name", selected_symbol),
        exchange=contract_info.get("exchange", "TAIFEX"),
        point_value=float(contract_info.get("point_value", 10.0)),
        tick_size=float(contract_info.get("tick_size", 1.0)),
        tax_rate=float(contract_info.get("tax_rate", 0.00002)),
        default_commission=float(contract_info.get("default_commission", 12.0)),
    )

    st.sidebar.caption(f"標的規格：{contract.name} | 1點 = {contract.point_value}元 | 手續費 = {contract.default_commission}元")

    # 2. 策略選擇
    strategy_files = list(Path("strategies").glob("*.json"))
    strategy_names = [f.name for f in strategy_files] if strategy_files else ["自訂策略"]
    selected_strategy_name = st.sidebar.selectbox("策略範本", strategy_names, index=0)

    initial_capital = st.sidebar.number_input("初始資金 (TWD)", min_value=10000.0, value=100000.0, step=10000.0)
    bt_timeframe = st.sidebar.selectbox("台新 API K 線週期", ["1m", "5m", "15m", "30m", "60m"], index=1)

    # 載入策略物件
    if strategy_files and selected_strategy_name != "自訂策略":
        strat_path = Path("strategies") / selected_strategy_name
        strategy_dict = load_strategy_from_file(strat_path)
    else:
        strategy_dict = {
            "id": "ma_cross_demo",
            "version": 1,
            "symbol": selected_symbol,
            "timeframe": bt_timeframe,
            "session": "day_and_night",
            "indicators": {
                "fast_ma": {"type": "SMA", "source": "close", "length": 5},
                "slow_ma": {"type": "SMA", "source": "close", "length": 20},
            },
            "entry_long": {"cross_above": ["fast_ma", "slow_ma"]},
            "entry_short": {"cross_below": ["fast_ma", "slow_ma"]},
            "exit": {
                "stop_loss": {"type": "fixed_points", "value": 60},
                "take_profit": {"type": "fixed_points", "value": 150},
                "trailing_stop": None,
            },
            "execution": {"signal_on": "bar_close", "fill_at": "next_bar_open", "order_type": "range_market", "slippage_points": 1.0},
            "position_sizing": {"type": "fixed", "contracts": 1},
        }

    strategy_dict["timeframe"] = bt_timeframe

    # 允許側邊欄微調均線參數與 AI 信心度門檻
    st.sidebar.subheader("🎛️ 參數快速微調")
    fast_len = st.sidebar.number_input("短均線週期 (Fast)", min_value=2, max_value=50, value=int(st.session_state.get("opt_fast_len", 5)))
    slow_len = st.sidebar.number_input("長均線週期 (Slow)", min_value=5, max_value=120, value=int(st.session_state.get("opt_slow_len", 20)))
    stop_loss_pts = st.sidebar.number_input("停損點數 (Stop Loss)", min_value=10, max_value=300, value=int(st.session_state.get("opt_sl_pts", 60)))

    st.sidebar.subheader("🧠 AI 信心度與狀態濾網")
    enable_conf_filter = st.sidebar.checkbox("啟用 AI 信心度閾值過濾", value=bool(st.session_state.get("opt_conf_enabled", False)))
    conf_threshold = st.sidebar.slider("進場信心度門檻 (%)", min_value=50, max_value=85, value=int(st.session_state.get("opt_conf_th", 60)), step=1)
    regime_adaptive = st.sidebar.checkbox("依市場狀態 (趨勢/盤整) 動態調節門檻", value=True)

    # 策略覆寫
    if "ma5" in strategy_dict.get("indicators", {}):
        strategy_dict["indicators"]["ma5"]["length"] = int(fast_len)
    elif "fast_ma" in strategy_dict.get("indicators", {}):
        strategy_dict["indicators"]["fast_ma"]["length"] = int(fast_len)

    if "ma20" in strategy_dict.get("indicators", {}):
        strategy_dict["indicators"]["ma20"]["length"] = int(slow_len)
    elif "slow_ma" in strategy_dict.get("indicators", {}):
        strategy_dict["indicators"]["slow_ma"]["length"] = int(slow_len)

    if "exit" in strategy_dict and "stop_loss" in strategy_dict["exit"] and strategy_dict["exit"]["stop_loss"]:
        strategy_dict["exit"]["stop_loss"]["value"] = float(stop_loss_pts)

    strategy_dict["confidence_filter"] = {
        "enabled": bool(enable_conf_filter),
        "threshold": float(conf_threshold),
        "regime_adaptive": bool(regime_adaptive),
    }

    try:
        strategy = StrategySchema.model_validate(strategy_dict)
    except Exception as e:
        st.error(f"❌ 策略驗證失敗: {e}")
        return

    # K 線唯一來源：台新 API (TaishinQuoteFeed)
    df_bars = feed.get_bars(timeframe=bt_timeframe, session=strategy.session)

    # 頁面標題
    st.title("🛡️ AI 自然語言自動下單與回測系統 (MVP)")
    st.markdown(f"**策略代號**: `{strategy.id}` (v{strategy.version}) | **標的**: `{contract.name} ({contract.symbol})` | **K 線來源**: `台新 API ({feed.symbol}, 共 {df_bars.height} 根 {bt_timeframe} K 棒)`")

    # 導覽分頁列 (採用 st.radio + Session State 持久化，解決 rerun 跳回第一頁問題)
    nav_tabs = [
        "📊 績效報表 (回測結果)",
        "🤖 AI 信心度與多模型戰情室",
        "🛡️ 可信度驗證報告 (門檻檢核)",
        "📡 即時行情串流 (零風險純報價)",
        "🎯 紙上交易與風控中心",
        "📜 Strategy Schema 結構",
    ]
    if "active_nav" not in st.session_state:
        st.session_state["active_nav"] = nav_tabs[0]

    selected_nav = st.radio(
        "導覽分頁",
        nav_tabs,
        index=nav_tabs.index(st.session_state["active_nav"]) if st.session_state["active_nav"] in nav_tabs else 0,
        horizontal=True,
        key="active_nav"
    )

    if selected_nav == nav_tabs[0]:
        if df_bars.height == 0:
            st.warning("⚠️ 目前尚未載入台新 API K 線資料。請切換至「📡 即時行情串流」分頁點擊「🚀 連線行情伺服器」以從台新 API 取得真實 K 線數據。")
        else:
            engine = EventDrivenEngine(strategy, contract, initial_capital)
            report = engine.run_backtest(df_bars, cost_multiplier=1.0)

            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("總淨利 (Net Profit)", f"{report.net_profit:,.0f} 元", delta=f"{report.net_profit:,.0f}")
            c2.metric("勝率 (Win Rate)", f"{report.win_rate:.1f} %")
            c3.metric("獲利因子 (Profit Factor)", f"{report.profit_factor:.2f}")
            c4.metric("最大回撤 (MDD)", f"{report.max_drawdown:,.0f} 元", f"-{report.max_drawdown_pct:.1f}%", delta_color="inverse")
            c5.metric("交易次數 (Trades)", f"{report.total_trades} 筆")
            c6.metric("每筆期望值 (Expectancy)", f"{report.expectancy:,.0f} 元")

            fig_equity = go.Figure()
            eq_x = list(range(len(report.equity_curve)))
            fig_equity.add_trace(go.Scatter(
                x=eq_x, y=report.equity_curve,
                mode='lines',
                name='策略權益 (Strategy Equity)',
                line=dict(color='#2962FF', width=2),
                fill='tozeroy',
                fillcolor='rgba(41, 98, 255, 0.08)'
            ))
            # 買進持有基準比較 (Buy & Hold Benchmark)
            if df_bars.height > 0:
                first_close = float(df_bars["close"][0])
                bh_curve = [
                    initial_capital + (float(c) - first_close) * contract.point_value * strategy.position_sizing.contracts
                    for c in df_bars["close"].to_list()
                ]
                fig_equity.add_trace(go.Scatter(
                    x=eq_x, y=bh_curve,
                    mode='lines',
                    name='買進持有基準 (Buy & Hold)',
                    line=dict(color='#FF9800', width=1.5, dash='dot'),
                ))
            fig_equity.update_layout(
                title=f"資金曲線 vs 買進持有基準 (基於台新 API 真實 {bt_timeframe} K 線，共 {df_bars.height} 根)",
                xaxis_title="K棒序號",
                yaxis_title="帳戶權益 (TWD)",
                template="plotly_dark",
                height=360,
                margin=dict(l=20, r=20, t=40, b=20),
                uirevision=f"eq_{strategy.id}_{feed.symbol}_{bt_timeframe}",
            )
            st.plotly_chart(fig_equity, width="stretch", config={"scrollZoom": True, "displayModeBar": True})

            # 水下回撤圖 (Underwater Drawdown Chart)
            peaks = np.maximum.accumulate(report.equity_curve) if report.equity_curve else np.array([initial_capital])
            dd_series = [float(eq - pk) for eq, pk in zip(report.equity_curve, peaks)]
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=eq_x, y=dd_series,
                mode='lines',
                name='水下回撤 (Drawdown)',
                line=dict(color='#F23645', width=1.5),
                fill='tozeroy',
                fillcolor='rgba(242, 54, 69, 0.2)'
            ))
            fig_dd.update_layout(
                title="水下回撤圖 (Underwater Drawdown Curve)",
                xaxis_title="K棒序號",
                yaxis_title="回撤金額 (TWD)",
                template="plotly_dark",
                height=220,
                margin=dict(l=20, r=20, t=35, b=20),
                uirevision=f"dd_{strategy.id}_{feed.symbol}_{bt_timeframe}",
            )
            st.plotly_chart(fig_dd, width="stretch", config={"scrollZoom": True, "displayModeBar": True})

            if report.trades:
                st.subheader("📋 交易明細表")
                trades_data = []
                for t in report.trades:
                    trades_data.append({
                        "方向": "多 (Long)" if t.direction == 1 else "空 (Short)",
                        "口數": t.contracts,
                        "進場K棒": t.entry_idx,
                        "出場K棒": t.exit_idx,
                        "進場價": t.entry_price,
                        "出場價": t.exit_price,
                        "手續費": round(t.commission, 1),
                        "期交稅": round(t.tax, 1),
                        "淨損益": round(t.net_pnl, 1),
                        "出場原因": t.exit_reason,
                    })
                df_trades = pd.DataFrame(trades_data)
                st.dataframe(df_trades, width="stretch", height=220)

    elif selected_nav == nav_tabs[1]:
        st.subheader("🤖 AI 信心度推論引擎、市場狀態辨識與多模型並行戰情室")
        st.caption("完整實作 AITrade 5 步驟量化流程：① K 線收集 → ② 多維度特徵工程 → ③ 模型推論信心度 (1~99%) → ④ 閾值 + 市場狀態動態調節 → ⑤ 多模型融合 (投票/平均/動態加權)。")

        if df_bars.height < 25:
            st.warning("⚠️ 請先連線台新 API 載入至少 25 根 K 線資料，以啟動 AI 特徵工程與多模型並行評估。")
        else:
            snap = ConfidenceEngine.get_latest_snapshot(
                df_bars, base_threshold=float(conf_threshold), regime_adaptive=bool(regime_adaptive)
            )
            e_col1, e_col2, e_col3 = st.columns([1.2, 1.2, 1.6])
            with e_col1:
                ens_method_label = st.selectbox(
                    "多模型融合演算法 (Ensemble Method)",
                    [
                        "dynamic_weighted (依近期回測損益動態加權)",
                        "average (所有啟用模型信心度取平均)",
                        "voting (多數決投票制)"
                    ],
                    index=0,
                )
                ens_method = ens_method_label.split(" ")[0]
            with e_col2:
                if snap:
                    st.metric(
                        "當前市場狀態 (Regime Detection)",
                        snap.regime.label_zh,
                        delta=f"ADX={snap.regime.adx_val} | 門檻調節 {snap.regime.threshold_offset:+.0f}%",
                    )
            with e_col3:
                if snap:
                    st.metric(
                        "設定門檻 vs 動態有效門檻",
                        f"{snap.effective_threshold:.1f} %",
                        delta=f"原始設定: {conf_threshold}% ({'已啟用狀態調節' if regime_adaptive else '固定門檻'})",
                    )

            ens_dec = MultiModelEnsemble.evaluate_ensemble(
                df_bars,
                contract=contract,
                initial_capital=initial_capital,
                method=ens_method,
                threshold=float(conf_threshold),
                regime_adaptive=bool(regime_adaptive),
            )

            # 頂部即時決策卡片 (仿照 AITrade Live Strategy Logic)
            st.markdown("#### ⚡ 即時 AI 信心度與策略規則決策 (Live Strategy Logic)")
            d_c1, d_c2, d_c3, d_c4 = st.columns([1.0, 1.0, 1.1, 1.3])
            d_c1.metric("📈 融合做多信心度", f"{ens_dec.ensemble_long_conf:.1f} %", delta="達標 (>有效門檻)" if ens_dec.ensemble_long_conf >= ens_dec.effective_threshold else "未過門檻")
            d_c2.metric("📉 融合做空信心度", f"{ens_dec.ensemble_short_conf:.1f} %", delta="達標 (>有效門檻)" if ens_dec.ensemble_short_conf >= ens_dec.effective_threshold else "未過門檻")
            d_c3.metric("🗳️ 多模型投票分布", f"多 {ens_dec.bull_votes} / 空 {ens_dec.bear_votes} / 平 {ens_dec.neutral_votes}", delta=f"共 {ens_dec.total_active_models} 組並行模型")
            act_str = (
                "🟢 觸發買進 (LONG)" if ens_dec.final_action == 1
                else ("🔴 觸發放空 (SHORT)" if ens_dec.final_action == -1 else "⚪ 觀望持平 (未過閾值)")
            )
            d_c4.metric("🎯 最終融合決策指令", act_str)

            st.code(
                f"// Live AI Strategy Rule ({ens_dec.regime_label})\n"
                f"當 ( 融合信心度 [多={ens_dec.ensemble_long_conf:.1f}%, 空={ens_dec.ensemble_short_conf:.1f}%] >= 有效門檻 {ens_dec.effective_threshold:.1f}% ) {{\n"
                f"    執行動作 = \"{act_str}\";\n"
                f"}} 否則 {{\n"
                f"    執行動作 = \"保持觀望 (機械化紀律執行，未達門檻絕不盲目出手)\";\n"
                f"}}",
                language="javascript",
            )

            # 多模型即時監控牆 (依損益自動排序)
            st.markdown("#### 🏆 多模型並行監控排行榜 (依台新真實 K 線回測淨利自動排序)")
            model_rows = []
            for idx_m, m in enumerate(ens_dec.models, start=1):
                sig_txt = "🟢 看多" if m.latest_signal == 1 else ("🔴 看空" if m.latest_signal == -1 else "⚪ 觀望")
                model_rows.append({
                    "排名": f"#{idx_m}",
                    "模型代號": m.model_id,
                    "策略架構說明": m.display_name,
                    "最新訊號": sig_txt,
                    "做多信心度 (%)": m.long_confidence,
                    "做空信心度 (%)": m.short_confidence,
                    "動態權重": f"{m.dynamic_weight * 100:.1f}%",
                    "回測總淨利 (TWD)": round(m.report.net_profit, 0),
                    "勝率 (%)": round(m.report.win_rate, 1),
                    "獲利因子 (PF)": m.report.profit_factor,
                    "最大回撤 (MDD)": round(m.report.max_drawdown, 0),
                    "交易筆數": m.report.total_trades,
                })
            st.dataframe(pd.DataFrame(model_rows), width="stretch", height=270)

            col_feat, col_th = st.columns(2)
            with col_feat:
                st.markdown("#### 🧬 最新 K 棒特徵工程因子拆解 (Feature Contributions)")
                if snap and snap.feature_contributions:
                    feat_items = [
                        {"特徵維度": k, "多空傾向強度 (-100~+100)": v}
                        for k, v in snap.feature_contributions.items()
                    ]
                    st.dataframe(pd.DataFrame(feat_items), width="stretch", height=220)
            with col_th:
                st.markdown("#### 🎚️ 信心度閾值高低取捨實測表 (Threshold Trade-off)")
                th_rows = MultiModelEnsemble.scan_threshold_tradeoff(
                    df_bars, strategy, contract, initial_capital, regime_adaptive=bool(regime_adaptive)
                )
                th_df = pd.DataFrame([
                    {
                        "門檻設定": f"{r.threshold:.0f}%",
                        "訊號次數": r.total_trades,
                        "單筆勝率 (%)": round(r.win_rate, 1),
                        "總淨利 (TWD)": round(r.net_profit, 0),
                        "MDD (TWD)": round(r.max_drawdown, 0),
                        "適合情境": r.suitability_label,
                    }
                    for r in th_rows
                ])
                st.dataframe(th_df, width="stretch", height=220)

            # 三步驟模型參數優化精靈 (Parameter Optimization Wizard)
            st.markdown("---")
            st.markdown("#### 🪄 三步驟模型參數優化精靈 (結合樣本外 OOS 防過擬合評分)")
            w_c1, w_c2 = st.columns([1.2, 2.8])
            with w_c1:
                run_wiz = st.button("🚀 自動尋找最佳參數與信心度閾值", width="stretch", type="primary")
            if run_wiz:
                fast_k = "fast_ma" if "fast_ma" in strategy.indicators else ("ma5" if "ma5" in strategy.indicators else list(strategy.indicators.keys())[0])
                slow_k = "slow_ma" if "slow_ma" in strategy.indicators else ("ma20" if "ma20" in strategy.indicators else list(strategy.indicators.keys())[-1])
                with st.spinner("正在多維度迭代搜尋最佳指標期數、停損點數與信心度閾值..."):
                    wiz_res = CredibilityValidator(strategy, contract, initial_capital).run_parameter_wizard(
                        df_bars, ind_x=fast_k, ind_y=slow_k, top_k=5
                    )
                    st.session_state["wizard_candidates"] = wiz_res

            if "wizard_candidates" in st.session_state and st.session_state["wizard_candidates"]:
                cands = st.session_state["wizard_candidates"]
                cand_rows = [
                    {
                        "推薦序": f"Top {i+1}",
                        "短週期": c.fast_length,
                        "長週期": c.slow_length,
                        "停損點數": c.stop_loss_pts,
                        "信心度門檻 (%)": f"{c.confidence_threshold:.0f}%" if c.confidence_threshold > 0 else "不濾網",
                        "IS 樣本內淨利": c.is_net_profit,
                        "OOS 樣本外淨利": c.oos_net_profit,
                        "總勝率 (%)": c.win_rate,
                        "最大回撤 (MDD)": c.max_drawdown,
                        "穩健度總分": c.score,
                    }
                    for i, c in enumerate(cands)
                ]
                st.dataframe(pd.DataFrame(cand_rows), width="stretch", height=200)
                best_c = cands[0]
                if st.button(f"✅ 一鍵套用 Top 1 最佳參數 (短={best_c.fast_length}, 長={best_c.slow_length}, 停損={best_c.stop_loss_pts:.0f}, 門檻={best_c.confidence_threshold:.0f}%)"):
                    st.session_state["opt_fast_len"] = best_c.fast_length
                    st.session_state["opt_slow_len"] = best_c.slow_length
                    st.session_state["opt_sl_pts"] = int(best_c.stop_loss_pts)
                    st.session_state["opt_conf_enabled"] = best_c.confidence_threshold > 0
                    if best_c.confidence_threshold >= 50:
                        st.session_state["opt_conf_th"] = int(best_c.confidence_threshold)
                    st.rerun()

    elif selected_nav == nav_tabs[2]:
        st.subheader("防過擬合與可信度檢驗")
        st.caption("依據規劃書標準：OOS 期望值為正、雙倍成本壓力測試獲利、熱力圖呈高原狀、蒙地卡羅 95% 信心水準通過，方可進入紙上交易。")

        if df_bars.height < 30:
            st.warning(f"⚠️ 目前台新 API 載入之 {bt_timeframe} K 棒數量為 {df_bars.height} 根（少於 30 根），請先連線台新 API 或切換為較短週期（如 1m / 5m）以進行完整可信度驗證。")
        else:
            validator = CredibilityValidator(strategy, contract, initial_capital)
            fast_key = "fast_ma" if "fast_ma" in strategy.indicators else "ma5"
            slow_key = "slow_ma" if "slow_ma" in strategy.indicators else "ma20"

            with st.spinner("正在以台新 API 真實 K 線進行 IS/OOS、雙倍成本壓力測試、參數高原掃描與 1,000 次蒙地卡羅模擬..."):
                val_res = validator.evaluate_all(
                    df_bars,
                    ind_x=fast_key,
                    x_values=[3, 5, 7, 10],
                    ind_y=slow_key,
                    y_values=[15, 20, 25, 30],
                    mc_iterations=1000
                )

            if val_res.overall_passed:
                st.success("🟢 **【驗證合格 - 准許上線】** 本策略已通過所有防過擬合門檻，可部署進入紙上交易 (Paper Trading)！")
            else:
                st.error("🔴 **【驗證未通過 - 阻擋上線】** 本策略未能通過所有可信度防護檢驗，嚴禁直接部署實盤！")

            for reason in val_res.reasons:
                st.markdown(f"- {reason}")

            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown("#### 1. 樣本內 (IS) vs 樣本外 (OOS) 切分檢驗")
                is_net = val_res.is_report.net_profit
                oos_net = val_res.oos_report.net_profit
                c_is, c_oos = st.columns(2)
                c_is.metric("IS (70% 歷史資料) 淨利", f"{is_net:,.0f} 元")
                c_oos.metric("OOS (30% 未見資料) 淨利", f"{oos_net:,.0f} 元", delta="合格" if val_res.oos_passed else "未達標")

                st.markdown("#### 2. 雙倍成本壓力測試 (Slippage & Commission x2)")
                st.metric("雙倍成本後淨利", f"{val_res.stress_report.net_profit:,.0f} 元",
                          delta="仍有正期望值" if val_res.stress_passed else "加倍後轉虧",
                          delta_color="normal" if val_res.stress_passed else "inverse")

                if val_res.monte_carlo_report:
                    st.markdown("#### 3. 蒙地卡羅 1,000 次模擬結果")
                    mc = val_res.monte_carlo_report
                    mc_c1, mc_c2 = st.columns(2)
                    mc_c1.metric("中位數 MDD", f"{mc.median_mdd:,.0f} 元")
                    mc_c2.metric("95% 信心最差回撤", f"{mc.mdd_95th:,.0f} 元", delta_color="inverse")

            with col_right:
                st.markdown("#### 4. 參數優化熱力圖 (高原 vs 尖峰)")
                heat_arr = np.array(val_res.heatmap_matrix)
                x_labels = [f"{fast_key}={x}" for x in [3, 5, 7, 10]]
                y_labels = [f"{slow_key}={y}" for y in [15, 20, 25, 30]]

                fig_heat = px.imshow(
                    heat_arr,
                    x=x_labels,
                    y=y_labels,
                    color_continuous_scale="RdYlGn",
                    title=f"雙均線淨利熱力圖 ({'高原特徵合格' if val_res.heatmap_plateau_passed else '孤立尖峰/過擬合'})"
                )
                fig_heat.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_heat, width="stretch")

            if val_res.monte_carlo_report and val_res.monte_carlo_report.mdd_distribution:
                st.markdown("#### 5. 蒙地卡羅最大回撤 (MDD) 機率分布圖")
                fig_mc = px.histogram(
                    x=val_res.monte_carlo_report.mdd_distribution,
                    nbins=30,
                    labels={"x": "MDD 金額 (TWD)"},
                    title="1,000 次隨機重排交易順序之 MDD 分布 (含 95% 信心臨界線)",
                    color_discrete_sequence=['#00BCD4']
                )
                fig_mc.add_vline(x=val_res.monte_carlo_report.mdd_95th, line_dash="dash", line_color="#FF5252",
                                 annotation_text=f"95% VaR: {val_res.monte_carlo_report.mdd_95th:,.0f} 元")
                fig_mc.update_layout(template="plotly_dark", height=280, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_mc, width="stretch")

            if val_res.walk_forward_report and val_res.walk_forward_report.folds:
                wf = val_res.walk_forward_report
                st.markdown(f"#### 6. 滾動視窗驗證 (Walk-Forward Analysis) — 跨期一致性: `{wf.consistency_ratio * 100:.0f}%` | 總樣本外損益: `{wf.total_oos_profit:,.0f} 元`")
                wf_rows = [
                    {
                        "滾動視窗折數": f"Fold #{f.fold_index}",
                        "訓練集 K 棒數": f.train_bars,
                        "測試集 K 棒數": f.test_bars,
                        "訓練集淨利 (TWD)": round(f.train_net_profit, 0),
                        "未見測試集淨利 (TWD)": round(f.test_net_profit, 0),
                        "測試集勝率 (%)": round(f.test_win_rate, 1),
                        "測試集交易數": f.test_trades,
                        "單折判定": "🟢 合格" if f.passed else "🔴 衰退",
                    }
                    for f in wf.folds
                ]
                st.dataframe(pd.DataFrame(wf_rows), width="stretch", height=160)

    elif selected_nav == nav_tabs[3]:
        st.subheader("📡 即時行情串流 (零風險純報價接收)")
        st.caption("透過台新 API (Sol_D 即時報價 + TechAnalysis K 線 API) 接收真實 K 線與即時 Tick。此處完全不發送任何委託下單。")
        st.info("💡 **功能狀態提示**：系統所有 K 線資料唯一來源為台新官方 API。期貨行情 API（微台/大台）目前待開通，底層邏輯與介面已保留；現階段可選擇「**2330 (台積電)**」進行實機連線與真實 K 線接收。")

        ts_col1, ts_col2, ts_col3 = st.columns(3)
        with ts_col1:
            show_plain = st.checkbox("👁️ 顯示完整身分證號", value=st.session_state["show_plain_id"], key="toggle_show_id")
            st.session_state["show_plain_id"] = show_plain

            current_raw = st.session_state["actual_sol_user_id"]
            display_val = current_raw if show_plain else mask_id_number(current_raw)

            user_input = st.text_input("台新身分證字號", value=display_val)
            if "*" not in user_input and user_input.strip() != "":
                st.session_state["actual_sol_user_id"] = user_input.strip()
                actual_user = user_input.strip()
            else:
                actual_user = st.session_state["actual_sol_user_id"]

        with ts_col2:
            st.write("")
            st.write("")
            ts_pwd = st.text_input("台新密碼", value=default_cfg.get("sol_password", ""), type="password")
        with ts_col3:
            st.write("")
            st.write("")
            watchlist_items = load_watchlist()
            default_idx = 0
            if "selected_watchlist_item" in st.session_state and st.session_state["selected_watchlist_item"] in watchlist_items:
                default_idx = watchlist_items.index(st.session_state["selected_watchlist_item"])
            ts_symbol = st.selectbox(
                "訂閱商品 (切換即自動更新行情與K線)",
                watchlist_items,
                index=default_idx,
                key="watchlist_selectbox",
            )
            st.session_state["selected_watchlist_item"] = ts_symbol
            clean_sym = ts_symbol.split(" ")[0]

            # 當使用者在下拉選單更換商品時，自動熱切換訂閱標的並載入新商品 K 線與成交價
            if clean_sym and feed.symbol != clean_sym:
                feed.switch_symbol(clean_sym, actual_user, ts_pwd)
                if len(feed.kbars_1m) == 0:
                    with st.spinner(f"正在從台新伺服器下載 {clean_sym} K 線..."):
                        for _ in range(12):
                            time.sleep(0.4)
                            if feed.ta_rcv_done or len(feed.kbars_1m) > 0 or "失敗" in feed.status_msg or "異常" in feed.status_msg:
                                break
                st.rerun()

        with st.expander("➕ / 🗑️ 增減「訂閱商品」清單 (直接新增股票或期貨代碼並自動存檔)"):
            wl_c1, wl_c2, wl_c3, wl_c4 = st.columns([1.2, 1.5, 1.0, 1.1])
            with wl_c1:
                new_sym_code = st.text_input("新增商品代碼", placeholder="例如: 2603 或 3231", key="in_new_sym_code")
            with wl_c2:
                new_sym_note = st.text_input("商品名稱/備註 (選填)", placeholder="例如: 長榮 - 股票連線", key="in_new_sym_note")
            with wl_c3:
                st.write("")
                if st.button("➕ 新增至清單", width="stretch", type="primary"):
                    ok_add, msg_add, updated_list = add_symbol_to_watchlist(new_sym_code, new_sym_note)
                    if ok_add:
                        st.session_state["selected_watchlist_item"] = updated_list[0]
                        st.toast(msg_add)
                        st.rerun()
                    else:
                        st.warning(msg_add)
            with wl_c4:
                st.write("")
                if st.button(f"🗑️ 移除目前選取 ({clean_sym})", width="stretch"):
                    ok_rm, msg_rm, updated_list = remove_symbol_from_watchlist(ts_symbol)
                    if ok_rm:
                        st.session_state["selected_watchlist_item"] = updated_list[0]
                        st.toast(msg_rm)
                        st.rerun()
                    else:
                        st.warning(msg_rm)
            st.caption("💡 提示：清單儲存於 `config/watchlist.json`，您可以在此直接輸入任何台股代碼（如 `2317`、`2454`、`2603`、`0050`）或期貨代碼（如 `TMFR1`、`TXFR1`）隨時增減。")

        btn_c1, btn_c2, btn_c3, btn_c4 = st.columns([1.15, 0.9, 1.15, 0.9])
        with btn_c1:
            st.write("")
            if st.button("🚀 連線行情伺服器", width="stretch", type="primary"):
                with st.spinner("正在連線台新伺服器並載入真實 K 線與即時報價..."):
                    feed.start(actual_user, ts_pwd, clean_sym)
                    for _ in range(20):
                        time.sleep(0.5)
                        if feed.ta_rcv_done or "失敗" in feed.status_msg or "異常" in feed.status_msg:
                            break
                    st.rerun()
        with btn_c2:
            st.write("")
            if st.button("🛑 中斷連線", width="stretch"):
                feed.stop()
                st.rerun()
        with btn_c3:
            refresh_opt = st.selectbox(
                "畫面自動刷新頻率 (僅讀本地記憶體，不耗 API 次數)",
                ["10 秒 (推薦)", "5 秒", "15 秒", "30 秒", "60 秒", "關閉自動刷新"],
                index=0,
                key="live_refresh_interval_opt",
            )
            if refresh_opt == "關閉自動刷新" or not feed.connected:
                quote_refresh_sec = None
                chart_refresh_sec = None
            else:
                quote_refresh_sec = int(refresh_opt.split(" ")[0])
                # K 線圖以 1 分鐘 K 為主，最低 30 秒才局部重繪一次，避免干擾看盤縮放
                chart_refresh_sec = max(30, quote_refresh_sec)
        with btn_c4:
            st.write("")
            if st.button("🔄 立即刷新", width="stretch"):
                st.rerun()

        # 1. 頂部即時報價數字看板 (獨立輕量 Fragment，依選取秒數局部跳動，不干擾下方 K 線圖)
        st.markdown("---")

        @st.fragment(run_every=quote_refresh_sec)
        def _render_live_quote_banner():
            m_c1, m_c2, m_c3, m_c4 = st.columns([1.8, 1.0, 0.8, 1.2])
            m_c1.metric("伺服器狀態", feed.status_msg)
            latest_tick = feed.get_latest_tick()
            if latest_tick:
                m_c2.metric("最新成交價", f"{latest_tick.price:,.1f}")
                m_c3.metric("單量", f"{latest_tick.volume}")
                m_c4.metric("最新成交時間", latest_tick.timestamp.strftime("%Y-%m-%d %H:%M:%S"))
            else:
                m_c2.metric("最新成交價", "等待台新 API 報價中...")
                m_c3.metric("單量", "--")
                m_c4.metric("最新成交時間", "--")

        _render_live_quote_banner()

        # 2. 台新 API 真實 K 棒圖表 (獨立 Fragment，最低 30 秒更新一次，看圖縮放零閃爍)
        k_hdr1, k_hdr2, k_hdr3 = st.columns([2.2, 0.9, 0.9])
        with k_hdr2:
            chart_tf = st.selectbox(
                "圖表顯示 K 線週期",
                ["1m", "5m", "15m", "30m", "60m"],
                index=0,
                key="live_chart_tf",
            )
        with k_hdr3:
            vis_bars_opt = st.selectbox(
                "預設畫面 K 棒數量",
                ["60 根", "120 根", "250 根", "500 根", "1000 根", "全部顯示"],
                index=1,
                key="live_chart_vis_bars",
            )

        @st.fragment(run_every=chart_refresh_sec)
        def _render_live_kchart_and_ticks():
            real_bars = feed.get_bars(timeframe=chart_tf, session="day_and_night")
            if real_bars.height > 0:
                st.markdown(f"**📊 台新 API `{feed.symbol}` {chart_tf} K 線圖 (共 {real_bars.height} 根完整載入)**")
                df_plot = real_bars.to_pandas()
                df_plot["time_label"] = df_plot["timestamp"].dt.strftime("%m-%d %H:%M")
                total_k = len(df_plot)
                if vis_bars_opt == "全部顯示":
                    vis_count = total_k
                else:
                    vis_count = int(vis_bars_opt.split(" ")[0])
                init_start = max(0, total_k - vis_count)
                init_end = max(0, total_k - 1)

                vis_slice = df_plot.iloc[init_start:]
                y_min = float(vis_slice["low"].min())
                y_max = float(vis_slice["high"].max())
                y_pad = max(1.0, (y_max - y_min) * 0.08)

                fig_k = go.Figure(data=[go.Candlestick(
                    x=df_plot["time_label"],
                    open=df_plot["open"],
                    high=df_plot["high"],
                    low=df_plot["low"],
                    close=df_plot["close"],
                    increasing_line_color="#ef5350",
                    decreasing_line_color="#26a69a",
                    name=f"{feed.symbol} ({chart_tf})"
                )])
                fig_k.update_layout(
                    template="plotly_dark",
                    height=440,
                    margin=dict(l=20, r=20, t=25, b=20),
                    xaxis_rangeslider_visible=False,
                    xaxis=dict(
                        type="category",
                        range=[init_start - 0.5, init_end + 2.5],
                        minallowed=-0.5,
                        maxallowed=total_k + 2.5,
                    ),
                    yaxis=dict(
                        range=[y_min - y_pad, y_max + y_pad],
                        fixedrange=False,
                    ),
                    dragmode="pan",
                    uirevision=f"kchart_{feed.symbol}_{chart_tf}_{vis_bars_opt}",
                )
                st.plotly_chart(
                    fig_k,
                    width="stretch",
                    key=f"plotly_live_k_{feed.symbol}_{chart_tf}_{vis_bars_opt}",
                    config={"scrollZoom": True, "displayModeBar": True},
                )
                st.caption("💡 操作提示：上方成交價依您選定的秒數（預設 10 秒）自動局部更新，下方 K 線圖則每 30 秒才更新一次（且完整保留您的縮放比例與位置），看盤操作完全不卡頓。")
            else:
                st.info("💡 請點擊上方「🚀 連線行情伺服器」，系統將透過台新官方 API 載入真實 1 分鐘 K 棒與即時成交報價。")

            recent_ticks = feed.get_recent_ticks()
            if recent_ticks:
                st.subheader(f"⚡ 即時成交明細 (最近 {min(15, len(recent_ticks))} 筆)")
                tick_rows = []
                for tk in reversed(recent_ticks[-15:]):
                    tick_rows.append({
                        "時間": tk.timestamp.strftime("%H:%M:%S"),
                        "成交價": tk.price,
                        "單筆數量": tk.volume,
                    })
                st.dataframe(pd.DataFrame(tick_rows), width="stretch", height=180)

        _render_live_kchart_and_ticks()

    elif selected_nav == nav_tabs[4]:
        st.subheader("🎯 紙上交易模擬撮合與風控中心 (Paper Trading & Risk Center)")
        st.caption("與回測共用同一套事件驅動核心，串接台新 API 真實 K 線與即時報價進行模擬撮合。支援單日最大虧損鎖定、單日獲利達標全平、模型退化自動停機與一鍵 Kill Switch。")

        # 初始化或更新紙上交易引擎單例
        if (
            "paper_trader" not in st.session_state
            or getattr(st.session_state["paper_trader"], "VERSION", None) != PaperTradingEngine.VERSION
            or st.session_state["paper_trader"].strategy.id != strategy.id
        ):
            st.session_state["paper_trader"] = PaperTradingEngine(strategy, contract, initial_capital)
        pt: PaperTradingEngine = st.session_state["paper_trader"]
        pt.strategy = strategy
        pt.contract = contract

        # 風控參數設定列 (新增：單日獲利達標全平、連續虧損模型退化暫停)
        r_col1, r_col2, r_col3, r_col4, r_col5 = st.columns(5)
        with r_col1:
            max_loss_in = st.number_input("單日最大虧損鎖定 (TWD)", min_value=500.0, max_value=100000.0, value=float(pt.risk_config.max_daily_loss), step=500.0)
            pt.risk_config.max_daily_loss = float(max_loss_in)
        with r_col2:
            prof_target_in = st.number_input("單日獲利達標全平 (0=不限)", min_value=0.0, max_value=200000.0, value=float(pt.risk_config.daily_profit_target), step=500.0)
            pt.risk_config.daily_profit_target = float(prof_target_in)
        with r_col3:
            consec_in = st.number_input("模型退化連虧停機筆數", min_value=0, max_value=20, value=int(pt.risk_config.max_consecutive_losses), step=1)
            pt.risk_config.max_consecutive_losses = int(consec_in)
        with r_col4:
            max_cnt_in = st.number_input("最大允許持倉口數", min_value=1, max_value=20, value=int(pt.risk_config.max_contracts), step=1)
            pt.risk_config.max_contracts = int(max_cnt_in)
        with r_col5:
            # 檢查五道可信度是否已通過
            cred_passed = False
            if df_bars.height >= 30:
                fast_k = "fast_ma" if "fast_ma" in strategy.indicators else "ma5"
                slow_k = "slow_ma" if "slow_ma" in strategy.indicators else "ma20"
                val_check = CredibilityValidator(strategy, contract, initial_capital).evaluate_all(
                    df_bars, ind_x=fast_k, x_values=[3, 5, 7, 10], ind_y=slow_k, y_values=[15, 20, 25, 30], mc_iterations=300
                )
                cred_passed = val_check.overall_passed
            allow_test_override = st.checkbox("允許測試模式略過可信度門檻 (僅限紙上模擬)", value=True)

        # 同步台新最新 K 線與報價
        latest_tk = feed.get_latest_tick()
        curr_mkt_price = latest_tk.price if latest_tk else (float(df_bars["close"][-1]) if df_bars.height > 0 else 0.0)
        pt.sync_with_market(df_bars, latest_price=curr_mkt_price)

        # 控制按鈕列 (含一鍵 Kill Switch)
        any_locked = pt.kill_switch_locked or pt.daily_loss_locked or pt.daily_profit_locked or pt.degradation_locked
        act_c1, act_c2, act_c3, act_c4 = st.columns(4)
        with act_c1:
            if st.button("▶️ 啟動紙上交易", width="stretch", type="primary", disabled=pt.is_active):
                ok, msg = pt.start_session(credibility_passed=(cred_passed or allow_test_override))
                if ok:
                    st.toast(msg)
                else:
                    st.error(msg)
                st.rerun()
        with act_c2:
            if st.button("⏸️ 暫停訊號執行", width="stretch", disabled=not pt.is_active):
                pt.stop_session()
                st.rerun()
        with act_c3:
            if st.button("🚨 一鍵緊急平倉 (Kill Switch)", width="stretch"):
                ks_msg = pt.trigger_kill_switch(current_price=curr_mkt_price)
                st.warning(ks_msg)
                st.rerun()
        with act_c4:
            if st.button("🔓 解除風控鎖定", width="stretch", disabled=not any_locked):
                pt.reset_risk_locks()
                st.rerun()

        st.markdown("---")
        # 即時帳戶與部位狀態卡片
        if pt.kill_switch_locked:
            status_label = "🚨 Kill Switch 鎖定"
        elif pt.daily_loss_locked:
            status_label = "🛑 單日虧損鎖定"
        elif pt.daily_profit_locked:
            status_label = "🏆 獲利達標全平鎖定"
        elif pt.degradation_locked:
            status_label = "⚠️ 模型退化連虧暫停"
        elif pt.is_active:
            status_label = "🟢 運行中"
        else:
            status_label = "⚪ 待命暫停"

        pos_label = (
            f"{'多單' if pt.position.direction == 1 else '空單'} {pt.position.contracts} 口 @ {pt.position.entry_price:,.1f}"
            if pt.position else "空手 (0 口)"
        )
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("引擎風控狀態", status_label)
        k2.metric("當前持倉部位", pos_label)
        k3.metric("未實現損益", f"{pt.get_unrealized_pnl(curr_mkt_price):,.0f} 元")
        k4.metric("已實現淨損益", f"{pt.realized_pnl:,.0f} 元")
        k5.metric("連續虧損筆數", f"{pt.consecutive_losses} 筆")
        k6.metric("帳戶總權益", f"{pt.get_total_equity(curr_mkt_price):,.0f} 元")

        if pt.audit_logs:
            st.subheader("📜 風控與委託稽核日誌 (Audit Log)")
            log_rows = [
                {
                    "時間": entry.timestamp,
                    "事件類型": entry.event_type,
                    "觸發價格": entry.price if entry.price is not None else "--",
                    "損益 (TWD)": entry.pnl if entry.pnl is not None else "--",
                    "詳細說明": entry.message,
                }
                for entry in reversed(pt.audit_logs[-20:])
            ]
            st.dataframe(pd.DataFrame(log_rows), width="stretch", height=240)

    elif selected_nav == nav_tabs[5]:
        st.subheader("🤖 AI 自然語言策略協作與 Strategy Schema 邏輯樹")
        st.markdown("輸入中文交易想法，系統將自動轉譯為嚴格驗證的 **Strategy Schema**、執行**強制停損安全審查**，並產生**視覺化策略邏輯樹**供確認。")

        nl_default = "當 5MA 向上突破 20MA 且 RSI(14) 大於 50 時做多，跌破時做空，停損 45 點，停利 120 點，移動停損 50 點"
        nl_prompt = st.text_area("💬 輸入中文策略描述", value=nl_default, height=90)

        ai_c1, ai_c2 = st.columns([1.2, 3])
        with ai_c1:
            if st.button("✨ AI 解析生成策略 Schema", width="stretch", type="primary"):
                try:
                    parsed = AIStrategyCopilot.parse_natural_language(
                        nl_prompt, symbol=selected_symbol, timeframe=bt_timeframe, session=strategy.session
                    )
                    st.session_state["ai_parsed_result"] = parsed
                except Exception as ex:
                    st.error(f"解析失敗: {ex}")

        if "ai_parsed_result" in st.session_state:
            parsed_res = st.session_state["ai_parsed_result"]
            for warn in parsed_res.clarifications:
                st.warning(warn)
            st.success(parsed_res.summary_text)
            if st.button("💾 儲存此 AI 策略至範本庫 (strategies/)", type="secondary"):
                out_path = Path("strategies") / f"{parsed_res.strategy.id}.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(parsed_res.strategy.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
                st.toast(f"已儲存至 {out_path}！可於左側邊欄「策略範本」選用。")
                st.rerun()

            active_disp_strat = parsed_res.strategy
        else:
            active_disp_strat = strategy

        tree_col, json_col = st.columns(2)
        with tree_col:
            st.markdown("#### 🌳 視覺化策略邏輯樹")
            tree_str = "\n".join(AIStrategyCopilot.generate_logic_tree_lines(active_disp_strat))
            st.code(tree_str, language="text")
        with json_col:
            st.markdown("#### 📜 Strategy Schema (JSON 規格)")
            st.json(active_disp_strat.model_dump())


if __name__ == "__main__":
    main()

# src/ui/excel_warroom.py
"""
Excel 戰情室即時看盤與雙向控制橋接模組
支援：
1. openpyxl 建立專業 Excel 戰情室範本 (AI_Trading_WarRoom.xlsx)
2. collect_warroom_snapshot() 收集行情、ORB 多空訊號、關鍵點位與持倉狀態
3. win32com 即時同步更新已開啟的 Excel 視窗與讀取控制格指令 (B4 自動開關, D4 手動指令)
4. 命令列支援 --init, --once, --loop
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime, timezone, timedelta, time as dt_time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import polars as pl
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from src.broker.taishin_quote import TaishinQuoteFeed
from src.core.orb_strategy import multi_wave_orb_params, compute_orb_state, OrbState
from src.core.paper_trader import (
    PaperTradingEngine,
    RiskConfig,
    default_ai_strategy,
    ensure_atr_exits,
)
from src.core.engine import ContractSpec
from src.core.key_levels import compute_key_levels, KeyLevels
from src.core.indicators import IndicatorEngine

logger = logging.getLogger(__name__)

DEFAULT_MAX_DAILY_LOSS = 5000.0


def load_excel_taishin_config() -> dict:
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

TW_TZ = timezone(timedelta(hours=8))
EXCEL_FILENAME = "AI_Trading_WarRoom.xlsx"
SNAPSHOT_PATH = Path("data/excel_live_snapshot.json")

# 顏色常數
COLOR_PRIMARY_GREEN = "107C41"
COLOR_SUBHEADER_GRAY = "E9ECEF"
COLOR_HIGHLIGHT_YELLOW = "FFF8E1"
COLOR_BG_CARD = "F8F9FA"
COLOR_TEXT_RED = "D93025"
COLOR_TEXT_GREEN = "188038"
COLOR_TEXT_MUTED = "6C757D"
COLOR_BORDER = "CED4DA"


def get_thin_border() -> Border:
    s = Side(style="thin", color=COLOR_BORDER)
    return Border(left=s, right=s, top=s, bottom=s)


def build_excel_template(filepath: str | Path = EXCEL_FILENAME) -> Path:
    """使用 openpyxl 建立標準三分頁 Excel 戰情室範本"""
    filepath = Path(filepath)
    wb = openpyxl.Workbook()

    thin_border = get_thin_border()

    font_title = Font(name="微軟正黑體", size=13, bold=True, color="FFFFFF")
    font_section = Font(name="微軟正黑體", size=11, bold=True, color="0C3820")
    font_bold = Font(name="微軟正黑體", size=10, bold=True)
    font_normal = Font(name="微軟正黑體", size=10)
    font_muted = Font(name="微軟正黑體", size=9, color=COLOR_TEXT_MUTED)

    fill_header = PatternFill(start_color=COLOR_PRIMARY_GREEN, end_color=COLOR_PRIMARY_GREEN, fill_type="solid")
    fill_sub = PatternFill(start_color=COLOR_SUBHEADER_GRAY, end_color=COLOR_SUBHEADER_GRAY, fill_type="solid")
    fill_ctrl = PatternFill(start_color="E7F1FF", end_color="E7F1FF", fill_type="solid")

    # ==========================
    # Sheet 1: 📊 戰情室即時看盤
    # ==========================
    ws1 = wb.active
    ws1.title = "📊 戰情室即時看盤"
    ws1.views.sheetView[0].showGridLines = True

    # Row 1: Title
    ws1.merge_cells("A1:J1")
    ws1["A1"] = "【AI 智慧看盤與自動交易戰情室 — EXCEL 即時終端 (v2.12.0)】"
    ws1["A1"].font = font_title
    ws1["A1"].fill = fill_header
    ws1["A1"].alignment = Alignment(horizontal="center", vertical="center")
    ws1.row_dimensions[1].height = 32

    # Row 2: Status
    ws1["A2"] = "最後更新:"
    ws1["B2"] = "尚未連線"
    ws1["C2"] = "盤別時段:"
    ws1["D2"] = "--"
    ws1["E2"] = "行情狀態:"
    ws1["F2"] = "未連線"
    ws1["G2"] = "保本/動態風控:"
    ws1["H2"] = "已啟用 (1.0 ATR)"
    ws1["I2"] = "帳戶權益:"
    ws1["J2"] = "$100,000"

    for c in ["A2", "C2", "E2", "G2", "I2"]:
        ws1[c].font = font_bold
        ws1[c].alignment = Alignment(horizontal="right", vertical="center")
    for c in ["B2", "D2", "F2", "H2", "J2"]:
        ws1[c].font = font_normal
        ws1[c].alignment = Alignment(horizontal="left", vertical="center")
    ws1.row_dimensions[2].height = 20

    # Row 4: Control Panel
    ws1.merge_cells("A4:J4")
    ws1["A4"] = "  🎛️ 【雙向控制項】 (直接在儲存格輸入即可互動)  |  B4 自動盯盤開關 [ON/OFF]  |  D4 手動指令 [無/平倉/FLATTEN]  |  F4 刷新秒數"
    ws1["A4"].font = font_bold
    ws1["A4"].fill = fill_ctrl
    ws1["A4"].alignment = Alignment(horizontal="left", vertical="center")
    ws1.row_dimensions[4].height = 22

    # Row 5: Control inputs
    ws1["A5"] = "自動操盤:"
    ws1["B5"] = "ON"
    ws1["C5"] = "手動緊急指令:"
    ws1["D5"] = "無"
    ws1["E5"] = "刷新週期(秒):"
    ws1["F5"] = 10
    ws1["G5"] = "說明:"
    ws1["H5"] = "在 D5 輸入「平倉」會立即強制市價平倉部位"

    for c in ["A5", "C5", "E5", "G5"]:
        ws1[c].font = font_bold
        ws1[c].alignment = Alignment(horizontal="right", vertical="center")
    for c in ["B5", "D5", "F5"]:
        ws1[c].font = Font(name="微軟正黑體", size=11, bold=True, color="004085")
        ws1[c].alignment = Alignment(horizontal="center", vertical="center")
        ws1[c].border = thin_border
    ws1["H5"].font = font_muted
    ws1.row_dimensions[5].height = 22

    # Section 1: Market Quote (Row 7~9)
    ws1["A7"] = "■ 微台指 (TMFR1) 即時行情與客觀關鍵價位"
    ws1["A7"].font = font_section

    quote_headers = ["商品代碼", "最新成交", "漲跌", "幅度", "今日開盤", "最高 / 最低", "當盤總量", "當日VWAP", "前日收盤", "預期波動區間(ATR)"]
    for idx, h in enumerate(quote_headers, start=1):
        cell = ws1.cell(row=8, column=idx, value=h)
        cell.font = font_bold
        cell.fill = fill_sub
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = thin_border
    ws1.row_dimensions[8].height = 20

    quote_init = ["TMFR1 (微台)", "--", "--", "--", "--", "-- / --", "--", "--", "--", "-- ~ --"]
    for idx, val in enumerate(quote_init, start=1):
        cell = ws1.cell(row=9, column=idx, value=val)
        cell.font = font_normal
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = thin_border
    ws1.row_dimensions[9].height = 22

    # Section 2: ORB Signal (Row 11~14)
    ws1["A11"] = "■ 日夜雙盤多波段 ORB 箱型突破訊號"
    ws1["A11"].font = font_section

    orb_headers = ["目前訊號狀態", "箱型上緣", "箱型下緣", "箱型寬度", "現價距突破", "量比 (前20均)", "K棒飽滿度", "突破品質", "訊號產生時間", "規則決策原因"]
    for idx, h in enumerate(orb_headers, start=1):
        cell = ws1.cell(row=12, column=idx, value=h)
        cell.font = font_bold
        cell.fill = fill_sub
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = thin_border
    ws1.row_dimensions[12].height = 20

    orb_init = ["⏳ 觀望 / 箱型形成中", "--", "--", "--", "--", "--", "--", "--", "--", "等待有效突破"]
    for idx, val in enumerate(orb_init, start=1):
        cell = ws1.cell(row=13, column=idx, value=val)
        cell.font = font_normal
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = thin_border
    ws1.row_dimensions[13].height = 24

    # Section 3: Paper Position (Row 15~18)
    ws1["A15"] = "■ 模擬倉部位與三段停損守護 (初始停損 ➔ 保本鎖利 ➔ 移動停利)"
    ws1["A15"].font = font_section

    pos_headers = ["持倉方向", "進場口數", "進場均價", "目前點數損益", "未實現損益(元)", "目前有效停損價", "停損保護模式", "目標停利價", "今日已實現損益", "策略名稱"]
    for idx, h in enumerate(pos_headers, start=1):
        cell = ws1.cell(row=16, column=idx, value=h)
        cell.font = font_bold
        cell.fill = fill_sub
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = thin_border
    ws1.row_dimensions[16].height = 20

    pos_init = ["無部位 (空手)", 0, "--", 0.0, "$0", "--", "未啟動", "--", "$0", "日夜雙盤多波段ORB+保本停損"]
    for idx, val in enumerate(pos_init, start=1):
        cell = ws1.cell(row=17, column=idx, value=val)
        cell.font = font_normal
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = thin_border
    ws1.row_dimensions[17].height = 24

    # Column widths
    col_widths = {1: 16, 2: 14, 3: 13, 4: 13, 5: 14, 6: 18, 7: 15, 8: 16, 9: 18, 10: 30}
    for col_idx, width in col_widths.items():
        ws1.column_dimensions[get_column_letter(col_idx)].width = width

    # ==========================
    # Sheet 2: 📑 歷史成交與日誌
    # ==========================
    ws2 = wb.create_sheet(title="📑 歷史成交與日誌")
    ws2.views.sheetView[0].showGridLines = True

    ws2.merge_cells("A1:H1")
    ws2["A1"] = "【模擬操盤對帳單與成交紀錄】"
    ws2["A1"].font = font_title
    ws2["A1"].fill = fill_header
    ws2["A1"].alignment = Alignment(horizontal="center", vertical="center")
    ws2.row_dimensions[1].height = 28

    trade_headers = ["交易序號", "進場時間", "出場時間", "商品", "方向", "進場價", "出場價", "損益(點)", "損益(元)", "出場原因"]
    for idx, h in enumerate(trade_headers, start=1):
        cell = ws2.cell(row=2, column=idx, value=h)
        cell.font = font_bold
        cell.fill = fill_sub
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = thin_border
    ws2.row_dimensions[2].height = 20

    ws2["A18"] = "【系統執行與風控日誌 (最近 15 筆)】"
    ws2["A18"].font = font_section
    log_headers = ["時間", "層級", "訊息"]
    for idx, h in enumerate(log_headers, start=1):
        cell = ws2.cell(row=19, column=idx, value=h)
        cell.font = font_bold
        cell.fill = fill_sub
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = thin_border

    ws2.column_dimensions["A"].width = 14
    ws2.column_dimensions["B"].width = 20
    ws2.column_dimensions["C"].width = 20
    ws2.column_dimensions["D"].width = 12
    ws2.column_dimensions["E"].width = 10
    ws2.column_dimensions["F"].width = 12
    ws2.column_dimensions["G"].width = 12
    ws2.column_dimensions["H"].width = 12
    ws2.column_dimensions["I"].width = 14
    ws2.column_dimensions["J"].width = 28

    # ==========================
    # Sheet 3: 📈 最新5分K與指標
    # ==========================
    ws3 = wb.create_sheet(title="📈 最新5分K與指標")
    ws3.views.sheetView[0].showGridLines = True

    ws3.merge_cells("A1:H1")
    ws3["A1"] = "【最近 50 根 5 分 K 棒與即時技術指標 (可自由拉公式)】"
    ws3["A1"].font = font_title
    ws3["A1"].fill = fill_header
    ws3["A1"].alignment = Alignment(horizontal="center", vertical="center")
    ws3.row_dimensions[1].height = 28

    k_headers = ["時間", "開盤價", "最高價", "最低價", "收盤價", "單根成交量", "累計成交量", "ATR(14)波動度"]
    for idx, h in enumerate(k_headers, start=1):
        cell = ws3.cell(row=2, column=idx, value=h)
        cell.font = font_bold
        cell.fill = fill_sub
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = thin_border
    ws3.row_dimensions[2].height = 20

    ws3.column_dimensions["A"].width = 20
    for col_c in ["B", "C", "D", "E", "F", "G", "H"]:
        ws3.column_dimensions[col_c].width = 14

    wb.save(filepath)
    logger.info("已建立 Excel 範本: %s", filepath)
    return filepath


def get_mock_or_cached_kbars() -> Tuple[pl.DataFrame, pl.DataFrame]:
    """若無網路連線或無行情 API 時，讀取快取 parquet 或合成假 K 棒"""
    cache_1m_path = Path("data/taishin_kbars_TMFR1_1m.parquet")
    history_path = Path("data/history/TMF_1m.parquet")

    df_1m: Optional[pl.DataFrame] = None
    if cache_1m_path.exists():
        try:
            df_1m = pl.read_parquet(cache_1m_path)
        except Exception:
            pass

    if df_1m is None and history_path.exists():
        try:
            df_1m = pl.read_parquet(history_path).tail(300)
        except Exception:
            pass

    if df_1m is None or len(df_1m) == 0:
        # 建立合成 1m K 棒
        now = datetime.now()
        rows = []
        base_p = 22800.0
        for i in range(120, -1, -1):
            t = (now - timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S")
            rows.append({
                "ts": t,
                "open": base_p + (i % 5),
                "high": base_p + (i % 5) + 8,
                "low": base_p + (i % 5) - 6,
                "close": base_p + (i % 5) + 2,
                "volume": 80.0 + (i % 10) * 10,
            })
        df_1m = pl.DataFrame(rows)

    # 確保 1m 欄位統一
    if "ts" not in df_1m.columns and "datetime" in df_1m.columns:
        df_1m = df_1m.rename({"datetime": "ts"})

    # 合成 5m K 棒
    try:
        df_5m = IndicatorEngine.resample_kbars(df_1m, "5m")
    except Exception:
        df_5m = df_1m.tail(30)

    return df_1m, df_5m


def collect_warroom_snapshot(
    feed: Optional[TaishinQuoteFeed] = None,
    pt: Optional[PaperTradingEngine] = None,
    symbol: str = "TMFR1",
) -> Dict[str, Any]:
    """
    收集所有戰情室數據、計算 ORB 訊號與停損守護，輸出標準 dict 並儲存至 JSON 快取。
    完全不依賴 COM 或 Excel 執行緒，適合測試與排程。
    """
    cfg = load_excel_taishin_config()
    now_dt = datetime.now(TW_TZ).replace(tzinfo=None)
    now_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")

    # 1. 取得行情快照與 K 棒
    bars_1m: Optional[pl.DataFrame] = None
    bars_5m: Optional[pl.DataFrame] = None
    curr_price: float = 0.0
    quote_status: str = "離線快取"

    if feed is None and cfg.get("sol_user_id") and cfg.get("sol_password"):
        try:
            feed = TaishinQuoteFeed.get_instance(cfg["sol_user_id"], cfg["sol_password"])
        except Exception as e:
            logger.debug("無法連接台新 API: %s", e)

    if feed is not None:
        try:
            snap = feed.get_market_snapshot(symbol)
            if snap and snap.current_price > 0:
                curr_price = float(snap.current_price)
                quote_status = "連線正常 (台新API)"
            bars_1m = feed.get_kbars(symbol, timeframe="1m", limit=300)
            bars_5m = feed.get_kbars(symbol, timeframe="5m", limit=100)
        except Exception as e:
            logger.debug("從 feed 讀取資料失敗: %s", e)

    if bars_1m is None or len(bars_1m) == 0:
        bars_1m, bars_5m = get_mock_or_cached_kbars()
        if curr_price <= 0 and len(bars_1m) > 0:
            curr_price = float(bars_1m["close"][-1])

    if curr_price <= 0 and len(bars_1m) > 0:
        curr_price = float(bars_1m["close"][-1])

    # 2. 計算關鍵價位
    session_open_t, session_close_t = dt_time(8, 45), dt_time(13, 45)
    key_levels: KeyLevels
    try:
        # 確保 timestamp 欄位名稱正確
        df_for_levels = bars_1m
        if "timestamp" not in df_for_levels.columns and "ts" in df_for_levels.columns:
            df_for_levels = df_for_levels.rename({"ts": "timestamp"})
        if "timestamp" in df_for_levels.columns and df_for_levels["timestamp"].dtype == pl.Utf8:
            df_for_levels = df_for_levels.with_columns(
                pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
            )
        key_levels = compute_key_levels(df_for_levels, now=now_dt, open_t=session_open_t, close_t=session_close_t)
    except Exception as e:
        logger.debug("計算客觀關鍵價位例外: %s", e)
        key_levels = KeyLevels()

    # 3. 計算 ORB 狀態
    orb_params = multi_wave_orb_params()
    orb_state: OrbState = compute_orb_state(bars_1m, now=now_dt, params=orb_params)

    # 4. 模擬倉引擎同步
    if pt is None:
        c_spec = ContractSpec.from_json("TMF")
        strat = default_ai_strategy("TMF")
        pt = PaperTradingEngine(
            strategy=strat,
            contract=c_spec,
            initial_capital=100_000.0,
            risk_config=RiskConfig(max_daily_loss=DEFAULT_MAX_DAILY_LOSS, max_contracts=1),
        )
        pt.load_state_from_disk()
        migrated, _ = ensure_atr_exits(pt.strategy)
        pt.strategy = migrated

    # 離線補盤與同步訊號
    if curr_price > 0:
        pt.catch_up_missed_bars(
            bars_1m, bars_5m, orb_params,
            allow_multiple_entries=(orb_params.max_entries_per_session > 1)
        )
        pt.sync_with_rule_signal(
            action=orb_state.action,
            current_price=curr_price,
            df_bars=bars_5m,
            flatten=orb_state.flatten,
            source_note=f"{orb_state.session_label}Excel同步",
            allow_multiple_entries=(orb_params.max_entries_per_session > 1),
            reentry_cooldown_sec=float(orb_params.reentry_cooldown_minutes * 60),
        )

    # 5. 整理行情數據
    last_1m = bars_1m[-1].to_dicts()[0] if len(bars_1m) > 0 else {}
    first_1m = bars_1m[0].to_dicts()[0] if len(bars_1m) > 0 else {}
    open_p = float(first_1m.get("open", curr_price))
    high_p = float(bars_1m["high"].max()) if len(bars_1m) > 0 else curr_price
    low_p = float(bars_1m["low"].min()) if len(bars_1m) > 0 else curr_price
    vol_sum = float(bars_1m["volume"].sum()) if len(bars_1m) > 0 else 0.0

    prev_close = key_levels.prev_close or open_p
    diff = curr_price - prev_close
    pct = (diff / prev_close * 100) if prev_close > 0 else 0.0

    # 6. 整理部位與停損數據
    pos = pt.position
    pos_direction = "無部位 (空手)"
    pos_qty = 0
    pos_entry = "--"
    pos_pts = 0.0
    pos_pnl = "$0"
    pos_stop = "--"
    pos_stop_mode = "未啟動"
    pos_tp = "--"

    if pos is not None:
        pos_direction = "買進多單 (BUY)" if pos.direction == 1 else "放空空單 (SELL)"
        pos_qty = pos.contracts
        pos_entry = f"{pos.entry_price:,.1f}"
        pos_pts = (curr_price - pos.entry_price) if pos.direction == 1 else (pos.entry_price - curr_price)
        pos_pnl = f"${pos_pts * pt.contract.point_value * pos.contracts:+,.0f}"

        # 停損與保本模式
        stop_p = pos.stop_loss_price
        tp_p = pos.take_profit_price
        pos_stop = f"{stop_p:,.1f}" if stop_p else "--"
        pos_tp = f"{tp_p:,.1f}" if tp_p else "--"

        # 檢查是否已觸發保本或移動停損
        if (stop_p and ((pos.direction == 1 and stop_p >= pos.entry_price) or (pos.direction == -1 and stop_p <= pos.entry_price))):
            pos_stop_mode = "🛡️ 保本鎖利停損 (Breakeven)"
        else:
            pos_stop_mode = "🔴 初始 ATR 停損"

    # 7. 整理歷史成交與近 30 根 K 棒
    trades = []
    for t in pt.closed_trades[-15:]:
        if hasattr(t, "to_dict"):
            trades.append(t.to_dict())
        elif hasattr(t, "__dict__"):
            trades.append({k: (str(v) if isinstance(v, (datetime, Path)) else v) for k, v in t.__dict__.items()})
        else:
            trades.append(dict(t))

    logs = []
    for log_item in pt.audit_logs[-15:]:
        if hasattr(log_item, "to_dict"):
            logs.append(log_item.to_dict())
        elif hasattr(log_item, "__dict__"):
            logs.append({k: (str(v) if isinstance(v, (datetime, Path)) else v) for k, v in log_item.__dict__.items()})
        else:
            logs.append(str(log_item))

    recent_kbars = []
    if bars_5m is not None and len(bars_5m) > 0:
        tail_5m = bars_5m.tail(30).to_dicts()
        for row in tail_5m:
            recent_kbars.append({
                "ts": str(row.get("ts", row.get("datetime", ""))),
                "open": float(row.get("open", 0)),
                "high": float(row.get("high", 0)),
                "low": float(row.get("low", 0)),
                "close": float(row.get("close", 0)),
                "volume": float(row.get("volume", 0)),
            })

    snapshot = {
        "timestamp": now_str,
        "symbol": symbol,
        "session_label": orb_state.session_label,
        "quote_status": quote_status,
        "market": {
            "current_price": curr_price,
            "diff": diff,
            "pct": pct,
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "volume_sum": vol_sum,
            "vwap": orb_state.session_vwap or key_levels.vwap or curr_price,
            "prev_close": prev_close,
            "range_high": key_levels.exp_upper,
            "range_low": key_levels.exp_lower,
        },
        "orb": {
            "action": orb_state.action,
            "action_desc": "做多 (BUY)" if orb_state.action == "BUY_LONG" else "做空 (SELL)" if orb_state.action == "SELL_SHORT" else "觀望中 (HOLD)",
            "range_high": orb_state.range_high,
            "range_low": orb_state.range_low,
            "range_width": (orb_state.range_high - orb_state.range_low) if (orb_state.range_high and orb_state.range_low) else None,
            "vol_ratio": orb_state.vol_ratio,
            "body_ratio": orb_state.body_ratio,
            "quality_note": orb_state.quality_note,
            "breakout_time": orb_state.breakout_ts,
            "reason": orb_state.reason,
        },
        "paper_trader": {
            "direction": pos_direction,
            "qty": pos_qty,
            "entry_price": pos_entry,
            "pts": pos_pts,
            "pnl": pos_pnl,
            "stop_loss": pos_stop,
            "stop_mode": pos_stop_mode,
            "take_profit": pos_tp,
            "realized_pnl": f"${pt.daily_realized_pnl:+,.0f}",
            "equity": f"${pt.get_total_equity():,.0f}",
        },
        "trades": trades,
        "logs": logs,
        "recent_5m": recent_kbars,
    }

    # 存檔至 JSON
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SNAPSHOT_PATH, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

    return snapshot


def execute_excel_commands(ws1, pt: PaperTradingEngine) -> Dict[str, Any]:
    """讀取 Excel 儲存格控制項（B5 自動開關、D5 手動指令）"""
    res = {"autopilot": True, "flatten_triggered": False}
    try:
        b5_val = str(ws1.Range("B5").Value or "").strip().upper()
        if b5_val in ["OFF", "FALSE", "0", "停用"]:
            res["autopilot"] = False

        d5_val = str(ws1.Range("D5").Value or "").strip()
        if d5_val in ["平倉", "FLATTEN", "市價平倉", "出場"]:
            if pt.position is not None:
                # 執行平倉 (若無 last_price 則以進場價或當前行情平倉)
                exit_p = pt.last_price if pt.last_price > 0 else pt.position.entry_price
                pt.close_position_manually(current_price=exit_p, reason="excel_manual_flatten")
                pt._log("USER", "透過 Excel 儲存格 D5 觸發手動平倉")
                pt.save_state_to_disk()
                res["flatten_triggered"] = True
            ws1.Range("D5").Value = "無 (已於平倉後重設)"
    except Exception as e:
        logger.debug("讀取 Excel 控制格時跳過 (可能使用者正在編輯中): %s", e)
    return res


def save_snapshot_to_xlsx_file(snapshot: Dict[str, Any], filepath: str | Path = EXCEL_FILENAME) -> bool:
    """直接使用 openpyxl 寫入實體 xlsx 檔案，即使 Excel 視窗未開也能保持檔案為最新"""
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            build_excel_template(filepath)

        wb = openpyxl.load_workbook(filepath)
        ws1 = wb["📊 戰情室即時看盤"]

        m = snapshot["market"]
        orb = snapshot["orb"]
        p = snapshot["paper_trader"]

        ws1["B2"].value = snapshot["timestamp"]
        ws1["D2"].value = snapshot["session_label"]
        ws1["F2"].value = snapshot["quote_status"]
        ws1["J2"].value = p["equity"]

        ws1["A9"].value = f"{snapshot['symbol']} (微台)"
        ws1["B9"].value = m["current_price"]
        ws1["C9"].value = m["diff"]
        ws1["D9"].value = f"{m['pct']:+.2f}%"
        ws1["E9"].value = m["open"]
        ws1["F9"].value = f"{m['high']:,.1f} / {m['low']:,.1f}"
        ws1["G9"].value = m["volume_sum"]
        ws1["H9"].value = m["vwap"]
        ws1["I9"].value = m["prev_close"]
        exp_range = f"{m['range_low']:,.0f} ~ {m['range_high']:,.0f}" if m.get("range_low") else "--"
        ws1["J9"].value = exp_range

        sig_display = f"🚀 {orb['action_desc']}" if orb["action"] != "HOLD" else "⏳ 觀望 / 箱型形成中"
        ws1["A13"].value = sig_display
        ws1["B13"].value = orb["range_high"] or "--"
        ws1["C13"].value = orb["range_low"] or "--"
        ws1["D13"].value = orb["range_width"] or "--"
        ws1["J13"].value = orb["reason"] or "--"

        ws1["A17"].value = p["direction"]
        ws1["B17"].value = p["qty"]
        ws1["C17"].value = p["entry_price"]
        ws1["D17"].value = p["pts"]
        ws1["E17"].value = p["pnl"]
        ws1["F17"].value = p["stop_loss"]
        ws1["G17"].value = p["stop_mode"]
        ws1["H17"].value = p["take_profit"]
        ws1["I17"].value = p["realized_pnl"]

        # 更新 Sheet 2
        try:
            ws2 = wb["📑 歷史成交與日誌"]
            for r_idx, t in enumerate(snapshot.get("trades", [])[-12:], start=3):
                ws2.cell(row=r_idx, column=1, value=t.get("trade_id", r_idx - 2))
                ws2.cell(row=r_idx, column=2, value=t.get("entry_time", ""))
                ws2.cell(row=r_idx, column=3, value=t.get("exit_time", ""))
                ws2.cell(row=r_idx, column=4, value=t.get("symbol", "TMF"))
                ws2.cell(row=r_idx, column=5, value=t.get("side", ""))
                ws2.cell(row=r_idx, column=6, value=t.get("entry_price", 0))
                ws2.cell(row=r_idx, column=7, value=t.get("exit_price", 0))
                ws2.cell(row=r_idx, column=8, value=t.get("profit_loss_pts", 0))
                ws2.cell(row=r_idx, column=9, value=t.get("profit_loss_twd", 0))
                ws2.cell(row=r_idx, column=10, value=t.get("exit_reason", ""))
        except Exception:
            pass

        # 更新 Sheet 3
        try:
            ws3 = wb["📈 最新5分K與指標"]
            for r_idx, k in enumerate(snapshot.get("recent_5m", [])[-30:], start=3):
                ws3.cell(row=r_idx, column=1, value=k.get("ts", ""))
                ws3.cell(row=r_idx, column=2, value=k.get("open", 0))
                ws3.cell(row=r_idx, column=3, value=k.get("high", 0))
                ws3.cell(row=r_idx, column=4, value=k.get("low", 0))
                ws3.cell(row=r_idx, column=5, value=k.get("close", 0))
                ws3.cell(row=r_idx, column=6, value=k.get("volume", 0))
        except Exception:
            pass

        wb.save(filepath)
        return True
    except Exception as e:
        logger.debug("save_snapshot_to_xlsx_file 跳過 (可能檔案被 Excel 鎖定中): %s", e)
        return False


def sync_snapshot_to_excel_app(
    snapshot: Dict[str, Any],
    excel_path: str | Path = EXCEL_FILENAME,
    pt: Optional[PaperTradingEngine] = None,
) -> bool:
    """使用 win32com 將 snapshot 即時寫入正開啟中的 Excel 視窗"""
    try:
        import win32com.client
    except ImportError:
        logger.error("未安裝 pywin32，無法啟用 win32com 即時同步")
        return False

    abs_path = os.path.abspath(excel_path)
    if not os.path.exists(abs_path):
        build_excel_template(abs_path)

    xl = None
    wb = None
    try:
        # 連接既有或開啟新的 Excel
        try:
            xl = win32com.client.GetActiveObject("Excel.Application")
        except Exception:
            xl = win32com.client.Dispatch("Excel.Application")

        xl.Visible = True
        xl.DisplayAlerts = False

        # 尋找使用者是否已自行打開該活頁簿；如果使用者關閉了，絕對不要擅自重新打開
        for open_wb in xl.Workbooks:
            if open_wb.FullName.lower() == abs_path.lower():
                wb = open_wb
                break
        if wb is None:
            # 使用者已關閉活頁簿，不自動彈出干擾使用者
            return False

        ws1 = wb.Worksheets("📊 戰情室即時看盤")

        # 讀取並執行雙向控制項
        if pt is not None:
            execute_excel_commands(ws1, pt)

        # 填入 Sheet 1 狀態列
        m = snapshot["market"]
        orb = snapshot["orb"]
        p = snapshot["paper_trader"]

        ws1.Range("B2").Value = snapshot["timestamp"]
        ws1.Range("D2").Value = snapshot["session_label"]
        ws1.Range("F2").Value = snapshot["quote_status"]
        ws1.Range("J2").Value = p["equity"]

        # 填入 Row 9 行情
        diff_str = f"{m['diff']:+.1f}"
        pct_str = f"{m['pct']:+.2f}%"
        ws1.Range("A9").Value = f"{snapshot['symbol']} (微台)"
        ws1.Range("B9").Value = m["current_price"]
        ws1.Range("C9").Value = diff_str
        ws1.Range("D9").Value = pct_str
        ws1.Range("E9").Value = m["open"]
        ws1.Range("F9").Value = f"{m['high']:,.1f} / {m['low']:,.1f}"
        ws1.Range("G9").Value = f"{m['volume_sum']:,.0f}"
        ws1.Range("H9").Value = f"{m['vwap']:,.1f}"
        ws1.Range("I9").Value = f"{m['prev_close']:,.1f}"
        exp_range = f"{m['range_low']:,.0f} ~ {m['range_high']:,.0f}" if m.get("range_low") else "--"
        ws1.Range("J9").Value = exp_range

        # 漲跌顏色
        if m["diff"] > 0:
            ws1.Range("B9:D9").Font.Color = 0x0000FF  # 紅色 (BGR)
        elif m["diff"] < 0:
            ws1.Range("B9:D9").Font.Color = 0x008800  # 綠色
        else:
            ws1.Range("B9:D9").Font.Color = 0x000000

        # 填入 Row 13 ORB 突破
        sig_display = f"🚀 {orb['action_desc']}" if orb["action"] != "HOLD" else "⏳ 觀望 / 箱型形成中"
        ws1.Range("A13").Value = sig_display
        ws1.Range("B13").Value = f"{orb['range_high']:,.1f}" if orb["range_high"] else "--"
        ws1.Range("C13").Value = f"{orb['range_low']:,.1f}" if orb["range_low"] else "--"
        ws1.Range("D13").Value = f"{orb['range_width']:,.1f}" if orb["range_width"] else "--"

        dist_str = "--"
        if orb["range_high"] and orb["range_low"]:
            if m["current_price"] > orb["range_high"]:
                dist_str = f"突破上緣 +{m['current_price'] - orb['range_high']:.1f}"
            elif m["current_price"] < orb["range_low"]:
                dist_str = f"跌破下緣 -{orb['range_low'] - m['current_price']:.1f}"
            else:
                dist_str = f"距上緣 {orb['range_high'] - m['current_price']:.1f} / 距下緣 {m['current_price'] - orb['range_low']:.1f}"
        ws1.Range("E13").Value = dist_str
        ws1.Range("F13").Value = f"{orb['vol_ratio']:.2f}x" if orb["vol_ratio"] else "--"
        ws1.Range("G13").Value = f"{orb['body_ratio']:.1%}" if orb["body_ratio"] else "--"
        ws1.Range("H13").Value = orb["quality_note"] or "--"
        ws1.Range("I13").Value = orb["breakout_time"] or "--"
        ws1.Range("J13").Value = orb["reason"] or "--"

        # 填入 Row 17 模擬倉持倉與停損
        ws1.Range("A17").Value = p["direction"]
        ws1.Range("B17").Value = p["qty"]
        ws1.Range("C17").Value = p["entry_price"]
        ws1.Range("D17").Value = f"{p['pts']:+.1f}" if p["qty"] > 0 else "0.0"
        ws1.Range("E17").Value = p["pnl"]
        ws1.Range("F17").Value = p["stop_loss"]
        ws1.Range("G17").Value = p["stop_mode"]
        ws1.Range("H17").Value = p["take_profit"]
        ws1.Range("I17").Value = p["realized_pnl"]
        ws1.Range("J17").Value = "日夜雙盤多波段ORB+保本停損"

        # Sheet 2 歷史明細
        try:
            ws2 = wb.Worksheets("📑 歷史成交與日誌")
            for r_idx, t in enumerate(snapshot.get("trades", [])[-12:], start=3):
                ws2.Range(f"A{r_idx}").Value = t.get("trade_id", r_idx - 2)
                ws2.Range(f"B{r_idx}").Value = t.get("entry_time", "")
                ws2.Range(f"C{r_idx}").Value = t.get("exit_time", "")
                ws2.Range(f"D{r_idx}").Value = t.get("symbol", "TMF")
                ws2.Range(f"E{r_idx}").Value = t.get("side", "")
                ws2.Range(f"F{r_idx}").Value = t.get("entry_price", 0)
                ws2.Range(f"G{r_idx}").Value = t.get("exit_price", 0)
                ws2.Range(f"H{r_idx}").Value = t.get("profit_loss_pts", 0)
                ws2.Range(f"I{r_idx}").Value = t.get("profit_loss_twd", 0)
                ws2.Range(f"J{r_idx}").Value = t.get("exit_reason", "")
        except Exception as e:
            logger.debug("更新 Sheet 2 失敗: %s", e)

        # Sheet 3 K 棒
        try:
            ws3 = wb.Worksheets("📈 最新5分K與指標")
            for r_idx, k in enumerate(snapshot.get("recent_5m", [])[-30:], start=3):
                ws3.Range(f"A{r_idx}").Value = k.get("ts", "")
                ws3.Range(f"B{r_idx}").Value = k.get("open", 0)
                ws3.Range(f"C{r_idx}").Value = k.get("high", 0)
                ws3.Range(f"D{r_idx}").Value = k.get("low", 0)
                ws3.Range(f"E{r_idx}").Value = k.get("close", 0)
                ws3.Range(f"F{r_idx}").Value = k.get("volume", 0)
        except Exception as e:
            logger.debug("更新 Sheet 3 失敗: %s", e)

        return True
    except Exception as e:
        logger.warning("同步 Excel 活頁簿時發生例外: %s", e)
        return False


def main():
    parser = argparse.ArgumentParser(description="Excel 戰情室即時看盤與橋接系統")
    parser.add_argument("--init", action="store_true", help="僅生成 AI_Trading_WarRoom.xlsx 範本")
    parser.add_argument("--once", action="store_true", help="抓取一次行情並更新至 Excel")
    parser.add_argument("--loop", action="store_true", help="持續以指定間隔秒數迴圈同步至 Excel")
    parser.add_argument("--interval", type=int, default=10, help="自動更新間隔秒數 (預設 10 秒)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.init:
        path = build_excel_template(EXCEL_FILENAME)
        print(f"✅ Excel 戰情室範本已成功建立: {path}")
        return

    # 初始化 PaperTrader
    c_spec = ContractSpec.from_json("TMF")
    strat = default_ai_strategy("TMF")
    pt = PaperTradingEngine(
        strategy=strat,
        contract=c_spec,
        initial_capital=100_000.0,
        risk_config=RiskConfig(max_daily_loss=DEFAULT_MAX_DAILY_LOSS, max_contracts=1),
    )
    pt.load_state_from_disk()

    if args.once or (not args.loop and not args.init):
        snap = collect_warroom_snapshot(pt=pt)
        saved_file = save_snapshot_to_xlsx_file(snap, EXCEL_FILENAME)
        ok = sync_snapshot_to_excel_app(snap, EXCEL_FILENAME, pt=pt)
        print(f"✅ [{snap['timestamp']}] 單次更新完成 (寫入檔案: {saved_file}, 同步至開啟中Excel: {ok}) - 微台現價: {snap['market']['current_price']}, 訊號: {snap['orb']['action']}")
        return

    # 迴圈定時同步
    print(f"🚀 啟動 Excel 戰情室即時監控迴圈 (每 {args.interval} 秒刷新)... (按 Ctrl+C 可停止)")
    while True:
        try:
            snap = collect_warroom_snapshot(pt=pt)
            save_snapshot_to_xlsx_file(snap, EXCEL_FILENAME)
            ok = sync_snapshot_to_excel_app(snap, EXCEL_FILENAME, pt=pt)
            print(f"[{snap['timestamp']}] 現價: {snap['market']['current_price']} | 訊號: {snap['orb']['action']} | 部位: {snap['paper_trader']['direction']} (同步開啟中Excel: {ok})")
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n🛑 已由使用者中斷監控。")
            break
        except Exception as e:
            print(f"⚠️ 監控迴圈例外: {e}")
            time.sleep(args.interval)


if __name__ == "__main__":
    main()

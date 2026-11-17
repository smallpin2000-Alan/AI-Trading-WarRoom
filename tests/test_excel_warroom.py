"""
tests/test_excel_warroom.py
單元測試：驗證 Excel 戰情室範本建立、快照數據收集、與雙向指令平倉邏輯
"""
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import openpyxl
import polars as pl
import pytest

from src.ui.excel_warroom import (
    build_excel_template,
    collect_warroom_snapshot,
    execute_excel_commands,
)
from src.core.paper_trader import (
    PaperTradingEngine,
    RiskConfig,
    default_ai_strategy,
    OpenPosition,
)
from src.core.engine import ContractSpec


def test_build_excel_template(tmp_path):
    target_xlsx = tmp_path / "test_warroom.xlsx"
    built_path = build_excel_template(target_xlsx)

    assert built_path.exists()
    wb = openpyxl.load_workbook(built_path)
    sheet_names = wb.sheetnames

    assert "📊 戰情室即時看盤" in sheet_names
    assert "📑 歷史成交與日誌" in sheet_names
    assert "📈 最新5分K與指標" in sheet_names

    ws1 = wb["📊 戰情室即時看盤"]
    assert "【AI 智慧看盤與自動交易戰情室" in str(ws1["A1"].value)
    assert ws1["B5"].value == "ON"
    assert ws1["D5"].value == "無"
    assert ws1["F5"].value == 10
    assert "微台" in str(ws1["A9"].value)


def test_collect_warroom_snapshot(tmp_path):
    strat = default_ai_strategy("TMF")
    c_spec = ContractSpec.from_json("TMF")
    pt = PaperTradingEngine(strategy=strat, contract=c_spec, initial_capital=100_000.0)

    snapshot = collect_warroom_snapshot(feed=None, pt=pt, symbol="TMFR1")

    assert "timestamp" in snapshot
    assert snapshot["symbol"] == "TMFR1"
    assert "market" in snapshot
    assert "orb" in snapshot
    assert "paper_trader" in snapshot
    assert "trades" in snapshot
    assert "logs" in snapshot

    m = snapshot["market"]
    assert m["current_price"] > 0
    assert "diff" in m
    assert "vwap" in m

    orb = snapshot["orb"]
    assert orb["action"] in ["BUY_LONG", "SELL_SHORT", "HOLD", "WAIT"]
    assert "action_desc" in orb

    p = snapshot["paper_trader"]
    assert "direction" in p
    assert "equity" in p


def test_execute_excel_commands_flatten():
    strat = default_ai_strategy("TMF")
    c_spec = ContractSpec.from_json("TMF")
    pt = PaperTradingEngine(strategy=strat, contract=c_spec, initial_capital=100_000.0)

    # 模擬有一個持倉 (direction: 1 為多, -1 為空)
    pt.position = OpenPosition(
        direction=1,
        contracts=1,
        entry_price=22800.0,
        entry_time="2026-11-17 08:45:00",
        stop_loss_price=22700.0,
        take_profit_price=23000.0,
    )

    mock_ws = MagicMock()
    mock_ws.Range.side_effect = lambda cell: {
        "B5": MagicMock(Value="ON"),
        "D5": MagicMock(Value="平倉"),
    }[cell]

    res = execute_excel_commands(mock_ws, pt)

    assert res["autopilot"] is True
    assert res["flatten_triggered"] is True
    assert pt.position is None  # 部位已被平倉
    assert len(pt.closed_trades) == 1
    trade = pt.closed_trades[0]
    reason = trade.exit_reason if hasattr(trade, "exit_reason") else trade.get("exit_reason", "")
    assert "excel_manual_flatten" in reason

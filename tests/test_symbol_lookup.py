# -*- coding: utf-8 -*-
import pytest
from src.data.symbol_lookup import get_symbol_name, format_symbol_display
from src.ui.dashboard import load_watchlist, add_symbol_to_watchlist


def test_builtin_and_futures_symbol_lookup():
    # 核心權值股
    assert get_symbol_name("2330") == "台積電"
    assert get_symbol_name("2317") == "鴻海"
    assert get_symbol_name("2603") == "長榮"
    assert get_symbol_name("0050") == "元大台灣50"

    # 期貨代碼
    assert get_symbol_name("TMFR1") == "微台指近月"
    assert get_symbol_name("TXFR1") == "大台指近月"
    assert get_symbol_name("MTXR1") == "小台指近月"


def test_format_symbol_display():
    assert format_symbol_display("2330") == "2330 (台積電)"
    assert format_symbol_display("2603") == "2603 (長榮)"
    assert format_symbol_display("TMFR1") == "TMFR1 (微台指近月)"
    # 已有格式者維持原樣
    assert format_symbol_display("2330 (台積電)") == "2330 (台積電)"


def test_add_symbol_auto_resolves_name(tmp_path):
    wl_file = tmp_path / "watchlist.json"
    # 不傳入名稱時，系統自動帶出「長榮」
    ok, msg, items = add_symbol_to_watchlist("2603", "", path=wl_file)
    assert ok is True
    assert items[0] == "2603 (長榮)"

    # 新增廣達，自動帶出「廣達」
    ok2, _, items2 = add_symbol_to_watchlist("2382", "", path=wl_file)
    assert ok2 is True
    assert items2[0] == "2382 (廣達)"

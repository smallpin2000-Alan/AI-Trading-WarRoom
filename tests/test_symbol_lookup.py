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


def test_save_disk_cache_is_atomic_and_logs_errors(tmp_path, monkeypatch, caplog):
    import json
    import logging
    import src.data.symbol_lookup as sl

    cache_file = tmp_path / "cfg" / "symbol_names_cache.json"
    monkeypatch.setattr(sl, "CACHE_PATH", cache_file)
    monkeypatch.setitem(sl._memory_cache, "9999", "測試股")
    sl._save_disk_cache()
    assert json.loads(cache_file.read_text(encoding="utf-8"))["9999"] == "測試股"
    assert [p.name for p in cache_file.parent.iterdir()] == ["symbol_names_cache.json"]

    # 取代失敗：舊檔保持完整、暫存檔清除、例外被記錄而非靜默吞掉
    before = cache_file.read_bytes()

    def boom(src, dst):
        raise PermissionError("file in use")

    monkeypatch.setattr(sl.os, "replace", boom)
    monkeypatch.setitem(sl._memory_cache, "8888", "另一檔")
    with caplog.at_level(logging.WARNING, logger="src.data.symbol_lookup"):
        sl._save_disk_cache()
    assert cache_file.read_bytes() == before
    assert [p.name for p in cache_file.parent.iterdir()] == ["symbol_names_cache.json"]
    assert any("symbol_names_cache" in r.getMessage() or "快取" in r.getMessage() for r in caplog.records)

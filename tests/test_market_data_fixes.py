# tests/test_market_data_fixes.py
"""
行情與資料模組修正的回歸測試（一律使用 tmp_path 或 store_dir=None，絕不寫入正式 data/）。
"""
from __future__ import annotations

import logging
import threading
import time as time_mod
from datetime import datetime, timedelta, time as dtime
from types import SimpleNamespace

import polars as pl
import pytest

import src.broker.taishin_quote as tq
from src.broker.taishin_quote import TaishinQuoteFeed, find_outlier_kbar_keys
from src.core.bar_builder import TickData, SharedBarBuilder
from src.data.kbar_store import KBarStore


def _bar(dt, price, vol=10, **extra):
    b = {"timestamp": dt, "open": price, "high": price + 1, "low": price - 1, "close": price, "volume": vol}
    b.update(extra)
    return b


def _series(start, n, base, step=0.0):
    return {
        start + timedelta(minutes=i): _bar(start + timedelta(minutes=i), base + step * i)
        for i in range(n)
    }


def _ts(dt: datetime) -> int:
    """台灣時間 (無 tzinfo) -> Unix timestamp"""
    return int(dt.replace(tzinfo=tq.TZ_TW).timestamp())


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _patch_http(monkeypatch, cnyes_data=None, yahoo=None, mis_quote=None):
    """以假物件取代 requests，測試中絕不連外"""
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        if "yahoo" in url:
            return _FakeResp(yahoo or {})
        if "cnyes" in url:
            return _FakeResp({"data": cnyes_data or {}})
        return _FakeResp({})

    def fake_post(url, **kwargs):
        calls.append(url)
        return _FakeResp(mis_quote or {"RtData": {"QuoteList": []}})

    monkeypatch.setattr(tq, "requests", SimpleNamespace(get=fake_get, post=fake_post))
    return calls


# ==================== KBarStore：合併寫入 / 明確刪除 / 原子寫入 / 壞檔隔離 ====================

def test_store_delete_keys_removes_only_listed_bars(tmp_path):
    store = KBarStore(data_dir=tmp_path)
    t0 = datetime(2026, 11, 10, 9, 1)
    bars = _series(t0, 3, 1000.0)
    store.save_kbars("2330", bars)

    t2 = t0 + timedelta(minutes=2)
    cnt = store.save_kbars("2330", {t2: _bar(t2, 1005.0)}, delete_keys={t0})
    assert cnt == 2
    loaded = store.load_kbars_dict("2330")
    assert sorted(loaded) == [t0 + timedelta(minutes=1), t2]
    assert loaded[t2]["close"] == 1005.0


def test_store_delete_keys_without_new_bars(tmp_path):
    store = KBarStore(data_dir=tmp_path)
    t0 = datetime(2026, 11, 10, 9, 1)
    store.save_kbars("2330", _series(t0, 3, 1000.0))
    store.save_kbars("2330", {}, delete_keys=[t0])
    assert t0 not in store.load_kbars_dict("2330")
    assert len(store.load_kbars_dict("2330")) == 2


def test_store_skips_synthetic_bars(tmp_path):
    store = KBarStore(data_dir=tmp_path)
    t0 = datetime(2026, 11, 10, 9, 1)
    t1 = t0 + timedelta(minutes=1)
    store.save_kbars("2330", {t0: _bar(t0, 1000.0), t1: _bar(t1, 1001.0, synthetic=True)})
    assert sorted(store.load_kbars_dict("2330")) == [t0]


def test_store_write_failure_keeps_old_file_and_cleans_tmp(tmp_path, monkeypatch):
    store = KBarStore(data_dir=tmp_path)
    t0 = datetime(2026, 11, 10, 9, 1)
    store.save_kbars("2330", _series(t0, 3, 1000.0))
    before = store._get_file_path("2330").read_bytes()

    def boom(self, *args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(pl.DataFrame, "write_parquet", boom)
    t9 = t0 + timedelta(minutes=9)
    with pytest.raises(RuntimeError):
        store.save_kbars("2330", {t9: _bar(t9, 1009.0)})
    monkeypatch.undo()

    assert store._get_file_path("2330").read_bytes() == before
    assert len(store.load_kbars_dict("2330")) == 3
    assert [p.name for p in tmp_path.iterdir()] == ["taishin_kbars_2330_1m.parquet"]


def test_store_replace_failure_keeps_old_file(tmp_path, monkeypatch):
    import src.data.kbar_store as ks

    store = KBarStore(data_dir=tmp_path)
    t0 = datetime(2026, 11, 10, 9, 1)
    store.save_kbars("2330", _series(t0, 3, 1000.0))
    before = store._get_file_path("2330").read_bytes()

    def locked(src, dst):
        raise PermissionError("file in use")

    monkeypatch.setattr(ks.os, "replace", locked)
    monkeypatch.setattr(ks, "_REPLACE_RETRY_DELAY_SEC", 0.0)
    t9 = t0 + timedelta(minutes=9)
    with pytest.raises(PermissionError):
        store.save_kbars("2330", {t9: _bar(t9, 1009.0)})
    monkeypatch.undo()

    assert store._get_file_path("2330").read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["taishin_kbars_2330_1m.parquet"]


def test_store_corrupt_file_is_quarantined_not_overwritten(tmp_path, caplog):
    store = KBarStore(data_dir=tmp_path)
    path = store._get_file_path("2330")
    path.write_bytes(b"this is not a parquet file")

    with caplog.at_level(logging.ERROR):
        assert store.load_kbars_dict("2330") == {}
    corrupt = tmp_path / (path.name + ".corrupt")
    assert corrupt.exists() and not path.exists()
    assert corrupt.read_bytes() == b"this is not a parquet file"
    assert any("corrupt" in r.getMessage() for r in caplog.records)

    # 之後的存檔會建立新檔，壞檔仍保留
    t0 = datetime(2026, 11, 10, 9, 1)
    store.save_kbars("2330", {t0: _bar(t0, 1000.0)})
    assert len(store.load_kbars_dict("2330")) == 1
    assert corrupt.exists()


def test_store_transient_read_error_aborts_save(tmp_path, monkeypatch):
    """讀取既有檔發生暫時性 I/O 錯誤時，不可當成沒有歷史而整檔覆蓋"""
    import src.data.kbar_store as ks

    store = KBarStore(data_dir=tmp_path)
    t0 = datetime(2026, 11, 10, 9, 1)
    store.save_kbars("2330", _series(t0, 3, 1000.0))
    before = store._get_file_path("2330").read_bytes()

    def locked(*args, **kwargs):
        raise PermissionError("file in use")

    monkeypatch.setattr(ks.pl, "read_parquet", locked)
    t9 = t0 + timedelta(minutes=9)
    with pytest.raises(OSError):
        store.save_kbars("2330", {t9: _bar(t9, 1009.0)})
    # load 仍維持「回傳空字典」的既有控制流程，但不可動到檔案
    assert store.load_kbars_dict("2330") == {}
    monkeypatch.undo()

    assert store._get_file_path("2330").read_bytes() == before
    assert not (tmp_path / "taishin_kbars_2330_1m.parquet.corrupt").exists()


def test_store_concurrent_saves_do_not_lose_bars(tmp_path):
    t0 = datetime(2026, 11, 10, 9, 0)
    errors = []

    def worker(idx):
        try:
            store = KBarStore(data_dir=tmp_path)
            for j in range(5):
                dt = t0 + timedelta(minutes=idx * 5 + j)
                store.save_kbars("2330", {dt: _bar(dt, 1000.0 + idx)})
        except Exception as ex:  # pragma: no cover - 失敗時由下方 assert 呈現
            errors.append(ex)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert errors == []
    assert len(KBarStore(data_dir=tmp_path).load_kbars_dict("2330")) == 30


# ==================== 項目 1：切換商品時舊商品 K 棒不可污染新商品 ====================

def _ta_rec(product, hhmm, price, date="20261110"):
    return SimpleNamespace(
        KBar=SimpleNamespace(
            Date=date, Product=product, TimeSn_Dply=hhmm,
            OPrice=price, HPrice=price + 1, LPrice=price - 1, CPrice=price, Quantity=10,
        )
    )


def test_stale_ta_callbacks_after_switch_are_dropped(tmp_path):
    feed = TaishinQuoteFeed(store_dir=tmp_path)
    feed.symbol = "2330"
    feed._on_ta_rcv_done(0, [_ta_rec("2330", 901, 1050.0)])
    assert len(feed.kbars_1m) == 1

    feed.switch_symbol("2317")
    assert feed.symbol == "2317" and feed.kbars_1m == {}

    # 舊商品 (2330) 遲到的歷史回補與即時更新都必須被丟棄
    feed._on_ta_rcv_done(0, [_ta_rec("2330", 902, 1051.0), _ta_rec("2330", 903, 1052.0)])
    feed._on_ta_update(0, None, _ta_rec("2330", 904, 1053.0))
    assert feed.kbars_1m == {}
    assert feed.store.load_kbars_dict("2317") == {}
    assert feed.ta_rcv_done is False

    # 新商品的資料正常接收
    feed._on_ta_rcv_done(0, [_ta_rec("2317", 901, 210.0)])
    feed._on_ta_update(0, None, _ta_rec("2317", 902, 211.0))
    assert len(feed.kbars_1m) == 2
    assert [b["close"] for b in feed.store.load_kbars_dict("2317").values()] == [210.0]
    # 2330 的 parquet 不受影響
    assert len(feed.store.load_kbars_dict("2330")) == 1


def test_ta_callbacks_without_product_field_still_accepted():
    """K 棒紀錄沒有 Product 欄位（或為空）時維持既有行為"""
    feed = TaishinQuoteFeed(store_dir=None)
    feed.symbol = "2330"
    feed.switch_symbol("2317")
    rec = SimpleNamespace(
        KBar=SimpleNamespace(Date="20261110", TimeSn_Dply=901, OPrice=210.0, HPrice=211.0, LPrice=209.0, CPrice=210.0, Quantity=5)
    )
    feed._on_ta_rcv_done(0, [rec])
    assert len(feed.kbars_1m) == 1


def test_switch_back_to_old_symbol_accepts_its_bars_again():
    feed = TaishinQuoteFeed(store_dir=None)
    feed.symbol = "2330"
    feed.switch_symbol("2317")
    feed.switch_symbol("2330")
    feed._on_ta_rcv_done(0, [_ta_rec("2330", 901, 1050.0)])
    assert len(feed.kbars_1m) == 1
    feed._on_ta_rcv_done(0, [_ta_rec("2317", 902, 210.0)])
    assert len(feed.kbars_1m) == 1


def test_start_while_connecting_clears_old_symbol_cache():
    feed = TaishinQuoteFeed(store_dir=None)
    feed.symbol = "2330"
    t0 = datetime(2026, 11, 10, 9, 1)
    feed.kbars_1m = {t0: _bar(t0, 1050.0)}
    feed.latest_tick = TickData(t0, 1050.0, 1)
    feed._connect_in_progress = True

    feed.start("A123456789", "pw", "2317")

    assert feed.symbol == "2317"
    assert feed.kbars_1m == {}
    assert feed.latest_tick is None
    # 舊商品遲到的回呼不可寫入
    feed._on_ta_rcv_done(0, [_ta_rec("2330", 902, 1051.0)])
    assert feed.kbars_1m == {}


def test_sync_aborts_when_symbol_switched_midway(tmp_path, monkeypatch):
    """HTTP 同步途中若商品被切換，舊商品資料不可寫入新商品的快取或 parquet"""
    feed = TaishinQuoteFeed(store_dir=tmp_path)
    feed.symbol = "2330"
    now = datetime(2026, 11, 10, 10, 0, 0)
    monkeypatch.setattr(tq, "_now_tw", lambda: now)
    d0 = datetime(2026, 11, 10, 9, 1)

    def fake_get(url, **kwargs):
        if "cnyes" in url:
            feed.switch_symbol("2317")  # 模擬另一條執行緒在請求途中切換商品
            return _FakeResp({"data": {"t": [_ts(d0)], "o": [1000], "h": [1001], "l": [999], "c": [1000], "v": [5]}})
        return _FakeResp({})

    monkeypatch.setattr(tq, "requests", SimpleNamespace(get=fake_get, post=fake_get))
    feed._sync_realtime_kbars_and_quote("2330")

    assert feed.symbol == "2317"
    assert feed.kbars_1m == {}
    assert feed.store.load_kbars_dict("2317") == {}
    assert feed.store.load_kbars_dict("2330") == {}


# ==================== 項目 2 / 8：同步不縮短歷史、假 K 棒不落盤 ====================

def test_sync_merges_instead_of_replacing_history(tmp_path, monkeypatch):
    store = KBarStore(data_dir=tmp_path)
    old0 = datetime(2026, 11, 9, 9, 1)
    store.save_kbars("2330", _series(old0, 3, 1000.0))

    feed = TaishinQuoteFeed(store_dir=tmp_path)  # 記憶體為空（只有部分資料的情境）
    feed.symbol = "2330"
    now = datetime(2026, 11, 10, 10, 0, 0)
    monkeypatch.setattr(tq, "_now_tw", lambda: now)
    d0 = datetime(2026, 11, 10, 9, 1)
    d1 = datetime(2026, 11, 10, 9, 5)
    _patch_http(monkeypatch, cnyes_data={
        "t": [_ts(d0), _ts(d1)], "o": [1001, 1005], "h": [1002, 1006], "l": [1000, 1004], "c": [1001, 1005], "v": [7, 8],
    })

    feed._sync_realtime_kbars_and_quote("2330")

    on_disk = store.load_kbars_dict("2330")
    # 歷史 3 根仍在 + 今日 2 根真實 K 棒；內插與即時補值的假 K 棒不落盤
    assert sorted(on_disk) == [old0, old0 + timedelta(minutes=1), old0 + timedelta(minutes=2), d0, d1]

    # 記憶體中仍保有補值 K 棒供顯示，且都帶 synthetic 標記
    interp_keys = [d0 + timedelta(minutes=i) for i in (1, 2, 3)]
    for k in interp_keys:
        assert feed.kbars_1m[k].get("synthetic") is True
    now_bar = now.replace(second=0)
    assert feed.kbars_1m[now_bar].get("synthetic") is True
    assert not feed.kbars_1m[d0].get("synthetic")
    assert feed.get_bars("1m", session="day_only").height == len(feed.kbars_1m)


def test_sync_purges_outlier_from_parquet_but_keeps_history(tmp_path, monkeypatch):
    store = KBarStore(data_dir=tmp_path)
    base = datetime(2026, 11, 9, 9, 0)
    bars = _series(base, 30, 27000.0, step=1.0)
    bad_dt = base + timedelta(minutes=15)
    bars[bad_dt] = _bar(bad_dt, 2380.0)  # 混入的個股測試異常值
    store.save_kbars("TMFR1", bars)

    feed = TaishinQuoteFeed(store_dir=tmp_path, auto_load_store=True)
    assert feed.symbol == "TMFR1" and len(feed.kbars_1m) == 30
    monkeypatch.setattr(tq, "_now_tw", lambda: datetime(2026, 11, 10, 10, 0, 0))
    _patch_http(monkeypatch)

    feed._sync_realtime_kbars_and_quote("TMFR1")

    on_disk = store.load_kbars_dict("TMFR1")
    assert bad_dt not in on_disk
    assert len(on_disk) == 29
    assert bad_dt not in feed.kbars_1m


# ==================== 項目 7：相對偏離的異常值判斷 ====================

def test_outlier_normal_futures_and_stock_not_flagged():
    fut = _series(datetime(2026, 11, 10, 8, 45), 300, 20000.0, step=0.5)
    stock = _series(datetime(2026, 11, 10, 9, 0), 270, 1000.0, step=0.05)
    assert find_outlier_kbar_keys(fut) == []
    assert find_outlier_kbar_keys(stock) == []


def test_outlier_overnight_gap_not_flagged():
    """跨日跳空（接近漲跌停 10%）屬正常資料，不可誤殺"""
    day1 = _series(datetime(2026, 11, 9, 9, 0), 60, 1000.0)
    day2 = _series(datetime(2026, 11, 10, 9, 0), 60, 1099.0)
    assert find_outlier_kbar_keys({**day1, **day2}) == []


def test_outlier_injected_values_flagged():
    fut = _series(datetime(2026, 11, 10, 8, 45), 100, 20000.0, step=0.5)
    bad_fut = datetime(2026, 11, 10, 9, 30)
    fut[bad_fut] = _bar(bad_fut, 2380.0)
    assert find_outlier_kbar_keys(fut) == [bad_fut]

    stock = _series(datetime(2026, 11, 10, 9, 0), 100, 1000.0)
    bad_stock = datetime(2026, 11, 10, 9, 40)
    stock[bad_stock] = _bar(bad_stock, 20000.0)
    assert find_outlier_kbar_keys(stock) == [bad_stock]

    # 只有 low 異常（例如錯誤的最低價）也要剔除
    low_spike = datetime(2026, 11, 10, 9, 50)
    stock2 = _series(datetime(2026, 11, 10, 9, 0), 100, 1000.0)
    stock2[low_spike]["low"] = 10.0
    assert find_outlier_kbar_keys(stock2) == [low_spike]


def test_outlier_missing_low_or_close_is_invalid():
    t0 = datetime(2026, 11, 10, 9, 0)
    bars = _series(t0, 10, 1000.0)
    no_low = t0 + timedelta(minutes=2)
    no_close = t0 + timedelta(minutes=4)
    zero_low = t0 + timedelta(minutes=6)
    del bars[no_low]["low"]
    del bars[no_close]["close"]
    bars[zero_low]["low"] = 0.0
    assert find_outlier_kbar_keys(bars) == [no_low, no_close, zero_low]


def test_get_bars_filters_outliers_for_stock_and_keeps_low_priced_futures():
    feed = TaishinQuoteFeed(store_dir=None)
    feed.symbol = "2330"
    stock = _series(datetime(2026, 11, 10, 9, 0), 60, 1000.0)
    bad = datetime(2026, 11, 10, 9, 30)
    stock[bad] = _bar(bad, 20000.0)
    feed.kbars_1m = stock
    df = feed.get_bars("1m", session="day_only")
    assert df.height == 59
    assert df["high"].max() < 1100

    # 指數低於 8000 的正常期貨資料不可再被硬編碼門檻整批剔除
    feed2 = TaishinQuoteFeed(store_dir=None)
    feed2.symbol = "TXFR1"
    feed2.kbars_1m = _series(datetime(2026, 11, 10, 8, 45), 60, 7500.0)
    assert feed2.get_bars("1m", session="day_only").height == 60


# ==================== 項目 9：cnyes TXF 分 K 不可蓋掉 TMF/MXF 的真實 K 棒 ====================

def _cnyes_payload(dts, price, vol):
    return {
        "t": [_ts(d) for d in dts],
        "o": [price] * len(dts), "h": [price + 2] * len(dts), "l": [price - 2] * len(dts),
        "c": [price] * len(dts), "v": [vol] * len(dts),
    }


def test_futures_sync_does_not_overwrite_tmf_real_bars_with_txf(monkeypatch):
    feed = TaishinQuoteFeed(store_dir=None)
    feed.symbol = "TMFR1"
    t0 = datetime(2026, 11, 10, 9, 0)
    t1, t2, t3 = t0 + timedelta(minutes=1), t0 + timedelta(minutes=2), t0 + timedelta(minutes=3)
    feed.kbars_1m = {t0: _bar(t0, 27000.0, vol=5), t1: _bar(t1, 27001.0, vol=6), t3: _bar(t3, 27003.0, vol=7)}
    monkeypatch.setattr(tq, "_now_tw", lambda: datetime(2026, 11, 10, 9, 5, 0))
    _patch_http(monkeypatch, cnyes_data=_cnyes_payload([t0, t1, t2], 27010.0, 9999))

    feed._sync_realtime_kbars_and_quote("TMFR1")

    assert feed.kbars_1m[t0]["volume"] == 5 and feed.kbars_1m[t0]["close"] == 27000.0
    assert feed.kbars_1m[t1]["volume"] == 6
    assert feed.kbars_1m[t2]["close"] == 27010.0  # 只補缺少的 key
    assert t3 in feed.kbars_1m  # 既有真實 K 棒不可被先刪


def test_futures_sync_txf_still_refreshes_from_cnyes(monkeypatch):
    feed = TaishinQuoteFeed(store_dir=None)
    feed.symbol = "TXFR1"
    t0 = datetime(2026, 11, 10, 9, 0)
    feed.kbars_1m = {t0: _bar(t0, 27000.0, vol=5)}
    monkeypatch.setattr(tq, "_now_tw", lambda: datetime(2026, 11, 10, 9, 5, 0))
    _patch_http(monkeypatch, cnyes_data=_cnyes_payload([t0], 27010.0, 321))

    feed._sync_realtime_kbars_and_quote("TXFR1")
    assert feed.kbars_1m[t0]["volume"] == 321 and feed.kbars_1m[t0]["close"] == 27010.0


def test_futures_sync_malformed_response_keeps_existing_bars(monkeypatch):
    """新資料驗證失敗（欄位長度不一致）時，既有 K 棒不可先被刪掉"""
    feed = TaishinQuoteFeed(store_dir=None)
    feed.symbol = "TXFR1"
    t0 = datetime(2026, 11, 10, 9, 0)
    t1 = t0 + timedelta(minutes=1)
    feed.kbars_1m = {t0: _bar(t0, 27000.0, vol=5), t1: _bar(t1, 27001.0, vol=6)}
    monkeypatch.setattr(tq, "_now_tw", lambda: datetime(2026, 11, 10, 9, 5, 0))
    payload = _cnyes_payload([t0, t1], 27010.0, 321)
    payload["c"] = [27010.0]  # 少一筆
    _patch_http(monkeypatch, cnyes_data=payload)

    feed._sync_realtime_kbars_and_quote("TXFR1")
    assert feed.kbars_1m[t0]["volume"] == 5
    assert feed.kbars_1m[t1]["volume"] == 6


# ==================== 項目 5 / 6：K 棒時間解析 ====================

def test_parse_kbar_datetime_midnight_bar_is_kept():
    assert TaishinQuoteFeed._parse_kbar_datetime("20261110", 0) == datetime(2026, 11, 10, 0, 0)
    assert TaishinQuoteFeed._parse_kbar_datetime("20261110", None) is None
    assert TaishinQuoteFeed._parse_kbar_datetime("20261110", -1) is None
    assert TaishinQuoteFeed._parse_kbar_datetime("20261110", 2460) is None


def test_upsert_kbar_midnight_and_missing_time():
    feed = TaishinQuoteFeed(store_dir=None)
    feed._on_ta_rcv_done(0, [_ta_rec("TMFR1", 0, 27000.0, date="20261110")])
    assert datetime(2026, 11, 10, 0, 0) in feed.kbars_1m

    feed2 = TaishinQuoteFeed(store_dir=None)
    no_time = SimpleNamespace(KBar=SimpleNamespace(Date="20261110", OPrice=1.0, HPrice=1.0, LPrice=1.0, CPrice=1.0, Quantity=1))
    feed2._on_ta_rcv_done(0, [no_time])
    assert feed2.kbars_1m == {}


def test_parse_kbar_datetime_night_session_uses_real_calendar_day():
    p = TaishinQuoteFeed._parse_kbar_datetime
    # 2026-11-09 為週一交易日：其夜盤自週五 (11/06) 15:00 開始，凌晨段落在週六 (11/07)
    assert datetime(2026, 11, 9).weekday() == 0
    assert p("20261109", 1500) == datetime(2026, 11, 6, 15, 0)
    assert p("20261109", 2359) == datetime(2026, 11, 6, 23, 59)
    assert p("20261109", 0) == datetime(2026, 11, 7, 0, 0)
    assert p("20261109", 300) == datetime(2026, 11, 7, 3, 0)
    assert p("20261109", 500) == datetime(2026, 11, 7, 5, 0)
    assert p("20261109", 845) == datetime(2026, 11, 9, 8, 45)
    assert p("20261109", 1345) == datetime(2026, 11, 9, 13, 45)
    # 週二交易日：夜盤自週一 15:00 開始，凌晨段即週二當天
    assert p("20261110", 1500) == datetime(2026, 11, 9, 15, 0)
    assert p("20261110", 300) == datetime(2026, 11, 10, 3, 0)
    assert p("20261110", 900) == datetime(2026, 11, 10, 9, 0)


def test_night_session_start_date_skips_weekend():
    f = TaishinQuoteFeed._night_session_start_date
    assert f(datetime(2026, 11, 10, 16, 0)) == datetime(2026, 11, 10).date()   # 週二夜盤
    assert f(datetime(2026, 11, 11, 2, 0)) == datetime(2026, 11, 10).date()    # 週三凌晨
    assert f(datetime(2026, 11, 7, 3, 0)) == datetime(2026, 11, 6).date()      # 週六凌晨 -> 週五
    assert f(datetime(2026, 11, 7, 16, 0)) == datetime(2026, 11, 6).date()     # 週六下午 -> 週五
    assert f(datetime(2026, 11, 8, 20, 0)) == datetime(2026, 11, 6).date()     # 週日晚上 -> 週五


# ==================== 項目 10：斷線後可重連、不造成登入風暴 ====================

def test_ta_disconnect_marks_reconnect_needed_with_cooldown(monkeypatch):
    feed = TaishinQuoteFeed(store_dir=None)
    feed.symbol = "TMFR1"
    feed.current_user_id, feed.current_password = "A123456789", "pw"
    feed._on_ta_conn(True)
    assert feed.ta_connected and feed.connected and not feed._needs_reconnect

    feed._on_ta_conn(False)
    assert feed.ta_connected is False
    assert feed._needs_reconnect is True

    calls = []
    monkeypatch.setattr(feed, "start", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(feed, "_sync_realtime_kbars_and_quote", lambda s: 0)

    # 寬限期內不重連（避免與 API 自身的自動重連搶登入）
    feed.ensure_fresh_data("TMFR1")
    assert calls == []

    # 寬限期過後觸發一次重連
    feed._reconnect_not_before = 0.0
    feed.ensure_fresh_data("TMFR1")
    assert len(calls) == 1

    # 連線恢復後不再重連
    feed._on_ta_conn(True)
    feed._reconnect_not_before = 0.0
    feed.ensure_fresh_data("TMFR1")
    assert len(calls) == 1


def test_reconnect_start_goes_through_full_connect_and_sets_cooldown(monkeypatch):
    started = []

    class _FakeThread:
        def __init__(self, target=None, args=(), daemon=None, **kwargs):
            self.target = target

        def start(self):
            started.append(self.target)

    feed = TaishinQuoteFeed(store_dir=None)
    feed.symbol = "TMFR1"
    feed.current_user_id, feed.current_password = "A123456789", "pw"
    feed.connected = True
    feed.ta = object()
    feed._needs_reconnect = True
    monkeypatch.setattr(feed, "_sync_realtime_kbars_and_quote", lambda s: 0)
    monkeypatch.setattr(tq.threading, "Thread", _FakeThread)

    feed.start("A123456789", "pw", "TMFR1")

    assert len(started) == 2  # TechAnalysis + Sol_D 兩條連線執行緒（測試中不實際執行）
    assert feed._connect_in_progress is True
    assert feed._needs_reconnect is False
    assert feed._reconnect_not_before >= time_mod.time() + tq.RECONNECT_COOLDOWN_SEC - 5

    # 冷卻時間內 ensure_fresh_data 不會再次觸發登入
    feed.connected = False
    calls = []
    monkeypatch.setattr(feed, "start", lambda *a, **k: calls.append(a))
    feed.ensure_fresh_data("TMFR1")
    assert calls == []

    # 兩條執行緒都結束才清除 _connect_in_progress
    feed._connect_thread_done()
    assert feed._connect_in_progress is True
    feed._connect_thread_done()
    assert feed._connect_in_progress is False


# ==================== 項目 11：快照不重複計量、Tick 時間戳 ====================

def test_snapshot_updates_price_without_adding_volume():
    feed = TaishinQuoteFeed(store_dir=None)
    feed.symbol = "2330"
    snap = lambda p, q, t: SimpleNamespace(TickData=SimpleNamespace(MatchPrice=p, MatchQty=q, MatchTime=t))

    feed._on_match_tick(SimpleNamespace(MatchPrice=1000.0, MatchQty=10, MatchTime="09:00:05"))
    feed._on_snapshot(snap(1000.0, 10, "09:00:05"))   # 同一筆成交的快照
    feed._on_snapshot(snap(1002.0, 3, "09:00:20"))
    bar = feed.kbars_1m[max(feed.kbars_1m)]
    assert bar["volume"] == 10
    assert bar["close"] == 1002.0 and bar["high"] == 1002.0
    assert feed.get_latest_tick().price == 1002.0


def test_snapshot_seeded_bar_volume_replaced_by_first_real_tick():
    feed = TaishinQuoteFeed(store_dir=None)
    feed.symbol = "2330"
    feed._on_snapshot(SimpleNamespace(TickData=SimpleNamespace(MatchPrice=1000.0, MatchQty=50, MatchTime="09:01:00")))
    bar = feed.kbars_1m[max(feed.kbars_1m)]
    assert bar["volume"] == 1  # 僅為佔位，維持 volume >= 1 的既有慣例
    feed._on_match_tick(SimpleNamespace(MatchPrice=1001.0, MatchQty=7, MatchTime="09:01:10"))
    feed._on_match_tick(SimpleNamespace(MatchPrice=1001.0, MatchQty=2, MatchTime="09:01:20"))
    assert bar["volume"] == 9
    assert bar["open"] == 1000.0 and bar["close"] == 1001.0


def test_parse_match_time_failure_returns_none_and_tick_is_dropped():
    feed = TaishinQuoteFeed(store_dir=None)
    assert feed._parse_match_time("") is None
    assert feed._parse_match_time("abc") is None
    assert feed._parse_match_time("25:00:00") is None

    received = []
    feed.register_on_tick(received.append)
    feed._on_match_tick(SimpleNamespace(MatchPrice=1000.0, MatchQty=1, MatchTime="??"))
    assert received == [] and feed.kbars_1m == {} and feed.get_latest_tick() is None


def test_parse_match_time_cross_midnight_and_weekend():
    feed = TaishinQuoteFeed(store_dir=None)
    # 剛過午夜收到 23:59:58 的 Tick -> 前一日
    now = datetime(2026, 11, 11, 0, 0, 5)
    assert feed._parse_match_time("23:59:58", now=now) == datetime(2026, 11, 10, 23, 59, 58)
    # 本機時鐘略慢：23:59:58 收到 00:00:01 的 Tick -> 隔日
    now2 = datetime(2026, 11, 10, 23, 59, 58)
    assert feed._parse_match_time("000001", now=now2) == datetime(2026, 11, 11, 0, 0, 1)
    # 時鐘小幅誤差內的未來時間不調整
    now3 = datetime(2026, 11, 10, 9, 0, 0)
    assert feed._parse_match_time("09:00:30", now=now3) == datetime(2026, 11, 10, 9, 0, 30)
    # 盤前收到前一交易日收盤的快照時間：週一 08:00 的 13:30 -> 上週五
    now4 = datetime(2026, 11, 9, 8, 0, 0)
    assert feed._parse_match_time("13:30:00", now=now4) == datetime(2026, 11, 6, 13, 30, 0)
    # 週六凌晨夜盤時間維持週六
    now5 = datetime(2026, 11, 7, 4, 0, 0)
    assert feed._parse_match_time("03:59:59", now=now5) == datetime(2026, 11, 7, 3, 59, 59)


# ==================== 項目 14：bar_builder 交易時段對齊與 1D ====================

def _minute_ticks(start, n, price=20000.0):
    return [TickData(start + timedelta(minutes=i), price + i, 1) for i in range(n)]


def test_bar_builder_default_alignment_unchanged():
    ticks = _minute_ticks(datetime(2026, 11, 10, 8, 45), 300)
    bars = SharedBarBuilder.ticks_to_bars(ticks, timeframe="60m", session="day_only")
    assert bars["timestamp"][0] == datetime(2026, 11, 10, 8, 0)
    assert bars.height == 6


def test_bar_builder_session_aligned_60m_and_30m():
    day = _minute_ticks(datetime(2026, 11, 10, 8, 45), 300)       # 08:45 ~ 13:44
    night = _minute_ticks(datetime(2026, 11, 10, 15, 0), 120)     # 15:00 ~ 16:59
    b60 = SharedBarBuilder.ticks_to_bars(day + night, timeframe="60m", session_aligned=True)
    assert b60["timestamp"].to_list() == (
        [datetime(2026, 11, 10, 8, 45) + timedelta(hours=i) for i in range(5)]
        + [datetime(2026, 11, 10, 15, 0), datetime(2026, 11, 10, 16, 0)]
    )
    assert b60["volume"].to_list() == [60] * 7
    assert b60["open"][0] == 20000.0 and b60["close"][0] == 20059.0

    b30 = SharedBarBuilder.ticks_to_bars(day, timeframe="30m", session="day_only", session_aligned=True)
    assert b30.height == 10
    assert b30["timestamp"][0] == datetime(2026, 11, 10, 8, 45)
    assert b30["timestamp"][1] == datetime(2026, 11, 10, 9, 15)

    # 個股以 09:00 為錨點
    stock = _minute_ticks(datetime(2026, 11, 10, 9, 0), 120, price=1000.0)
    bs = SharedBarBuilder.ticks_to_bars(stock, timeframe="60m", session="day_only", session_aligned=True, day_anchor=dtime(9, 0))
    assert bs["timestamp"].to_list() == [datetime(2026, 11, 10, 9, 0), datetime(2026, 11, 10, 10, 0)]


def test_bar_builder_1d_groups_night_session_into_trading_day():
    ticks = [
        TickData(datetime(2026, 11, 6, 15, 0, 5), 100.0, 1),   # 週五夜盤 -> 週一交易日
        TickData(datetime(2026, 11, 7, 3, 0, 0), 90.0, 2),     # 週六凌晨 -> 週一交易日
        TickData(datetime(2026, 11, 9, 9, 0, 0), 120.0, 3),    # 週一日盤
        TickData(datetime(2026, 11, 9, 15, 0, 0), 130.0, 4),   # 週一夜盤 -> 週二交易日
        TickData(datetime(2026, 11, 10, 2, 0, 0), 125.0, 5),   # 週二凌晨 -> 週二交易日
    ]
    bars = SharedBarBuilder.ticks_to_bars(ticks, timeframe="1D")
    assert bars["timestamp"].to_list() == [datetime(2026, 11, 9), datetime(2026, 11, 10)]
    assert bars["open"].to_list() == [100.0, 130.0]
    assert bars["high"].to_list() == [120.0, 130.0]
    assert bars["low"].to_list() == [90.0, 125.0]
    assert bars["close"].to_list() == [120.0, 125.0]
    assert bars["volume"].to_list() == [6, 9]


def test_bar_builder_rejects_unsupported_timeframe():
    ticks = _minute_ticks(datetime(2026, 11, 10, 8, 45), 10)
    for bad in ("abc", "1W", "0m", "7x"):
        with pytest.raises(ValueError):
            SharedBarBuilder.ticks_to_bars(ticks, timeframe=bad)

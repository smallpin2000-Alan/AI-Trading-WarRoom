# tests/test_taifex_history.py
"""期交所逐筆成交資料 → 1 分 K 的轉換規則（以 11/10 實檔與台新資料比對後確定）"""
import zipfile
from datetime import datetime

import polars as pl

from src.data.taifex_history import build_history, parse_daily_zip

HEADER = "成交日期,商品代號,到期月份(週別),成交時間,成交價格,成交數量(B+S),近月價格,遠月價格,開盤集合競價 "


def _write_zip(path, lines):
    content = "\n".join([HEADER, *lines]) + "\n"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(path.stem + ".csv", content.encode("big5"))
    return path


def test_parse_builds_minute_bars_labeled_by_minute_end(tmp_path):
    zp = _write_zip(tmp_path / "Daily_2026_11_10.zip", [
        "20261110,TMF    ,202611     ,090000,40000,4,-,-,*",
        "20261110,TMF    ,202611     ,090030,40010,2,-,-, ",
        "20261110,TMF    ,202611     ,090059,39990,2,-,-, ",
        "20261110,TMF    ,202611     ,090100,39995,6,-,-, ",
        "20261110,TX     ,202611     ,090000,40000,9,-,-, ",            # 其他商品不可混入
        "20261110,TMF    ,202611/202612,090010,50,2,40000,40050, ",     # 價差單不可混入
        "20261110,TMF    ,202612     ,090020,40100,2,-,-, ",            # 遠月成交量少，不採用
    ])
    bars = parse_daily_zip(zp, "TMF")
    assert bars["timestamp"].to_list() == [datetime(2026, 11, 10, 9, 1), datetime(2026, 11, 10, 9, 2)]
    first = bars.row(0, named=True)
    assert (first["open"], first["high"], first["low"], first["close"]) == (40000, 40010, 39990, 39990)
    assert first["volume"] == 4  # 買賣雙邊合計 8 口 ÷ 2
    assert bars.row(1, named=True)["volume"] == 3


def test_near_month_chosen_per_session_by_volume(tmp_path):
    """結算日：日盤仍以到期月份為主，夜盤換成次月"""
    zp = _write_zip(tmp_path / "Daily_2026_11_18.zip", [
        "20261117,TMF    ,202611     ,150000,40000,2,-,-, ",
        "20261117,TMF    ,202612     ,150000,40050,20,-,-, ",
        "20261118,TMF    ,202611     ,090000,40100,20,-,-, ",
        "20261118,TMF    ,202612     ,090000,40150,2,-,-, ",
    ])
    bars = parse_daily_zip(zp, "TMF")
    by_ts = {r["timestamp"]: r["close"] for r in bars.iter_rows(named=True)}
    assert by_ts[datetime(2026, 11, 17, 15, 1)] == 40050  # 夜盤用 202612
    assert by_ts[datetime(2026, 11, 18, 9, 1)] == 40100   # 日盤用 202611


def test_build_history_merges_files_and_dedupes(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_zip(raw / "Daily_2026_11_09.zip", ["20261109,TMF    ,202611     ,090000,40000,2,-,-, "])
    _write_zip(raw / "Daily_2026_11_10.zip", [
        "20261109,TMF    ,202611     ,150000,40020,2,-,-, ",
        "20261110,TMF    ,202611     ,090000,40040,2,-,-, ",
    ])
    out = tmp_path / "TMF_1m.parquet"
    n = build_history(raw, "TMF", out)
    df = pl.read_parquet(out)
    assert n == df.height == 3
    assert df["timestamp"].is_sorted()
    assert df["timestamp"].n_unique() == df.height


def test_corrupt_zip_is_skipped_not_fatal(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "Daily_2026_11_06.zip").write_bytes(b"not a zip")
    _write_zip(raw / "Daily_2026_11_10.zip", ["20261110,TMF    ,202611     ,090000,40040,2,-,-, "])
    assert build_history(raw, "TMF", tmp_path / "out.parquet") == 1


def test_download_recent_skips_existing_and_rejects_foreign_links(tmp_path):
    from src.data import taifex_history as th

    good = tmp_path / "src.zip"
    _write_zip(good, ["20261110,TMF    ,202611     ,090000,40040,2,-,-, "])
    page = (
        "window.open('https://www.taifex.com.tw/file/taifex/Dailydownload/DailydownloadCSV/Daily_2026_11_10.zip')"
        "window.open('https://www.taifex.com.tw/file/taifex/Dailydownload/DailydownloadCSV/Daily_2026_11_09.zip')"
        "window.open('https://evil.example.com/file/taifex/Dailydownload/DailydownloadCSV/Daily_2026_11_08.zip')"
    )
    calls = []

    def fake_fetch(url):
        calls.append(url)
        if url == th.LIST_PAGE_URL:
            return page.encode()
        if url.endswith("11_09.zip"):
            return b"broken"
        return good.read_bytes()

    raw = tmp_path / "raw"
    raw.mkdir()
    new = th.download_recent(raw, fetch=fake_fetch, sleep=lambda s: None)
    assert new == ["Daily_2026_11_10.zip"]
    assert not (raw / "Daily_2026_11_09.zip").exists(), "損毀檔不可留下"
    assert not list(raw.glob("*.part"))
    assert not any("evil" in u for u in calls), "不可下載非期交所網址"

    calls.clear()
    assert th.download_recent(raw, fetch=fake_fetch, sleep=lambda s: None) == []
    assert not any(u.endswith("11_10.zip") for u in calls), "已存在的檔案不可重複下載"

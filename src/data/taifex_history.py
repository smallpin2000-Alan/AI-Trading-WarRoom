# src/data/taifex_history.py
"""
期交所「前 30 個交易日期貨每筆成交資料」→ 歷史 1 分 K（供回測與 AI 訊號評估使用）。

資料來源：https://www.taifex.com.tw/cht/3/dlFutPrevious30DaysSalesData
每個交易日一個 Daily_YYYY_MM_DD.zip（內含 Big5 編碼 CSV），涵蓋「前一晚夜盤 + 當日日盤」。

轉換規則（2026-11-11 以 11/10 實檔與台新 TMFR1 K 棒比對確定，300 根中 291 根收盤完全相同）：
- 只取單式委託；到期月份含「/」者為價差單，排除。
- 近月合約：日盤、夜盤各自取成交量最大的到期月份（結算日日盤仍是到期月、夜盤換次月）。
- K 棒時間標在該分鐘「結束」時刻（09:00:00~09:00:59 的成交 → 09:01），與台新慣例一致。
- 成交數量欄位為買賣雙邊合計，÷ 2 才是成交口數。

本模組只讀取已下載的檔案與寫出 K 棒，不處理任何下單。
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import threading
import time
import zipfile
from collections import defaultdict
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import polars as pl
import requests

logger = logging.getLogger(__name__)

LIST_PAGE_URL = "https://www.taifex.com.tw/cht/3/dlFutPrevious30DaysSalesData"
# 只接受期交所官網 CSV 版下載路徑，避免頁面內容被竄改時下載到其他網址
_CSV_ZIP_PATTERN = re.compile(
    r"https://www\.taifex\.com\.tw/file/taifex/Dailydownload/DailydownloadCSV/(Daily_\d{4}_\d{2}_\d{2}\.zip)"
)
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (AI-Trading-WarRoom history fetcher)"}
HTTP_TIMEOUT_SEC = 60
DOWNLOAD_INTERVAL_SEC = 1.0  # 每檔間隔，避免對期交所造成負擔
MAX_ZIP_BYTES = 50 * 1024 * 1024  # 單檔上限（實際約 3 MB），超過視為異常

RAW_DIR = Path("data/history/taifex_raw")
HISTORY_DIR = Path("data/history")
CSV_ENCODING = "big5"
NIGHT_SESSION_START_HOUR = 15
NIGHT_SESSION_END_HOUR = 6  # 夜盤收在 05:00，以 06:00 為界判斷凌晨段

_COL_DATE, _COL_PRODUCT, _COL_MONTH, _COL_TIME, _COL_PRICE, _COL_QTY = range(6)


def history_path(product: str) -> Path:
    return HISTORY_DIR / f"{product.upper()}_1m.parquet"


def _session_of(ts: datetime) -> str:
    return "night" if (ts.hour >= NIGHT_SESSION_START_HOUR or ts.hour < NIGHT_SESSION_END_HOUR) else "day"


def _read_outright_ticks(zip_path: Path, product: str) -> List[Tuple[datetime, str, float, int]]:
    """讀出指定商品的單式成交：(時間, 到期月份, 價格, 雙邊數量)"""
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise ValueError(f"{zip_path.name} 內沒有 CSV 檔")
        text = z.read(names[0]).decode(CSV_ENCODING)

    target = product.upper()
    ticks: List[Tuple[datetime, str, float, int]] = []
    for line in text.splitlines()[1:]:
        fields = [x.strip() for x in line.split(",")]
        if len(fields) <= _COL_QTY or fields[_COL_PRODUCT] != target or "/" in fields[_COL_MONTH]:
            continue
        try:
            ts = datetime.strptime(fields[_COL_DATE] + fields[_COL_TIME].zfill(6), "%Y%m%d%H%M%S")
            ticks.append((ts, fields[_COL_MONTH], float(fields[_COL_PRICE]), int(fields[_COL_QTY])))
        except ValueError:
            logger.warning("略過無法解析的成交列（%s）：%s", zip_path.name, line[:80])
    return ticks


def _near_month_by_session(ticks: List[Tuple[datetime, str, float, int]]) -> Dict[str, str]:
    volume: Dict[Tuple[str, str], int] = defaultdict(int)
    for ts, month, _, qty in ticks:
        volume[(_session_of(ts), month)] += qty
    chosen: Dict[str, str] = {}
    for (session, month), vol in volume.items():
        if session not in chosen or vol > volume[(session, chosen[session])]:
            chosen[session] = month
    return chosen


def parse_daily_zip(zip_path: Path, product: str = "TMF") -> pl.DataFrame:
    """將一個 Daily_YYYY_MM_DD.zip 轉成近月合約的 1 分 K（timestamp, open, high, low, close, volume）。"""
    ticks = _read_outright_ticks(Path(zip_path), product)
    near = _near_month_by_session(ticks)
    rows = [(ts, price, qty) for ts, month, price, qty in ticks if near.get(_session_of(ts)) == month]
    if not rows:
        return pl.DataFrame(schema={"timestamp": pl.Datetime, "open": pl.Float64, "high": pl.Float64,
                                    "low": pl.Float64, "close": pl.Float64, "volume": pl.Int64})
    df = pl.DataFrame(rows, schema=["ts", "price", "qty"], orient="row").sort("ts", maintain_order=True)
    bars = df.group_by_dynamic("ts", every="1m", label="right", closed="left").agg(
        pl.col("price").first().alias("open"),
        pl.col("price").max().alias("high"),
        pl.col("price").min().alias("low"),
        pl.col("price").last().alias("close"),
        (pl.col("qty").sum() // 2).cast(pl.Int64).alias("volume"),
    )
    return bars.rename({"ts": "timestamp"}).select(["timestamp", "open", "high", "low", "close", "volume"])


def build_history(raw_dir: Path = RAW_DIR, product: str = "TMF", out_path: Path | None = None) -> int:
    """
    合併 raw_dir 內所有 Daily_*.zip 為單一歷史 1 分 K parquet（同一分鐘以較新的檔案為準），回傳 K 棒數。
    損毀或格式不符的檔案會記錄錯誤後略過。
    """
    out = Path(out_path) if out_path is not None else history_path(product)
    frames: List[pl.DataFrame] = []
    for zp in sorted(Path(raw_dir).glob("Daily_*.zip")):
        try:
            frames.append(parse_daily_zip(zp, product))
        except (zipfile.BadZipFile, ValueError, UnicodeDecodeError) as ex:
            logger.error("略過無法讀取的期交所檔案 %s：%s", zp.name, ex)
    if not frames:
        return 0
    merged = (
        pl.concat(frames)
        .unique(subset="timestamp", keep="last", maintain_order=True)
        .sort("timestamp")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp")
    merged.write_parquet(tmp)
    os.replace(tmp, out)
    return merged.height


def list_available_files(page_html: str) -> List[Tuple[str, str]]:
    """從期交所下載頁 HTML 取出 (檔名, 網址)，依檔名去重並排序。"""
    found = {m.group(1): m.group(0) for m in _CSV_ZIP_PATTERN.finditer(page_html)}
    return sorted(found.items())


def download_recent(
    raw_dir: Path = RAW_DIR,
    fetch: Callable[[str], bytes] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> List[str]:
    """
    下載期交所頁面上列出、但 raw_dir 尚未存在的每日逐筆資料檔，回傳新下載的檔名。
    每檔先寫暫存檔、確認為有效 zip 後才改名；單檔失敗只記錄錯誤，不中斷其他檔。
    """
    def _default_fetch(url: str) -> bytes:
        resp = requests.get(url, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT_SEC)
        resp.raise_for_status()
        return resp.content

    get = fetch or _default_fetch
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    files = list_available_files(get(LIST_PAGE_URL).decode("utf-8", errors="replace"))
    if not files:
        raise ValueError("期交所下載頁沒有找到任何每日資料檔連結（頁面格式可能已變更）")

    downloaded: List[str] = []
    for name, url in files:
        target = raw / name
        if target.exists():
            continue
        tmp = target.with_name(f"{name}.part")
        try:
            data = get(url)
            if len(data) > MAX_ZIP_BYTES:
                raise ValueError(f"檔案大小 {len(data)} bytes 超過上限")
            tmp.write_bytes(data)
            with zipfile.ZipFile(tmp) as z:
                if z.testzip() is not None:
                    raise zipfile.BadZipFile("壓縮檔內容損毀")
            os.replace(tmp, target)
            downloaded.append(name)
        except (requests.RequestException, zipfile.BadZipFile, ValueError, OSError) as ex:
            logger.error("下載 %s 失敗：%s", name, ex)
            tmp.unlink(missing_ok=True)
        sleep(DOWNLOAD_INTERVAL_SEC)
    return downloaded


# 每日自動更新：期交所當日檔案在日盤收盤（13:45）後上架，15:30 後更新可確保取得當天資料
DAILY_UPDATE_AFTER = dtime(15, 30)
UPDATE_CHECK_INTERVAL_SEC = 1800
_TW_TZ = timezone(timedelta(hours=8))


def _now_tw() -> datetime:
    return datetime.now(_TW_TZ).replace(tzinfo=None)


def is_update_due(now: datetime, last_run: datetime | None) -> bool:
    """
    是否該執行歷史資料更新：
    - 程式啟動後還沒跑過 → 先跑一次（補齊缺少的日子）
    - 今天 15:30 之後、且上次執行早於今天 15:30 → 再跑一次（取得當天收盤資料）
    """
    if last_run is None:
        return True
    today_cutoff = datetime.combine(now.date(), DAILY_UPDATE_AFTER)
    return now >= today_cutoff and last_run < today_cutoff


def update_history(
    product: str = "TMF",
    raw_dir: Path = RAW_DIR,
    out_path: Path | None = None,
    fetch: Callable[[str], bytes] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Tuple[List[str], int]:
    """下載缺少的每日檔案；有新檔案或輸出檔不存在時重建歷史 K 棒。回傳 (新下載檔名, K 棒數；未重建為 0)。"""
    out = Path(out_path) if out_path is not None else history_path(product)
    new_files = download_recent(raw_dir, fetch=fetch, sleep=sleep)
    if new_files or not out.exists():
        return new_files, build_history(raw_dir, product, out)
    return new_files, 0


class DailyHistoryUpdater(threading.Thread):
    """背景執行緒：定期檢查並更新期交所歷史資料（失敗只記錄，不影響主程式）。"""

    def __init__(self, product: str = "TMF", interval_sec: float = UPDATE_CHECK_INTERVAL_SEC):
        super().__init__(name=f"taifex-history-{product}", daemon=True)
        self.product = product
        self.interval_sec = interval_sec
        self.last_run: datetime | None = None
        self.last_result: str = "尚未執行"

    def run_once(self) -> None:
        now = _now_tw()
        if not is_update_due(now, self.last_run):
            return
        try:
            new_files, n = update_history(self.product)
            self.last_result = f"{now:%m/%d %H:%M} 新增 {len(new_files)} 檔" + (f"，重建 {n} 根 K 棒" if n else "")
            logger.info("期交所歷史資料更新完成：%s", self.last_result)
            self.last_run = now
        except Exception as ex:  # 網路或期交所頁面異常：記錄後下次再試，不更新 last_run
            self.last_result = f"{now:%m/%d %H:%M} 更新失敗：{ex}"
            logger.exception("期交所歷史資料自動更新失敗")

    def run(self) -> None:
        while True:
            self.run_once()
            time.sleep(self.interval_sec)


def main() -> None:
    parser = argparse.ArgumentParser(description="期交所逐筆成交資料 → 歷史 1 分 K")
    parser.add_argument("--product", default="TMF")
    parser.add_argument("--download", action="store_true", help="先從期交所下載尚未取得的每日檔案")
    args = parser.parse_args()
    if args.download:
        new_files, _ = update_history(args.product)
        print(f"新下載 {len(new_files)} 個檔案：{', '.join(new_files) or '（無）'}")
    n = build_history(RAW_DIR, args.product)
    print(f"已輸出 {history_path(args.product)}：{n} 根 1 分 K")


if __name__ == "__main__":
    main()

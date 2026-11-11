# src/data/kbar_store.py
from __future__ import annotations
import logging
import os
import shutil
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Iterable, Optional
import polars as pl

logger = logging.getLogger(__name__)

# os.replace 在 Windows 上若目標檔被其他程式（防毒、雲端硬碟同步）短暫佔用會失敗，重試次數與間隔
_REPLACE_RETRIES = 5
_REPLACE_RETRY_DELAY_SEC = 0.1


class KBarStore:
    """
    台新 API 歷史與即時 K 線本地與 Google Cloud Storage 持久化儲存層 (Parquet Storage)
    - 僅儲存來自台新官方 API 的真實 1 分鐘 K 棒（帶有 synthetic 標記的補值/內插 K 棒一律不落盤）。
    - 支援啟動時毫秒級預載入，並於收到台新新 K 棒時自動去重合併存檔與背景非阻塞雲端同步。
    - 寫入採「暫存檔 + os.replace」原子取代，並以每商品檔案鎖保護 save / load，避免半寫入或交錯覆蓋。
    """

    DEFAULT_GCS_BUCKET = os.getenv("GCS_BUCKET", "point-api-512405-stock-history").strip()
    _last_gcs_upload_ts: Dict[str, float] = {}
    # 每個 Parquet 檔一把鎖（跨 KBarStore 實例共用，以檔案絕對路徑為 key）
    _file_locks: Dict[str, threading.RLock] = {}
    _file_locks_guard = threading.Lock()

    def __init__(self, data_dir: str | Path = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, symbol: str) -> Path:
        clean = "".join(c for c in symbol.strip() if c.isalnum() or c in ("_", "-"))
        return self.data_dir / f"taishin_kbars_{clean}_1m.parquet"

    @classmethod
    def _lock_for(cls, file_path: Path) -> threading.RLock:
        key = os.path.abspath(str(file_path))
        with cls._file_locks_guard:
            lock = cls._file_locks.get(key)
            if lock is None:
                lock = threading.RLock()
                cls._file_locks[key] = lock
            return lock

    @staticmethod
    def _tmp_path(file_path: Path, tag: str) -> Path:
        return file_path.with_name(f"{file_path.name}.{tag}.{os.getpid()}.{threading.get_ident()}.tmp")

    @staticmethod
    def _remove_quietly(path: Path) -> None:
        try:
            if path.exists():
                path.unlink()
        except OSError as ex:
            logger.warning("KBarStore: 無法移除暫存檔 %s: %s", path, ex)

    @staticmethod
    def _atomic_replace(tmp_path: Path, file_path: Path) -> None:
        """以 os.replace 原子取代目標檔；目標檔被短暫佔用時重試，仍失敗則拋出例外（舊檔保持不變）"""
        last_ex: Optional[OSError] = None
        for attempt in range(_REPLACE_RETRIES):
            try:
                os.replace(tmp_path, file_path)
                return
            except PermissionError as ex:
                last_ex = ex
                logger.warning(
                    "KBarStore: 取代 %s 失敗 (第 %d 次，檔案可能被佔用): %s", file_path, attempt + 1, ex
                )
                time.sleep(_REPLACE_RETRY_DELAY_SEC)
        assert last_ex is not None
        raise last_ex

    def _quarantine_corrupt(self, file_path: Path, ex: Exception) -> None:
        """將無法解析的壞檔改名為 .corrupt 保留現場，避免之後被當成『沒有歷史』而整檔覆蓋"""
        corrupt_path = file_path.with_name(file_path.name + ".corrupt")
        if corrupt_path.exists():
            # 已有第一份壞檔備份時，後續壞檔固定寫到 .corrupt.latest，避免無限累積
            corrupt_path = file_path.with_name(file_path.name + ".corrupt.latest")
        os.replace(file_path, corrupt_path)
        logger.error("KBarStore: %s 無法解析 (%s)，已改名為 %s 保留", file_path, ex, corrupt_path)

    def _read_existing(self, file_path: Path, is_stock: bool) -> Dict[datetime, Dict[str, Any]]:
        """
        讀取既有 Parquet（呼叫端須已持有該檔的鎖）。
        - 檔案不存在：回傳空字典。
        - I/O 類錯誤（檔案被佔用等暫時性問題）：拋出 OSError，由呼叫端決定是否中止寫入。
        - 內容損毀：改名為 .corrupt 後回傳空字典。
        """
        if not file_path.exists():
            return {}
        try:
            df = pl.read_parquet(file_path).sort("timestamp")
            res: Dict[datetime, Dict[str, Any]] = {}
            for row in df.iter_rows(named=True):
                dt = row["timestamp"]
                if is_stock and not ((9, 0) <= (dt.hour, dt.minute) <= (13, 35)):
                    continue
                res[dt] = {
                    "timestamp": dt,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": int(row["volume"]),
                }
            return res
        except OSError:
            raise
        except Exception as ex:
            self._quarantine_corrupt(file_path, ex)
            return {}

    def save_kbars(
        self,
        symbol: str,
        kbars_dict: Dict[datetime, Dict[str, Any]],
        replace_all: bool = False,
        force_gcs: bool = False,
        delete_keys: Optional[Iterable[datetime]] = None,
    ) -> int:
        """
        將記憶體中的 1m K 棒與本地 Parquet 合併去重後寫入硬碟（並於雲端環境同步至 GCS）
        - delete_keys：要從歷史中明確刪除的 K 棒時間（例如已判定為異常值者），其餘歷史一律保留。
        - 帶有 synthetic 標記（補值/內插，僅供顯示）的 K 棒不會寫入。
        - replace_all=True 會以傳入內容整檔取代（僅為相容保留，一般同步路徑請勿使用）。
        """
        delete_set = set(delete_keys) if delete_keys else set()
        if not kbars_dict and not delete_set:
            return 0
        file_path = self._get_file_path(symbol)
        is_stock = symbol.strip().isdigit()
        real_bars = {dt: b for dt, b in (kbars_dict or {}).items() if not b.get("synthetic")}

        with self._lock_for(file_path):
            if replace_all:
                merged: Dict[datetime, Dict[str, Any]] = dict(real_bars)
            else:
                # 讀取失敗（I/O 錯誤）時直接拋出，不可把『讀不到』當成『沒有歷史』而覆蓋掉舊檔
                merged = self._read_existing(file_path, is_stock)
                merged.update(real_bars)

            for dk in delete_set:
                merged.pop(dk, None)

            if is_stock:
                merged = {
                    dt: b for dt, b in merged.items()
                    if (9, 0) <= (dt.hour, dt.minute) <= (13, 35)
                }
            if not merged and not (delete_set and file_path.exists()):
                return 0

            sorted_bars = [merged[dt] for dt in sorted(merged.keys())]
            df = pl.DataFrame(
                {
                    "timestamp": [b["timestamp"] for b in sorted_bars],
                    "open": [float(b["open"]) for b in sorted_bars],
                    "high": [float(b["high"]) for b in sorted_bars],
                    "low": [float(b["low"]) for b in sorted_bars],
                    "close": [float(b["close"]) for b in sorted_bars],
                    "volume": [int(b["volume"]) for b in sorted_bars],
                },
                schema={
                    "timestamp": pl.Datetime,
                    "open": pl.Float64,
                    "high": pl.Float64,
                    "low": pl.Float64,
                    "close": pl.Float64,
                    "volume": pl.Int64,
                },
            )
            # 先寫暫存檔再原子取代，任何中途失敗都不會破壞既有 Parquet
            tmp_path = self._tmp_path(file_path, "write")
            try:
                df.write_parquet(tmp_path)
                self._atomic_replace(tmp_path, file_path)
            except Exception:
                self._remove_quietly(tmp_path)
                raise

        # 若在 Cloud Run 雲端環境，於背景執行緒非阻塞上傳至 GCS 永久保存（每檔至少間隔 120 秒，絕不卡住前端網頁）
        now_ts = time.time()
        last_up = self._last_gcs_upload_ts.get(file_path.name, 0.0)
        if (os.getenv("K_SERVICE") or os.getenv("GCS_BUCKET")) and self.DEFAULT_GCS_BUCKET and (force_gcs or (now_ts - last_up >= 120.0)):
            self._last_gcs_upload_ts[file_path.name] = now_ts
            bucket_name = self.DEFAULT_GCS_BUCKET
            blob_name = f"kbars/{file_path.name}"

            def _bg_upload():
                # 先在鎖內複製一份快照再上傳，避免上傳途中檔案被下一次存檔取代
                snap_path = self._tmp_path(file_path, "upload")
                try:
                    with self._lock_for(file_path):
                        shutil.copyfile(file_path, snap_path)
                    from google.cloud import storage
                    client = storage.Client()
                    bucket = client.bucket(bucket_name)
                    blob = bucket.blob(blob_name)
                    blob.upload_from_filename(str(snap_path))
                except Exception as ex:
                    logger.warning("KBarStore: 上傳 %s 至 GCS 失敗: %s", blob_name, ex)
                finally:
                    self._remove_quietly(snap_path)

            threading.Thread(target=_bg_upload, daemon=True).start()

        return len(sorted_bars)

    def _download_from_gcs(self, file_path: Path) -> None:
        """本地無檔時自 GCS 下載（先寫暫存檔再原子取代，下載中斷不會留下半個 Parquet）"""
        tmp_path = self._tmp_path(file_path, "download")
        try:
            from google.cloud import storage
            client = storage.Client()
            bucket = client.bucket(self.DEFAULT_GCS_BUCKET)
            blob = bucket.blob(f"kbars/{file_path.name}")
            if blob.exists():
                blob.download_to_filename(str(tmp_path))
                self._atomic_replace(tmp_path, file_path)
        except Exception as ex:
            logger.warning("KBarStore: 自 GCS 下載 %s 失敗: %s", file_path.name, ex)
        finally:
            self._remove_quietly(tmp_path)

    def load_kbars_dict(self, symbol: str) -> Dict[datetime, Dict[str, Any]]:
        """從本地 Parquet（或 Google Cloud Storage）讀取指定商品的台新真實 1m K 棒字典"""
        file_path = self._get_file_path(symbol)
        is_stock = symbol.strip().isdigit()
        with self._lock_for(file_path):
            if not file_path.exists() and (os.getenv("K_SERVICE") or os.getenv("GCS_BUCKET")) and self.DEFAULT_GCS_BUCKET:
                self._download_from_gcs(file_path)

            try:
                return self._read_existing(file_path, is_stock)
            except OSError as ex:
                # 暫時性 I/O 錯誤：不動檔案，僅記錄並回傳空字典（save_kbars 走 _read_existing 會直接中止寫入）
                logger.error("KBarStore: 讀取 %s 發生 I/O 例外: %s", file_path, ex)
                return {}

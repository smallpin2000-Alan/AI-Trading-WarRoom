# src/data/kbar_store.py
from __future__ import annotations
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import polars as pl


class KBarStore:
    """
    台新 API 歷史與即時 K 線本地與 Google Cloud Storage 持久化儲存層 (Parquet Storage)
    - 僅儲存來自台新官方 API 的真實 1 分鐘 K 棒。
    - 支援啟動時毫秒級預載入，並於收到台新新 K 棒時自動去重合併存檔與雲端同步。
    """

    DEFAULT_GCS_BUCKET = os.getenv("GCS_BUCKET", "point-api-512405-stock-history").strip()

    def __init__(self, data_dir: str | Path = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, symbol: str) -> Path:
        clean = "".join(c for c in symbol.strip() if c.isalnum() or c in ("_", "-"))
        return self.data_dir / f"taishin_kbars_{clean}_1m.parquet"

    def save_kbars(self, symbol: str, kbars_dict: Dict[datetime, Dict[str, Any]]) -> int:
        """將記憶體中的 1m K 棒與本地 Parquet 合併去重後寫入硬碟（並於雲端環境同步至 GCS）"""
        if not kbars_dict:
            return 0
        file_path = self._get_file_path(symbol)

        merged: Dict[datetime, Dict[str, Any]] = self.load_kbars_dict(symbol)
        merged.update(kbars_dict)

        sorted_bars = [merged[dt] for dt in sorted(merged.keys())]
        df = pl.DataFrame(
            {
                "timestamp": [b["timestamp"] for b in sorted_bars],
                "open": [float(b["open"]) for b in sorted_bars],
                "high": [float(b["high"]) for b in sorted_bars],
                "low": [float(b["low"]) for b in sorted_bars],
                "close": [float(b["close"]) for b in sorted_bars],
                "volume": [int(b["volume"]) for b in sorted_bars],
            }
        )
        df.write_parquet(file_path)

        # 若在 Cloud Run 雲端環境，將最新台新 K 線同步上傳至 GCS 永久保存
        if (os.getenv("K_SERVICE") or os.getenv("GCS_BUCKET")) and self.DEFAULT_GCS_BUCKET:
            try:
                from google.cloud import storage
                client = storage.Client()
                bucket = client.bucket(self.DEFAULT_GCS_BUCKET)
                blob = bucket.blob(f"kbars/{file_path.name}")
                blob.upload_from_filename(str(file_path))
            except Exception:
                pass

        return len(sorted_bars)

    def load_kbars_dict(self, symbol: str) -> Dict[datetime, Dict[str, Any]]:
        """從本地 Parquet（或 Google Cloud Storage）讀取指定商品的台新真實 1m K 棒字典"""
        file_path = self._get_file_path(symbol)
        if not file_path.exists() and (os.getenv("K_SERVICE") or os.getenv("GCS_BUCKET")) and self.DEFAULT_GCS_BUCKET:
            try:
                from google.cloud import storage
                client = storage.Client()
                bucket = client.bucket(self.DEFAULT_GCS_BUCKET)
                blob = bucket.blob(f"kbars/{file_path.name}")
                if blob.exists():
                    blob.download_to_filename(str(file_path))
            except Exception:
                pass

        if not file_path.exists():
            return {}
        try:
            df = pl.read_parquet(file_path).sort("timestamp")
            res: Dict[datetime, Dict[str, Any]] = {}
            for row in df.iter_rows(named=True):
                dt = row["timestamp"]
                res[dt] = {
                    "timestamp": dt,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": int(row["volume"]),
                }
            return res
        except Exception as ex:
            print(f"KBarStore: 讀取 {file_path} 例外: {ex}")
            return {}

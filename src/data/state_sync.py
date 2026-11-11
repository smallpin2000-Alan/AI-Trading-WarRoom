# src/data/state_sync.py
"""
模擬倉狀態檔與 Google Cloud Storage 的同步（僅在 Cloud Run 上啟用）。

Cloud Run 容器是無狀態的：縮容到 0 或重新部署後，本機 data/ 會回到映像檔內的版本。
因此在雲端時以 GCS 為準：
- 讀檔前先從 GCS 下載最新狀態（下載失敗則沿用本機檔）。
- 每次存檔後在背景把同一份內容上傳到 GCS。
本機執行（沒有 K_SERVICE 環境變數）時完全不動作。
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

STATE_BLOB_PREFIX = "state/"
_upload_lock = threading.Lock()


def _bucket_name() -> str:
    return os.getenv("GCS_BUCKET", "").strip()


def is_enabled() -> bool:
    """只在 Cloud Run（有 K_SERVICE）且設定了 GCS_BUCKET 時啟用。"""
    return bool(os.getenv("K_SERVICE")) and bool(_bucket_name())


def _blob(file_name: str):
    from google.cloud import storage  # 延遲匯入：本機與測試不需要此套件

    return storage.Client().bucket(_bucket_name()).blob(f"{STATE_BLOB_PREFIX}{file_name}")


def download_if_available(path: Path) -> bool:
    """從 GCS 下載狀態檔覆蓋本機檔（先寫暫存檔再替換）；成功回傳 True。"""
    if not is_enabled():
        return False
    tmp = path.with_name(f"{path.name}.{os.getpid()}.download")
    try:
        blob = _blob(path.name)
        if not blob.exists():
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(tmp))
        os.replace(tmp, path)
        return True
    except Exception:
        logger.exception("從 GCS 下載模擬倉狀態失敗，沿用本機檔 %s", path)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def upload_in_background(path: Path, content: str) -> None:
    """在背景執行緒把狀態內容上傳到 GCS（上傳的是傳入的字串快照，不會讀到寫到一半的檔案）。"""
    if not is_enabled():
        return

    def _run():
        with _upload_lock:  # 依序上傳，避免舊內容晚到而覆蓋新內容
            try:
                _blob(path.name).upload_from_string(content, content_type="application/json")
            except Exception:
                logger.exception("模擬倉狀態上傳 GCS 失敗（本機檔已保存）")

    threading.Thread(target=_run, daemon=True).start()

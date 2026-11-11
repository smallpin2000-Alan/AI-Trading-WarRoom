# src/ui/admin_gate.py
"""
戰情室「寫入操作」的管理密碼保護（選用）。

網站以公開方式部署時，任何訪客都能按下重置、解鎖風控、修改停損參數等按鈕。
設定環境變數 WARROOM_ADMIN_PASSWORD 後，這些寫入操作必須先輸入管理密碼解鎖；
未設定時維持原本行為（不擋），並由介面顯示提醒。

本模組不依賴 Streamlit，方便單元測試。
"""
from __future__ import annotations

import hmac
import os
import threading
import time
from typing import Callable, List

ADMIN_PASSWORD_ENV = "WARROOM_ADMIN_PASSWORD"
# 全站在一個時間窗內允許的密碼錯誤次數，超過即暫時鎖定（防暴力猜測）
MAX_FAILED_ATTEMPTS = 8
FAILED_WINDOW_SEC = 600.0


def get_admin_password() -> str:
    return os.getenv(ADMIN_PASSWORD_ENV, "").strip()


def is_protection_enabled() -> bool:
    return bool(get_admin_password())


def verify_admin_password(candidate: str, expected: str) -> bool:
    """以固定時間比較驗證密碼；未設定管理密碼時一律回傳 False（不可用空密碼解鎖）。"""
    if not expected:
        return False
    return hmac.compare_digest((candidate or "").encode("utf-8"), expected.encode("utf-8"))


class FailedAttemptThrottle:
    """全站共用的密碼錯誤次數節流器（執行緒安全）。"""

    def __init__(
        self,
        max_attempts: int = MAX_FAILED_ATTEMPTS,
        window_sec: float = FAILED_WINDOW_SEC,
        clock: Callable[[], float] = time.time,
    ):
        self._max_attempts = max_attempts
        self._window_sec = window_sec
        self._clock = clock
        self._lock = threading.Lock()
        self._failures: List[float] = []

    def _recent(self) -> List[float]:
        cutoff = self._clock() - self._window_sec
        return [t for t in self._failures if t >= cutoff]

    def is_locked(self) -> bool:
        with self._lock:
            self._failures = self._recent()
            return len(self._failures) >= self._max_attempts

    def record_failure(self) -> None:
        with self._lock:
            self._failures = [*self._recent(), self._clock()]

    def seconds_until_unlock(self) -> int:
        with self._lock:
            recent = self._recent()
            if len(recent) < self._max_attempts:
                return 0
            return max(0, int(recent[0] + self._window_sec - self._clock()))

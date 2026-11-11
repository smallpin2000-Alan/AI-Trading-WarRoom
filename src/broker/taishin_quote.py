# src/broker/taishin_quote.py
from __future__ import annotations
import os
import sys
import time
import logging
import datetime
import threading
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any
import requests
import polars as pl

from src.core.bar_builder import TickData, SharedBarBuilder
from src.data.kbar_store import KBarStore


logger = logging.getLogger(__name__)

TZ_TW = datetime.timezone(datetime.timedelta(hours=8))

# 異常值判斷：K 棒價格與鄰近 K 棒收盤價中位數相比，偏離超過此比例才剔除（期貨與個股共用）
# 台股單日漲跌停 10%，同日最大振幅約 22%，取 25% 可確保正常行情（含跨日跳空）不被誤殺
OUTLIER_MAX_DEVIATION_RATIO = 0.25
OUTLIER_NEIGHBOR_WINDOW = 10   # 前後各取幾根鄰近 K 棒
OUTLIER_MIN_NEIGHBORS = 3      # 鄰近 K 棒不足此數時不做偏離判斷（僅檢查欄位有效性）

# Tick 時間與本機時鐘可容許的誤差；超過此誤差的「未來時間」視為前一日（跨午夜）
TICK_CLOCK_SKEW_TOLERANCE = datetime.timedelta(minutes=10)

# 斷線重連節流：斷線後先等待寬限期（讓 API 自身的自動重連先嘗試），每次自動登入之間至少間隔冷卻時間
RECONNECT_GRACE_SEC = 20.0
RECONNECT_COOLDOWN_SEC = 60.0


def _now_tw() -> datetime.datetime:
    """永遠回傳台灣時間 (UTC+8, 無 tzinfo)，不受 Cloud Run 預設 UTC+0 時區影響"""
    return datetime.datetime.now(TZ_TW).replace(tzinfo=None, microsecond=0)


def _from_ts_tw(ts_val: int | float) -> datetime.datetime:
    """將 Unix Timestamp 強制轉換為台灣時間 (UTC+8, 無 tzinfo)"""
    return datetime.datetime.fromtimestamp(int(ts_val), tz=TZ_TW).replace(tzinfo=None, second=0, microsecond=0)


class _SymbolSwitched(Exception):
    """同步途中商品已被切換：中止本次同步，不可把舊商品資料寫進新商品"""


def _valid_price(val: Any) -> Optional[float]:
    """轉為正的有限浮點數；缺值、非數字、<= 0 或 NaN/Inf 一律回傳 None"""
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")) or f <= 0:
        return None
    return f


def find_outlier_kbar_keys(kbars: Dict[datetime.datetime, Dict[str, Any]]) -> List[datetime.datetime]:
    """
    找出應剔除的異常 K 棒時間（依時間排序回傳）：
    - 缺 low / close（或非正數）的 K 棒視為無效。
    - 任一 OHLC 價格與鄰近 K 棒（前後各 OUTLIER_NEIGHBOR_WINDOW 根）收盤價中位數相比，
      偏離超過 OUTLIER_MAX_DEVIATION_RATIO 者視為異常值。
    為避免誤殺，鄰近 K 棒數為偶數時需同時偏離上、下中位數才剔除（兩種價位各佔一半時不判定）。
    """
    bad: List[datetime.datetime] = []
    valid: List[tuple] = []  # (dt, close, [該棒所有有效價格])
    for dt in sorted(kbars.keys()):
        b = kbars[dt]
        low = _valid_price(b.get("low"))
        close = _valid_price(b.get("close"))
        if low is None or close is None:
            bad.append(dt)
            continue
        prices = [low, close]
        for fld in ("open", "high"):
            pv = _valid_price(b.get(fld))
            if pv is not None:
                prices.append(pv)
        valid.append((dt, close, prices))

    if len(valid) > OUTLIER_MIN_NEIGHBORS:
        # 快速路徑：全體價格範圍都在容許比例內時，不可能有任何一根偏離鄰近中位數
        g_min = min(min(v[2]) for v in valid)
        g_max = max(max(v[2]) for v in valid)
        if g_max > g_min * (1.0 + OUTLIER_MAX_DEVIATION_RATIO):
            closes = [v[1] for v in valid]
            total = len(valid)
            for i, (dt, _c, prices) in enumerate(valid):
                lo_i = max(0, i - OUTLIER_NEIGHBOR_WINDOW)
                hi_i = min(total, i + OUTLIER_NEIGHBOR_WINDOW + 1)
                neigh = sorted(closes[lo_i:i] + closes[i + 1:hi_i])
                if len(neigh) < OUTLIER_MIN_NEIGHBORS:
                    continue
                med_lo = neigh[(len(neigh) - 1) // 2]
                med_hi = neigh[len(neigh) // 2]
                for pv in prices:
                    if (
                        abs(pv / med_lo - 1.0) > OUTLIER_MAX_DEVIATION_RATIO
                        and abs(pv / med_hi - 1.0) > OUTLIER_MAX_DEVIATION_RATIO
                    ):
                        bad.append(dt)
                        break
    return sorted(bad)


class TaishinQuoteFeed:
    """
    台新/元富行情與技術分析 K 線接收模組 (Market Data Feed)
    - 採用官方 SolPYAPI (PY_TradeD / Sol_D) 接收即時 Tick 與快照。
    - 採用官方 tech_analysis_api_v2 (TechAnalysis) 接收台新真實歷史與即時 1 分鐘 K 線。
    - 當期貨 API 尚未開通或於 Linux Cloud Run 環境時，自動串接台灣期貨交易所官網 API (mis.taifex.com.tw)
      與即時 1 分鐘 K 線通道進行 0 秒延遲同步，確保期貨與個股皆永遠保持當日最新分 K。
    - 純報價接收 (Market Data Only)，絕不發送任何交易下單委託指令。
    """

    VERSION = 22
    _last_sync_ts: Dict[str, float] = {}
    _last_server_pull_dt: Dict[str, datetime.datetime] = {}
    _last_server_bar_dt: Dict[str, datetime.datetime] = {}
    _last_server_source: Dict[str, str] = {}

    def __init__(self, store_dir: Optional[str | Path] = "data", auto_load_store: bool = False):
        self.sol_d = None
        self.ta = None
        self._k_configs = []
        self.connected = False
        self.ta_connected = False
        self.ta_rcv_done = False
        self.status_msg = "未連線"
        self.symbol = "TMFR1"
        self.current_user_id = ""
        self.current_password = ""
        self._conn_lock = threading.Lock()
        self._connect_in_progress = False
        self._connect_pending = 0          # 尚未結束的連線執行緒數（TechAnalysis + Sol_D）
        self._needs_reconnect = False      # 曾連線成功後被斷線，需重新登入
        self._reconnect_not_before = 0.0   # 自動（重）連線最早可執行的時間 (time.time())
        self._symbol_gen = 0               # 商品世代編號：每次切換商品遞增，供進行中的同步比對
        self._stale_symbols: set = set()   # 已切換離開的舊商品代碼（其遲到的 K 棒回呼一律丟棄）
        self._bad_tick_time_count = 0

        self.store: Optional[KBarStore] = KBarStore(store_dir) if store_dir else None
        self.raw_ticks: List[TickData] = []
        self.kbars_1m: Dict[datetime.datetime, Dict[str, Any]] = (
            self.store.load_kbars_dict(self.symbol) if (self.store and auto_load_store) else {}
        )
        self.latest_tick: Optional[TickData] = None
        if self.kbars_1m:
            last_dt = max(self.kbars_1m.keys())
            last_b = self.kbars_1m[last_dt]
            self.latest_tick = TickData(
                timestamp=last_dt,
                price=float(last_b["close"]),
                volume=int(last_b["volume"]),
            )
            self.status_msg = f"已載入本地台新 K 線快取 ({self.symbol}: {len(self.kbars_1m)} 根)"
        self.lock = threading.Lock()
        self.is_running = False
        self.on_tick_callbacks: List[Callable[[TickData], None]] = []

    def _mark_server_pulled(
        self,
        symbol: str,
        source: str,
        latest_bar_dt: Optional[datetime.datetime] = None,
    ):
        """僅在確實從遠端伺服器拉取到最新有效資料時才更新最後更新時間（讀快取絕不呼叫）"""
        sym = (symbol or self.symbol).strip().upper()
        now_tw = _now_tw()
        self._last_server_pull_dt[sym] = now_tw
        if latest_bar_dt is not None:
            self._last_server_bar_dt[sym] = latest_bar_dt
        elif self.kbars_1m:
            self._last_server_bar_dt[sym] = max(self.kbars_1m.keys())
        self._last_server_source[sym] = source

    def get_server_sync_info(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """回傳指定商品最後一次『確實從伺服器拉到資料』的驗證資訊"""
        sym = (symbol or self.symbol).strip().upper()
        pull_dt = self._last_server_pull_dt.get(sym)
        bar_dt = self._last_server_bar_dt.get(sym)
        source = self._last_server_source.get(sym, "")
        if pull_dt is None:
            return {
                "is_synced": False,
                "pull_time_str": "尚未從伺服器拉取（目前顯示快取）",
                "bar_time_str": bar_dt.strftime("%Y-%m-%d %H:%M") if bar_dt else "--",
                "source": "本地 / 雲端快取",
                "seconds_ago": -1,
            }
        sec_ago = max(0, int((_now_tw() - pull_dt).total_seconds()))
        return {
            "is_synced": True,
            "pull_time_str": pull_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "bar_time_str": bar_dt.strftime("%Y-%m-%d %H:%M") if bar_dt else "--",
            "source": source,
            "seconds_ago": sec_ago,
        }

    def register_on_tick(self, callback: Callable[[TickData], None]):
        """註冊 Tick 到達時的回呼函式"""
        self.on_tick_callbacks.append(callback)

    def _parse_match_time(
        self, time_val: Any, now: Optional[datetime.datetime] = None
    ) -> Optional[datetime.datetime]:
        """
        解析時間格式 (支援 HH:MM:SS 或 HHMMSS 或整數)，統一採用台灣時間 (UTC+8)
        - 解析失敗回傳 None（不可把無法解析的時間當成現在時間）。
        - 來源只有時分秒沒有日期：若比現在晚超過時鐘容許誤差，代表是跨午夜前（或前一交易日）的成交，視為前一日；
          反之若比現在早將近一整天（本機時鐘略慢、剛跨午夜），視為隔日。
        - 週日全天與週六 05:00 之後沒有任何盤，日期再往前推到最近的交易時段。
        """
        if now is None:
            now = _now_tw()
        t_str = str(time_val).strip() if time_val is not None else ""
        try:
            if ":" in t_str:
                parts = t_str.split(":")
                hh = int(parts[0])
                mm = int(parts[1])
                ss = int(float(parts[2])) if len(parts) > 2 else 0
            else:
                digits = t_str.split(".")[0]
                if not digits.isdigit():
                    return None
                clean = digits.zfill(6)
                hh = int(clean[0:2])
                mm = int(clean[2:4])
                ss = int(clean[4:6])
            dt = now.replace(hour=hh, minute=mm, second=ss, microsecond=0)
        except (ValueError, IndexError):
            return None

        if dt - now > TICK_CLOCK_SKEW_TOLERANCE:
            dt -= datetime.timedelta(days=1)
        elif now - dt > datetime.timedelta(days=1) - TICK_CLOCK_SKEW_TOLERANCE:
            dt += datetime.timedelta(days=1)
        while dt.weekday() == 6 or (dt.weekday() == 5 and dt.time() > datetime.time(5, 0)):
            dt -= datetime.timedelta(days=1)
        return dt

    def _note_bad_tick_time(self, source: str, raw: Any):
        """記錄無法解析的 Tick 時間（前 5 次與之後每 1000 次記一筆，避免洗版）"""
        self._bad_tick_time_count += 1
        if self._bad_tick_time_count <= 5 or self._bad_tick_time_count % 1000 == 0:
            logger.warning(
                "TaishinQuoteFeed: %s 的成交時間 %r 無法解析，已略過該筆 (累計 %d 筆)",
                source, raw, self._bad_tick_time_count,
            )

    @staticmethod
    def _night_session_start_date(now: datetime.datetime) -> datetime.date:
        """回傳目前（或最近一次）夜盤的開始日曆日：15:00 前屬前一日開始的夜盤；週末沒有夜盤開盤，退回週五。"""
        start = now.date() if now.hour >= 15 else (now.date() - datetime.timedelta(days=1))
        if start.weekday() == 5:
            start -= datetime.timedelta(days=1)
        elif start.weekday() == 6:
            start -= datetime.timedelta(days=2)
        return start

    def _save_store_safely(self, symbol: str, kbars: Dict[datetime.datetime, Dict[str, Any]], **kwargs):
        """存檔失敗只記錄錯誤、不中斷行情流程（KBarStore 已保證失敗時舊檔不受影響）"""
        if not self.store:
            return
        try:
            self.store.save_kbars(symbol, kbars, **kwargs)
        except Exception as ex:
            logger.error("TaishinQuoteFeed: %s K 棒存檔失敗（舊檔保留）: %s", symbol, ex)

    @staticmethod
    def _parse_kbar_datetime(date_str: str, time_sn_dply: Optional[int]) -> Optional[datetime.datetime]:
        """
        解析台新 TechAnalysis TKBarRec 的日期與時間 (YYYYMMDD 交易日 + HHMM)，回傳真實日曆時間：
        - 時間缺值 (None) 才視為無效；0 代表 00:00 的 K 棒，必須保留。
        - 15:00 之後的夜盤屬於「下一個交易日」，真實日曆日為前一個交易日（週一交易日 -> 上週五）。
        - 夜盤跨午夜後的凌晨段 (00:00 ~ 05:00) 真實日曆日為夜盤開始日的隔天：
          一般即交易日當天，但週一交易日的凌晨段是週六（與 cnyes / 期交所 MIS 路徑的真實日曆日一致）。
        僅依週末推算，不含國定假日行事曆。
        """
        if time_sn_dply is None or time_sn_dply < 0:
            return None
        try:
            dt = datetime.datetime.strptime(str(date_str).strip(), "%Y%m%d")
            weekday = dt.weekday()
            if time_sn_dply >= 1500:
                if weekday == 0:
                    cal_dt = dt - datetime.timedelta(days=3)
                elif weekday == 6:
                    cal_dt = dt - datetime.timedelta(days=2)
                else:
                    cal_dt = dt - datetime.timedelta(days=1)
            elif time_sn_dply <= 500 and weekday == 0:
                cal_dt = dt - datetime.timedelta(days=2)
            else:
                cal_dt = dt
            hh = time_sn_dply // 100
            mm = time_sn_dply % 100
            return cal_dt.replace(hour=hh, minute=mm, second=0, microsecond=0)
        except (ValueError, TypeError):
            return None

    def _upsert_kbar_from_rec(self, kb: Any):
        """將台新 TechAnalysis 的 TKBarRec 寫入 1m K 棒快取"""
        if kb is None:
            return
        if self._is_stale_product(kb):
            return
        t_raw = getattr(kb, "TimeSn_Dply", None)
        if t_raw is None or t_raw == "":
            return
        t_val = int(t_raw)
        dt = self._parse_kbar_datetime(str(getattr(kb, "Date", "")), t_val)
        if dt is None:
            return

        o_p = float(getattr(kb, "OPrice", 0.0) or 0.0)
        h_p = float(getattr(kb, "HPrice", 0.0) or 0.0)
        l_p = float(getattr(kb, "LPrice", 0.0) or 0.0)
        c_p = float(getattr(kb, "CPrice", 0.0) or 0.0)
        if c_p <= 0:
            return
        qty = int(getattr(kb, "Quantity", 0) or 0)
        vol = max(1, qty)

        self.kbars_1m[dt] = {
            "timestamp": dt,
            "open": o_p,
            "high": h_p,
            "low": l_p,
            "close": c_p,
            "volume": vol,
        }

    def _is_stale_product(self, kb: Any) -> bool:
        """
        K 棒紀錄 (TKBarRec.Product) 若屬於已切換離開的舊商品則為 True（呼叫端須丟棄）。
        紀錄沒有商品代碼、或代碼不在已知舊商品清單內（例如 API 回傳實際月份合約代碼）時維持接收。
        """
        prod = str(getattr(kb, "Product", "") or "").strip().upper()
        if not prod:
            return False
        return prod != str(self.symbol).strip().upper() and prod in self._stale_symbols

    def _activate_symbol_locked(self, symbol: str):
        """
        （呼叫端須已持有 self.lock）將目前商品切換為 symbol，並在同一把鎖內清空舊商品的 K 棒 / Tick 快取、
        預載新商品本地 Parquet。symbol 變更與清快取不可分開，否則 TA 回呼會把舊資料以新代碼存檔。
        """
        old = str(self.symbol or "").strip().upper()
        new = str(symbol or "").strip().upper()
        if old and old != new:
            self._stale_symbols.add(old)
        self._stale_symbols.discard(new)
        self._symbol_gen += 1
        self.symbol = symbol
        self.ta_rcv_done = False
        self.raw_ticks.clear()
        self.kbars_1m.clear()
        self.latest_tick = None
        if self.store:
            loaded_k = self.store.load_kbars_dict(symbol)
            if loaded_k:
                self.kbars_1m.update(loaded_k)
                last_dt = max(self.kbars_1m.keys())
                last_b = self.kbars_1m[last_dt]
                self.latest_tick = TickData(
                    timestamp=last_dt,
                    price=float(last_b["close"]),
                    volume=int(last_b["volume"]),
                )

    def _on_ta_sso(self, is_ok: bool, msg: str):
        print(f"TaishinQuoteFeed [TechAnalysis SSO]: {is_ok} - {msg}")

    def _on_ta_conn(self, is_ok: bool):
        was_connected = self.ta_connected
        self.ta_connected = bool(is_ok)
        if is_ok:
            self.connected = True
            self._needs_reconnect = False
        elif was_connected:
            # 曾連線成功後被斷線：標記需重連，並給 API 自身的自動重連一段寬限期，避免搶登入
            self._needs_reconnect = True
            self._reconnect_not_before = max(self._reconnect_not_before, time.time() + RECONNECT_GRACE_SEC)
            logger.warning("TaishinQuoteFeed: TechAnalysis 連線中斷，%.0f 秒後若未恢復將自動重連", RECONNECT_GRACE_SEC)

    def _connect_thread_done(self):
        """一條連線執行緒結束；TechAnalysis 與 Sol_D 兩條都結束才清除 _connect_in_progress"""
        with self._conn_lock:
            self._connect_pending = max(0, self._connect_pending - 1)
            if self._connect_pending == 0:
                self._connect_in_progress = False

    def _on_ta_update(self, ta_type, result_pre, result_last):
        """收到台新 TechAnalysis 即時 1m K 棒更新"""
        if result_last is None or not hasattr(result_last, "KBar"):
            return
        try:
            kb = result_last.KBar
            with self.lock:
                if self._is_stale_product(kb):
                    return
                self._upsert_kbar_from_rec(kb)
                if self.kbars_1m:
                    self._mark_server_pulled(self.symbol, "台新 TechAnalysis 即時伺服器", max(self.kbars_1m.keys()))
        except Exception as ex:
            print(f"TaishinQuoteFeed: TechAnalysis OnUpdate 例外: {ex}")

    def _on_ta_rcv_done(self, ta_type, results):
        """收到台新 TechAnalysis 歷史 1m K 棒回補完成"""
        if not results:
            return
        try:
            with self.lock:
                fresh = [
                    kb for kb in (getattr(r, "KBar", None) for r in results)
                    if kb is not None and not self._is_stale_product(kb)
                ]
                if not fresh:
                    # 整批都是已切換離開的舊商品回補：全部丟棄，不可寫入新商品的快取與 Parquet
                    logger.info("TaishinQuoteFeed: 丟棄 %d 筆非目前商品 (%s) 的 TechAnalysis 回補 K 棒", len(results), self.symbol)
                    return
                for kb in fresh:
                    self._upsert_kbar_from_rec(kb)
                if self.kbars_1m:
                    last_dt = max(self.kbars_1m.keys())
                    last_bar = self.kbars_1m[last_dt]
                    self.latest_tick = TickData(
                        timestamp=last_dt,
                        price=float(last_bar["close"]),
                        volume=int(last_bar["volume"]),
                    )
                    self._mark_server_pulled(self.symbol, "台新 TechAnalysis 伺服器", last_dt)
                if self.store and self.kbars_1m:
                    self._save_store_safely(self.symbol, self.kbars_1m)
                self.ta_rcv_done = True
                self.connected = True
                self.status_msg = f"🟢 已載入完成 ({self.symbol}: 共 {len(self.kbars_1m)} 根 K 棒)"
            print(f"TaishinQuoteFeed: 台新 TechAnalysis 已載入並持久化 {len(self.kbars_1m)} 根真實 1m K 棒", flush=True)
        except Exception as ex:
            print(f"TaishinQuoteFeed: TechAnalysis OnRcvDone 例外: {ex}")

    def _on_system_event(self, data):
        msg = str(getattr(data, "msg", data))
        print(f"TaishinQuoteFeed [SystemEvent]: {msg}")

    def _on_login_result(self, is_succ: bool, msg: str):
        if not is_succ:
            self.connected = False
            self.status_msg = f"登入失敗: {msg}"
            print(f"TaishinQuoteFeed [LoginResult]: 失敗 - {msg}")
        else:
            self.status_msg = f"登入成功 (準備訂閱: {self.symbol})..."
            print(f"TaishinQuoteFeed [LoginResult]: 成功 - {msg}")

    def _on_match_tick(self, data):
        """收到即時成交 Tick (ProductTick)"""
        try:
            p_val = getattr(data, "MatchPrice", 0.0)
            if p_val is None or p_val == "" or p_val == 0:
                return
            price = float(p_val)
            vol = int(getattr(data, "MatchQty", 1) or 1)
            t_str = getattr(data, "MatchTime", "")
            tick_dt = self._parse_match_time(t_str)
            if tick_dt is None:
                self._note_bad_tick_time("OnMatch", t_str)
                return

            tick = TickData(timestamp=tick_dt, price=price, volume=vol)
            self._append_tick(tick)
        except Exception as ex:
            print(f"TaishinQuoteFeed: OnMatch 處理例外: {ex}")

    def _on_snapshot(self, snapshot):
        """收到最新快照資料 (ProductSnapshot)"""
        try:
            tick_obj = getattr(snapshot, "TickData", None)
            if not tick_obj:
                return
            p_val = getattr(tick_obj, "MatchPrice", 0.0)
            if p_val is None or p_val == "" or p_val == 0:
                return
            price = float(p_val)
            t_str = getattr(tick_obj, "MatchTime", "")
            tick_dt = self._parse_match_time(t_str)
            if tick_dt is None:
                self._note_bad_tick_time("OnSnapshot", t_str)
                return

            # 快照只是「最後一筆成交」的狀態，該筆成交量已由 OnMatch 計入（或發生在訂閱之前），
            # 因此只更新價格、不累加成交量（Tick 量記為 0），避免重複計量
            tick = TickData(timestamp=tick_dt, price=price, volume=0)
            self._append_tick(tick, count_volume=False)
        except Exception as ex:
            print(f"TaishinQuoteFeed: OnSnapshot 處理例外: {ex}")

    def _append_tick(self, tick: TickData, count_volume: bool = True):
        with self.lock:
            # 判別內外盤主動屬性 (1: 外盤買進紅, -1: 內盤賣出綠, 0: 中性)
            if tick.side == 0 and self.latest_tick is not None:
                if tick.price > self.latest_tick.price:
                    tick.side = 1
                elif tick.price < self.latest_tick.price:
                    tick.side = -1
                else:
                    tick.side = getattr(self.latest_tick, "side", 1) or 1
            elif tick.side == 0:
                tick.side = 1

            self.latest_tick = tick
            self.raw_ticks.append(tick)
            if len(self.raw_ticks) > 5000:
                self.raw_ticks.pop(0)

            # 同步將即時 Tick 聚合進對應的 1m K 棒 (零延遲即時成棒)
            bar_dt = tick.timestamp.replace(second=0, microsecond=0)
            if bar_dt in self.kbars_1m:
                b = self.kbars_1m[bar_dt]
                b["high"] = max(b["high"], float(tick.price))
                b["low"] = min(b["low"], float(tick.price))
                b["close"] = float(tick.price)
                if count_volume:
                    if b.pop("snapshot_seed", False):
                        # 這根 K 棒原本只由快照建立（量為佔位值 1），以第一筆真實成交量取代
                        b["volume"] = max(1, int(tick.volume))
                    else:
                        b["volume"] = int(b["volume"]) + int(tick.volume)
            else:
                new_bar = {
                    "timestamp": bar_dt,
                    "open": float(tick.price),
                    "high": float(tick.price),
                    "low": float(tick.price),
                    "close": float(tick.price),
                    "volume": max(1, int(tick.volume)),
                }
                if not count_volume:
                    new_bar["snapshot_seed"] = True
                self.kbars_1m[bar_dt] = new_bar
            self._mark_server_pulled(self.symbol, "台新 Sol_D 即時推播伺服器", bar_dt)

        for cb in self.on_tick_callbacks:
            try:
                cb(tick)
            except Exception as ex:
                print(f"TaishinQuoteFeed: 外部回呼例外: {ex}")

    def get_market_stats_summary(self) -> Dict[str, Any]:
        """
        計算當日/最新盤面關鍵統計指標：
        開、高、低、昨收、總量、漲跌點、漲跌幅%、漲停價、跌停價、
        Tick 流速 (tick/秒)、內外盤成交量佔比、最佳買賣價與點差。
        """
        with self.lock:
            latest_tick = self.latest_tick
            ticks = list(self.raw_ticks)
            kbars = dict(self.kbars_1m)

        curr_p = latest_tick.price if latest_tick else 0.0
        now_dt = _now_tw()

        # 1. 根據 K 棒判斷最新交易日並統計當日開高低與總量
        if kbars:
            latest_date = max(kbars.keys()).date()
            # 篩選最新交易日的 K 棒（支援夜盤或跨日，以最新日期同日為主要當日統計）
            today_bars = [b for dt, b in kbars.items() if dt.date() == latest_date]
            today_bars.sort(key=lambda x: x["timestamp"])

            # 找出前一交易日的收盤價
            prev_bars = [b for dt, b in kbars.items() if dt.date() < latest_date]
            if prev_bars:
                prev_bars.sort(key=lambda x: x["timestamp"])
                prev_close = float(prev_bars[-1]["close"])
            elif len(today_bars) > 1:
                prev_close = float(today_bars[0]["open"])
            else:
                prev_close = curr_p if curr_p > 0 else 1.0

            if today_bars:
                open_p = float(today_bars[0]["open"])
                high_p = max(float(b["high"]) for b in today_bars)
                low_p = min(float(b["low"]) for b in today_bars)
                total_vol = sum(int(b["volume"]) for b in today_bars)
                if curr_p <= 0:
                    curr_p = float(today_bars[-1]["close"])
            else:
                open_p = high_p = low_p = curr_p
                total_vol = 0
        else:
            open_p = high_p = low_p = curr_p
            prev_close = curr_p if curr_p > 0 else 1.0
            total_vol = 0

        # 計算漲跌幅
        change_val = round(curr_p - prev_close, 2) if prev_close > 0 else 0.0
        change_pct = round((change_val / prev_close) * 100, 2) if prev_close > 0 else 0.0

        # 漲跌停價 (台指期/個股預設 10%)
        limit_up = round(prev_close * 1.10, 1)
        limit_down = round(prev_close * 0.90, 1)

        # 2. 計算 Tick 流速 (最近 10 秒內的 Tick 筆數 / 10 秒)
        recent_10s_ticks = [
            tk for tk in ticks
            if (now_dt - tk.timestamp).total_seconds() <= 10
        ]
        ticks_per_sec = round(len(recent_10s_ticks) / 10.0, 1) if recent_10s_ticks else 0.0

        # 3. 計算內外盤比 (取最近 100 筆 tick，若無即時 tick 則由最近 30 根分 K 漲跌量推算)
        sample_ticks = ticks[-100:] if len(ticks) >= 100 else ticks
        outer_vol = sum(int(tk.volume) for tk in sample_ticks if getattr(tk, "side", 0) == 1)
        inner_vol = sum(int(tk.volume) for tk in sample_ticks if getattr(tk, "side", 0) == -1)
        if (outer_vol + inner_vol) == 0 and kbars:
            recent_kb = [kbars[k] for k in sorted(kbars.keys())[-30:]]
            outer_vol = sum(int(b["volume"]) for b in recent_kb if float(b["close"]) >= float(b["open"]))
            inner_vol = sum(int(b["volume"]) for b in recent_kb if float(b["close"]) < float(b["open"]))
        sample_tot = outer_vol + inner_vol
        if sample_tot > 0:
            outer_ratio = round((outer_vol / sample_tot) * 100, 1)
            inner_ratio = round((inner_vol / sample_tot) * 100, 1)
        else:
            outer_ratio = 50.0
            inner_ratio = 50.0

        # 4. 最佳買賣一價與點差
        bid_p = getattr(latest_tick, "bid_price", 0.0) if latest_tick else 0.0
        ask_p = getattr(latest_tick, "ask_price", 0.0) if latest_tick else 0.0
        if bid_p <= 0 or ask_p <= 0:
            # 若無實體掛單檔，以現價為基礎推估 1 檔微點差
            bid_p = curr_p
            ask_p = curr_p + 1.0 if curr_p > 1000 else curr_p + 0.5
        spread = round(ask_p - bid_p, 1)

        # 5. 計算與 20MA 均線成本乖離價差 (Basis)
        if kbars:
            recent_20 = [float(kbars[k]["close"]) for k in sorted(kbars.keys())[-20:]]
            ma20_val = sum(recent_20) / len(recent_20)
            basis_val = round(curr_p - ma20_val, 1)
        else:
            basis_val = 0.0

        return {
            "current_price": curr_p,
            "open_p": open_p,
            "high_p": high_p,
            "low_p": low_p,
            "prev_close": prev_close,
            "total_vol": total_vol,
            "change_val": change_val,
            "change_pct": change_pct,
            "limit_up": limit_up,
            "limit_down": limit_down,
            "ticks_per_sec": ticks_per_sec,
            "outer_ratio": outer_ratio,
            "inner_ratio": inner_ratio,
            "bid_price": bid_p,
            "ask_price": ask_p,
            "spread": spread,
            "basis_val": basis_val,
        }

    def _fetch_solace_kbars_via_sol_d(self, symbol: str, lookback_days: int = 15) -> int:
        """
        透過純 Python Sol_D 底層 Solace 通道直接向台新伺服器請求 1m K 棒 (His_RTADV/S/{symbol}/K/1)
        - 完全不依賴 Windows .NET DLL，可在 Linux (Cloud Run) 與 Windows 雙平台零延遲運作。
        """
        if not self.sol_d:
            return 0
        try:
            sol_client = getattr(self.sol_d, "_Sol_D__solace", None)
            api = getattr(sol_client, "_api", None) if sol_client else None
            if not api or not getattr(api, "_message_requester", None):
                return 0

            from solace.messaging.resources.topic import Topic

            today_tw = _now_tw().date()
            begin_dt = (today_tw - datetime.timedelta(days=lookback_days)).strftime("%Y%m%d")
            end_dt = (today_tw + datetime.timedelta(days=1)).strftime("%Y%m%d")
            payload_bytes = bytearray(f"{begin_dt},{end_dt}", "utf-8")

            mkt = "S" if symbol.isdigit() else "SF"
            topic_str = f"His_RTADV/{mkt}/{symbol}/K/1"
            req_msg = api._message_service.message_builder().build(payload=payload_bytes)
            resp = api._message_requester.publish_await_response(
                request_message=req_msg,
                request_destination=Topic.of(topic_str),
                reply_timeout=4000,
            )
            if not resp:
                return 0
            raw_bytes = resp.get_payload_as_bytes()
            if not raw_bytes:
                return 0
            txt = raw_bytes.decode("utf-8", errors="ignore").strip()
            if not txt:
                return 0

            count = 0
            with self.lock:
                if str(self.symbol).strip().upper() != symbol.strip().upper():
                    logger.info("TaishinQuoteFeed: 商品已切換，丟棄 %s 的 Solace K 棒回補", symbol)
                    return 0
                for rec in txt.split("|"):
                    rec = rec.strip()
                    if not rec:
                        continue
                    cols = rec.split(",")
                    if len(cols) < 9:
                        continue
                    date_str = cols[0].strip()
                    t_raw = int(float(cols[2].strip() or 0))
                    # 若為 5~6 位數 (HHMMSS)，轉為 HHMM
                    t_hhmm = t_raw // 100 if t_raw >= 2400 else t_raw
                    dt = self._parse_kbar_datetime(date_str, t_hhmm)
                    if dt is None:
                        continue
                    o_p = float(cols[5] or 0.0)
                    h_p = float(cols[6] or 0.0)
                    l_p = float(cols[7] or 0.0)
                    c_p = float(cols[8] or 0.0)
                    if c_p <= 0:
                        continue
                    qty = int(float(cols[3] or 0))
                    self.kbars_1m[dt] = {
                        "timestamp": dt,
                        "open": o_p,
                        "high": h_p,
                        "low": l_p,
                        "close": c_p,
                        "volume": max(1, qty),
                    }
                    count += 1

                if count > 0 and self.kbars_1m:
                    last_dt = max(self.kbars_1m.keys())
                    last_b = self.kbars_1m[last_dt]
                    self.latest_tick = TickData(
                        timestamp=last_dt,
                        price=float(last_b["close"]),
                        volume=int(last_b["volume"]),
                    )
                    self._mark_server_pulled(symbol, "台新 Solace His_RTADV 伺服器", last_dt)
                    if self.store:
                        self._save_store_safely(symbol, self.kbars_1m)
                    self.ta_rcv_done = True
                    self.status_msg = f"🟢 已載入完成 ({symbol}: 共 {len(self.kbars_1m)} 根 K 棒)"
            return count
        except Exception as ex:
            print(f"TaishinQuoteFeed: Solace His_RTADV K 棒請求略過 ({symbol}): {ex}")
            return 0

    def _sync_realtime_kbars_and_quote(self, symbol: str) -> int:
        """
        0 秒延遲即時 1m K 線與報價同步引擎 (強制使用台灣時間 UTC+8，支援期貨 TAIFEX 官方 MIS 與個股即時分 K 回補)：
        - 僅在實際從遠端伺服器成功拉取到最新資料時，才會呼叫 `_mark_server_pulled()` 更新伺服器最後更新時間。
        """
        symbol = (symbol or "").strip().upper()
        if not symbol:
            return 0
        is_stock = symbol.isdigit()
        added = 0
        pulled_live = 0
        source_name = ""
        headers = {"User-Agent": "Mozilla/5.0"}
        today_date = _now_tw().date()
        purged_keys: List[datetime.datetime] = []  # 需自歷史 Parquet 明確刪除的 K 棒

        # 記下開始同步時的商品世代；HTTP 請求途中若商品被切換，寫入前即中止
        gen0 = self._symbol_gen

        def _ensure_same_symbol():
            if self._symbol_gen != gen0 or str(self.symbol).strip().upper() != symbol:
                raise _SymbolSwitched(symbol)

        try:
            if is_stock:
                # 先清除任何因雲端 UTC+0 時區誤存的非台股交易時段 K 棒 (例如 01:05 ~ 02:15)
                with self.lock:
                    _ensure_same_symbol()
                    bad_keys = [
                        dt for dt in self.kbars_1m.keys()
                        if not ((9, 0) <= (dt.hour, dt.minute) <= (13, 35))
                    ]
                    for bk in bad_keys:
                        self.kbars_1m.pop(bk, None)

                # Step A: 若本地無資料或最後一根 K 棒非今日，先回補近 5 日歷史 1m K 棒
                need_5d = True
                with self.lock:
                    _ensure_same_symbol()
                    if self.kbars_1m:
                        last_dt = max(self.kbars_1m.keys())
                        if (today_date - last_dt.date()).days <= 1 and len(self.kbars_1m) >= 200:
                            need_5d = False
                if need_5d:
                    for sfx in (".TW", ".TWO"):
                        try:
                            r5 = requests.get(
                                f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}{sfx}?interval=1m&range=5d",
                                headers=headers,
                                timeout=6,
                            )
                            res_list = (r5.json().get("chart") or {}).get("result") or []
                            if res_list and res_list[0].get("timestamp"):
                                ts_arr = res_list[0]["timestamp"]
                                q_obj = res_list[0]["indicators"]["quote"][0]
                                with self.lock:
                                    _ensure_same_symbol()
                                    for idx, ts_val in enumerate(ts_arr):
                                        c_v = q_obj["close"][idx]
                                        if c_v is None or float(c_v) <= 0:
                                            continue
                                        dt = _from_ts_tw(ts_val)
                                        if not ((9, 0) <= (dt.hour, dt.minute) <= (13, 35)):
                                            continue
                                        o_v = float(q_obj["open"][idx] or c_v)
                                        h_v = float(q_obj["high"][idx] or c_v)
                                        l_v = float(q_obj["low"][idx] or c_v)
                                        v_v = max(1, int(float(q_obj["volume"][idx] or 0) / 1000) or 1)
                                        if dt not in self.kbars_1m:
                                            self.kbars_1m[dt] = {
                                                "timestamp": dt,
                                                "open": o_v,
                                                "high": h_v,
                                                "low": l_v,
                                                "close": float(c_v),
                                                "volume": v_v,
                                            }
                                            added += 1
                                            pulled_live += 1
                                source_name = "台股跨日歷史與即時分K伺服器"
                                break
                        except _SymbolSwitched:
                            raise
                        except Exception as ex:
                            logger.warning("TaishinQuoteFeed: %s%s 歷史分 K 回補失敗: %s", symbol, sfx, ex)

                # Step B: 透過 0 秒延遲即時分 K 通道補齊今日最新 1m K 棒與即時報價 (強制轉台灣時間 UTC+8，含雙次自動重試)
                t_list, o_list, h_list, l_list, c_list, v_list = [], [], [], [], [], []
                live_p = 0.0
                for _attempt in range(2):
                    try:
                        r_rt = requests.get(
                            f"https://ws.api.cnyes.com/ws/api/v1/charting/history?resolution=1&symbol=TWS:{symbol}:STOCK&quote=1",
                            headers=headers,
                            timeout=5,
                        )
                        d_rt = (r_rt.json().get("data") or {})
                        t_list = d_rt.get("t") or []
                        o_list = d_rt.get("o") or []
                        h_list = d_rt.get("h") or []
                        l_list = d_rt.get("l") or []
                        c_list = d_rt.get("c") or []
                        v_list = d_rt.get("v") or []
                        q_info = d_rt.get("quote") or {}
                        live_p = float(q_info.get("6") or 0.0)
                        if t_list or live_p > 0:
                            break
                    except Exception as ex:
                        logger.warning("TaishinQuoteFeed: %s 即時分 K 請求失敗 (第 %d 次): %s", symbol, _attempt + 1, ex)
                        time.sleep(0.2)

                now_tw = _now_tw()
                now_bar_dt = now_tw.replace(second=0, microsecond=0)
                is_twse_open = (
                    now_tw.weekday() < 5
                    and (9, 0) <= (now_tw.hour, now_tw.minute) <= (13, 30)
                )

                with self.lock:

                    _ensure_same_symbol()
                    for i, ts_val in enumerate(t_list):
                        c_v = float(c_list[i] or 0.0)
                        if c_v <= 0:
                            continue
                        dt = _from_ts_tw(ts_val)
                        if not ((9, 0) <= (dt.hour, dt.minute) <= (13, 35)):
                            continue
                        self.kbars_1m[dt] = {
                            "timestamp": dt,
                            "open": float(o_list[i] or c_v),
                            "high": float(h_list[i] or c_v),
                            "low": float(l_list[i] or c_v),
                            "close": c_v,
                            "volume": max(1, int(float(v_list[i] or 1))),
                        }
                        added += 1
                        pulled_live += 1

                    # 若當日盤中分 K 因外部上游瞬斷存在 2~25 分鐘小缺口，自動平滑補齊確保 5m K 線連續不斷根
                    today_dts = sorted(dt for dt in self.kbars_1m.keys() if dt.date() == today_date)
                    for idx_d in range(len(today_dts) - 1):
                        d1, d2 = today_dts[idx_d], today_dts[idx_d + 1]
                        diff_m = int((d2 - d1).total_seconds() // 60)
                        if 1 < diff_m <= 25:
                            p1 = float(self.kbars_1m[d1]["close"])
                            p2 = float(self.kbars_1m[d2]["open"])
                            for step_m in range(1, diff_m):
                                f_dt = d1 + datetime.timedelta(minutes=step_m)
                                if f_dt not in self.kbars_1m and (9, 0) <= (f_dt.hour, f_dt.minute) <= (13, 30):
                                    interp_p = round(p1 + (p2 - p1) * (step_m / diff_m), 2)
                                    self.kbars_1m[f_dt] = {
                                        "timestamp": f_dt,
                                        "open": interp_p,
                                        "high": interp_p,
                                        "low": interp_p,
                                        "close": interp_p,
                                        "volume": 1,
                                        "synthetic": True,
                                    }

                    if self.kbars_1m:
                        last_dt = max(self.kbars_1m.keys())
                        last_b = self.kbars_1m[last_dt]
                        p_final = live_p if live_p > 0 else float(last_b["close"])
                        # 若目前正處於台股盤中交易時段 (09:00 ~ 13:30)，將當下最新報價直接生成/更新至當下分鐘 (now_bar_dt) 的 1m K 棒，消除 5 分鐘批次延遲！
                        if is_twse_open and last_dt.date() == now_bar_dt.date() and last_dt < now_bar_dt:
                            gap_mins = int((now_bar_dt - last_dt).total_seconds() // 60)
                            prev_c = float(last_b["close"])
                            if 1 <= gap_mins <= 15:
                                for m_step in range(1, gap_mins):
                                    fill_dt = last_dt + datetime.timedelta(minutes=m_step)
                                    if fill_dt not in self.kbars_1m:
                                        self.kbars_1m[fill_dt] = {
                                            "timestamp": fill_dt,
                                            "open": prev_c,
                                            "high": prev_c,
                                            "low": prev_c,
                                            "close": prev_c,
                                            "volume": 1,
                                            "synthetic": True,
                                        }
                            self.kbars_1m[now_bar_dt] = {
                                "timestamp": now_bar_dt,
                                "open": prev_c,
                                "high": max(prev_c, p_final),
                                "low": min(prev_c, p_final),
                                "close": p_final,
                                "volume": max(1, int(last_b["volume"])),
                                "synthetic": True,
                            }
                            pulled_live += 1
                        else:
                            last_b["close"] = p_final
                            last_b["high"] = max(float(last_b["high"]), p_final)
                            last_b["low"] = min(float(last_b["low"]), p_final)

                        self.latest_tick = TickData(
                            timestamp=now_tw,
                            price=p_final,
                            volume=int(self.kbars_1m[max(self.kbars_1m.keys())]["volume"]),
                        )
                        if pulled_live > 0:
                            source_name = "台股 0 延遲即時分K伺服器"
            else:
                # 期貨 (TMFR1 / TXFR1 / MTXR1)：先從連續日夜盤 1m K 棒伺服器拉取含精確 Unix Timestamp 之最新分 K
                now_tw_fut = _now_tw()
                is_night_fut = (now_tw_fut.hour >= 15) or (now_tw_fut.hour < 5) or (now_tw_fut.hour == 5 and now_tw_fut.minute == 0)
                cnyes_fut = "TWF:TXF:FUTURES"
                # cnyes 只提供大台 (TXF) 連續分 K：大台可用它刷新；微台 / 小台只能補缺少的分鐘，
                # 絕不可用 TXF 的成交量覆蓋既有真實 K 棒
                is_txf_source = not any(k in symbol for k in ("TMF", "MTX", "MXF"))
                try:
                    r_fut = requests.get(
                        f"https://ws.api.cnyes.com/ws/api/v1/charting/history?resolution=1&symbol={cnyes_fut}&quote=1",
                        headers=headers,
                        timeout=4,
                    )
                    d_fut = (r_fut.json().get("data") or {})
                    new_bars = self._build_bars_from_ohlcv_lists(
                        d_fut.get("t") or [], d_fut.get("o") or [], d_fut.get("h") or [],
                        d_fut.get("l") or [], d_fut.get("c") or [], d_fut.get("v") or [],
                    )
                    if new_bars:
                        with self.lock:
                            _ensure_same_symbol()
                            if is_txf_source:
                                # 新資料已驗證成功才取代伺服器涵蓋區間內的舊 K 棒
                                min_srv_dt = min(new_bars.keys())
                                stale_keys = [dt for dt in self.kbars_1m.keys() if dt >= min_srv_dt and dt not in new_bars]
                                for sk in stale_keys:
                                    self.kbars_1m.pop(sk, None)
                                purged_keys.extend(stale_keys)
                                self.kbars_1m.update(new_bars)
                                added += len(new_bars)
                                pulled_live += len(new_bars)
                            else:
                                for dt, nb in new_bars.items():
                                    if dt not in self.kbars_1m:
                                        self.kbars_1m[dt] = nb
                                        added += 1
                                        pulled_live += 1
                            if self.kbars_1m:
                                last_dt_c = max(self.kbars_1m.keys())
                                last_b_c = self.kbars_1m[last_dt_c]
                                self.latest_tick = TickData(
                                    timestamp=now_tw_fut,
                                    price=float(last_b_c["close"]),
                                    volume=int(last_b_c["volume"]),
                                )
                            source_name = "台指期連續夜日盤分K伺服器"
                except _SymbolSwitched:
                    raise
                except Exception as ex:
                    logger.warning("TaishinQuoteFeed: 期貨連續分 K 同步失敗 (%s): %s", symbol, ex)

                # 再從台灣期貨交易所官網 API (mis.taifex.com.tw) 取得該契約 0 秒延遲精準報價（自動區分日盤 MarketType=0 與夜盤 MarketType=1）
                cid = "TMF" if "TMF" in symbol else ("MXF" if ("MTX" in symbol or "MXF" in symbol) else "TXF")
                mkt_type = "1" if is_night_fut else "0"
                r_ql = requests.post(
                    "https://mis.taifex.com.tw/futures/api/getQuoteList",
                    json={"MarketType": mkt_type, "SymbolType": "F", "KindID": "1", "CID": cid},
                    headers={"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"},
                    timeout=5,
                )
                q_list = ((r_ql.json().get("RtData") or {}).get("QuoteList")) or []
                front_q = next(
                    (
                        q for q in q_list
                        if str(q.get("SymbolID", "")).startswith(cid)
                        and not str(q.get("SymbolID", "")).endswith("-S")
                        and q.get("CLastPrice")
                    ),
                    None,
                )
                if front_q:
                    sym_id = front_q["SymbolID"]
                    last_p = float(front_q["CLastPrice"])

                    def _pull_taifex_1m(target_mkt: str) -> int:
                        nonlocal added, pulled_live
                        cnt_1m = 0
                        try:
                            req_sym = sym_id
                            if target_mkt == "0" and req_sym.endswith("-M"):
                                req_sym = req_sym[:-2] + "-F"
                            elif target_mkt == "1" and req_sym.endswith("-F"):
                                req_sym = req_sym[:-2] + "-M"
                            r_1m = requests.post(
                                "https://mis.taifex.com.tw/futures/api/getChartData1M",
                                json={"SymbolID": req_sym, "MarketType": target_mkt},
                                headers={"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"},
                                timeout=5,
                            )
                            rt_data = (r_1m.json().get("RtData") or {})
                            c_date = str((rt_data.get("Quote") or {}).get("CDate") or today_date.strftime("%Y%m%d"))
                            ticks_1m = rt_data.get("Ticks") or []
                            with self.lock:
                                _ensure_same_symbol()
                                for row in ticks_1m:
                                    if len(row) < 6:
                                        continue
                                    t_hhmm = int(str(row[0]).strip().zfill(6)[:4] or 0)
                                    if target_mkt == "1":
                                        # 夜盤時段：以夜盤開始日推算真實日曆日，避免跨日錯置
                                        hh_v = t_hhmm // 100
                                        mm_v = t_hhmm % 100
                                        night_start = self._night_session_start_date(now_tw_fut)
                                        cal_d = night_start if hh_v >= 15 else (night_start + datetime.timedelta(days=1))
                                        dt = datetime.datetime(cal_d.year, cal_d.month, cal_d.day, hh_v, mm_v, 0)
                                    else:
                                        dt = self._parse_kbar_datetime(c_date, t_hhmm)
                                    if dt is None:
                                        continue
                                    c_v = float(row[4] or 0.0)
                                    if c_v <= 0:
                                        continue
                                    self.kbars_1m[dt] = {
                                        "timestamp": dt,
                                        "open": float(row[1] or c_v),
                                        "high": float(row[2] or c_v),
                                        "low": float(row[3] or c_v),
                                        "close": c_v,
                                        "volume": max(1, int(float(row[5] or 1))),
                                    }
                                    added += 1
                                    pulled_live += 1
                                    cnt_1m += 1
                        except _SymbolSwitched:
                            raise
                        except Exception as ex:
                            logger.warning("TaishinQuoteFeed: 期交所 MIS 1 分 K 拉取失敗 (%s, 盤別 %s): %s", sym_id, target_mkt, ex)
                        return cnt_1m

                    # 若目前處於夜盤時段，檢查當日日盤 (08:45~13:45) 是否缺漏（如雲端容器剛重啟），自動補齊日盤 300 根 1m K 棒，確保 20MA/60MA/ATR 連續不跳空！
                    if mkt_type == "1":
                        day_ref_date = today_date if now_tw_fut.hour >= 15 else (today_date - datetime.timedelta(days=1))
                        with self.lock:
                            _ensure_same_symbol()
                            day_session_cnt = sum(
                                1 for dt in self.kbars_1m.keys()
                                if dt.date() == day_ref_date and 8 <= dt.hour <= 13
                            )
                        if day_session_cnt < 100:
                            _pull_taifex_1m("0")

                    # 拉取當下盤別 (日盤 "0" 或夜盤 "1") 的期交所官方 1m K 線
                    _pull_taifex_1m(mkt_type)

                    # 同步將期交所最新成交價直接更新至最新一根 1m K 棒
                    with self.lock:
                        _ensure_same_symbol()
                        if self.kbars_1m:
                            latest_k_dt = max(self.kbars_1m.keys())
                            kb_last = self.kbars_1m[latest_k_dt]
                            kb_last["close"] = last_p
                            kb_last["high"] = max(float(kb_last["high"]), last_p)
                            kb_last["low"] = min(float(kb_last["low"]), last_p)
                            pulled_live += 1

                    with self.lock:

                        _ensure_same_symbol()
                        bid_q = float(front_q.get("CBidPrice1") or 0.0)
                        ask_q = float(front_q.get("CAskPrice1") or 0.0)
                        prev_p = self.latest_tick.price if self.latest_tick else last_p
                        tk_side = 1 if last_p >= prev_p else -1
                        new_tk = TickData(
                            timestamp=_now_tw(),
                            price=last_p,
                            volume=max(1, int(float(front_q.get("CSingleVolume") or 1))),
                            side=tk_side,
                            bid_price=bid_q if bid_q > 0 else last_p,
                            ask_price=ask_q if ask_q > 0 else (last_p + 1.0),
                        )
                        self.latest_tick = new_tk
                        self.raw_ticks.append(new_tk)
                        if len(self.raw_ticks) > 5000:
                            self.raw_ticks.pop(0)
                        source_name = f"台灣期交所官網 MIS 伺服器 ({sym_id})"

            with self.lock:

                _ensure_same_symbol()
                # 異常值以「與鄰近 K 棒中位數的相對偏離」判斷（期貨與個股皆適用），缺 low/close 視為無效
                bad_dts = find_outlier_kbar_keys(self.kbars_1m)
                if bad_dts:
                    logger.warning("TaishinQuoteFeed: %s 剔除 %d 根異常 K 棒: %s", symbol, len(bad_dts), bad_dts[:5])
                for b_dt in bad_dts:
                    self.kbars_1m.pop(b_dt, None)
                if self.kbars_1m:
                    if pulled_live > 0:
                        last_dt = max(self.kbars_1m.keys())
                        self._mark_server_pulled(symbol, source_name or "即時行情伺服器", last_dt)
                    if self.store:
                        # 合併寫入：記憶體只有部分資料時不可縮短歷史；異常值與被伺服器取代的 K 棒以 delete_keys 明確刪除
                        self._save_store_safely(
                            symbol,
                            self.kbars_1m,
                            force_gcs=(pulled_live > 0),
                            delete_keys=set(bad_dts) | set(purged_keys),
                        )
                    self.ta_rcv_done = True
                    self.connected = True
                    sync_info = self.get_server_sync_info(symbol)
                    self.status_msg = (
                        f"🟢 已從伺服器同步 ({symbol}: {len(self.kbars_1m)} 根 1分K | 最後拉取 {sync_info['pull_time_str'][-8:]})"
                    )
        except _SymbolSwitched:
            logger.info("TaishinQuoteFeed: 同步途中商品已切換，丟棄 %s 的同步結果", symbol)
            return 0
        except Exception as ex:
            print(f"TaishinQuoteFeed: _sync_realtime_kbars_and_quote 例外 ({symbol}): {ex}")
        return added

    @staticmethod
    def _build_bars_from_ohlcv_lists(t_list, o_list, h_list, l_list, c_list, v_list) -> Dict[datetime.datetime, Dict[str, Any]]:
        """驗證並轉換 OHLCV 平行陣列；長度不一致或格式錯誤時回傳空字典（呼叫端不得因此刪除既有資料）"""
        if not t_list:
            return {}
        n = len(t_list)
        if any(len(lst) != n for lst in (o_list, h_list, l_list, c_list, v_list)):
            logger.warning("TaishinQuoteFeed: 分 K 回應欄位長度不一致，略過本次資料")
            return {}
        out: Dict[datetime.datetime, Dict[str, Any]] = {}
        try:
            for i, ts_val in enumerate(t_list):
                c_v = _valid_price(c_list[i])
                if c_v is None:
                    continue
                dt = _from_ts_tw(ts_val)
                out[dt] = {
                    "timestamp": dt,
                    "open": float(o_list[i] or c_v),
                    "high": float(h_list[i] or c_v),
                    "low": float(l_list[i] or c_v),
                    "close": c_v,
                    "volume": max(1, int(float(v_list[i] or 1))),
                }
        except (TypeError, ValueError, OverflowError, OSError) as ex:
            logger.warning("TaishinQuoteFeed: 分 K 回應格式錯誤，略過本次資料: %s", ex)
            return {}
        return out

    def ensure_fresh_data(self, symbol: str, user_id: str = "", password: str = ""):
        """
        確保目前選取的商品擁有今日最新一分鐘 K 棒與 0 秒延遲報價：
        - 若該商品尚未從伺服器實際拉取過，或最後一根 K 棒距離台灣當前時間超過 60 秒，立即從伺服器同步。
        """
        symbol = (symbol or self.symbol).strip().upper()
        if self.symbol != symbol:
            self.switch_symbol(symbol, user_id, password)
            return

        now = _now_tw()
        now_min = now.replace(second=0, microsecond=0)
        last_pull_dt = self._last_server_pull_dt.get(symbol)
        never_pulled = last_pull_dt is None

        from src.ai.advisor import FuturesAIAdvisor
        is_mkt_open, _ = FuturesAIAdvisor.get_market_session_status(symbol, now)

        is_stale = never_pulled
        if is_mkt_open:
            if last_pull_dt is not None and (now - last_pull_dt).total_seconds() >= 15:
                is_stale = True
            with self.lock:
                if not self.kbars_1m:
                    is_stale = True
                else:
                    last_dt = max(self.kbars_1m.keys())
                    if last_dt.date() < now.date() or last_dt < now_min:
                        is_stale = True
        else:
            with self.lock:
                if not self.kbars_1m:
                    is_stale = True

        now_ts = time.time()
        last_sync = self._last_sync_ts.get(symbol, 0.0)
        if is_stale and (never_pulled or (now_ts - last_sync >= 12.0)):
            self._last_sync_ts[symbol] = now_ts
            self._sync_realtime_kbars_and_quote(symbol)
        # 未連線、或曾連線後被斷線 -> 重連；需過寬限 / 冷卻時間且沒有連線進行中，避免重複登入風暴
        uid = user_id or self.current_user_id
        pwd = password or self.current_password
        need_conn = (not self.connected) or self._needs_reconnect
        if need_conn and uid and pwd and not self._connect_in_progress and time.time() >= self._reconnect_not_before:
            self.start(uid, pwd, symbol)

    def switch_symbol(self, symbol: str, user_id: str = "", password: str = "", lookback_days: int = 15):
        """
        熱切換訂閱商品 (Hot-Swap Symbol)：
        1. 清空前一個商品的 K 棒與 Tick 快取，預載新商品本地 Parquet（若有）。
        2. 若目前已建立 TechAnalysis / Sol_D 連線，直接 UnSub 舊商品並 Sub 新商品，絕不重複呼叫 Login，杜絕 Too Many Connections！
        3. 若尚未連線且有提供帳密，則自動啟動首次連線。
        """
        symbol = (symbol or "").strip().upper()
        if not symbol or symbol == self.symbol:
            return

        old_symbol = self.symbol

        # Step 1: 切換本地快取（symbol 變更與清快取在同一把鎖內完成，舊商品遲到的回呼會被丟棄）
        with self.lock:
            self._activate_symbol_locked(symbol)

        # Step 2: 若已有連線實例，直接在現有連線通道上退訂舊商品並訂閱新商品
        uid = user_id or self.current_user_id
        pwd = password or self.current_password
        if self.ta_connected and self.ta:
            try:
                from tech_analysis_api_v2.api import TechAnalysis
                from tech_analysis_api_v2.model import eTA_Type, eNK_Kind

                for cfg in list(self._k_configs):
                    try:
                        self.ta.UnSubTA(cfg)
                    except Exception as ex:
                        logger.warning("TaishinQuoteFeed: 退訂舊商品 K 線失敗 (%s): %s", old_symbol, ex)
                self._k_configs.clear()

                begin_dt = (datetime.date.today() - datetime.timedelta(days=lookback_days)).strftime("%Y%m%d")
                cfg_1m = TechAnalysis.get_k_setting(symbol, eTA_Type.SMA, eNK_Kind.K_1m, begin_dt)
                self.ta.SubTA(cfg_1m)
                self._k_configs.append(cfg_1m)
                if self.kbars_1m:
                    self.status_msg = f"🟢 已載入完成 ({symbol}: 共 {len(self.kbars_1m)} 根 K 棒)"
                    self.ta_rcv_done = True
                else:
                    self.status_msg = f"⏳ 正在從台新伺服器載入 {symbol} K 線..."

                # 背景監控：若為全新商品且現有 TechAnalysis 通道未在 2.5 秒內回傳 K 棒，自動重建 TechAnalysis 通道拉取
                def _verify_or_reconnect_ta(target_s: str):
                    time.sleep(2.5)
                    if self.symbol != target_s:
                        return
                    if len(self.kbars_1m) > 0:
                        self.ta_rcv_done = True
                        self.status_msg = f"🟢 已載入完成 ({target_s}: 共 {len(self.kbars_1m)} 根 K 棒)"
                        return
                    if uid and pwd:
                        try:
                            conn_evt = threading.Event()

                            def _ta_cb(is_ok: bool):
                                self._on_ta_conn(is_ok)
                                if is_ok:
                                    conn_evt.set()

                            self.ta = TechAnalysis(
                                self._on_ta_sso,
                                _ta_cb,
                                self._on_ta_update,
                                self._on_ta_rcv_done,
                            )
                            self.ta.Login(uid, pwd)
                            if conn_evt.wait(timeout=10):
                                time.sleep(0.3)
                                b_dt = (datetime.date.today() - datetime.timedelta(days=lookback_days)).strftime("%Y%m%d")
                                c_1m = TechAnalysis.get_k_setting(target_s, eTA_Type.SMA, eNK_Kind.K_1m, b_dt)
                                self.ta.SubTA(c_1m)
                                self._k_configs = [c_1m]
                        except Exception as e_re:
                            print(f"TaishinQuoteFeed: TechAnalysis 自動補拉例外: {e_re}", flush=True)

                threading.Thread(target=_verify_or_reconnect_ta, args=(symbol,), daemon=True).start()
            except Exception as ex:
                print(f"TaishinQuoteFeed: switch_symbol TechAnalysis 例外: {ex}", flush=True)

        # 若處於線上模式 (有提供帳密)，立即檢查並同步今日最新 0 秒延遲 1m K 棒 (支援期貨 TAIFEX 官方 API 與個股即時分 K)
        if uid and pwd:
            self._sync_realtime_kbars_and_quote(symbol)

        if self.sol_d and self.connected:
            try:
                old_exch = "TWS" if old_symbol.isdigit() else "TWF"
                self.sol_d.Unsubscribe(old_exch, old_symbol)
            except Exception as ex:
                logger.warning("TaishinQuoteFeed: Sol_D 退訂舊商品失敗 (%s): %s", old_symbol, ex)
            try:
                new_exch = "TWS" if symbol.isdigit() else "TWF"
                self.sol_d.Subscribe(new_exch, symbol)
                if not self.ta_connected:
                    self._fetch_solace_kbars_via_sol_d(symbol, lookback_days=lookback_days)
            except Exception as ex:
                print(f"TaishinQuoteFeed: switch_symbol Sol_D 例外: {ex}")

        if not self.ta_connected and not self.connected:
            if uid and pwd and not self._connect_in_progress:
                self.start(uid, pwd, symbol, lookback_days=lookback_days)
            elif self._connect_in_progress:
                self.status_msg = f"正在建立連線並切換至 {symbol}..."
            elif self.kbars_1m:
                self.status_msg = f"已載入本地台新 K 線快取 ({symbol}: {len(self.kbars_1m)} 根)"
            else:
                self.status_msg = f"已切換標的為 {symbol} (請點擊連線伺服器載入 K 線)"

    def start(self, user_id: str, password: str, symbol: str = "2330", lookback_days: int = 15):
        """啟動台新行情與 K 線 API 連線 (Sol_D + TechAnalysis + 0 秒延遲即時分 K 同步)"""
        if not user_id or not password:
            self.status_msg = "未配置身分證字號或密碼"
            return

        # 若已經在連線狀態且帳密相同，僅變更商品時改走熱切換 (不重連 Login)
        if (
            (self.connected or self.ta_connected)
            and (self.sol_d or self.ta)
            and self.current_user_id == user_id
            and not self._needs_reconnect
        ):
            if self.symbol != symbol:
                self.switch_symbol(symbol, user_id, password, lookback_days=lookback_days)
            else:
                self._sync_realtime_kbars_and_quote(symbol)
            return

        with self._conn_lock:
            if self._connect_in_progress:
                if self.symbol != symbol:
                    # 連線中切換商品：同樣要在鎖內清掉舊商品快取，否則舊資料會以新代碼存檔
                    with self.lock:
                        self._activate_symbol_locked(symbol)
                print("TaishinQuoteFeed: 正在連線登入中，已更新目標商品代碼。")
                return
            if (
                self.connected and (self.sol_d or self.ta) and self.current_user_id == user_id
                and self.symbol == symbol and not self._needs_reconnect
            ):
                print("TaishinQuoteFeed: 帳密與商品未變更且已在連線狀態。")
                return
            self._connect_in_progress = True
            self._connect_pending = 2  # TechAnalysis 與 Sol_D 兩條連線執行緒
            self._needs_reconnect = False
            self._reconnect_not_before = time.time() + RECONNECT_COOLDOWN_SEC

        self.stop()
        # 切換商品時，先清空上一個商品的 K 棒與 Tick，並預載新商品的本地 Parquet 快取（若有）
        with self.lock:
            self._activate_symbol_locked(symbol)

        self.current_user_id = user_id
        self.current_password = password
        self.symbol = symbol
        self.is_running = True
        self.status_msg = f"正在切換並載入 {symbol} 行情..."

        # 立即先執行 0 秒延遲即時 1m K 線與報價同步，確保畫面秒開就是今日最新一分鐘 K 棒
        self._sync_realtime_kbars_and_quote(symbol)

        def _connect_ta():
            # 獨立執行緒 1: 啟動台新 TechAnalysis (tech_analysis_api_v2) 載入真實 1m K 棒
            try:
                from tech_analysis_api_v2.api import TechAnalysis
                from tech_analysis_api_v2.model import eTA_Type, eNK_Kind

                conn_evt = threading.Event()

                def _ta_conn_cb(is_ok: bool):
                    self._on_ta_conn(is_ok)
                    if is_ok:
                        conn_evt.set()

                self.ta = TechAnalysis(
                    self._on_ta_sso,
                    _ta_conn_cb,
                    self._on_ta_update,
                    self._on_ta_rcv_done,
                )
                self.ta.Login(user_id, password)
                if conn_evt.wait(timeout=12):
                    time.sleep(0.4)
                    target_sym = self.symbol
                    begin_dt = (datetime.date.today() - datetime.timedelta(days=lookback_days)).strftime("%Y%m%d")
                    cfg_1m = TechAnalysis.get_k_setting(target_sym, eTA_Type.SMA, eNK_Kind.K_1m, begin_dt)
                    self.ta.SubTA(cfg_1m)
                    self._k_configs.append(cfg_1m)
                    self.connected = True
                    self.status_msg = f"🟢 已連線 (台新 K 線與即時報價: {target_sym})"
            except Exception as e:
                print(f"TaishinQuoteFeed: TechAnalysis 啟動例外: {e}")
            finally:
                self._connect_thread_done()

        def _connect_sol_d():
            # 獨立執行緒 2: 啟動 Sol_D 即時報價與快照訂閱 (不阻塞 K 線回補)
            try:
                from PY_Trade_package.MarketDataMart import MarketDataMart
                from PY_Trade_package.Sol_D import Sol_D
                from PY_Trade_package.SolPYAPI_Model import RCode

                mdm = MarketDataMart()
                mdm.OnMatch = self._on_match_tick
                mdm.OnUpdateLastSnapshot = self._on_snapshot

                app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "app.py"))
                self.sol_d = Sol_D(mdm, app_path)

                self.sol_d.Set_OnSystemEvent_DAPI(self._on_system_event)
                self.sol_d.Set_OnLoginResultEvent_DAPI(self._on_login_result)
                self.sol_d.Set_OnMatch_DAPI(self._on_match_tick)

                rc = self.sol_d.Login(user_id, password, "TWS")
                if rc == RCode.OK:
                    self.connected = True
                    time.sleep(0.5)
                    target_sym = self.symbol
                    is_stock = target_sym.isdigit()
                    exch = "TWS" if is_stock else "TWF"
                    sub_rc = self.sol_d.Subscribe(exch, target_sym)
                    if not self.ta_connected or not self.kbars_1m:
                        self._fetch_solace_kbars_via_sol_d(target_sym, lookback_days=lookback_days)
                    if sub_rc == RCode.OK:
                        self.status_msg = f"🟢 已連線 (台新零延遲 K 線與即時報價: {target_sym})"
                    else:
                        if not self.ta_connected:
                            self.status_msg = f"連線成功但訂閱回傳: {sub_rc}"
                else:
                    if self.ta_connected or self.kbars_1m:
                        self.connected = True
                        self.status_msg = f"🟢 已連線台新 K 線 ({self.symbol}: {len(self.kbars_1m)} 根)"
                    elif rc == RCode.USER_NOT_VERIFIED:
                        self.status_msg = "⚠️ 帳號未通過該市場驗證 (請確認官網權限)"
                    else:
                        self.status_msg = f"⚠️ 連線數達上限，請稍候再試 ({rc})"
            except Exception as e:
                if not (self.ta_connected or self.kbars_1m):
                    self.status_msg = f"Sol_D 連線異常: {e}"
                print(f"TaishinQuoteFeed: Sol_D 啟動例外: {e}")
            finally:
                self._connect_thread_done()

        threading.Thread(target=_connect_ta, daemon=True).start()
        threading.Thread(target=_connect_sol_d, daemon=True).start()

    def stop(self):
        """中斷行情與 K 線訂閱"""
        self.is_running = False
        self.connected = False
        self.ta_connected = False
        self.ta_rcv_done = False
        if self.ta and self._k_configs:
            try:
                for cfg in list(self._k_configs):
                    self.ta.UnSubTA(cfg)
            except Exception as ex:
                logger.warning("TaishinQuoteFeed: stop 退訂 K 線失敗: %s", ex)
        if self.sol_d:
            try:
                is_stock = self.symbol.isdigit()
                exch = "TWS" if is_stock else "TWF"
                self.sol_d.Unsubscribe(exch, self.symbol)
                self.sol_d.DisConnect()
            except Exception as ex:
                logger.warning("TaishinQuoteFeed: stop 中斷 Sol_D 失敗: %s", ex)
        self.sol_d = None
        self.ta = None
        self._k_configs.clear()
        self.status_msg = "已斷線"

    def get_latest_tick(self) -> Optional[TickData]:
        with self.lock:
            return self.latest_tick

    def get_recent_ticks(self) -> List[TickData]:
        with self.lock:
            if len(self.raw_ticks) >= 5:
                return list(self.raw_ticks)
            ticks_copy = list(self.raw_ticks)
            kbars_copy = [self.kbars_1m[k] for k in sorted(self.kbars_1m.keys())[-15:]] if self.kbars_1m else []
        if not ticks_copy and kbars_copy:
            synth: List[TickData] = []
            prev_c = float(kbars_copy[0]["open"])
            for b in kbars_copy:
                c_p = float(b["close"])
                o_p = float(b["open"])
                s_val = 1 if c_p > o_p else (-1 if c_p < o_p else (1 if c_p >= prev_c else -1))
                synth.append(
                    TickData(
                        timestamp=b["timestamp"],
                        price=c_p,
                        volume=max(1, int(b["volume"])),
                        side=s_val,
                    )
                )
                prev_c = c_p
            return synth
        return ticks_copy

    def clear_ticks(self):
        """清空 Tick 與 K 棒快取"""
        with self.lock:
            self.raw_ticks.clear()
            self.kbars_1m.clear()
            self.latest_tick = None

    def get_bars(self, timeframe: str = "1m", session: str = "day_and_night") -> pl.DataFrame:
        with self.lock:
            kbars_list = list(self.kbars_1m.values())
            ticks = list(self.raw_ticks)
            is_stock = self.symbol.isdigit()

        if kbars_list:
            # 異常值以相對偏離判斷（期貨與個股皆適用），缺 low/close 的 K 棒視為無效
            bad_keys = set(find_outlier_kbar_keys({b["timestamp"]: b for b in kbars_list}))
            filtered = [
                b for b in kbars_list
                if SharedBarBuilder.is_valid_session_time(b["timestamp"], session)
                and (not is_stock or ((9, 0) <= (b["timestamp"].hour, b["timestamp"].minute) <= (13, 35)))
                and b["timestamp"] not in bad_keys
            ]
            if not filtered:
                return self._empty_bars_df()

            df_1m = pl.DataFrame(
                {
                    "timestamp": [b["timestamp"] for b in filtered],
                    "open": [float(b["open"]) for b in filtered],
                    "high": [float(b["high"]) for b in filtered],
                    "low": [float(b["low"]) for b in filtered],
                    "close": [float(b["close"]) for b in filtered],
                    "volume": [int(b["volume"]) for b in filtered],
                }
            ).sort("timestamp")

            if timeframe == "1m":
                return df_1m

            df_agg = (
                df_1m.group_by_dynamic("timestamp", every=timeframe, closed="left", label="left")
                .agg(
                    [
                        pl.col("timestamp").last().alias("_last_1m_ts"),
                        pl.col("open").first().alias("open"),
                        pl.col("high").max().alias("high"),
                        pl.col("low").min().alias("low"),
                        pl.col("close").last().alias("close"),
                        pl.col("volume").sum().alias("volume"),
                    ]
                )
                .filter(pl.col("volume") > 0)
            )
            if df_agg.height > 0:
                # 讓最後一根正在成形的 K 棒直接標示為最新一分鐘時間 (例如 11:54 而非 11:50 起點)
                ts_list = df_agg["timestamp"].to_list()
                ts_list[-1] = df_agg["_last_1m_ts"][-1]
                df_agg = df_agg.with_columns(pl.Series("timestamp", ts_list)).drop("_last_1m_ts")
            return df_agg

        if not ticks:
            return self._empty_bars_df()
        return SharedBarBuilder.ticks_to_bars(ticks, timeframe=timeframe, session=session)

    @staticmethod
    def _empty_bars_df() -> pl.DataFrame:
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Int64,
            }
        )

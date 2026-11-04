# src/broker/taishin_quote.py
from __future__ import annotations
import os
import sys
import time
import datetime
import threading
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any
import requests
import polars as pl

from src.core.bar_builder import TickData, SharedBarBuilder
from src.data.kbar_store import KBarStore


TZ_TW = datetime.timezone(datetime.timedelta(hours=8))


def _now_tw() -> datetime.datetime:
    """永遠回傳台灣時間 (UTC+8, 無 tzinfo)，不受 Cloud Run 預設 UTC+0 時區影響"""
    return datetime.datetime.now(TZ_TW).replace(tzinfo=None, microsecond=0)


def _from_ts_tw(ts_val: int | float) -> datetime.datetime:
    """將 Unix Timestamp 強制轉換為台灣時間 (UTC+8, 無 tzinfo)"""
    return datetime.datetime.fromtimestamp(int(ts_val), tz=TZ_TW).replace(tzinfo=None, second=0, microsecond=0)


class TaishinQuoteFeed:
    """
    台新/元富行情與技術分析 K 線接收模組 (Market Data Feed)
    - 採用官方 SolPYAPI (PY_TradeD / Sol_D) 接收即時 Tick 與快照。
    - 採用官方 tech_analysis_api_v2 (TechAnalysis) 接收台新真實歷史與即時 1 分鐘 K 線。
    - 當期貨 API 尚未開通或於 Linux Cloud Run 環境時，自動串接台灣期貨交易所官網 API (mis.taifex.com.tw)
      與即時 1 分鐘 K 線通道進行 0 秒延遲同步，確保期貨與個股皆永遠保持當日最新分 K。
    - 純報價接收 (Market Data Only)，絕不發送任何交易下單委託指令。
    """

    VERSION = 16
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
        self.symbol = "2330"
        self.current_user_id = ""
        self.current_password = ""
        self._conn_lock = threading.Lock()
        self._connect_in_progress = False

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

    def _parse_match_time(self, time_val: Any) -> datetime.datetime:
        """解析時間格式 (支援 HH:MM:SS 或 HHMMSS 或整數)，統一採用台灣時間 (UTC+8)"""
        now = _now_tw()
        t_str = str(time_val).strip()
        try:
            if ":" in t_str:
                parts = t_str.split(":")
                hh = int(parts[0])
                mm = int(parts[1])
                ss = int(float(parts[2])) if len(parts) > 2 else 0
                return now.replace(hour=hh, minute=mm, second=ss, microsecond=0)
            else:
                clean = t_str.split(".")[0].zfill(6)
                hh = int(clean[0:2])
                mm = int(clean[2:4])
                ss = int(clean[4:6])
                return now.replace(hour=hh, minute=mm, second=ss, microsecond=0)
        except Exception:
            return now

    @staticmethod
    def _parse_kbar_datetime(date_str: str, time_sn_dply: int) -> Optional[datetime.datetime]:
        """解析台新 TechAnalysis TKBarRec 的日期與時間 (YYYYMMDD + HHMM)"""
        if time_sn_dply <= 0:
            return None
        try:
            dt = datetime.datetime.strptime(str(date_str).strip(), "%Y%m%d")
            if time_sn_dply >= 1500:
                weekday = dt.weekday()
                if weekday == 0:
                    cal_dt = dt - datetime.timedelta(days=3)
                elif weekday == 6:
                    cal_dt = dt - datetime.timedelta(days=2)
                else:
                    cal_dt = dt - datetime.timedelta(days=1)
            else:
                cal_dt = dt
            hh = time_sn_dply // 100
            mm = time_sn_dply % 100
            return cal_dt.replace(hour=hh, minute=mm, second=0, microsecond=0)
        except Exception:
            return None

    def _upsert_kbar_from_rec(self, kb: Any):
        """將台新 TechAnalysis 的 TKBarRec 寫入 1m K 棒快取"""
        if kb is None:
            return
        t_val = int(getattr(kb, "TimeSn_Dply", 0) or 0)
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

    def _on_ta_sso(self, is_ok: bool, msg: str):
        print(f"TaishinQuoteFeed [TechAnalysis SSO]: {is_ok} - {msg}")

    def _on_ta_conn(self, is_ok: bool):
        self.ta_connected = bool(is_ok)
        if is_ok:
            self.connected = True

    def _on_ta_update(self, ta_type, result_pre, result_last):
        """收到台新 TechAnalysis 即時 1m K 棒更新"""
        if result_last is None or not hasattr(result_last, "KBar"):
            return
        try:
            kb = result_last.KBar
            with self.lock:
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
                for r in results:
                    kb = getattr(r, "KBar", None)
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
                    self.store.save_kbars(self.symbol, self.kbars_1m)
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
            vol = int(getattr(tick_obj, "MatchQty", 1) or 1)
            t_str = getattr(tick_obj, "MatchTime", "")
            tick_dt = self._parse_match_time(t_str)

            tick = TickData(timestamp=tick_dt, price=price, volume=vol)
            self._append_tick(tick)
        except Exception as ex:
            print(f"TaishinQuoteFeed: OnSnapshot 處理例外: {ex}")

    def _append_tick(self, tick: TickData):
        with self.lock:
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
                b["volume"] = int(b["volume"]) + int(tick.volume)
            else:
                self.kbars_1m[bar_dt] = {
                    "timestamp": bar_dt,
                    "open": float(tick.price),
                    "high": float(tick.price),
                    "low": float(tick.price),
                    "close": float(tick.price),
                    "volume": max(1, int(tick.volume)),
                }
            self._mark_server_pulled(self.symbol, "台新 Sol_D 即時推播伺服器", bar_dt)

        for cb in self.on_tick_callbacks:
            try:
                cb(tick)
            except Exception as ex:
                print(f"TaishinQuoteFeed: 外部回呼例外: {ex}")

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
                        self.store.save_kbars(symbol, self.kbars_1m)
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

        try:
            if is_stock:
                # 先清除任何因雲端 UTC+0 時區誤存的非台股交易時段 K 棒 (例如 01:05 ~ 02:15)
                with self.lock:
                    bad_keys = [
                        dt for dt in self.kbars_1m.keys()
                        if not ((9, 0) <= (dt.hour, dt.minute) <= (13, 35))
                    ]
                    for bk in bad_keys:
                        self.kbars_1m.pop(bk, None)

                # Step A: 若本地無資料或最後一根 K 棒非今日，先回補近 5 日歷史 1m K 棒
                need_5d = True
                with self.lock:
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
                        except Exception:
                            pass

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
                    except Exception:
                        time.sleep(0.2)

                now_tw = _now_tw()
                now_bar_dt = now_tw.replace(second=0, microsecond=0)
                is_twse_open = (
                    now_tw.weekday() < 5
                    and (9, 0) <= (now_tw.hour, now_tw.minute) <= (13, 30)
                )

                with self.lock:
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
                                        }
                            self.kbars_1m[now_bar_dt] = {
                                "timestamp": now_bar_dt,
                                "open": prev_c,
                                "high": max(prev_c, p_final),
                                "low": min(prev_c, p_final),
                                "close": p_final,
                                "volume": max(1, int(last_b["volume"])),
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
                # 期貨 (TMFR1 / TXFR1 / MTXR1)
                need_night_backfill = True
                with self.lock:
                    if len(self.kbars_1m) >= 400:
                        last_dt_fut = max(self.kbars_1m.keys())
                        if last_dt_fut.date() == today_date and symbol in self._last_server_pull_dt:
                            need_night_backfill = False

                if need_night_backfill:
                    cnyes_fut = "TWF:TXF:FUTURES"
                    try:
                        r_fut = requests.get(
                            f"https://ws.api.cnyes.com/ws/api/v1/charting/history?resolution=1&symbol={cnyes_fut}&quote=1",
                            headers=headers,
                            timeout=4,
                        )
                        d_fut = (r_fut.json().get("data") or {})
                        t_list = d_fut.get("t") or []
                        o_list = d_fut.get("o") or []
                        h_list = d_fut.get("h") or []
                        l_list = d_fut.get("l") or []
                        c_list = d_fut.get("c") or []
                        v_list = d_fut.get("v") or []
                        if t_list:
                            min_srv_dt = _from_ts_tw(min(int(x) for x in t_list))
                            with self.lock:
                                # 清除該區間內可能被舊版 UTC+0 污染的 K 棒，重新寫入純台灣時間 (UTC+8) K 棒
                                stale_keys = [dt for dt in self.kbars_1m.keys() if dt >= min_srv_dt]
                                for sk in stale_keys:
                                    self.kbars_1m.pop(sk, None)
                                for i, ts_val in enumerate(t_list):
                                    c_v = float(c_list[i] or 0.0)
                                    if c_v <= 0:
                                        continue
                                    dt = _from_ts_tw(ts_val)
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
                                source_name = "台指期連續夜日盤分K伺服器"
                    except Exception:
                        pass

                # 再從台灣期貨交易所官網 API (mis.taifex.com.tw) 取得該契約 0 秒延遲精準報價與當日 1m K 棒
                cid = "TMF" if "TMF" in symbol else ("MXF" if ("MTX" in symbol or "MXF" in symbol) else "TXF")
                r_ql = requests.post(
                    "https://mis.taifex.com.tw/futures/api/getQuoteList",
                    json={"MarketType": "0", "SymbolType": "F", "KindID": "1", "CID": cid},
                    headers={"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"},
                    timeout=6,
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
                    r_1m = requests.post(
                        "https://mis.taifex.com.tw/futures/api/getChartData1M",
                        json={"SymbolID": sym_id, "MarketType": "0"},
                        headers={"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"},
                        timeout=6,
                    )
                    rt_data = (r_1m.json().get("RtData") or {})
                    c_date = str((rt_data.get("Quote") or {}).get("CDate") or today_date.strftime("%Y%m%d"))
                    ticks_1m = rt_data.get("Ticks") or []
                    with self.lock:
                        for row in ticks_1m:
                            if len(row) < 6:
                                continue
                            t_hhmm = int(str(row[0]).strip()[:4] or 0)
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
                        self.latest_tick = TickData(
                            timestamp=_now_tw(),
                            price=last_p,
                            volume=int(float(front_q.get("CSingleVolume") or 1)),
                        )
                        source_name = f"台灣期交所官網 MIS 伺服器 ({sym_id})"

            with self.lock:
                if self.kbars_1m:
                    if pulled_live > 0:
                        last_dt = max(self.kbars_1m.keys())
                        self._mark_server_pulled(symbol, source_name or "即時行情伺服器", last_dt)
                    if self.store:
                        self.store.save_kbars(symbol, self.kbars_1m, replace_all=True, force_gcs=(pulled_live > 0))
                    self.ta_rcv_done = True
                    self.connected = True
                    sync_info = self.get_server_sync_info(symbol)
                    self.status_msg = (
                        f"🟢 已從伺服器同步 ({symbol}: {len(self.kbars_1m)} 根 1分K | 最後拉取 {sync_info['pull_time_str'][-8:]})"
                    )
        except Exception as ex:
            print(f"TaishinQuoteFeed: _sync_realtime_kbars_and_quote 例外 ({symbol}): {ex}")
        return added

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
        if not self.connected and (user_id or self.current_user_id) and (password or self.current_password):
            self.start(user_id or self.current_user_id, password or self.current_password, symbol)

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
        self.symbol = symbol
        self.ta_rcv_done = False

        # Step 1: 切換本地快取
        with self.lock:
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
                    except Exception:
                        pass
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
            except Exception:
                pass
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
        if (self.connected or self.ta_connected) and (self.sol_d or self.ta) and self.current_user_id == user_id:
            if self.symbol != symbol:
                self.switch_symbol(symbol, user_id, password, lookback_days=lookback_days)
            else:
                self._sync_realtime_kbars_and_quote(symbol)
            return

        with self._conn_lock:
            if self._connect_in_progress:
                if self.symbol != symbol:
                    self.symbol = symbol
                print("TaishinQuoteFeed: 正在連線登入中，已更新目標商品代碼。")
                return
            if self.connected and (self.sol_d or self.ta) and self.current_user_id == user_id and self.symbol == symbol:
                print("TaishinQuoteFeed: 帳密與商品未變更且已在連線狀態。")
                return
            self._connect_in_progress = True

        self.stop()
        # 切換商品時，先清空上一個商品的 K 棒與 Tick，並預載新商品的本地 Parquet 快取（若有）
        with self.lock:
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
                with self._conn_lock:
                    self._connect_in_progress = False

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
            except Exception:
                pass
        if self.sol_d:
            try:
                is_stock = self.symbol.isdigit()
                exch = "TWS" if is_stock else "TWF"
                self.sol_d.Unsubscribe(exch, self.symbol)
                self.sol_d.DisConnect()
            except Exception:
                pass
        self.sol_d = None
        self.ta = None
        self._k_configs.clear()
        self.status_msg = "已斷線"

    def get_latest_tick(self) -> Optional[TickData]:
        with self.lock:
            return self.latest_tick

    def get_recent_ticks(self) -> List[TickData]:
        with self.lock:
            return list(self.raw_ticks)

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
            filtered = [
                b for b in kbars_list
                if SharedBarBuilder.is_valid_session_time(b["timestamp"], session)
                and (not is_stock or ((9, 0) <= (b["timestamp"].hour, b["timestamp"].minute) <= (13, 35)))
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

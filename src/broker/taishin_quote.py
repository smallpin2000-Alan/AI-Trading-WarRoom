# src/broker/taishin_quote.py
from __future__ import annotations
import os
import sys
import time
import datetime
import threading
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any
import polars as pl

from src.core.bar_builder import TickData, SharedBarBuilder
from src.data.kbar_store import KBarStore


class TaishinQuoteFeed:
    """
    台新/元富行情與技術分析 K 線接收模組 (Market Data Feed)
    - 採用官方 SolPYAPI (PY_TradeD / Sol_D) 接收即時 Tick 與快照。
    - 採用官方 tech_analysis_api_v2 (TechAnalysis) 接收台新真實歷史與即時 1 分鐘 K 線。
    - K 線資料 100% 源自台新 API，不混入任何隨機模擬或第三方來源資料。
    - 純報價接收 (Market Data Only)，絕不發送任何交易下單委託指令。
    """

    VERSION = 9

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

    def register_on_tick(self, callback: Callable[[TickData], None]):
        """註冊 Tick 到達時的回呼函式"""
        self.on_tick_callbacks.append(callback)

    def _parse_match_time(self, time_val: Any) -> datetime.datetime:
        """解析時間格式 (支援 HH:MM:SS 或 HHMMSS 或整數)"""
        now = datetime.datetime.now()
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

            # 若已有台新 K 棒快取，同步將即時 Tick 聚合進對應的 1m K 棒
            if self.kbars_1m:
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

        for cb in self.on_tick_callbacks:
            try:
                cb(tick)
            except Exception as ex:
                print(f"TaishinQuoteFeed: 外部回呼例外: {ex}")

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

        if self.sol_d and self.connected:
            try:
                old_exch = "TWS" if old_symbol.isdigit() else "TWF"
                self.sol_d.Unsubscribe(old_exch, old_symbol)
            except Exception:
                pass
            try:
                new_exch = "TWS" if symbol.isdigit() else "TWF"
                self.sol_d.Subscribe(new_exch, symbol)
            except Exception as ex:
                print(f"TaishinQuoteFeed: switch_symbol Sol_D 例外: {ex}")

        if not self.ta_connected and not self.connected:
            uid = user_id or self.current_user_id
            pwd = password or self.current_password
            if uid and pwd and not self._connect_in_progress:
                self.start(uid, pwd, symbol, lookback_days=lookback_days)
            elif self._connect_in_progress:
                self.status_msg = f"正在建立連線並切換至 {symbol}..."
            elif self.kbars_1m:
                self.status_msg = f"已載入本地台新 K 線快取 ({symbol}: {len(self.kbars_1m)} 根)"
            else:
                self.status_msg = f"已切換標的為 {symbol} (請點擊連線伺服器載入 K 線)"

    def start(self, user_id: str, password: str, symbol: str = "2330", lookback_days: int = 15):
        """啟動台新行情與 K 線 API 連線 (Sol_D + TechAnalysis)"""
        if not user_id or not password:
            self.status_msg = "未配置身分證字號或密碼"
            return

        # 若已經在連線狀態且帳密相同，僅變更商品時改走熱切換 (不重連 Login)
        if (self.connected or self.ta_connected) and (self.sol_d or self.ta) and self.current_user_id == user_id:
            if self.symbol != symbol:
                self.switch_symbol(symbol, user_id, password, lookback_days=lookback_days)
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
                    if sub_rc == RCode.OK:
                        self.status_msg = f"🟢 已連線 (台新 K 線與即時報價: {target_sym})"
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

        if kbars_list:
            filtered = [
                b for b in kbars_list
                if SharedBarBuilder.is_valid_session_time(b["timestamp"], session)
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

            return (
                df_1m.group_by_dynamic("timestamp", every=timeframe, closed="left", label="left")
                .agg(
                    [
                        pl.col("open").first().alias("open"),
                        pl.col("high").max().alias("high"),
                        pl.col("low").min().alias("low"),
                        pl.col("close").last().alias("close"),
                        pl.col("volume").sum().alias("volume"),
                    ]
                )
                .filter(pl.col("volume") > 0)
            )

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

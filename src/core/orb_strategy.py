# src/core/orb_strategy.py
"""
開盤區間突破（Opening Range Breakout, ORB）規則 —— 取代 AI 判斷多空方向。

規則（只做日盤）：
1. 開盤區間：日盤開盤後前 range_minutes 分鐘（預設 08:45~09:15）的最高價 / 最低價。
2. 進場：區間完成後，第一根「收盤價」突破上緣 → 做多；跌破下緣 → 做空。
   當日只認第一次突破，之後反向突破不反手；last_entry 之後的突破不進場。
3. 出場：沿用模擬倉的 ATR 停損 / 停利 / 移動鎖利；flatten_at（預設 13:30）強制平倉，不留倉過夜。

資料慣例：1 分 K 的時間標在「分鐘結束」（08:46 代表 08:45:00~08:45:59），
與期交所歷史資料（taifex_history）及台新 K 棒一致。

參考：Tsai et al. (2019) "Assessing the Profitability of Timely Opening Range Breakout on
Index Futures Markets", IEEE Access 7:32061-32071（台指期 2003~2013 實證）。
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Literal, Optional

import polars as pl

from src.core.exits import compute_exit_levels
from src.core.indicators import IndicatorEngine
from src.schema.strategy import StrategySchema

Action = Literal["BUY_LONG", "SELL_SHORT", "WAIT"]


@dataclass(frozen=True)
class OrbParams:
    session_open: time = time(8, 45)
    session_close: time = time(13, 45)
    range_minutes: int = 30
    last_entry: time = time(12, 30)
    flatten_at: time = time(13, 30)
    buffer_points: float = 0.0      # 突破需超過區間邊界的點數（0 = 收盤價超過即算）
    cost_points: float = 2.0        # 回測用：來回手續費 + 稅 + 滑價（點）
    atr_length: int = 14            # 回測用：以 5 分 K 計算 ATR
    # 日夜雙盤與多波段箱型突破擴充參數（預設維持單筆日盤以相容既有測試）
    allow_night_session: bool = False
    night_open: time = time(15, 0)
    night_close: time = time(5, 0)
    night_last_entry: time = time(4, 30)
    night_flatten_at: time = time(4, 45)
    max_entries_per_session: int = 1
    rolling_box_minutes: int = 30
    reentry_cooldown_minutes: int = 20
    buffer_atr_mult: float = 0.1
    max_range_atr_mult: float = 3.0


def multi_wave_orb_params() -> OrbParams:
    """日夜雙盤＋多波段箱型突破預設參數（有賺錢機會即進場，每盤最多 3 趟、出場後冷卻 20 分鐘）"""
    return OrbParams(
        last_entry=time(13, 15),
        allow_night_session=True,
        max_entries_per_session=3,
        rolling_box_minutes=30,
        reentry_cooldown_minutes=20,
        buffer_atr_mult=0.1,
        max_range_atr_mult=3.0,
    )


@dataclass(frozen=True)
class OrbState:
    trading_date: date
    range_ready: bool
    range_high: Optional[float]
    range_low: Optional[float]
    action: Action
    flatten: bool
    breakout_ts: Optional[datetime] = None
    breakout_price: Optional[float] = None
    reason: str = ""
    session_label: str = "日盤"


def _session_bounds(d: date, p: OrbParams) -> tuple[datetime, datetime, datetime, datetime, datetime]:
    open_dt = datetime.combine(d, p.session_open)
    return (
        open_dt,
        open_dt + timedelta(minutes=p.range_minutes),
        datetime.combine(d, p.last_entry),
        datetime.combine(d, p.flatten_at),
        datetime.combine(d, p.session_close),
    )


def _day_bars(bars_1m: pl.DataFrame, d: date, p: OrbParams, until: Optional[datetime] = None) -> pl.DataFrame:
    open_dt, _, _, _, close_dt = _session_bounds(d, p)
    end = min(close_dt, until) if until else close_dt
    return bars_1m.filter((pl.col("timestamp") > open_dt) & (pl.col("timestamp") <= end)).sort("timestamp")


def _find_breakout(day: pl.DataFrame, d: date, p: OrbParams) -> tuple[bool, Optional[float], Optional[float], int, Optional[int]]:
    """回傳 (區間是否完成, 上緣, 下緣, 突破方向, 突破那根在 day 中的索引)"""
    _, range_end, last_entry_dt, _, _ = _session_bounds(d, p)
    ts = day["timestamp"].to_list()
    rng = day.filter(pl.col("timestamp") <= range_end)
    ready = any(t >= range_end for t in ts)
    if rng.height == 0 or not ready:
        return ready, None, None, 0, None
    hi, lo = float(rng["high"].max()), float(rng["low"].min())
    closes = day["close"].to_list()
    for i, t in enumerate(ts):
        if t <= range_end or t > last_entry_dt:
            continue
        if closes[i] > hi + p.buffer_points:
            return True, hi, lo, 1, i
        if closes[i] < lo - p.buffer_points:
            return True, hi, lo, -1, i
    return True, hi, lo, 0, None


def _active_session_bounds(
    now: datetime, p: OrbParams
) -> tuple[bool, str, date, datetime, datetime, datetime, datetime, datetime]:
    """
    回傳 (是否在盤中, 盤別標籤, 所屬日曆日, 開盤時間, 區間完成時間, 最後進場時間, 強制平倉時間, 收盤時間)。
    當 allow_night_session=True 時，自動識別日盤 (08:45~13:45) 與夜盤 (15:00~隔日05:00)。
    """
    d = now.date()
    day_open, day_rng_end, day_last_entry, day_flatten, day_close = _session_bounds(d, p)
    if day_open <= now <= day_close:
        return True, "日盤", d, day_open, day_rng_end, day_last_entry, day_flatten, day_close

    if p.allow_night_session:
        hm = (now.hour, now.minute)
        if hm >= (p.night_open.hour, p.night_open.minute) or hm <= (p.night_close.hour, p.night_close.minute):
            s_date = d if now.hour >= p.night_open.hour else (d - timedelta(days=1))
            if s_date.weekday() == 5:
                s_date -= timedelta(days=1)
            elif s_date.weekday() == 6:
                s_date -= timedelta(days=2)
            n_open = datetime.combine(s_date, p.night_open)
            n_rng_end = n_open + timedelta(minutes=p.range_minutes)
            next_d = s_date + timedelta(days=1)
            n_last_entry = datetime.combine(next_d, p.night_last_entry)
            n_flatten = datetime.combine(next_d, p.night_flatten_at)
            n_close = datetime.combine(next_d, p.night_close)
            return True, "夜盤", s_date, n_open, n_rng_end, n_last_entry, n_flatten, n_close

    return False, "休市", d, day_open, day_rng_end, day_last_entry, day_flatten, day_close


def compute_orb_state(bars_1m: pl.DataFrame, now: datetime, params: OrbParams = OrbParams()) -> OrbState:
    """以「現在時間」之前已完成的 1 分 K 計算當日/當盤 ORB 與波段箱型突破狀態（不使用未來資料）。"""
    p = params
    if not p.allow_night_session and p.max_entries_per_session <= 1:
        d = now.date()
        open_dt, _, _, flatten_dt, close_dt = _session_bounds(d, p)
        in_day_session = open_dt <= now <= close_dt
        flatten = (not in_day_session) or now >= flatten_dt

        day = _day_bars(bars_1m, d, p, until=now)
        ready, hi, lo, direction, idx = _find_breakout(day, d, p)

        if not ready:
            return OrbState(d, False, hi, lo, "WAIT", flatten, reason="開盤區間尚未完成")
        if direction == 0:
            return OrbState(d, True, hi, lo, "WAIT", flatten, reason="價格仍在開盤區間內")
        row = day.row(idx, named=True)
        action: Action = "BUY_LONG" if direction == 1 else "SELL_SHORT"
        if flatten:
            action = "WAIT"
        return OrbState(
            d, True, hi, lo, action, flatten,
            breakout_ts=row["timestamp"], breakout_price=float(row["close"]),
            reason=("收盤突破開盤區間上緣" if direction == 1 else "收盤跌破開盤區間下緣"),
        )

    in_sess, sess_label, s_date, open_dt, range_end, last_entry_dt, flatten_dt, close_dt = _active_session_bounds(now, p)
    flatten = (not in_sess) or now >= flatten_dt
    if bars_1m is None or bars_1m.height == 0:
        return OrbState(s_date, False, None, None, "WAIT", flatten, reason="尚無 K 棒資料", session_label=sess_label)

    end_dt = min(close_dt, now) if in_sess else close_dt
    sess_df = bars_1m.filter((pl.col("timestamp") > open_dt) & (pl.col("timestamp") <= end_dt)).sort("timestamp")
    ts_list = sess_df["timestamp"].to_list()
    rng_df = sess_df.filter(pl.col("timestamp") <= range_end)
    ready = any(t >= range_end for t in ts_list)
    if rng_df.height == 0 or not ready:
        return OrbState(s_date, False, None, None, "WAIT", flatten, reason=f"{sess_label}開盤區間形成中", session_label=sess_label)

    orb_hi, orb_lo = float(rng_df["high"].max()), float(rng_df["low"].min())
    rows = sess_df.rows(named=True)
    n = len(rows)
    lb = max(10, p.rolling_box_minutes)

    # 計算最近完成的波段箱型（排除最後 1 根正在判斷突破的 K 棒）
    if n > lb + 1:
        box_slice = rows[max(0, n - 1 - lb):n - 1]
        box_hi = max(float(r["high"]) for r in box_slice)
        box_lo = min(float(r["low"]) for r in box_slice)
    else:
        box_hi, box_lo = orb_hi, orb_lo

    atr = _atr_before(bars_1m, rows[-1]["timestamp"], p.atr_length)
    if atr is None or math.isnan(atr) or atr <= 0:
        atr = max(20.0, (orb_hi - orb_lo) * 0.5)
    buf = p.buffer_points + atr * p.buffer_atr_mult

    # 掃描本盤最新一波有效突破訊號（支援初波開盤區間突破 ＋ 後續 30 分鐘箱型再突破）
    last_dir = 0
    last_brk_ts: Optional[datetime] = None
    last_brk_p: Optional[float] = None
    last_hi, last_lo = orb_hi, orb_lo
    last_reason = f"價格於{sess_label}區間整理中，等待突破訊號"

    # 若開盤區間過大（> max_range_atr_mult * ATR），改看滾動箱型突破，避免開盤暴衝後追高殺低
    orb_too_wide = p.max_range_atr_mult > 0 and (orb_hi - orb_lo) > atr * p.max_range_atr_mult

    for i, r in enumerate(rows):
        t = r["timestamp"]
        if t <= range_end or t > last_entry_dt:
            continue
        sub_lb = rows[max(0, i - lb):i]
        if len(sub_lb) >= 10:
            cur_box_hi = max(float(b["high"]) for b in sub_lb)
            cur_box_lo = min(float(b["low"]) for b in sub_lb)
        else:
            cur_box_hi, cur_box_lo = orb_hi, orb_lo

        # 第一波優先看開盤區間（若未過寬），後續波段看近 30 分鐘箱型高低點
        use_box = (last_brk_ts is not None and (t - last_brk_ts).total_seconds() >= p.reentry_cooldown_minutes * 60) or orb_too_wide
        hi_lvl = cur_box_hi if use_box else orb_hi
        lo_lvl = cur_box_lo if use_box else orb_lo
        tag = f"{sess_label}近{lb}分箱型" if use_box else f"{sess_label}開盤區間"

        c = float(r["close"])
        if c > hi_lvl + buf:
            last_dir = 1
            last_brk_ts = t
            last_brk_p = c
            last_hi, last_lo = hi_lvl, lo_lvl
            last_reason = f"收盤突破{tag}上緣 ({hi_lvl:,.0f})"
        elif c < lo_lvl - buf:
            last_dir = -1
            last_brk_ts = t
            last_brk_p = c
            last_hi, last_lo = hi_lvl, lo_lvl
            last_reason = f"收盤跌破{tag}下緣 ({lo_lvl:,.0f})"

    # 若最後一波突破後，最新收盤價已退回目前箱型內部且超過 15 分鐘，視為該波段結束、轉為等待下一波箱型突破
    latest_c = float(rows[-1]["close"])
    if last_dir != 0 and last_brk_ts is not None:
        mins_since_brk = (rows[-1]["timestamp"] - last_brk_ts).total_seconds() / 60.0
        if mins_since_brk > 15 and box_lo <= latest_c <= box_hi:
            if ( last_dir == 1 and latest_c < last_brk_p ) or ( last_dir == -1 and latest_c > last_brk_p ):
                last_dir = 0
                last_hi, last_lo = box_hi, box_lo
                last_reason = f"前波突破已收斂，目前監控{sess_label}近{lb}分箱型 ({box_lo:,.0f}~{box_hi:,.0f}) 等待下一波突破"

    if last_dir == 0:
        disp_hi = box_hi if orb_too_wide else last_hi
        disp_lo = box_lo if orb_too_wide else last_lo
        return OrbState(s_date, True, disp_hi, disp_lo, "WAIT", flatten, reason=last_reason, session_label=sess_label)

    act: Action = "BUY_LONG" if last_dir == 1 else "SELL_SHORT"
    if flatten or now > last_entry_dt:
        act = "WAIT"
    return OrbState(
        s_date, True, last_hi, last_lo, act, flatten,
        breakout_ts=last_brk_ts, breakout_price=last_brk_p,
        reason=last_reason, session_label=sess_label,
    )


def _atr_before(bars_1m: pl.DataFrame, before: datetime, length: int) -> float:
    """以 before 之前已完成的 5 分 K 計算 ATR（不含未來資料）。"""
    hist = bars_1m.filter(pl.col("timestamp") <= before).with_columns(
        (pl.col("timestamp") - pl.duration(minutes=1)).alias("start")
    )
    m5 = hist.group_by_dynamic("start", every="5m", label="left", closed="left").agg(
        pl.col("open").first(), pl.col("high").max(), pl.col("low").min(), pl.col("close").last()
    )
    if m5.height <= length:
        return float("nan")
    val = m5.select(IndicatorEngine.calculate_atr(m5, length)).to_series()[-1]
    return float(val) if val is not None else float("nan")


def backtest_orb(
    bars_1m: pl.DataFrame,
    params: OrbParams = OrbParams(),
    strategy: Optional[StrategySchema] = None,
    atr_value: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    回測 ORB / 全時段多波段突破：
    - 當 params 為預設單筆日盤模式時，逐日回測日盤第一次突破。
    - 當 params 啟用日夜雙盤或多波段模式 (allow_night_session=True 或 max_entries_per_session > 1) 時，
      支援日盤＋夜盤雙盤開盤區間與滾動箱型突破多次進場（出場後經冷卻時間可再次捕捉下一波突破）。
    """
    if strategy is None:
        from src.core.paper_trader import default_ai_strategy  # 延遲匯入避免循環相依

        strategy = default_ai_strategy("TMF")
    p = params
    df = bars_1m.sort("timestamp")
    trades: List[Dict[str, Any]] = []

    if not p.allow_night_session and p.max_entries_per_session <= 1:
        dates = sorted({t.date() for t in df["timestamp"].to_list()})
        for d in dates:
            day = _day_bars(df, d, p)
            ready, hi, lo, direction, idx = _find_breakout(day, d, p)
            if not ready or direction == 0 or idx is None or idx + 1 >= day.height:
                continue
            rows = day.rows(named=True)
            entry_row = rows[idx + 1]
            entry = float(entry_row["open"])
            atr = atr_value if atr_value is not None else _atr_before(df, rows[idx]["timestamp"], p.atr_length)
            if atr is None or math.isnan(atr) or atr <= 0:
                continue
            atr_map = {ln: atr for ln in (strategy.exit.stop_loss.atr_length if strategy.exit.stop_loss else 14,
                                          strategy.exit.trailing_stop.atr_length if strategy.exit.trailing_stop else 14,
                                          strategy.exit.take_profit.atr_length if strategy.exit.take_profit else 14)}
            flatten_dt = datetime.combine(d, p.flatten_at)
            hi_since = lo_since = entry
            exit_price, exit_reason, exit_time = None, "", None

            for r in rows[idx + 1:]:
                stop, tp, stop_reason = compute_exit_levels(strategy, direction, entry, hi_since, lo_since, atr_map, atr_map)
                bar_open, bar_high, bar_low = float(r["open"]), float(r["high"]), float(r["low"])
                if stop is not None and ((direction == 1 and bar_low <= stop) or (direction == -1 and bar_high >= stop)):
                    exit_price = min(stop, bar_open) if direction == 1 else max(stop, bar_open)
                    exit_reason, exit_time = stop_reason, r["timestamp"]
                    break
                if tp is not None and ((direction == 1 and bar_high >= tp) or (direction == -1 and bar_low <= tp)):
                    exit_price = max(tp, bar_open) if direction == 1 else min(tp, bar_open)
                    exit_reason, exit_time = "take_profit", r["timestamp"]
                    break
                if r["timestamp"] >= flatten_dt:
                    exit_price, exit_reason, exit_time = float(r["close"]), "time_exit", r["timestamp"]
                    break
                hi_since, lo_since = max(hi_since, bar_high), min(lo_since, bar_low)
            if exit_price is None:  # 資料在收盤前中斷：以最後一根收盤平倉
                exit_price, exit_reason, exit_time = float(rows[-1]["close"]), "end_of_data", rows[-1]["timestamp"]

            trades.append({
                "date": d,
                "direction": direction,
                "range_high": hi,
                "range_low": lo,
                "entry_time": entry_row["timestamp"],
                "entry_price": entry,
                "exit_time": exit_time,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "atr": atr,
                "pnl_points": (exit_price - entry) * direction - p.cost_points,
            })
        return trades

    # --- 日夜雙盤＋多波段箱型突破快速回測 ---
    all_rows = df.rows(named=True)
    if not all_rows:
        return []

    # 預先計算 5m ATR 索引加速查詢
    m5_ts: List[datetime] = []
    m5_atr: List[Optional[float]] = []
    if atr_value is None:
        df_start = df.with_columns((pl.col("timestamp") - pl.duration(minutes=1)).alias("start"))
        m5 = df_start.group_by_dynamic("start", every="5m", label="left", closed="left").agg(
            pl.col("timestamp").last().alias("ts_end"),
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
        )
        if m5.height > p.atr_length:
            m5 = m5.with_columns(IndicatorEngine.calculate_atr(m5, p.atr_length).alias("atr_col"))
            m5_ts = m5["ts_end"].to_list()
            m5_atr = m5["atr_col"].to_list()

    def _lookup_atr(dt_val: datetime) -> float:
        if atr_value is not None:
            return atr_value
        idx_a = bisect.bisect_right(m5_ts, dt_val) - 1
        if idx_a >= p.atr_length and m5_atr[idx_a] is not None:
            return float(m5_atr[idx_a])
        return float("nan")

    sessions: List[tuple[tuple[str, date], List[Dict[str, Any]]]] = []
    curr_sess: List[Dict[str, Any]] = []
    curr_kind: Optional[tuple[str, date]] = None
    for r in all_rows:
        t = r["timestamp"]
        hm = (t.hour, t.minute)
        if (p.session_open.hour, p.session_open.minute) < hm <= (p.session_close.hour, p.session_close.minute):
            kind = ("DAY", t.date())
        elif p.allow_night_session and (
            hm > (p.night_open.hour, p.night_open.minute) or hm <= (p.night_close.hour, p.night_close.minute)
        ):
            s_d = t.date() if t.hour >= p.night_open.hour else (t.date() - timedelta(days=1))
            kind = ("NIGHT", s_d)
        else:
            continue
        if kind != curr_kind:
            if curr_sess and curr_kind is not None:
                sessions.append((curr_kind, curr_sess))
            curr_sess = [r]
            curr_kind = kind
        else:
            curr_sess.append(r)
    if curr_sess and curr_kind is not None:
        sessions.append((curr_kind, curr_sess))

    atr_lengths = {
        strategy.exit.stop_loss.atr_length if strategy.exit.stop_loss else 14,
        strategy.exit.trailing_stop.atr_length if strategy.exit.trailing_stop else 14,
        strategy.exit.take_profit.atr_length if strategy.exit.take_profit else 14,
    }
    lb = max(10, p.rolling_box_minutes)

    for (s_type, s_date), s_rows in sessions:
        if len(s_rows) < p.range_minutes + 5:
            continue
        t0 = s_rows[0]["timestamp"] - timedelta(minutes=1)
        range_end = t0 + timedelta(minutes=p.range_minutes)
        if s_type == "DAY":
            last_entry = datetime.combine(s_date, p.last_entry)
            flatten_dt = datetime.combine(s_date, p.flatten_at)
        else:
            next_d = s_rows[-1]["timestamp"].date()
            last_entry = datetime.combine(next_d, p.night_last_entry)
            flatten_dt = datetime.combine(next_d, p.night_flatten_at)

        rng_bars = [r for r in s_rows if r["timestamp"] <= range_end]
        if not rng_bars:
            continue
        orb_hi = max(float(r["high"]) for r in rng_bars)
        orb_lo = min(float(r["low"]) for r in rng_bars)
        orb_w = orb_hi - orb_lo

        i = 0
        n = len(s_rows)
        sess_trades = 0
        last_exit_ts: Optional[datetime] = None
        while i < n - 1:
            r = s_rows[i]
            ts = r["timestamp"]
            if ts <= range_end or ts > last_entry or sess_trades >= p.max_entries_per_session:
                i += 1
                continue
            if last_exit_ts is not None and (ts - last_exit_ts).total_seconds() < p.reentry_cooldown_minutes * 60:
                i += 1
                continue

            lb_bars = s_rows[max(0, i - lb):i]
            if len(lb_bars) < 15:
                i += 1
                continue
            box_hi = max(float(b["high"]) for b in lb_bars)
            box_lo = min(float(b["low"]) for b in lb_bars)
            atr = _lookup_atr(ts)
            if atr is None or math.isnan(atr) or atr <= 0:
                i += 1
                continue
            if p.max_range_atr_mult > 0 and orb_w > atr * p.max_range_atr_mult:
                i += 1
                continue

            hi_lvl = orb_hi if sess_trades == 0 else box_hi
            lo_lvl = orb_lo if sess_trades == 0 else box_lo
            buf = p.buffer_points + atr * p.buffer_atr_mult
            c = float(r["close"])
            direction = 1 if c > hi_lvl + buf else (-1 if c < lo_lvl - buf else 0)
            if direction == 0:
                i += 1
                continue

            entry_row = s_rows[i + 1]
            entry = float(entry_row["open"])
            atr_map = {ln: atr for ln in atr_lengths}
            hi_since = lo_since = entry
            exit_price, exit_reason, exit_time = None, "", None
            exit_i = i + 1

            for j in range(i + 1, n):
                rj = s_rows[j]
                stop, tp, stop_reason = compute_exit_levels(strategy, direction, entry, hi_since, lo_since, atr_map, atr_map)
                bo, bh, bl = float(rj["open"]), float(rj["high"]), float(rj["low"])
                if stop is not None and ((direction == 1 and bl <= stop) or (direction == -1 and bh >= stop)):
                    exit_price = min(stop, bo) if direction == 1 else max(stop, bo)
                    exit_reason, exit_time = stop_reason, rj["timestamp"]
                    exit_i = j
                    break
                if tp is not None and ((direction == 1 and bh >= tp) or (direction == -1 and bl <= tp)):
                    exit_price = max(tp, bo) if direction == 1 else min(tp, bo)
                    exit_reason, exit_time = "take_profit", rj["timestamp"]
                    exit_i = j
                    break
                if rj["timestamp"] >= flatten_dt:
                    exit_price, exit_reason, exit_time = float(rj["close"]), "time_exit", rj["timestamp"]
                    exit_i = j
                    break
                hi_since, lo_since = max(hi_since, bh), min(lo_since, bl)
            if exit_price is None:
                exit_price, exit_reason, exit_time = float(s_rows[-1]["close"]), "end_of_data", s_rows[-1]["timestamp"]
                exit_i = n - 1

            trades.append({
                "date": s_date,
                "session": s_type,
                "direction": direction,
                "range_high": hi_lvl,
                "range_low": lo_lvl,
                "entry_time": entry_row["timestamp"],
                "entry_price": entry,
                "exit_time": exit_time,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "atr": atr,
                "pnl_points": (exit_price - entry) * direction - p.cost_points,
            })
            sess_trades += 1
            last_exit_ts = exit_time
            i = exit_i + 1

    return trades

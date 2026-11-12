# tests/test_key_levels.py
"""客觀關鍵價位：前一日高低收、當日 VWAP、以平均日振幅推估的預期波動區間（1 分 K 時間標在分鐘結束）"""
from datetime import date, datetime, time, timedelta

import polars as pl
import pytest

from src.core.key_levels import compute_key_levels, expected_range_hit_rate

OPEN_T, CLOSE_T = time(8, 45), time(13, 45)


def _day(d: date, o: float, h: float, l: float, c: float, vol: int = 10, n: int = 300) -> list[dict]:
    """產生一天日盤 1 分 K：第一根開在 o，中間觸及 h 與 l，最後收在 c"""
    rows = []
    start = datetime.combine(d, OPEN_T) + timedelta(minutes=1)
    for i in range(n):
        p = o if i == 0 else (h if i == n // 3 else (l if i == 2 * n // 3 else c))
        rows.append({"timestamp": start + timedelta(minutes=i), "open": p, "high": p, "low": p, "close": p, "volume": vol})
    return rows


def _df(rows):
    return pl.DataFrame(rows)


def test_previous_day_high_low_close_uses_day_session_only():
    d1, d2 = date(2026, 11, 9), date(2026, 11, 10)
    night = [{"timestamp": datetime(2026, 11, 9, 20, 0), "open": 99999.0, "high": 99999.0, "low": 1.0, "close": 99999.0, "volume": 1}]
    rows = _day(d1, 40000, 40300, 39800, 40100) + night + _day(d2, 40100, 40200, 40000, 40150, n=30)
    kl = compute_key_levels(_df(rows), now=datetime(2026, 11, 10, 9, 20), open_t=OPEN_T, close_t=CLOSE_T)
    assert (kl.prev_high, kl.prev_low, kl.prev_close) == (40300, 39800, 40100)
    assert kl.today_open == 40100


def test_vwap_is_volume_weighted_and_excludes_future_bars():
    d = date(2026, 11, 10)
    t = datetime.combine(d, OPEN_T)
    rows = [
        {"timestamp": t + timedelta(minutes=1), "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1},
        {"timestamp": t + timedelta(minutes=2), "open": 200.0, "high": 200.0, "low": 200.0, "close": 200.0, "volume": 3},
        {"timestamp": t + timedelta(minutes=3), "open": 900.0, "high": 900.0, "low": 900.0, "close": 900.0, "volume": 100},
    ]
    kl = compute_key_levels(_df(rows), now=t + timedelta(minutes=2), open_t=OPEN_T, close_t=CLOSE_T)
    assert kl.vwap == pytest.approx((100 * 1 + 200 * 3) / 4)


def test_expected_range_from_average_daily_range():
    rows = []
    d0 = date(2026, 10, 1)
    for i in range(12):  # 每天振幅 400 點
        rows += _day(d0 + timedelta(days=i), 40000, 40200, 39800, 40000)
    today = d0 + timedelta(days=12)
    rows += _day(today, 40050, 40060, 40040, 40050, n=10)
    kl = compute_key_levels(_df(rows), now=datetime.combine(today, time(9, 0)), open_t=OPEN_T, close_t=CLOSE_T,
                            adr_days=10, adr_mult=0.75)
    assert kl.adr == pytest.approx(400)
    assert kl.exp_upper == pytest.approx(40050 + 300)
    assert kl.exp_lower == pytest.approx(40050 - 300)


def test_expected_range_unavailable_with_too_few_days():
    rows = _day(date(2026, 11, 9), 40000, 40200, 39800, 40000) + _day(date(2026, 11, 10), 40000, 40010, 39990, 40000, n=10)
    kl = compute_key_levels(_df(rows), now=datetime(2026, 11, 10, 9, 0), open_t=OPEN_T, close_t=CLOSE_T, adr_days=10)
    assert kl.exp_upper is None and kl.exp_lower is None


def test_hit_rate_counts_days_fully_inside_range():
    rows = []
    d0 = date(2026, 10, 1)
    for i in range(10):
        rows += _day(d0 + timedelta(days=i), 40000, 40200, 39800, 40000)          # 振幅 400
    rows += _day(d0 + timedelta(days=10), 40000, 40100, 39900, 40000)            # 區間 ±300 內 → 命中
    rows += _day(d0 + timedelta(days=11), 40000, 40500, 39900, 40000)            # 上破 → 未命中
    rate, n = expected_range_hit_rate(_df(rows), OPEN_T, CLOSE_T, adr_days=10, adr_mult=0.75)
    assert n == 2 and rate == pytest.approx(0.5)

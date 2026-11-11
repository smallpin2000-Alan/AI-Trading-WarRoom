# scripts/backtest_orb.py
"""以期交所歷史 1 分 K 回測開盤區間突破（ORB）規則。執行：python -X utf8 scripts/backtest_orb.py [--range 30]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.orb_strategy import OrbParams, backtest_orb  # noqa: E402

POINT_VALUE = 10.0  # 微台每點 10 元


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--range", type=int, default=30, help="開盤區間分鐘數")
    args = ap.parse_args()
    bars = pl.read_parquet(ROOT / "data/history/TMF_1m.parquet")
    trades = pl.DataFrame(backtest_orb(bars, OrbParams(range_minutes=args.range)))
    if trades.height == 0:
        print("沒有任何交易")
        return
    r = trades["pnl_points"]
    wins, losses = r.filter(r > 0), r.filter(r <= 0)
    pf = wins.sum() / abs(losses.sum()) if losses.len() and losses.sum() else float("inf")
    print(f"開盤區間 {args.range} 分鐘：{trades.height} 筆（{trades['date'].min()} ~ {trades['date'].max()}）")
    print(f"勝率 {(r > 0).mean() * 100:.1f}%  平均 {r.mean():+.1f} 點  合計 {r.sum():+.0f} 點（{r.sum() * POINT_VALUE:+,.0f} 元/口）  獲利因子 {pf:.2f}")
    print(f"最大單筆虧損 {r.min():.0f} 點  最大單筆獲利 {r.max():.0f} 點")
    print("出場原因：", dict(trades.group_by("exit_reason").len().iter_rows()))
    half = trades.height // 2
    for name, part in (("前半", trades.head(half)), ("後半", trades.tail(trades.height - half))):
        pr = part["pnl_points"]
        print(f"  {name} {part.height} 筆：勝率 {(pr > 0).mean() * 100:.0f}%  合計 {pr.sum():+.0f} 點")


if __name__ == "__main__":
    main()

# scripts/evaluate_signals.py
"""
AI 訊號方向評估（不含停損停利）：
以期交所歷史 1 分 K 合成 5 分 K，逐根「只用到該根收盤為止的資料」呼叫 FuturesAIAdvisor.analyze_market，
對每個 BUY_LONG / SELL_SHORT 訊號，以下一根開盤價進場，量測 15 / 30 / 60 分鐘後的方向與點數報酬（扣成本）。

執行：python -X utf8 scripts/evaluate_signals.py [--window 300] [--limit 0]
輸出：data/history/signal_eval_TMF.parquet（每個訊號一列）與終端摘要。
不寫入任何正式資料（record_prediction 已被停用）。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from unittest import mock

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ai.advisor import FuturesAIAdvisor  # noqa: E402
from src.ai.cloud_intel import CloudIntelligenceHub  # noqa: E402

HISTORY_1M = ROOT / "data/history/TMF_1m.parquet"
OUT_PATH = ROOT / "data/history/signal_eval_TMF.parquet"
HORIZONS = (3, 6, 12)       # 5 分 K 根數 → 15 / 30 / 60 分鐘
ROUND_TRIP_COST_PTS = 2.0   # 微台來回手續費 + 稅 + 1 tick 滑價的保守估計（點）
MIN_BARS = 60               # 暖機根數


def load_5m(path: Path) -> pl.DataFrame:
    """1 分 K（時間標在分鐘結束）→ 5 分 K（時間標在 K 棒開始，與戰情室一致）。"""
    m1 = pl.read_parquet(path).sort("timestamp").with_columns(
        (pl.col("timestamp") - pl.duration(minutes=1)).alias("start")
    )
    return (
        m1.group_by_dynamic("start", every="5m", label="left", closed="left")
        .agg(
            pl.col("open").first(), pl.col("high").max(), pl.col("low").min(),
            pl.col("close").last(), pl.col("volume").sum(),
        )
        .rename({"start": "timestamp"})
    )


def evaluate(df5: pl.DataFrame, window: int, limit: int) -> pl.DataFrame:
    opens = df5["open"].to_list()
    closes = df5["close"].to_list()
    stamps = df5["timestamp"].to_list()
    last_i = df5.height - 1 - max(HORIZONS)
    idxs = list(range(MIN_BARS, last_i))
    if limit:
        idxs = idxs[:limit]

    rows = []
    t0 = time.time()
    with mock.patch.object(CloudIntelligenceHub, "record_prediction", lambda *a, **k: None):
        for n, i in enumerate(idxs, 1):
            hist = df5.slice(max(0, i + 1 - window), min(window, i + 1))
            adv = FuturesAIAdvisor.analyze_market(hist, symbol="TMFR1", timeframe="5m", fast_local_only=True)
            if adv.action != "WAIT":
                d = 1 if adv.action == "BUY_LONG" else -1
                entry = opens[i + 1]
                rec = {
                    "timestamp": stamps[i], "action": adv.action, "direction": d,
                    "confidence": float(adv.confidence), "regime": adv.regime_label,
                    "entry": entry, "atr": float(adv.atr_val),
                }
                for h in HORIZONS:
                    rec[f"ret_{h * 5}m"] = (closes[i + h] - entry) * d - ROUND_TRIP_COST_PTS
                rows.append(rec)
            if n % 500 == 0:
                print(f"  進度 {n}/{len(idxs)}，已 {time.time() - t0:.0f} 秒，訊號 {len(rows)} 筆", flush=True)
    return pl.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=300, help="每次分析只給最近幾根 5 分 K（與戰情室實際量級相近）")
    ap.add_argument("--limit", type=int, default=0, help="只評估前 N 根（測速用）")
    args = ap.parse_args()

    df5 = load_5m(HISTORY_1M)
    print(f"5 分 K：{df5.height} 根（{df5['timestamp'].min()} ~ {df5['timestamp'].max()}）")
    t0 = time.time()
    sig = evaluate(df5, args.window, args.limit)
    print(f"耗時 {time.time() - t0:.0f} 秒，訊號 {sig.height} 筆")
    if sig.height and not args.limit:
        sig.write_parquet(OUT_PATH)
        print(f"已輸出 {OUT_PATH}")


if __name__ == "__main__":
    main()

# tests/test_copilot_and_store.py
from datetime import datetime, timedelta
import json
from pathlib import Path
import polars as pl

from src.data.kbar_store import KBarStore
from src.ai.strategy_copilot import AIStrategyCopilot
from src.core.indicators import IndicatorEngine
from src.schema.strategy import StrategySchema
from src.core.engine import EventDrivenEngine, ContractSpec


def test_kbar_store_save_and_merge(tmp_path):
    """測試台新 K 線本地 Parquet 儲存與去重合併"""
    store = KBarStore(data_dir=tmp_path)
    t1 = datetime(2026, 10, 22, 9, 1)
    t2 = datetime(2026, 10, 22, 9, 2)
    t3 = datetime(2026, 10, 22, 9, 3)

    batch1 = {
        t1: {"timestamp": t1, "open": 2380.0, "high": 2385.0, "low": 2375.0, "close": 2380.0, "volume": 100},
        t2: {"timestamp": t2, "open": 2380.0, "high": 2390.0, "low": 2380.0, "close": 2385.0, "volume": 150},
    }
    cnt1 = store.save_kbars("2330", batch1)
    assert cnt1 == 2

    # 第二批更新 t2 並新增 t3
    batch2 = {
        t2: {"timestamp": t2, "open": 2380.0, "high": 2395.0, "low": 2380.0, "close": 2390.0, "volume": 200},
        t3: {"timestamp": t3, "open": 2390.0, "high": 2400.0, "low": 2385.0, "close": 2395.0, "volume": 120},
    }
    cnt2 = store.save_kbars("2330", batch2)
    assert cnt2 == 3

    loaded = store.load_kbars_dict("2330")
    assert len(loaded) == 3
    assert loaded[t2]["close"] == 2390.0
    assert loaded[t2]["volume"] == 200


def test_ai_copilot_natural_language_and_safety_stop_loss():
    """測試 AI 中文自然語言解析、停損強制補齊提醒與邏輯樹產生"""
    # 1. 含完整停損與 RSI 濾網的口語
    res1 = AIStrategyCopilot.parse_natural_language(
        "當 5EMA 向上突破 20EMA 且 RSI(14) 大於 55 時做多，跌破時做空，停損 40 點，停利 100 點，移動停損 50 點"
    )
    assert res1.strategy.indicators["fast_ma"].type == "EMA"
    assert res1.strategy.indicators["fast_ma"].length == 5
    assert "rsi" in res1.strategy.indicators
    assert res1.strategy.exit.stop_loss.value == 40.0
    assert res1.strategy.exit.take_profit.value == 100.0
    assert res1.strategy.exit.trailing_stop.value == 50.0
    assert len(res1.logic_tree_lines) >= 6

    # 2. 未提及停損的口語 -> 自動補齊安全停損並發出安全提醒
    res2 = AIStrategyCopilot.parse_natural_language("突破 20 根最高價做多，跌破最低價做空")
    assert res2.strategy.exit.stop_loss is not None
    assert any("交易安全提醒" in w for w in res2.clarifications)


def test_all_12_indicators_and_strategy_templates():
    """測試全部 12 種技術指標計算與 strategies/ 目錄下所有範本合法性"""
    t0 = datetime(2026, 10, 22, 9, 0)
    n = 40
    df = pl.DataFrame({
        "timestamp": [t0 + timedelta(minutes=i) for i in range(n)],
        "open": [20000.0 + i * 2 for i in range(n)],
        "high": [20010.0 + i * 2 for i in range(n)],
        "low": [19990.0 + i * 2 for i in range(n)],
        "close": [20005.0 + i * 2 for i in range(n)],
        "volume": [100 + i for i in range(n)],
    })

    specs = {
        "sma": {"type": "SMA", "length": 5},
        "ema": {"type": "EMA", "length": 5},
        "rsi": {"type": "RSI", "length": 6},
        "atr": {"type": "ATR", "length": 5},
        "k": {"type": "KD_K", "length": 9},
        "d": {"type": "KD_D", "length": 9},
        "macd": {"type": "MACD", "length": 12},
        "adx": {"type": "ADX", "length": 7},
        "bbu": {"type": "BB_UPPER", "length": 10},
        "bbl": {"type": "BB_LOWER", "length": 10},
        "hh": {"type": "HIGHEST", "length": 10, "source": "high"},
        "ll": {"type": "LOWEST", "length": 10, "source": "low"},
    }
    out = IndicatorEngine.compute_indicators(df, specs)
    for col_name in specs.keys():
        assert col_name in out.columns

    spec_contract = ContractSpec("TMF", "微型臺指", "TAIFEX", 10.0, 1.0, 0.00002, 12.0)
    for p in Path("strategies").glob("*.json"):
        data = json.loads(p.read_text(encoding="utf-8"))
        strat = StrategySchema.model_validate(data)
        eng = EventDrivenEngine(strat, spec_contract, 100_000.0)
        rep = eng.run_backtest(df)
        assert len(rep.equity_curve) == n

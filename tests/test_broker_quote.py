# tests/test_broker_quote.py
from datetime import datetime
from types import SimpleNamespace
from src.broker.taishin_quote import TaishinQuoteFeed
from src.core.bar_builder import TickData


def test_taishin_quote_feed_time_parsing():
    """測試台新行情模組的時間解析與 TickData 轉換"""
    feed = TaishinQuoteFeed()

    # 測試冒號格式 HH:MM:SS
    dt1 = feed._parse_match_time("09:15:30")
    assert dt1.hour == 9
    assert dt1.minute == 15
    assert dt1.second == 30

    # 測試純數字格式 HHMMSS
    dt2 = feed._parse_match_time("143000")
    assert dt2.hour == 14
    assert dt2.minute == 30
    assert dt2.second == 0


def test_taishin_quote_feed_aggregation():
    """測試將台新推送的 Tick 即時聚合成 K 棒"""
    feed = TaishinQuoteFeed()

    ticks = [
        TickData(datetime(2026, 10, 22, 9, 0, 1), 20100.0, 10),
        TickData(datetime(2026, 10, 22, 9, 0, 30), 20120.0, 5),
        TickData(datetime(2026, 10, 22, 9, 0, 50), 20090.0, 8),
        TickData(datetime(2026, 10, 22, 9, 1, 10), 20150.0, 12),
    ]
    with feed.lock:
        feed.raw_ticks = list(ticks)
        feed.latest_tick = ticks[-1]

    bars = feed.get_bars(timeframe="1m", session="day_only")
    assert bars.height == 2
    assert bars["open"][0] == 20100.0
    assert bars["high"][0] == 20120.0
    assert bars["low"][0] == 20090.0
    assert bars["close"][0] == 20090.0
    assert bars["volume"][0] == 23


def test_taishin_quote_feed_on_match_and_snapshot_callbacks():
    """測試台新 Sol_D API OnMatch 與 OnUpdateLastSnapshot 事件回呼及 K 棒合成"""
    feed = TaishinQuoteFeed()
    received = []
    feed.register_on_tick(lambda tk: received.append(tk))

    snap_obj = SimpleNamespace(
        TickData=SimpleNamespace(MatchPrice=2380.0, MatchQty=5, MatchTime="09:00:05")
    )
    tick_obj = SimpleNamespace(MatchPrice=2385.0, MatchQty=10, MatchTime="09:00:45")

    feed._on_snapshot(snap_obj)
    feed._on_match_tick(tick_obj)

    assert len(received) == 2
    assert feed.get_latest_tick().price == 2385.0
    assert len(feed.get_recent_ticks()) == 2

    bars = feed.get_bars(timeframe="1m", session="day_only")
    assert bars.height == 1
    assert bars["open"][0] == 2380.0
    assert bars["close"][0] == 2385.0
    assert bars["volume"][0] == 15


def test_taishin_tech_analysis_kbar_callbacks():
    """測試台新 TechAnalysis API 1m K 棒接收與週期轉換"""
    feed = TaishinQuoteFeed()

    raw_results = [
        SimpleNamespace(
            KBar=SimpleNamespace(
                Date="20261022",
                TimeSn_Dply=901,
                OPrice=2380.0,
                HPrice=2390.0,
                LPrice=2375.0,
                CPrice=2385.0,
                Quantity=120,
                Volume=120,
            )
        ),
        SimpleNamespace(
            KBar=SimpleNamespace(
                Date="20261022",
                TimeSn_Dply=902,
                OPrice=2385.0,
                HPrice=2395.0,
                LPrice=2380.0,
                CPrice=2390.0,
                Quantity=80,
                Volume=200,
            )
        ),
    ]
    feed._on_ta_rcv_done(0, raw_results)

    bars_1m = feed.get_bars(timeframe="1m", session="day_only")
    assert bars_1m.height == 2
    assert bars_1m["open"][0] == 2380.0
    assert bars_1m["close"][1] == 2390.0
    assert bars_1m["volume"][0] == 120
    assert bars_1m["volume"][1] == 80

    bars_5m = feed.get_bars(timeframe="5m", session="day_only")
    assert bars_5m.height == 1
    assert bars_5m["open"][0] == 2380.0
    assert bars_5m["high"][0] == 2395.0
    assert bars_5m["low"][0] == 2375.0
    assert bars_5m["close"][0] == 2390.0
    assert bars_5m["volume"][0] == 200


def test_taishin_switch_symbol_clears_old_data_and_loads_new(tmp_path):
    """測試更換訂閱商品時，會立即清空前商品的 K 線與成交價，並載入新商品快取"""
    feed = TaishinQuoteFeed(store_dir=tmp_path, auto_load_store=False)
    feed.symbol = "2330"

    # 先灌入 2330 的 K 棒 (價格 1050)
    raw_2330 = [
        SimpleNamespace(
            KBar=SimpleNamespace(
                Date="20261022", TimeSn_Dply=901,
                OPrice=1050.0, HPrice=1055.0, LPrice=1045.0, CPrice=1050.0, Quantity=500
            )
        )
    ]
    feed._on_ta_rcv_done(0, raw_2330)
    assert feed.get_latest_tick().price == 1050.0

    # 預先存一份 2317 的 K 棒進 Parquet (價格 210)
    dt_2317 = datetime(2026, 10, 22, 9, 1)
    feed.store.save_kbars("2317", {
        dt_2317: {"timestamp": dt_2317, "open": 210.0, "high": 212.0, "low": 209.0, "close": 211.5, "volume": 300}
    })

    # 切換至 2317
    feed.switch_symbol("2317")
    assert feed.symbol == "2317"
    assert feed.get_latest_tick() is not None
    assert feed.get_latest_tick().price == 211.5
    bars_2317 = feed.get_bars(timeframe="1m", session="day_only")
    assert bars_2317.height == 1
    assert bars_2317["close"][0] == 211.5


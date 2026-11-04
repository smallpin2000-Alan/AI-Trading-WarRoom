# tests/test_advisor.py
from datetime import datetime, timedelta
import polars as pl
from src.ai.advisor import FuturesAIAdvisor, FuturesAdvice


def test_futures_ai_advisor_fallback():
    now = datetime.now()
    dates = [now - timedelta(minutes=5 * i) for i in range(35)][::-1]
    # 構造多頭走勢
    df = pl.DataFrame({
        "timestamp": dates,
        "open": [20000.0 + i * 15.0 for i in range(35)],
        "high": [20010.0 + i * 15.0 for i in range(35)],
        "low": [19990.0 + i * 15.0 for i in range(35)],
        "close": [20008.0 + i * 15.0 for i in range(35)],
        "volume": [100 + i * 10 for i in range(35)],
    })

    advice = FuturesAIAdvisor.analyze_market(df, symbol="TMF", timeframe="5m")
    assert isinstance(advice, FuturesAdvice)
    assert advice.symbol == "TMF"
    assert advice.action in ["BUY_LONG", "SELL_SHORT", "WAIT"]
    assert advice.long_probability >= 0.0
    assert advice.short_probability >= 0.0
    assert len(advice.rationale) > 0
    assert advice.source in ["本地量化多空模型", "Gemini AI"]


def test_futures_ai_advisor_empty():
    df_empty = pl.DataFrame({"timestamp": [], "open": [], "high": [], "low": [], "close": [], "volume": []})
    advice = FuturesAIAdvisor.analyze_market(df_empty, symbol="TMF", timeframe="5m")
    assert advice.action == "WAIT"
    assert "數據回補中" in advice.action_label


def test_stock_ai_advisor_unit_and_label():
    now = datetime.now()
    dates = [now - timedelta(minutes=5 * i) for i in range(35)][::-1]
    df_stock = pl.DataFrame({
        "timestamp": dates,
        "open": [1000.0 + i * 2.0 for i in range(35)],
        "high": [1005.0 + i * 2.0 for i in range(35)],
        "low": [995.0 + i * 2.0 for i in range(35)],
        "close": [1003.0 + i * 2.0 for i in range(35)],
        "volume": [500 + i * 20 for i in range(35)],
    })

    assert FuturesAIAdvisor.is_futures("TMF") is True
    assert FuturesAIAdvisor.is_futures("TMFR1") is True
    assert FuturesAIAdvisor.is_futures("TXFR1") is True
    assert FuturesAIAdvisor.is_futures("2330") is False
    assert FuturesAIAdvisor.is_futures("2317") is False

    advice = FuturesAIAdvisor.analyze_market(df_stock, symbol="2330", timeframe="5m")
    assert advice.symbol == "2330"
    assert "元" in advice.stop_loss
    assert "元" in advice.target_price
    assert "元" in advice.entry_zone

    levels = advice.extract_price_levels(fallback_price=1070.0)
    assert levels["entry"] > 0
    assert levels["stop_loss"] > 0
    assert levels["target"] > 0


def test_cloud_intel_integration():
    from src.ai.cloud_intel import CloudIntelSnapshot
    now = datetime.now()
    dates = [now - timedelta(minutes=5 * i) for i in range(35)][::-1]
    df = pl.DataFrame({
        "timestamp": dates,
        "open": [22000.0 + i * 10.0 for i in range(35)],
        "high": [22015.0 + i * 10.0 for i in range(35)],
        "low": [21990.0 + i * 10.0 for i in range(35)],
        "close": [22010.0 + i * 10.0 for i in range(35)],
        "volume": [200 + i * 5 for i in range(35)],
    })
    snap = CloudIntelSnapshot(
        symbol="TMFR1",
        fetched_at="20:40:00",
        global_bias_label="🟢 國際盤偏多助漲",
        global_summary_text="那指期 🔺+0.85% ｜ 台積ADR 🔺+1.20%",
        news_headlines=["外資期貨淨空單大減，台指期夜盤強彈"],
        news_sentiment_label="🟢 消息面偏多",
        gcs_connected=True,
        institutional_note="雲端籌碼偏多",
        is_in_morning_picks=True,
        ai_win_rate=75.0,
        ai_verified_count=8,
    )
    prompt_ctx = snap.to_prompt_context()
    assert "Google 雲端四維度即時情報庫" in prompt_ctx
    assert "75%" in prompt_ctx

    advice = FuturesAIAdvisor.analyze_market(df, symbol="TMFR1", timeframe="5m", cloud_intel=snap)
    assert advice.action in ["BUY_LONG", "SELL_SHORT", "WAIT"]
    assert advice.long_probability >= 50.0


def test_market_session_status_open_and_close():
    # 週三 10:30 -> 台股開盤、期貨日盤開盤
    dt_open = datetime(2026, 11, 4, 10, 30)
    assert FuturesAIAdvisor.get_market_session_status("2330", dt_open)[0] is True
    assert FuturesAIAdvisor.get_market_session_status("TMFR1", dt_open)[0] is True

    # 週三 14:26 -> 台股已收盤 (13:30)、期貨日盤已收盤 (13:45 ~ 15:00 休息)
    dt_closed_afternoon = datetime(2026, 11, 4, 14, 26)
    is_stk_open, stk_msg = FuturesAIAdvisor.get_market_session_status("2330", dt_closed_afternoon)
    is_fut_open, fut_msg = FuturesAIAdvisor.get_market_session_status("TMFR1", dt_closed_afternoon)
    assert is_stk_open is False and "已收盤" in stk_msg
    assert is_fut_open is False and "15:00 夜盤開盤" in fut_msg

    # 週三 15:30 -> 台股仍收盤、期貨夜盤開盤
    dt_night = datetime(2026, 11, 4, 15, 30)
    assert FuturesAIAdvisor.get_market_session_status("2330", dt_night)[0] is False
    assert FuturesAIAdvisor.get_market_session_status("TMFR1", dt_night)[0] is True




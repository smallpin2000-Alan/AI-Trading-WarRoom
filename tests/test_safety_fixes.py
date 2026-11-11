# tests/test_safety_fixes.py
"""
2026-11-11 全專案審查後的安全性修正回歸測試：
- AI 顧問在爆量長 K 棒下不可當機
- 停損守護不受「暫停 / 風控鎖定 / 無 K 棒」影響
- 風控計帳（換日歸零、連虧休眠、反手存檔、交易紀錄不截斷）
- 對話框指令（否定句不執行、平倉不解鎖、使用模擬倉專屬價格）
"""
from datetime import datetime, timedelta

import polars as pl
import pytest

from src.ai.advisor import FuturesAIAdvisor
from src.core import paper_trader as pt_module
from src.core.engine import ContractSpec
from src.core.paper_trader import PaperTradingEngine, RiskConfig
from src.schema.strategy import StrategySchema


def _strategy() -> StrategySchema:
    return StrategySchema.model_validate({
        "id": "safety_test_v1",
        "version": 1,
        "symbol": "TMF",
        "timeframe": "5m",
        "session": "day_and_night",
        "indicators": {
            "ma5": {"type": "SMA", "source": "close", "length": 5},
            "ma20": {"type": "SMA", "source": "close", "length": 20},
        },
        "entry_long": {"cross_above": ["ma5", "ma20"]},
        "entry_short": {"cross_below": ["ma5", "ma20"]},
        "exit": {
            "stop_loss": {"type": "fixed_points", "value": 45.0},
            "take_profit": {"type": "fixed_points", "value": 60.0},
            "trailing_stop": {"type": "points", "value": 15.0, "activation_points": 35.0},
        },
    })


def _contract() -> ContractSpec:
    return ContractSpec(
        symbol="TMF", name="微型臺指", exchange="TAIFEX", point_value=10.0, tick_size=1.0,
        tax_rate=0.00002, default_commission=12.0, initial_margin=16100.0, maintenance_margin=12400.0,
    )


def _engine(**risk_kwargs) -> PaperTradingEngine:
    risk = RiskConfig(**{"max_daily_loss": 5000.0, "max_contracts": 1, **risk_kwargs})
    eng = PaperTradingEngine(_strategy(), _contract(), initial_capital=100_000.0, risk_config=risk)
    ok, _ = eng.start_session(credibility_passed=True)
    assert ok
    return eng


def _bars(n: int = 30, base: float = 20000.0, start: datetime = datetime(2026, 11, 10, 9, 0)) -> pl.DataFrame:
    closes = [base + i for i in range(n)]
    return pl.DataFrame({
        "timestamp": [start + timedelta(minutes=5 * i) for i in range(n)],
        "open": closes,
        "high": [c + 5 for c in closes],
        "low": [c - 5 for c in closes],
        "close": closes,
        "volume": [100] * n,
    })


def _open_long(eng: PaperTradingEngine, price: float = 20000.0) -> None:
    assert eng._open_new_position(direction=1, raw_price=price, bar_time_str="2026-11-10 09:05", reason="test")
    assert eng.position is not None


# ---------------------------------------------------------------------------
# AI 顧問
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bearish", [True, False])
def test_analyze_market_survives_climax_bar(bearish):
    """最新一根為 300 點爆量長 K（振幅 >= 1.85 倍 ATR）時不可拋例外，且不得追價"""
    df = _bars(80)
    rows = df.to_dicts()
    last = rows[-1]
    o = last["open"]
    if bearish:
        last.update(high=o + 5, low=o - 300, close=o - 290)
    else:
        last.update(high=o + 300, low=o - 5, close=o + 290)
    advice = FuturesAIAdvisor.analyze_market(pl.DataFrame(rows), symbol="TMFR1", timeframe="5m", fast_local_only=True)
    forbidden = "SELL_SHORT" if bearish else "BUY_LONG"
    assert advice.action != forbidden


# ---------------------------------------------------------------------------
# 停損守護
# ---------------------------------------------------------------------------
def test_stop_loss_fires_while_session_paused():
    """暫停自動操盤（is_active=False）後，既有持倉仍必須受停損保護"""
    eng = _engine()
    _open_long(eng)
    eng.is_active = False
    res = eng.check_position_stops(current_price=19900.0, df_bars=_bars())
    assert eng.position is None
    assert res and "停損" in res


def test_stop_loss_fires_while_degradation_locked():
    eng = _engine()
    _open_long(eng)
    eng.is_active = False
    eng.degradation_locked = True
    eng.check_position_stops(current_price=19900.0, df_bars=_bars())
    assert eng.position is None
    assert eng.degradation_locked


def test_stop_loss_fires_without_bars():
    """K 棒尚未載入時，固定點數停損仍須生效"""
    eng = _engine()
    _open_long(eng)
    eng.check_position_stops(current_price=19900.0, df_bars=_bars(0))
    assert eng.position is None


def test_sync_with_market_guards_position_while_paused():
    eng = _engine()
    _open_long(eng, price=20029.0)
    eng.is_active = False
    eng.sync_with_market(_bars(), latest_price=19900.0)
    assert eng.position is None


def test_no_new_entry_while_paused():
    eng = _engine()
    eng.is_active = False
    res = eng.sync_with_ai_advice("BUY_LONG", 90.0, 20030.0, _bars())
    assert eng.position is None
    assert res == "系統暫停或風控鎖定中"


# ---------------------------------------------------------------------------
# 風控計帳
# ---------------------------------------------------------------------------
def test_degradation_lock_triggers_on_ai_path():
    """AI 即時路徑連續停損達上限時，必須觸發模型退化休眠"""
    eng = _engine(max_consecutive_losses=2)
    for _ in range(2):
        _open_long(eng)
        eng.check_position_stops(current_price=19900.0, df_bars=_bars())
        assert eng.position is None
    assert eng.degradation_locked
    assert not eng.is_active


def test_reverse_close_is_persisted_even_if_reopen_fails(tmp_path):
    """反手時原部位已平倉但新倉被攔截，平倉紀錄仍須落盤"""
    eng = _engine()
    _open_long(eng)
    eng.order_safety_locked = True  # 新倉會被下單安全鎖攔截
    eng.sync_with_ai_advice("SELL_SHORT", 90.0, 20010.0, _bars())
    assert eng.position is None
    assert len(eng.closed_trades) == 1

    restored = PaperTradingEngine(_strategy(), _contract())
    assert restored.load_state_from_disk()
    assert len(restored.closed_trades) == 1


def test_state_roundtrip_keeps_all_trades_and_cooldown(tmp_path):
    """超過 100 筆交易後存檔再讀回，已實現損益不可跳動；冷卻時間戳須持久化"""
    eng = _engine(max_consecutive_losses=0, max_daily_loss=10_000_000.0)
    for _ in range(105):
        _open_long(eng)
        eng._close_current_position(20010.0, exit_idx=-1, reason="test")
    eng._last_stop_out_ts = 1234.5
    eng._last_stop_out_dir = -1
    eng._last_entry_bar_ts = datetime(2026, 11, 10, 9, 5)
    path = tmp_path / "state.json"
    eng.save_state_to_disk(path)

    restored = PaperTradingEngine(_strategy(), _contract())
    assert restored.load_state_from_disk(path)
    assert len(restored.closed_trades) == 105
    assert restored.realized_pnl == eng.realized_pnl
    assert restored._last_stop_out_ts == 1234.5
    assert restored._last_stop_out_dir == -1
    assert restored._last_entry_bar_ts == datetime(2026, 11, 10, 9, 5)


def test_corrupt_state_file_is_preserved_not_silently_ignored(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{ this is not json", encoding="utf-8")
    eng = PaperTradingEngine(_strategy(), _contract())
    assert eng.load_state_from_disk(path) is False
    assert list(tmp_path.glob("state.json.corrupt*")), "損毀的狀態檔應改名保留以便事後救援"


def test_daily_loss_lock_resets_on_new_trading_day(monkeypatch):
    """單日虧損鎖定在換交易日後自動解除並重新計帳（但不自動恢復操盤）"""
    monkeypatch.setattr(pt_module, "current_trading_day", lambda: "2026-11-10")
    eng = _engine(max_daily_loss=500.0)
    _open_long(eng)
    assert eng.check_daily_loss_limit(19900.0)
    assert eng.daily_loss_locked

    # 同一交易日：仍鎖定
    assert eng.check_daily_loss_limit(19900.0)
    assert eng.daily_loss_locked

    monkeypatch.setattr(pt_module, "current_trading_day", lambda: "2026-11-11")
    assert not eng.check_daily_loss_limit(19900.0)
    assert not eng.daily_loss_locked
    assert not eng.is_active  # 換日只解鎖，不擅自重新啟動


def test_daily_profit_target_uses_daily_pnl_not_cumulative(monkeypatch):
    """獲利達標以「當日」損益計算；解鎖後不可因歷史累計獲利立刻再鎖"""
    monkeypatch.setattr(pt_module, "current_trading_day", lambda: "2026-11-10")
    eng = _engine(daily_profit_target=500.0)
    _open_long(eng)
    assert eng.check_daily_profit_target(20100.0)
    assert eng.daily_profit_locked

    eng.reset_risk_locks(reset_daily_accounting=True)
    assert not eng.check_daily_profit_target(20100.0)
    assert not eng.daily_profit_locked


def test_trading_day_groups_night_session_with_next_day():
    f = pt_module.trading_day_of
    assert f(datetime(2026, 11, 10, 10, 0)) == "2026-11-10"   # 週二日盤
    assert f(datetime(2026, 11, 10, 15, 0)) == "2026-11-11"   # 週二夜盤屬於週三
    assert f(datetime(2026, 11, 11, 4, 59)) == "2026-11-11"   # 跨午夜仍屬週三
    assert f(datetime(2026, 11, 11, 9, 0)) == "2026-11-11"


def test_kill_switch_rejects_cross_instrument_price_without_raising():
    """Kill Switch 收到明顯不屬於該商品的價格時：仍鎖定系統，但不得以錯價結算，也不得拋例外"""
    eng = _engine()
    _open_long(eng)
    msg = eng.trigger_kill_switch(current_price=1050.0)
    assert eng.kill_switch_locked
    assert eng.position is not None
    assert len(eng.closed_trades) == 0
    assert "拒絕" in msg


def test_reset_does_not_grant_credibility():
    eng = PaperTradingEngine(_strategy(), _contract())
    assert eng.credibility_verified is False
    eng.reset_all_simulation(save_to_disk=False)
    assert eng.credibility_verified is False


def test_unverified_start_requires_explicit_simulation_override():
    """未通過五道檢核：預設禁止啟動；明確確認後可啟動，但必須留下稽核紀錄且不得標記為已通過"""
    eng = PaperTradingEngine(_strategy(), _contract())
    ok, _ = eng.start_session(credibility_passed=False)
    assert not ok and not eng.is_active

    ok, msg = eng.start_session(credibility_passed=False, simulation_override=True)
    assert ok and eng.is_active
    assert eng.credibility_verified is False
    assert "未通過五道可信度檢核" in msg
    assert eng.audit_logs[-1].event_type == "RISK_BREACH"


def test_short_reverse_reason_label():
    """持多單時出現空方訊號，原因標籤應為 reverse（原本誤判為 entry_short）"""
    eng = _engine()
    assert eng._entry_reason(target_dir=-1, pos_dir=1) == "reverse"
    assert eng._entry_reason(target_dir=-1, pos_dir=0) == "entry_short"
    assert eng._entry_reason(target_dir=1, pos_dir=-1) == "reverse"
    assert eng._entry_reason(target_dir=1, pos_dir=0) == "entry_long"


# ---------------------------------------------------------------------------
# 對話框指令
# ---------------------------------------------------------------------------
def _chat(eng, msg, **kw):
    return FuturesAIAdvisor.chat_with_advisor(
        user_msg=msg, symbol=kw.pop("symbol", "TMFR1"), timeframe="5m",
        current_price=kw.pop("current_price", 20000.0), paper_trader=eng, gemini_api_key="", **kw,
    )


@pytest.mark.parametrize("msg", ["不要重置", "重置會怎樣？", "為什麼會被重置", "先別平倉", "可以平倉嗎"])
def test_chat_negation_or_question_does_not_execute(msg):
    eng = _engine()
    _open_long(eng)
    reply = _chat(eng, msg)
    assert eng.position is not None
    assert len(eng.closed_trades) == 0
    assert "未執行" in reply


def test_chat_close_keeps_risk_locks_and_does_not_reactivate():
    eng = _engine()
    _open_long(eng)
    eng.is_active = False
    eng.daily_loss_locked = True
    reply = _chat(eng, "幫我平倉", trader_price=20010.0)
    assert eng.position is None
    assert eng.daily_loss_locked, "平倉指令不可順手解除單日虧損鎖"
    assert not eng.is_active, "平倉指令不可擅自重新啟動自動操盤"
    assert not eng.kill_switch_locked
    assert "平倉" in reply


def test_chat_close_uses_trader_price_not_viewed_symbol_price():
    """看板顯示個股（約 1000 元）時下平倉指令，必須用微台專屬價格結算"""
    eng = _engine()
    _open_long(eng)
    _chat(eng, "平倉", symbol="2330", current_price=1050.0, trader_price=20010.0)
    assert eng.position is None
    assert eng.closed_trades[-1].exit_price == pytest.approx(20010.0, abs=5)


def test_chat_close_with_wrong_instrument_price_is_rejected():
    eng = _engine()
    _open_long(eng)
    reply = _chat(eng, "平倉", symbol="2330", current_price=1050.0)
    assert eng.position is not None
    assert len(eng.closed_trades) == 0
    assert "拒絕" in reply or "失敗" in reply


def test_chat_pause_and_resume_commands_work():
    eng = _engine()
    reply = _chat(eng, "暫停操盤")
    assert not eng.is_active
    assert "暫停" in reply
    reply = _chat(eng, "啟動操盤")
    assert eng.is_active
    assert "啟動" in reply


def test_chat_resume_blocked_when_not_credibility_verified():
    eng = _engine()
    eng.stop_session()
    eng.credibility_verified = False
    reply = _chat(eng, "啟動操盤")
    assert not eng.is_active
    assert "可信度" in reply


def test_chat_read_only_mode_executes_nothing():
    eng = _engine()
    _open_long(eng)
    reply = _chat(eng, "平倉", allow_commands=False)
    assert eng.position is not None
    assert "唯讀" in reply


def test_state_sync_disabled_locally(monkeypatch, tmp_path):
    """本機（無 K_SERVICE）時不得嘗試連線 GCS"""
    from src.data import state_sync
    monkeypatch.delenv("K_SERVICE", raising=False)
    monkeypatch.setenv("GCS_BUCKET", "dummy-bucket")
    assert state_sync.is_enabled() is False
    assert state_sync.download_if_available(tmp_path / "x.json") is False


def test_state_sync_uploads_saved_content_on_cloud_run(monkeypatch, tmp_path):
    from src.data import state_sync
    uploaded = []
    monkeypatch.setenv("K_SERVICE", "svc")
    monkeypatch.setenv("GCS_BUCKET", "dummy-bucket")
    monkeypatch.setattr(state_sync, "upload_in_background", lambda path, content: uploaded.append((path.name, content)))
    eng = _engine()
    eng.save_state_to_disk(tmp_path / "state.json")
    assert uploaded and uploaded[0][0] == "state.json"
    assert uploaded[0][1] == (tmp_path / "state.json").read_text(encoding="utf-8")

# tests/conftest.py
"""
全域測試隔離：確保任何單元測試都不會寫入正式的 data/ 目錄。

背景：analyze_market 在開盤時段會呼叫 CloudIntelligenceHub.record_prediction，
PaperTradingEngine 的撮合路徑會呼叫 save_state_to_disk，兩者預設都寫入 data/。
"""
import pytest

from src.ai.cloud_intel import CloudIntelligenceHub
from src.core import paper_trader as paper_trader_module


@pytest.fixture(autouse=True)
def _isolate_production_data(monkeypatch, tmp_path):
    # AI 預測紀錄：測試中一律不落盤、不上傳 GCS
    monkeypatch.setattr(
        CloudIntelligenceHub,
        "record_prediction",
        classmethod(lambda cls, *args, **kwargs: None),
    )
    # 模擬倉狀態檔：改寫到每個測試專屬的暫存目錄
    monkeypatch.setattr(
        paper_trader_module,
        "DEFAULT_STATE_PATH",
        tmp_path / "paper_trading_state.json",
    )
    yield

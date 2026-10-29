# app.py
"""
AI 自動下單與回測系統 - 入口檔案
執行指令：streamlit run app.py
"""
import sys
import importlib
import src.core.indicators
import src.ai.confidence_engine
import src.ai.ensemble
import src.core.engine
import src.core.validator
import src.core.paper_trader
import src.broker.taishin_quote
import src.ui.dashboard

# 強制每次 rerun 時熱重載最新模組
importlib.reload(src.core.indicators)
importlib.reload(src.ai.confidence_engine)
importlib.reload(src.ai.ensemble)
importlib.reload(src.core.engine)
importlib.reload(src.core.validator)
importlib.reload(src.core.paper_trader)
importlib.reload(src.broker.taishin_quote)
importlib.reload(src.ui.dashboard)

if __name__ == "__main__":
    src.ui.dashboard.main()

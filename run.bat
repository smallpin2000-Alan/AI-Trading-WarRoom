@echo off
chcp 65001 >nul
title AI 智慧看盤與多空決策戰情室 v2.7.9
echo ========================================================
echo   🎯 AI 智慧看盤與多空決策戰情室 (v2.7.9)
echo   正在啟動服務並自動開啟瀏覽器...
echo ========================================================
echo.

cd /d "%~dp0"

:: 設定 UTF-8 編碼環境變數
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

:: 優先使用系統 python 啟動
python -X utf8 -m streamlit run app.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo 系統預設 python 啟動失敗，嘗試使用 Python 3.14 絕對路徑...
    "C:\Users\alan\AppData\Local\Programs\Python\Python314\python.exe" -X utf8 -m streamlit run app.py
)

pause

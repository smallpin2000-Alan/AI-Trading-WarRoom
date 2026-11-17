@echo off
chcp 65001 >nul
title AI 智慧看盤與多空決策 — Excel 戰情室即時終端

echo ============================================================
echo   🚀 正在啟動 AI 交易戰情室 — EXCEL 即時看盤與雙向控制終端
echo ============================================================
echo.
echo [1/3] 檢查 Python 與 Excel 環境...
python -c "import openpyxl, win32com.client; print('OK: Python 套件環境正常')" 2>nul
if errorlevel 1 (
    echo [錯誤] 缺少 openpyxl 或 pywin32 套件，正在嘗試安裝...
    pip install openpyxl pywin32
)

echo.
echo [2/3] 檢查 Excel 戰情室範本...
if not exist "AI_Trading_WarRoom.xlsx" (
    echo [建立] 正在建立 AI_Trading_WarRoom.xlsx 專業活頁簿...
    python -m src.ui.excel_warroom --init
)

echo.
echo [3/3] 啟動即時看盤同步 (每 10 秒刷新一次)...
echo 提示：
echo   - Excel 視窗將自動開啟，數據會自動定時填入
echo   - 可直接在 Excel 儲存格 B5 切換 ON/OFF 控制自動操盤
echo   - 可直接在 Excel 儲存格 D5 輸入「平倉」手動緊急市價出場
echo   - 欲停止請關閉此視窗或按下 Ctrl+C
echo.

python -m src.ui.excel_warroom --loop --interval 10

pause

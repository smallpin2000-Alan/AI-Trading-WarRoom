@echo off
title AI Auto Trading and Backtesting System
echo ========================================================
echo   AI Auto Trading and Backtesting System
echo   Starting server and opening browser...
echo ========================================================
echo.
cd /d "%~dp0"
python -m streamlit run app.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Failed with python, trying direct executable path...
    "C:\Users\alan\AppData\Local\Programs\Python\Python314\python.exe" -m streamlit run app.py
)
pause

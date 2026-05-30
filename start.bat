@echo off
title Portfolio Intelligence Agent Server

echo ===================================================
echo   Starting Portfolio Intelligence Backend Server
echo ===================================================
echo.
echo Step 1: Verifying/Installing python dependencies...
pip install -r requirements.txt
echo.
echo Step 2: Launching FastAPI app on http://127.0.0.1:8000
echo.
uvicorn main:app --reload
pause

@echo off
rem 在 Windows 上运行豆宝
cd /d "%~dp0.."
if not exist data mkdir data
start "" /min python server.py --port 8100
timeout /t 3 /nobreak >nul
start "" http://127.0.0.1:8100

@echo off
setlocal
cd /d "%~dp0"
echo Starting NIFTY Options Dashboard...
where py >nul 2>nul
if %errorlevel%==0 (
  start "NIFTY Live Backend" cmd /k py server.py
  timeout /t 2 >nul
  start "" nifty-options-terminal-dashboard.html
  goto :eof
)
where python >nul 2>nul
if %errorlevel%==0 (
  start "NIFTY Live Backend" cmd /k python server.py
  timeout /t 2 >nul
  start "" nifty-options-terminal-dashboard.html
  goto :eof
)
echo Python not found. Install from https://www.python.org/downloads/
pause

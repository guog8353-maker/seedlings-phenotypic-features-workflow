@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "PYTHON_EXE="
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
if not defined PYTHON_EXE (
  for /f "delims=" %%I in ('where python 2^>nul') do (
    if not defined PYTHON_EXE set "PYTHON_EXE=%%I"
  )
)
if not defined PYTHON_EXE (
  echo [错误] 未找到 Python，请先安装 Python 3.12。
  pause
  exit /b 1
)

"%PYTHON_EXE%" "main.py"
echo.
echo 程序退出代码：%errorlevel%
pause
endlocal

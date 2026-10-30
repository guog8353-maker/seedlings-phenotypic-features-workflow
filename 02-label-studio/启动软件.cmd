@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "PYTHON_EXE="
if exist ".venv\Scripts\pythonw.exe" set "PYTHON_EXE=%CD%\.venv\Scripts\pythonw.exe"
if not defined PYTHON_EXE (
  for /f "delims=" %%I in ('where pythonw 2^>nul') do (
    if not defined PYTHON_EXE set "PYTHON_EXE=%%I"
  )
)

if not defined PYTHON_EXE (
  echo [错误] 未找到 Python。请先安装 Python 3.12 和 requirements.txt 中的依赖。
  echo 也可以在本目录创建 .venv 虚拟环境后重新双击本脚本。
  pause
  exit /b 1
)

set "LOG_FILE=%CD%\launcher_error.log"
if exist "%LOG_FILE%" del "%LOG_FILE%"
"%PYTHON_EXE%" "main.py" >"%LOG_FILE%" 2>&1
if errorlevel 1 (
  echo [错误] 软件启动失败，将打开错误日志。
  if exist "%LOG_FILE%" start notepad "%LOG_FILE%"
  pause
  exit /b 1
)
if exist "%LOG_FILE%" del "%LOG_FILE%"
endlocal

@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

REM Switch to script directory for portable deployment
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Prefer bundled Python interpreter (portable mode)
if exist "%SCRIPT_DIR%\.python\python.exe" (
    set "PYTHON=%SCRIPT_DIR%\.python\python.exe"
    goto :run
)

REM Try GPU conda environment first, then CPU environment
where conda >nul 2>nul
if %errorlevel% == 0 (
    call conda activate automl-platform-gpu >nul 2>nul
    if !errorlevel! == 0 (
        set "PYTHON=python"
        goto :run
    )
    call conda activate automl-platform >nul 2>nul
    if !errorlevel! == 0 (
        set "PYTHON=python"
        goto :run
    )
)

REM Fallback to system PATH python
set "PYTHON=python"

:run
"%PYTHON%" app.py
if %errorlevel% neq 0 (
    echo.
    echo Program exited with error. Please check messages above.
    pause
)
endlocal

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$localPython = Join-Path $scriptDir ".venv\Scripts\python.exe"
if (Test-Path $localPython) {
    $pythonExe = $localPython
} else {
    $command = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        Write-Host "错误：未找到 Python，请先安装 Python 3.12。" -ForegroundColor Red
        Read-Host "按 Enter 键继续"
        exit 1
    }
    $pythonExe = $command.Source
}

& $pythonExe "main.py"
if ($LASTEXITCODE -ne 0) {
    Write-Host "程序异常退出，错误代码：$LASTEXITCODE" -ForegroundColor Red
    Read-Host "按 Enter 键继续"
}

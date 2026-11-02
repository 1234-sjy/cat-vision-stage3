param([string]$PythonExe = $env:CATVISION_PYTHON, [switch]$FastMode)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not $PythonExe) { $PythonExe = Join-Path $projectRoot '.venv/Scripts/python.exe' }
if (-not (Test-Path -LiteralPath $PythonExe)) { throw '尚未恢复运行环境，请先运行“安装环境.cmd”。' }
& $PythonExe -c "from catvision.server import model_files_ready; from catvision.common import ROOT; raise SystemExit(0 if model_files_ready(ROOT) else 1)"
if ($LASTEXITCODE -ne 0) { throw '模型文件不齐，请按说明恢复 weights 文件夹。' }
Write-Host '打开 http://127.0.0.1:8765 使用项目；在此窗口按 Ctrl+C 停止服务。'
if ($FastMode) {
    Write-Host '正在预加载模型；准备完成后识别更快，模型将常驻显存。'
    & $PythonExe -m catvision.server --fast-demo
} else {
    & $PythonExe -m catvision.server
}

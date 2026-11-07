param([switch]$DownloadModels)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$projectPython = Join-Path $projectRoot '.venv/Scripts/python.exe'
Set-Location -LiteralPath $projectRoot
if (-not (Test-Path -LiteralPath $projectPython)) {
    & python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw '请先安装 Python 3.12，并勾选加入 PATH。' }
}
& $projectPython -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)"
if ($LASTEXITCODE -ne 0) { throw '本项目要求Python 3.12，请用该版本创建环境。' }
& $projectPython -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'PyTorch安装未完成，已停止，不自动重试。' }
& $projectPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw '依赖安装未完成，已停止，不自动重试。' }
if ($DownloadModels) {
    & $projectPython scripts/download_models.py --only all
    if ($LASTEXITCODE -ne 0) { throw '模型下载未完成，已停止，请保留日志。' }
}
Write-Host '依赖安装成功。使用方法见 README.md。'

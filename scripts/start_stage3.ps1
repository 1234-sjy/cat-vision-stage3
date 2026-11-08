param([string]$Distribution = 'CatVision', [switch]$Build, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$taskEncoding = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = $taskEncoding
[Console]::OutputEncoding = $taskEncoding
$taskRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'stage3_wsl_session.ps1')
Start-CatVisionWslSession -ProjectRoot $taskRoot -Distribution $Distribution
$linuxRoot = & wsl.exe -d $Distribution -u root --exec wslpath -u $taskRoot
if ($LASTEXITCODE -ne 0) {
    throw '未找到项目Linux环境，请先按 README.md 中的 Windows 启动说明准备 WSL2。'
}
$taskArguments = @('-d',$Distribution,'-u','root','--cd',$linuxRoot.Trim(),'--exec','bash','scripts/start_stage3.sh')
if ($Build) { $taskArguments += '--build' }
& wsl.exe @taskArguments
if ($LASTEXITCODE -ne 0) { throw "Linux启动失败，退出码 $LASTEXITCODE；请保留屏幕日志。" }
if (-not $NoBrowser) { Start-Process 'http://127.0.0.1:8080' }

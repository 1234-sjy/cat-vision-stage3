param([string]$Distribution = 'CatVision')
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'stage3_wsl_session.ps1')
$linuxRoot = & wsl.exe -d $Distribution -u root --exec wslpath -u $taskRoot
if ($LASTEXITCODE -ne 0) { throw 'The WSL project directory is unavailable.' }
& wsl.exe -d $Distribution -u root --cd $linuxRoot.Trim() --exec docker compose stop
if ($LASTEXITCODE -ne 0) { throw 'Containers did not stop; the WSL session has been kept open.' }
Stop-CatVisionWslSession -ProjectRoot $taskRoot

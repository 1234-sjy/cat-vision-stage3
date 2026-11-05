function Get-CatVisionWslSession {
    param([string]$StatePath)
    if (-not (Test-Path -LiteralPath $StatePath)) { return $null }
    try {
        $session = Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json
        $sessionProcess = Get-Process -Id $session.process_id -ErrorAction Stop
        $recordedStart = ([DateTimeOffset]$session.started_utc).UtcDateTime
        if ($sessionProcess.ProcessName -ne 'wsl' -or
            $sessionProcess.StartTime.ToUniversalTime().Ticks -ne $recordedStart.Ticks) {
            return $null
        }
        return [pscustomobject]@{ Process = $sessionProcess; Distribution = $session.distribution }
    } catch { return $null }
}

function Start-CatVisionWslSession {
    param([string]$ProjectRoot, [string]$Distribution)
    $statePath = Join-Path $ProjectRoot 'runtime/stage3-wsl-session.json'
    $existing = Get-CatVisionWslSession -StatePath $statePath
    if ($null -ne $existing) {
        if ($existing.Distribution -ne $Distribution) {
            throw 'Stop the existing WSL session before selecting another distribution.'
        }
        return
    }
    # systemd services alone do not keep a WSL distribution running.
    # Keep an idle Linux process attached until the explicit stop command.
    if ($Distribution -notmatch '^[A-Za-z0-9_.-]+$') {
        throw 'Use a WSL distribution name containing letters, digits, dots, underscores or hyphens.'
    }
    $arguments = @('-d', $Distribution, '-u', 'root', '--exec', 'sleep', 'infinity')
    New-Item -ItemType Directory -Path (Split-Path -Parent $statePath) -Force | Out-Null
    $logPrefix = Join-Path $ProjectRoot ('runtime/stage3-wsl-session-' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ'))
    # Keep early WSL errors in local logs instead of a hidden console.
    $sessionProcess = Start-Process -FilePath 'wsl.exe' -ArgumentList $arguments -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput ($logPrefix + '.out.log') -RedirectStandardError ($logPrefix + '.err.log')
    if ($sessionProcess.WaitForExit(1000)) {
        throw ('The WSL session exited early. See ' + $logPrefix + '.err.log')
    }
    [pscustomobject]@{
        process_id = $sessionProcess.Id
        started_utc = $sessionProcess.StartTime.ToUniversalTime().ToString('o')
        distribution = $Distribution
    } | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding UTF8
}

function Stop-CatVisionWslSession {
    param([string]$ProjectRoot)
    $statePath = Join-Path $ProjectRoot 'runtime/stage3-wsl-session.json'
    $existing = Get-CatVisionWslSession -StatePath $statePath
    if ($null -ne $existing) { Stop-Process -Id $existing.Process.Id -ErrorAction Stop }
    if (Test-Path -LiteralPath $statePath) { Remove-Item -LiteralPath $statePath }
}

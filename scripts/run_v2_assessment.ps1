param(
    [Parameter(Mandatory=$true)][string]$PythonExe,
    [Parameter(Mandatory=$true)][string]$DataRoot,
    [Parameter(Mandatory=$true)][string]$Stage1Predictions
)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$stateFile=Join-Path $projectRoot 'runtime/model-task-state.json'
$modelState=Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
if ($modelState.task -eq 'train-swin-v2' -and $modelState.state -eq 'running') {
    $runningTraining=Get-CimInstance Win32_Process -Filter "ProcessId=$($modelState.pid)"
    if (-not $runningTraining -or $runningTraining.CommandLine -notmatch 'scripts[/\\]train_transformer.py') {
        throw 'Training state is stale or process identity changed; no recovery attempted.'
    }
    Write-Output 'WAITING FOR THE ALREADY RUNNING TRAINING; no model task started.'
    Wait-Process -Id $modelState.pid
}
$modelState=Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
if ($modelState.task -ne 'train-swin-v2' -or $modelState.state -ne 'completed') {
    throw 'Training did not complete successfully; assessment stopped without retry.'
}
function Run-Stage([string[]]$TaskArguments) {
    Write-Output ('STAGE '+($TaskArguments -join ' '))
    & $PythonExe -u -X utf8 @TaskArguments
    if ($LASTEXITCODE -ne 0) { throw "Stage stopped with exit $LASTEXITCODE; no next stage, retry or restart." }
}
Run-Stage @('scripts/select_transformer.py','--data-root',$DataRoot)
Run-Stage @('scripts/freeze_v2.py')
Run-Stage @('scripts/evaluate.py','--route','traditional','--set','comparison-v2','--data-root',$DataRoot)
Run-Stage @('scripts/evaluate_original_classifier.py','--data-root',$DataRoot)
Run-Stage @('scripts/evaluate.py','--route','traditional','--set','comparison','--data-root',$DataRoot)
Run-Stage @('scripts/evaluate.py','--route','vlm','--set','comparison-v2','--data-root',$DataRoot)
Run-Stage @('scripts/make_report_v2.py','--data-root',$DataRoot,'--stage1-predictions',$Stage1Predictions)
Write-Output 'ALL V2 ASSESSMENT STAGES COMPLETED; NO AUTOMATIC RETRIES.'

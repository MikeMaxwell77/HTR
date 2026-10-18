# Examples: .\docker.ps1 Setup; .\docker.ps1 Smoke; .\docker.ps1 Train --epochs 50
param(
    [ValidateSet('Setup', 'Check', 'Smoke', 'Train', 'Continuous', 'Logs', 'Status', 'Stop')]
    [string]$Action = 'Setup',
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$TrainingArguments
)
$ErrorActionPreference = 'Stop'
$dockerCommand = Get-Command docker -ErrorAction SilentlyContinue
$dockerExecutable = if ($dockerCommand) { $dockerCommand.Source } else { $null }
if (-not $dockerExecutable) {
    foreach ($candidate in @(
        "$env:LOCALAPPDATA/Programs/DockerDesktop/resources/bin/docker.exe",
        "$env:ProgramFiles/Docker/Docker/resources/bin/docker.exe"
    )) {
        if (Test-Path -LiteralPath $candidate) {
            $dockerExecutable = $candidate
            break
        }
    }
}
if (-not $dockerExecutable) { throw 'Docker CLI not found. Install Docker Desktop, then retry.' }
# Credential helpers and Compose plugins may also need the Docker bin directory.
$env:PATH = "$(Split-Path -Parent $dockerExecutable);$env:PATH"

function Invoke-Docker {
    & $dockerExecutable @args
    if ($LASTEXITCODE -ne 0) { throw "Docker failed with exit code $LASTEXITCODE." }
}

Push-Location $PSScriptRoot
try {
    & $dockerExecutable info --format '{{.OSType}}'
    if ($LASTEXITCODE -ne 0) { throw 'Start Docker Desktop with its Linux engine, then retry.' }
    if ($Action -in @('Setup', 'Smoke', 'Train')) { Invoke-Docker compose build }
    switch ($Action) {
        'Setup' { Invoke-Docker compose run --rm --entrypoint python train -m docker.gpu_check }
        'Check' { Invoke-Docker compose run --rm --entrypoint python train -m docker.gpu_check }
        'Smoke' {
            # Separate artifacts so this check cannot overwrite a trained model.
            $runName = 'docker-smoke-' + [guid]::NewGuid().ToString('N')
            Invoke-Docker compose run --rm train --limit 100 --epochs 1 --batch-size 2 --output-dir "/app/models/$runName"
        }
        'Train' { Invoke-Docker compose run --rm train @TrainingArguments }
        'Continuous' { Invoke-Docker compose -f compose.yaml -f compose.continuous.yaml up -d --build train }
        'Logs' {
            Write-Host 'Following NEW training output. It may arrive in chunks. Ctrl+C exits this viewer; training continues.'
            Invoke-Docker logs --follow --tail 0 htr-training
        }
        'Status' {
            Invoke-Docker inspect htr-training --format '{{.State.Status}} (restarts: {{.RestartCount}})'
            Invoke-Docker stats --no-stream htr-training
            if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
                & nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv
            }
        }
        'Stop' { Invoke-Docker compose -f compose.yaml -f compose.continuous.yaml stop -t 60 train }
    }
}
finally { Pop-Location }

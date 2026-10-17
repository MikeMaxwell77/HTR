# Create or refresh the persistent Python environment. Safe to run again.
$ErrorActionPreference = 'Stop'
$environmentPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $environmentPython)) {
    & py -3.12 -m venv (Join-Path $PSScriptRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Could not create .venv. Install Python 3.12 and retry.' }
}
$requirements = Join-Path $PSScriptRoot 'scripts/training/requirements-lock.txt'
if (-not (Test-Path -LiteralPath $requirements)) {
    $requirements = Join-Path $PSScriptRoot 'scripts/training/requirements.txt'
}
& $environmentPython -m pip install -r $requirements
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $environmentPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed.' }
Write-Host 'Environment ready. Run: python train.py --words-dir "C:/datasets/IAM/words"'

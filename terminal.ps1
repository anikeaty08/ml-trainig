$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $RootDir ".venv"

if (-not (Test-Path $VenvDir)) {
    throw "Missing .venv. Run install.ps1 first."
}

. (Join-Path $VenvDir "Scripts\\Activate.ps1")
python -m backend.cli

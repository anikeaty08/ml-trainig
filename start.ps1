$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $RootDir ".venv"

if (-not (Test-Path $VenvDir)) {
    throw "Missing .venv. Run install.ps1 first."
}

. (Join-Path $VenvDir "Scripts\\Activate.ps1")
$env:LOKY_MAX_CPU_COUNT = [string][Environment]::ProcessorCount
python (Join-Path $RootDir "scripts\\start_server.py")

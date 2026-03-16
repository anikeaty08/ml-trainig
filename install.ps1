$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $RootDir ".venv"
$AppOrigin = "http://127.0.0.1:38475"

function Resolve-Python {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return "py -3"
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return "python"
    }
    throw "Python 3.10+ is required."
}

function Resolve-Npm {
    if (Get-Command npm.cmd -ErrorAction SilentlyContinue) {
        return "npm.cmd"
    }
    if (Get-Command npm -ErrorAction SilentlyContinue) {
        return "npm"
    }
    throw "npm is required."
}

$PythonCmd = Resolve-Python
$NpmCmd = Resolve-Npm

if (-not (Test-Path $VenvDir)) {
    Invoke-Expression "$PythonCmd -m venv `"$VenvDir`""
}

$ActivateScript = Join-Path $VenvDir "Scripts\\Activate.ps1"
. $ActivateScript

python -m pip install --upgrade pip
python -m pip install -r (Join-Path $RootDir "requirements.txt")
python -m pip install -r (Join-Path $RootDir "requirements-full.txt")
python (Join-Path $RootDir "scripts\\download_models.py")
python (Join-Path $RootDir "scripts\\init_db.py")

Push-Location (Join-Path $RootDir "frontend")
& $NpmCmd install
& $NpmCmd run build
Pop-Location

$ServerScript = Join-Path $RootDir "scripts\\start_server.py"
Start-Process python -ArgumentList "`"$ServerScript`""
Start-Sleep -Seconds 5
Start-Process $AppOrigin

Write-Host "ML Pipeline Agent ready at $AppOrigin"

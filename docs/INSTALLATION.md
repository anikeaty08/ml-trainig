# Installation

## Quick start

### macOS / Linux

```bash
chmod +x install.sh
./install.sh
```

### Windows

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\install.ps1
```

## What the installer does

1. Creates `.venv`
2. Installs Python dependencies
3. Generates lightweight built-in models locally
4. Initializes SQLite
5. Installs frontend dependencies
6. Builds the React frontend
7. Starts backend and frontend servers

## Start later

### macOS / Linux

```bash
./start.sh
```

### Windows

```powershell
.\start.ps1
```

## Requirements

- Python 3.10+
- Node.js 18+
- Enough disk for Python packages, frontend dependencies, and generated artifacts

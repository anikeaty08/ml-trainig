# ML Pipeline Agent

Local-first autonomous ML pipeline for messy datasets. Download it, run one install command, upload a CSV, and the app detects whether the job is tabular, text-heavy, or time-series, then cleans the data, trains multiple models, compares them, and gives you downloadable artifacts plus a terminal-style agent console for provider/model selection.

## What is included

- FastAPI backend with local job queue, SQLite tracking, WebSocket progress, and artifact downloads
- React frontend with direct upload flow, job history, model comparison, and agent console
- Autonomous CSV cleaning, target inference, dataset fitness analysis, and multi-model training
- Broader autonomous modes for tabular datasets, text-in-CSV workflows, and time-series forecasting from timestamped CSVs
- OpenClaw-style provider-first LLM selection with `provider/model` identifiers and optional fallback chains
- One-command setup scripts for macOS, Linux, and Windows

## Quick start

### macOS / Linux

```bash
git clone https://github.com/your-username/ml-pipeline-agent.git
cd ml-pipeline-agent
chmod +x install.sh
./install.sh
```

### Windows

Double-click `install.bat` or run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\install.ps1
```

The installer:

1. Creates a local Python virtual environment
2. Installs Python dependencies
3. Builds lightweight built-in baseline models locally
4. Installs frontend dependencies and builds the React app
5. Initializes SQLite
6. Starts the backend on `http://localhost:8000`
7. Starts the frontend on `http://localhost:3000`
8. Opens your browser automatically

## Current MVP scope

- Fully implemented: tabular CSV/TSV datasets, text-in-CSV datasets, and timestamped CSV forecasting datasets
- Included models: logistic regression, random forest, extra trees, gradient boosting, histogram gradient boosting, SVM, KNN, naive Bayes, ridge/linear regression, and voting ensemble
- Real-time job progress over WebSocket
- Downloadable model, cleaned dataset, generated training code, HTML report, predictions, and comparison CSV
- Optional local/cloud-style provider settings for the agent console, including primary plus fallback model refs

## Notes

- No login is required for the local app.
- User data stays inside `backend/data/`.
- The agent console supports local providers first, with optional API-key-based providers if the user chooses.
- Browser-login style auth is represented in the model-selection flow, but the MVP prioritizes local endpoints and API-key flows.

# ML Pipeline Agent

Local-first autonomous ML pipeline for messy datasets. Download it, run one install command, open the local app on `http://127.0.0.1:38475`, choose what packs you want on this machine, and start training. No app login. Everything stays on the device.

## What is included

- FastAPI backend with local job queue, SQLite tracking, WebSocket progress, setup state, and artifact downloads
- React frontend with a first-run setup wizard, download manager, direct upload flow, job history, model comparison, and agent console
- Autonomous local training for tabular, text, time-series, image-archive, and audio-archive workflows
- Device-local pack manager for install profiles such as `Core`, `Analyst`, `Vision`, `Research`, and `Full`
- OpenClaw-style provider-first agent routing with `provider/model` identifiers, fallback chains, allowlists, and local auth profiles
- Terminal setup wizard plus browser setup wizard
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
6. Starts the app on `http://127.0.0.1:38475`
7. Opens your browser automatically
8. Lets you choose packs and providers inside the setup wizard

## Start after install

### macOS / Linux

```bash
./start.sh
```

### Windows

Double-click `start.bat` or run:

```powershell
.\start.ps1
```

## Current product flow

1. Install and start the app
2. Complete the no-login setup wizard
3. Choose a setup profile and local packs
4. Configure optional agent providers
5. Upload a local dataset or paste a dataset URL / Kaggle link
6. Watch the app detect the modality, clean data, and train multiple models with live feedback
7. Review rankings, diagnostics, and exports

## Setup wizard

- `Core`: fastest install, tabular-first
- `Analyst`: tabular + boosting + time series classic + text classic
- `Vision`: tabular + boosting + transfer-learning image pack
- `Research`: broad local ML/DL stack without every heavy transformer pack
- `Full`: all declared packs

The terminal flow also supports setup on first start:

```bash
./terminal.sh
```

or on Windows:

```powershell
.\terminal.ps1
```

If no setup profile has been saved yet, the terminal will offer the setup wizard automatically.

## Model families tracked by the app

- Tabular classification/regression: Logistic Regression, Random Forest, Extra Trees, Gradient Boosting, Histogram Gradient Boosting, SVM, KNN, Naive Bayes, Ridge, Lasso, MLP, XGBoost, LightGBM, Voting Ensemble
- Text: TF-IDF classical pipelines plus transformer-target packs
- Time series: lag models with room for ARIMA, Prophet, LSTM, and smoothing packs
- Images: handcrafted feature flow now, transfer-learning packs tracked through the download manager
- Audio: spectrogram-feature flow now, richer audio packs tracked through the download manager

## Downloads and packs

Open the local download manager in the app to:

- see installed packs
- install deferred packs later
- remove packs
- review approximate storage use
- match packs to the recommended local profile

## Notes

- No login is required for the local app.
- User data and setup state stay inside `backend/data/`.
- The agent console supports local providers first, with optional API-key-based providers if the user chooses.
- Browser-login style auth is represented in the agent routing flow, while the app itself remains no-login.

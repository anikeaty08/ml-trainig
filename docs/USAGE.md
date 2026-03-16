# Usage

1. Open `http://127.0.0.1:38475`
2. Complete the setup wizard if this machine has not been configured yet
3. Choose a setup profile and packs
4. Configure optional agent providers if you want chat assistance
5. Upload a local dataset or paste a dataset URL / Kaggle link
6. Watch real-time progress with current model family and neural architecture details
7. Review rankings, diagnostics, and downloads

## Setup and downloads

- Setup wizard stores configuration on the device through the backend, not in browser localStorage
- Download manager shows installed packs, deferred packs, and approximate storage footprint
- Profiles include `Core`, `Analyst`, `Vision`, `Research`, and `Full`

## Agent console

- Provider-first routing
- Primary `provider/model` plus fallback chain
- Separate image-model route and optional allowlist
- Local auth profiles per provider with ordered rotation
- Settings persisted on the device through the local backend
- Supports local endpoints like Ollama and LM Studio first
- Supports onboarding commands, dataset summary commands, model scans, and local search

## Dataset modes

- Tabular: mixed numeric/categorical CSVs for classification or regression
- Text: CSVs with a strong text column plus a target label/value
- Time series: CSVs with a datetime column plus a numeric target
- Images: ZIP archives with class folders like `cats/*.png` and `dogs/*.png`
- Audio: ZIP archives with class folders that contain WAV-like files

## Terminal workflow

- Start with `./terminal.sh` or `.\terminal.ps1`
- If setup is incomplete, the terminal offers the setup wizard automatically
- Use `packs`, `pack install <pack_id>`, `pack remove <pack_id>`, `setup`, `train <path>`, and `/dataset summary`

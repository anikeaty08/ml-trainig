# Architecture

## Backend

- FastAPI for HTTP and WebSocket endpoints
- SQLite for local job and result tracking
- In-memory queue with a single background worker
- Device-local setup state and pack management through `backend/data/setup_state.json`
- Scikit-learn pipelines for autonomous tabular, text, image-feature, audio-feature, and lag-based time-series workflows
- Optional TensorFlow / Keras and PyTorch neural backends that join the comparison table when available

## Frontend

- React SPA with setup wizard, download manager, dashboard, upload, processing, results, and agent selector
- Agent provider settings and setup state stored on the device through backend APIs
- Live processing view shows current model family, training strategy, and neural architecture information when available

## Data flow

Setup -> choose profile/packs -> configure providers -> upload or fetch dataset -> detect -> clean -> analyze -> recommend -> train multiple families -> error analysis -> generate artifacts -> download

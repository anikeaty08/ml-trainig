# Architecture

## Backend

- FastAPI for HTTP and WebSocket endpoints
- SQLite for local job and result tracking
- In-memory queue with a single background worker
- Scikit-learn pipelines for autonomous CSV workflows

## Frontend

- React SPA with dashboard, upload, processing, results, and model selector
- Agent provider settings stored in browser localStorage

## Data flow

Upload -> detect -> clean -> analyze -> recommend -> train -> error analysis -> generate artifacts -> download

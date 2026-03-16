# Usage

1. Open `http://localhost:3000`
2. Configure the agent model provider if needed
3. Upload a CSV or TSV dataset
4. Watch real-time progress
5. Review the ranked model comparison and download artifacts

## Agent console

- Provider-first routing
- Primary `provider/model` plus fallback chain
- Settings persisted in browser localStorage
- Supports local endpoints like Ollama and LM Studio first

## Dataset modes

- Tabular: mixed numeric/categorical CSVs for classification or regression
- Text: CSVs with a strong text column plus a target label/value
- Time series: CSVs with a datetime column plus a numeric target

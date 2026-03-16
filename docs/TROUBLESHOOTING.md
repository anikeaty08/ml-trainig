# Troubleshooting

## Frontend build fails

- Confirm Node.js 18+ is installed
- Run `npm install` inside `frontend/`

## Backend import errors

- Activate `.venv`
- Run `python -m pip install -r requirements.txt`

## No remote models appear in the selector

- Confirm your local provider is running
- Check the base URL in the model selector
- Use `Fetch models` to probe the provider endpoint

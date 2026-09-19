# Contributing

Thanks for helping improve Predict & Segment.

## Development

```bash
python -m venv .venv
pip install -r requirements.txt
pytest -q
streamlit run app.py
```

## Pull requests

Please keep changes focused, explain the motivation, and add or update tests when behavior changes.

## Code quality

Prefer:
- small reusable functions
- clear names
- deterministic random seeds
- validation at data boundaries
- tests for model logic

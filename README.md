# SDV Credit Union Demo

Run a 6-step Streamlit flow for multi-table credit union data:

1. Load CSV/Parquet tables or the bundled sample.
2. Detect or upload SDV metadata.
3. Configure and apply simple constraints.
4. Train or upload an `HMASynthesizer` model.
5. Generate and download synthetic tables.
6. Evaluate real vs synthetic data with SDMetrics.

## Getting started

```bash
pip install -r requirements.txt
streamlit run app.py
```

The sample data lives in `data/` and includes members, accounts, and transactions tables with relationships detected automatically.

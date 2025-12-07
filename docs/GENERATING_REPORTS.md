# Generating Reports

This document shows how to reproduce the notebook's visual and interactive
reports from the command line using the included helper script
`scripts/generate_report.py`.

1. (Optional) Install parquet engines to enable Parquet read/write. If you want
   Parquet support, install (conda recommended):

```bash
conda install -c conda-forge pyarrow fastparquet
# or with pip:
pip install pyarrow fastparquet
```

2. Install Python requirements (or your environment of choice):

```bash
pip install -r requirements.txt
```

3. Run the report generator (from repository root):

```bash
python scripts/generate_report.py
```

Outputs will be written to `reports/visualizations/` and `reports/interactive/`.
A top-level index is available at `reports/index.html`.

## Notes

- The script is intentionally simple: it calls the project's ingestion and
  feature-engineering modules, creates a few representative PNGs and one
  interactive HTML, and writes them under `reports/`.
- If you prefer the notebook workflow, the notebook `notebooks/01_data_exploration.ipynb`
  contains the more extensive plotting and report-generation pipeline used
  during development.

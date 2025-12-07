#!/usr/bin/env python3
"""Pipeline runner: fetch data, build features, run experiment, backtest, generate report."""

import subprocess
import sys
import os
from pathlib import Path

SCRIPTS = [
    "fetch_data.py",
    "build_features.py",
    "run_experiment.py",
    "backtest_batch.py",
    "generate_report.py",
]

SCRIPT_DIR = Path(__file__).parent


def run_script(script):
    print(f"Running {script}...")
    result = subprocess.run([sys.executable, str(SCRIPT_DIR / script)])
    if result.returncode != 0:
        print(f"Error running {script}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    for script in SCRIPTS:
        run_script(script)
    print("Pipeline complete.")

"""
Shared utilities for the six-week ML project.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]
BINARY_FEATURES = ["sex", "fbs", "exang"]

def load_data():
    """Download the UCI Heart Disease dataset and normalize column names."""
    heart = fetch_ucirepo(id=45)
    X = heart.data.features.copy()
    y = heart.data.targets.copy()

    # Normalize missing-value markers and column names.
    X = X.replace("?", np.nan)
    X.columns = [str(c).strip().lower() for c in X.columns]
    y = y.iloc[:, 0].replace("?", np.nan)

    # Convert all feature columns to numeric where possible.
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    y = pd.to_numeric(y, errors="coerce")
    mask = y.notna()
    X, y = X.loc[mask].copy(), y.loc[mask].astype(int)

    # UCI target: 0 = no disease; 1-4 = disease presence.
    y = (y > 0).astype(int)

    # Use the commonly used 13-feature subset if available.
    desired = [
        "age","sex","cp","trestbps","chol","fbs","restecg",
        "thalach","exang","oldpeak","slope","ca","thal"
    ]
    if all(c in X.columns for c in desired):
        X = X[desired]

    # 'ca' is categorical/discrete in the source dataset.
    if "ca" in X.columns and "ca" not in CATEGORICAL_FEATURES:
        pass

    return X, y

def split_data(X, y):
    return train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

def save_json(data, filename):
    (RESULTS / filename).write_text(
        json.dumps(data, indent=2, default=str), encoding="utf-8"
    )

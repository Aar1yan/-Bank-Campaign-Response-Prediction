"""Trains and serializes the final model.

Run from the project root as:

    python -m src.train

Uses the exact hyperparameters and feature policy already selected in
notebooks 05-07 - no search, no tuning, no test-set involvement.
"""

import json
import warnings

import joblib
import pandas as pd

from src import config

# `campaign`'s natural right-skew produces a few large scaled values that trigger
# harmless RuntimeWarnings during the solver's optimization steps - investigated
# and confirmed benign in notebook 05 (the model still converges to a stable solution)
warnings.filterwarnings("ignore", category=RuntimeWarning)
from src.pipeline import build_pipeline


def load_training_data():
    """Reproduces the same chronological train split used in notebooks 05-07.

    Returns raw, not-yet-engineered feature columns - the pipeline's own
    feature-engineering step (src/pipeline.py) handles that internally.
    """
    df = pd.read_csv(config.CLEANED_DATA_PATH)

    train_size = int(len(df) * config.TRAIN_SPLIT_RATIO)
    train_df = df.iloc[:train_size]

    feature_columns = [
        col for col in train_df.columns
        if col not in [config.TARGET_COLUMN] + config.DROPPED_FEATURES
    ]
    X_train = train_df[feature_columns]
    y_train = (train_df[config.TARGET_COLUMN] == config.POSITIVE_CLASS).astype(int)

    return X_train, y_train


def train_and_save():
    X_train, y_train = load_training_data()

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.MODEL_PATH)

    with open(config.THRESHOLD_PATH, "w") as f:
        json.dump({"threshold": config.SELECTED_THRESHOLD}, f, indent=2)

    print(f"Trained on {len(X_train)} rows")
    print(f"Saved pipeline to {config.MODEL_PATH}")
    print(f"Saved threshold ({config.SELECTED_THRESHOLD}) to {config.THRESHOLD_PATH}")

    return pipeline


if __name__ == "__main__":
    train_and_save()

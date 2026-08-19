"""Reusable prediction logic - the only place the Streamlit app should call into.

    customer_data (raw input, no `y`, no `duration`)
        -> saved pipeline (feature engineering + preprocessing + model, all in one)
        -> probability
        -> threshold
        -> prediction

The saved pipeline is fully self-contained (src/pipeline.py), so this module
does not repeat any feature-engineering or preprocessing logic - it only
loads the artifact and applies the threshold.
"""

import json

import joblib

from src import config

_pipeline = None
_threshold = None


def load_pipeline():
    """Loads the saved pipeline once and reuses it on later calls."""
    global _pipeline
    if _pipeline is None:
        _pipeline = joblib.load(config.MODEL_PATH)
    return _pipeline


def load_threshold():
    global _threshold
    if _threshold is None:
        with open(config.THRESHOLD_PATH) as f:
            _threshold = json.load(f)["threshold"]
    return _threshold


def validate_customer_data(customer_data):
    """Rejects clearly impossible values before they reach the pipeline.

    A model will still return a number for nonsense input (e.g. age = -5) -
    that number isn't meaningful, so it's rejected here instead.
    """
    if "age" in customer_data.columns and (customer_data["age"] <= 0).any():
        raise ValueError("age must be greater than 0")
    if "campaign" in customer_data.columns and (customer_data["campaign"] < 1).any():
        raise ValueError("campaign must be >= 1 (it counts this planned call too)")
    if "previous" in customer_data.columns and (customer_data["previous"] < 0).any():
        raise ValueError("previous must be >= 0")


def predict_probability(customer_data):
    """customer_data: a DataFrame of one or more raw customer records.

    Must contain the same pre-call columns as the cleaned training data
    (age, job, campaign, education, ... ) - never `y`, never `duration`.
    Returns the predicted probability of subscribing for each row.
    """
    validate_customer_data(customer_data)
    pipeline = load_pipeline()
    return pipeline.predict_proba(customer_data)[:, 1]


def predict(customer_data, threshold=None):
    """Returns 1 (yes) / 0 (no) predictions using the given or saved threshold."""
    if threshold is None:
        threshold = load_threshold()
    probabilities = predict_probability(customer_data)
    return (probabilities >= threshold).astype(int)

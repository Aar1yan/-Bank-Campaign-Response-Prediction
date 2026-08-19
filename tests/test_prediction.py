"""Tests for src/predict.py - model loading, prediction behavior, and the
guarantees the saved production pipeline is supposed to provide.
"""

import joblib
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config, predict

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def test_model_file_exists():
    assert config.MODEL_PATH.exists(), (
        f"Expected a trained model at {config.MODEL_PATH}. Run `python -m src.train` first."
    )


def test_model_loads_as_pipeline():
    """Checked by stage count/type, not exact step names, so this doesn't
    break if a step is renamed - it still catches a pipeline that's missing
    its preprocessing or classifier stage.
    """
    pipeline = predict.load_pipeline()
    assert isinstance(pipeline, Pipeline)

    assert len(pipeline.steps) == 3
    step_types = [type(transformer) for _, transformer in pipeline.steps]
    assert ColumnTransformer in step_types
    assert LogisticRegression in step_types


# ---------------------------------------------------------------------------
# Threshold loading
# ---------------------------------------------------------------------------


def test_threshold_file_exists():
    assert config.THRESHOLD_PATH.exists(), (
        f"Expected a threshold file at {config.THRESHOLD_PATH}. Run `python -m src.train` first."
    )


def test_threshold_is_a_valid_probability():
    threshold = predict.load_threshold()
    assert isinstance(threshold, (int, float))
    assert 0 < threshold < 1


# ---------------------------------------------------------------------------
# Valid prediction
# ---------------------------------------------------------------------------


def test_valid_customer_produces_probability_between_0_and_1(valid_customer_df):
    probability = predict.predict_probability(valid_customer_df)[0]
    assert 0.0 <= probability <= 1.0


def test_valid_customer_produces_valid_prediction_label(valid_customer_df):
    prediction = predict.predict(valid_customer_df)[0]
    assert prediction in (0, 1)


# ---------------------------------------------------------------------------
# Probability / threshold consistency
# ---------------------------------------------------------------------------


def test_prediction_agrees_with_threshold(valid_customer_df):
    threshold = predict.load_threshold()
    probability = predict.predict_probability(valid_customer_df)[0]
    prediction = predict.predict(valid_customer_df, threshold=threshold)[0]

    expected = 1 if probability >= threshold else 0
    assert prediction == expected


# ---------------------------------------------------------------------------
# Serialization consistency
# ---------------------------------------------------------------------------


def test_prediction_consistent_before_and_after_serialization(tmp_path, valid_customer_df):
    """Fits the exact same pipeline/training data as src/train.py (no new ML
    decisions), predicts with it in memory, then saves + reloads it and
    predicts again. The saved artifact must behave like the pipeline that
    produced it.
    """
    from src.pipeline import build_pipeline
    from src.train import load_training_data

    X_train, y_train = load_training_data()
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    probability_before = pipeline.predict_proba(valid_customer_df)[:, 1][0]

    saved_path = tmp_path / "test_model.joblib"
    joblib.dump(pipeline, saved_path)
    reloaded_pipeline = joblib.load(saved_path)
    probability_after = reloaded_pipeline.predict_proba(valid_customer_df)[:, 1][0]

    assert probability_after == pytest.approx(probability_before, abs=1e-9)


# ---------------------------------------------------------------------------
# duration / target protection
# ---------------------------------------------------------------------------


def test_prediction_does_not_require_duration_or_target(valid_customer_dict):
    """`duration` was excluded for leakage (notebook 03) and `y` is the label,
    not an input - the raw record used across this file already omits both;
    this test makes that guarantee explicit.
    """
    assert "duration" not in valid_customer_dict
    assert "y" not in valid_customer_dict

    df = pd.DataFrame([valid_customer_dict])
    probability = predict.predict_probability(df)[0]
    assert 0.0 <= probability <= 1.0


# ---------------------------------------------------------------------------
# Missing input
# ---------------------------------------------------------------------------


def test_missing_required_column_raises_clear_error(valid_customer_dict):
    incomplete = valid_customer_dict.copy()
    del incomplete["job"]
    df = pd.DataFrame([incomplete])

    with pytest.raises(ValueError, match="job"):
        predict.predict_probability(df)


# ---------------------------------------------------------------------------
# Invalid numeric input - rejected by predict.validate_customer_data
# ---------------------------------------------------------------------------


def test_negative_age_is_rejected(valid_customer_dict):
    customer = valid_customer_dict.copy()
    customer["age"] = -5
    df = pd.DataFrame([customer])

    with pytest.raises(ValueError, match="age"):
        predict.predict_probability(df)


def test_campaign_below_1_is_rejected(valid_customer_dict):
    customer = valid_customer_dict.copy()
    customer["campaign"] = -1
    df = pd.DataFrame([customer])

    with pytest.raises(ValueError, match="campaign"):
        predict.predict_probability(df)


def test_negative_previous_is_rejected(valid_customer_dict):
    customer = valid_customer_dict.copy()
    customer["previous"] = -2
    df = pd.DataFrame([customer])

    with pytest.raises(ValueError, match="previous"):
        predict.predict_probability(df)


# ---------------------------------------------------------------------------
# Invalid categorical input - the pipeline has no manual category
# validation, so this documents its real, intentional behavior instead:
# OneHotEncoder's handle_unknown="ignore" (src/data_preprocessing.py)
# silently zero-encodes a category it never saw in training, rather than
# crashing or returning nonsense.
# ---------------------------------------------------------------------------


def test_unseen_category_still_produces_a_valid_probability(valid_customer_dict):
    customer = valid_customer_dict.copy()
    customer["job"] = "astronaut"  # not a category the encoder ever saw
    df = pd.DataFrame([customer])

    probability = predict.predict_probability(df)[0]
    assert 0.0 <= probability <= 1.0

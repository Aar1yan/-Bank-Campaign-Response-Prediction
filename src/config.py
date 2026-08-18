"""Project-wide paths, constants, and configuration.

Centralizing these here means every script (train.py, predict.py, evaluate.py)
references the same paths and settings instead of hard-coding them separately.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "bank-additional-full.csv"
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "bank_marketing_cleaned.csv"
FEATURES_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "bank_marketing_features.csv"

# Model artifacts
MODEL_PATH = PROJECT_ROOT / "models" / "final_model.joblib"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "threshold.json"

# Reproducibility
RANDOM_STATE = 42

# Same chronological 80/20 split used in notebooks 05-08 (no shuffling - the
# data is time-ordered, so a random split would leak future information into training)
TRAIN_SPLIT_RATIO = 0.8

# Target and feature policy (notebook 03: `duration` is unavailable before the
# call takes place, so it is never part of the model's input features)
TARGET_COLUMN = "y"
POSITIVE_CLASS = "yes"
DROPPED_FEATURES = ["duration"]

# Columns the ColumnTransformer selects, AFTER the pipeline's feature-engineering
# step has run (notebooks 05-07) - this is why some names are engineered
# (age_group, education_grouped, ...) while others are raw (age, education, ...);
# both the raw and engineered versions are deliberately kept side by side.
# `pdays` is intentionally excluded: its 999 sentinel value dominates the
# column and destabilizes StandardScaler (investigated in notebook 05); its
# useful signal is already captured by `was_previously_contacted`.
NUMERIC_FEATURES = [
    "age",
    "campaign",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
    "contacts_before_this_call",
    "was_previously_contacted",
]
CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
    "age_group",
    "education_grouped",
]

# Final model (notebook 06 tuning confirmed these are the best configuration
# found - they match scikit-learn's defaults)
LOGISTIC_REGRESSION_PARAMS = {
    "C": 1,
    "class_weight": None,
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
}

# Operating threshold selected in notebook 07 via out-of-fold cross-validation
# on training data only (never re-optimized here)
SELECTED_THRESHOLD = 0.15

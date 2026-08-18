"""Reusable preprocessing logic - scaling and encoding, nothing fitted at import time.

`build_preprocessor()` returns an unfitted ColumnTransformer. It only learns
anything (means, standard deviations, category lists) when `.fit()` is called
as part of the full pipeline on training data - never on the whole dataset.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor():
    """Numeric features are scaled; categorical features are one-hot encoded.

    `handle_unknown='ignore'` protects prediction time against a category the
    encoder never saw during training, instead of raising an error.
    """
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])

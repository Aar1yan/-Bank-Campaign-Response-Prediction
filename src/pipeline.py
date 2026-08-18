"""Assembles the complete, self-contained production pipeline.

This is the one object that gets trained and serialized to
models/final_model.joblib - raw customer columns in, subscription
probability out:

    Raw input -> feature engineering -> preprocessing -> Logistic Regression

`FunctionTransformer` wraps the plain `add_engineered_features` function
(src/feature_engineering.py) so it can be a pipeline step like any other
scikit-learn transformer, without writing a custom transformer class.
Because it's inside the pipeline, the exact same feature-engineering code
runs during both training (src/train.py) and prediction (src/predict.py) -
there's no second copy of this logic anywhere.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.config import LOGISTIC_REGRESSION_PARAMS
from src.data_preprocessing import build_preprocessor
from src.feature_engineering import add_engineered_features


def build_pipeline():
    """Returns an unfitted Pipeline: feature engineering -> ColumnTransformer -> LogisticRegression."""
    return Pipeline([
        ("feature_engineering", FunctionTransformer(add_engineered_features)),
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)),
    ])

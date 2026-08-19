"""Shared fixtures for the test suite."""

import pandas as pd
import pytest


@pytest.fixture
def valid_customer_dict():
    """One representative, raw customer record - the exact raw columns the
    production pipeline expects. No `y`, no `duration` (notebook 03: `duration`
    is unknown before the call happens and is never a model input).
    """
    return {
        "age": 35,
        "job": "admin.",
        "marital": "married",
        "education": "university.degree",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "mon",
        "campaign": 1,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp.var.rate": 1.1,
        "cons.price.idx": 93.994,
        "cons.conf.idx": -36.4,
        "euribor3m": 4.857,
        "nr.employed": 5191.0,
    }


@pytest.fixture
def valid_customer_df(valid_customer_dict):
    """The same record as a one-row DataFrame - what predict.py expects."""
    return pd.DataFrame([valid_customer_dict])

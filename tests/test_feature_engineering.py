"""Tests for src/feature_engineering.py - the notebook 04 feature logic that
must behave identically at training time and prediction time.
"""

import pandas as pd

from src.feature_engineering import add_engineered_features


def test_adds_the_four_notebook_04_features(valid_customer_dict):
    df = pd.DataFrame([valid_customer_dict])
    engineered = add_engineered_features(df)

    for column in [
        "was_previously_contacted", "age_group",
        "contacts_before_this_call", "education_grouped",
    ]:
        assert column in engineered.columns


def test_was_previously_contacted_reflects_previous_count():
    df = pd.DataFrame([
        {"age": 35, "campaign": 2, "previous": 0, "education": "university.degree"},
        {"age": 35, "campaign": 2, "previous": 3, "education": "university.degree"},
    ])
    engineered = add_engineered_features(df)
    assert engineered["was_previously_contacted"].tolist() == [0, 1]


def test_contacts_before_this_call_excludes_the_planned_call():
    # `campaign` counts the not-yet-made call too (notebook 03), so 4 contacts
    # this campaign means only 3 were known before dialing
    df = pd.DataFrame([{"age": 35, "campaign": 4, "previous": 0, "education": "university.degree"}])
    engineered = add_engineered_features(df)
    assert engineered["contacts_before_this_call"].iloc[0] == 3


def test_illiterate_is_grouped_into_basic_4y():
    df = pd.DataFrame([{"age": 35, "campaign": 1, "previous": 0, "education": "illiterate"}])
    engineered = add_engineered_features(df)
    assert engineered["education_grouped"].iloc[0] == "basic.4y"


def test_does_not_modify_input_in_place():
    df = pd.DataFrame([{"age": 35, "campaign": 1, "previous": 0, "education": "university.degree"}])
    original_columns = list(df.columns)
    add_engineered_features(df)
    assert list(df.columns) == original_columns


def test_engineered_features_do_not_require_y_or_duration(valid_customer_dict):
    """valid_customer_dict has no y or duration - if this raises, feature
    engineering would secretly depend on one of them.
    """
    assert "y" not in valid_customer_dict
    assert "duration" not in valid_customer_dict

    df = pd.DataFrame([valid_customer_dict])
    add_engineered_features(df)  # should not raise

"""Feature engineering decisions established in notebook 04.

`add_engineered_features` is the single source of truth for these transformations -
it is used both to build the training data and to transform new customer input
at prediction time, so the two can never drift apart.
"""

import pandas as pd

AGE_BINS = [17, 25, 35, 45, 55, 65, 99]
AGE_LABELS = ["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]


def add_engineered_features(df):
    """Adds the four notebook 04 features to a copy of df. Does not modify df in place."""
    df = df.copy()

    # Reliable "contacted before" flag: `previous` agrees with `poutcome` on every
    # row, unlike the raw `pdays == 999` sentinel (notebook 04 investigation)
    df["was_previously_contacted"] = (df["previous"] > 0).astype(int)

    # Age shows a U-shaped relationship with the target (notebook 04 EDA), which a
    # single continuous value can't represent - kept alongside `age`, not instead of it
    df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS, include_lowest=True)

    # `campaign` counts the current, not-yet-made contact (notebook 03); this reflects
    # only contacts truly known before dialing
    df["contacts_before_this_call"] = df["campaign"] - 1

    # `illiterate` is too small a category (18 rows) to estimate reliably on its own
    df["education_grouped"] = df["education"].replace({"illiterate": "basic.4y"})

    return df

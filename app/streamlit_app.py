"""Streamlit application.

UI only - all ML logic (feature engineering, preprocessing, the model, and
the decision threshold) lives in models/final_model.joblib, loaded and
applied through src/predict.py. This file never touches that logic
directly, only calls into src/predict.py.
"""

import sys
from pathlib import Path

# `streamlit run app/streamlit_app.py` only puts this file's own folder (app/)
# on sys.path, not the project root - add it explicitly so `from src import ...` works.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from src import predict

st.set_page_config(page_title="Term Deposit Subscription Predictor", page_icon="📞")

MONTH_OPTIONS = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
DAY_OPTIONS = ["mon", "tue", "wed", "thu", "fri"]
POUTCOME_OPTIONS = ["nonexistent", "failure", "success"]
YES_NO_UNKNOWN = ["no", "unknown", "yes"]
JOB_OPTIONS = ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
               "self-employed", "services", "student", "technician", "unemployed", "unknown"]
EDUCATION_OPTIONS = ["basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
                     "professional.course", "university.degree", "unknown"]


@st.cache_resource
def get_pipeline():
    return predict.load_pipeline()


@st.cache_resource
def get_threshold():
    return predict.load_threshold()


st.title("Term Deposit Subscription Predictor")
st.caption(
    "Estimates the probability that a bank customer will subscribe to a term deposit, "
    "using only information available immediately before a planned marketing call."
)

try:
    pipeline = get_pipeline()
    threshold = get_threshold()
except Exception:
    st.error(
        "Model files not found. Train and save the production pipeline first by running "
        "`python -m src.train` from the project root, then restart this app."
    )
    st.stop()

with st.form("customer_form"):
    st.subheader("Customer Information")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=17, max_value=98, value=40, step=1)
        job = st.selectbox("Job", JOB_OPTIONS)
        marital = st.selectbox("Marital status", ["married", "single", "divorced", "unknown"])
        education = st.selectbox("Education", EDUCATION_OPTIONS)
    with col2:
        default = st.selectbox("Has credit in default?", YES_NO_UNKNOWN)
        housing = st.selectbox("Has housing loan?", YES_NO_UNKNOWN)
        loan = st.selectbox("Has personal loan?", YES_NO_UNKNOWN)

    st.subheader("Contact Information")
    col3, col4, col5 = st.columns(3)
    with col3:
        contact = st.selectbox("Contact method", ["cellular", "telephone"])
    with col4:
        month = st.selectbox("Planned contact month", MONTH_OPTIONS)
    with col5:
        day_of_week = st.selectbox("Planned contact day", DAY_OPTIONS)

    st.subheader("Campaign Information")
    col6, col7, col8 = st.columns(3)
    with col6:
        campaign = st.number_input(
            "Contact number this campaign (including this planned call)",
            min_value=1, max_value=60, value=1, step=1,
        )
    with col7:
        previous = st.number_input(
            "Contacts before this campaign", min_value=0, max_value=10, value=0, step=1,
        )
    with col8:
        poutcome = st.selectbox("Previous campaign outcome", POUTCOME_OPTIONS)

    st.subheader("Economic / Context Information")
    st.caption(
        "Macroeconomic indicators around the time of contact - these describe the "
        "broader economy, not this specific customer."
    )
    col9, col10 = st.columns(2)
    with col9:
        emp_var_rate = st.number_input(
            "Employment variation rate", min_value=-3.4, max_value=1.4, value=1.1,
            step=0.1, format="%.1f",
        )
        cons_price_idx = st.number_input(
            "Consumer price index", min_value=92.2, max_value=94.8, value=93.9,
            step=0.01, format="%.3f",
        )
        cons_conf_idx = st.number_input(
            "Consumer confidence index", min_value=-50.8, max_value=-26.9, value=-36.4,
            step=0.1, format="%.1f",
        )
    with col10:
        euribor3m = st.number_input(
            "Euribor 3-month rate", min_value=0.6, max_value=5.1, value=4.86,
            step=0.01, format="%.3f",
        )
        nr_employed = st.number_input(
            "Number of employees economy-wide (thousands)", min_value=4963.6,
            max_value=5228.1, value=5191.0, step=0.1, format="%.1f",
        )

    submitted = st.form_submit_button("Predict Subscription Probability")

if submitted:
    customer = pd.DataFrame([{
        "age": age, "job": job, "marital": marital, "education": education,
        "default": default, "housing": housing, "loan": loan,
        "contact": contact, "month": month, "day_of_week": day_of_week,
        "campaign": campaign, "previous": previous, "poutcome": poutcome,
        "emp.var.rate": emp_var_rate, "cons.price.idx": cons_price_idx,
        "cons.conf.idx": cons_conf_idx, "euribor3m": euribor3m, "nr.employed": nr_employed,
    }])

    try:
        probability = predict.predict_probability(customer)[0]
        prediction_label = predict.predict(customer, threshold=threshold)[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    prediction = "Likely to Subscribe" if prediction_label == 1 else "Unlikely to Subscribe"

    st.subheader("Prediction Result")

    result_col1, result_col2 = st.columns(2)
    with result_col1:
        st.metric("Estimated Subscription Probability", f"{probability:.1%}")
    with result_col2:
        st.metric("Decision Threshold", f"{threshold:.0%}")

    if prediction_label == 1:
        st.success(f"Prediction: {prediction}")
    else:
        st.warning(f"Prediction: {prediction}")

    st.write(
        f"This customer is classified as **{prediction.lower()}** based on the trained "
        f"model's estimated probability ({probability:.1%}) compared against the selected "
        f"decision threshold ({threshold:.0%}). This is a model-based estimate, not a "
        f"guarantee of actual customer behavior."
    )

st.divider()

with st.expander("About the Model"):
    st.markdown(
        f"- **Model type:** Logistic Regression\n"
        f"- **Prediction target:** Whether a customer subscribes to a term deposit\n"
        f"- **Historical dataset:** UCI Bank Marketing Dataset - a Portuguese bank's phone "
        f"campaigns, 2008-2010\n"
        f"- **Decision threshold:** {threshold:.0%}, selected via cross-validation on "
        f"training data\n"
        f"\n"
        f"Predictions are estimated probabilities based on patterns in historical data, "
        f"not guarantees of individual customer behavior."
    )

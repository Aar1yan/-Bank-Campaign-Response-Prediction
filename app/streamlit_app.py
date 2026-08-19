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

st.set_page_config(
    page_title="Term Deposit Subscription Predictor", page_icon="📞", layout="wide",
)

# Presentational only - typography, spacing, cards, and the hero/section
# styling below. The base color theme lives in .streamlit/config.toml; this
# just layers spacing/typography/card treatment on top. No widget logic here.
#
# `layout="wide"` above removes Streamlit's default "centered" mode (which caps
# .block-container at ~730px regardless of screen size). Wide mode itself has
# no max-width at all, so it's capped again here at ~90-95% of common desktop
# widths instead of stretching edge to edge.
st.markdown(
    """
    <style>
    html {
        font-size: 15px;
    }
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .block-container {
        width: 95% !important;
        max-width: 1900px !important;
        margin: 0 auto !important;
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
    }

    /* ---- Hero header ---- */
    .hero {
        position: relative;
        overflow: hidden;
        border-radius: 20px;
        padding: 3.5rem 3.5rem;
        margin-bottom: 2.5rem;
        border: 1px solid rgba(255, 255, 255, 0.06);
        background:
            radial-gradient(circle at 12% 20%, rgba(59, 91, 254, 0.28), transparent 45%),
            radial-gradient(circle at 88% 25%, rgba(139, 92, 246, 0.22), transparent 50%),
            linear-gradient(135deg, #0B1020 0%, #10162B 60%, #131A2E 100%);
    }
    .hero-inner {
        position: relative;
        z-index: 1;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 2rem;
        flex-wrap: wrap;
    }
    .hero-content { flex: 1 1 480px; max-width: 760px; }
    .hero-badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 999px;
        background: rgba(139, 92, 246, 0.15);
        border: 1px solid rgba(139, 92, 246, 0.4);
        color: #C4B5FD;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        margin-bottom: 1.25rem;
    }
    .hero-title-row {
        display: flex;
        align-items: center;
        gap: 1.25rem;
        margin-bottom: 1.1rem;
    }
    .hero-icon-box {
        flex-shrink: 0;
        width: 4.5rem;
        height: 4.5rem;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.1rem;
        background: rgba(59, 91, 254, 0.14);
        border: 1.5px solid rgba(96, 140, 255, 0.5);
        box-shadow: 0 0 24px rgba(59, 91, 254, 0.25);
    }
    .hero-title {
        font-size: 3.1rem;
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -0.01em;
        color: #F8FAFC;
        text-transform: uppercase;
        margin: 0;
    }
    .hero-subtitle {
        color: #A5B4CB;
        font-size: 1.2rem;
        line-height: 1.6;
        max-width: 660px;
        margin-bottom: 1.5rem;
    }
    .hero-pills { display: flex; flex-wrap: wrap; gap: 0.7rem; }
    .hero-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.5rem 1.05rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #CBD5E1;
        font-size: 0.95rem;
        font-weight: 500;
    }
    .hero-decoration {
        flex-shrink: 0;
        filter: drop-shadow(0 0 26px rgba(96, 140, 255, 0.35));
    }
    @media (max-width: 900px) {
        .hero { padding: 2rem; }
        .hero-title { font-size: 1.7rem; }
        .hero-decoration { display: none; }
    }

    /* ---- Section headings (icon badge + bold label), used inside cards ---- */
    .section-heading {
        display: flex;
        align-items: center;
        gap: 0.9rem;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid rgba(139, 92, 246, 0.25);
    }
    .section-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 3rem;
        height: 3rem;
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(59, 91, 254, 0.28), rgba(139, 92, 246, 0.28));
        border: 1px solid rgba(139, 92, 246, 0.4);
        font-size: 1.5rem;
    }
    .section-heading-text {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: 0.01em;
        text-transform: uppercase;
        color: #FFFFFF;
    }

    /* ---- Cards (bordered containers) ---- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px !important;
        border-color: rgba(255, 255, 255, 0.08) !important;
        padding: 0.5rem !important;
    }

    /* ---- Inputs ---- */
    [data-testid="stWidgetLabel"] p {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        color: #E2E8F0 !important;
    }
    .stNumberInput input, .stTextInput input {
        font-size: 1.15rem !important;
        border-radius: 10px !important;
        padding-top: 0.6rem !important;
        padding-bottom: 0.6rem !important;
    }
    div[data-baseweb="select"] > div {
        font-size: 1.15rem !important;
        border-radius: 10px !important;
        min-height: 2.9rem !important;
    }
    [data-testid="stCaptionContainer"] {
        font-size: 1.02rem !important;
    }

    /* ---- Primary button (predict) ---- */
    button[kind="primary"], button[kind="primaryFormSubmit"] {
        background: linear-gradient(90deg, #3B5BFE, #8B5CF6) !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 800 !important;
        font-size: 1.2rem !important;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        padding: 1.1rem 1.5rem !important;
        box-shadow: 0 8px 24px rgba(88, 80, 236, 0.35);
        transition: filter 0.15s ease;
    }
    button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover {
        filter: brightness(1.08);
    }

    /* ---- About the Model expander ---- */
    [data-testid="stExpander"] {
        border-radius: 14px !important;
        border-color: rgba(255, 255, 255, 0.08) !important;
    }
    [data-testid="stExpander"] summary p {
        font-size: 1.2rem !important;
        font-weight: 700 !important;
    }

    /* ---- Prediction result card ---- */
    .result-card {
        border-radius: 18px;
        padding: 2rem 2.25rem;
        margin-top: 0.5rem;
        border: 1px solid;
    }
    .result-positive {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.10), rgba(34, 197, 94, 0.02));
        border-color: rgba(34, 197, 94, 0.35);
    }
    .result-negative {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.10), rgba(239, 68, 68, 0.02));
        border-color: rgba(239, 68, 68, 0.35);
    }
    .result-stats {
        display: flex;
        flex-wrap: wrap;
        gap: 2.5rem;
        margin: 1.25rem 0 1.25rem 3rem;
    }
    .result-stat { min-width: 220px; }
    .result-stat-label {
        color: #94A3B8;
        font-size: 0.95rem;
        margin-bottom: 0.4rem;
    }
    .result-stat-value {
        font-size: 2.4rem;
        font-weight: 800;
    }
    .result-stat-value.neutral { color: #60A5FA; }
    .result-positive .result-stat-value.accent { color: #4ADE80; }
    .result-negative .result-stat-value.accent { color: #F87171; }
    .result-explainer {
        color: #94A3B8;
        font-size: 1rem;
        line-height: 1.6;
        margin-left: 3rem;
        max-width: 900px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def section_heading(icon, text):
    """Renders an icon-badge + bold label, styled consistently across every card."""
    st.markdown(
        f'<div class="section-heading">'
        f'<span class="section-icon">{icon}</span>'
        f'<span class="section-heading-text">{text}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


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


st.markdown(
    """
    <div class="hero">
      <div class="hero-inner">
        <div class="hero-content">
          <div class="hero-badge">AI POWERED</div>
          <div class="hero-title-row">
            <div class="hero-icon-box">📈</div>
            <div class="hero-title">Term Deposit<br>Subscription Predictor</div>
          </div>
          <div class="hero-subtitle">
            Estimate the probability that a bank customer will subscribe to a term deposit,
            using only information available immediately before a planned marketing call.
          </div>
          <div class="hero-pills">
            <span class="hero-pill">🌐 Machine Learning Model</span>
            <span class="hero-pill">📊 Data-Driven Insights</span>
            <span class="hero-pill">🎯 Smart Predictions</span>
          </div>
        </div>
        <svg class="hero-decoration" width="110" height="110" viewBox="0 0 24 24" fill="none" stroke-width="1.3">
          <defs>
            <linearGradient id="heroGrad" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stop-color="#60A5FA"/>
              <stop offset="100%" stop-color="#A78BFA"/>
            </linearGradient>
          </defs>
          <path d="M3 10l9-6 9 6" stroke="url(#heroGrad)" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M4 10v9M9 10v9M15 10v9M20 10v9" stroke="url(#heroGrad)" stroke-linecap="round"/>
          <path d="M2 21h20" stroke="url(#heroGrad)" stroke-linecap="round"/>
        </svg>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
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
    with st.container(border=True):
        section_heading("👤", "Customer Information")
        col1, col2, col3_cust = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=17, max_value=98, value=40, step=1)
            job = st.selectbox("Job", JOB_OPTIONS)
        with col2:
            marital = st.selectbox("Marital status", ["married", "single", "divorced", "unknown"])
            education = st.selectbox("Education", EDUCATION_OPTIONS)
        with col3_cust:
            default = st.selectbox("Has credit in default?", YES_NO_UNKNOWN)
            housing = st.selectbox("Has housing loan?", YES_NO_UNKNOWN)
            loan = st.selectbox("Has personal loan?", YES_NO_UNKNOWN)

    contact_col, campaign_col = st.columns(2)
    with contact_col:
        with st.container(border=True):
            section_heading("📞", "Contact Information")
            contact = st.selectbox("Contact method", ["cellular", "telephone"])
            month = st.selectbox("Planned contact month", MONTH_OPTIONS)
            day_of_week = st.selectbox("Planned contact day", DAY_OPTIONS)

    with campaign_col:
        with st.container(border=True):
            section_heading("📣", "Campaign Information")
            campaign = st.number_input(
                "Contact number this campaign (including this planned call)",
                min_value=1, max_value=60, value=1, step=1,
            )
            previous = st.number_input(
                "Contacts before this campaign", min_value=0, max_value=10, value=0, step=1,
            )
            poutcome = st.selectbox("Previous campaign outcome", POUTCOME_OPTIONS)

    with st.container(border=True):
        section_heading("🌐", "Economic / Context Information")
        st.caption(
            "Macroeconomic indicators around the time of contact - these describe the "
            "broader economy, not this specific customer."
        )
        col9, col10, col11 = st.columns(3)
        with col9:
            emp_var_rate = st.number_input(
                "Employment variation rate", min_value=-3.4, max_value=1.4, value=1.1,
                step=0.1, format="%.1f",
            )
            euribor3m = st.number_input(
                "Euribor 3-month rate", min_value=0.6, max_value=5.1, value=4.86,
                step=0.01, format="%.3f",
            )
        with col10:
            cons_price_idx = st.number_input(
                "Consumer price index", min_value=92.2, max_value=94.8, value=93.9,
                step=0.01, format="%.3f",
            )
            nr_employed = st.number_input(
                "Number of employees economy-wide (thousands)", min_value=4963.6,
                max_value=5228.1, value=5191.0, step=0.1, format="%.1f",
            )
        with col11:
            cons_conf_idx = st.number_input(
                "Consumer confidence index", min_value=-50.8, max_value=-26.9, value=-36.4,
                step=0.1, format="%.1f",
            )

    st.write("")
    submitted = st.form_submit_button(
        "📈 Predict Subscription Probability", type="primary", use_container_width=True,
    )

with st.expander("ℹ️ About the Model"):
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

    is_positive = prediction_label == 1
    prediction = "Likely to Subscribe" if is_positive else "Unlikely to Subscribe"
    result_class = "result-positive" if is_positive else "result-negative"
    mark = "✓" if is_positive else "✕"

    st.markdown(
        f"""
        <div class="result-card {result_class}">
          <div class="section-heading">
            <span class="section-icon">📈</span>
            <span class="section-heading-text">Prediction Result</span>
          </div>
          <div class="result-stats">
            <div class="result-stat">
              <div class="result-stat-label">Estimated Subscription Probability</div>
              <div class="result-stat-value accent">{probability:.1%}</div>
            </div>
            <div class="result-stat">
              <div class="result-stat-label">Decision Threshold</div>
              <div class="result-stat-value neutral">{threshold:.0%}</div>
            </div>
            <div class="result-stat">
              <div class="result-stat-label">Prediction</div>
              <div class="result-stat-value accent">{mark} {prediction}</div>
            </div>
          </div>
          <div class="result-explainer">
            This customer is classified as <strong>{prediction.lower()}</strong> based on the trained
            model's estimated probability ({probability:.1%}) compared against the selected
            decision threshold ({threshold:.0%}). This is a model-based estimate, not a
            guarantee of actual customer behavior.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

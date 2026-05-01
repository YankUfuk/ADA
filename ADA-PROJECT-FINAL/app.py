import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# file paths
APP_DIR = Path(__file__).resolve().parent
PIPELINE_FILE = APP_DIR / "rf_pipeline.sav"
DATA_FILE = APP_DIR / "bank-additional" / "bank-additional.csv"

# app setup
st.set_page_config(
    page_title="Bank Campaign Subscription Predictor",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0B0F19;
        color: #E5E7EB;
    }
    h1, h2, h3 {
        color: #F9FAFB;
    }
    p, label, span, div {
        color: #E5E7EB;
    }
    div[data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-left: 4px solid #3B82F6;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        color: #F9FAFB;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] div {
        color: #F9FAFB;
    }
    div[data-baseweb="select"] div,
    input,
    textarea {
        color: #111827;
    }
    div[data-baseweb="select"] span {
        color: #111827;
    }
    .stButton > button {
        background-color: #3B82F6;
        color: #FFFFFF;
        border-radius: 0.4rem;
        border: none;
    }
    .stTabs [data-baseweb="tab"] {
        color: #E5E7EB;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# load model
@st.cache_resource
def load_pipeline():
    with open(PIPELINE_FILE, "rb") as f:
        return pickle.load(f)

pipeline_model = load_pipeline()

# load dataset
@st.cache_data
def read_bank_data():
    return pd.read_csv(DATA_FILE, sep=";")

bank_data = read_bank_data()

st.title("Bank Campaign Subscription Predictor")
st.markdown(
    "This application provides a short exploration of the bank marketing dataset "
    "and estimates whether a customer is likely to subscribe to a term deposit"
)

overview_tab, prediction_tab = st.tabs(["Exploratory Data Analysis", "Customer Prediction"])

# eda tab
with overview_tab:
    st.header("Exploratory Data Analysis")

    total_records = len(bank_data)
    subscribed_count = int((bank_data["y"] == "yes").sum())
    not_subscribed_count = int((bank_data["y"] == "no").sum())

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Total Records", total_records)
    metric_col2.metric("Subscribed", subscribed_count)
    metric_col3.metric("Not Subscribed", not_subscribed_count)

    st.divider()

    st.subheader("Subscription Outcome Distribution")
    target_fig, target_ax = plt.subplots()
    bank_data["y"].value_counts().plot(kind="bar", ax=target_ax, color=["#2563EB", "#64748B"])
    target_ax.set_xlabel("Subscription")
    target_ax.set_ylabel("Count")
    target_ax.set_title("Target Class Distribution (y)")
    st.pyplot(target_fig)

    st.markdown("""
- The dataset is **imbalanced**.
- This justifies the use of **SMOTE** during training.
""")

    st.divider()

    if "duration" in bank_data.columns:
        st.subheader("Call Duration vs Target")
        duration_fig, duration_ax = plt.subplots()
        sns.histplot(data=bank_data, x="duration", hue="y", bins=40, ax=duration_ax)
        duration_ax.set_title("Duration vs Subscription Outcome")
        st.pyplot(duration_fig)

        st.warning("""
**Data Leakage Notice**

`duration` is only known **after** the call is completed.
It was excluded from model training so the prediction stays realistic.
""")

    st.divider()

    st.subheader("Numerical Feature Correlation")
    numeric_data = bank_data.select_dtypes(include=["number"])
    corr_fig, corr_ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_data.corr(), cmap="YlGnBu", ax=corr_ax)
    corr_ax.set_title("Feature Correlation Matrix")
    st.pyplot(corr_fig)

    st.divider()

    st.subheader("Sample Records")
    st.dataframe(bank_data.head())

# prediction tab
with prediction_tab:
    st.header("Customer Simulation & Prediction")

    customer_col, campaign_col, economic_col = st.columns(3)

    with customer_col:
        st.subheader("Customer Info")
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30,
            step=1
            )
        job = st.selectbox("Job", [
            "admin.","technician","management","blue-collar",
            "services","housemaid","retired","self-employed",
            "entrepreneur","unemployed","student"
        ])
        education = st.selectbox("Education", [
            "basic.4y","basic.6y","basic.9y",
            "high.school","professional.course",
            "university.degree","illiterate"
        ])
        campaign = st.number_input(
            "Campaign Calls",
            min_value=1,
            max_value=50,
            value=2,
            step=1
            )
        housing = st.selectbox("Housing Loan", ["yes", "no"])

    with campaign_col:
        st.subheader("Campaign Info")
        loan = st.selectbox("Personal Loan", ["yes", "no"])
        contact = st.selectbox("Contact Type", ["cellular", "telephone"])
        month = st.selectbox("Month", [
            "jan","feb","mar","apr","may","jun",
            "jul","aug","sep","oct","nov","dec"
        ])
        poutcome = st.selectbox("Previous Campaign Outcome", ["success", "failure", "unknown"])
        emp_var_rate = st.selectbox("Employment Variation Rate",
                                    [-3.4, -2.9, -2.1, -1.8, -1.1, 0.1, 1.1])

    with economic_col:
        st.subheader("Economic Info")
        cons_price_idx = st.selectbox("Consumer Price Index",
                                      [92.2, 92.8, 93.2, 93.9, 94.4])
        cons_conf_idx = st.selectbox("Consumer Confidence Index",
                                     [-50.8, -46.2, -42.0, -36.4])
        euribor3m = st.selectbox("Euribor 3M Rate",
                                 [0.7, 1.1, 1.3, 1.5, 2.0, 3.0, 4.0])
        nr_employed = st.selectbox("Number of Employees",
                                   [4963.6, 5008.7, 5076.2, 5099.1, 5228.1])
        pdays = st.number_input(
            "Days Since Last Contact (999 = never)",
            min_value=0,
            max_value=999,
            value=999,
            step=1
            )

        previous = st.number_input(
            "Previous Contacts",
            min_value=0,
            max_value=20,
            value=0,
            step=1
            )


    # build input row for prediction
    customer_input = pd.DataFrame([{
        "age": age,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "emp.var.rate": emp_var_rate,
        "cons.price.idx": cons_price_idx,
        "cons.conf.idx": cons_conf_idx,
        "euribor3m": euribor3m,
        "nr.employed": nr_employed,
        "education": education,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "poutcome": poutcome,
        "job": job,

        "has_university_degree": int(education == "university.degree"),
        "married": 0,
        "is_high_campaign": int(campaign > 3),
        "white_collar": int(job in ["admin.","technician","management"]),
        "blue_collar": int(job in ["blue-collar","services","housemaid"]),
        "other_collar": int(job in ["retired","self-employed","entrepreneur","unemployed","student"]),
        "econ_interact": euribor3m * emp_var_rate,
        "age_euribor_interact": age * euribor3m,
        "a_age_euribor_interact": age * euribor3m,
        "m_age_euribor_interact": age * euribor3m,
        "r_age_euribor_interact": age * euribor3m,
        "campaign_conf_interact": campaign * cons_conf_idx,
        "cpi_euribor_diff": cons_price_idx - euribor3m,
        "previous_contact": int(pdays != 999),
        "has_multiple_loans": int(housing == "yes" and loan == "yes"),
        "economic_stress": int(emp_var_rate < 0)
    }])

    st.divider()

    # show prediction result
    if st.button("Predict"):
        prediction = pipeline_model.predict(customer_input)[0]
        subscription_probability = pipeline_model.predict_proba(customer_input)[0][1]

        st.metric("Estimated Subscription Probability", f"{subscription_probability:.2%}")

        if prediction == 1:
            st.success("Prediction Subscribe")
        else:
            st.error("Prediction Not Subscribe")

        st.caption(
            "This result is based on the selected customer campaign and economic inputs"
        )


st.markdown("---")
st.caption("ADS 542 Statistical Learning | Final Project | Yankı Ufuk Dülger")

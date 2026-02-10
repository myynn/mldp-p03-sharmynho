import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="StrokeGuard", page_icon="🩺", layout="centered")

@st.cache_resource
def load_artifacts():
    return joblib.load("strokeguard_artifacts.joblib")

art = load_artifacts()
model = art["model"]
FINAL_THRESHOLD = float(art["final_threshold"])
GLUCOSE_CAP = float(art["glucose_cap"])
BMI_MEDIAN_TRAIN = float(art["bmi_median_train"])
SELECTED_FEATURES = art["selected_features"]


def set_bg_image(image_path: str):
    import base64
    with open(image_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/jpg;base64,{b64}") no-repeat center center fixed;
            background-size: cover;
        }}

        header[data-testid="stHeader"] {{ background: rgba(0,0,0,0) !important; }}
        div[data-testid="stToolbar"] {{ visibility: hidden !important; height: 0 !important; }}
        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}

        .block-container {{
            padding-top: 0.7rem !important;
            padding-bottom: 1.2rem !important;
            max-width: 980px !important;
        }}

        .brand-wrap {{
            margin: 8px auto 18px auto;
            padding: 12px 14px;
            border-radius: 18px;
            background: rgba(255,255,255,0.90);
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.18);
        }}
        .brand {{
            display:flex;
            align-items:center;
            gap:12px;
            padding: 0;
        }}
        .brand-badge {{
            width:40px; height:40px;
            border-radius:10px;
            background: rgba(0, 90, 255, 0.9);
            display:flex; align-items:center; justify-content:center;
            color:white;
            font-size:18px;
            font-weight:700;
        }}
        .brand-title {{
            font-size:20px;
            font-weight:800;
            color:#0f172a;
            margin:0;
            line-height:1.1;
        }}
        .brand-subtitle {{
            font-size:12px;
            color:#334155;
            margin:0;
        }}

        .center-title {{
            text-align:center;
            font-size: 42px;
            font-weight: 900;
            color: #0f172a;
            margin: 10px 0 4px 0;
        }}
        .center-sub {{
            text-align:center;
            font-size: 16px;
            color: #0f172a;
            margin: 0 0 18px 0;
        }}

        div[data-testid="stForm"] {{
            background: rgba(255,255,255,0.92) !important;
            border-radius: 18px !important;
            padding: 22px 22px 18px 22px !important;
            box-shadow: 0 12px 35px rgba(15, 23, 42, 0.25) !important;
            border: 1px solid rgba(15,23,42,0.10) !important;
        }}

        label {{
            color: #0f172a !important;
            font-weight: 700 !important;
        }}

        input, textarea {{
            background: #111827 !important;
            color: #ffffff !important;
            border: 1.5px solid rgba(255,255,255,0.35) !important;
            border-radius: 12px !important;
        }}

        div[data-baseweb="select"] > div {{
            background: #111827 !important;
            color: #ffffff !important;
            border: 1.5px solid rgba(255,255,255,0.35) !important;
            border-radius: 12px !important;
        }}
        div[data-baseweb="select"] span {{
            color: #ffffff !important;
        }}

        div[data-testid="stNumberInput"] button {{
            background: #111827 !important;
            color: #ffffff !important;
            border: 1.5px solid rgba(255,255,255,0.35) !important;
        }}
        div[data-testid="stNumberInput"] svg {{
            fill: #ffffff !important;
        }}

        div[data-testid="stForm"] button[kind="primary"] {{
            background: #2563eb !important;
            color: #ffffff !important;
            border-radius: 12px !important;
            font-weight: 800 !important;
            padding: 10px 14px !important;
            width: 55% !important;
            min-width: 240px !important;
            display: block !important;
            margin: 12px auto 0 auto !important;
        }}

        code {{
            color: #ffffff !important;
            background: #111827 !important;
            padding: 2px 8px !important;
            border-radius: 10px !important;
        }}

        div[data-baseweb="tooltip"] {{
            background: #ffffff !important;
            color: #0f172a !important;
            border-radius: 10px !important;
            border: 1px solid rgba(15,23,42,0.15) !important;
        }}

        .result-wrap {{
            background: rgba(255,255,255,0.92);
            border-radius: 18px;
            padding: 18px 18px 14px 18px;
            margin-top: 16px;
            box-shadow: 0 12px 35px rgba(15, 23, 42, 0.25);
            border: 1px solid rgba(15,23,42,0.10);
        }}
        .result-wrap, .result-wrap * {{
            color: #0f172a !important;
        }}
        .result-title {{
            text-align:center;
            font-size: 28px;
            font-weight: 900;
            margin: 0 0 10px 0;
        }}

        .risk-box {{
            border-radius: 12px;
            padding: 12px 14px;
            margin: 10px 0 10px 0;
            font-weight: 800;
        }}
        .risk-high {{
            background: rgba(239, 68, 68, 0.18);
            border: 1px solid rgba(239, 68, 68, 0.35);
        }}
        .risk-low {{
            background: rgba(34, 197, 94, 0.18);
            border: 1px solid rgba(34, 197, 94, 0.35);
        }}

        .bar {{
            height: 10px;
            background: rgba(15,23,42,0.18);
            border-radius: 999px;
            overflow: hidden;
            margin-top: 8px;
        }}
        .bar > div {{
            height: 100%;
            width: 0%;
            background: #2563eb;
        }}

        div[data-testid="stForm"] div[data-testid="stFormSubmitButton"] {{
            display: flex !important;
            justify-content: center !important;
        }}
        div[data-testid="stForm"] div[data-testid="stFormSubmitButton"] > button,
        div[data-testid="stForm"] button[kind="primary"] {{
            width: 55% !important;
            min-width: 240px !important;
            display: block !important;
            margin: 12px auto 0 auto !important;
        }}

        div[data-testid="stNumberInput"] button {{
            background: #111827 !important;
            border: 1.5px solid rgba(255,255,255,0.35) !important;
        }}
        div[data-testid="stNumberInput"] svg {{
            fill: #ffffff !important;
        }}

        .result-wrap, .result-wrap * {{
            color: #0f172a !important;
        }}
        .result-wrap code, .result-wrap code * {{
            color: #ffffff !important;
            background: #111827 !important;
            padding: 2px 10px !important;
            border-radius: 999px !important;
            font-weight: 800 !important;
        }}

        .bar {{
            height: 12px !important;
            background: rgba(15,23,42,0.12) !important;
            border-radius: 999px !important;
            overflow: hidden !important;
        }}
        .bar > div {{
            height: 100% !important;
            background: #2563eb !important;
            width: 0%;
            border-radius: 999px !important;
        }}

        .result-outside-title{{
            text-align:center;
            font-size: 34px;
            font-weight: 900;
            color:#0f172a;
            margin: 22px 0 10px 0;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_image("strokeimage.jpg")

st.markdown(
    """
    <div class="brand-wrap">
        <div class="brand">
            <div class="brand-badge">≋</div>
            <div>
                <p class="brand-title">StrokeGuard</p>
                <p class="brand-subtitle">Advanced Stroke Risk Assessment System</p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


def build_feature_row(
    gender: str,
    age: int,
    hypertension: str,
    ever_married: str,
    work_type: str,
    avg_glucose_level: float,
    bmi: float,
    smoking_status: str
) -> pd.DataFrame:

    glucose = min(float(avg_glucose_level), GLUCOSE_CAP)
    bmi_val = float(bmi) if bmi is not None else BMI_MEDIAN_TRAIN

    row = {f: 0 for f in SELECTED_FEATURES}

    if "age" in row:
        row["age"] = int(age)
    if "hypertension" in row:
        row["hypertension"] = 1 if hypertension == "Yes" else 0
    if "avg_glucose_level" in row:
        row["avg_glucose_level"] = glucose
    if "bmi" in row:
        row["bmi"] = bmi_val

    if "gender_Male" in row:
        row["gender_Male"] = 1 if gender == "Male" else 0
    if "ever_married_Yes" in row:
        row["ever_married_Yes"] = 1 if ever_married == "Yes" else 0
    if "work_type_Private" in row:
        row["work_type_Private"] = 1 if work_type == "Private" else 0
    if "work_type_Self-employed" in row:
        row["work_type_Self-employed"] = 1 if work_type == "Self-employed" else 0
    if "smoking_status_never smoked" in row:
        row["smoking_status_never smoked"] = 1 if smoking_status == "never smoked" else 0
    if "smoking_status_smokes" in row:
        row["smoking_status_smokes"] = 1 if smoking_status == "smokes" else 0

    return pd.DataFrame([row], columns=SELECTED_FEATURES)


#UI
st.markdown('<div class="center-title">Patient Risk Assessment</div>', unsafe_allow_html=True)
st.markdown('<div class="center-sub">Complete the form below for your stroke risk analysis</div>', unsafe_allow_html=True)

with st.form("stroke_form", clear_on_submit=False):

    gender = st.selectbox("Gender *", ["Select gender...", "Female", "Male"], index=0)
    hypertension = st.selectbox("Do you have hypertension? *", ["Select...", "No", "Yes"], index=0)
    ever_married = st.selectbox("Have you ever been married? *", ["Select...", "No", "Yes"], index=0)
    work_type = st.selectbox("What is your work type? *", ["Select work type...", "Government job", "Private", "Self-employed"], index=0)
    smoking_status = st.selectbox("Smoking Status *", ["Select smoking status...", "formerly smoked", "never smoked", "smokes"], index=0)

    age = st.number_input("Age *", min_value=0, max_value=120, value=0, step=1)

    avg_glucose_level = st.number_input(
        "Average Glucose Level (mg/dL) *",
        min_value=0.00, max_value=400.00,
        value=0.00, step=0.01, format="%.2f",
        help="Example: 105.50 (2 decimal places)"
    )
    st.caption("Normal range: 70–100 mg/dL")

    bmi = st.number_input(
        "Body Mass Index (BMI) *",
        min_value=0.0, max_value=80.0,
        value=0.0, step=0.1, format="%.1f",
        help="Example: 24.5 (1 decimal place)"
    )
    st.caption("Normal range: 18.5–24.9")

    submitted = st.form_submit_button("Analyse stroke risk")


# validation and prediction
if submitted:
    errors = []

    # dropdown validations
    if gender.startswith("Select"):
        errors.append("Please select **Gender**.")
    if hypertension.startswith("Select"):
        errors.append("Please select **Hypertension** status.")
    if ever_married.startswith("Select"):
        errors.append("Please select **Marital status**.")
    if work_type.startswith("Select"):
        errors.append("Please select **Work type**.")
    if smoking_status.startswith("Select"):
        errors.append("Please select **Smoking status**.")

    # numeric validations
    # numeric validations (now using number_input)
    if age == 0:
        errors.append("Please enter **Age** (cannot be 0).")
    if avg_glucose_level == 0:
        errors.append("Please enter **Average glucose level** (cannot be 0).")
    if bmi == 0:
        errors.append("Please enter **BMI** (cannot be 0).")

    if round(avg_glucose_level, 2) != avg_glucose_level:
        errors.append("Average glucose level must be **2 decimal places** (e.g., 105.50).")

    if round(bmi, 1) != bmi:
        errors.append("BMI must be **1 decimal place** (e.g., 24.5).")

    if errors:
        st.error("Please fix the following input issues:\n\n- " + "\n- ".join(errors))
    else:
        X_new = build_feature_row(
            gender=gender,
            age=age,
            hypertension=hypertension,
            ever_married=ever_married,
            work_type=work_type,
            avg_glucose_level=avg_glucose_level,
            bmi=bmi,
            smoking_status=smoking_status
        )

        prob = float(model.predict_proba(X_new)[:, 1][0])
        pred = int(prob >= FINAL_THRESHOLD)
        pct = int(min(max(prob, 0.0), 1.0) * 100)

        if pred == 1:
            risk_box = f"""
            <div class="risk-box risk-high">
                Prediction: Higher stroke risk (screening estimate)
            </div>
            """
        else:
            risk_box = f"""
            <div class="risk-box risk-low">
                Prediction: Lower stroke risk (screening estimate)
            </div>
            """
        st.markdown('<div class="result-outside-title">Result</div>', unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="result-wrap">
                <p><b>Predicted stroke risk probability:</b> <code>{prob:.4f}</code></p>
                {risk_box}
                <p style="margin:0 0 6px 0; font-size: 14px;">
                    This is a screening estimate, not a medical diagnosis. Please consult a healthcare professional if you have concerns.
                </p>
                <div class="bar"><div style="width:{pct}%;"></div></div>
            </div>
            """,
            unsafe_allow_html=True
        )
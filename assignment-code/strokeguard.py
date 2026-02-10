import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="StrokeGuard", page_icon="🩺", layout="wide")

# Load model + thresholds + feature list

@st.cache_resource
def load_artifacts():
    return joblib.load("strokeguard_artifacts.joblib")

art = load_artifacts()
model = art["model"]
FINAL_THRESHOLD = float(art["final_threshold"])
GLUCOSE_CAP = float(art["glucose_cap"])
BMI_MEDIAN_TRAIN = float(art["bmi_median_train"])
SELECTED_FEATURES = art["selected_features"]

#background and css
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

        /* Top-left brand area spacing */
        .brand {{
            display:flex;
            align-items:center;
            gap:12px;
            padding: 14px 10px 6px 10px;
        }}
        .brand-badge {{
            width:40px;
            height:40px;
            border-radius:10px;
            background: rgba(0, 90, 255, 0.9);
            display:flex;
            align-items:center;
            justify-content:center;
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

        /* Center card */
        .card {{
            background: rgba(255,255,255,0.92);
            border-radius: 18px;
            padding: 26px 26px 18px 26px;
            box-shadow: 0 12px 35px rgba(15, 23, 42, 0.20);
            max-width: 980px;  /* wider so less scrolling */
            margin: 18px auto 30px auto;
        }}

        .card h2 {{
            text-align:center;
            margin: 0 0 4px 0;
            color:#0f172a;
            font-weight:800;
        }}

        .card p {{
            text-align:center;
            margin: 0 0 18px 0;
            color:#475569;
            font-size: 13px;
        }}

        /* Make Streamlit widgets look tighter inside card */
        div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stForm"]) {{
            padding-top: 0px;
        }}

        /* Button styling */
        .stButton button, .stForm button {{
            width: 100%;
            border-radius: 12px;
            padding: 10px 14px;
            font-weight: 700;
        }}

        /* Reduce extra space above/below */
        .block-container {{
            padding-top: 0.6rem;
            padding-bottom: 1rem;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_image("strokeimage.jpg")

#header at top left of my app
st.markdown(
    """
    <div class="brand">
        <div class="brand-badge">≋</div>
        <div>
            <p class="brand-title">StrokeGuard</p>
            <p class="brand-subtitle">Advanced Stroke Risk Assessment System</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

#model input row for my selected features
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
    """
    Creates a single-row DataFrame with EXACT columns = SELECTED_FEATURES.
    Uses the same one-hot baseline logic you had with drop_first=True.
    """

    #numeric preprocessing
    glucose = min(float(avg_glucose_level), GLUCOSE_CAP)  # winsorisation cap
    bmi_val = float(bmi) if bmi is not None else BMI_MEDIAN_TRAIN

    # Start all features at 0
    row = {f: 0 for f in SELECTED_FEATURES}

    # Continuous and binary base features
    if "age" in row:
        row["age"] = int(age)
    if "hypertension" in row:
        row["hypertension"] = 1 if hypertension == "Yes" else 0
    if "avg_glucose_level" in row:
        row["avg_glucose_level"] = glucose
    if "bmi" in row:
        row["bmi"] = bmi_val

    # OHE columns that exist in my selected features
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

#main ui
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("<h2>Patient Risk Assessment</h2>", unsafe_allow_html=True)
st.markdown("<p>Complete the form below for your stroke risk analysis</p>", unsafe_allow_html=True)

with st.form("stroke_form", clear_on_submit=False):

    c1, c2 = st.columns(2, gap="large")

    with c1:
        gender = st.selectbox("Gender *", ["Female", "Male"], index=0)
        age = st.number_input("Age *", min_value=0, max_value=120, value=30, step=1)
        hypertension = st.selectbox("Do you have hypertension? *", ["No", "Yes"], index=0)
        ever_married = st.selectbox("Have you ever been married? *", ["No", "Yes"], index=0)

    with c2:
        work_type = st.selectbox("What is your work type? *", ["Govt_job", "Private", "Self-employed"], index=0)

        avg_glucose_level = st.number_input(
            "Average Glucose Level (mg/dL) *",
            min_value=0.00, max_value=400.00, value=100.00, step=0.01, format="%.2f",
            help="Example: 105.50 (2 decimal places)"
        )

        bmi = st.number_input(
            "Body Mass Index (BMI) *",
            min_value=0.0, max_value=80.0, value=24.5, step=0.1, format="%.1f",
            help="Example: 24.5 (1 decimal place)"
        )

        smoking_status = st.selectbox(
            "Smoking Status *",
            ["formerly smoked", "never smoked", "smokes"],
            index=0
        )

    submitted = st.form_submit_button("Analyse stroke risk")

#validation and prediction
if submitted:
    errors = []

    # Age must be whole number
    if int(age) != age:
        errors.append("Age must be a **whole number**.")

    # Glucose 2dp validation
    if round(avg_glucose_level, 2) != avg_glucose_level:
        errors.append("Average glucose level must be **2 decimal places**.")

    # BMI 1dp validation
    if round(bmi, 1) != bmi:
        errors.append("BMI must be **1 decimal place**.")

    if age < 0 or age > 120:
        errors.append("Age must be between **0 and 120**.")
    if avg_glucose_level <= 0:
        errors.append("Average glucose level must be **greater than 0**.")
    if bmi <= 0:
        errors.append("BMI must be **greater than 0**.")

    if errors:
        st.error("Please fix the following input issues:\n\n- " + "\n- ".join(errors))
    else:
        X_new = build_feature_row(
            gender=gender,
            age=int(age),
            hypertension=hypertension,
            ever_married=ever_married,
            work_type=work_type,
            avg_glucose_level=float(avg_glucose_level),
            bmi=float(bmi),
            smoking_status=smoking_status
        )

        prob = float(model.predict_proba(X_new)[:, 1][0])
        pred = int(prob >= FINAL_THRESHOLD)

        st.markdown("---")
        st.subheader("Result")

        st.write(f"**Predicted stroke risk probability:** `{prob:.4f}`")
        st.write(f"**Threshold used:** `{FINAL_THRESHOLD:.2f}`")

        if pred == 1:
            st.error(" **Higher risk detected (Predicted: Stroke = 1).** Please consider medical follow-up.")
        else:
            st.success(" **Lower risk detected (Predicted: Stroke = 0).**")

        st.progress(min(max(prob, 0.0), 1.0))

st.markdown("</div>", unsafe_allow_html=True)
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).with_name("rf_pipeline.pkl")
NUMERICAL_FEATURES = [
    "age",
    "resting_blood_pressure",
    "cholesterol",
    "max_heart_rate",
    "st_depression",
]


@st.cache_resource
def load_pipeline():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Run the notebook cell that saves rf_pipeline.pkl first."
        )

    pipeline = joblib.load(MODEL_PATH)
    required_keys = {"model", "scaler", "features"}
    missing_keys = required_keys.difference(pipeline)
    if missing_keys:
        raise ValueError(
            f"The saved model is missing required keys: {sorted(missing_keys)}"
        )
    return pipeline


st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤",
    layout="centered",
)

st.title("Heart Disease Predictor")
st.write("Enter patient measurements to generate a model prediction.")
st.caption("This tool is for educational purposes and is not a medical diagnosis.")

try:
    pipeline = load_pipeline()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

model = pipeline["model"]
scaler = pipeline["scaler"]
features = pipeline["features"]

with st.form("patient_form"):
    st.subheader("Patient information")

    age = st.number_input("Age", min_value=1, max_value=120, value=54, step=1)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda value: "Female" if value == 0 else "Male")
    chest_pain_type = st.number_input("Chest pain type", min_value=0, max_value=3, value=2, step=1)
    resting_blood_pressure = st.number_input(
        "Resting blood pressure", min_value=50, max_value=250, value=130, step=1
    )
    cholesterol = st.number_input("Cholesterol", min_value=50, max_value=700, value=246, step=1)
    fasting_blood_sugar = st.selectbox(
        "Fasting blood sugar > 120 mg/dl", options=[0, 1], format_func=lambda value: "No" if value == 0 else "Yes"
    )
    ecg = st.number_input("ECG result", min_value=0, max_value=2, value=1, step=1)
    max_heart_rate = st.number_input("Maximum heart rate", min_value=50, max_value=250, value=150, step=1)
    exercise_induced_chest_pain = st.selectbox(
        "Exercise-induced chest pain", options=[0, 1], format_func=lambda value: "No" if value == 0 else "Yes"
    )
    st_depression = st.number_input("ST depression", min_value=0.0, max_value=10.0, value=1.2, step=0.1)
    st_slope = st.number_input("ST slope", min_value=0, max_value=2, value=1, step=1)
    stained_blood_vessels = st.number_input("Number of stained blood vessels", min_value=0, max_value=4, value=0, step=1)
    blood_disorder = st.number_input("Blood disorder", min_value=0, max_value=3, value=2, step=1)

    submitted = st.form_submit_button("Predict")

if submitted:
    patient_data = pd.DataFrame(
        [[
            age,
            sex,
            chest_pain_type,
            resting_blood_pressure,
            cholesterol,
            fasting_blood_sugar,
            ecg,
            max_heart_rate,
            exercise_induced_chest_pain,
            st_depression,
            st_slope,
            stained_blood_vessels,
            blood_disorder,
        ]],
        columns=features,
    )

    patient_data_scaled = patient_data.copy()
    patient_data_scaled[NUMERICAL_FEATURES] = scaler.transform(
        patient_data[NUMERICAL_FEATURES]
    )

    prediction = int(model.predict(patient_data_scaled)[0])
    probabilities = model.predict_proba(patient_data_scaled)[0]
    probability_no_disease = probabilities[0]
    probability_disease = probabilities[1]

    st.subheader("Prediction")
    if prediction == 1:
        st.error("The model predicts a higher likelihood of heart disease.")
    else:
        st.success("The model predicts a lower likelihood of heart disease.")

    col1, col2 = st.columns(2)
    col1.metric("No heart disease", f"{probability_no_disease:.1%}")
    col2.metric("Heart disease", f"{probability_disease:.1%}")

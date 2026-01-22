import streamlit as st
import joblib
import numpy as np

# ---------- Page config ----------
st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    layout="centered"
)

# ---------- Title ----------
st.title("❤️ Heart Disease Risk Prediction")
st.write("Enter patient data to predict cardiovascular disease risk.")

# ---------- Load model ----------
model = joblib.load("heart_disease_model.pkl")

# ---------- Inputs ----------
age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex (0 = female, 1 = male)", [0, 1])
cp = st.selectbox("Chest pain type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting blood pressure (mm Hg)", 80, 250, 120)
chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (0 = no, 1 = yes)", [0, 1])

restecg = st.selectbox("Resting ECG results (0–2)", [0, 1, 2])
thalach = st.number_input("Maximum heart rate achieved", 60, 220, 150)
exang = st.selectbox("Exercise induced angina (0 = no, 1 = yes)", [0, 1])
oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 10.0, 1.0)
slope = st.selectbox("Slope of peak exercise ST segment (0–2)", [0, 1, 2])
ca = st.selectbox("Number of major vessels (0–4)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (0–3)", [0, 1, 2, 3])

# ---------- Predict ----------
if st.button("🔍 Predict risk"):

    features = np.array([[age, sex, cp, trestbps, chol, fbs,
                           restecg, thalach, exang, oldpeak,
                           slope, ca, thal]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    st.markdown("---")

    if prediction == 1:
        st.error(f"⚠️ High risk of heart disease\n\nProbability: {probability:.2%}")
    else:
        st.success(f"✅ Low risk of heart disease\n\nProbability: {probability:.2%}")

    st.info("⚕️ This result is for educational purposes only and not a medical diagnosis.")

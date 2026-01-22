import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Heart Disease Risk Prediction", layout="centered")
st.title("❤️ Heart Disease Risk Prediction")
st.write("Enter patient data to predict cardiovascular disease risk.")

model = joblib.load("heart_disease_model.pkl")
n_expected = getattr(model, "n_features_in_", None)
st.caption(f"✅ Model expects: {n_expected} features")

# --- Base features ---
age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex (0=female, 1=male)", [0, 1])
cp = st.selectbox("Chest pain type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting blood pressure (mm Hg)", 80, 250, 120)
chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (0=no, 1=yes)", [0, 1])

# --- Common extra features (UCI-like) ---
st.subheader("Additional features")
restecg = st.selectbox("Resting ECG results (0–2)", [0, 1, 2])
thalach = st.number_input("Max heart rate achieved (thalach)", 60, 220, 150)
exang = st.selectbox("Exercise induced angina (0=no, 1=yes)", [0, 1])
oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 10.0, 1.0)
slope = st.selectbox("Slope of peak exercise ST segment (0–2)", [0, 1, 2])
ca = st.selectbox("Number of major vessels (ca: 0–4)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (thal: 0–3)", [0, 1, 2, 3])

# --- Two unknown features to reach 15 ---
st.subheader("Extra features (your model needs them)")
extra_1 = st.number_input("Extra feature 1 (unknown)", value=0.0)
extra_2 = st.number_input("Extra feature 2 (unknown)", value=0.0)

if st.button("🔍 Predict risk"):
    try:
        if n_expected != 15 and n_expected is not None:
            st.error(f"Model expects {n_expected} features, but this app is configured for 15.")
            st.stop()

        features = np.array([[
            age, sex, cp, trestbps, chol, fbs,
            restecg, thalach, exang, oldpeak, slope, ca, thal,
            extra_1, extra_2
        ]], dtype=float)

        pred = model.predict(features)[0]

        prob = None
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(features)[0][1])

        st.markdown("---")
        if int(pred) == 1:
            st.error("⚠️ High risk of heart disease")
        else:
            st.success("✅ Low risk of heart disease")

        if prob is not None:
            st.write(f"Probability (class 1): **{prob:.2%}**")

        st.info("⚕️ Educational use only. Not a medical diagnosis.")

    except Exception as e:
        st.error("❌ Prediction failed.")
        st.code(str(e))

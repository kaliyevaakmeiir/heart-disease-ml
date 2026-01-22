import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Heart Risk Predictor", page_icon="❤️")

st.title("❤️ Heart Disease Risk Prediction")
st.write("Simple model with 5 features")

# Load model
model = joblib.load("heart_disease_model.pkl")

# Inputs
age = st.number_input("Age", min_value=1, max_value=120, value=45)
sex = st.selectbox("Sex (0 = female, 1 = male)", [0, 1])
trestbps = st.number_input("Resting blood pressure (mm Hg)", min_value=80, max_value=250, value=120)
chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting blood sugar > 120 mg/dl", [0, 1])

if st.button("Predict risk"):
    try:
        X = np.array([[age, sex, trestbps, chol, fbs]])
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]

        if prediction == 1:
            st.error(f"⚠️ High risk of heart disease ({probability:.2%})")
        else:
            st.success(f"✅ Low risk of heart disease ({probability:.2%})")

    except Exception as e:
        st.error("❌ Prediction failed")
        st.exception(e)

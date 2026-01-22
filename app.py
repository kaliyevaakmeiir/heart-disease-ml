import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Heart Disease Risk Prediction", layout="centered")

st.title("❤️ Heart Disease Risk Prediction")
st.write("Enter patient data to predict cardiovascular disease risk.")

# --- Load model ---
model = joblib.load("heart_disease_model.pkl")

# --- Detect how many features the model expects ---
n_expected = getattr(model, "n_features_in_", None)

st.caption(f"✅ Model expects: {n_expected} features" if n_expected is not None else "⚠️ Could not detect n_features_in_")

# --- Inputs (base 6) ---
age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex (0 = female, 1 = male)", [0, 1])
cp = st.selectbox("Chest pain type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting blood pressure (mm Hg)", 80, 250, 120)
chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (0 = no, 1 = yes)", [0, 1])

# --- Extra 7 (only if model expects 13) ---
restecg = thalach = exang = oldpeak = slope = ca = thal = None
if n_expected == 13:
    st.subheader("Additional features")
    restecg = st.selectbox("Resting ECG results (0–2)", [0, 1, 2])
    thalach = st.number_input("Maximum heart rate achieved", 60, 220, 150)
    exang = st.selectbox("Exercise induced angina (0 = no, 1 = yes)", [0, 1])
    oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope of peak exercise ST segment (0–2)", [0, 1, 2])
    ca = st.selectbox("Number of major vessels (0–4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (0–3)", [0, 1, 2, 3])

if st.button("🔍 Predict risk"):
    try:
        if n_expected == 6 or n_expected is None:
            features = np.array([[age, sex, cp, trestbps, chol, fbs]])
        elif n_expected == 13:
            features = np.array([[age, sex, cp, trestbps, chol, fbs,
                                  restecg, thalach, exang, oldpeak,
                                  slope, ca, thal]])
        else:
            st.error(f"Model expects {n_expected} features. This app supports only 6 or 13. Retrain model or adjust inputs.")
            st.stop()

        pred = model.predict(features)[0]

        # probability (if available)
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
        st.error("❌ Prediction failed. Most likely feature mismatch with the trained model.")
        st.code(str(e))

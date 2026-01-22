import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    layout="centered"
)

st.title("❤️ Heart Disease Risk Prediction")
st.write("Enter patient data to predict cardiovascular disease risk.")

# ----------------------------
# Load model
# ----------------------------
MODEL_PATH = "heart_disease_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Cannot load model file: {MODEL_PATH}")
    st.exception(e)
    st.stop()

# ----------------------------
# Inputs (classic UCI Heart Disease features)
# ----------------------------
age = st.number_input("Age", min_value=1, max_value=120, value=45, step=1)

sex = st.selectbox("Sex (0 = female, 1 = male)", [0, 1], index=0)

cp = st.selectbox("Chest pain type (0–3)", [0, 1, 2, 3], index=1)

trestbps = st.number_input("Resting blood pressure (mm Hg)", min_value=50, max_value=250, value=120, step=1)

chol = st.number_input("Cholesterol (mg/dl)", min_value=50, max_value=700, value=200, step=1)

fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (0 = no, 1 = yes)", [0, 1], index=0)

restecg = st.selectbox("Resting ECG results (0–2)", [0, 1, 2], index=0)

thalach = st.number_input("Max heart rate achieved (thalach)", min_value=50, max_value=250, value=150, step=1)

exang = st.selectbox("Exercise induced angina (0 = no, 1 = yes)", [0, 1], index=0)

oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

slope = st.selectbox("Slope of peak exercise ST segment (0–2)", [0, 1, 2], index=1)

ca = st.selectbox("Number of major vessels colored by fluoroscopy (0–4)", [0, 1, 2, 3, 4], index=0)

thal = st.selectbox("Thalassemia (0–3)", [0, 1, 2, 3], index=2)

# ----------------------------
# Model expects features by column names (Pipeline + ColumnTransformer)
# We'll build a dataframe with names.
# Some models expect 13 features, some 15. We'll handle both.
# ----------------------------

base_feature_names_13 = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal"
]

# If your model expects 15 features, we add two extra placeholders.
# (We can't know their real meaning without training code, but this prevents crash.)
extra_feature_names_15 = ["extra_1", "extra_2"]

st.markdown("---")

# Detect expected number of features if possible
expected_n = None
try:
    if hasattr(model, "n_features_in_"):
        expected_n = int(model.n_features_in_)
except Exception:
    expected_n = None

# If pipeline uses transformers, feature count may not be easily accessible.
# We'll infer from training expectation heuristics.
# Default behavior:
# - If model expects 15 -> ask extra_1 and extra_2
# - Else -> use only 13
use_15 = False
if expected_n == 15:
    use_15 = True
elif expected_n == 13:
    use_15 = False
else:
    # If unknown, try 13 first (most common for UCI dataset)
    use_15 = False

# Optional inputs for 15-feature models
extra_1 = 0.0
extra_2 = 0.0

if use_15:
    st.subheader("Extra features (your model needs them)")
    extra_1 = st.number_input("Extra feature 1 (unknown)", value=0.0, step=1.0)
    extra_2 = st.number_input("Extra feature 2 (unknown)", value=0.0, step=1.0)

# ----------------------------
# Prediction
# ----------------------------
if st.button("🔍 Predict risk"):
    try:
        values_13 = [
            age, sex, cp, trestbps, chol, fbs,
            restecg, thalach, exang, oldpeak,
            slope, ca, thal
        ]

        if use_15:
            feature_names = base_feature_names_13 + extra_feature_names_15
            values = [values_13 + [extra_1, extra_2]]
        else:
            feature_names = base_feature_names_13
            values = [values_13]

        # IMPORTANT: pass DataFrame with column names
        X = pd.DataFrame(values, columns=feature_names)

        # Predict class
        pred = model.predict(X)[0]

        # Predict probability if available
        prob = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            # class "1" probability usually at index 1
            if proba is not None and len(proba[0]) > 1:
                prob = float(proba[0][1])

        # Output
        if int(pred) == 1:
            st.error("⚠️ Prediction: Higher risk of heart disease (class = 1)")
        else:
            st.success("✅ Prediction: Lower risk of heart disease (class = 0)")

        if prob is not None:
            st.info(f"Estimated probability (class=1): **{prob:.2f}**")

        # Debug view (optional)
        with st.expander("Show input data sent to the model"):
            st.dataframe(X)

    except Exception as e:
        st.error("❌ Prediction failed.")
        st.write(str(e))
        st.exception(e)

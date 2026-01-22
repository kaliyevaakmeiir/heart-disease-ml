import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Heart Disease Risk Prediction", layout="centered")

st.title("❤️ Heart Disease Risk Prediction")
st.write("Enter patient data to predict cardiovascular disease risk.")

model = joblib.load("heart_disease_model.pkl")

age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex (0 = female, 1 = male)", [0, 1])
cp = st.number_input("Chest pain type", 0, 3, 1)
trestbps = st.number_input("Resting blood pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 400, 200)
fbs = st.selectbox("Fasting blood sugar > 120 mg/dl", [0, 1])

                   


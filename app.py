import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page config
st.set_page_config(page_title="Cervical Cancer Risk Prediction", layout="centered")

st.title("Cervical Cancer Risk Prediction System")
st.write("Prediction based on machine learning model")

# Load artifacts
path = "pkl_files/"
model = joblib.load(path + "rf_model_deploy.pkl")
imputer = joblib.load(path + "imputer_top.pkl")
scaler = joblib.load(path + "scaler_top.pkl")
top_features = joblib.load(path + "top_features.pkl")

st.header("Enter Patient Information")

# Create input fields dynamically
input_data = {}

for feature in top_features:
    value = st.number_input(f"{feature}", value=0.0)
    input_data[feature] = value

# Prediction button
if st.button("Predict Risk"):

    input_df = pd.DataFrame([input_data])

    # Apply preprocessing
    input_array = imputer.transform(input_df)
    input_array = scaler.transform(input_array)

    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"High Risk Detected (Probability: {probability:.2f})")
    else:
        st.success(f"Low Risk (Probability: {probability:.2f})")

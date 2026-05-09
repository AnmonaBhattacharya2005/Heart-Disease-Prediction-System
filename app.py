# ==========================================
# Heart Disease Prediction using Streamlit
# Using your heart.csv dataset
# ==========================================

# Save this file as app.py

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ------------------------------------------
# Load Dataset
# ------------------------------------------

df = pd.read_csv("heart.csv")

# ------------------------------------------
# Split Features and Target
# ------------------------------------------

X = df.drop("target", axis=1)
y = df["target"]

# ------------------------------------------
# Train Test Split
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ------------------------------------------
# Train ML Model
# ------------------------------------------

model = RandomForestClassifier()

model.fit(X_train, y_train)

# ------------------------------------------
# Model Accuracy
# ------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# ------------------------------------------
# Streamlit UI
# ------------------------------------------

st.title("❤️ Heart Disease Prediction App")

st.write("Machine Learning + Streamlit Project")

st.success(f"Model Accuracy: {accuracy * 100:.2f}%")

st.subheader("Enter Patient Details")

# ------------------------------------------
# User Inputs
# ------------------------------------------

age = st.slider("Age", 20, 80, 40)

sex = st.selectbox(
    "Sex",
    [0, 1],
    format_func=lambda x: "Female" if x == 0 else "Male"
)

cp = st.selectbox(
    "Chest Pain Type",
    [0, 1, 2, 3]
)

trestbps = st.slider(
    "Resting Blood Pressure",
    80,
    200,
    120
)

chol = st.slider(
    "Cholesterol",
    100,
    400,
    200
)

fbs = st.selectbox(
    "Fasting Blood Sugar",
    [0, 1]
)

restecg = st.selectbox(
    "Rest ECG",
    [0, 1, 2]
)

thalach = st.slider(
    "Maximum Heart Rate",
    60,
    220,
    150
)

exang = st.selectbox(
    "Exercise Induced Angina",
    [0, 1]
)

oldpeak = st.slider(
    "Oldpeak",
    0.0,
    6.0,
    1.0
)

slope = st.selectbox(
    "Slope",
    [0, 1, 2]
)

ca = st.selectbox(
    "Number of Major Vessels",
    [0, 1, 2, 3, 4]
)

thal = st.selectbox(
    "Thal",
    [0, 1, 2, 3]
)

# ------------------------------------------
# Prediction
# ------------------------------------------

if st.button("Predict"):

    input_data = np.array([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

# ------------------------------------------
# Show Dataset
# ------------------------------------------

if st.checkbox("Show Dataset"):
    st.write(df.head())

# ------------------------------------------
# Dataset Information
# ------------------------------------------

if st.checkbox("Show Dataset Info"):
    st.write(df.describe())
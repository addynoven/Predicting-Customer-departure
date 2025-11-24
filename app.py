import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="centered"
)

# ---- LOAD MODEL ----
model = tf.keras.models.load_model("model.h5")

with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("onehot_encoder_geo.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


# ---- HEADER ----
st.markdown(
    """
    <h1 style="text-align:center; margin-bottom:0;">Customer Churn Prediction</h1>
    <p style="text-align:center; color:#666; font-size:18px;">
        Enter customer details to estimate their churn probability.
    </p>
    """,
    unsafe_allow_html=True,
)

st.write("---")

# ---- FORM STYLE ----
form_container = st.container()
with form_container:

    col1, col2 = st.columns(2)

    with col1:
        geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
        gender = st.selectbox("Gender", label_encoder_gender.classes_)
        age = st.slider("Age", 18, 92)
        credit_score = st.number_input("Credit Score", min_value=0)

    with col2:
        balance = st.number_input("Balance", min_value=0.0)
        estimated_salary = st.number_input("Estimated Salary", min_value=0.0)
        tenure = st.slider("Tenure (Years)", 0, 10)
        num_of_products = st.slider("Number of Products", 1, 4)

    col3, col4 = st.columns(2)

    with col3:
        has_cr_card = st.selectbox("Has Credit Card", [0, 1])

    with col4:
        is_active_member = st.selectbox("Is Active Member", [0, 1])


# ---- PREPARE INPUT ----
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})

# One-hot encode
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
)

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scaling
input_data_scaled = scaler.transform(input_data)


# ---- PREDICT ----
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# ---- DISPLAY RESULT ----
st.write("---")
st.subheader("Prediction Result")

if prediction_proba > 0.5:
    color = "#ffcccc"
    text = "Customer is likely to churn"
else:
    color = "#ccffcc"
    text = "Customer is not likely to churn"

st.markdown(
    f"""
    <div style="
        background-color:{color};
        padding:20px;
        border-radius:10px;
        text-align:center;
        font-size:20px;
        border:1px solid #ddd;
    ">
        <b>Churn Probability: {prediction_proba:.2f}</b><br><br>
        {text}
    </div>
    """,
    unsafe_allow_html=True,
)

import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide",
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
body {
    background-color: #0f1117;
}
h1, h2, h3, label {
    color: #ffffff;
}
.css-1d391kg, .css-1offfwp, .stSlider label, .stSelectbox label, .stNumberInput label {
    color: #ffffff !important;
}
.container-card {
    background: rgba(255,255,255,0.04);
    padding: 25px;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
.result-card {
    padding: 30px;
    border-radius: 14px;
    text-align: center;
    font-size: 22px;
    font-weight: 600;
    box-shadow: 0 4px 25px rgba(0,0,0,0.4);
}
</style>
""", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
model = tf.keras.models.load_model("model.h5")

with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("onehot_encoder_geo.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


# ---------- HEADER ----------
st.markdown(
    """
    <h1 style="text-align:center; margin-bottom:0;">Customer Churn Prediction</h1>
    <p style="text-align:center; color:#9a9a9a; font-size:18px; margin-top:-5px;">
        Enter customer details to estimate their churn probability.
    </p>
    """,
    unsafe_allow_html=True,
)


# ---------- INPUT CARD ----------
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

st.markdown('</div>', unsafe_allow_html=True)



# ---------- INFERENCE ----------
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

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
)

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
input_data_scaled = scaler.transform(input_data)


prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]



# ---------- RESULT ----------
st.write("")
st.markdown("### Prediction Result")

if prediction_proba > 0.5:
    bg_color = "rgba(255, 70, 70, 0.25)"
    text = "Customer is likely to churn"
else:
    bg_color = "rgba(100, 255, 100, 0.25)"
    text = "Customer is not likely to churn"

st.markdown(
    f"""
    <div class="result-card" style="background:{bg_color}">
        <div>Churn Probability: {prediction_proba:.2f}</div>
        <div style="margin-top:10px; font-size:18px; color:#e8e8e8;">{text}</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------- FOOTER ----------
st.markdown(
    """
    <p style="text-align:center; margin-top:40px; color:#666;">Built with Streamlit</p>
    """,
    unsafe_allow_html=True,
)



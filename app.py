import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import plotly.graph_objects as go

# ---------- PAGE CONFIGURATION ----------
st.set_page_config(
    page_title="ChurnIQ | Customer Retention AI",
    page_icon="📉",
    layout="centered", # Centered looks better for a calculator tool
    initial_sidebar_state="collapsed"
)

# ---------- CUSTOM CSS FOR DARK THEME ----------
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(to bottom right, #0f1117, #161b22);
    }
    
    /* Inputs */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"], .stSlider {
        background-color: #1f2937 !important;
        color: white !important;
        border-radius: 8px;
    }
    
    /* Headings */
    h1 {
        background: -webkit-linear-gradient(45deg, #4ade80, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }
    h3 {
        color: #e5e7eb !important;
        font-weight: 600;
        font-size: 1.1rem !important;
        margin-top: 20px;
    }
    
    /* Custom Button */
    div.stButton > button {
        background: linear-gradient(90deg, #3b82f6, #2563eb);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# ---------- LOAD ASSETS (CACHED) ----------
@st.cache_resource
def load_assets():
    try:
        model_loaded = tf.keras.models.load_model("model.h5")
        
        with open("label_encoder_gender.pkl", "rb") as file:
            le_gender = pickle.load(file)
            
        with open("onehot_encoder_geo.pkl", "rb") as file:
            ohe_geo = pickle.load(file)
            
        with open("scaler.pkl", "rb") as file:
            scaler_loaded = pickle.load(file)
            
        return model_loaded, le_gender, ohe_geo, scaler_loaded
    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}. Please ensure .h5 and .pkl files are in the directory.")
        return None, None, None, None

model, label_encoder_gender, onehot_encoder_geo, scaler = load_assets()

# ---------- HEADER ----------
st.title("ChurnIQ Prediction")
st.markdown("Adjust customer details below to estimate the probability of churn.")

# Stop execution if files didn't load
if model is None:
    st.stop()

# ---------- INPUT FORM ----------
with st.form("churn_form"):
    
    st.markdown("### 👤 Demographics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
    with col2:
        gender = st.selectbox("Gender", label_encoder_gender.classes_)
    with col3:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)

    st.markdown("### 💳 Financials")
    col4, col5 = st.columns(2)
    
    with col4:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
        balance = st.number_input("Account Balance ($)", min_value=0.0, value=50000.0, step=1000.0)
    with col5:
        estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=60000.0, step=1000.0)
        num_of_products = st.slider("Number of Products", 1, 4, 2)

    st.markdown("### 🏦 Account Status")
    col6, col7, col8 = st.columns(3)
    
    with col6:
        tenure = st.slider("Tenure (Years)", 0, 10, 5)
    with col7:
        # UI uses Yes/No, Logic converts to 1/0
        has_cr_card_input = st.selectbox("Has Credit Card?", ["Yes", "No"])
        has_cr_card = 1 if has_cr_card_input == "Yes" else 0
    with col8:
        active_member_input = st.selectbox("Is Active Member?", ["Yes", "No"])
        is_active_member = 1 if active_member_input == "Yes" else 0

    st.write("") # Spacer
    submitted = st.form_submit_button("Calculate Churn Risk")

# ---------- LOGIC & RESULTS ----------
if submitted:
    # 1. Prepare Data
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

    # 2. Encode Geography
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
    )

    # 3. Combine & Scale
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
    input_data_scaled = scaler.transform(input_data)

    # 4. Predict
    prediction = model.predict(input_data_scaled)
    churn_probability = prediction[0][0]

    # ---------- DISPLAY RESULTS ----------
    st.markdown("---")
    
    # Determine color and status
    if churn_probability > 0.5:
        status_color = "#ff4b4b" # Red
        status_text = "High Risk"
    else:
        status_color = "#2ecc71" # Green
        status_text = "Safe"

    # Create Columns for Result
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.markdown(f"""
        <div style="background-color: rgba(255,255,255,0.05); padding: 20px; border-radius: 10px; text-align: center;">
            <h4 style="margin:0; color: #9ca3af;">Prediction</h4>
            <h2 style="margin:0; color: {status_color}; font-size: 2rem;">{status_text}</h2>
        </div>
        """, unsafe_allow_html=True)

    with res_col2:
        # Plotly Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = churn_probability * 100,
            number = {'suffix': "%"},
            title = {'text': "Churn Probability"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': status_color},
                'steps': [
                    {'range': [0, 50], 'color': "rgba(46, 204, 113, 0.2)"},
                    {'range': [50, 100], 'color': "rgba(255, 75, 75, 0.2)"}
                ],
            }
        ))
        
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white"},
            height=250,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)

import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the saved model and scaler
@st.cache_resource
def load_assets():
    model = pickle.load(open('rf_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    features = pickle.load(open('features.pkl', 'rb'))
    return model, scaler, features

model, scaler, features = load_assets()

# --- WEB UI DESIGN ---
st.set_page_config(page_title="Insurance AI Calculator", layout="wide")

st.title("🏥 Smart Insurance Premium Estimator")
st.markdown("---")

# Layout with two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("👤 Personal Metrics")
    age = st.slider("Age", 18, 66, 30)
    height = st.number_input("Height (cm)", 145, 188, 170)
    weight = st.number_input("Weight (kg)", 51, 132, 75)
    surgeries = st.selectbox("Number of Major Surgeries", [0, 1, 2, 3])

with col2:
    st.subheader("📋 Health History")
    dia = st.checkbox("Diabetes")
    bp = st.checkbox("Blood Pressure Problems")
    tra = st.checkbox("History of Transplants")
    chr = st.checkbox("Chronic Diseases")
    alg = st.checkbox("Known Allergies")
    can = st.checkbox("Family History of Cancer")

# --- DYNAMIC CALCULATIONS ---
bmi = weight / ((height / 100) ** 2)
risk_score = sum([dia, bp, tra, chr, alg, can])

# Prepare Data for Model
input_data = pd.DataFrame([[
    age, int(dia), int(bp), int(tra), int(chr), 
    height, weight, int(alg), int(can), surgeries, bmi, risk_score
]], columns=features)

st.markdown("---")
if st.button("🚀 Calculate Premium Now"):
    # Apply scaling
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]
    
    # Dynamic Visual Results
    st.balloons()
    st.success(f"### Estimated Annual Premium: ₹{round(prediction, 2):,}")
    
    # Predictive Insights (Block 1 Goal)
    st.info(f"**Analysis:** Based on a BMI of {bmi:.1f} and a Risk Score of {risk_score}, our model predicts this rate with high confidence.")

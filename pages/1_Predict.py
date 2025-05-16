import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page config
st.set_page_config(
    page_title="Cardiovascular Disease Prediction",
    page_icon="🎯",
    layout="wide"
)

# Title and description
st.title("Cardiovascular Disease Prediction")
st.write("""
Enter the patient's information below to predict the likelihood of cardiovascular disease.
""")

# Load the model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('final_model.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Create input fields
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    height = st.number_input("Height (cm)", min_value=0, max_value=250, value=170)
    weight = st.number_input("Weight (kg)", min_value=0, max_value=300, value=70)
    ap_hi = st.number_input("Systolic Blood Pressure", min_value=0, max_value=300, value=120)
    ap_lo = st.number_input("Diastolic Blood Pressure", min_value=0, max_value=200, value=80)

with col2:
    cholesterol = st.selectbox("Cholesterol Level", ["Normal", "Above Normal", "Well Above Normal"])
    gluc = st.selectbox("Glucose Level", ["Normal", "Above Normal", "Well Above Normal"])
    smoke = st.checkbox("Smoking")
    alco = st.checkbox("Alcohol Consumption")
    active = st.checkbox("Physical Activity")

# Convert inputs to model format
def prepare_input():
    gender_val = 1 if gender == "Male" else 2
    cholesterol_map = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}
    gluc_map = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}
    
    # Calculate BMI
    height_m = height / 100
    bmi = weight / (height_m * height_m)
    
    # Calculate pulse pressure
    pulse_pressure = ap_hi - ap_lo
    
    # Calculate mean arterial pressure
    map_value = (ap_hi + 2 * ap_lo) / 3
    
    input_data = {
        'age': age,
        'gender': gender_val,
        'height': height,
        'weight': weight,
        'ap_hi': ap_hi,
        'ap_lo': ap_lo,
        'cholesterol': cholesterol_map[cholesterol],
        'gluc': gluc_map[gluc],
        'smoke': int(smoke),
        'alco': int(alco),
        'active': int(active),
        'bmi': bmi,
        'pulse_pressure': pulse_pressure,
        'map': map_value,
        'age_years': age,
        'weight_kg': weight
    }
    
    # Order features to match model
    feature_order = [
        'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
        'cholesterol', 'gluc', 'smoke', 'alco', 'active',
        'bmi', 'pulse_pressure', 'map', 'age_years', 'weight_kg'
    ]
    
    return pd.DataFrame([input_data])[feature_order]

# Make prediction
if st.button("Predict"):
    try:
        model = load_model()
        if model is None:
            st.error("Could not load the model. Please check the model file.")
        else:
            input_df = prepare_input()
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)
            
            st.write("---")
            if prediction[0] == 1:
                st.error("⚠️ High Risk of Cardiovascular Disease")
            else:
                st.success("✅ Low Risk of Cardiovascular Disease")
                
            st.write(f"Probability of Cardiovascular Disease: {probability[0][1]:.2%}")
            
            # Add risk factors analysis
            st.write("### Risk Factors Analysis")
            risk_factors = []
            if age > 50: risk_factors.append("Age > 50")
            if ap_hi > 140: risk_factors.append("High Systolic Blood Pressure")
            if ap_lo > 90: risk_factors.append("High Diastolic Blood Pressure")
            if bmi > 30: risk_factors.append("High BMI")
            if smoke: risk_factors.append("Smoking")
            if cholesterol != "Normal": risk_factors.append("Elevated Cholesterol")
            if gluc != "Normal": risk_factors.append("Elevated Glucose")
            if not active: risk_factors.append("Physical Inactivity")
            
            if risk_factors:
                st.write("Identified Risk Factors:")
                for factor in risk_factors:
                    st.write(f"- {factor}")
            else:
                st.write("No significant risk factors identified.")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

# Add footer
st.write("---")
st.write("Developed for Cardiovascular Disease Prediction") 
# streamlit_sickle_crisis_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

# ------------------------------
# Helper functions
# ------------------------------

def load_synthetic_data():
    # Load the synthetic dataset we generated earlier
    return pd.read_csv("sickle_synth.csv")

def preprocess_data(df, features, scaler=None):
    X = df[features]
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, scaler

def categorize_risk(prob):
    if prob < 0.3:
        return "Low"
    elif prob < 0.6:
        return "Medium"
    else:
        return "High"

def provide_tips(row):
    tips = []
    if row['malaria_level'] > 30:
        tips.append("Check for malaria and treat promptly")
    if row['typhoid'] == 1:
        tips.append("Monitor for typhoid symptoms")
    if row['pcv'] < 28:
        tips.append("Consider consulting doctor about low PCV")
    if row['temperature'] > 38:
        tips.append("Fever detected: hydrate and rest")
    if row['o2_sat'] < 94:
        tips.append("Low oxygen: seek medical attention if symptoms worsen")
    if row['pain_score'] > 5:
        tips.append("High pain score: monitor and manage pain")
    if row['ventilation'] == 1:
        tips.append("Recent ventilation: high risk, monitor closely")
    return tips

# ------------------------------
# Main App
# ------------------------------

st.set_page_config(page_title="Sickle Cell Crisis Predictor", layout="wide")
st.title("Sickle Cell Crisis Risk Predictor 🩸")

# Load synthetic data
synthetic_data = load_synthetic_data()
features = ['age', 'malaria_level', 'typhoid', 'pcv', 'temperature',
            'freq_crisis_last_year', 'ventilation', 'o2_sat', 'pain_score']

# ------------------------------
# Sidebar for CSV upload or manual input
# ------------------------------
st.sidebar.header("Input Options")
input_method = st.sidebar.radio("Select input method:", ["Manual Input", "CSV Upload"])

# Initialize empty dataframe for predictions
input_df = pd.DataFrame()

if input_method == "Manual Input":
    st.sidebar.subheader("Enter patient data")
    age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=10)
    sex = st.sidebar.selectbox("Sex", ["M", "F"])
    malaria_level = st.sidebar.slider("Malaria Level (0-100)", 0.0, 100.0, 10.0)
    typhoid = st.sidebar.selectbox("Typhoid Presence", [0,1])
    pcv = st.sidebar.slider("PCV (%)", 12.0, 45.0, 30.0)
    temperature = st.sidebar.slider("Temperature (°C)", 34.0, 41.0, 37.0)
    freq_crisis_last_year = st.sidebar.slider("Frequency of Crises Last Year", 0, 15, 0)
    ventilation = st.sidebar.selectbox("Recent Ventilation", [0,1])
    o2_sat = st.sidebar.slider("Oxygen Saturation (%)", 70.0, 100.0, 96.0)
    pain_score = st.sidebar.slider("Pain Score (0-10)", 0, 10, 2)

    input_df = pd.DataFrame([{
        'age': age,
        'malaria_level': malaria_level,
        'typhoid': typhoid,
        'pcv': pcv,
        'temperature': temperature,
        'freq_crisis_last_year': freq_crisis_last_year,
        'ventilation': ventilation,
        'o2_sat': o2_sat,
        'pain_score': pain_score
    }])

elif input_method == "CSV Upload":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.write("Uploaded CSV preview:")
        st.dataframe(input_df.head())

# ------------------------------
# Model Training (Logistic Regression)
# ------------------------------

if st.button("Predict Crisis Risk"):
    # Combine synthetic + uploaded data if CSV provided
    if not input_df.empty and input_method == "CSV Upload":
        combined_data = pd.concat([synthetic_data, input_df], ignore_index=True)
    else:
        combined_data = synthetic_data.copy()

    # Preprocess
    X_scaled, scaler = preprocess_data(combined_data, features)
    
    # Split X for prediction
    X_pred_scaled = scaler.transform(input_df[features])
    
    # Train logistic regression on combined data (synthetic + uploaded)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, combined_data['crisis_next_30d'])
    
    # Predict probabilities
    probs = model.predict_proba(X_pred_scaled)[:,1]
    risk_categories = [categorize_risk(p) for p in probs]
    
    # Add predictions to dataframe
    results_df = input_df.copy()
    results_df['crisis_prob'] = np.round(probs,3)
    results_df['risk_category'] = risk_categories
    results_df['tips'] = results_df.apply(provide_tips, axis=1)
    
    # Display results
    st.subheader("Prediction Results")
    st.dataframe(results_df)
    
    # Option to download predictions
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name='crisis_predictions.csv',
        mime='text/csv'
    )
    
    st.success("Prediction complete! ✅")

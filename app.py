import numpy as np
import joblib
import streamlit as st

# Load trained models
scaler = joblib.load('scaler.pkl')
RF_model = joblib.load('RF_model.pkl')

# Streamlit UI
st.title("Multi-Model Seat Prediction")
st.write("Enter details to predict the number of seats using different models")

# Input fields
votes = st.number_input("Votes", min_value=0, value=100000)
validVotesPercentage = st.number_input("Valid Votes Percentage", min_value=0.0, max_value=100.0)
totalMandates = st.number_input("Total Mandates", min_value=0, value=10)
Hondt = st.number_input("Hondt", min_value=0)

def predict(model, input_values):
    input_scaled = scaler.transform(input_values)
    return round(model.predict(input_scaled)[0])

if st.button("Predict Seats"):
    try:
        input_values = np.array([votes, validVotesPercentage, totalMandates , Hondt]).reshape(1, -1)
        dhondt_pred = predict(RF_model, input_values)
        
        st.write("### Predictions")
        st.write(f"D'Hondt Method: {dhondt_pred} seats")
    except Exception as e:
        st.error(f"Error: {e}")

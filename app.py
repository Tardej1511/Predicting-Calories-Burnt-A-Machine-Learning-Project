import streamlit as st
import numpy as np
import pandas as pd
import pickle

#load model
rfr = pickle.load(open('rfr.pkl','rb'))

#web app
st.title("Caloried Burn Prediction")


# Load the pre-trained model
with open('rfr.pkl', 'rb') as file:
    rfr = pickle.load(file)

# Load training data for reference (to use columns/features)
x_train = pd.read_csv('x_train.csv')

for estimator in rfr.estimators_:
    if not hasattr(estimator, 'monotonic_cst'):
        estimator.monotonic_cst = None


# Define the prediction function
def pred(Gender, Age, Height, Weight, Duration, Heart_rate, Body_temp):
    # Convert Gender to numeric (e.g., male: 1, female: 0)
    Gender = 1 if Gender == 'male' else 0

    # Create feature array
    features = np.array([[Gender, Age, Height, Weight, Duration, Heart_rate, Body_temp]])

    # Predict using the model
    prediction = rfr.predict(features)
    return prediction[0]  # Return the predicted calorie burn


# Streamlit web app
st.title("Calories Burn Prediction")

# Input fields
Gender = st.selectbox('Gender', ['male', 'female'])
Age = st.slider('Age', 10, 100, step=1)
Height = st.number_input('Height (cm)', min_value=100, max_value=250, step=1)
Weight = st.number_input('Weight (kg)', min_value=30, max_value=200, step=1)
Duration = st.number_input('Duration (minutes)', min_value=1, max_value=300, step=1)
Heart_rate = st.number_input('Heart Rate (bpm)', min_value=40, max_value=200, step=1)
Body_temp = st.number_input('Body Temperature (°C)', min_value=30.0, max_value=45.0, step=0.1)

# Predict button
if st.button('Predict'):
    result = pred(Gender, Age, Height, Weight, Duration, Heart_rate, Body_temp)
    st.write(f"Estimated Calories Burned: **{result:.2f} kcal**")

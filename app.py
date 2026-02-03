import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load Random Forest model and the exact column list
model = joblib.load('bike_rf_model.pkl')
model_columns = joblib.load('model_columns.pkl') # Ensure this contains the 14 names below

st.set_page_config(page_title="SME Bike Analytics", layout="wide")
st.title("🚲 SME Bike Demand: Random Forest Engine")

# --- MANUAL INPUT SECTION ---
st.sidebar.header("Quick Check (Manual Input)")

# Creating inputs for the 10 base features
temp = st.sidebar.slider("Temperature (°C)", 0.0, 45.0, 20.0)
atemp = st.sidebar.slider("Feels-Like Temp (°C)", 0.0, 50.0, 22.0)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 50)
windspeed = st.sidebar.slider("Windspeed (km/h)", 0.0, 60.0, 10.0)
hour = st.sidebar.number_input("Time (Hour 0-23)", 0, 23, 12)
month = st.sidebar.slider("Month (1-12)", 1, 12, 6)
day = st.sidebar.slider("Day of Month", 1, 31, 15)
weather = st.sidebar.selectbox("Weather Condition (1:Clear, 4:Heavy Rain)", [1, 2, 3, 4])
holiday = st.sidebar.radio("Is it a Holiday?", [0, 1])
workingday = st.sidebar.radio("Is it a Working Day?", [0, 1])
season_choice = st.sidebar.selectbox("Season", ["spring", "summer", "fall", "winter"])

if st.sidebar.button("Predict"):
    # Create a single row with all 14 columns initialized to 0
    input_data = pd.DataFrame(0.0, index=[0], columns=model_columns)
    
    # Fill numerical and categorical columns
    input_data['temp'] = temp
    input_data['atemp'] = atemp
    input_data['humidity'] = humidity
    input_data['windspeed'] = windspeed
    input_data['time'] = hour
    input_data['month'] = month
    input_data['day'] = day
    input_data['weather'] = weather
    input_data['holiday'] = holiday
    input_data['workingday'] = workingday
    
    # Fill One-Hot Encoded season columns
    if season_choice in input_data.columns:
        input_data[season_choice] = 1.0

    # Ensure the order is exactly what the model expects
    input_data = input_data[model_columns]

    prediction = model.predict(input_data)
    st.sidebar.metric("Bikes Needed", int(prediction[0]))

# --- BULK CSV SECTION ---
st.header("Bulk Restock Planning")
st.write("Upload a CSV with the following columns: " + ", ".join(model_columns))

uploaded_file = st.file_uploader("Upload your Weekly Forecast", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Check if all required columns are in the uploaded file
    missing_cols = [col for col in model_columns if col not in df.columns]
    
    if not missing_cols:
        # Use only the 14 columns in the correct order for prediction
        predictions = model.predict(df[model_columns])
        df['predicted_demand'] = predictions.astype(int)
        
        st.write("### Predicted Demand Results", df.head())
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Restock Report", data=csv, file_name='restock_report.csv')
    else:
        st.error(f"The uploaded CSV is missing these columns: {missing_cols}")
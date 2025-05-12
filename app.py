import streamlit as st
import joblib
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
# Load the image for the bus (as in your previous code)
img = Image.open("Bus-image.png")

# Set the title of the Streamlit app
st.title("Simulasi Kalibrasi Sensor Ban Bus")
st.markdown("Silakan input tegangan (dalam mV) pada masing-masing ban:")

# Display the image in the center column of a 3-column layout
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.image(img, caption="Klik tombol + pada posisi ban", width=200)

# Initialize session state if not already initialized
if 'voltages' not in st.session_state:
    st.session_state.voltages = {'Depan Kiri': None, 'Depan Kanan': None, 'Belakang Kiri': None, 'Belakang Kanan': None}
if 'prediksi' not in st.session_state:
    st.session_state.prediksi = {'Depan Kiri': None, 'Depan Kanan': None, 'Belakang Kiri': None, 'Belakang Kanan': None}

# Load the dataset (CSV file)
def load_data():
    file_path = 'truck_tire_depth_dataset_indonesia.csv'  # File is in the same folder as app.py
    df = pd.read_csv(file_path)
    return df

# Train model using the dataset
def train_model(df):
    X = df["CCD_Reading_mV"].values.reshape(-1, 1)  # CCD readings as feature
    y = df["Tread_Depth_mm"].values  # Depth as target

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict and calculate error metrics
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Output the model and metrics
    st.write(f"Model: depth_mm = {model.coef_[0]:.5f} * mV + {model.intercept_:.2f}")
    st.write(f"Mean Absolute Error (MAE): {mae:.3f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.3f}")

    # Save the model to a file
    with open("tire_depth_model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)
    
    return model

# Load the trained model
def load_model():
    try:
        model = joblib.load("tire_depth_model.pkl")
        return model
    except FileNotFoundError:
        return None

# Classify tire condition based on predicted tread depth
def classify_condition(depth):
    if depth < 1.6:
        return "Harus Diganti (Danger)"
    elif 1.6 <= depth < 3.0:
        return "Hampir Aus (Warning)"
    elif 3.0 <= depth < 5.0:
        return "Normal (Usia Pakai Pertengahan)"
    else:
        return "Baru (Optimal)"

# Function to handle user input and make predictions
def input_ban(posisi, model):
    tegangan = st.number_input(f"{posisi} (mV)", min_value=0.0, step=10.0, key=posisi)
    if tegangan > 0:
        # Use the model to predict the tread depth
        pred = model.predict([[tegangan]])[0]
        st.session_state.voltages[posisi] = tegangan
        st.session_state.prediksi[posisi] = pred
        condition = classify_condition(pred)
        st.write(f"→ Kedalaman: **{pred:.2f} mm**")
        st.write(f"→ Kondisi Ban: **{condition}**")

# Load the dataset and train the model if it's not trained yet
df = load_data()
model = load_model()

if model is None:
    st.write("Model tidak ditemukan, melatih model baru...")
    model = train_model(df)

# Display tire positions and inputs for the user
st.markdown("### Posisi Ban dan Input")
col1, col2 = st.columns(2)
with col1:
    input_ban("Depan Kiri", model)
with col2:
    input_ban("Depan Kanan", model)

col3, col4 = st.columns(2)
with col3:
    input_ban("Belakang Kiri", model)
with col4:
    input_ban("Belakang Kanan", model)

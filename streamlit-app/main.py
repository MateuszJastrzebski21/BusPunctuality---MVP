import os
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib
from prepare_input import prepare_input
from datetime import datetime



BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Definicja modelu FCNN (taka sama jak w treningu)
class FCNN(nn.Module):
    def __init__(self, input_size):
        super(FCNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.layers(x)

# Ścieżki do plików
MODEL_PATH = os.path.join(BASE_DIR, "fcnn_model.pth")
SCALER_X_PATH = os.path.join(BASE_DIR, "scaler_X.pkl")
SCALER_Y_PATH = os.path.join(BASE_DIR, "scaler_y.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "onehot_encoder.pkl")
PARQUET_PATH = os.path.join(BASE_DIR, "data", "huge_delays_removed_240625.parquet")

# Wczytanie modelu i pomocniczych obiektów
model = FCNN(input_size=19)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

scaler_X = joblib.load(SCALER_X_PATH)
scaler_y = joblib.load(SCALER_Y_PATH)
encoder = joblib.load(ENCODER_PATH)

# Interfejs użytkownika
st.title("Bus Delay Prediction")

stop_name = st.text_input("Stop Name (Przystanek nazwa)")
line = st.text_input("Line (np. 126)")
date = st.date_input("Date")
time = st.time_input("Time")

if st.button("Predict"):
    try:
        input_features = prepare_input(stop_name, line, date, time, PARQUET_PATH, encoder)

        # Skalowanie cech
        input_scaled = scaler_X.transform([input_features])
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

        # Predykcja
        with torch.no_grad():
            prediction_scaled = model(input_tensor)
            prediction = scaler_y.inverse_transform(prediction_scaled.numpy())[0][0]

        st.success(f"Predicted trip time: {prediction:.2f} seconds ({prediction/60:.2f} minutes)")

    except Exception as e:
        st.error(f"Error: {e}")

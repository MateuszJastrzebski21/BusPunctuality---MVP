# MAIN FUNCTION
# create ui in streamlit, load(open) models
# build form to get data about bus_line_number, bus_stop_name, date and time of departure
# on button save, run loaded model with prepared form data and return result for both models

# Interface

import streamlit as st
import pandas as pd

st.title("Bus Delay Prediction")

stop = st.text_input("Bus Stop")
line = st.text_input("Bus Line")
date = st.date_input("Date")
time = st.time_input("Time")

if st.button("Predict"):
    # input_df = prepare_input(stop, line, date, time)
    delay = 6.78 # model.predict(input_df)[0]
    st.success(f"Predicted delay: {delay:.2f} minutes")

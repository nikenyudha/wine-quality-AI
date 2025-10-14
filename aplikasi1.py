import streamlit as st
import pandas as pd
import joblib

model = joblib.load('best_pipeline.pkl')

st.title("🍷 Prediksi Kualitas Wine")

alcohol = st.number_input("Kadar Alkohol", 0.0, 20.0, 10.0)
sulphates = st.number_input("Sulphates", 0.0, 2.0, 0.5)
citric_acid = st.number_input("Citric Acid", 0.0, 1.0, 0.3)

input_data = pd.DataFrame({
    'alcohol': [alcohol],
    'sulphates': [sulphates],
    'citric acid': [citric_acid]
})

if st.button("Prediksi"):
    pred = model.predict(input_data)
    st.success(f"Hasil Prediksi Kualitas Wine: {pred[0]:.2f}")
    
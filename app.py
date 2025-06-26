import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model_dt.pkl")

st.title("Prediksi Kelangsungan Hidup Kuda 🐴")
st.write("Masukkan data pasien kuda di bawah ini:")

# Form input fitur (contoh: sesuaikan dengan fitur dari X_test/X_train)
age = st.selectbox("Usia", [1, 2, 3])
temp = st.number_input("Suhu tubuh (°C)", min_value=35.0, max_value=42.0, value=38.0)
pulse = st.number_input("Detak jantung", min_value=30, max_value=180, value=70)
resp = st.number_input("Frekuensi napas", min_value=10, max_value=100, value=30)

# Lanjutkan sesuai dengan fitur lain yang dipakai model...

# Prediksi
if st.button("Prediksi"):
    # Sesuaikan urutan dengan training data
    data = pd.DataFrame([{
        "age": age,
        "rectal_temp": temp,
        "pulse": pulse,
        "respiratory_rate": resp,
        # tambahkan fitur lain...
    }])
    
    prediction = model.predict(data)[0]
    label = "Selamat" if prediction == 1 else "Tidak Selamat"
    
    st.subheader("Hasil Prediksi:")
    st.success(f"Kuda diprediksi: **{label}**")

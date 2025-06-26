import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Menggunakan cache agar model hanya dimuat sekali untuk efisiensi
@st.cache_resource
def load_model():
    """Memuat pipeline model dari file .pkl"""
    try:
        pipeline = joblib.load('horse_colic_pipeline.pkl')
        return pipeline
    except FileNotFoundError:
        st.error("Error: File model 'horse_colic_pipeline.pkl' tidak ditemukan.")
        return None

# Memuat model saat aplikasi dimulai
pipeline = load_model()

# Konfigurasi dan judul halaman
st.set_page_config(page_title="Prediksi Kolik Kuda", page_icon="🐴", layout="wide")
st.title('🐴 Aplikasi Prediksi Kelangsungan Hidup Kuda')
st.write("Aplikasi ini menggunakan model **Decision Tree** untuk memprediksi kondisi kuda berdasarkan data klinis.")

# Hanya jalankan sisa aplikasi jika model berhasil dimuat
if pipeline:
    # Tata letak 2 kolom
    col1, col2 = st.columns(2)

    # Opsi input yang user-friendly
    surgery_options = {1.0: 'Ya, pernah dioperasi', 2.0: 'Tidak pernah dioperasi'}
    age_options = {1.0: 'Dewasa (> 6 bulan)', 2.0: 'Muda (< 6 bulan)'}
    pain_options = {
        1.0: 'Waspada, tidak sakit', 2.0: 'Depresi', 3.0: 'Nyeri ringan intermiten',
        4.0: 'Nyeri parah intermiten', 5.0: 'Nyeri parah berkelanjutan'
    }

    with col1:
        st.header("Data Klinis Utama")
        pulse = st.number_input('Denyut Nadi (pulse)', 30, 200, 60)
        rectal_temp = st.number_input('Suhu Rektal (°C)', 35.0, 42.0, 38.0, 0.1)
        respiratory_rate = st.number_input('Laju Pernapasan', 8, 100, 24)
        packed_cell_volume = st.number_input('Volume Sel Darah (%)', 20.0, 80.0, 45.0, 0.1)

    with col2:
        st.header("Observasi Kondisi")
        pain_label = st.selectbox('Tingkat Rasa Sakit (pain)', options=list(pain_options.values()))
        surgery_label = st.selectbox('Pernah Dioperasi? (surgery)', options=list(surgery_options.values()))
        age_label = st.selectbox('Usia (age)', options=list(age_options.values()))

    if st.button('Lakukan Prediksi', type="primary", use_container_width=True):
        # Mengonversi input teks kembali ke nilai numerik
        pain = [k for k, v in pain_options.items() if v == pain_label][0]
        surgery = [k for k, v in surgery_options.items() if v == surgery_label][0]
        age = [k for k, v in age_options.items() if v == age_label][0]
        
        # Membuat DataFrame tunggal dari input
        # Fitur lain diisi dengan NaN agar ditangani oleh imputer di dalam pipeline
        input_data = pd.DataFrame({
            'rectal_temperature': [rectal_temp], 'pulse': [pulse], 'respiratory_rate': [respiratory_rate],
            'packed_cell_volume': [packed_cell_volume], 'total_protein': [np.nan], 'surgery': [surgery],
            'age': [age], 'pain': [pain], 'temp_of_extremities': [np.nan], 'peripheral_pulse': [np.nan],
            'mucous_membrane': [np.nan], 'capillary_refill_time': [np.nan], 'peristalsis': [np.nan],
            'abdominal_distension': [np.nan], 'nasogastric_tube': [np.nan], 'nasogastric_reflux': [np.nan],
            'rectal_exam_feces': [np.nan], 'abdomen': [np.nan], 'abdomo_appearance': [np.nan],
            'abdomo_protein': [np.nan], 'surgical_lesion': [np.nan], 'lesion_1': [np.nan], 'lesion_2': [np.nan],
            'lesion_3': [np.nan], 'cp_data': [np.nan]
        })
        
        # Melakukan prediksi
        prediction = pipeline.predict(input_data)[0]
        prediction_proba = pipeline.predict_proba(input_data)[0]
        
        # Menampilkan hasil
        st.subheader('Hasil Prediksi:')
        if prediction == 1:
            st.success('**SELAMAT (Lived)**')
            st.write(f"Keyakinan model: **{prediction_proba[1]*100:.2f}%**")
        else:
            st.error('**TIDAK SELAMAT (Died or Euthanized)**')
            st.write(f"Keyakinan model: **{prediction_proba[0]*100:.2f}%**")
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Menggunakan cache agar model hanya dimuat sekali untuk efisiensi
@st.cache_resource
def load_model():
    """
    Fungsi ini memuat pipeline model yang telah disimpan dari file .pkl.
    Menangani error jika file tidak ditemukan.
    """
    try:
        pipeline = joblib.load('horse_colic_pipeline.pkl')
        return pipeline
    except FileNotFoundError:
        st.error("Error: File model 'horse_colic_pipeline.pkl' tidak ditemukan.")
        st.warning("Pastikan file model berada di folder yang sama dengan file 'app.py' ini.")
        return None

# Memuat pipeline model ke dalam aplikasi
pipeline = load_model()

# --- ANTARMUKA PENGGUNA (USER INTERFACE) ---

# Konfigurasi halaman web (judul di tab browser dan ikon)
st.set_page_config(page_title="Prediksi Kolik Kuda", page_icon="🐴", layout="wide")

# Judul utama aplikasi
st.title('🐴 Aplikasi Prediksi Kelangsungan Hidup Kuda')

# Deskripsi singkat aplikasi
st.write("""
Aplikasi ini adalah hasil deployment dari model machine learning **Decision Tree**. 
Model ini dilatih untuk memprediksi apakah seekor kuda akan selamat atau tidak 
berdasarkan data klinis yang dimasukkan.
""")

# Hanya tampilkan sisa aplikasi jika model berhasil dimuat
if pipeline:
    # Membuat 2 kolom untuk tata letak input yang lebih rapi
    col1, col2 = st.columns(2)

    # Opsi input yang user-friendly untuk dropdown menu
    surgery_options = {1.0: 'Ya, pernah dioperasi', 2.0: 'Tidak pernah dioperasi'}
    age_options = {1.0: 'Dewasa (> 6 bulan)', 2.0: 'Muda (< 6 bulan)'}
    pain_options = {
        1.0: 'Waspada, tidak sakit', 2.0: 'Depresi', 3.0: 'Nyeri ringan intermiten',
        4.0: 'Nyeri parah intermiten', 5.0: 'Nyeri parah berkelanjutan'
    }

    # --- INPUT FORM DI KOLOM 1 ---
    with col1:
        st.header("Data Klinis Utama")
        pulse = st.number_input(
            label='Denyut Nadi (pulse)', 
            min_value=30, max_value=200, value=60, step=1,
            help="Denyut jantung per menit. Normal untuk kuda dewasa: 30-40."
        )
        rectal_temp = st.number_input(
            label='Suhu Rektal (°C)', 
            min_value=35.0, max_value=42.0, value=38.0, step=0.1,
            help="Dalam Celcius. Normal: 37.8°C."
        )
        respiratory_rate = st.number_input(
            label='Laju Pernapasan', 
            min_value=8, max_value=100, value=24, step=1,
            help="Napas per menit. Normal: 8-10."
        )
        packed_cell_volume = st.number_input(
            label='Volume Sel Darah (%)', 
            min_value=20.0, max_value=80.0, value=45.0, step=0.1,
            help="Persentase sel darah merah dalam darah. Normal: 30-50%."
        )

    # --- INPUT FORM DI KOLOM 2 ---
    with col2:
        st.header("Observasi Kondisi")
        pain_label = st.selectbox('Tingkat Rasa Sakit (pain)', options=list(pain_options.values()))
        surgery_label = st.selectbox('Pernah Dioperasi? (surgery)', options=list(surgery_options.values()))
        age_label = st.selectbox('Usia (age)', options=list(age_options.values()))

    # --- TOMBOL PREDIKSI DAN LOGIKA BACKEND ---
    if st.button('Lakukan Prediksi', type="primary", use_container_width=True):
        
        # Mengonversi kembali input dari label teks ke nilai numerik yang dipahami model
        pain = [k for k, v in pain_options.items() if v == pain_label][0]
        surgery = [k for k, v in surgery_options.items() if v == surgery_label][0]
        age = [k for k, v in age_options.items() if v == age_label][0]
        
        # Membuat DataFrame dari input pengguna.
        # Nama kolom harus sama persis dengan yang digunakan saat training.
        # Fitur yang tidak diinput pengguna akan diisi dengan np.nan,
        # dan akan ditangani secara otomatis oleh imputer di dalam pipeline kita.
        input_data = pd.DataFrame({
            'rectal_temperature': [rectal_temp], 'pulse': [pulse], 'respiratory_rate': [respiratory_rate],
            'packed_cell_volume': [packed_cell_volume], 'total_protein': [np.nan], 'surgery': [surgery],
            'age': [age], 'pain': [pain], 'temp_of_extremities': [np.nan], 'peripheral_pulse': [np.nan],
            'mucous_membranes': [np.nan], 'capillary_refill_time': [np.nan], 'peristalsis': [np.nan],
            'abdominal_distension': [np.nan], 'nasogastric_tube': [np.nan], 'nasogastric_reflux': [np.nan],
            'rectal_examination_feces': [np.nan], 'abdomen': [np.nan], 'abdomo_appearance': [np.nan],
            'abdomo_protein': [np.nan], 'surgical_lesion': [np.nan], 'lesion_site': [0], 'lesion_type': [0],
            'lesion_subtype': [0], 'cp_data': [np.nan]
        })
        
        # Melakukan prediksi dan mendapatkan probabilitas
        prediction = pipeline.predict(input_data)[0]
        prediction_proba = pipeline.predict_proba(input_data)[0]
        
        # Menampilkan hasil prediksi
        st.subheader('Hasil Prediksi:')
        if prediction == 1:
            st.success('**SELAMAT (Lived)**')
            st.write(f"Keyakinan model bahwa kuda akan selamat: **{prediction_proba[1]*100:.2f}%**")
        else:
            st.error('**TIDAK SELAMAT (Died or Euthanized)**')
            st.write(f"Keyakinan model bahwa kuda tidak akan selamat: **{prediction_proba[0]*100:.2f}%**")

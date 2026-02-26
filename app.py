import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# --- 1. Muat Model dan Scaler yang Tersimpan ---
@st.cache_resource
def load_model_and_scaler():
    try:
        with open('gradient_boosting_model.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
        with open('feature_scaler.pkl', 'rb') as file:
            loaded_scaler = pickle.load(file)
        return loaded_model, loaded_scaler
    except FileNotFoundError:
        st.error("File model atau scaler tidak ditemukan. Pastikan 'gradient_boosting_model.pkl' dan 'feature_scaler.pkl' ada di direktori yang sama.")
        st.stop()

loaded_model, loaded_scaler = load_model_and_scaler()

# --- 2. Konfigurasi Aplikasi Streamlit ---
st.set_page_config(page_title="Prediksi Gaji Awal Lulusan Vokasi", layout="centered")
st.title("💰 Prediksi Gaji Awal Lulusan Pelatihan Vokasi")
st.markdown("Aplikasi ini memprediksi gaji awal (dalam Juta Rupiah) berdasarkan profil peserta pelatihan.")

# --- 3. Input Data dari Pengguna ---
st.header("Input Data Peserta")

usia = st.slider("Usia (tahun)", min_value=18, max_value=60, value=25)
durasi_jam = st.slider("Durasi Pelatihan (jam)", min_value=20, max_value=100, value=40)
nilai_ujian = st.slider("Nilai Ujian (skala 0-100)", min_value=0.0, max_value=100.0, value=75.0, step=0.1)

pendidikan_options = ['SMA', 'SMK', 'D3', 'S1', 'SMP'] # Dari df_bersih['Pendidikan'].unique()
pendidikan = st.selectbox("Pendidikan Terakhir", pendidikan_options)

jurusan_options = ['Administrasi', 'Teknik Las', 'Desain Grafis', 'Teknik Listrik', 'Perhotelan'] # Dari df_bersih['Jurusan'].unique()
jurusan = st.selectbox("Jurusan Pelatihan", jurusan_options)

jenis_kelamin_options = ['Laki-laki', 'Wanita'] # Dari df_bersih['Jenis_Kelamin'].unique() setelah cleaning
jenis_kelamin = st.selectbox("Jenis Kelamin", jenis_kelamin_options)

status_bekerja_options = ['Belum Bekerja', 'Sudah Bekerja'] # Dari df_bersih['Status_Bekerja'].unique()
status_bekerja = st.selectbox("Status Bekerja Sebelumnya", status_bekerja_options)

# --- 4. Tombol Prediksi ---
if st.button("Prediksi Gaji Awal"): 
    # --- 5. Preprocessing Data Input Baru ---
    input_data = {
        'Usia': usia,
        'Durasi_Jam': durasi_jam,
        'Nilai_Ujian': nilai_ujian,
        'Pendidikan': pendidikan,
        'Jurusan': jurusan,
        'Jenis_Kelamin': jenis_kelamin,
        'Status_Bekerja': status_bekerja
    }
    df_new_sample = pd.DataFrame([input_data])

    # Re-initialize LabelEncoders with all known categories from training
    le_pendidikan = LabelEncoder()
    le_pendidikan.fit(pendidikan_options) # Fit with all possible options
    df_new_sample['Pendidikan'] = le_pendidikan.transform(df_new_sample['Pendidikan'])

    le_jurusan = LabelEncoder()
    le_jurusan.fit(jurusan_options) # Fit with all possible options
    df_new_sample['Jurusan'] = le_jurusan.transform(df_new_sample['Jurusan'])

    # One-Hot Encoding
    original_one_hot_columns = [
        'Jenis_Kelamin_Laki-laki', 'Jenis_Kelamin_Wanita',
        'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja'
    ]
    df_one_hot_temp = pd.get_dummies(df_new_sample[['Jenis_Kelamin', 'Status_Bekerja']])
    df_one_hot_processed = df_one_hot_temp.reindex(columns=original_one_hot_columns, fill_value=0)
    df_one_hot_processed = df_one_hot_processed.astype(int)

    # Mengidentifikasi kolom numerik
    numerical_cols = ['Usia', 'Durasi_Jam', 'Nilai_Ujian']

    # Menggabungkan semua fitur yang sudah di-preproses
    feature_cols = ['Pendidikan', 'Jurusan', 'Jenis_Kelamin_Laki-laki', 'Jenis_Kelamin_Wanita',
                    'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja',
                    'Usia', 'Durasi_Jam', 'Nilai_Ujian']

    final_input_dict = {}
    final_input_dict['Pendidikan'] = df_new_sample['Pendidikan'].values
    final_input_dict['Jurusan'] = df_new_sample['Jurusan'].values
    for col in original_one_hot_columns:
        final_input_dict[col] = df_one_hot_processed[col].values
    for col in numerical_cols:
        final_input_dict[col] = df_new_sample[col].values

    df_final_input_for_prediction = pd.DataFrame(final_input_dict, columns=feature_cols)

    # Scaling
    scaled_input_data_array = loaded_scaler.transform(df_final_input_for_prediction)
    scaled_input_data_df = pd.DataFrame(scaled_input_data_array, columns=feature_cols)

    # --- 6. Melakukan Prediksi ---
    prediction = loaded_model.predict(scaled_input_data_df)

    # --- 7. Tampilkan Hasil Prediksi ---
    st.subheader("Hasil Prediksi Gaji Awal")
    st.success(f"Prediksi Gaji Awal: **{prediction[0]:.2f} Juta Rupiah**")
    st.info("*(Prediksi ini adalah estimasi dan dapat bervariasi tergantung pada banyak faktor)*")

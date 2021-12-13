import streamlit as st
import pickle
import numpy as np


def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        datas = pickle.load(file)
    return datas

datas = load_model()

regressor = datas["model"]
LE_kabupaten = datas["nama_kabupaten_kota"]
LE_tahun = datas["tahun"]

def show_predict_page():
    st.title("Prediksi Kenaikan Kendaraan Pada Kabupaten di Jawa Barat")

    st.write("""### We need some information to predict ###""")

    kabupaten = (
        'KABUPATEN BOGOR', 
        'KABUPATEN SUKABUMI',
        'KABUPATEN CIANJUR',
        'KABUPATEN BANDUNG', 
        'KABUPATEN GARUT', 
        'KABUPATEN TASIKMALAYA',
        'KABUPATEN CIAMIS', 
        'KABUPATEN KUNINGAN', 
        'KABUPATEN CIREBON',
        'KABUPATEN MAJALENGKA', 
        'KABUPATEN SUMEDANG',
        'KABUPATEN INDRAMAYU', 
        'KABUPATEN SUBANG', 
        'KABUPATEN PURWAKARTA',
        'KABUPATEN KARAWANG', 
        'KABUPATEN BEKASI',
        'KABUPATEN BANDUNG BARAT', 
        'KOTA BOGOR', 
        'KOTA SUKABUMI',
        'KOTA BANDUNG', 
        'KOTA CIREBON', 
        'KOTA BEKASI', 
        'KOTA DEPOK',
        'KOTA CIMAHI', 
        'KOTA TASIKMALAYA', 
        'KOTA BANJAR',
        'KABUPATEN PANGANDARAN'
    )


    kabupaten = st.selectbox("kabupaten", kabupaten)

    tahun = st.slider("tahun", 2013, 2025, 2013)

    ok = st.button("-- Calculate --")
    if ok:
        x = np.array([[kabupaten, tahun ]])
        x[:, 0] = LE_kabupaten.transform(x[:,0])
        x[:, 1] = LE_tahun.transform(x[:,1])
        x = x.astype(float)

        hasil = regressor.predict(x)
        st.subheader(f"jumlah peningkatan kendaraan :  {hasil[0]:.2f}")

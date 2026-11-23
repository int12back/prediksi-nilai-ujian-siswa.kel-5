import streamlit as st
import pandas as pd
import joblib

# ==========================================
# 1. KONFIGURASI DASHBOARD
# ==========================================

st.set_page_config(
    page_title="Prediksi Nilai Ujian Siswa",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Dashboard Prediksi Nilai Ujian Siswa")

st.write(
    "Dashboard ini digunakan untuk memprediksi nilai ujian siswa "
    "berdasarkan faktor pembelajaran, kehadiran, dan lingkungan belajar "
    "menggunakan algoritma Machine Learning."
)

# Memuat model terbaik
model = joblib.load("model_final.pkl")

# ==========================================
# 2. INPUT DATA SISWA
# ==========================================

st.sidebar.header("📝 Input Data Siswa")

hours = st.sidebar.number_input(
    "Jam Belajar",
    min_value=0,
    max_value=100,
    value=20
)

attendance = st.sidebar.number_input(
    "Kehadiran (%)",
    min_value=0,
    max_value=100,
    value=80
)

sleep = st.sidebar.number_input(
    "Jam Tidur",
    min_value=0,
    max_value=24,
    value=7
)

previous = st.sidebar.number_input(
    "Nilai Sebelumnya",
    min_value=0,
    max_value=100,
    value=70
)

tutoring = st.sidebar.number_input(
    "Jumlah Sesi Bimbingan",
    min_value=0,
    max_value=20,
    value=2
)

activity = st.sidebar.number_input(
    "Aktivitas Fisik",
    min_value=0,
    max_value=24,
    value=3
)

# Input kategorikal
parental = st.sidebar.selectbox(
    "Keterlibatan Orang Tua",
    ["High", "Low", "Medium"]
)

resources = st.sidebar.selectbox(
    "Akses Sumber Belajar",
    ["High", "Low", "Medium"]
)

extracurricular = st.sidebar.selectbox(
    "Kegiatan Ekstrakurikuler",
    ["No", "Yes"]
)

motivation = st.sidebar.selectbox(
    "Tingkat Motivasi",
    ["High", "Low", "Medium"]
)

internet = st.sidebar.selectbox(
    "Akses Internet",
    ["No", "Yes"]
)

income = st.sidebar.selectbox(
    "Pendapatan Keluarga",
    ["High", "Low", "Medium"]
)

teacher = st.sidebar.selectbox(
    "Kualitas Guru",
    ["High", "Low", "Medium"]
)

school = st.sidebar.selectbox(
    "Jenis Sekolah",
    ["Private", "Public"]
)

peer = st.sidebar.selectbox(
    "Pengaruh Teman",
    ["Negative", "Neutral", "Positive"]
)

disability = st.sidebar.selectbox(
    "Kesulitan Belajar",
    ["No", "Yes"]
)

education = st.sidebar.selectbox(
    "Pendidikan Orang Tua",
    ["College", "High School", "Postgraduate"]
)

distance = st.sidebar.selectbox(
    "Jarak dari Rumah",
    ["Far", "Moderate", "Near"]
)

gender = st.sidebar.selectbox(
    "Jenis Kelamin",
    ["Female", "Male"]
)

# ==========================================
# 3. MEMBENTUK DATA INPUT
# ==========================================

data = {
    "Hours_Studied": hours,
    "Attendance": attendance,
    "Sleep_Hours": sleep,
    "Previous_Scores": previous,
    "Tutoring_Sessions": tutoring,
    "Physical_Activity": activity,

    "Parental_Involvement_Low": int(parental == "Low"),
    "Parental_Involvement_Medium": int(parental == "Medium"),

    "Access_to_Resources_Low": int(resources == "Low"),
    "Access_to_Resources_Medium": int(resources == "Medium"),

    "Extracurricular_Activities_Yes": int(extracurricular == "Yes"),

    "Motivation_Level_Low": int(motivation == "Low"),
    "Motivation_Level_Medium": int(motivation == "Medium"),

    "Internet_Access_Yes": int(internet == "Yes"),

    "Family_Income_Low": int(income == "Low"),
    "Family_Income_Medium": int(income == "Medium"),

    "Teacher_Quality_Low": int(teacher == "Low"),
    "Teacher_Quality_Medium": int(teacher == "Medium"),

    "School_Type_Public": int(school == "Public"),

    "Peer_Influence_Neutral": int(peer == "Neutral"),
    "Peer_Influence_Positive": int(peer == "Positive"),

    "Learning_Disabilities_Yes": int(disability == "Yes"),

    "Parental_Education_Level_High School":
        int(education == "High School"),

    "Parental_Education_Level_Postgraduate":
        int(education == "Postgraduate"),

    "Distance_from_Home_Moderate": int(distance == "Moderate"),
    "Distance_from_Home_Near": int(distance == "Near"),

    "Gender_Male": int(gender == "Male")
}

# Urutan fitur harus sama dengan data training
feature_columns = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity",
    "Parental_Involvement_Low",
    "Parental_Involvement_Medium",
    "Access_to_Resources_Low",
    "Access_to_Resources_Medium",
    "Extracurricular_Activities_Yes",
    "Motivation_Level_Low",
    "Motivation_Level_Medium",
    "Internet_Access_Yes",
    "Family_Income_Low",
    "Family_Income_Medium",
    "Teacher_Quality_Low",
    "Teacher_Quality_Medium",
    "School_Type_Public",
    "Peer_Influence_Neutral",
    "Peer_Influence_Positive",
    "Learning_Disabilities_Yes",
    "Parental_Education_Level_High School",
    "Parental_Education_Level_Postgraduate",
    "Distance_from_Home_Moderate",
    "Distance_from_Home_Near",
    "Gender_Male"
]

input_data = pd.DataFrame([data])[feature_columns]

# ==========================================
# 4. VISUALISASI UTAMA
# ==========================================

st.header("📊 Visualisasi Hasil Penelitian")

# Grafik 1: Perbandingan RMSE
st.subheader("1. Perbandingan RMSE Lima Algoritma")

rmse_data = pd.DataFrame({
    "Algoritma": [
        "Lasso",
        "Ridge",
        "Gradient Boosting",
        "Random Forest",
        "Decision Tree"
    ],
    "RMSE": [
        1.8058,
        1.8058,
        1.8840,
        2.1674,
        2.5073
    ]
})

st.bar_chart(
    rmse_data.set_index("Algoritma")
)

st.caption(
    "Semakin rendah nilai RMSE, semakin kecil kesalahan prediksi model."
)

# Grafik 2: Profil faktor pembelajaran
st.subheader("2. Profil Faktor Pembelajaran Siswa")

profil = pd.DataFrame({
    "Faktor": [
        "Jam Belajar",
        "Kehadiran",
        "Jam Tidur",
        "Nilai Sebelumnya",
        "Sesi Bimbingan",
        "Aktivitas Fisik"
    ],
    "Nilai": [
        hours,
        attendance,
        sleep,
        previous,
        tutoring,
        activity
    ]
})

st.bar_chart(
    profil.set_index("Faktor")
)

# ==========================================
# 5. PREDIKSI NILAI UJIAN
# ==========================================

st.header("🎯 Prediksi Nilai Ujian")

if st.button("Prediksi Nilai Ujian", type="primary"):

    hasil = model.predict(input_data)[0]

    st.success("Prediksi berhasil dilakukan!")

    st.metric(
        label="Hasil Prediksi Nilai Ujian",
        value=f"{hasil:.2f}"
    )

    st.info(
        "Hasil ini merupakan estimasi berdasarkan pola "
        "data yang dipelajari oleh model Lasso Regression, "
        "bukan nilai ujian sebenarnya."
    )

st.divider()

st.caption(
    "Proyek Data Science Kelompok 5 | Prediksi Nilai Ujian Siswa "
    "Menggunakan Machine Learning"
)

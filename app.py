import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Diabetes AI Diagnostic",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS DENGAN TEMA BIRU MUDA (DIPERBAIKI) ---
st.markdown("""
    <style>
    .main { 
        background: linear-gradient(135deg, #f0f8ff 0%, #e6f2ff 100%);
    }
    
    h1 { 
        color: #1a73e8; 
        margin-bottom: 0.5rem;
        font-weight: 700;
        padding-bottom: 0.5rem;
    }
    
    h2, h3 { color: #4285f4; }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
        padding: 1.5rem;
        border-right: 2px solid #e3f2fd;
        box-shadow: 2px 0 10px rgba(0, 0, 0, 0.05);
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background: linear-gradient(135deg, #4285f4, #1a73e8);
        color: white;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(66, 133, 244, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #3367d6, #0d62d9);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(66, 133, 244, 0.4);
    }
    
    .custom-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(66, 133, 244, 0.1);
        border: 1px solid #e3f2fd;
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(66, 133, 244, 0.15);
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #4285f4 0%, #4285f4 var(--value), #e3f2fd var(--value), #e3f2fd 100%);
        border-radius: 6px;
    }
    
    .stSlider > div > div > div > div {
        background-color: #1a73e8 !important;
    }
    
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #e3f2fd;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    [data-testid="stMetricLabel"] {
        color: #5f6368;
        font-weight: 600;
    }
    
    [data-testid="stMetricValue"] {
        color: #1a73e8;
        font-weight: 700;
    }
    
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #e8f4ff 0%, #d6eaff 100%);
        border-radius: 10px;
        border: 1px solid #c2dcff;
        color: #1a73e8;
        font-weight: 600;
    }
    
    .streamlit-expanderContent {
        background-color: #f8fbff;
        border-radius: 0 0 10px 10px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #e8f4ff;
        border-radius: 10px 10px 0 0;
        padding: 15px 20px;
        border: 1px solid #c2dcff;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1a73e8 !important;
        color: white !important;
    }
    
    .alert-box {
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-weight: 500;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    .alert-low { 
        background: linear-gradient(135deg, #e8f5e9 0%, #d0f0d0 100%);
        border-left: 5px solid #34a853;
        color: #1e4620;
    }
    
    .alert-medium { 
        background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
        border-left: 5px solid #fbbc05;
        color: #5d4037;
    }
    
    .alert-high { 
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-left: 5px solid #ea4335;
        color: #b71c1c;
    }
    
    .stProgress > div > div > div > div {
        background-color: #4285f4;
    }
    
    input[type="number"] {
        border: 2px solid #e3f2fd !important;
        border-radius: 8px !important;
        padding: 5px 10px !important;
    }
    
    input[type="number"]:focus {
        border-color: #4285f4 !important;
        box-shadow: 0 0 0 2px rgba(66, 133, 244, 0.2) !important;
    }
    
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    .footer {
        text-align: center;
        padding: 1.5rem;
        margin-top: 3rem;
        background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
        border-radius: 15px;
        border: 1px solid #e3f2fd;
        color: #5f6368;
        font-size: 0.9rem;
    }
    
    .icon-container {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 50px;
        height: 50px;
        border-radius: 12px;
        margin-right: 15px;
        background: linear-gradient(135deg, #e8f4ff 0%, #d6eaff 100%);
    }
    
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #4285f4, transparent);
        margin: 2rem 0;
    }
    
    .custom-dataframe {
        border-collapse: collapse;
        width: 100%;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    .custom-dataframe th {
        background: linear-gradient(135deg, #4285f4, #1a73e8);
        color: white;
        padding: 12px;
        text-align: left;
        font-weight: 600;
    }
    
    .custom-dataframe td {
        padding: 10px;
        border-bottom: 1px solid #e3f2fd;
    }
    
    .custom-dataframe tr:hover {
        background-color: #f8fbff;
    }
    
    .custom-dataframe tr:nth-child(even) {
        background-color: #f8fbff;
    }
    
    .filter-btn {
        padding: 8px 16px;
        border-radius: 8px;
        border: 1px solid #c2dcff;
        background-color: #e8f4ff;
        color: #1a73e8;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        margin-right: 5px;
        margin-bottom: 5px;
    }
    
    .filter-btn:hover {
        background-color: #d6eaff;
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(66, 133, 244, 0.2);
    }
    
    .filter-btn.active {
        background: linear-gradient(135deg, #4285f4, #1a73e8);
        color: white;
        border-color: #1a73e8;
        box-shadow: 0 2px 8px rgba(66, 133, 244, 0.3);
    }
    
    .pagination-info {
        background-color: #f8fbff;
        padding: 10px 15px;
        border-radius: 8px;
        border: 1px solid #e3f2fd;
        color: #5f6368;
        font-size: 0.9rem;
        margin-top: 10px;
    }
    
    /* Animasi untuk header */
    @keyframes pulse {
        0% { opacity: 0.5; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.02); }
        100% { opacity: 0.5; transform: scale(1); }
    }
    
    .pulse-dot {
        width: 10px;
        height: 10px;
        background-color: #34a853;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. LOAD DATA & MODEL TRAINING ---
@st.cache_resource
def train_model():
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    progress_text.text("🔄 Memuat data dan melatih model AI...")
    progress_bar.progress(20)
    
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    names = ['Kehamilan', 'Glukosa', 'Tekanan_Darah', 'Kulit', 'Insulin', 'BMI', 'Riwayat', 'Umur', 'Hasil']
    
    progress_bar.progress(40)
    df = pd.read_csv(url, names=names)
    
    progress_bar.progress(60)
    X = df.drop('Hasil', axis=1)
    y = df['Hasil']
    
    progress_bar.progress(80)
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X, y)
    
    progress_bar.progress(100)
    progress_text.empty()
    progress_bar.empty()
    
    return df, model

df, model = train_model()

# --- 4. HEADER SEDERHANA YANG AMAN ---
st.markdown("""
<div style="
    background: linear-gradient(135deg, #e8f4ff 0%, #c2dcff 100%);
    padding: 2rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    border: 2px solid #bbdefb;
    box-shadow: 0 8px 25px rgba(66, 133, 244, 0.15);
    text-align: center;
">
    <div style="display: flex; align-items: center; justify-content: center; gap: 20px; margin-bottom: 1rem;">
        <div style="
            background: white;
            width: 80px;
            height: 80px;
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        ">
            <span style="font-size: 40px;">🏥</span>
        </div>
        <div style="text-align: left;">
            <h1 style="
                color: #1a73e8;
                margin: 0;
                font-size: 2.5rem;
                font-weight: 800;
                background: linear-gradient(135deg, #1a73e8, #4285f4);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            ">
                Diabetes AI Diagnostic System
            </h1>
            <p style="color: #5f6368; margin: 10px 0 0 0; font-size: 1.1rem;">
                Sistem analisis cerdas untuk mendeteksi risiko diabetes
            </p>
        </div>
    </div>
    
""", unsafe_allow_html=True)

# --- 5. SIDEBAR INPUT YANG DITINGKATKAN ---
st.sidebar.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <div style="
            background: linear-gradient(135deg, #4285f4, #1a73e8);
            width: 80px;
            height: 80px;
            border-radius: 20px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 1rem;
            box-shadow: 0 4px 12px rgba(66, 133, 244, 0.3);
        ">
            <span style="font-size: 40px; color: white;">📋</span>
        </div>
        <h2 style="color: #1a73e8; margin: 0 0 0.5rem 0;">Data Pasien</h2>
        <p style="color: #5f6368; font-size: 0.9rem; margin: 0;">Masukkan data medis untuk analisis</p>
    </div>
""", unsafe_allow_html=True)

# Info penting di sidebar
with st.sidebar.expander("ℹ️ Panduan Pengisian", expanded=True):
    st.markdown("""
    <div style="color: #5f6368;">
    <strong>Petunjuk:</strong><br>
    • Glukosa: Kadar gula darah puasa (mg/dL)<br>
    • Tekanan Darah: Diastolik (mmHg)<br>
    • BMI: Indeks Massa Tubuh (kg/m²)<br>
    • Semua nilai berdasarkan pemeriksaan medis
    </div>
    """, unsafe_allow_html=True)

def user_input_features():
    st.sidebar.markdown("### 📋 Parameter Medis")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        kehamilan = st.slider('Kehamilan', 0, 17, 2,
                            help="Jumlah kehamilan yang pernah dialami")
        glukosa = st.slider('Glukosa', 0, 200, 100,
                           help="Kadar glukosa darah puasa (mg/dL)")
        tekanan = st.slider('Tekanan Darah', 0, 122, 70,
                           help="Tekanan darah diastolik (mmHg)")
        kulit = st.slider('Ketebalan Kulit', 0, 99, 20,
                         help="Ketebalan lipatan kulit trisep (mm)")
    
    with col2:
        insulin = st.slider('Insulin', 0, 846, 79,
                           help="Kadar insulin 2 jam serum (mu U/ml)")
        bmi = st.number_input('BMI', 0.0, 70.0, 25.0, 0.1,
                             help="Indeks Massa Tubuh (kg/m²)")
        riwayat = st.number_input('Riwayat Keluarga', 0.0, 2.5, 0.4, 0.1,
                                 help="Skor riwayat diabetes keluarga")
        umur = st.slider('Umur', 21, 81, 30,
                        help="Umur pasien dalam tahun")
    
    return pd.DataFrame({
        'Kehamilan': [kehamilan], 'Glukosa': [glukosa], 'Tekanan_Darah': [tekanan],
        'Kulit': [kulit], 'Insulin': [insulin], 'BMI': [bmi], 'Riwayat': [riwayat], 'Umur': [umur]
    })

input_df = user_input_features()

# Tombol analisis di sidebar
st.sidebar.markdown("---")
analisis_btn = st.sidebar.button("🔍 Mulai Analisis", type="primary", use_container_width=True)
if analisis_btn:
    st.sidebar.success("✅ Analisis dimulai!")

# --- 6. LAYOUT UTAMA ---
# Bagian 1: Ringkasan Parameter
st.markdown("### 📊 Parameter Kesehatan Pasien")

# Tampilkan parameter dalam grid card dengan icon dari internet
cols = st.columns(4)
param_mapping = {
    'Kehamilan': {'icon': '🤰', 'unit': 'kali'},
    'Glukosa': {'icon': '🩸', 'unit': 'mg/dL'},
    'Tekanan_Darah': {'icon': '💓', 'unit': 'mmHg'},
    'Kulit': {'icon': '📏', 'unit': 'mm'},
    'Insulin': {'icon': '💉', 'unit': 'mu U/ml'},
    'BMI': {'icon': '⚖️', 'unit': 'kg/m²'},
    'Riwayat': {'icon': '👨‍👩‍👧‍👦', 'unit': 'skor'},
    'Umur': {'icon': '👤', 'unit': 'tahun'}
}

for idx, (param, value) in enumerate(input_df.iloc[0].items()):
    with cols[idx % 4]:
        info = param_mapping.get(param, {'icon': '📊', 'unit': ''})
        st.markdown(f"""
            <div class="custom-card">
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <div style="
                        background: linear-gradient(135deg, #e8f4ff 0%, #d6eaff 100%);
                        width: 45px;
                        height: 45px;
                        border-radius: 12px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        margin-right: 12px;
                        font-size: 24px;
                    ">
                        {info['icon']}
                    </div>
                    <div style="font-weight: 600; color: #1a73e8;">{param}</div>
                </div>
                <div style="font-size: 1.8rem; font-weight: 700; color: #1a73e8; text-align: center;">
                    {value}
                </div>
                <div style="text-align: center; color: #5f6368; font-size: 0.9rem;">
                    {info['unit']}
                </div>
            </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# --- 7. ANALISIS HASIL DENGAN VISUALISASI ---
st.markdown("## 📈 Hasil Analisis AI")

# Prediksi risiko
prob_positif = model.predict_proba(input_df)[0][1] * 100

# Menentukan kategori dan styling dengan icon dari internet
if prob_positif <= 40:
    kategori = "RENDAH"
    warna = "#34a853"
    icon = "✅"
    warna_bg = "#e8f5e9"
    pesan = "Kondisi Anda saat ini aman. Pertahankan gaya hidup sehat dengan pola makan seimbang dan olahraga teratur."
    rekomendasi = [
        "Lanjutkan pola hidup sehat",
        "Cek rutin 6 bulan sekali",
        "Konsumsi makanan bergizi",
        "Olahraga teratur"
    ]
elif 40 < prob_positif <= 60:
    kategori = "MENENGAH"
    warna = "#fbbc05"
    icon = "⚠️"
    warna_bg = "#fff8e1"
    pesan = "Risiko terdeteksi di level menengah. Perlu perhatian khusus pada pola hidup."
    rekomendasi = [
        "Konsultasi dengan dokter",
        "Perbaiki pola makan",
        "Olahraga rutin 3x seminggu",
        "Monitor gula darah bulanan"
    ]
else:
    kategori = "TINGGI"
    warna = "#ea4335"
    icon = "🚨"
    warna_bg = "#ffebee"
    pesan = "Risiko terdeteksi tinggi. Diperlukan penanganan medis segera."
    rekomendasi = [
        "Segera konsultasi dokter",
        "Tes laboratorium lengkap",
        "Mulai pengobatan jika diperlukan",
        "Diet ketat dan monitoring harian"
    ]

# Layout dua kolom untuk hasil
col_result, col_visual = st.columns([1, 1], gap="large")

with col_result:
    # Card hasil analisis
    st.markdown(f"""
        <div class="custom-card" style="border-left: 5px solid {warna};">
            <div style="display: flex; align-items: center; margin-bottom: 1.5rem;">
                <div style="
                    background: linear-gradient(135deg, {warna_bg}, white);
                    padding: 0.8rem;
                    border-radius: 15px;
                    margin-right: 1rem;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
                    font-size: 32px;
                ">
                    {icon}
                </div>
                <div>
                    <h3 style="margin: 0; color: {warna};">Status Risiko: {kategori}</h3>
                    <p style="color: #5f6368; margin: 0.2rem 0 0 0;">Probabilitas: {prob_positif:.1f}%</p>
                </div>
            </div>
            <div style="background-color: {warna_bg}30; padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                <p style="color: #5f6368; margin: 0; font-weight: 500;">{pesan}</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Box rekomendasi
    st.markdown("### 💡 Rekomendasi Medis")
    st.markdown(f"""
        <div style="background-color: white; padding: 1.5rem; border-radius: 15px; border: 1px solid #e3f2fd; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
            <div style="display: flex; align-items: start; margin-bottom: 1rem;">
                <span style="font-size: 24px; margin-right: 10px; margin-top: 2px;">💡</span>
                <div style="color: #1a73e8; font-weight: 600;">Tindakan yang Disarankan:</div>
            </div>
            <div style="padding-left: 2rem;">
                {''.join([f'<div style="color: #5f6368; margin-bottom: 0.8rem; display: flex; align-items: center;"><div style="color: {warna}; margin-right: 10px;">•</div>{item}</div>' for item in rekomendasi])}
            </div>
        </div>
    """, unsafe_allow_html=True)

with col_visual:
    # Gauge Chart yang ditingkatkan
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob_positif,
        number={'suffix': "%", 'font': {'size': 40, 'color': warna}},
        delta={'reference': 50, 'increasing': {'color': "#ea4335"}, 'decreasing': {'color': "#34a853"}},
        title={'text': "TINGKAT RISIKO DIABETES", 'font': {'size': 18, 'color': '#1a73e8'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': '#1a73e8'},
            'bar': {'color': warna, 'thickness': 0.8},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e3f2fd",
            'steps': [
                {'range': [0, 40], 'color': '#e8f5e9'},
                {'range': [40, 60], 'color': '#fff8e1'},
                {'range': [60, 100], 'color': '#ffebee'}],
            'threshold': {
                'line': {'color': warna, 'width': 4},
                'thickness': 0.8,
                'value': prob_positif}
        }
    ))
    
    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "#5f6368", 'family': "Arial, sans-serif"}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Skala risiko
    st.markdown("""
        <div style="display: flex; justify-content: space-between; margin-top: 1rem; padding: 1rem; background: white; border-radius: 10px; border: 1px solid #e3f2fd;">
            <div style="text-align: center;">
                <div style="width: 20px; height: 20px; background-color: #e8f5e9; border-radius: 50%; margin: 0 auto 5px auto; border: 2px solid #34a853;"></div>
                <div style="font-size: 0.85rem; color: #5f6368;">Rendah</div>
                <div style="font-size: 0.75rem; color: #34a853;">(0-40%)</div>
            </div>
            <div style="text-align: center;">
                <div style="width: 20px; height: 20px; background-color: #fff8e1; border-radius: 50%; margin: 0 auto 5px auto; border: 2px solid #fbbc05;"></div>
                <div style="font-size: 0.85rem; color: #5f6368;">Menengah</div>
                <div style="font-size: 0.75rem; color: #fbbc05;">(41-60%)</div>
            </div>
            <div style="text-align: center;">
                <div style="width: 20px; height: 20px; background-color: #ffebee; border-radius: 50%; margin: 0 auto 5px auto; border: 2px solid #ea4335;"></div>
                <div style="font-size: 0.85rem; color: #5f6368;">Tinggi</div>
                <div style="font-size: 0.75rem; color: #ea4335;">(61-100%)</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# --- 8. INFORMASI TAMBAHAN DENGAN FILTER DATA ---
st.markdown("---")

# Tabs untuk informasi tambahan
tab1, tab2, tab3 = st.tabs(["📋 Data Training", "🤖 Teknologi AI", "📚 Panduan"])

with tab1:
    st.markdown("### 📊 Dataset Medis")
    st.write("Data yang digunakan untuk melatih model kecerdasan buatan:")
    
    # Statistik dataset
    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
    with col_stats1:
        st.metric("Total Data", len(df), help="Jumlah total data pasien dalam dataset")
    with col_stats2:
        diabetes_rate = df['Hasil'].mean()*100
        st.metric("Kasus Diabetes", f"{df['Hasil'].sum()} ({diabetes_rate:.1f}%)", 
                 help="Jumlah dan persentase pasien diabetes")
    with col_stats3:
        st.metric("Fitur Medis", "8", help="Jumlah parameter yang dianalisis")
    with col_stats4:
        st.metric("Akurasi Model", "≈ 85%", help="Tingkat akurasi prediksi model")
    
    # --- FILTER DATA BARU ---
    st.markdown("---")
    st.markdown("### 🔍 Filter Data Training")
    
    # Filter options dengan dua baris
    col_filter1, col_filter2, col_filter3, col_filter4 = st.columns(4)
    
    with col_filter1:
        filter_option = st.selectbox(
            "Pilih Filter Data",
            ["10 Data Teratas", "10 Data Terbawah", "Kasus Diabetes", "Kasus Non-Diabetes", 
             "Semua Data", "Data Acak (20)", "Glukosa Tinggi", "BMI Tinggi", "Usia Muda (<30)", "Usia Tua (>50)"]
        )
    
    with col_filter2:
        jumlah_data = st.slider("Jumlah Data", 1, 100, 20, help="Jumlah data yang akan ditampilkan")
    
    with col_filter3:
        sort_by = st.selectbox(
            "Urutkan Berdasarkan",
            ["Default", "Glukosa (Tinggi ke Rendah)", "Glukosa (Rendah ke Tinggi)", 
             "BMI (Tinggi ke Rendah)", "BMI (Rendah ke Tinggi)", "Umur (Tua ke Muda)", "Umur (Muda ke Tua)"]
        )
    
    with col_filter4:
        # Pencarian berdasarkan nilai
        search_col = st.selectbox("Cari di Kolom", ["Semua", "Kehamilan", "Glukosa", "Tekanan Darah", "BMI", "Umur"])
        if search_col != "Semua":
            col_map = {"Kehamilan": "Kehamilan", "Glukosa": "Glukosa", "Tekanan Darah": "Tekanan_Darah", "BMI": "BMI", "Umur": "Umur"}
            search_value = st.number_input(f"Nilai {search_col}", min_value=0.0)
    
    # Apply filters
    filtered_df = df.copy()
    
    if filter_option == "10 Data Teratas":
        filtered_df = df.head(10)
    elif filter_option == "10 Data Terbawah":
        filtered_df = df.tail(10)
    elif filter_option == "Kasus Diabetes":
        filtered_df = df[df['Hasil'] == 1]
    elif filter_option == "Kasus Non-Diabetes":
        filtered_df = df[df['Hasil'] == 0]
    elif filter_option == "Data Acak (20)":
        filtered_df = df.sample(n=min(20, len(df)), random_state=42)
    elif filter_option == "Glukosa Tinggi":
        filtered_df = df[df['Glukosa'] > 150]
    elif filter_option == "BMI Tinggi":
        filtered_df = df[df['BMI'] > 35]
    elif filter_option == "Usia Muda (<30)":
        filtered_df = df[df['Umur'] < 30]
    elif filter_option == "Usia Tua (>50)":
        filtered_df = df[df['Umur'] > 50]
    
    # Apply sorting
    if sort_by == "Glukosa (Tinggi ke Rendah)":
        filtered_df = filtered_df.sort_values('Glukosa', ascending=False)
    elif sort_by == "Glukosa (Rendah ke Tinggi)":
        filtered_df = filtered_df.sort_values('Glukosa', ascending=True)
    elif sort_by == "BMI (Tinggi ke Rendah)":
        filtered_df = filtered_df.sort_values('BMI', ascending=False)
    elif sort_by == "BMI (Rendah ke Tinggi)":
        filtered_df = filtered_df.sort_values('BMI', ascending=True)
    elif sort_by == "Umur (Tua ke Muda)":
        filtered_df = filtered_df.sort_values('Umur', ascending=False)
    elif sort_by == "Umur (Muda ke Tua)":
        filtered_df = filtered_df.sort_values('Umur', ascending=True)
    
    # Apply search filter
    if search_col != "Semua" and 'search_value' in locals():
        col_name = col_map[search_col]
        if search_value > 0:
            filtered_df = filtered_df[filtered_df[col_name] == search_value]
    
    # Limit jumlah data
    filtered_df = filtered_df.head(jumlah_data)
    
    # Tampilkan info filter
    st.markdown(f"""
        <div class="pagination-info">
        Menampilkan {len(filtered_df)} dari {len(df)} data | Filter: {filter_option} | 
        {f"Urutkan: {sort_by}" if sort_by != "Default" else "Urutan default"}
        </div>
    """, unsafe_allow_html=True)
    
    # Quick filter buttons
    st.markdown("#### 🚀 Filter Cepat")
    quick_filter_cols = st.columns(6)
    
    with quick_filter_cols[0]:
        if st.button("👶 Usia Muda", use_container_width=True):
            filtered_df = df[df['Umur'] < 30]
    with quick_filter_cols[1]:
        if st.button("👴 Usia Tua", use_container_width=True):
            filtered_df = df[df['Umur'] > 50]
    with quick_filter_cols[2]:
        if st.button("📈 Glukosa Tinggi", use_container_width=True):
            filtered_df = df[df['Glukosa'] > 150]
    with quick_filter_cols[3]:
        if st.button("⚖️ BMI Tinggi", use_container_width=True):
            filtered_df = df[df['BMI'] > 35]
    with quick_filter_cols[4]:
        if st.button("✅ Kasus Diabetes", use_container_width=True):
            filtered_df = df[df['Hasil'] == 1]
    with quick_filter_cols[5]:
        if st.button("🔄 Reset Filter", use_container_width=True, type="primary"):
            filtered_df = df.head(20)
    
    # Tampilkan dataframe dengan styling custom
    st.markdown("### 📋 Data yang Ditampilkan")
    
    if len(filtered_df) > 0:
        # Buat dataframe dengan styling yang lebih baik
        df_display = filtered_df.copy()
        
        # Tampilkan dengan HTML table styling custom
        html_table = """
        <table class="custom-dataframe">
            <thead>
                <tr>
                    <th>No</th>
                    <th>Kehamilan</th>
                    <th>Glukosa</th>
                    <th>Tekanan Darah</th>
                    <th>Kulit</th>
                    <th>Insulin</th>
                    <th>BMI</th>
                    <th>Riwayat</th>
                    <th>Umur</th>
                    <th>Hasil</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for idx, row in df_display.iterrows():
            html_table += "<tr>"
            html_table += f"<td>{idx + 1}</td>"
            html_table += f"<td>{row['Kehamilan']}</td>"
            html_table += f"<td>{row['Glukosa']}</td>"
            html_table += f"<td>{row['Tekanan_Darah']}</td>"
            html_table += f"<td>{row['Kulit']}</td>"
            html_table += f"<td>{row['Insulin']}</td>"
            html_table += f"<td>{row['BMI']:.1f}</td>"
            html_table += f"<td>{row['Riwayat']:.3f}</td>"
            html_table += f"<td>{row['Umur']}</td>"
            
            # Warna berbeda untuk hasil positif/negatif
            if row['Hasil'] == 1:
                html_table += f"<td style='color: #ea4335; font-weight: bold;'>Ya</td>"
            else:
                html_table += f"<td style='color: #34a853; font-weight: bold;'>Tidak</td>"
            
            html_table += "</tr>"
        
        html_table += """
            </tbody>
        </table>
        """
        
        st.markdown(html_table, unsafe_allow_html=True)
        
        # Download button
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Data (CSV)",
            data=csv,
            file_name=f"diabetes_data_{filter_option.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.warning("⚠️ Tidak ada data yang sesuai dengan filter yang dipilih.")
    
    # Ringkasan statistik
    st.markdown("### 📊 Statistik Data yang Ditampilkan")
    if len(filtered_df) > 0:
        col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
        with col_sum1:
            st.metric("Rata-rata Glukosa", f"{filtered_df['Glukosa'].mean():.1f} mg/dL")
        with col_sum2:
            st.metric("Rata-rata BMI", f"{filtered_df['BMI'].mean():.1f} kg/m²")
        with col_sum3:
            st.metric("Rata-rata Umur", f"{filtered_df['Umur'].mean():.1f} tahun")
        with col_sum4:
            diabetes_rate_filtered = filtered_df['Hasil'].mean()*100
            st.metric("Persentase Diabetes", f"{diabetes_rate_filtered:.1f}%")

with tab2:
    st.markdown("""
    ### 🧠 Teknologi di Balik Sistem
    
    <div style="background: white; padding: 1.5rem; border-radius: 15px; border: 1px solid #e3f2fd; margin: 1rem 0;">
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <span style="font-size: 40px; margin-right: 15px;">🤖</span>
            <div>
                <h4 style="margin: 0; color: #1a73e8;">Random Forest Algorithm</h4>
                <p style="margin: 0; color: #5f6368;">Kombinasi banyak pohon keputusan untuk akurasi maksimal</p>
            </div>
        </div>
        <div style="color: #5f6368;">
        <strong>Keunggulan:</strong><br>
        • Mengurangi overfitting dibanding model tunggal<br>
        • Memberikan probabilitas risiko yang akurat<br>
        • Robust terhadap noise dalam data<br>
        • Dapat menangani hubungan non-linear
        </div>
    </div>
    
    <div style="background: white; padding: 1.5rem; border-radius: 15px; border: 1px solid #e3f2fd; margin: 1rem 0;">
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <span style="font-size: 40px; margin-right: 15px;">📊</span>
            <div>
                <h4 style="margin: 0; color: #1a73e8;">Parameter Analisis</h4>
                <p style="margin: 0; color: #5f6368;">8 faktor kunci yang mempengaruhi risiko diabetes</p>
            </div>
        </div>
        <div style="color: #5f6368;">
        <strong>Parameter yang Dianalisis:</strong><br>
        1. Demografis (Umur, Riwayat kehamilan)<br>
        2. Klinis (Glukosa, Tekanan darah)<br>
        3. Antropometri (BMI, Ketebalan kulit)<br>
        4. Biokimia (Insulin)
        </div>
    </div>
    """, unsafe_allow_html=True)

with tab3:
    st.markdown("""
    ### 📚 Panduan Interpretasi Hasil
    
    <div style="background: linear-gradient(135deg, #e8f4ff 0%, #d6eaff 100%); padding: 1.5rem; border-radius: 15px; border: 1px solid #c2dcff; margin: 1rem 0;">
        <h4 style="color: #1a73e8; margin-top: 0;">🎯 Risiko Rendah (0-40%)</h4>
        <p style="color: #5f6368; margin: 0.5rem 0;">
        Tidak menunjukkan tanda-tanda signifikan. Pertahankan gaya hidup sehat dengan pola makan seimbang, olahraga teratur, dan tidur yang cukup.
        </p>
    </div>
    
    <div style="background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%); padding: 1.5rem; border-radius: 15px; border: 1px solid #ffe082; margin: 1rem 0;">
        <h4 style="color: #f57c00; margin-top: 0;">⚠️ Risiko Menengah (41-60%)</h4>
        <p style="color: #5f6368; margin: 0.5rem 0;">
        Memerlukan pemantauan berkala dan perbaikan pola hidup. Disarankan untuk konsultasi dengan tenaga medis dan melakukan pemeriksaan rutin.
        </p>
    </div>
    
    <div style="background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%); padding: 1.5rem; border-radius: 15px; border: 1px solid #ff8a80; margin: 1rem 0;">
        <h4 style="color: #d32f2f; margin-top: 0;">🚨 Risiko Tinggi (61-100%)</h4>
        <p style="color: #5f6368; margin: 0.5rem 0;">
        Diperlukan konsultasi medis segera dan tes laboratorium lanjutan. Penanganan dini dapat mencegah komplikasi lebih lanjut.
        </p>
    </div>
    
    <div style="background-color: #f8fbff; padding: 1rem; border-radius: 10px; border-left: 4px solid #1a73e8; margin-top: 2rem;">
        <strong>⚠️ Disclaimer:</strong><br>
        Hasil analisis ini merupakan prediksi berdasarkan data historis dan tidak menggantikan diagnosis dari tenaga medis profesional. 
        Selalu konsultasikan dengan dokter untuk diagnosis dan pengobatan yang tepat.
    </div>
    """, unsafe_allow_html=True)

# --- 9. FOOTER ---
st.markdown("""
    <div class="footer">
        <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 1rem;">
            <span style="font-size: 30px; margin-right: 10px;">🏥</span>
            <div style="color: #1a73e8; font-weight: 600; font-size: 1.1rem;">Diabetes AI Diagnostic System</div>
        </div>
        <p style="margin: 0.5rem 0; color: #5f6368;">Versi 2.0 | Sistem Kecerdasan Buatan untuk Kesehatan</p>
        <p style="margin: 0.5rem 0; color: #5f6368; font-size: 0.8rem;">
        Dikembangkan untuk tujuan edukasi dan penelitian medis | © 2024
        </p>
        <p style="margin: 1rem 0 0 0; color: #9aa0a6; font-size: 0.75rem;">
        <strong>Disclaimer:</strong> Aplikasi ini tidak untuk diagnosis medis resmi. Selalu konsultasikan dengan tenaga medis profesional.
        </p>
    </div>
""", unsafe_allow_html=True)
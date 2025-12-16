import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. KONFIGURASI HALAMAN & STATE
# ==========================================
st.set_page_config(
    page_title="Stress Level Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inisialisasi Session State
if 'form_step' not in st.session_state:
    st.session_state.form_step = 1
if 'inputs' not in st.session_state:
    st.session_state.inputs = {}
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'api_result' not in st.session_state:
    st.session_state.api_result = None

# Fungsi Navigasi Form
def next_step():
    st.session_state.form_step += 1

def prev_step():
    st.session_state.form_step -= 1

def reset_app():
    st.session_state.form_step = 1
    st.session_state.inputs = {}
    st.session_state.analysis_done = False
    st.session_state.api_result = None

# ==========================================
# 2. MODERN CSS & ICONS (NO EMOJIS)
# ==========================================
st.markdown("""
<script src="https://unpkg.com/@phosphor-icons/web"></script>
<link rel="stylesheet" type="text/css" href="https://unpkg.com/@phosphor-icons/web/src/bold.css">

<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');

    /* Global Reset */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #1e293b;
        scroll-behavior: smooth;
    }

    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding-top: 1rem;
        padding-bottom: 5rem;
    }

    /* Hero Section */
    .hero-container {
        text-align: center;
        padding: 4rem 1rem;
        background: linear-gradient(180deg, #f8fafc 0%, #0001C 100%);
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 3rem;
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        letter-spacing: -0.05em;
        color: #FFFFFF;
        margin-bottom: 1rem;
    }
    .hero-subtitle {
        font-size: 1.25rem;
        color: #64748b;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* Info Cards */
    .info-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin-bottom: 4rem;
    }
    .info-card {
        background: linear-gradient(180deg, #f8fafc 0%, #0001C 100%);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s;
    }
    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .card-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
        color: #3b82f6;
    }
    .card-title {
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
        color: #FFFFFF;
    }
    .card-desc {
        font-size: 0.95rem;
        color: #64748b;
        line-height: 1.5;
    }

    /* Form Container */
    .form-container {
        background: linear-gradient(180deg, #f8fafc 0%, #0001C 100%);
F        border-radius: 24px;
        padding: 2.5rem;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.05);
        margin-top: 2rem;
    }
    
    /* Custom Slider Label */
    .stSlider label {
        font-weight: 600;
        color: #475569;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        border: none;
        transition: all 0.2s;
    }
    .primary-btn button {
        background-color: #0f172a;
        color: white;
    }
    .secondary-btn button {
        background-color: #f1f5f9;
        color: #475569;
    }

    /* Result Section */
    .result-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .level-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 99px;
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. HERO SECTION
# ==========================================
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Stress Level Analytics</div>
    <div class="hero-subtitle">
        Sistem cerdas berbasis Machine Learning untuk menganalisis pola stres Anda berdasarkan faktor fisik, mental, dan lingkungan.
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 4. EDUKASI (PENGERTIAN)
# ==========================================
st.markdown("### Memahami Kondisi Mental")
st.markdown("""
<div class="info-grid">
    <div class="info-card">
        <div class="card-icon"><i class="ph-bold ph-smiley"></i></div>
        <div class="card-title">No Stress / Calm</div>
        <div class="card-desc">Kondisi keseimbangan ideal di mana tubuh dan pikiran rileks, memungkinkan pemulihan energi yang optimal.</div>
    </div>
    <div class="info-card">
        <div class="card-icon"><i class="ph-bold ph-lightning"></i></div>
        <div class="card-title">Eustress</div>
        <div class="card-desc">Stres positif yang memotivasi. Respon tubuh terhadap tantangan yang dapat diatasi, meningkatkan fokus dan kinerja.</div>
    </div>
    <div class="info-card">
        <div class="card-icon"><i class="ph-bold ph-warning-circle"></i></div>
        <div class="card-title">Distress</div>
        <div class="card-desc">Stres negatif yang berlebihan. Tekanan melebihi kemampuan koping, berpotensi mengganggu kesehatan fisik dan mental.</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 5. ANALISIS WIZARD (SATU HALAMAN)
# ==========================================
st.markdown("---")
st.markdown("<h3 style='text-align:center; margin-top: 2rem;'>Mulai Analisis</h3>", unsafe_allow_html=True)

# Container Form
with st.container():
    # Progress Indicator
    if not st.session_state.analysis_done:
        progress = (st.session_state.form_step / 3) * 100
        st.progress(int(progress))
        st.caption(f"Step {st.session_state.form_step} of 3")

    st.markdown('<div class="form-container">', unsafe_allow_html=True)

    # --- TAMPILAN JIKA BELUM ANALISIS ---
    if not st.session_state.analysis_done:
        
        # STEP 1: Mental & Pribadi
        if st.session_state.form_step == 1:
            st.markdown("#### Faktor Mental & Emosional")
            c1, c2 = st.columns(2)
            
            with c1:
                st.session_state.inputs['anxiety_level'] = st.slider("Tingkat Kecemasan (0-21)", 0, 21, key='anx')
                st.session_state.inputs['depression'] = st.slider("Tingkat Depresi (0-27)", 0, 27, key='dep')
                st.session_state.inputs['self_esteem'] = st.slider("Harga Diri (0-30)", 0, 30, key='self')
            
            with c2:
                st.markdown("**Riwayat Kesehatan Mental**")
                mh_choice = st.radio("History", ["Tidak", "Ya"], horizontal=True, label_visibility="collapsed")
                st.session_state.inputs['mental_health_history'] = 1 if mh_choice == "Ya" else 0
                
                st.write("") # Spacer
                st.session_state.inputs['headache'] = st.slider("Frekuensi Sakit Kepala (0-5)", 0, 5, key='head')

            # Navigasi Step 1
            st.write("---")
            col_r = st.columns([6, 1])[1]
            with col_r:
                if st.button("Next ➔", type="primary"):
                    next_step()
                    st.rerun()

        # STEP 2: Lingkungan & Akademik
        elif st.session_state.form_step == 2:
            st.markdown("#### Faktor Akademik & Lingkungan")
            c1, c2 = st.columns(2)
            
            with c1:
                st.session_state.inputs['academic_performance'] = st.slider("Performa Akademik (0-5)", 0, 5, key='acad')
                st.session_state.inputs['study_load'] = st.slider("Beban Studi (0-5)", 0, 5, key='load')
                st.session_state.inputs['teacher_student_relationship'] = st.slider("Hubungan dgn Pengajar (0-5)", 0, 5, key='rel')
                st.session_state.inputs['future_career_concerns'] = st.slider("Kekhawatiran Karir (0-5)", 0, 5, key='car')
            
            with c2:
                st.session_state.inputs['peer_pressure'] = st.slider("Tekanan Teman Sebaya (0-5)", 0, 5, key='peer')
                st.session_state.inputs['social_support'] = st.slider("Dukungan Sosial (0-3)", 0, 3, key='soc')
                st.session_state.inputs['bullying'] = st.slider("Tingkat Perundungan (0-5)", 0, 5, key='bull')
                st.session_state.inputs['living_conditions'] = st.slider("Kondisi Tempat Tinggal (0-5)", 0, 5, key='live')

            # Navigasi Step 2
            st.write("---")
            cb, cn = st.columns([1, 6])
            with cb:
                if st.button("Back"):
                    prev_step()
                    st.rerun()
            with cn:
                # Kolom kosong untuk alignment tombol Next ke kanan
                sub_c = st.columns([5, 1])[1]
                with sub_c:
                    if st.button("Next ➔", type="primary"):
                        next_step()
                        st.rerun()

        # STEP 3: Fisik & Submit
        elif st.session_state.form_step == 3:
            st.markdown("#### Faktor Kesehatan Fisik")
            c1, c2 = st.columns(2)
            
            with c1:
                st.session_state.inputs['sleep_quality'] = st.slider("Kualitas Tidur (0-5)", 0, 5, key='sleep')
                st.session_state.inputs['blood_pressure'] = st.slider("Tekanan Darah (1-3)", 1, 3, key='bp', help="1=Normal, 2=Sedang, 3=Tinggi")
            
            with c2:
                st.session_state.inputs['breathing_problem'] = st.slider("Masalah Pernapasan (0-5)", 0, 5, key='breath')
                st.session_state.inputs['noise_level'] = st.slider("Kebisingan Lingkungan (0-5)", 0, 5, key='noise')
            
            # Input tambahan (Hidden/Defaults or Extra) untuk melengkapi API
            st.session_state.inputs['basic_needs'] = st.slider("Pemenuhan Kebutuhan Dasar (0-5)", 0, 5, key='basic')
            st.session_state.inputs['extracurricular_activities'] = st.slider("Aktivitas Ekstrakurikuler (0-5)", 0, 5, key='extra')

            # Navigasi Step 3 (Submit)
            st.write("---")
            cb, cn = st.columns([1, 6])
            with cb:
                if st.button("Back"):
                    prev_step()
                    st.rerun()
            with cn:
                if st.button("Analisis Sekarang", type="primary", use_container_width=True):
                    with st.spinner("Memproses data..."):
                        try:
                            # Kirim ke API
                            response = requests.post(
                                "http://localhost:5000/predict",
                                json=st.session_state.inputs,
                                headers={"Content-Type": "application/json"}
                            )
                            if response.status_code == 200:
                                st.session_state.api_result = response.json()
                                st.session_state.analysis_done = True
                                st.rerun()
                            else:
                                st.error("Gagal terhubung ke API.")
                        except Exception as e:
                            st.error(f"Koneksi Error: {e}")

    # --- TAMPILAN HASIL (RESULT) ---
    else:
        result = st.session_state.api_result
        lvl = result['stress_level']
        desc = result['description']
        probs = result['probabilities']

        # Styling Warna Hasil
        colors = ["#10b981", "#3b82f6", "#f59e0b"] # Green, Blue, Orange
        bg_colors = ["#d1fae5", "#dbeafe", "#fef3c7"]
        color = colors[lvl] if lvl < 3 else "#ef4444"
        bg_color = bg_colors[lvl] if lvl < 3 else "#fee2e2"

        st.markdown(f"""
        <div style="text-align:center; padding: 2rem;">
            <div style="background-color: {bg_color}; color: {color}; display: inline-block; padding: 0.5rem 1rem; border-radius: 50px; font-weight: bold; margin-bottom: 1rem;">
                STRESS LEVEL {lvl}
            </div>
            <h2 style="color: #1e293b; font-size: 2.5rem; margin: 0;">{desc}</h2>
        </div>
        """, unsafe_allow_html=True)

        col_res1, col_res2 = st.columns([2, 1])

        with col_res1:
            st.markdown("##### Probabilitas Kategori")
            # Dinamis Chart Logic
            num_classes = len(probs)
            labels = ["No Stress", "Eustress", "Distress"][:num_classes]
            values = [probs[f"stress_level_{i}"] for i in range(num_classes)]
            
            df_chart = pd.DataFrame({"Label": labels, "Value": values})
            
            fig, ax = plt.subplots(figsize=(6, 3))
            fig.patch.set_alpha(0)
            ax.set_facecolor('#ffffff00')
            bars = ax.barh(df_chart["Label"], df_chart["Value"], color=[colors[i] for i in range(num_classes)])
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.xaxis.set_visible(False)
            
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, f'{width:.1%}', va='center')
            
            st.pyplot(fig)

        with col_res2:
            st.markdown("##### Rekomendasi")
            rec_text = {
                0: ["Pertahankan pola tidur", "Lanjutkan aktivitas sosial", "Bantu orang sekitar"],
                1: ["Ambil jeda istirahat", "Kurangi sedikit beban kerja", "Olahraga ringan"],
                2: ["Konsultasi profesional", "Praktik mindfulness", "Kurangi drastis pemicu stres"]
            }
            
            for rec in rec_text.get(lvl, []):
                st.markdown(f"""
                <div style="display:flex; align-items:center; margin-bottom:0.5rem; color:#475569;">
                    <i class="ph-bold ph-check-circle" style="color:{color}; margin-right:0.5rem;"></i> {rec}
                </div>
                """, unsafe_allow_html=True)

        st.write("---")
        if st.button("Reset Analisis"):
            reset_app()
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True) # End Form Container
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Student Outcome Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a3a6c 0%, #2563eb 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-header h1 { font-size: 2.2rem; margin: 0; }
    .main-header p  { font-size: 1rem; opacity: 0.85; margin-top: 0.5rem; }

    .result-graduate {
        background: linear-gradient(135deg, #064e3b 0%, #059669 100%);
        padding: 2rem; border-radius: 12px; color: white; text-align: center;
    }
    .result-dropout {
        background: linear-gradient(135deg, #7f1d1d 0%, #dc2626 100%);
        padding: 2rem; border-radius: 12px; color: white; text-align: center;
    }
    .result-title { font-size: 2rem; font-weight: 700; margin: 0; }
    .result-subtitle { font-size: 1.1rem; opacity: 0.9; margin-top: 0.4rem; }

    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-card .label { font-size: 0.8rem; color: #64748b; font-weight: 600; }
    .metric-card .value { font-size: 1.6rem; font-weight: 700; color: #1e293b; }

    .section-header {
        font-size: 1.1rem; font-weight: 700;
        color: #1e3a8a; border-bottom: 2px solid #3b82f6;
        padding-bottom: 0.3rem; margin-bottom: 1rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%);
        color: white; font-weight: 700; font-size: 1.1rem;
        border: none; border-radius: 8px;
        padding: 0.7rem 2rem; width: 100%;
    }
    .stButton>button:hover { background: #1e40af; }
    .info-box {
        background: #eff6ff; border-left: 4px solid #3b82f6;
        padding: 0.8rem 1rem; border-radius: 0 8px 8px 0;
        font-size: 0.9rem; color: #1e40af;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Load Models (cached)
# ──────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = os.path.dirname(__file__)
    models = {
        "🤖 Random Forest (Advanced)": joblib.load(os.path.join(base, "random_forest_pipeline (1).joblib")),
        "📊 Logistic Regression":       joblib.load(os.path.join(base, "logistic_regression_tuned.joblib")),
        "🌳 Decision Tree":             joblib.load(os.path.join(base, "decision_tree_pipeline.joblib")),
    }
    meta = joblib.load(os.path.join(base, "model_metadata.joblib"))
    return models, meta

models, meta = load_models()
THRESHOLD = meta.get("threshold", 0.3)

# ──────────────────────────────────────────────
# Label Mappings  (decoded from OHE categories)
# ──────────────────────────────────────────────
MARITAL_STATUS = {
    1: "Single", 2: "Married", 3: "Widower",
    4: "Divorced", 5: "Facto Union", 6: "Legally Separated"
}
APPLICATION_MODE = {
    1: "1st phase – general contingent",
    2: "Ordinance No. 612/93",
    5: "1st phase – special contingent (Azores Island)",
    7: "Holders of other higher courses",
    10: "Ordinance No. 854-B/99",
    15: "International student (bachelor)",
    16: "1st phase – special contingent (Madeira Island)",
    17: "2nd phase – general contingent",
    18: "3rd phase – general contingent",
    26: "Ordinance No. 533-A/99 b2 (different plan)",
    27: "Ordinance No. 533-A/99 b3 (other institution)",
    39: "Over 23 years old",
    42: "Transfer",
    43: "Change of course",
    44: "Technological specialization diploma holders",
    51: "Change of institution/course",
    53: "Short cycle diploma holders",
    57: "Change of institution/course (international)",
}
COURSE = {
    33: "Biofuel Production Technologies", 171: "Animation and Multimedia Design",
    8014: "Social Service (evening)", 9003: "Agronomy",
    9070: "Communication Design", 9085: "Veterinary Nursing",
    9119: "Informatics Engineering", 9130: "Equinculture",
    9147: "Management", 9238: "Social Service",
    9254: "Tourism", 9500: "Nursing",
    9556: "Oral Hygiene", 9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication", 9853: "Basic Education",
    9991: "Management (evening)",
}
PREV_QUAL = {
    1: "Secondary education", 2: "Higher education – bachelor's degree",
    3: "Higher education – degree", 4: "Higher education – master's",
    5: "Higher education – doctorate", 6: "Frequency of higher education",
    9: "12th year – not completed", 10: "11th year – not completed",
    12: "Other – 11th year", 15: "10th year",
    19: "10th year – not completed", 38: "Basic education 3rd cycle",
    39: "Basic education 2nd cycle", 40: "Technological specialization",
    42: "Higher education – degree (1st cycle)", 43: "Professional higher technical course",
}
NATIONALITY = {
    1: "Portuguese", 2: "German", 6: "Spanish", 11: "Italian",
    13: "Dutch", 14: "English", 17: "Lithuanian", 21: "Angolan",
    22: "Cape Verdean", 24: "Guinean", 25: "Mozambican", 26: "Santomean",
    41: "Turkish", 62: "Brazilian", 100: "Romanian", 101: "Moldova (Republic of)",
    103: "Mexican", 105: "Ukrainian", 109: "Russian",
}
PARENT_QUAL = {
    1: "Secondary Education – 12th Year", 2: "Higher Education – Bachelor's",
    3: "Higher Education – Degree", 4: "Higher Education – Master's",
    5: "Higher Education – Doctorate", 6: "Frequency of Higher Education",
    9: "12th Year – Not Completed", 10: "11th Year – Not Completed",
    11: "7th Year (Old)", 12: "Other – 11th Year", 14: "10th Year",
    19: "10th Year – Not Completed", 22: "8th Year Schooling",
    26: "Unknown", 29: "Can Read without 4th Year",
    30: "Basic Education 1st Cycle (4th/5th Year)",
    34: "Basic Education 2nd Cycle (6th/7th/8th Year)",
    35: "Technological Specialization Course",
    36: "Higher Education – Degree (1st Cycle)",
    37: "Specialized Higher Studies",
    38: "Professional Higher Technical Course",
    39: "Higher Education – Master (2nd Cycle)",
    40: "Higher Education – Doctorate (3rd Cycle)",
    41: "Frequency of Higher Education",
    42: "12th Year – Not Completed",
    43: "Basic Education 3rd Cycle", 44: "Unknown",
}
OCCUPATION = {
    0: "Student", 1: "Legislative/Executive Representatives",
    2: "Intellectual/Scientific Activities", 3: "Intermediate Technicians",
    4: "Administrative Staff", 5: "Personal Services/Security/Safety",
    6: "Agricultural/Fishery Workers", 7: "Craft/Trade Workers",
    8: "Plant/Machine Operators", 9: "Unskilled Workers",
    10: "Armed Forces", 90: "Other Situation", 99: "(blank)",
    122: "Health professionals", 123: "Teachers", 125: "ICT Specialists",
    131: "Science/Engineering Technicians", 132: "Health Technicians",
    134: "Legal/Social/Cultural Technicians", 141: "Office Workers",
    143: "Data/Accounting/Statistical Operators", 144: "Other Admin Support",
    151: "Personal Service Workers", 152: "Sellers",
    153: "Personal Care Workers", 171: "Skilled Construction Workers",
    175: "Skilled Food Industry Workers", 191: "Cleaning Workers",
    192: "Unskilled Agriculture Workers", 193: "Unskilled Industry Workers",
    194: "Meal Preparation Assistants",
}
FATHER_OCCUPATION_EXTRA = {
    101: "Armed Forces Officers", 102: "Armed Forces Sergeants",
    103: "Other Armed Forces", 112: "Directors of Administrative Services",
    121: "Commercial/Service Directors", 124: "Hotel/Catering/Trade Directors",
    135: "IT Technicians", 154: "Other Health Technicians",
    161: "Market-Oriented Farmers", 163: "Subsistence Farmers",
    172: "Skilled Metallurgy Workers", 174: "Skilled Electricity Workers",
    181: "Process Control Operators", 182: "Fixed Plant Operators",
    183: "Other Plant Operators", 195: "Street Vendors",
}

ALL_FATHER_OCC = {**OCCUPATION, **FATHER_OCCUPATION_EXTRA}

# ──────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────
def selectbox_mapped(label, mapping, help_text=None):
    options = list(mapping.keys())
    labels  = [f"{k} – {v}" for k, v in mapping.items()]
    choice  = st.selectbox(label, options, format_func=lambda x: f"{x} – {mapping[x]}", help=help_text)
    return choice

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🎓 Student Outcome Predictor</h1>
  <p>Prediksi kelulusan atau dropout mahasiswa menggunakan Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# SIDEBAR – Model selection + Info
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Pengaturan Model")
    selected_model_name = st.selectbox("Pilih Model Prediksi", list(models.keys()))
    selected_model = models[selected_model_name]

    st.markdown("---")
    st.markdown("### 📌 Informasi Model")
    st.markdown(f"""
    - **Model aktif:** {selected_model_name}
    - **Threshold:** `{THRESHOLD}`
    - **Output:** Lulus (1) / Dropout (0)
    """)
    st.markdown("---")
    st.markdown("""
    <div class="info-box">
    💡 Model menggunakan data akademik, ekonomi, dan demografis mahasiswa untuk memprediksi kemungkinan kelulusan.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Model tersedia: Random Forest · Logistic Regression · Decision Tree")

# ──────────────────────────────────────────────
# FORM INPUT
# ──────────────────────────────────────────────
st.markdown("## 📋 Data Mahasiswa")
st.markdown("Isi informasi di bawah ini untuk mendapatkan prediksi.")

tab1, tab2, tab3, tab4 = st.tabs([
    "👤 Data Pribadi & Pendaftaran",
    "💰 Kondisi Ekonomi & Sosial",
    "📚 Akademik Semester 1",
    "📚 Akademik Semester 2"
])

with tab1:
    st.markdown('<div class="section-header">Data Pribadi</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Jenis Kelamin", [1, 0], format_func=lambda x: "Laki-laki" if x == 1 else "Perempuan")
        age_at_enrollment = st.number_input("Usia saat Mendaftar", min_value=17, max_value=70, value=20)
        marital_status = selectbox_mapped("Status Pernikahan", MARITAL_STATUS)
    with c2:
        nationality = selectbox_mapped("Kewarganegaraan", NATIONALITY)
        international = st.selectbox("Mahasiswa Internasional?", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
        displaced = st.selectbox("Displaced (Pindahan)?", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    with c3:
        special_needs = st.selectbox("Kebutuhan Pendidikan Khusus?", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
        debtor = st.selectbox("Memiliki Tunggakan?", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
        tuition_fees = st.selectbox("Uang Kuliah Lancar?", [1, 0], format_func=lambda x: "Ya" if x == 1 else "Tidak")

    st.markdown('<div class="section-header">Data Pendaftaran</div>', unsafe_allow_html=True)
    c4, c5, c6 = st.columns(3)
    with c4:
        application_mode = selectbox_mapped("Mode Pendaftaran", APPLICATION_MODE)
        application_order = st.number_input("Urutan Pilihan Program", min_value=0, max_value=9, value=1,
                                            help="0 = pilihan pertama, 9 = pilihan terakhir")
    with c5:
        course = selectbox_mapped("Program Studi", COURSE)
        attendance = st.selectbox("Waktu Kuliah", [1, 0], format_func=lambda x: "Pagi (Daytime)" if x == 1 else "Malam (Evening)")
    with c6:
        prev_qual = selectbox_mapped("Kualifikasi Sebelumnya", PREV_QUAL)
        prev_qual_grade = st.number_input("Nilai Kualifikasi Sebelumnya", min_value=0.0, max_value=200.0, value=130.0, step=0.1)
    
    admission_grade = st.number_input("Nilai Penerimaan (Admission Grade)", min_value=0.0, max_value=200.0, value=130.0, step=0.1)
    scholarship = st.selectbox("Penerima Beasiswa?", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")

with tab2:
    st.markdown('<div class="section-header">Latar Belakang Keluarga</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Ibu**")
        mothers_qual = selectbox_mapped("Pendidikan Ibu", PARENT_QUAL)
        mothers_occ   = selectbox_mapped("Pekerjaan Ibu", OCCUPATION)
    with c2:
        st.markdown("**Ayah**")
        fathers_qual = selectbox_mapped("Pendidikan Ayah", PARENT_QUAL)
        fathers_occ   = selectbox_mapped("Pekerjaan Ayah", ALL_FATHER_OCC)

    st.markdown('<div class="section-header">Kondisi Makroekonomi</div>', unsafe_allow_html=True)
    c3, c4, c5 = st.columns(3)
    with c3:
        unemployment_rate = st.number_input("Tingkat Pengangguran (%)", min_value=0.0, max_value=30.0, value=10.8, step=0.1,
                                            help="Tingkat pengangguran di negara asal saat mendaftar")
    with c4:
        inflation_rate = st.number_input("Tingkat Inflasi (%)", min_value=-5.0, max_value=20.0, value=1.4, step=0.1)
    with c5:
        gdp = st.number_input("GDP (pertumbuhan, %)", min_value=-10.0, max_value=10.0, value=1.74, step=0.01)

with tab3:
    st.markdown('<div class="section-header">Kinerja Akademik – Semester 1</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        cu1_credited  = st.number_input("SKS yang Dikreditkan (Sem 1)", min_value=0, max_value=20, value=0)
        cu1_enrolled  = st.number_input("SKS yang Diambil (Sem 1)", min_value=0, max_value=30, value=6)
        cu1_evals     = st.number_input("Jumlah Evaluasi (Sem 1)", min_value=0, max_value=45, value=6)
    with c2:
        cu1_approved  = st.number_input("SKS yang Lulus (Sem 1)", min_value=0, max_value=30, value=5)
        cu1_grade     = st.number_input("Rata-rata Nilai (Sem 1)", min_value=0.0, max_value=20.0, value=13.0, step=0.1)
    with c3:
        cu1_no_eval   = st.number_input("SKS tanpa Evaluasi (Sem 1)", min_value=0, max_value=20, value=0)

with tab4:
    st.markdown('<div class="section-header">Kinerja Akademik – Semester 2</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        cu2_credited  = st.number_input("SKS yang Dikreditkan (Sem 2)", min_value=0, max_value=20, value=0)
        cu2_enrolled  = st.number_input("SKS yang Diambil (Sem 2)", min_value=0, max_value=30, value=6)
        cu2_evals     = st.number_input("Jumlah Evaluasi (Sem 2)", min_value=0, max_value=45, value=6)
    with c2:
        cu2_approved  = st.number_input("SKS yang Lulus (Sem 2)", min_value=0, max_value=30, value=5)
        cu2_grade     = st.number_input("Rata-rata Nilai (Sem 2)", min_value=0.0, max_value=20.0, value=13.0, step=0.1)
    with c3:
        cu2_no_eval   = st.number_input("SKS tanpa Evaluasi (Sem 2)", min_value=0, max_value=20, value=0)

# ──────────────────────────────────────────────
# PREDICT BUTTON
# ──────────────────────────────────────────────
st.markdown("---")
col_btn = st.columns([1, 2, 1])
with col_btn[1]:
    predict_btn = st.button("🔍 Prediksi Sekarang")

# ──────────────────────────────────────────────
# PREDICTION LOGIC
# ──────────────────────────────────────────────
if predict_btn:
    input_data = pd.DataFrame([{
        "Marital status":                              marital_status,
        "Application mode":                            application_mode,
        "Application order":                           application_order,
        "Course":                                      course,
        "Daytime/evening attendance":                  attendance,
        "Previous qualification":                      prev_qual,
        "Previous qualification (grade)":              prev_qual_grade,
        "Nacionality":                                 nationality,
        "Mother's qualification":                      mothers_qual,
        "Father's qualification":                      fathers_qual,
        "Mother's occupation":                         mothers_occ,
        "Father's occupation":                         fathers_occ,
        "Admission grade":                             admission_grade,
        "Displaced":                                   displaced,
        "Educational special needs":                   special_needs,
        "Debtor":                                      debtor,
        "Tuition fees up to date":                     tuition_fees,
        "Gender":                                      gender,
        "Scholarship holder":                          scholarship,
        "Age at enrollment":                           age_at_enrollment,
        "International":                               international,
        "Curricular units 1st sem (credited)":         cu1_credited,
        "Curricular units 1st sem (enrolled)":         cu1_enrolled,
        "Curricular units 1st sem (evaluations)":      cu1_evals,
        "Curricular units 1st sem (approved)":         cu1_approved,
        "Curricular units 1st sem (grade)":            cu1_grade,
        "Curricular units 1st sem (without evaluations)": cu1_no_eval,
        "Curricular units 2nd sem (credited)":         cu2_credited,
        "Curricular units 2nd sem (enrolled)":         cu2_enrolled,
        "Curricular units 2nd sem (evaluations)":      cu2_evals,
        "Curricular units 2nd sem (approved)":         cu2_approved,
        "Curricular units 2nd sem (grade)":            cu2_grade,
        "Curricular units 2nd sem (without evaluations)": cu2_no_eval,
        "Unemployment rate":                           unemployment_rate,
        "Inflation rate":                              inflation_rate,
        "GDP":                                         gdp,
    }])

    try:
        proba = selected_model.predict_proba(input_data)[0]
        prob_graduate = proba[1]
        prob_dropout  = proba[0]
        prediction    = 1 if prob_graduate >= THRESHOLD else 0
    except Exception as e:
        st.error(f"Error prediksi: {e}")
        st.stop()

    # Result display
    st.markdown("---")
    st.markdown("## 📊 Hasil Prediksi")

    if prediction == 1:
        st.markdown(f"""
        <div class="result-graduate">
          <div class="result-title">✅ DIPREDIKSI LULUS</div>
          <div class="result-subtitle">Mahasiswa ini memiliki kemungkinan tinggi untuk menyelesaikan studi.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-dropout">
          <div class="result-title">⚠️ RISIKO DROPOUT</div>
          <div class="result-subtitle">Mahasiswa ini berisiko tidak menyelesaikan studi. Perlu perhatian lebih.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Probability metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="label">PROBABILITAS LULUS</div>
          <div class="value" style="color:#059669">{prob_graduate:.1%}</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="label">PROBABILITAS DROPOUT</div>
          <div class="value" style="color:#dc2626">{prob_dropout:.1%}</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
          <div class="label">THRESHOLD PREDIKSI</div>
          <div class="value" style="color:#2563eb">{THRESHOLD:.2f}</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
          <div class="label">MODEL DIGUNAKAN</div>
          <div class="value" style="font-size:0.9rem; color:#1e293b">{selected_model_name.split(" ",1)[1]}</div>
        </div>""", unsafe_allow_html=True)

    # Probability bar chart
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📈 Distribusi Probabilitas")
    bar_df = pd.DataFrame({
        "Hasil": ["Lulus", "Dropout"],
        "Probabilitas": [prob_graduate, prob_dropout]
    })
    st.bar_chart(bar_df.set_index("Hasil"), use_container_width=True, color=["#059669"])

    # Key risk factors summary
    st.markdown("### 🔎 Ringkasan Faktor Kunci")
    cA, cB = st.columns(2)
    with cA:
        st.markdown("**Faktor Akademik**")
        total_approved = cu1_approved + cu2_approved
        total_enrolled = (cu1_enrolled + cu2_enrolled) or 1
        approval_rate  = total_approved / total_enrolled
        avg_grade      = (cu1_grade + cu2_grade) / 2

        if approval_rate >= 0.8:
            st.success(f"✅ Tingkat kelulusan SKS: {approval_rate:.0%} (Baik)")
        elif approval_rate >= 0.5:
            st.warning(f"⚠️ Tingkat kelulusan SKS: {approval_rate:.0%} (Perlu perhatian)")
        else:
            st.error(f"❌ Tingkat kelulusan SKS: {approval_rate:.0%} (Kritis)")

        if avg_grade >= 14:
            st.success(f"✅ Rata-rata nilai: {avg_grade:.1f}/20 (Bagus)")
        elif avg_grade >= 10:
            st.warning(f"⚠️ Rata-rata nilai: {avg_grade:.1f}/20 (Cukup)")
        else:
            st.error(f"❌ Rata-rata nilai: {avg_grade:.1f}/20 (Rendah)")

    with cB:
        st.markdown("**Faktor Ekonomi & Sosial**")
        if tuition_fees == 1:
            st.success("✅ Uang kuliah: Lancar")
        else:
            st.error("❌ Uang kuliah: Tidak lancar")

        if scholarship == 1:
            st.success("✅ Penerima beasiswa")
        else:
            st.info("ℹ️ Bukan penerima beasiswa")

        if debtor == 1:
            st.error("❌ Memiliki tunggakan pembayaran")
        else:
            st.success("✅ Tidak ada tunggakan")

    # All models comparison
    st.markdown("---")
    st.markdown("### 🆚 Perbandingan Semua Model")
    comp_data = []
    for mname, mobj in models.items():
        try:
            p = mobj.predict_proba(input_data)[0]
            pred_label = "✅ Lulus" if p[1] >= THRESHOLD else "⚠️ Dropout"
            comp_data.append({
                "Model": mname,
                "Probabilitas Lulus": f"{p[1]:.1%}",
                "Probabilitas Dropout": f"{p[0]:.1%}",
                "Prediksi": pred_label
            })
        except:
            comp_data.append({"Model": mname, "Probabilitas Lulus": "Error",
                               "Probabilitas Dropout": "Error", "Prediksi": "Error"})

    st.dataframe(pd.DataFrame(comp_data).set_index("Model"), use_container_width=True)

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown("---")
st.caption("🎓 Student Outcome Predictor | Berbasis Machine Learning | Data: Academic, Socioeconomic & Demographic Factors")
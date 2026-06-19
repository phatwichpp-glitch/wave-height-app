"""
app.py — Entry point for the Wave Height Analysis application.
Streamlit multi-page app using st.navigation.
"""

import streamlit as st

st.set_page_config(
    page_title="ระบบวิเคราะห์ความสูงคลื่น IMU Buoy",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def setup_thai_font():
    import subprocess
    import os
    import matplotlib.font_manager as fm
    font_path = "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf"
    if not os.path.exists(font_path):
        subprocess.run(
            ["apt-get", "install", "-y", "-q", "fonts-noto"],
            capture_output=True
        )
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        import matplotlib.pyplot as plt
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = prop.get_name()


setup_thai_font()

# ─── Sidebar global settings ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌊 การตั้งค่าส่วนกลาง")
    st.markdown("---")

    fs = st.number_input(
        "อัตราการสุ่มตัวอย่าง fs (Hz)",
        min_value=1.0, max_value=100.0, value=5.0, step=0.5,
        help="ความถี่ในการเก็บข้อมูล"
    )
    epoch_sec = st.number_input(
        "ความยาว Epoch (วินาที)",
        min_value=10, max_value=3600, value=60, step=10,
        help="ความยาวของช่วงเวลาในการวิเคราะห์แต่ละ Epoch"
    )
    hp_cutoff = st.number_input(
        "High-pass cutoff (Hz)",
        min_value=0.01, max_value=0.5, value=0.05, step=0.01,
        format="%.3f",
        help="ความถี่ตัดผ่าน High-pass filter"
    )
    lp_cutoff = st.number_input(
        "Low-pass cutoff (Hz)",
        min_value=0.5, max_value=5.0, value=2.0, step=0.1,
        help="ความถี่ตัดผ่าน Low-pass filter"
    )

    # Nyquist warning
    nyquist = fs / 2.0
    if lp_cutoff > nyquist:
        st.warning(f"⚠️ Low-pass cutoff ({lp_cutoff} Hz) > Nyquist ({nyquist} Hz)")
    if fs < 2 * lp_cutoff:
        st.warning(f"⚠️ fs ต่ำกว่า Nyquist 2× ความถี่คลื่น")

    st.markdown("---")
    st.caption("พัฒนาสำหรับ IMU Buoy (9-axis)\nความสูงคลื่นนัยสำคัญ (Hs)")

# Store settings in session_state for all pages
st.session_state["fs"] = fs
st.session_state["epoch_sec"] = int(epoch_sec)
st.session_state["hp_cutoff"] = hp_cutoff
st.session_state["lp_cutoff"] = lp_cutoff

# ─── Navigation ───────────────────────────────────────────────────────────────
pages = [
    st.Page("pages/1_upload.py",     title="📁 อัปโหลดข้อมูล",       icon="📁"),
    st.Page("pages/2_processing.py", title="⚙️ ประมวลผลสัญญาณ",      icon="⚙️"),
    st.Page("pages/3_spectrum.py",   title="📊 การวิเคราะห์สเปกตรัม", icon="📊"),
    st.Page("pages/4_comparison.py", title="📈 เปรียบเทียบวิธีการ",   icon="📈"),
    st.Page("pages/5_report.py",     title="📤 ส่งออกรายงาน",         icon="📤"),
]

pg = st.navigation(pages)
pg.run()

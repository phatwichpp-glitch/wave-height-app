"""
app.py — Entry point for the Wave Height Analysis application.
Streamlit multi-page app using st.navigation.
"""

import streamlit as st

st.set_page_config(
    page_title="IMU Buoy Wave Height Analyzer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Sidebar global settings ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌊 Global Settings")
    st.markdown("---")

    fs = st.number_input(
        "Sampling Rate fs (Hz)",
        min_value=1.0, max_value=100.0, value=5.0, step=0.5,
        help="Data acquisition frequency"
    )
    epoch_sec = st.number_input(
        "Epoch Length (seconds)",
        min_value=10, max_value=3600, value=60, step=10,
        help="Duration of each analysis epoch"
    )
    hp_cutoff = st.number_input(
        "High-pass Cutoff (Hz)",
        min_value=0.01, max_value=0.5, value=0.05, step=0.01,
        format="%.3f",
        help="High-pass filter cutoff frequency"
    )
    lp_cutoff = st.number_input(
        "Low-pass Cutoff (Hz)",
        min_value=0.5, max_value=5.0, value=2.0, step=0.1,
        help="Low-pass filter cutoff frequency"
    )

    # Nyquist warning
    nyquist = fs / 2.0
    if lp_cutoff > nyquist:
        st.warning(f"⚠️ Low-pass cutoff ({lp_cutoff} Hz) > Nyquist ({nyquist} Hz)")
    if fs < 2 * lp_cutoff:
        st.warning(f"⚠️ fs is below 2x wave frequency Nyquist limit")

    st.markdown("---")
    st.caption("Developed for 9-axis IMU Buoy\nSignificant Wave Height (Hs) Analysis")

# Store settings in session_state for all pages
st.session_state["fs"] = fs
st.session_state["epoch_sec"] = int(epoch_sec)
st.session_state["hp_cutoff"] = hp_cutoff
st.session_state["lp_cutoff"] = lp_cutoff

# ─── Navigation ───────────────────────────────────────────────────────────────
pages = [
    st.Page("pages/1_upload.py",     title="Upload Data",        icon="📁"),
    st.Page("pages/2_processing.py", title="Signal Processing",  icon="⚙️"),
    st.Page("pages/3_spectrum.py",   title="Spectral Analysis",  icon="📊"),
    st.Page("pages/4_comparison.py", title="Method Comparison",  icon="📈"),
    st.Page("pages/5_report.py",     title="Export Report",      icon="📤"),
]

pg = st.navigation(pages)
pg.run()

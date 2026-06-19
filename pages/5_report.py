"""
pages/5_report.py
Export Report page — PDF (ReportLab) and Excel (openpyxl).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import numpy as np

from core.export import build_excel, build_pdf

st.title("📤 Export Report")

if "results" not in st.session_state:
    st.warning("⚠️ Please process data on the ⚙️ page first")
    st.stop()

results = st.session_state["results"]
n_results = len(results)
n_samples = st.session_state.get("n_samples_raw", 0)
t_start = st.session_state.get("t_start", "")
t_end = st.session_state.get("t_end", "")
date_range = f"{t_start} → {t_end}" if t_start and t_end else "Not specified"

st.markdown(f"""
**Summary:**
- Time range: **{date_range}**
- Sample count: **{n_samples:,}**
- Epoch count: **{n_results}**
""")

# ─── Epoch selector for spectrum sheet / PDF spectrum page ────────────────────
epoch_for_spectrum = st.number_input(
    "Epoch for the Spectrum page in the report",
    min_value=1, max_value=n_results, value=1, step=1
) - 1  # convert to 0-based

st.markdown("---")

# ─── Excel Export ─────────────────────────────────────────────────────────────
st.markdown("### 📊 Export Excel")
st.markdown("""
The Excel file contains 3 sheets:
- **Summary** — statistics for each Hs method
- **Time Series** — full epoch table
- **Spectrum** — f and Pxx values for the selected epoch
""")

if st.button("📥 Generate and Download Excel", type="primary"):
    with st.spinner("Generating Excel file..."):
        try:
            excel_bytes = build_excel(results, selected_epoch_idx=epoch_for_spectrum)
            st.download_button(
                label="⬇️ Download Excel Report",
                data=excel_bytes,
                file_name="wave_height_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
            st.success("✅ Excel file generated successfully!")
        except Exception as e:
            st.error(f"An error occurred while generating the Excel file: {e}")

st.markdown("---")

# ─── PDF Export ───────────────────────────────────────────────────────────────
st.markdown("### 📄 Export PDF")
st.markdown("""
The PDF file contains 5 pages:
- Page 1: Title, time range, statistics summary table
- Page 2: Hs time series chart (all 3 methods)
- Page 3: Energy spectrum chart
- Page 4: Scatter plot comparing methods
- Page 5: Full epoch table
""")

# Check if Thai font is available for PDF
THAI_FONT_PATH = "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf"
if not os.path.exists(THAI_FONT_PATH):
    st.warning("⚠️ Thai font not found — the PDF will use Helvetica instead")

if st.button("📥 Generate and Download PDF", type="primary"):
    with st.spinner("Generating PDF file (may take 10-30 seconds)..."):
        try:
            pdf_bytes = build_pdf(
                results=results,
                epoch_idx=epoch_for_spectrum,
                date_range=date_range,
                n_samples=n_samples,
            )
            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name="wave_height_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            st.success("✅ PDF file generated successfully!")
        except Exception as e:
            st.error(f"An error occurred while generating the PDF file: {e}")

st.markdown("---")

# ─── Inline Preview ───────────────────────────────────────────────────────────
st.markdown("### 👀 Preview of the Selected Epoch's Data for the Report")

r = results[epoch_for_spectrum]

def fmt_v(v, dec=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.{dec}f}"

preview_data = {
    "Parameter": [
        "Hs_rect (m)", "Hs_spec (m)", "Hs_zc (m)",
        "Tp (s)", "Tm02 (s)", "fp (Hz)", "H_max (m)", "Wave count"
    ],
    "Value": [
        fmt_v(r.get("Hs_rect")),
        fmt_v(r.get("Hs_spec")),
        fmt_v(r.get("Hs_zc")),
        fmt_v(r.get("Tp"), 2),
        fmt_v(r.get("Tm02"), 2),
        fmt_v(r.get("fp"), 4),
        fmt_v(r.get("H_max")),
        str(r.get("wave_count", 0)),
    ]
}

import pandas as pd
st.table(pd.DataFrame(preview_data))

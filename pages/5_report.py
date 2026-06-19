"""
pages/5_report.py
Export Report page — PDF (ReportLab) and Excel (openpyxl).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import numpy as np
import pandas as pd

from core.export import build_excel, build_pdf

st.title("📤 Export Report")

if "results" not in st.session_state:
    st.warning("⚠️ Please process data on the ⚙️ Processing page first.")
    st.stop()

results   = st.session_state["results"]
n_results = len(results)
n_samples = st.session_state.get("n_samples_raw", 0)
t_start   = st.session_state.get("t_start", "")
t_end     = st.session_state.get("t_end", "")
date_range = f"{t_start} → {t_end}" if t_start and t_end else "N/A"

st.markdown(f"""
**Summary:**
- Time range: **{date_range}**
- Total samples: **{n_samples:,}**
- Epochs processed: **{n_results}**
""")

epoch_for_spectrum = st.number_input(
    "Epoch for spectrum page in report",
    min_value=1, max_value=n_results, value=1, step=1
) - 1

st.markdown("---")

# ── Excel ─────────────────────────────────────────────────────────────────────
st.markdown("### 📊 Excel Export")
st.markdown("""
Excel file contains 3 sheets:
- **Summary** — statistics per Hs method
- **Timeseries** — full epoch table
- **Spectrum** — f and Pxx for selected epoch
""")

if st.button("📥 Generate & Download Excel", type="primary"):
    with st.spinner("Building Excel file..."):
        try:
            excel_bytes = build_excel(results, selected_epoch_idx=epoch_for_spectrum)
            st.download_button(
                label="⬇️ Download Excel Report",
                data=excel_bytes,
                file_name="wave_height_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
            st.success("✅ Excel file created!")
        except Exception as e:
            st.error(f"Excel generation error: {e}")

st.markdown("---")

# ── PDF ───────────────────────────────────────────────────────────────────────
st.markdown("### 📄 PDF Export")
st.markdown("""
PDF report contains 5 pages:
- Page 1: Title, date range, summary statistics table
- Page 2: Hs time series chart (all 3 methods)
- Page 3: Energy spectrum chart
- Page 4: Scatter plot comparison
- Page 5: Full epoch table
""")

if st.button("📥 Generate & Download PDF", type="primary"):
    with st.spinner("Building PDF report (may take 10-30 seconds)..."):
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
            st.success("✅ PDF file created!")
        except Exception as e:
            st.error(f"PDF generation error: {e}")

st.markdown("---")

# ── Inline Preview ────────────────────────────────────────────────────────────
st.markdown("### 👀 Selected Epoch Preview")

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

st.table(pd.DataFrame(preview_data))

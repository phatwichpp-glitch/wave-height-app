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

st.title("📤 ส่งออกรายงาน")

if "results" not in st.session_state:
    st.warning("⚠️ กรุณาประมวลผลข้อมูลที่หน้า ⚙️ ก่อน")
    st.stop()

results = st.session_state["results"]
n_results = len(results)
n_samples = st.session_state.get("n_samples_raw", 0)
t_start = st.session_state.get("t_start", "")
t_end = st.session_state.get("t_end", "")
date_range = f"{t_start} → {t_end}" if t_start and t_end else "ไม่ระบุ"

st.markdown(f"""
**ข้อมูลสรุป:**
- ช่วงเวลา: **{date_range}**
- จำนวนตัวอย่าง: **{n_samples:,}**
- จำนวน Epoch: **{n_results}**
""")

# ─── Epoch selector for spectrum sheet / PDF spectrum page ────────────────────
epoch_for_spectrum = st.number_input(
    "Epoch สำหรับหน้า Spectrum ใน Report",
    min_value=1, max_value=n_results, value=1, step=1
) - 1  # convert to 0-based

st.markdown("---")

# ─── Excel Export ─────────────────────────────────────────────────────────────
st.markdown("### 📊 ส่งออก Excel")
st.markdown("""
ไฟล์ Excel ประกอบด้วย 3 Sheet:
- **สรุป** — ค่าสถิติแต่ละวิธี Hs
- **อนุกรมเวลา** — ตาราง Epoch ทั้งหมด
- **สเปกตรัม** — ค่า f และ Pxx ของ Epoch ที่เลือก
""")

if st.button("📥 สร้างและดาวน์โหลด Excel", type="primary"):
    with st.spinner("กำลังสร้างไฟล์ Excel..."):
        try:
            excel_bytes = build_excel(results, selected_epoch_idx=epoch_for_spectrum)
            st.download_button(
                label="⬇️ ดาวน์โหลด Excel Report",
                data=excel_bytes,
                file_name="wave_height_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
            st.success("✅ สร้างไฟล์ Excel สำเร็จ!")
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการสร้าง Excel: {e}")

st.markdown("---")

# ─── PDF Export ───────────────────────────────────────────────────────────────
st.markdown("### 📄 ส่งออก PDF")
st.markdown("""
ไฟล์ PDF ประกอบด้วย 5 หน้า:
- หน้า 1: หัวเรื่อง, ช่วงเวลา, ตารางสรุปสถิติ
- หน้า 2: กราฟอนุกรมเวลา Hs (ทั้ง 3 วิธี)
- หน้า 3: กราฟสเปกตรัมพลังงาน
- หน้า 4: Scatter Plot เปรียบเทียบวิธีการ
- หน้า 5: ตาราง Epoch ทั้งหมด
""")

# Check if Thai font is available for PDF
THAI_FONT_PATH = "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf"
if not os.path.exists(THAI_FONT_PATH):
    st.warning("⚠️ ไม่พบฟอนต์ภาษาไทย — PDF จะใช้ฟอนต์ Helvetica แทน")

if st.button("📥 สร้างและดาวน์โหลด PDF", type="primary"):
    with st.spinner("กำลังสร้างไฟล์ PDF (อาจใช้เวลา 10-30 วินาที)..."):
        try:
            pdf_bytes = build_pdf(
                results=results,
                epoch_idx=epoch_for_spectrum,
                date_range=date_range,
                n_samples=n_samples,
            )
            st.download_button(
                label="⬇️ ดาวน์โหลด PDF Report",
                data=pdf_bytes,
                file_name="wave_height_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            st.success("✅ สร้างไฟล์ PDF สำเร็จ!")
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการสร้าง PDF: {e}")

st.markdown("---")

# ─── Inline Preview ───────────────────────────────────────────────────────────
st.markdown("### 👀 ตัวอย่างข้อมูล Epoch ที่เลือกสำหรับ Report")

r = results[epoch_for_spectrum]

def fmt_v(v, dec=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.{dec}f}"

preview_data = {
    "พารามิเตอร์": [
        "Hs_rect (m)", "Hs_spec (m)", "Hs_zc (m)",
        "Tp (s)", "Tm02 (s)", "fp (Hz)", "H_max (m)", "จำนวนคลื่น"
    ],
    "ค่า": [
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

"""
pages/2_processing.py
Signal Processing & Hs Calculation page.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.signal_proc import process_epoch
from core.wave_methods import compute_all_methods

st.title("⚙️ ประมวลผลสัญญาณและคำนวณ Hs")

# ─── Check prerequisite ───────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.warning("⚠️ กรุณาอัปโหลดข้อมูลที่หน้า 📁 ก่อน")
    st.stop()

df: pd.DataFrame = st.session_state["df"]
fs: float = st.session_state.get("fs", 5.0)
epoch_sec: int = st.session_state.get("epoch_sec", 60)
hp_cutoff: float = st.session_state.get("hp_cutoff", 0.05)
lp_cutoff: float = st.session_state.get("lp_cutoff", 2.0)

epoch_samples = int(fs * epoch_sec)
min_samples_spectral = 64

st.markdown(f"""
**พารามิเตอร์การประมวลผล:**
- fs = **{fs} Hz** | Epoch = **{epoch_sec} วินาที** ({epoch_samples} ตัวอย่าง)
- Bandpass: **{hp_cutoff}–{lp_cutoff} Hz** | High-pass drift: **{hp_cutoff} Hz**
""")

# ─── Validate required columns ────────────────────────────────────────────────
for col in ["accel_x", "accel_y", "accel_z"]:
    if col not in df.columns:
        st.error(f"ไม่พบคอลัมน์ '{col}' ในข้อมูล")
        st.stop()

# ─── Epoch splitting ──────────────────────────────────────────────────────────
n_total = len(df)
n_epochs = n_total // epoch_samples

if n_epochs == 0:
    st.error(f"ข้อมูลน้อยเกินไป ({n_total} ตัวอย่าง) สำหรับ 1 Epoch ({epoch_samples} ตัวอย่าง)")
    st.stop()

st.info(f"🔢 จำนวน Epoch ที่จะวิเคราะห์: **{n_epochs}**")

# ─── Run processing ───────────────────────────────────────────────────────────
run_btn = st.button("▶️ เริ่มประมวลผล", type="primary", use_container_width=True)

if run_btn or ("results" not in st.session_state):
    results = []
    progress = st.progress(0, text="กำลังประมวลผล Epoch...")
    status_text = st.empty()

    ax_all = df["accel_x"].values.astype(float)
    ay_all = df["accel_y"].values.astype(float)
    az_all = df["accel_z"].values.astype(float)

    # Timestamps for epoch labelling
    has_ts = "timestamp" in df.columns
    if has_ts:
        ts_all = df["timestamp"].values

    for i in range(n_epochs):
        i_start = i * epoch_samples
        i_end = i_start + epoch_samples

        ax_ep = ax_all[i_start:i_end]
        ay_ep = ay_all[i_start:i_end]
        az_ep = az_all[i_start:i_end]

        # Skip epoch if > 10% NaN
        nan_frac = np.mean(np.isnan(ax_ep) | np.isnan(ay_ep) | np.isnan(az_ep))
        if nan_frac > 0.10:
            status_text.text(f"Epoch {i+1}: ข้ามเนื่องจาก NaN {nan_frac*100:.1f}%")
            continue

        # Fill remaining NaN with interpolation
        for arr_ref in [ax_ep, ay_ep, az_ep]:
            nan_idx = np.isnan(arr_ref)
            if nan_idx.any():
                arr_ref[nan_idx] = np.interp(
                    np.where(nan_idx)[0],
                    np.where(~nan_idx)[0],
                    arr_ref[~nan_idx]
                )

        # Warn if too few samples for spectral
        if epoch_samples < min_samples_spectral:
            status_text.text(f"Epoch {i+1}: ⚠️ น้อยกว่า {min_samples_spectral} ตัวอย่าง — Spectral อาจไม่แม่นยำ")

        try:
            disp = process_epoch(ax_ep, ay_ep, az_ep, fs, hp_cutoff, lp_cutoff)
        except Exception as e:
            status_text.text(f"Epoch {i+1}: เกิดข้อผิดพลาด: {e}")
            continue

        ep_start = float(i_start) / fs
        ep_end = float(i_end) / fs

        if has_ts:
            try:
                ep_start = pd.Timestamp(ts_all[i_start]).isoformat()
                ep_end = pd.Timestamp(ts_all[min(i_end, len(ts_all)-1)]).isoformat()
            except Exception:
                pass

        result = compute_all_methods(
            ax_ep, ay_ep, az_ep, disp, fs,
            lowcut=hp_cutoff, highcut=lp_cutoff,
            epoch_start=ep_start, epoch_end=ep_end
        )
        result["disp"] = disp  # store displacement for spectrum page
        results.append(result)

        progress.progress((i + 1) / n_epochs,
                          text=f"Epoch {i+1}/{n_epochs}")

    progress.progress(1.0, text="✅ ประมวลผลเสร็จสิ้น")
    status_text.empty()

    if not results:
        st.error("ไม่มี Epoch ที่ผ่านการประมวลผล")
        st.stop()

    st.session_state["results"] = results
    st.success(f"✅ ประมวลผล {len(results)} Epoch เสร็จสิ้น")

elif "results" in st.session_state:
    results = st.session_state["results"]
    st.info(f"ℹ️ ใช้ผลลัพธ์จากการประมวลผลล่าสุด ({len(results)} Epoch) — กด ▶️ เพื่อประมวลผลใหม่")
else:
    st.stop()

results = st.session_state.get("results", [])
if not results:
    st.stop()

# ─── Display Results ──────────────────────────────────────────────────────────
st.markdown("### 📊 ผลลัพธ์การคำนวณ Hs")

result_rows = []
for i, r in enumerate(results):
    row = {
        "Epoch": i + 1,
        "Hs_rect (m)": round(r.get("Hs_rect", np.nan), 4),
        "Hs_spec (m)": round(r.get("Hs_spec", np.nan) if not np.isnan(r.get("Hs_spec", np.nan)) else np.nan, 4),
        "Hs_zc (m)":   round(r.get("Hs_zc", np.nan) if not np.isnan(r.get("Hs_zc", np.nan)) else np.nan, 4),
        "Tp (s)":       round(r.get("Tp", np.nan), 2) if not np.isnan(r.get("Tp", np.nan)) else np.nan,
        "Tm02 (s)":     round(r.get("Tm02", np.nan), 2) if not np.isnan(r.get("Tm02", np.nan)) else np.nan,
        "H_max (m)":    round(r.get("H_max", np.nan), 4) if not np.isnan(r.get("H_max", np.nan)) else np.nan,
        "wave_count":   r.get("wave_count", 0),
    }
    result_rows.append(row)

results_df = pd.DataFrame(result_rows)
st.dataframe(results_df, use_container_width=True)

# ─── Displacement Preview ─────────────────────────────────────────────────────
st.markdown("### 🌊 ตัวอย่างการกระจัดคลื่น (Epoch แรก)")

if results and "disp" in results[0]:
    disp0 = results[0]["disp"]
    t_axis = np.arange(len(disp0)) / fs

    fig, ax = plt.subplots(figsize=(12, 3), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(t_axis, disp0, color="steelblue", linewidth=0.8)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    hs_r = results[0].get("Hs_rect", np.nan)
    if not np.isnan(hs_r):
        ax.axhline(hs_r / 2, color="red", linewidth=1, linestyle=":",
                   label=f"+Hs/2 = {hs_r/2:.3f} m")
        ax.axhline(-hs_r / 2, color="red", linewidth=1, linestyle=":")
    ax.set_xlabel("เวลา (วินาที)")
    ax.set_ylabel("การกระจัด (m)")
    ax.set_title("การกระจัดคลื่นหลังจากการประมวลผล (Epoch 1)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ─── Quick stats ──────────────────────────────────────────────────────────────
st.markdown("### 📐 สถิติสรุป")

def safe_vals(key):
    return np.array([r[key] for r in results if not np.isnan(r.get(key, np.nan))])

col1, col2, col3 = st.columns(3)
for col_ui, key, label in [
    (col1, "Hs_rect", "Hs_rect"),
    (col2, "Hs_spec", "Hs_spec"),
    (col3, "Hs_zc",   "Hs_zc"),
]:
    vals = safe_vals(key)
    if len(vals):
        col_ui.metric(f"เฉลี่ย {label}", f"{np.mean(vals):.3f} m")
        col_ui.caption(f"สูงสุด {np.max(vals):.3f} m | ต่ำสุด {np.min(vals):.3f} m | σ {np.std(vals):.3f} m")
    else:
        col_ui.metric(f"เฉลี่ย {label}", "N/A")

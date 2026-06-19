"""
pages/2_processing.py
Signal Processing & Hs Calculation page.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.signal_proc import process_epoch
from core.wave_methods import compute_all_methods

st.title("⚙️ Signal Processing & Hs Calculation")

if "df" not in st.session_state:
    st.warning("⚠️ Please upload data on the 📁 Upload page first.")
    st.stop()

df: pd.DataFrame = st.session_state["df"]
fs: float        = st.session_state.get("fs", 5.0)
epoch_sec: int   = st.session_state.get("epoch_sec", 60)
hp_cutoff: float = st.session_state.get("hp_cutoff", 0.05)
lp_cutoff: float = st.session_state.get("lp_cutoff", 2.0)
epoch_samples    = int(fs * epoch_sec)

st.markdown(f"""
**Processing parameters:**
- fs = **{fs} Hz** | Epoch = **{epoch_sec} s** ({epoch_samples} samples)
- Bandpass: **{hp_cutoff}–{lp_cutoff} Hz**
""")

for col in ["accel_x", "accel_y", "accel_z"]:
    if col not in df.columns:
        st.error(f"Column '{col}' not found in data.")
        st.stop()

n_epochs = len(df) // epoch_samples
if n_epochs == 0:
    st.error(f"Not enough data ({len(df)} samples) for 1 epoch ({epoch_samples} samples).")
    st.stop()

st.info(f"🔢 Epochs to process: **{n_epochs}**")

run_btn = st.button("▶️ Run Processing", type="primary", use_container_width=True)

if run_btn or "results" not in st.session_state:
    results = []
    progress = st.progress(0, text="Processing epochs...")
    status_text = st.empty()

    ax_all = df["accel_x"].values.astype(float)
    ay_all = df["accel_y"].values.astype(float)
    az_all = df["accel_z"].values.astype(float)
    has_ts = "timestamp" in df.columns
    if has_ts:
        ts_all = df["timestamp"].values

    for i in range(n_epochs):
        i0, i1 = i * epoch_samples, (i + 1) * epoch_samples
        ax_ep = ax_all[i0:i1].copy()
        ay_ep = ay_all[i0:i1].copy()
        az_ep = az_all[i0:i1].copy()

        nan_frac = np.mean(np.isnan(ax_ep) | np.isnan(ay_ep) | np.isnan(az_ep))
        if nan_frac > 0.10:
            status_text.text(f"Epoch {i+1}: skipped ({nan_frac*100:.1f}% NaN)")
            continue

        for arr in [ax_ep, ay_ep, az_ep]:
            nan_idx = np.isnan(arr)
            if nan_idx.any():
                arr[nan_idx] = np.interp(np.where(nan_idx)[0],
                                         np.where(~nan_idx)[0], arr[~nan_idx])

        try:
            disp = process_epoch(ax_ep, ay_ep, az_ep, fs, hp_cutoff, lp_cutoff)
        except Exception as e:
            status_text.text(f"Epoch {i+1}: error — {e}")
            continue

        ep_start = float(i0) / fs
        ep_end   = float(i1) / fs
        if has_ts:
            try:
                ep_start = pd.Timestamp(ts_all[i0]).isoformat()
                ep_end   = pd.Timestamp(ts_all[min(i1, len(ts_all)-1)]).isoformat()
            except Exception:
                pass

        result = compute_all_methods(ax_ep, ay_ep, az_ep, disp, fs,
                                     lowcut=hp_cutoff, highcut=lp_cutoff,
                                     epoch_start=ep_start, epoch_end=ep_end)
        result["disp"] = disp
        results.append(result)
        progress.progress((i + 1) / n_epochs, text=f"Epoch {i+1}/{n_epochs}")

    progress.progress(1.0, text="✅ Done")
    status_text.empty()

    if not results:
        st.error("No epochs were successfully processed.")
        st.stop()

    st.session_state["results"] = results
    st.success(f"✅ Processed {len(results)} epochs.")

elif "results" in st.session_state:
    results = st.session_state["results"]
    st.info(f"Using cached results ({len(results)} epochs) — press ▶️ to reprocess.")
else:
    st.stop()

results = st.session_state.get("results", [])
if not results:
    st.stop()

# ── Results Table ─────────────────────────────────────────────────────────────
st.markdown("### 📊 Hs Results per Epoch")

rows = []
for i, r in enumerate(results):
    def rv(k): return round(r.get(k, np.nan), 4) if not np.isnan(r.get(k, np.nan)) else np.nan
    rows.append({
        "Epoch": i + 1,
        "Hs_rect (m)": rv("Hs_rect"),
        "Hs_spec (m)": rv("Hs_spec"),
        "Hs_zc (m)":   rv("Hs_zc"),
        "Tp (s)":      round(r.get("Tp", np.nan), 2) if not np.isnan(r.get("Tp", np.nan)) else np.nan,
        "Tm02 (s)":    round(r.get("Tm02", np.nan), 2) if not np.isnan(r.get("Tm02", np.nan)) else np.nan,
        "H_max (m)":   rv("H_max"),
        "wave_count":  r.get("wave_count", 0),
    })

st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ── Displacement Preview ──────────────────────────────────────────────────────
st.markdown("### 🌊 Displacement Preview (Epoch 1)")

if results and "disp" in results[0]:
    disp0 = results[0]["disp"]
    t_axis = np.arange(len(disp0)) / fs
    hs_r = results[0].get("Hs_rect", np.nan)

    fig, ax = plt.subplots(figsize=(12, 3), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(t_axis, disp0, color="steelblue", linewidth=0.8)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    if not np.isnan(hs_r):
        ax.axhline( hs_r / 2, color="red", linewidth=1, linestyle=":",
                   label=f"+Hs/2 = {hs_r/2:.3f} m")
        ax.axhline(-hs_r / 2, color="red", linewidth=1, linestyle=":")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Displacement (m)")
    ax.set_title("Wave Displacement after Processing (Epoch 1)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ── Summary Stats ─────────────────────────────────────────────────────────────
st.markdown("### 📐 Summary Statistics")

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
        col_ui.metric(f"Mean {label}", f"{np.mean(vals):.3f} m")
        col_ui.caption(f"Max {np.max(vals):.3f} m | Min {np.min(vals):.3f} m | SD {np.std(vals):.3f} m")
    else:
        col_ui.metric(f"Mean {label}", "N/A")

"""
pages/3_spectrum.py
Spectral Analysis page.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from core.signal_proc import compute_spectrogram
from core.wave_methods import jonswap_spectrum

st.title("📊 การวิเคราะห์สเปกตรัม")

if "results" not in st.session_state:
    st.warning("⚠️ กรุณาประมวลผลข้อมูลที่หน้า ⚙️ ก่อน")
    st.stop()

results = st.session_state["results"]
fs = st.session_state.get("fs", 5.0)
hp_cutoff = st.session_state.get("hp_cutoff", 0.05)
lp_cutoff = st.session_state.get("lp_cutoff", 2.0)
n_results = len(results)

# ─── Epoch Selector ───────────────────────────────────────────────────────────
st.markdown("### 🔢 เลือก Epoch")
epoch_idx = st.slider("Epoch", min_value=1, max_value=n_results,
                       value=1, step=1) - 1

r = results[epoch_idx]
f_arr = r.get("f_arr")
Pxx_arr = r.get("Pxx_arr")
Hs_spec = r.get("Hs_spec", np.nan)
Tp = r.get("Tp", np.nan)
Tm02 = r.get("Tm02", np.nan)
fp = r.get("fp", np.nan)
disp = r.get("disp")

# ─── JONSWAP toggle ───────────────────────────────────────────────────────────
show_jonswap = st.toggle("แสดง JONSWAP Reference Spectrum", value=False)

# ─── Energy Spectrum Plot ─────────────────────────────────────────────────────
st.markdown("### 🌊 สเปกตรัมพลังงาน S(f)")

if f_arr is not None and Pxx_arr is not None and len(f_arr) > 0:
    col_chart, col_stats = st.columns([2, 1])

    with col_chart:
        fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="white")
        ax.set_facecolor("white")

        ax.fill_between(f_arr, Pxx_arr, alpha=0.35, color="steelblue")
        ax.plot(f_arr, Pxx_arr, color="steelblue", linewidth=1.5,
                label="สเปกตรัม Welch")

        # JONSWAP overlay
        if show_jonswap and not np.isnan(Hs_spec) and not np.isnan(Tp) and Tp > 0:
            S_jonswap = jonswap_spectrum(f_arr, Hs_spec, Tp)
            ax.plot(f_arr, S_jonswap, color="darkorange", linewidth=1.5,
                    linestyle="--", label="JONSWAP (γ=3.3)")

        # Peak frequency line
        if not np.isnan(fp):
            ax.axvline(fp, color="red", linestyle="--", linewidth=1.2,
                       label=f"fp = {fp:.4f} Hz")

        ax.set_xlabel("ความถี่ (Hz)", fontsize=11)
        ax.set_ylabel("ความหนาแน่นสเปกตรัม (m²/Hz)", fontsize=11)
        ax.set_title(f"สเปกตรัมพลังงาน — Epoch {epoch_idx + 1}", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_stats:
        st.markdown("#### พารามิเตอร์คลื่น")

        def fmt(v, dec=3, unit=""):
            if v is None or np.isnan(v):
                return "N/A"
            return f"{v:.{dec}f} {unit}".strip()

        st.metric("Hs_spec", fmt(Hs_spec, 3, "m"))
        st.metric("Tp (คาบคลื่นหลัก)", fmt(Tp, 2, "s"))
        st.metric("Tm02 (คาบเฉลี่ย)", fmt(Tm02, 2, "s"))
        st.metric("fp (ความถี่หลัก)", fmt(fp, 4, "Hz"))

        hs_r = r.get("Hs_rect", np.nan)
        hs_z = r.get("Hs_zc", np.nan)
        h_max = r.get("H_max", np.nan)
        wc = r.get("wave_count", 0)

        st.markdown("---")
        st.metric("Hs_rect", fmt(hs_r, 3, "m"))
        st.metric("Hs_zc", fmt(hs_z, 3, "m"))
        st.metric("H_max", fmt(h_max, 3, "m"))
        st.metric("จำนวนคลื่น", str(wc))

else:
    st.warning(f"ไม่มีข้อมูลสเปกตรัมสำหรับ Epoch {epoch_idx + 1}")

# ─── Spectrogram ─────────────────────────────────────────────────────────────
st.markdown("### 🗺️ สเปกโตแกรม (Time-Frequency)")

if disp is not None and len(disp) >= 64:
    try:
        nperseg = min(256, len(disp) // 4)
        if nperseg < 8:
            nperseg = 8
        t_sg, f_sg, Sxx = compute_spectrogram(disp, fs, nperseg=nperseg)

        # Mask to wave frequency band
        band_mask = (f_sg >= hp_cutoff) & (f_sg <= lp_cutoff)
        f_band = f_sg[band_mask]
        Sxx_band = Sxx[band_mask, :]

        if Sxx_band.size > 0 and np.any(Sxx_band > 0):
            fig2, ax2 = plt.subplots(figsize=(12, 4), facecolor="white")
            ax2.set_facecolor("white")

            Sxx_plot = np.maximum(Sxx_band, 1e-12)
            pcm = ax2.pcolormesh(
                t_sg, f_band, Sxx_plot,
                norm=LogNorm(vmin=Sxx_plot.min(), vmax=Sxx_plot.max()),
                cmap="viridis", shading="gouraud"
            )
            cbar = fig2.colorbar(pcm, ax=ax2, label="กำลัง (m²/Hz)")
            cbar.ax.tick_params(labelsize=8)

            ax2.set_xlabel("เวลา (วินาที)", fontsize=11)
            ax2.set_ylabel("ความถี่ (Hz)", fontsize=11)
            ax2.set_title(f"สเปกโตแกรม — Epoch {epoch_idx + 1}", fontsize=12)
            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)
        else:
            st.info("สัญญาณในช่วงความถี่คลื่นมีพลังงานต่ำเกินไป")

    except Exception as e:
        st.warning(f"ไม่สามารถสร้างสเปกโตแกรม: {e}")
else:
    st.info("ต้องการข้อมูลการกระจัดอย่างน้อย 64 ตัวอย่างสำหรับสเปกโตแกรม")

# ─── Multi-Epoch Spectrum Overlay ─────────────────────────────────────────────
st.markdown("### 📈 เปรียบเทียบสเปกตรัมระหว่าง Epoch")

max_overlay = min(n_results, 10)
selected_epochs = st.multiselect(
    "เลือก Epoch ที่ต้องการเปรียบเทียบ (สูงสุด 10)",
    options=list(range(1, n_results + 1)),
    default=[1] if n_results >= 1 else [],
    max_selections=max_overlay,
    format_func=lambda x: f"Epoch {x}"
)

if len(selected_epochs) > 0:
    fig3, ax3 = plt.subplots(figsize=(11, 4.5), facecolor="white")
    ax3.set_facecolor("white")
    cmap = plt.cm.get_cmap("tab10", len(selected_epochs))

    for ci, ep_num in enumerate(selected_epochs):
        idx = ep_num - 1
        if 0 <= idx < n_results:
            rv = results[idx]
            fa = rv.get("f_arr")
            pa = rv.get("Pxx_arr")
            if fa is not None and pa is not None and len(fa) > 0:
                ax3.plot(fa, pa, color=cmap(ci), linewidth=1.2,
                         alpha=0.8, label=f"Epoch {ep_num}")

    ax3.set_xlabel("ความถี่ (Hz)", fontsize=10)
    ax3.set_ylabel("ความหนาแน่นสเปกตรัม (m²/Hz)", fontsize=10)
    ax3.set_title("เปรียบเทียบสเปกตรัมหลาย Epoch", fontsize=11)
    ax3.legend(fontsize=8, loc="upper right")
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(left=0)
    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

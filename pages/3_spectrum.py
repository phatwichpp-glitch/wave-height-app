"""
pages/3_spectrum.py
Spectral Analysis page.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from core.signal_proc import compute_spectrogram
from core.wave_methods import jonswap_spectrum

st.title("📊 Spectral Analysis")

if "results" not in st.session_state:
    st.warning("⚠️ Please process data on the ⚙️ Processing page first.")
    st.stop()

results    = st.session_state["results"]
fs         = st.session_state.get("fs", 5.0)
hp_cutoff  = st.session_state.get("hp_cutoff", 0.05)
lp_cutoff  = st.session_state.get("lp_cutoff", 2.0)
n_results  = len(results)

# ── Epoch Selector ────────────────────────────────────────────────────────────
st.markdown("### 🔢 Select Epoch")
epoch_idx = st.slider("Epoch", min_value=1, max_value=n_results, value=1) - 1

r       = results[epoch_idx]
f_arr   = r.get("f_arr")
Pxx_arr = r.get("Pxx_arr")
Hs_spec = r.get("Hs_spec", np.nan)
Tp      = r.get("Tp", np.nan)
Tm02    = r.get("Tm02", np.nan)
fp      = r.get("fp", np.nan)
disp    = r.get("disp")

show_jonswap = st.toggle("Show JONSWAP Reference Spectrum", value=False)

# ── Energy Spectrum ───────────────────────────────────────────────────────────
st.markdown("### 🌊 Energy Spectrum S(f)")

def fmt(v, dec=3, unit=""):
    return "N/A" if (v is None or np.isnan(v)) else f"{v:.{dec}f} {unit}".strip()

if f_arr is not None and Pxx_arr is not None and len(f_arr) > 0:
    col_chart, col_stats = st.columns([2, 1])

    with col_chart:
        fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="white")
        ax.set_facecolor("white")
        ax.fill_between(f_arr, Pxx_arr, alpha=0.35, color="steelblue")
        ax.plot(f_arr, Pxx_arr, color="steelblue", linewidth=1.5, label="Welch spectrum")

        if show_jonswap and not np.isnan(Hs_spec) and not np.isnan(Tp) and Tp > 0:
            S_j = jonswap_spectrum(f_arr, Hs_spec, Tp)
            ax.plot(f_arr, S_j, color="darkorange", linewidth=1.5,
                    linestyle="--", label="JONSWAP (γ=3.3)")

        if not np.isnan(fp):
            ax.axvline(fp, color="red", linestyle="--", linewidth=1.2,
                       label=f"fp = {fp:.4f} Hz")

        ax.set_xlabel("Frequency (Hz)", fontsize=11)
        ax.set_ylabel("Spectral Density (m²/Hz)", fontsize=11)
        ax.set_title(f"Energy Spectrum — Epoch {epoch_idx + 1}", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_stats:
        st.markdown("#### Wave Parameters")
        st.metric("Hs_spec",        fmt(Hs_spec, 3, "m"))
        st.metric("Tp (peak period)", fmt(Tp, 2, "s"))
        st.metric("Tm02 (mean period)", fmt(Tm02, 2, "s"))
        st.metric("fp (peak freq)",  fmt(fp, 4, "Hz"))
        st.markdown("---")
        st.metric("Hs_rect",  fmt(r.get("Hs_rect"), 3, "m"))
        st.metric("Hs_zc",    fmt(r.get("Hs_zc"),   3, "m"))
        st.metric("H_max",    fmt(r.get("H_max"),    3, "m"))
        st.metric("Wave count", str(r.get("wave_count", 0)))
else:
    st.warning(f"No spectrum data for Epoch {epoch_idx + 1}.")

# ── Spectrogram ───────────────────────────────────────────────────────────────
st.markdown("### 🗺️ Spectrogram (Time-Frequency)")

if disp is not None and len(disp) >= 64:
    try:
        nperseg = max(8, min(256, len(disp) // 4))
        t_sg, f_sg, Sxx = compute_spectrogram(disp, fs, nperseg=nperseg)
        band_mask = (f_sg >= hp_cutoff) & (f_sg <= lp_cutoff)
        f_band   = f_sg[band_mask]
        Sxx_band = Sxx[band_mask, :]

        if Sxx_band.size > 0 and np.any(Sxx_band > 0):
            fig2, ax2 = plt.subplots(figsize=(12, 4), facecolor="white")
            ax2.set_facecolor("white")
            Sxx_plot = np.maximum(Sxx_band, 1e-12)
            pcm = ax2.pcolormesh(t_sg, f_band, Sxx_plot,
                                 norm=LogNorm(vmin=Sxx_plot.min(), vmax=Sxx_plot.max()),
                                 cmap="viridis", shading="gouraud")
            fig2.colorbar(pcm, ax=ax2, label="Power (m²/Hz)")
            ax2.set_xlabel("Time (s)", fontsize=11)
            ax2.set_ylabel("Frequency (Hz)", fontsize=11)
            ax2.set_title(f"Spectrogram — Epoch {epoch_idx + 1}", fontsize=12)
            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)
        else:
            st.info("Signal power in wave frequency band is too low.")
    except Exception as e:
        st.warning(f"Cannot generate spectrogram: {e}")
else:
    st.info("At least 64 displacement samples required for spectrogram.")

# ── Multi-Epoch Spectrum Overlay ──────────────────────────────────────────────
st.markdown("### 📈 Multi-Epoch Spectrum Comparison")

max_overlay = min(n_results, 10)
selected_epochs = st.multiselect(
    "Select epochs to compare (max 10)",
    options=list(range(1, n_results + 1)),
    default=[1] if n_results >= 1 else [],
    max_selections=max_overlay,
    format_func=lambda x: f"Epoch {x}"
)

if selected_epochs:
    fig3, ax3 = plt.subplots(figsize=(11, 4.5), facecolor="white")
    ax3.set_facecolor("white")
    cmap = plt.cm.get_cmap("tab10", len(selected_epochs))
    for ci, ep_num in enumerate(selected_epochs):
        idx = ep_num - 1
        if 0 <= idx < n_results:
            rv = results[idx]
            fa, pa = rv.get("f_arr"), rv.get("Pxx_arr")
            if fa is not None and pa is not None and len(fa) > 0:
                ax3.plot(fa, pa, color=cmap(ci), linewidth=1.2,
                         alpha=0.8, label=f"Epoch {ep_num}")
    ax3.set_xlabel("Frequency (Hz)", fontsize=10)
    ax3.set_ylabel("Spectral Density (m²/Hz)", fontsize=10)
    ax3.set_title("Spectrum Comparison — Multiple Epochs", fontsize=11)
    ax3.legend(fontsize=8, loc="upper right")
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(left=0)
    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

"""
pages/4_comparison.py
Method Comparison Dashboard page.
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
from scipy.stats import pearsonr

st.title("📈 แดชบอร์ดเปรียบเทียบวิธีการคำนวณ Hs")

if "results" not in st.session_state:
    st.warning("⚠️ กรุณาประมวลผลข้อมูลที่หน้า ⚙️ ก่อน")
    st.stop()

results = st.session_state["results"]
n = len(results)
epochs = np.arange(1, n + 1)

Hs_rect = np.array([r.get("Hs_rect", np.nan) for r in results])
Hs_spec = np.array([r.get("Hs_spec", np.nan) for r in results])
Hs_zc   = np.array([r.get("Hs_zc", np.nan) for r in results])
Tp      = np.array([r.get("Tp", np.nan) for r in results])
Tm02    = np.array([r.get("Tm02", np.nan) for r in results])
H_max   = np.array([r.get("H_max", np.nan) for r in results])


def safe_rmse(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if np.sum(mask) < 2:
        return np.nan
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def safe_r2(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if np.sum(mask) < 2:
        return np.nan
    r, _ = pearsonr(a[mask], b[mask])
    return float(r ** 2)


# ─── 1. Hs Time Series ────────────────────────────────────────────────────────
st.markdown("### 🌊 อนุกรมเวลา Hs (ทั้ง 3 วิธี)")

fig1, ax1 = plt.subplots(figsize=(12, 4), facecolor="white")
ax1.set_facecolor("white")
ax1.plot(epochs, Hs_rect, "b-o", markersize=3, linewidth=1, label="Hs_rect (สี่เหลี่ยม)")
ax1.plot(epochs, Hs_spec, "r-s", markersize=3, linewidth=1, label="Hs_spec (สเปกตรัม)")
ax1.plot(epochs, Hs_zc,   "g-^", markersize=3, linewidth=1, label="Hs_zc (ข้ามศูนย์)")
ax1.set_xlabel("Epoch", fontsize=10)
ax1.set_ylabel("Hs (m)", fontsize=10)
ax1.set_title("อนุกรมเวลาความสูงคลื่นนัยสำคัญ (Hs)", fontsize=11)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
st.pyplot(fig1)
plt.close(fig1)

# ─── 2. Scatter Plots ─────────────────────────────────────────────────────────
st.markdown("### 📐 Scatter Plot เปรียบเทียบ")

fig2, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")

scatter_pairs = [
    (Hs_rect, Hs_spec, "Hs_rect (m)", "Hs_spec (m)", "red", "Hs_rect vs Hs_spec"),
    (Hs_rect, Hs_zc,   "Hs_rect (m)", "Hs_zc (m)",  "green", "Hs_rect vs Hs_zc"),
]

for ax, (x_arr, y_arr, xlabel, ylabel, color, title) in zip(axes, scatter_pairs):
    ax.set_facecolor("white")
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if np.sum(mask) > 1:
        x_m, y_m = x_arr[mask], y_arr[mask]
        ax.scatter(x_m, y_m, c=color, alpha=0.6, s=30, edgecolors="white", linewidths=0.3)

        lim_min = min(x_m.min(), y_m.min()) * 0.95
        lim_max = max(x_m.max(), y_m.max()) * 1.05
        ax.plot([lim_min, lim_max], [lim_min, lim_max],
                "k--", linewidth=1, label="1:1")

        r2 = safe_r2(x_arr, y_arr)
        rmse = safe_rmse(x_arr, y_arr)
        ax.set_title(f"{title}\nR²={r2:.3f}  RMSE={rmse:.4f} m", fontsize=10)
        ax.legend(fontsize=8)
    else:
        ax.set_title(f"{title}\nข้อมูลไม่เพียงพอ", fontsize=10)

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.3)

fig2.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

# ─── 3. Metrics Table ─────────────────────────────────────────────────────────
st.markdown("### 📋 ตารางสถิติแต่ละวิธี")

rmse_rect_spec = safe_rmse(Hs_rect, Hs_spec)
rmse_zc_spec   = safe_rmse(Hs_zc, Hs_spec)

rows = []
for label, arr in [("Hs_rect", Hs_rect), ("Hs_spec", Hs_spec), ("Hs_zc", Hs_zc)]:
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        rows.append({"วิธี": label, "ค่าเฉลี่ย": "-", "ค่าสูงสุด": "-",
                      "ค่าต่ำสุด": "-", "S.D.": "-", "RMSE vs Hs_spec": "-"})
        continue
    rmse_vs = (rmse_rect_spec if label == "Hs_rect"
               else (0.0 if label == "Hs_spec" else rmse_zc_spec))
    rows.append({
        "วิธี": label,
        "ค่าเฉลี่ย (m)": f"{np.mean(valid):.4f}",
        "ค่าสูงสุด (m)": f"{np.max(valid):.4f}",
        "ค่าต่ำสุด (m)": f"{np.min(valid):.4f}",
        "S.D. (m)": f"{np.std(valid):.4f}",
        "RMSE vs Hs_spec (m)": f"{rmse_vs:.4f}" if not np.isnan(rmse_vs) else "-",
    })

st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ─── 4. Bias Histogram ────────────────────────────────────────────────────────
st.markdown("### 📊 ฮิสโตแกรม Bias (Hs_rect − Hs_spec)")

mask_both = np.isfinite(Hs_rect) & np.isfinite(Hs_spec)
if np.sum(mask_both) > 1:
    bias = Hs_rect[mask_both] - Hs_spec[mask_both]
    fig3, ax3 = plt.subplots(figsize=(8, 3.5), facecolor="white")
    ax3.set_facecolor("white")
    ax3.hist(bias, bins=min(20, max(5, n // 3)), color="steelblue",
             edgecolor="white", alpha=0.8)
    ax3.axvline(float(np.mean(bias)), color="red", linestyle="--", linewidth=1.2,
                label=f"Bias เฉลี่ย = {np.mean(bias):.4f} m")
    ax3.axvline(0, color="black", linewidth=0.8, linestyle="-")
    ax3.set_xlabel("Hs_rect − Hs_spec (m)", fontsize=10)
    ax3.set_ylabel("ความถี่", fontsize=10)
    ax3.set_title("การกระจาย Bias ระหว่าง Hs_rect และ Hs_spec", fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)
else:
    st.info("ข้อมูลไม่เพียงพอสำหรับฮิสโตแกรม Bias")

# ─── 5. Wave Period Panel ─────────────────────────────────────────────────────
st.markdown("### ⏱️ อนุกรมเวลาคาบคลื่น (Tp และ Tm02)")

mask_tp = np.isfinite(Tp)
mask_tm = np.isfinite(Tm02)

if mask_tp.any() or mask_tm.any():
    fig4, ax4 = plt.subplots(figsize=(12, 3.5), facecolor="white")
    ax4.set_facecolor("white")
    if mask_tp.any():
        ax4.plot(epochs[mask_tp], Tp[mask_tp], "b-o", markersize=3,
                 linewidth=1, label="Tp (คาบคลื่นหลัก)")
    if mask_tm.any():
        ax4.plot(epochs[mask_tm], Tm02[mask_tm], "r-s", markersize=3,
                 linewidth=1, label="Tm02 (คาบเฉลี่ย)")
    ax4.set_xlabel("Epoch", fontsize=10)
    ax4.set_ylabel("คาบ (วินาที)", fontsize=10)
    ax4.set_title("อนุกรมเวลาคาบคลื่น", fontsize=11)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    fig4.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)
else:
    st.info("ไม่มีข้อมูลคาบคลื่น")

# ─── 6. H_max vs Hs_zc ───────────────────────────────────────────────────────
st.markdown("### 🔺 H_max และ Hs_zc อนุกรมเวลา")

mask_hmax = np.isfinite(H_max)
mask_hszc = np.isfinite(Hs_zc)

if mask_hmax.any() or mask_hszc.any():
    fig5, ax5 = plt.subplots(figsize=(12, 3.5), facecolor="white")
    ax5.set_facecolor("white")
    if mask_hmax.any():
        ax5.plot(epochs[mask_hmax], H_max[mask_hmax], "r-o", markersize=3,
                 linewidth=1, label="H_max")
    if mask_hszc.any():
        ax5.fill_between(epochs[mask_hszc], Hs_zc[mask_hszc], alpha=0.3,
                         color="green")
        ax5.plot(epochs[mask_hszc], Hs_zc[mask_hszc], "g-s", markersize=3,
                 linewidth=1, label="Hs_zc")
    ax5.set_xlabel("Epoch", fontsize=10)
    ax5.set_ylabel("ความสูงคลื่น (m)", fontsize=10)
    ax5.set_title("H_max และ Hs_zc อนุกรมเวลา", fontsize=11)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    fig5.tight_layout()
    st.pyplot(fig5)
    plt.close(fig5)
else:
    st.info("ไม่มีข้อมูล H_max / Hs_zc")

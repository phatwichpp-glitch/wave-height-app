import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

st.set_page_config(page_title="Wave Height Calculator", page_icon="🌊", layout="wide")

# ─────────────────────────────────────────────
# Core calculation
# ─────────────────────────────────────────────
def calculate_hs(az_array, fs=5.0):
    """
    คำนวณ Significant Wave Height (Hs) จาก raw acceleration
    โดยวิธี Rectangular Integration (double cumsum)
    สูตร: Hs = 4 × std(displacement)
    """
    # 1. Detrend: ลบค่าเฉลี่ยออก
    Az = az_array - np.mean(az_array)

    # 2. Integrate #1: Acceleration → Velocity
    V = np.cumsum(Az / fs)
    V -= np.mean(V)          # ตัด DC drift

    # 3. Integrate #2: Velocity → Displacement
    S = np.cumsum(V / fs)
    S -= np.mean(S)          # ตัด DC drift

    # 4. Hs = 4 × σ(displacement)
    return 4.0 * np.std(S)


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("🌊 ระบบคำนวณ Wave Height (Hs)")
st.caption("คำนวณด้วยวิธี Rectangular Integration Method  |  Hs = 4 × σ(displacement)")

st.divider()

# ── Upload ──
uploaded_files = st.file_uploader(
    "📂 อัปโหลดไฟล์ CSV (เลือกได้หลายไฟล์)",
    type=["csv"],
    accept_multiple_files=True,
)

# ── Settings (sidebar) ──
with st.sidebar:
    st.header("⚙️ ตั้งค่าการคำนวณ")
    fs = st.number_input("Sampling rate (Hz)", min_value=1.0, max_value=100.0, value=5.0, step=0.5)
    epoch_sec = st.number_input("ความยาว epoch (วินาที)", min_value=10, max_value=600, value=60, step=10)
    samples_per_epoch = int(fs * epoch_sec)
    st.caption(f"= {samples_per_epoch} samples / epoch")

    st.divider()
    st.header("📊 ตัวเลือกกราฟ")
    show_raw = st.checkbox("แสดงกราฟ displacement ตัวอย่าง", value=False)

# ── Process ──
if uploaded_files:
    if st.button("🚀 เริ่มประมวลผล", type="primary"):
        with st.spinner("กำลังอ่านและรวมไฟล์..."):
            df_list = [pd.read_csv(f) for f in uploaded_files]
            full_df = pd.concat(df_list, ignore_index=True)

        with st.spinner("กำลังจัดเรียงและ interpolate..."):
            full_df["timestamp"] = pd.to_datetime(full_df["timestamp"], utc=True)
            full_df = (
                full_df.sort_values("timestamp")
                .drop_duplicates(subset=["timestamp"])
                .set_index("timestamp")
            )

            freq_str = f"{int(1000 / fs)}ms"
            reg_index = pd.date_range(
                start=full_df.index.min().ceil(freq_str),
                end=full_df.index.max().floor(freq_str),
                freq=freq_str,
            )
            df_combined = full_df.reindex(
                full_df.index.union(reg_index).sort_values()
            )
            df_combined["accel_z"] = df_combined["accel_z"].interpolate(method="time")
            df_reg = df_combined.loc[reg_index].copy()

        with st.spinner("กำลังคำนวณ Hs แต่ละ epoch..."):
            results = []
            n_epochs = len(df_reg) // samples_per_epoch
            progress = st.progress(0, text="คำนวณ epoch...")

            for i in range(n_epochs):
                epoch_df = df_reg.iloc[i * samples_per_epoch : (i + 1) * samples_per_epoch]
                if epoch_df["accel_z"].isna().any():
                    continue
                hs_val = calculate_hs(epoch_df["accel_z"].values, fs=fs)
                results.append(
                    {
                        "epoch_start": epoch_df.index[0],
                        "epoch_end": epoch_df.index[-1],
                        "Hs_m": hs_val,
                    }
                )
                progress.progress((i + 1) / n_epochs, text=f"epoch {i+1}/{n_epochs}")

            progress.empty()

        results_df = pd.DataFrame(results)

        if results_df.empty:
            st.error("❌ ข้อมูลไม่สมบูรณ์ ไม่สามารถคำนวณได้")
        else:
            st.success(f"✅ คำนวณสำเร็จ! พบข้อมูลทั้งหมด {len(results_df)} รอบ")

            # ── Metrics ──
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Hs เฉลี่ย", f"{results_df['Hs_m'].mean():.4f} ม.")
            col2.metric("Hs สูงสุด", f"{results_df['Hs_m'].max():.4f} ม.")
            col3.metric("Hs ต่ำสุด", f"{results_df['Hs_m'].min():.4f} ม.")
            col4.metric("จำนวน epoch", f"{len(results_df)} รอบ")

            st.divider()

            # ── Main plot: Hs over time ──
            st.subheader("📈 Significant Wave Height (Hs) ตามเวลา")
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(
                results_df["epoch_start"],
                results_df["Hs_m"],
                marker="o",
                markersize=4,
                linewidth=1.2,
                color="#1f77b4",
                label="Hs (m)",
            )
            ax.axhline(results_df["Hs_m"].mean(), color="orange", linestyle="--", linewidth=1, label="Hs เฉลี่ย")
            ax.set_ylabel("Wave Height (meters)")
            ax.set_xlabel("เวลา")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            plt.xticks(rotation=30, ha="right")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)

            # ── Optional: displacement sample ──
            if show_raw and len(df_reg) >= samples_per_epoch:
                st.subheader("🔬 ตัวอย่าง Displacement (epoch แรก)")
                sample = df_reg.iloc[:samples_per_epoch]["accel_z"].values
                Az = sample - np.mean(sample)
                V = np.cumsum(Az / fs); V -= np.mean(V)
                S = np.cumsum(V / fs); S -= np.mean(S)
                t = np.arange(len(S)) / fs

                fig2, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
                axes[0].plot(t, Az, color="#e74c3c", linewidth=0.8)
                axes[0].set_ylabel("Acceleration (m/s²)")
                axes[0].set_title("Acceleration (detrended)")
                axes[1].plot(t, V, color="#2ecc71", linewidth=0.8)
                axes[1].set_ylabel("Velocity (m/s)")
                axes[1].set_title("Velocity (integrated)")
                axes[2].plot(t, S, color="#3498db", linewidth=0.8)
                axes[2].set_ylabel("Displacement (m)")
                axes[2].set_xlabel("เวลา (วินาที)")
                axes[2].set_title(f"Displacement  |  Hs = 4×σ = {4*np.std(S):.4f} ม.")
                for a in axes:
                    a.grid(True, alpha=0.3)
                fig2.tight_layout()
                st.pyplot(fig2)

            # ── Distribution ──
            st.subheader("📊 การกระจายตัวของ Hs")
            fig3, ax3 = plt.subplots(figsize=(7, 3))
            ax3.hist(results_df["Hs_m"], bins=30, color="#1f77b4", edgecolor="white", alpha=0.85)
            ax3.axvline(results_df["Hs_m"].mean(), color="orange", linestyle="--", label="ค่าเฉลี่ย")
            ax3.set_xlabel("Hs (m)")
            ax3.set_ylabel("จำนวน epoch")
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            fig3.tight_layout()
            st.pyplot(fig3)

            # ── Download ──
            st.divider()
            st.download_button(
                "⬇️ ดาวน์โหลดไฟล์ผลลัพธ์ (CSV)",
                data=results_df.to_csv(index=False).encode("utf-8"),
                file_name="hs_output.csv",
                mime="text/csv",
            )
else:
    st.info("👆 กรุณาอัปโหลดไฟล์ CSV เพื่อเริ่มต้น")

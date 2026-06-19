import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Wave Height Calculator", page_icon="🌊")

def calculate_hs(az_array, fs=5.0):
    W = 1.0
    Az1 = az_array - np.mean(az_array)
    Az2 = Az1 - np.mean(Az1)
    Az_cal = (Az2 - np.mean(Az2)) * W
    dV = Az_cal * (1.0 / fs)
    dV1 = dV - np.mean(dV)
    dV2 = dV1 - np.mean(dV1)
    V = np.cumsum(dV2 - np.mean(dV2)) 
    dS = V * (1.0 / fs)
    dS1 = dS - np.mean(dS)
    dS2 = dS1 - np.mean(dS1)
    S = np.cumsum(dS2 - np.mean(dS2))
    return 1.0 * S[-1]

st.title("🌊 ระบบคำนวณ Wave Height (Hs)")
st.write("คำนวณด้วยวิธี Rectangular Integration Method")

uploaded_files = st.file_uploader("📂 อัปโหลดไฟล์ CSV (เลือกได้หลายไฟล์)", type=['csv'], accept_multiple_files=True)

if uploaded_files:
    if st.button("🚀 เริ่มประมวลผล", type="primary"):
        with st.spinner("กำลังคำนวณ..."):
            df_list = [pd.read_csv(f) for f in uploaded_files]
            full_df = pd.concat(df_list, ignore_index=True)
            full_df['timestamp'] = pd.to_datetime(full_df['timestamp'], utc=True)
            full_df = full_df.sort_values('timestamp').drop_duplicates(subset=['timestamp']).set_index('timestamp')
            
            reg_index = pd.date_range(start=full_df.index.min().ceil('200ms'), end=full_df.index.max().floor('200ms'), freq='200ms')
            df_combined = full_df.reindex(full_df.index.union(reg_index).sort_values())
            df_combined['accel_z'] = df_combined['accel_z'].interpolate(method='time')
            df_reg = df_combined.loc[reg_index].copy()
            
            results = []
            for i in range(len(df_reg) // 300):
                epoch_df = df_reg.iloc[i * 300 : (i + 1) * 300]
                if epoch_df['accel_z'].isna().any(): continue
                hs_val = calculate_hs(epoch_df['accel_z'].values, fs=5.0)
                results.append({'epoch_start': epoch_df.index[0], 'epoch_end': epoch_df.index[-1], 'Hs_m': hs_val})
                
            results_df = pd.DataFrame(results)
            
            if not results_df.empty:
                st.success(f"✅ คำนวณสำเร็จ! พบข้อมูลทั้งหมด {len(results_df)} รอบ")
                col1, col2 = st.columns(2)
                col1.metric("Hs เฉลี่ย", f"{results_df['Hs_m'].mean():.4f} ม.")
                col2.metric("Hs สูงสุด", f"{results_df['Hs_m'].max():.4f} ม.")
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(results_df['epoch_start'], results_df['Hs_m'], marker='o', color='#1f77b4')
                ax.set_ylabel('Wave Height (meters)')
                ax.grid(True)
                st.pyplot(fig)
                
                st.download_button("⬇️ ดาวน์โหลดไฟล์ผลลัพธ์ (CSV)", data=results_df.to_csv(index=False).encode('utf-8'), file_name='hs_output.csv', mime='text/csv')
            else:
                st.error("ข้อมูลไม่สมบูรณ์ ไม่สามารถคำนวณได้")

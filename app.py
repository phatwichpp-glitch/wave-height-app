import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
import io

def calculate_hs(az_array, fs=5.0):
    W, AQWA_Factor = 1.0, 1.0
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
    return AQWA_Factor * S[-1]

print("🌊 ระบบคำนวณ Wave Height (Hs) - Rectangular Integration Method")
print("📥 กรุณากดปุ่ม Choose Files ด้านล่าง เพื่ออัปโหลดไฟล์ CSV (เลือกพร้อมกันได้หลายไฟล์)")

# 1. ให้ผู้ใช้อัปโหลดไฟล์ผ่านหน้าเว็บ
uploaded = files.upload()

if uploaded:
    print("\nกำลังประมวลผลข้อมูล... ⏳")
    df_list = []
    for filename in uploaded.keys():
        df = pd.read_csv(io.BytesIO(uploaded[filename]))
        df_list.append(df)
        
    full_df = pd.concat(df_list, ignore_index=True)
    full_df['timestamp'] = pd.to_datetime(full_df['timestamp'], utc=True)
    full_df = full_df.sort_values('timestamp').drop_duplicates(subset=['timestamp']).set_index('timestamp')
    
    # 2. Resample 5Hz
    start_time, end_time = full_df.index.min().ceil('200ms'), full_df.index.max().floor('200ms')
    reg_index = pd.date_range(start=start_time, end=end_time, freq='200ms')
    combined_index = full_df.index.union(reg_index).sort_values()
    df_combined = full_df.reindex(combined_index)
    df_combined['accel_z'] = df_combined['accel_z'].interpolate(method='time')
    df_reg = df_combined.loc[reg_index].copy()
    
    # 3. คำนวณ Epoch ละ 300 samples
    samples_per_epoch = 300
    n_epochs = len(df_reg) // samples_per_epoch
    results = []
    
    for i in range(n_epochs):
        epoch_df = df_reg.iloc[i * samples_per_epoch : (i + 1) * samples_per_epoch]
        if epoch_df['accel_z'].isna().any(): continue
        hs_val = calculate_hs(epoch_df['accel_z'].values, fs=5.0)
        results.append({
            'epoch_start': epoch_df.index[0],
            'epoch_end': epoch_df.index[-1],
            'Hs_m': hs_val
        })
        
    results_df = pd.DataFrame(results)
    
    # 4. แสดงผลลัพธ์
    if not results_df.empty:
        print("\n✅ คำนวณเสร็จสิ้น! สรุปผลลัพธ์:")
        print(f"- จำนวนรอบที่คำนวณได้: {len(results_df)} Epochs")
        print(f"- Hs เฉลี่ย: {results_df['Hs_m'].mean():.4f} m")
        print(f"- Hs สูงสุด: {results_df['Hs_m'].max():.4f} m\n")
        
        # วาดกราฟ
        plt.figure(figsize=(10, 4))
        plt.plot(results_df['epoch_start'], results_df['Hs_m'], marker='o', color='tab:blue')
        plt.title('Wave Height (Hs) Over Time')
        plt.xlabel('Time (UTC)')
        plt.ylabel('Hs (meters)')
        plt.grid(True)
        plt.show()
        
        # เซฟไฟล์ให้ดาวน์โหลด
        results_df.to_csv('hs_output.csv', index=False)
        print("⬇️ กำลังดาวน์โหลดไฟล์ hs_output.csv อัตโนมัติ...")
        files.download('hs_output.csv')
    else:
        print("❌ ไม่สามารถคำนวณได้ ข้อมูลอาจไม่สมบูรณ์")
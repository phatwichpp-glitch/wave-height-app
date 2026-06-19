"""
pages/1_upload.py
Data Upload & Preview page.
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import zscore

st.title("📁 อัปโหลดและตรวจสอบข้อมูล")

# ─── Expected columns ─────────────────────────────────────────────────────────
REQUIRED_COLS = ["timestamp", "accel_x", "accel_y", "accel_z",
                  "gyro_x", "gyro_y", "gyro_z"]
OPTIONAL_COLS = ["mag_x", "mag_y", "mag_z",
                  "latitude", "longitude", "pressure", "temperature"]
ALL_EXPECTED = REQUIRED_COLS + OPTIONAL_COLS


def detect_delimiter(sample: str) -> str:
    """Auto-detect CSV delimiter."""
    comma_count = sample.count(",")
    semi_count = sample.count(";")
    return ";" if semi_count > comma_count else ","


def parse_timestamp(series: pd.Series) -> pd.Series:
    """Try ISO8601 parse, then Unix epoch."""
    try:
        parsed = pd.to_datetime(series, utc=True)
        return parsed
    except Exception:
        pass
    try:
        parsed = pd.to_datetime(series.astype(float), unit="s", utc=True)
        return parsed
    except Exception:
        pass
    return pd.to_datetime(series, infer_datetime_format=True, utc=True)


@st.cache_data(show_spinner=False)
def load_and_merge_csvs(files_bytes_list: list) -> pd.DataFrame:
    """Load, merge, sort, and deduplicate multiple CSV files."""
    dfs = []
    for name, content in files_bytes_list:
        try:
            sample = content[:2048].decode("utf-8", errors="replace")
            delim = detect_delimiter(sample)
            df = pd.read_csv(
                io.BytesIO(content),
                delimiter=delim,
                encoding="utf-8",
                on_bad_lines="skip",
            )
            df.attrs["source_file"] = name
            dfs.append(df)
        except Exception as e:
            st.error(f"ไม่สามารถอ่านไฟล์ '{name}': {e}")
    if not dfs:
        return pd.DataFrame()
    merged = pd.concat(dfs, ignore_index=True)
    return merged


def auto_map_columns(df_cols: list) -> dict:
    """
    Try to auto-map detected column names to expected field names.
    Returns a dict {expected_name: detected_name or None}.
    """
    mapping = {}
    lower_cols = {c.lower().strip(): c for c in df_cols}
    aliases = {
        "timestamp":    ["timestamp", "time", "datetime", "t", "date"],
        "accel_x":      ["accel_x", "ax", "acc_x", "acceleration_x", "a_x"],
        "accel_y":      ["accel_y", "ay", "acc_y", "acceleration_y", "a_y"],
        "accel_z":      ["accel_z", "az", "acc_z", "acceleration_z", "a_z"],
        "gyro_x":       ["gyro_x", "gx", "gyro_x", "angular_x"],
        "gyro_y":       ["gyro_y", "gy", "gyro_y", "angular_y"],
        "gyro_z":       ["gyro_z", "gz", "gyro_z", "angular_z"],
        "mag_x":        ["mag_x", "mx", "mag_x", "magnetometer_x"],
        "mag_y":        ["mag_y", "my", "mag_y", "magnetometer_y"],
        "mag_z":        ["mag_z", "mz", "mag_z", "magnetometer_z"],
        "latitude":     ["latitude", "lat"],
        "longitude":    ["longitude", "lon", "lng"],
        "pressure":     ["pressure", "press", "baro"],
        "temperature":  ["temperature", "temp", "t_celsius"],
    }
    for expected, alias_list in aliases.items():
        found = None
        for alias in alias_list:
            if alias in lower_cols:
                found = lower_cols[alias]
                break
        mapping[expected] = found
    return mapping


def gap_detection(timestamps: pd.Series, median_dt: float) -> int:
    """Count gaps larger than 2× median_dt."""
    diffs = timestamps.diff().dt.total_seconds().dropna()
    gaps = (diffs > 2 * median_dt).sum()
    return int(gaps)


# ─── File Uploader ────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "อัปโหลดไฟล์ CSV จาก IMU Buoy (รองรับหลายไฟล์)",
    type=["csv"],
    accept_multiple_files=True,
    help="รองรับ delimiter , หรือ ; และ timestamp แบบ ISO8601 หรือ Unix epoch"
)

if not uploaded_files:
    st.info("📂 กรุณาอัปโหลดไฟล์ CSV ของ IMU Buoy อย่างน้อย 1 ไฟล์")
    st.stop()

# Load files
with st.spinner("กำลังโหลดและรวมไฟล์..."):
    files_bytes = [(f.name, f.read()) for f in uploaded_files]
    raw_df = load_and_merge_csvs(files_bytes)

if raw_df.empty:
    st.error("ไม่พบข้อมูลในไฟล์ที่อัปโหลด")
    st.stop()

st.success(f"✅ โหลดสำเร็จ: {len(raw_df):,} แถว × {len(raw_df.columns)} คอลัมน์")

# ─── Column Mapping UI ────────────────────────────────────────────────────────
st.markdown("### 🗂️ การจับคู่คอลัมน์")
auto_map = auto_map_columns(raw_df.columns.tolist())

col_options = ["(ไม่มี)"] + raw_df.columns.tolist()

with st.expander("ปรับแต่งการจับคู่คอลัมน์", expanded=any(v is None for k, v in auto_map.items() if k in REQUIRED_COLS)):
    final_map = {}
    cols_ui = st.columns(3)
    for i, expected in enumerate(ALL_EXPECTED):
        auto_val = auto_map.get(expected)
        default_idx = col_options.index(auto_val) if auto_val in col_options else 0
        selected = cols_ui[i % 3].selectbox(
            f"{expected}",
            options=col_options,
            index=default_idx,
            key=f"col_map_{expected}"
        )
        final_map[expected] = None if selected == "(ไม่มี)" else selected

# ─── Build mapped DataFrame ───────────────────────────────────────────────────
try:
    rename_dict = {v: k for k, v in final_map.items() if v is not None and v != k}
    df = raw_df.rename(columns=rename_dict)

    # Validate required columns
    missing_req = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_req:
        st.error(f"คอลัมน์ที่จำเป็นไม่ครบ: {missing_req}")
        st.stop()

    # Parse timestamp
    df["timestamp"] = parse_timestamp(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการประมวลผลข้อมูล: {e}")
    st.stop()

# ─── Resample to uniform fs ───────────────────────────────────────────────────
fs = st.session_state.get("fs", 5.0)
epoch_sec = st.session_state.get("epoch_sec", 60)

try:
    df = df.set_index("timestamp")
    period_us = int(1e6 / fs)  # microseconds
    df_resampled = df.resample(f"{period_us}us").interpolate(method="time")
    df_resampled = df_resampled.reset_index()
    df_resampled = df_resampled.rename(columns={"index": "timestamp"})
except Exception as e:
    st.warning(f"การ Resample ล้มเหลว: {e} — ใช้ข้อมูลดิบแทน")
    df_resampled = df.reset_index()

# ─── Data Quality Report ──────────────────────────────────────────────────────
st.markdown("### 📋 รายงานคุณภาพข้อมูล")

if "timestamp" in df_resampled.columns:
    t_sorted = df_resampled["timestamp"].dropna().sort_values()
    if len(t_sorted) > 1:
        diffs_s = t_sorted.diff().dt.total_seconds().dropna()
        median_dt = float(diffs_s.median())
        detected_fs = round(1.0 / median_dt, 3) if median_dt > 0 else 0.0
        t_start = t_sorted.iloc[0]
        t_end = t_sorted.iloc[-1]
        duration_s = (t_end - t_start).total_seconds()
        n_gaps = gap_detection(t_sorted, median_dt)
    else:
        detected_fs = fs
        t_start = t_end = None
        duration_s = 0.0
        n_gaps = 0
else:
    detected_fs = fs
    t_start = t_end = None
    duration_s = 0.0
    n_gaps = 0

qc1, qc2, qc3, qc4 = st.columns(4)
qc1.metric("จำนวนตัวอย่าง", f"{len(df_resampled):,}")
qc2.metric("ช่วงเวลา", f"{duration_s/60:.1f} นาที" if duration_s else "-")
qc3.metric("fs ที่ตรวจพบ", f"{detected_fs} Hz")
qc4.metric("ช่องว่างข้อมูล (gaps)", str(n_gaps))

# Missing % per column
numeric_cols = [c for c in REQUIRED_COLS + OPTIONAL_COLS if c in df_resampled.columns]
miss_pct = {c: round(100 * df_resampled[c].isna().mean(), 2) for c in numeric_cols}
miss_df = pd.DataFrame.from_dict(miss_pct, orient="index", columns=["% ข้อมูลหาย"])
miss_df.index.name = "คอลัมน์"

# Spike detection
spike_counts = {}
for c in ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]:
    if c in df_resampled.columns:
        series = df_resampled[c].dropna().values
        if len(series) > 3:
            with np.errstate(invalid="ignore"):
                z = np.abs((series - np.mean(series)) / (np.std(series) + 1e-10))
            spike_counts[c] = int((z > 5).sum())

if spike_counts:
    spike_df = pd.DataFrame.from_dict(spike_counts, orient="index",
                                       columns=["จำนวน Spike (|z|>5)"])
    miss_df = miss_df.join(spike_df, how="left")
    miss_df["จำนวน Spike (|z|>5)"] = miss_df["จำนวน Spike (|z|>5)"].fillna(0).astype(int)

st.dataframe(miss_df, use_container_width=True)

# ─── Raw Signal Preview ────────────────────────────────────────────────────────
st.markdown("### 📉 ตัวอย่างสัญญาณดิบ")

plot_cols = [c for c in ["accel_z", "gyro_z", "pressure"] if c in df_resampled.columns]
if not plot_cols:
    plot_cols = [c for c in ["accel_x", "accel_y"] if c in df_resampled.columns]

max_preview = int(fs * 300)  # up to 5 minutes for preview
df_preview = df_resampled.iloc[:max_preview]

if plot_cols and "timestamp" in df_preview.columns:
    fig, axes = plt.subplots(len(plot_cols), 1,
                              figsize=(12, 2.5 * len(plot_cols)),
                              facecolor="white", sharex=True)
    if len(plot_cols) == 1:
        axes = [axes]

    labels_th = {
        "accel_z": "ความเร่งแนวดิ่ง az (m/s²)",
        "gyro_z":  "ไจโรสโคป gz (°/s)",
        "pressure": "ความดัน (hPa)",
        "accel_x": "ความเร่ง ax (m/s²)",
        "accel_y": "ความเร่ง ay (m/s²)",
    }
    colors_ = ["steelblue", "coral", "seagreen", "mediumpurple"]

    for idx, (col, ax) in enumerate(zip(plot_cols, axes)):
        ax.set_facecolor("white")
        t_vals = df_preview["timestamp"]
        y_vals = df_preview[col]
        ax.plot(t_vals, y_vals, color=colors_[idx % len(colors_)], linewidth=0.7)
        ax.set_ylabel(labels_th.get(col, col), fontsize=9)
        ax.grid(True, alpha=0.3)
        # Mark spikes
        if col in spike_counts and spike_counts[col] > 0:
            series_vals = df_preview[col].values
            mean_v = np.nanmean(series_vals)
            std_v = np.nanstd(series_vals)
            if std_v > 0:
                z_vals = np.abs((series_vals - mean_v) / std_v)
                spike_mask = z_vals > 5
                ax.scatter(t_vals[spike_mask], y_vals[spike_mask],
                           color="red", s=10, zorder=5, label="Spike")
                ax.legend(fontsize=7)

    axes[-1].set_xlabel("เวลา")
    fig.suptitle("สัญญาณดิบจาก IMU Buoy", fontsize=11, y=1.01)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
else:
    st.warning("ไม่พบคอลัมน์ accel_z / gyro_z / pressure สำหรับการแสดงผล")

# ─── Preview Table ────────────────────────────────────────────────────────────
st.markdown("### 🔍 ตัวอย่างข้อมูล (50 แถวแรก)")
preview_cols = ["timestamp"] + [c for c in ALL_EXPECTED if c in df_resampled.columns and c != "timestamp"]
st.dataframe(df_resampled[preview_cols].head(50), use_container_width=True)

# ─── Store in session state ───────────────────────────────────────────────────
st.session_state["df"] = df_resampled
st.session_state["detected_fs"] = detected_fs
st.session_state["t_start"] = str(t_start) if t_start else ""
st.session_state["t_end"] = str(t_end) if t_end else ""
st.session_state["n_samples_raw"] = len(df_resampled)

st.success("✅ ข้อมูลพร้อมสำหรับการประมวลผล — ไปที่หน้า ⚙️ ประมวลผลสัญญาณ")

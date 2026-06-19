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

st.title("📁 Upload & Inspect Data")

REQUIRED_COLS = ["timestamp", "accel_x", "accel_y", "accel_z",
                 "gyro_x", "gyro_y", "gyro_z"]
OPTIONAL_COLS = ["mag_x", "mag_y", "mag_z",
                 "latitude", "longitude", "pressure", "temperature"]
ALL_EXPECTED = REQUIRED_COLS + OPTIONAL_COLS


def detect_delimiter(sample: str) -> str:
    return ";" if sample.count(";") > sample.count(",") else ","


def parse_timestamp(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, utc=True)
    except Exception:
        pass
    try:
        return pd.to_datetime(series.astype(float), unit="s", utc=True)
    except Exception:
        pass
    return pd.to_datetime(series, infer_datetime_format=True, utc=True)


@st.cache_data(show_spinner=False)
def load_and_merge_csvs(files_bytes_list: list) -> pd.DataFrame:
    dfs = []
    for name, content in files_bytes_list:
        try:
            sample = content[:2048].decode("utf-8", errors="replace")
            delim = detect_delimiter(sample)
            df = pd.read_csv(io.BytesIO(content), delimiter=delim,
                             encoding="utf-8", on_bad_lines="skip")
            dfs.append(df)
        except Exception as e:
            st.error(f"Cannot read file '{name}': {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def auto_map_columns(df_cols: list) -> dict:
    lower_cols = {c.lower().strip(): c for c in df_cols}
    aliases = {
        "timestamp":   ["timestamp", "time", "datetime", "t", "date"],
        "accel_x":     ["accel_x", "ax", "acc_x", "acceleration_x", "a_x"],
        "accel_y":     ["accel_y", "ay", "acc_y", "acceleration_y", "a_y"],
        "accel_z":     ["accel_z", "az", "acc_z", "acceleration_z", "a_z"],
        "gyro_x":      ["gyro_x", "gx", "angular_x"],
        "gyro_y":      ["gyro_y", "gy", "angular_y"],
        "gyro_z":      ["gyro_z", "gz", "angular_z"],
        "mag_x":       ["mag_x", "mx", "magnetometer_x"],
        "mag_y":       ["mag_y", "my", "magnetometer_y"],
        "mag_z":       ["mag_z", "mz", "magnetometer_z"],
        "latitude":    ["latitude", "lat"],
        "longitude":   ["longitude", "lon", "lng"],
        "pressure":    ["pressure", "press", "baro"],
        "temperature": ["temperature", "temp", "t_celsius"],
    }
    mapping = {}
    for expected, alias_list in aliases.items():
        found = next((lower_cols[a] for a in alias_list if a in lower_cols), None)
        mapping[expected] = found
    return mapping


def gap_detection(timestamps: pd.Series, median_dt: float) -> int:
    diffs = timestamps.diff().dt.total_seconds().dropna()
    return int((diffs > 2 * median_dt).sum())


# ── File Uploader ─────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload CSV files from IMU Buoy (multiple files supported)",
    type=["csv"], accept_multiple_files=True,
    help="Supports , or ; delimiter and ISO8601 / Unix epoch timestamps"
)

if not uploaded_files:
    st.info("📂 Please upload at least one IMU Buoy CSV file.")
    st.stop()

with st.spinner("Loading and merging files..."):
    files_bytes = [(f.name, f.read()) for f in uploaded_files]
    raw_df = load_and_merge_csvs(files_bytes)

if raw_df.empty:
    st.error("No data found in uploaded files.")
    st.stop()

st.success(f"✅ Loaded: {len(raw_df):,} rows × {len(raw_df.columns)} columns")

# ── Column Mapping UI ─────────────────────────────────────────────────────────
st.markdown("### 🗂️ Column Mapping")
auto_map = auto_map_columns(raw_df.columns.tolist())
col_options = ["(none)"] + raw_df.columns.tolist()

with st.expander("Customize column mapping",
                 expanded=any(v is None for k, v in auto_map.items() if k in REQUIRED_COLS)):
    final_map = {}
    cols_ui = st.columns(3)
    for i, expected in enumerate(ALL_EXPECTED):
        auto_val = auto_map.get(expected)
        default_idx = col_options.index(auto_val) if auto_val in col_options else 0
        selected = cols_ui[i % 3].selectbox(
            expected, options=col_options, index=default_idx, key=f"col_map_{expected}"
        )
        final_map[expected] = None if selected == "(none)" else selected

# ── Build mapped DataFrame ────────────────────────────────────────────────────
try:
    rename_dict = {v: k for k, v in final_map.items() if v is not None and v != k}
    df = raw_df.rename(columns=rename_dict)
    missing_req = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_req:
        st.error(f"Required columns missing: {missing_req}")
        st.stop()
    df["timestamp"] = parse_timestamp(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
except Exception as e:
    st.error(f"Error processing data: {e}")
    st.stop()

# ── Resample ──────────────────────────────────────────────────────────────────
fs = st.session_state.get("fs", 5.0)
try:
    df = df.set_index("timestamp")
    period_us = int(1e6 / fs)
    df_resampled = df.resample(f"{period_us}us").interpolate(method="time").reset_index()
    df_resampled = df_resampled.rename(columns={"index": "timestamp"})
except Exception as e:
    st.warning(f"Resampling failed: {e} — using raw data.")
    df_resampled = df.reset_index()

# ── Data Quality Report ───────────────────────────────────────────────────────
st.markdown("### 📋 Data Quality Report")

if "timestamp" in df_resampled.columns:
    t_sorted = df_resampled["timestamp"].dropna().sort_values()
    if len(t_sorted) > 1:
        diffs_s = t_sorted.diff().dt.total_seconds().dropna()
        median_dt = float(diffs_s.median())
        detected_fs = round(1.0 / median_dt, 3) if median_dt > 0 else 0.0
        t_start, t_end = t_sorted.iloc[0], t_sorted.iloc[-1]
        duration_s = (t_end - t_start).total_seconds()
        n_gaps = gap_detection(t_sorted, median_dt)
    else:
        detected_fs, t_start, t_end, duration_s, n_gaps = fs, None, None, 0.0, 0
else:
    detected_fs, t_start, t_end, duration_s, n_gaps = fs, None, None, 0.0, 0

qc1, qc2, qc3, qc4 = st.columns(4)
qc1.metric("Total Samples", f"{len(df_resampled):,}")
qc2.metric("Duration", f"{duration_s/60:.1f} min" if duration_s else "-")
qc3.metric("Detected fs", f"{detected_fs} Hz")
qc4.metric("Data Gaps", str(n_gaps))

numeric_cols = [c for c in ALL_EXPECTED if c in df_resampled.columns]
miss_pct = {c: round(100 * df_resampled[c].isna().mean(), 2) for c in numeric_cols}
miss_df = pd.DataFrame.from_dict(miss_pct, orient="index", columns=["% Missing"])
miss_df.index.name = "Column"

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
                                      columns=["Spikes (|z|>5)"])
    miss_df = miss_df.join(spike_df, how="left")
    miss_df["Spikes (|z|>5)"] = miss_df["Spikes (|z|>5)"].fillna(0).astype(int)

st.dataframe(miss_df, use_container_width=True)

# ── Raw Signal Preview ────────────────────────────────────────────────────────
st.markdown("### 📉 Raw Signal Preview")

plot_cols = [c for c in ["accel_z", "gyro_z", "pressure"] if c in df_resampled.columns]
if not plot_cols:
    plot_cols = [c for c in ["accel_x", "accel_y"] if c in df_resampled.columns]

max_preview = int(fs * 300)
df_preview = df_resampled.iloc[:max_preview]

labels = {
    "accel_z": "Vertical Accel az (m/s²)",
    "gyro_z":  "Gyroscope gz (°/s)",
    "pressure": "Pressure (hPa)",
    "accel_x": "Accel ax (m/s²)",
    "accel_y": "Accel ay (m/s²)",
}
colors_ = ["steelblue", "coral", "seagreen", "mediumpurple"]

if plot_cols and "timestamp" in df_preview.columns:
    fig, axes = plt.subplots(len(plot_cols), 1,
                             figsize=(12, 2.5 * len(plot_cols)),
                             facecolor="white", sharex=True)
    if len(plot_cols) == 1:
        axes = [axes]
    for idx, (col, ax) in enumerate(zip(plot_cols, axes)):
        ax.set_facecolor("white")
        ax.plot(df_preview["timestamp"], df_preview[col],
                color=colors_[idx % len(colors_)], linewidth=0.7)
        ax.set_ylabel(labels.get(col, col), fontsize=9)
        ax.grid(True, alpha=0.3)
        if col in spike_counts and spike_counts[col] > 0:
            vals = df_preview[col].values
            mean_v, std_v = np.nanmean(vals), np.nanstd(vals)
            if std_v > 0:
                z_vals = np.abs((vals - mean_v) / std_v)
                mask = z_vals > 5
                ax.scatter(df_preview["timestamp"][mask], df_preview[col][mask],
                           color="red", s=10, zorder=5, label="Spike")
                ax.legend(fontsize=7)
    axes[-1].set_xlabel("Time")
    fig.suptitle("Raw IMU Buoy Signals", fontsize=11, y=1.01)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ── Preview Table ─────────────────────────────────────────────────────────────
st.markdown("### 🔍 Data Preview (first 50 rows)")
preview_cols = ["timestamp"] + [c for c in ALL_EXPECTED
                                if c in df_resampled.columns and c != "timestamp"]
st.dataframe(df_resampled[preview_cols].head(50), use_container_width=True)

# ── Store in session state ────────────────────────────────────────────────────
st.session_state["df"] = df_resampled
st.session_state["detected_fs"] = detected_fs
st.session_state["t_start"] = str(t_start) if t_start else ""
st.session_state["t_end"] = str(t_end) if t_end else ""
st.session_state["n_samples_raw"] = len(df_resampled)

st.success("✅ Data ready — go to ⚙️ Signal Processing")

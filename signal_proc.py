"""
core/signal_proc.py
Signal processing pipeline for IMU buoy wave height analysis.
"""

import numpy as np
from scipy.signal import butter, filtfilt, welch, spectrogram


def tilt_correct(ax: np.ndarray, ay: np.ndarray, az: np.ndarray) -> np.ndarray:
    """
    Step 1: Complementary-filter tilt correction.
    Returns the tilt-corrected vertical acceleration az_corrected.
    """
    pitch = np.arctan2(ax, np.sqrt(ay ** 2 + az ** 2))
    roll = np.arctan2(ay, np.sqrt(ax ** 2 + az ** 2))
    cos_pitch = np.cos(pitch)
    cos_roll = np.cos(roll)
    # Prevent division by near-zero
    cos_pitch = np.where(np.abs(cos_pitch) < 1e-6, 1e-6, cos_pitch)
    cos_roll = np.where(np.abs(cos_roll) < 1e-6, 1e-6, cos_roll)
    az_corrected = az / cos_pitch / cos_roll
    return az_corrected


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    low = np.clip(low, 1e-6, 1 - 1e-6)
    high = np.clip(high, 1e-6, 1 - 1e-6)
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_highpass(cutoff: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    normal_cutoff = np.clip(normal_cutoff, 1e-6, 1 - 1e-6)
    b, a = butter(order, normal_cutoff, btype="high")
    return b, a


def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float,
                    fs: float, order: int = 4) -> np.ndarray:
    """
    Step 2: 4th-order Butterworth bandpass filter.
    """
    if len(data) < 3 * order * 2:
        return data.copy()
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, data)


def highpass_filter(data: np.ndarray, cutoff: float, fs: float,
                    order: int = 4) -> np.ndarray:
    """High-pass Butterworth filter."""
    if len(data) < 3 * order * 2:
        return data.copy()
    b, a = butter_highpass(cutoff, fs, order)
    return filtfilt(b, a, data)


def double_integrate(az_filtered: np.ndarray, fs: float) -> np.ndarray:
    """
    Step 3: Double integration of filtered acceleration → displacement (m).
    Removes mean velocity and mean displacement (DC drift removal).
    """
    velocity = np.cumsum(az_filtered) / fs
    velocity -= np.mean(velocity)
    displacement = np.cumsum(velocity) / fs
    displacement -= np.mean(displacement)
    return displacement


def process_epoch(ax: np.ndarray, ay: np.ndarray, az: np.ndarray,
                  fs: float, lowcut: float = 0.05, highcut: float = 2.0
                  ) -> np.ndarray:
    """
    Full pipeline for one epoch:
      1) Tilt correct
      2) Bandpass filter
      3) Double integrate
      4) High-pass filter displacement

    Returns displacement array (m).
    """
    az_corr = tilt_correct(ax, ay, az)
    az_filt = bandpass_filter(az_corr, lowcut, highcut, fs)
    disp = double_integrate(az_filt, fs)
    disp_filt = highpass_filter(disp, lowcut, fs)
    return disp_filt


def detect_spikes(series: np.ndarray, threshold: float = 5.0) -> np.ndarray:
    """Return boolean mask of spikes (|z-score| > threshold)."""
    mean = np.nanmean(series)
    std = np.nanstd(series)
    if std == 0:
        return np.zeros(len(series), dtype=bool)
    z = np.abs((series - mean) / std)
    return z > threshold


def compute_spectrogram(displacement: np.ndarray, fs: float,
                         nperseg: int = 256):
    """
    Compute short-time Fourier transform spectrogram.
    Returns (t, f, Sxx) arrays.
    """
    f, t, Sxx = spectrogram(displacement, fs=fs, nperseg=nperseg,
                             noverlap=nperseg // 2)
    return t, f, Sxx

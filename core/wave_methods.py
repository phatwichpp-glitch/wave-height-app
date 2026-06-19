"""
core/wave_methods.py
Significant wave height (Hs) calculation using three methods.
"""

import numpy as np
from scipy.signal import welch
from scipy.integrate import trapezoid
from typing import Dict, Any, Optional


def hs_rectangular(disp: np.ndarray) -> float:
    """
    Method A: Rectangular integration.
    Hs = 4 * std(displacement)
    """
    return 4.0 * float(np.std(disp))


def hs_spectral(disp: np.ndarray, fs: float,
                lowcut: float = 0.05, highcut: float = 2.0
                ) -> Dict[str, Any]:
    """
    Method B: Spectral analysis using Welch method.
    Returns dict with Hs_spec, Tp, Tm02, f_arr, Pxx_arr.
    """
    N = len(disp)
    nperseg = min(256, max(16, N // 4))
    f, Pxx = welch(disp, fs=fs, nperseg=nperseg)

    mask = (f >= lowcut) & (f <= highcut)
    if not np.any(mask):
        return {
            "Hs_spec": np.nan, "Tp": np.nan, "Tm02": np.nan,
            "f_arr": f, "Pxx_arr": Pxx, "fp": np.nan
        }

    f_m = f[mask]
    Pxx_m = Pxx[mask]

    m0 = float(trapezoid(Pxx_m, f_m))
    m2 = float(trapezoid(Pxx_m * f_m ** 2, f_m))

    Hs_spec = 4.0 * np.sqrt(max(m0, 0.0))
    fp_idx = np.argmax(Pxx_m)
    fp = f_m[fp_idx]
    Tp = 1.0 / fp if fp > 0 else np.nan
    Tm02 = np.sqrt(m0 / m2) if m2 > 0 else np.nan

    return {
        "Hs_spec": float(Hs_spec),
        "Tp": float(Tp),
        "Tm02": float(Tm02),
        "f_arr": f_m,
        "Pxx_arr": Pxx_m,
        "fp": float(fp),
    }


def hs_zero_crossing(disp: np.ndarray, fs: float) -> Dict[str, Any]:
    """
    Method C: Zero-crossing analysis.
    Extracts individual waves, sorts by height,
    Hs = mean of top 1/3 wave heights.
    Returns dict with Hs_zc, H_max, T_mean, wave_count.
    """
    s = disp - np.mean(disp)

    # Find zero-crossing indices (sign changes)
    signs = np.sign(s)
    signs[signs == 0] = 1  # treat exact zero as positive
    crossings = np.where(np.diff(signs))[0]

    if len(crossings) < 4:
        return {
            "Hs_zc": np.nan,
            "H_max": np.nan,
            "T_mean": np.nan,
            "wave_count": 0,
        }

    # Pair consecutive crossings into full waves (two half-cycles each)
    wave_heights = []
    wave_periods = []

    # Use up-crossing method: start at up-crossing (negative→positive)
    up_crossings = []
    for i in range(len(crossings)):
        idx = crossings[i]
        if signs[idx] < 0 and signs[idx + 1] > 0:  # up-crossing
            up_crossings.append(idx)

    for k in range(len(up_crossings) - 1):
        i_start = up_crossings[k]
        i_end = up_crossings[k + 1]
        wave_seg = s[i_start:i_end + 1]
        if len(wave_seg) < 2:
            continue
        H = float(np.max(wave_seg) - np.min(wave_seg))
        T = (i_end - i_start) / fs
        wave_heights.append(H)
        wave_periods.append(T)

    if len(wave_heights) == 0:
        return {
            "Hs_zc": np.nan,
            "H_max": np.nan,
            "T_mean": np.nan,
            "wave_count": 0,
        }

    wave_heights = np.array(wave_heights)
    wave_periods = np.array(wave_periods)

    sorted_h = np.sort(wave_heights)[::-1]
    n_third = max(1, len(sorted_h) // 3)
    Hs_zc = float(np.mean(sorted_h[:n_third]))
    H_max = float(sorted_h[0])
    T_mean = float(np.mean(wave_periods))

    return {
        "Hs_zc": Hs_zc,
        "H_max": H_max,
        "T_mean": T_mean,
        "wave_count": len(wave_heights),
    }


def jonswap_spectrum(f: np.ndarray, Hs: float, Tp: float,
                     gamma: float = 3.3) -> np.ndarray:
    """
    JONSWAP reference spectrum.
    S(f) = alpha * g^2 * (2pi)^{-4} * f^{-5} * exp(-5/4*(fp/f)^4)
           * gamma^exp(-0.5*((f-fp)/(sigma*fp))^2)
    """
    if Tp <= 0 or Hs <= 0:
        return np.zeros_like(f)

    fp = 1.0 / Tp
    g = 9.81

    # Determine alpha from Hs (approximate)
    # alpha chosen so that integral gives (Hs/4)^2
    # Use simplified approach: scale after computing shape
    sigma = np.where(f <= fp, 0.07, 0.09)
    r = np.exp(-0.5 * ((f - fp) / (sigma * fp)) ** 2)

    # PM spectrum base
    with np.errstate(divide="ignore", invalid="ignore"):
        S_pm = (5.0 / 16.0) * (Hs ** 2) * (fp ** 4) * (f ** (-5)) * \
               np.exp(-1.25 * (fp / f) ** 4)
        S_pm = np.where(np.isfinite(S_pm), S_pm, 0.0)

    S_jonswap = S_pm * (gamma ** r)

    # Normalize to match requested Hs
    from scipy.integrate import trapezoid as trapz
    mask = f > 0
    m0 = trapz(S_jonswap[mask], f[mask])
    if m0 > 0:
        scale = (Hs / 4.0) ** 2 / m0
        S_jonswap *= scale

    return S_jonswap


def compute_all_methods(ax: np.ndarray, ay: np.ndarray, az: np.ndarray,
                         disp: np.ndarray, fs: float,
                         lowcut: float = 0.05, highcut: float = 2.0,
                         epoch_start: Optional[float] = None,
                         epoch_end: Optional[float] = None) -> Dict[str, Any]:
    """
    Compute all three Hs methods for a single epoch.
    Returns a unified result dict.
    """
    Hs_rect = hs_rectangular(disp)
    spec = hs_spectral(disp, fs, lowcut, highcut)
    zc = hs_zero_crossing(disp, fs)

    result = {
        "epoch_start": epoch_start,
        "epoch_end": epoch_end,
        "Hs_rect": Hs_rect,
        "Hs_spec": spec["Hs_spec"],
        "Hs_zc": zc["Hs_zc"],
        "Tp": spec["Tp"],
        "Tm02": spec["Tm02"],
        "fp": spec.get("fp", np.nan),
        "H_max": zc["H_max"],
        "T_mean": zc["T_mean"],
        "wave_count": zc["wave_count"],
        "f_arr": spec["f_arr"],
        "Pxx_arr": spec["Pxx_arr"],
    }
    return result

"""
app.py — Entry point for the Wave Height Analysis application.
Streamlit multi-page app using st.navigation.
"""

import streamlit as st

st.set_page_config(
    page_title="IMU Buoy Wave Height Analysis System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def setup_thai_font() -> str:
    """
    Locate an installed Thai-capable font and register it globally with
    matplotlib so every page (each of which does its own `import
    matplotlib.pyplot as plt`) picks up the same font, since rcParams are
    process-global, not per-module.

    Runtime apt-get is NOT used here: Streamlit Community Cloud has no sudo
    at runtime, so packages must come from packages.txt at build time
    instead. This function only *discovers* whatever was installed there.

    Returns the resolved font family name actually applied (for diagnostics).
    """
    import glob
    import os
    import matplotlib
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    # Candidate paths, in priority order. Covers fonts-noto-core
    # (NotoSansThai) and fonts-thai-tlwg (Garuda/Loma/Sarabun/etc.) as
    # declared in packages.txt, plus a couple of common alternate locations.
    candidates = [
        "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
        "/usr/share/fonts/truetype/tlwg/Garuda.ttf",
        "/usr/share/fonts/truetype/tlwg/Loma.ttf",
        "/usr/share/fonts/truetype/tlwg/Sarabun.ttf",
        "/usr/share/fonts/truetype/thai-tlwg/Garuda.ttf",
    ]
    found_path = next((p for p in candidates if os.path.exists(p)), None)

    # If none of the known paths hit, do a broader filesystem search as a
    # fallback (covers slightly different package layouts/versions).
    if not found_path:
        glob_patterns = [
            "/usr/share/fonts/**/NotoSansThai*.ttf",
            "/usr/share/fonts/**/Garuda*.ttf",
            "/usr/share/fonts/**/Sarabun*.ttf",
            "/usr/share/fonts/**/Loma*.ttf",
            "/usr/share/fonts/**/*Thai*.ttf",
        ]
        for pattern in glob_patterns:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                found_path = matches[0]
                break

    if found_path:
        fm.fontManager.addfont(found_path)
        prop = fm.FontProperties(fname=found_path)
        family_name = prop.get_name()
        # Apply globally so every page's `plt` (same process, same
        # matplotlib rcParams singleton) renders Thai correctly.
        plt.rcParams["font.family"] = family_name
        matplotlib.rcParams["font.family"] = family_name
        plt.rcParams["axes.unicode_minus"] = False
        return family_name

    # No Thai font found at all — likely packages.txt wasn't picked up by
    # the build, or the deploy needs a reboot after editing packages.txt.
    return ""


_resolved_font = setup_thai_font()
st.session_state["_thai_font_family"] = _resolved_font
if not _resolved_font:
    st.sidebar.warning(
        "⚠️ No Thai-capable font found on this system — charts may render "
        "as boxes/garbled characters\n\n"
        "Make sure a `packages.txt` file exists at the repo root containing "
        "`fonts-thai-tlwg` and `fonts-noto-core`, then reboot the app in "
        "Streamlit Cloud (Manage app → Reboot) so the build reinstalls the fonts."
    )

# ─── Sidebar global settings ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌊 Global Settings")
    st.markdown("---")

    fs = st.number_input(
        "Sampling rate fs (Hz)",
        min_value=1.0, max_value=100.0, value=5.0, step=0.5,
        help="Data acquisition frequency"
    )
    epoch_sec = st.number_input(
        "Epoch length (seconds)",
        min_value=10, max_value=3600, value=60, step=10,
        help="Length of each analysis epoch"
    )
    hp_cutoff = st.number_input(
        "High-pass cutoff (Hz)",
        min_value=0.01, max_value=0.5, value=0.05, step=0.01,
        format="%.3f",
        help="High-pass filter cutoff frequency"
    )
    lp_cutoff = st.number_input(
        "Low-pass cutoff (Hz)",
        min_value=0.5, max_value=5.0, value=2.0, step=0.1,
        help="Low-pass filter cutoff frequency"
    )

    # Nyquist warning
    nyquist = fs / 2.0
    if lp_cutoff > nyquist:
        st.warning(f"⚠️ Low-pass cutoff ({lp_cutoff} Hz) > Nyquist ({nyquist} Hz)")
    if fs < 2 * lp_cutoff:
        st.warning(f"⚠️ fs is below 2× the wave frequency (Nyquist criterion)")

    st.markdown("---")
    st.caption("Built for IMU Buoy (9-axis)\nSignificant Wave Height (Hs)")

# Store settings in session_state for all pages
st.session_state["fs"] = fs
st.session_state["epoch_sec"] = int(epoch_sec)
st.session_state["hp_cutoff"] = hp_cutoff
st.session_state["lp_cutoff"] = lp_cutoff

# ─── Navigation ───────────────────────────────────────────────────────────────
pages = [
    st.Page("pages/1_upload.py",     title="📁 Data Upload",       icon="📁"),
    st.Page("pages/2_processing.py", title="⚙️ Signal Processing",      icon="⚙️"),
    st.Page("pages/3_spectrum.py",   title="📊 Spectral Analysis", icon="📊"),
    st.Page("pages/4_comparison.py", title="📈 Method Comparison",   icon="📈"),
    st.Page("pages/5_report.py",     title="📤 Export Report",         icon="📤"),
]

pg = st.navigation(pages)
pg.run()

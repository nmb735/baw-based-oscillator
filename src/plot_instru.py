# =================================================================================================================================== #
# ----------------------------------------------------------- DESCRIPTION ----------------------------------------------------------- #
# Instrumentation lab plots: filter Bode diagram, amplifier Vin/Vout, multiplexer transient.                                        #
# Simulation data from SPICE (.txt AC/transient exports); measurements from bench tab-delimited files.                               #
# Author: Nedal M. Benelmekki                                                                                                        #
# =================================================================================================================================== #



# =================================================================================================================================== #
# --------------------------------------------------------- EXTERNAL IMPORTS -------------------------------------------------------- #
import os                        # Operating system interfaces.                                                                       #
import re                        # Regular expressions.                                                                               #
import sys                       # System-specific parameters and functions.                                                          #
import numpy             as np   # Numerical computing.                                                                               #
import matplotlib.pyplot as plt  # Plotting.                                                                                          #
import skrf              as rf   # RF/Microwave engineering (Smith chart).                                                            #
# =================================================================================================================================== #



# =================================================================================================================================== #
# ----------------------------------------------------- DIRECTORY CONFIGURATION ----------------------------------------------------- #
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))   # Directory of this script (src/).                                       #
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)                 # Project root = parent of src/.                                         #
sys.path.insert(0, _PROJECT_ROOT)                            # Add project root to sys.path for absolute imports.                     #
# =================================================================================================================================== #



# =================================================================================================================================== #
# ------------------------------------------------------- STYLE CONFIGURATION ------------------------------------------------------- #
import style.style as style                                                                                                           #
from style.style import (                                                                                                             #
    TITLE_SIZE, LABEL_SIZE, LEGEND_SIZE, TICK_SIZE,                                                                  #
    FIG_WIDTH, FIG_HEIGHT, LINE_WIDTH,                                                                                                #
    apply_minor_grid,                                                                                                                 #
)                                                                                                                                     #
from src.parse_measurements import S2PParser                                                                                          #
# =================================================================================================================================== #



# =================================================================================================================================== #
# --------------------------------------------------------- COLOUR ALIASES ---------------------------------------------------------- #
_SIM_COLOR  = '#0047AB'   # Cobalt blue  — simulation / Vin                                                                          #
_MEAS_COLOR = '#C41E3A'   # Cardinal red — measurement / Vout                                                                        #
# =================================================================================================================================== #



# =================================================================================================================================== #
# --------------------------------------------------------- LOCAL PARSERS ----------------------------------------------------------- #
def _parse_spice_ac(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse a SPICE AC analysis export.

    Expected format (tab-separated, one header row):
        Freq.    V(vout)/V(vin)
        1.0e+00  (-3.668e-06dB,-7.281e-03°)

    Parameters:
        - path [str] : Absolute path to the SPICE .txt file.

    Returns:
        - freq_hz [ndarray] : Frequency in Hz.
        - gain_dB [ndarray] : Gain magnitude in dB.
    """
    freqs: list[float] = []
    gains: list[float] = []
    with open(path, encoding='latin-1') as fh:
        next(fh)                                   # skip header
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) < 2:
                continue
            try:
                freq = float(parts[0])
                m    = re.search(r'\(([^d]+)dB', parts[1])
                if m:
                    freqs.append(freq)
                    gains.append(float(m.group(1)))
            except (ValueError, AttributeError):
                continue
    return np.array(freqs), np.array(gains)


def _parse_filter_meas(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse bench filter measurement file (European comma-decimal separators).

    Expected columns (tab-separated, one header row):
        Freq [Hz]   Vin [V]   Vout [V]   G [dB]

    Parameters:
        - path [str] : Absolute path to the measurement .txt file.

    Returns:
        - freq_hz [ndarray] : Frequency in Hz.
        - gain_dB [ndarray] : Gain in dB.
    """
    freqs: list[float] = []
    gains: list[float] = []
    with open(path, encoding='utf-8') as fh:
        next(fh)
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 4:
                continue
            try:
                freqs.append(float(parts[0].replace(',', '.')))
                gains.append(float(parts[3].replace(',', '.')))
            except ValueError:
                continue
    return np.array(freqs), np.array(gains)


def _parse_amp_sim(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse a SPICE DC-sweep export for the amplifier.

    Expected columns (tab-separated, one header row):
        v1   V(vin)   V(vout)

    Parameters:
        - path [str] : Absolute path to the SPICE .txt file.

    Returns:
        - vin  [ndarray] : Input voltage V(vin) in V.
        - vout [ndarray] : Output voltage V(vout) in V.
    """
    vins:  list[float] = []
    vouts: list[float] = []
    with open(path, encoding='utf-8') as fh:
        next(fh)
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            try:
                vins.append(float(parts[1]))   # V(vin)
                vouts.append(float(parts[2]))  # V(vout)
            except ValueError:
                continue
    return np.array(vins), np.array(vouts)


def _parse_amp_meas(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse bench amplifier measurement file.

    Expected columns (tab-separated, one header row):
        Vin [V]   Vout [V]   Gain

    Parameters:
        - path [str] : Absolute path to the measurement .txt file.

    Returns:
        - vin  [ndarray] : Input voltage (V).
        - vout [ndarray] : Output voltage (V).
        - gain [ndarray] : Linear voltage gain.
    """
    vins:  list[float] = []
    vouts: list[float] = []
    gains: list[float] = []
    with open(path, encoding='utf-8') as fh:
        next(fh)
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            try:
                vins.append(float(parts[0]))
                vouts.append(float(parts[1]))
                gains.append(float(parts[2]))
            except ValueError:
                continue
    return np.array(vins), np.array(vouts), np.array(gains)


def _parse_multiplexer(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse SPICE transient export for the multiplexer circuit.

    Expected columns (tab-separated, one header row):
        time   V(vin)   V(vout)

    Parameters:
        - path [str] : Absolute path to the SPICE .txt file.

    Returns:
        - time [ndarray] : Time in seconds.
        - vin  [ndarray] : Input voltage (V).
        - vout [ndarray] : Output voltage (V).
    """
    times: list[float] = []
    vins:  list[float] = []
    vouts: list[float] = []
    with open(path, encoding='utf-8') as fh:
        next(fh)
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            try:
                times.append(float(parts[0]))
                vins.append(float(parts[1]))
                vouts.append(float(parts[2]))
            except ValueError:
                continue
    return np.array(times), np.array(vins), np.array(vouts)
# =================================================================================================================================== #



# =================================================================================================================================== #
# -------------------------------------------------------- CUTOFF FREQUENCY HELPER --------------------------------------------------- #
def _find_cutoff(freq: np.ndarray, gain_dB: np.ndarray) -> float | None:
    """
    Interpolate the -3 dB cutoff frequency using log-space interpolation.

    The passband reference is the gain at the first (lowest) frequency point.
    Scans for the first frequency where gain drops 3 dB below that reference.

    Parameters:
        - freq    [ndarray] : Frequency array in Hz (monotonically increasing).
        - gain_dB [ndarray] : Gain array in dB.

    Returns:
        - fc [float] : Cutoff frequency in Hz, or None if no crossing found.
    """
    passband = gain_dB[0]
    target   = passband - 3.0
    for i in range(len(gain_dB) - 1):
        if gain_dB[i] >= target > gain_dB[i + 1]:
            t = (target - gain_dB[i]) / (gain_dB[i + 1] - gain_dB[i])
            return 10.0 ** (np.log10(freq[i]) + t * (np.log10(freq[i + 1]) - np.log10(freq[i])))
    return None
# =================================================================================================================================== #



# =================================================================================================================================== #
# ----------------------------------------------- PLOT 1 : FILTER BODE DIAGRAM ------------------------------------------------------ #
def plot_filter_bode() -> plt.Figure:
    """
    Bode magnitude plot of the low-pass filter.

    Simulation  — blue dashed line parsed from SPICE AC analysis.
    Measurement — red solid line from bench measurements.
    Cutoff frequencies are computed from the data (log-interpolated -3 dB point).

    Returns:
        - fig [plt.Figure] : Completed figure.
    """
    freq_sim,  gain_sim  = _parse_spice_ac(os.path.join(_SCRIPT_DIR, 'filter_sim.txt'))
    freq_meas, gain_meas = _parse_filter_meas(os.path.join(_SCRIPT_DIR, 'filter_meas.txt'))

    fc_sim  = _find_cutoff(freq_sim,  gain_sim)
    fc_meas = _find_cutoff(freq_meas, gain_meas)

    sim_label  = rf'Simulation ($f_c = {fc_sim:.1f}\,\mathrm{{Hz}}$)'  if fc_sim  else r'Simulation'
    meas_label = rf'Measurement ($f_c = {fc_meas:.1f}\,\mathrm{{Hz}}$)' if fc_meas else r'Measurement'

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    ax.semilogx(freq_sim,  gain_sim,  color=_SIM_COLOR,  linestyle='--', linewidth=3.0,
                label=sim_label)
    ax.semilogx(freq_meas, gain_meas, color=_MEAS_COLOR, linestyle='-',  linewidth=3.0,
                label=meas_label)

    # -3 dB reference line
    ax.axhline(-3, color='#666666', linestyle=':', linewidth=1.0, alpha=0.8)

    # Vertical markers at each computed cutoff
    if fc_sim:
        ax.axvline(fc_sim,  color=_SIM_COLOR,  linestyle=':', linewidth=1.0, alpha=0.7)
    if fc_meas:
        ax.axvline(fc_meas, color=_MEAS_COLOR, linestyle=':', linewidth=1.0, alpha=0.7)

    ax.set_xlabel(r'Frequency [Hz]')
    ax.set_ylabel(r'Gain [dB]')
    ax.set_title(r'Low-Pass Filter Bode Diagram')
    ax.legend()
    apply_minor_grid(ax)

    return fig
# =================================================================================================================================== #



# =================================================================================================================================== #
# ----------------------------------------------- PLOT 2 : AMPLIFIER Vin vs Vout ---------------------------------------------------- #
def plot_amp_vin_vout() -> plt.Figure:
    """
    Vin vs Vout characteristic of the amplifier.

    Simulation  — blue dashed line from SPICE DC sweep (V(vin) vs V(vout)).
    Measurement — red solid line with markers; average gains shown in the legend.

    Returns:
        - fig [plt.Figure] : Completed figure.
    """
    vin_sim,  vout_sim  = _parse_amp_sim(os.path.join(_SCRIPT_DIR, 'amp_sim.txt'))
    vin_meas, vout_meas, gain_meas = _parse_amp_meas(os.path.join(_SCRIPT_DIR, 'amp_meas.txt'))

    # Average simulation gain (exclude Vin = 0 to avoid divide-by-zero)
    mask         = vin_sim > 0
    avg_gain_sim = float(np.mean(vout_sim[mask] / vin_sim[mask]))

    # Sort and deduplicate measurements by Vin
    sort_idx  = np.argsort(vin_meas)
    vin_meas  = vin_meas[sort_idx]
    vout_meas = vout_meas[sort_idx]
    gain_meas = gain_meas[sort_idx]
    _, unique = np.unique(vin_meas, return_index=True)
    vin_meas  = vin_meas[unique]
    vout_meas = vout_meas[unique]
    gain_meas = gain_meas[unique]
    avg_gain_meas = float(np.mean(gain_meas))

    # Linear fit through measurement points, evaluated over the simulation x-range
    coeffs   = np.polyfit(vin_meas, vout_meas, 1)
    vin_fit  = np.linspace(vin_sim[0], vin_sim[-1], 300)
    vout_fit = np.polyval(coeffs, vin_fit)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    ax.plot(vin_sim,  vout_sim,
            color=_SIM_COLOR,  linestyle='--', linewidth=3.0,
            label=rf'Simulation (avg gain $= {avg_gain_sim:.2f}$)')
    ax.plot(vin_fit,  vout_fit,
            color=_MEAS_COLOR, linestyle='-',  linewidth=3.0,
            label=rf'Measurement fit (avg gain $= {avg_gain_meas:.2f}$)')

    ax.set_xlabel(r'$V_{\mathrm{in}}$ [V]')
    ax.set_ylabel(r'$V_{\mathrm{out}}$ [V]')
    ax.set_title(r'Amplifier: $V_{\mathrm{in}}$ vs $V_{\mathrm{out}}$')
    ax.legend()
    apply_minor_grid(ax)

    return fig
# =================================================================================================================================== #



# =================================================================================================================================== #
# -------------------------------------------- PLOT 3 : MULTIPLEXER TRANSIENT RESPONSE ---------------------------------------------- #
def plot_multiplexer() -> plt.Figure:
    """
    Time-domain voltage plot of the multiplexer SPICE simulation.

    V(vin)  — blue solid line.
    V(vout) — red solid line.

    Returns:
        - fig [plt.Figure] : Completed figure.
    """
    time, vin, vout = _parse_multiplexer(os.path.join(_SCRIPT_DIR, 'multiplexer.txt'))

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    ax.plot(time, vin,  color=_SIM_COLOR, linestyle='-', linewidth=3.0,
            label=r'$V_{\mathrm{in}}$')
    ax.plot(time, vout, color=_MEAS_COLOR, linestyle='--', linewidth=3.0,
            label=r'$V_{\mathrm{out}}$')

    ax.set_xlabel(r'Time (s)')
    ax.set_ylabel(r'Voltage (V)')
    ax.set_title(r'Transistor Simulation')
    ax.legend()
    apply_minor_grid(ax)

    return fig
# =================================================================================================================================== #



# =================================================================================================================================== #
# ------------------------------------------ PLOT 4 : SAMPLE-AND-HOLD DROOP DURING CONVERSION ---------------------------------------- #
def plot_droop() -> plt.Figure:
    """
    Voltage droop of the held capacitor during the ADC conversion window.

    Three curves show droop slope = -I_F / C_H for I_F = 30 pA:
        - C_H = 857 pF  : -0.0350 µV/µs  (minimum — just touches spec at t ≈ 112 µs)
        - C_H = 1 nF    : -0.0300 µV/µs
        - C_H = 1 µF    : -3e-5  µV/µs   (chosen — essentially flat)

    A horizontal dashed line marks the -3.922 µV specification.
    A vertical dotted line marks the acquisition end at t = 114 µs.

    Returns:
        - fig [plt.Figure] : Completed figure.
    """
    _ACQTIME_US = 114.0   # µs — total ADC conversion window

    t = np.linspace(0, _ACQTIME_US*1.5, 1000)   # µs

    # Droop: v_H(t) - v_H(0)  [µV].  slope [µV/µs] = I_F[A] / C_H[F]  (units cancel).
    droop_857pF = -((30E-12)/(872E-12))  * t
    droop_1nF   = -((30E-12)/(1E-9))  * t
    droop_1uF   = -((30E-12)/(0.01E-6))  * t

    _SPEC_UV   = -3.922
    _COL_857   = '#E07B00'   # amber  — 857 pF (minimum)
    _COL_1nF   = '#228B22'   # forest green — 1 nF
    _COL_1uF   = '#111111'   # near-black — 1 µF (chosen)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    # Specification threshold
    ax.axhline(_SPEC_UV, color=_MEAS_COLOR, linestyle='--', linewidth=3.0,
               label=rf'$\Delta V_{{H,\max}} = {_SPEC_UV}\,\mu\mathrm{{V}}$ (spec)')

    # Droop curves
    ax.plot(t, droop_857pF, color=_COL_857, linestyle='-',  linewidth=4.5,
            label=r'$C_H = 872\,\mathrm{pF}$ (minimum)')
    ax.plot(t, droop_1nF,   color=_COL_1nF, linestyle=':',  linewidth=4.5,
            label=r'$C_H = 1\,\mathrm{nF}$')
    ax.plot(t, droop_1uF,   color=_COL_1uF, linestyle='-.',  linewidth=4.5,
            label=r'$C_H = 0.01\,\mu\mathrm{F}$')

    # Vertical marker at acquisition end — crosses each droop line and the spec
    ax.axvline(_ACQTIME_US, color='#555555', linestyle=':', linewidth=1.5,
               label=rf'$\Delta t = {int(_ACQTIME_US)}\,\mu\mathrm{{s}}$ (acquisition end)')

    # Dot markers at crossing points (t = 114 µs on each droop curve)
    ax.plot(_ACQTIME_US, -((30E-12)/(872E-12))  * _ACQTIME_US, 'o', color=_COL_857,
            markersize=14, markeredgewidth=1.2, markeredgecolor='white', zorder=5)
    ax.plot(_ACQTIME_US, -((30E-12)/(1E-9))  * _ACQTIME_US, 'o', color=_COL_1nF,
            markersize=14, markeredgewidth=1.2, markeredgecolor='white', zorder=5)
    ax.plot(_ACQTIME_US, -((30E-12)/(0.01E-6))  * _ACQTIME_US, 'o', color=_COL_1uF,
            markersize=14, markeredgewidth=1.2, markeredgecolor='white', zorder=5)

    ax.set_xlim(0, _ACQTIME_US*1.5)
    ax.set_ylim(-8.5, 0.5)
    ax.set_xlabel(r'Hold time $\Delta t$ ($\mu$s)')
    ax.set_ylabel(r'$v_H - v_H(0)$ ($\mu$V)')
    ax.set_title(r'Conversion Droop for different $C_H$')
    ax.legend(loc='lower left')
    apply_minor_grid(ax)

    return fig
# =================================================================================================================================== #



# =================================================================================================================================== #
# --------------------------------------------------------------- MAIN -------------------------------------------------------------- #
if __name__ == '__main__':
    _OUT = os.path.join(_PROJECT_ROOT, 'output')
    os.makedirs(_OUT, exist_ok=True)

    fig1 = plot_filter_bode()
    fig1.savefig(os.path.join(_OUT, 'filter_bode.pdf'))
    plt.close(fig1)

    fig2 = plot_amp_vin_vout()
    fig2.savefig(os.path.join(_OUT, 'amp_vin_vout.pdf'))
    plt.close(fig2)

    fig3 = plot_multiplexer()
    fig3.savefig(os.path.join(_OUT, 'multiplexer_transient.pdf'))
    plt.close(fig3)

    fig4 = plot_droop()
    fig4.savefig(os.path.join(_OUT, 'droop.pdf'))
    plt.close(fig4)
# =================================================================================================================================== #

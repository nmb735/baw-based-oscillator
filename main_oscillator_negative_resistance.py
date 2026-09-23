# =================================================================================================================================== #
# ----------------------------------------------------------- DESCRIPTION ----------------------------------------------------------- #
# Oscillator script: parse Keysight ADS "List" exports and produce the publication-quality oscillator figures for the                #
# negative-resistance topology at nominal Vcc = 3.3 V (no transistor / resonator figures - that data wasn't exported for this        #
# topology).                                                                                                                          #
# Author: Nedal M. Benelmekki                                                                                                         #
# =================================================================================================================================== #



# =================================================================================================================================== #
# --------------------------------------------------------- EXTERNAL IMPORTS -------------------------------------------------------- #
import os                                              # Operating system interfaces.                                                #
from src.parse_ads       import ADSListParser          # ADS Data Display "List" .csv parser.                                        #
from src.plot_oscillator import OscillatorPlotter      # Oscillator figure plotting class.                                           #
# =================================================================================================================================== #



# =================================================================================================================================== #
# --------------------------------------------------------------- PATHS ------------------------------------------------------------- #
_ROOT      = os.path.dirname(os.path.abspath(__file__))                                                       # Project root.         #
DATA_DIR   = os.path.join(_ROOT, 'data', '2026-09-22', 'simulations', 'ADS', 'Negative-Resistance-3p3V')      # ADS export dir.       #
OUTPUT_DIR = os.path.join(_ROOT, 'output', '2026-09-22', 'simulations', 'ADS', 'Negative-Resistance-3p3V')    # Figure output dir.    #
# =================================================================================================================================== #



# =================================================================================================================================== #
# ------------------------------------------------------------- DESIGN CONSTANTS ---------------------------------------------------- #
CARRIER_GHZ        = 2.970    # Oscillator carrier frequency, read from the clean transient-FFT peak (transient/HB reference).       #
F_MARK_OLG_GHZ      = 2.970    # Open-loop gain: nominal marker frequency (mag + phase), snapped to the nearest exported sample -    #
                                # NOT a solved crossing (see nominal-marker note below).                                             #
F_MARK_NYQUIST_GHZ  = 2.983    # OscTest Nyquist: frequency marked on both panels (loop phase wraps through +/-180 deg here, not 0). #
PN_MARK_OFFSET_HZ   = 1e5      # Phase-noise offset frequency to mark.                                                               #
ZO_OSCTEST_OHM      = 21.5     # OscTest reference (probe) impedance - reused from the Pierce sim; confirm against the NR schematic.  #
VCC_NOMINAL         = 3.3      # Nominal supply voltage for the single-operating-point HB figure.                                     #
# NOTE: open_loop_gain_magnitude.csv / open_loop_gain_phase.csv (Gc) are numerically degenerate across the whole 2.85-3.10 GHz        #
# sweep - despite 32001 points, dB(Gc) has only ~1,600 distinct values and phase(Gc) snaps to a handful of exact atan()-type angles   #
# (0, 45, 90, 26.565, 56.31, 63.435 deg). That's consistent with a broken OscTest/Gc probe setup in the NR schematic, not real loop-  #
# gain data, so no 0 dB / 0 deg crossing from it can be trusted. The magnitude/phase panels are still plotted from the raw export,    #
# but marked at the nearest sample to F_MARK_OLG_GHZ (the carrier) rather than a solved crossing - re-check the probe setup in ADS.   #
OLG_NOMINAL_NOTE    = r'nominal marker (Gc probe data inconclusive)'                                                                  #
# =================================================================================================================================== #



# =================================================================================================================================== #
# ------------------------------------------------------------ PARSE ADS EXPORTS ---------------------------------------------------- #
p = lambda name: ADSListParser(os.path.join(DATA_DIR, name))

olg_mag          = p('open_loop_gain_magnitude.csv')
olg_phase        = p('open_loop_gain_phase.csv')
osctest_mag      = p('nyquist_s11_db.csv')
osctest_phase    = p('nyquist_s11_phase.csv')
tran_time        = p('tran_vout_time.csv')
tran_fft         = p('tran_vout_fft.csv')
hb_waveform      = p('hb_waveform.csv')
hb_spectrum      = p('hb_spectrum.csv')
phase_noise      = p('phase_noise.csv').rename({
    'indep(__d, 1)':             'offset_Hz',
    'plot_vs(pnmx, noisefreq)':  'L_dBcHz',
})

for parser in (olg_mag, olg_phase, osctest_mag, osctest_phase, tran_time, tran_fft,
               hb_waveform, hb_spectrum, phase_noise):
    print(parser)
# =================================================================================================================================== #



# =================================================================================================================================== #
# ------------------------------------------------------------------ PLOT ----------------------------------------------------------- #
plotter = OscillatorPlotter(OUTPUT_DIR)

# ── Randall-Hock open-loop gain (nominal marker - see NOTE above) ──────────────────────────────────────────────────────────────── #
plotter.plot_open_loop_gain(
    olg_mag, olg_phase,
    f_mark_mag_ghz=F_MARK_OLG_GHZ, f_mark_phase_ghz=F_MARK_OLG_GHZ,
    nominal_marker=True, nominal_note=OLG_NOMINAL_NOTE,
    filename='open_loop_gain',
)

# ── OscTest Nyquist criterion (loop phase wraps through +/-180 deg, not 0) ─────────────────────────────────────────────────────── #
plotter.plot_osctest_nyquist(
    osctest_mag, osctest_phase, f_mark_ghz=F_MARK_NYQUIST_GHZ, Zo=ZO_OSCTEST_OHM,
    phase_target_deg=180.0,
    filename='osctest_nyquist',
)

# ── Transient output + steady-state spectrum ────────────────────────────────────────────────────────────────────────────────────── #
plotter.plot_transient(tran_time, tran_fft, filename='transient_output')

# ── Harmonic-balance steady state: single operating point + family vs. Vcc ─────────────────────────────────────────────────────── #
plotter.plot_hb_operating_point(hb_waveform, hb_spectrum, vcc=VCC_NOMINAL, filename='hb_operating_point')
plotter.plot_hb_family(hb_waveform, hb_spectrum, filename='hb_family_vs_vcc')

# ── Phase noise ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────── #
plotter.plot_phase_noise(
    phase_noise, offset_col='offset_Hz', pn_col='L_dBcHz', carrier_ghz=CARRIER_GHZ,
    mark_offset_hz=PN_MARK_OFFSET_HZ, filename='phase_noise',
)
# =================================================================================================================================== #

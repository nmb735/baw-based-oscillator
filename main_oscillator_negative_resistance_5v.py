# =================================================================================================================================== #
# ----------------------------------------------------------- DESCRIPTION ----------------------------------------------------------- #
# Oscillator script: parse Keysight ADS "List" exports and produce the oscillator figures for the negative-resistance topology at    #
# Vcc = 5 V. No transistor / resonator figures - that data wasn't exported for this topology at either bias point.                   #
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
_ROOT      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(_ROOT, 'data', '2026-09-22', 'simulations', 'ADS', 'Negative-Resistance-5V')
OUTPUT_DIR = os.path.join(_ROOT, 'output', '2026-09-22', 'simulations', 'ADS', 'Negative-Resistance-5V')
# =================================================================================================================================== #



# =================================================================================================================================== #
# ------------------------------------------------------------- DESIGN CONSTANTS ---------------------------------------------------- #
CARRIER_GHZ        = 2.960    # Oscillator carrier frequency, read from the clean transient-FFT peak (dominant over the 2nd/3rd       #
                                # harmonics at 5.925/8.885 GHz, so unambiguous).                                                       #
F_MARK_OLG_GHZ      = 2.960    # Open-loop gain: nominal marker frequency (mag + phase), snapped to the nearest exported sample -     #
                                # NOT a solved crossing. Same Gc-probe degeneracy as the 3.3 V case (~5.5% distinct values over       #
                                # 32001 points, phase snapping to exact atan()-type angles) - see main_oscillator_negative_resistance.py.
F_MARK_NYQUIST_GHZ  = 3.047    # OscTest Nyquist: frequency marked on both panels (loop phase wraps through +/-180 deg here, not 0,   #
                                # same convention as the 3.3 V case). Sits ~87 MHz above the carrier - a larger small-signal/large-    #
                                # signal pulling gap than the 3.3 V case's ~13 MHz, plausible given the bias-dependent negative-       #
                                # resistance value, but worth a second look against the schematic.                                    #
PN_MARK_OFFSET_HZ   = 1e5      # Phase-noise offset frequency to mark.                                                                #
ZO_OSCTEST_OHM      = 21.5     # OscTest reference (probe) impedance - reused from the Pierce sim; confirm against the NR schematic.  #
VCC_NOMINAL         = 5.0      # Nominal supply voltage for the single-operating-point HB figure.                                     #
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

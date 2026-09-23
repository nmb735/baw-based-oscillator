# =================================================================================================================================== #
# ----------------------------------------------------------- DESCRIPTION ----------------------------------------------------------- #
# Oscillator script: parse Keysight ADS "List" exports and produce the oscillator figures for the Pierce topology at Vcc = 5 V.       #
# No transistor / resonator figures - that data wasn't re-exported for this bias point (see the 3.3 V figures for those).            #
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
DATA_DIR   = os.path.join(_ROOT, 'data', '2026-09-22', 'simulations', 'ADS', 'Pierce-5V')
OUTPUT_DIR = os.path.join(_ROOT, 'output', '2026-09-22', 'simulations', 'ADS', 'Pierce-5V')
# =================================================================================================================================== #



# =================================================================================================================================== #
# ------------------------------------------------------------- DESIGN CONSTANTS ---------------------------------------------------- #
# Determined the same way as the 3.3 V figures: carrier from the clean transient-FFT peak, OLG/Nyquist marks from the genuine
# zero-phase crossing nearest that carrier (this Gc export is well-behaved here, ~49% distinct values over 501 points - no
# nominal-marker workaround needed, unlike the negative-resistance topology).
F_MARK_OLG_GHZ       = 2.972    # Randall-Hock open-loop gain: solved zero-phase crossing (mag panel snaps to the nearest sample).    #
F_MARK_NYQUIST_GHZ   = 2.972    # OscTest Nyquist: solved 0 deg crossing.                                                             #
CARRIER_GHZ          = 2.970    # Oscillator carrier frequency (transient FFT peak).                                                  #
PN_MARK_OFFSET_HZ    = 1e5      # Phase-noise offset frequency to mark.                                                               #
ZO_OSCTEST_OHM       = 21.5     # OscTest reference (probe) impedance - reused from the 3.3 V Pierce sim; confirm against schematic.  #
VCC_NOMINAL          = 5.0      # Nominal supply voltage for the single-operating-point HB figure.                                    #
# =================================================================================================================================== #



# =================================================================================================================================== #
# ------------------------------------------------------------ PARSE ADS EXPORTS ---------------------------------------------------- #
p = lambda name: ADSListParser(os.path.join(DATA_DIR, name))

olg_mag          = p('open_gain_loop_magnitude.csv')
olg_phase        = p('open_gain_loop_phase.csv')
osctest_mag      = p('nyquist_magnitude.csv')
osctest_phase    = p('nyquist_phase.csv')
tran_time        = p('tran_vout.csv')
tran_fft         = p('tran_vout_fft.csv')
hb_waveform      = p('HB_waveform.csv')
hb_spectrum      = p('HB_spectrum.csv')
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

# ── Randall-Hock open-loop gain ─────────────────────────────────────────────────────────────────────────────────────────────────── #
plotter.plot_open_loop_gain(
    olg_mag, olg_phase,
    f_mark_mag_ghz=F_MARK_OLG_GHZ, f_mark_phase_ghz=F_MARK_OLG_GHZ,
    filename='open_loop_gain',
)

# ── OscTest Nyquist criterion ────────────────────────────────────────────────────────────────────────────────────────────────────── #
plotter.plot_osctest_nyquist(
    osctest_mag, osctest_phase, f_mark_ghz=F_MARK_NYQUIST_GHZ, Zo=ZO_OSCTEST_OHM,
    phase_target_deg=0.0,
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

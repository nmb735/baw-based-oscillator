# =================================================================================================================================== #
# ----------------------------------------------------------- DESCRIPTION ----------------------------------------------------------- #
# Oscillator script: parse Keysight ADS "List" exports and produce the eight publication-quality oscillator figures.                  #
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
_ROOT      = os.path.dirname(os.path.abspath(__file__))                           # Project root directory.                          #
DATA_DIR   = os.path.join(_ROOT, 'data', '2026-09-22', 'simulations', 'ADS')       # ADS List-export directory.                       #
OUTPUT_DIR = os.path.join(_ROOT, 'output', '2026-09-22', 'simulations', 'ADS')     # Figure output directory.                         #
# =================================================================================================================================== #



# =================================================================================================================================== #
# ------------------------------------------------------------- DESIGN CONSTANTS ---------------------------------------------------- #
F_MARK_OLG_MAG_GHZ   = 2.961    # Randall-Hock corrected open-loop gain: frequency marked on the magnitude panel.                     #
F_MARK_OLG_PHASE_GHZ = 2.963    # Randall-Hock corrected open-loop gain: frequency marked on the phase panel.                         #
F_MARK_NYQUIST_GHZ   = 2.965    # OscTest Nyquist: frequency marked on both panels (phase = 0 crossing).                              #
CARRIER_GHZ          = 2.976    # Oscillator carrier frequency (transient/spectrum inset centre, phase-noise reference).              #
PN_MARK_OFFSET_HZ    = 1e5      # Phase-noise offset frequency to mark.                                                               #
ZO_OSCTEST_OHM       = 21.5     # OscTest reference (probe) impedance.                                                                #
VCC_NOMINAL          = 3.3      # Nominal supply voltage for the single-operating-point HB figure.                                    #
# =================================================================================================================================== #



# =================================================================================================================================== #
# ------------------------------------------------------------ PARSE ADS EXPORTS ---------------------------------------------------- #
p = lambda name: ADSListParser(os.path.join(DATA_DIR, name))

transistor_iv    = p('transistor_iv.csv')
transistor_gain  = p('transistor_gain.csv')
resonator        = p('resonator_sparams_1real_2mbvd.csv')
olg_mag          = p('open_loop_gain_magnitude.csv')
olg_phase        = p('open_loop_gain_phase.csv')
osctest_mag      = p('nyquist_s11_dB.csv')
osctest_phase    = p('nyquist_s11_phase.csv')
tran_time        = p('tran_vout_time.csv')
tran_fft         = p('tran_vout_time_fft.csv')
hb_waveform      = p('hb_waveform.csv')
hb_spectrum      = p('hb_spectrum.csv')
phase_noise      = p('phase_noise.csv').rename({
    'indep(__d, 1)':             'offset_Hz',
    'plot_vs(pnmx, noisefreq)':  'L_dBcHz',
})

for parser in (transistor_iv, transistor_gain, resonator, olg_mag, olg_phase,
               osctest_mag, osctest_phase, tran_time, tran_fft,
               hb_waveform, hb_spectrum, phase_noise):
    print(parser)
# =================================================================================================================================== #



# =================================================================================================================================== #
# ------------------------------------------------------------------ PLOT ----------------------------------------------------------- #
plotter = OscillatorPlotter(OUTPUT_DIR)

# ── Transistor ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────── #
plotter.plot_transistor_iv(transistor_iv, filename='transistor_iv')
plotter.plot_transistor_gain(transistor_gain, filename='transistor_hfe')

# ── BAW resonator: measured vs. mBVD equivalent circuit ────────────────────────────────────────────────────────────────────────── #
# NOTE: assumes the file's own column order ("...1real_2mbvd") -> dB(Z(1,1)) = measured, dB(Z(2,2)) = mBVD model.
# Swap col_measured/col_model below if that assumption is backwards.
plotter.plot_resonator_reflection(
    resonator, col_measured='dB(Z(1,1))', col_model='dB(Z(2,2))',
    filename='resonator_reflection',
)

# ── Randall-Hock corrected open-loop gain ───────────────────────────────────────────────────────────────────────────────────────── #
plotter.plot_open_loop_gain(
    olg_mag, olg_phase,
    f_mark_mag_ghz=F_MARK_OLG_MAG_GHZ, f_mark_phase_ghz=F_MARK_OLG_PHASE_GHZ,
    filename='open_loop_gain',
)

# ── OscTest Nyquist criterion ────────────────────────────────────────────────────────────────────────────────────────────────────── #
plotter.plot_osctest_nyquist(
    osctest_mag, osctest_phase, f_mark_ghz=F_MARK_NYQUIST_GHZ, Zo=ZO_OSCTEST_OHM,
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

# =================================================================================================================================== #
# ----------------------------------------------------------- DESCRIPTION ----------------------------------------------------------- #
# Comparison script: re-parse the Pierce and negative-resistance Keysight ADS "List" exports (both at Vcc = 5 V) and produce         #
# overlay figures comparing the two topologies. No transistor / resonator comparison - that data only exists for 3.3 V Pierce.       #
# Author: Nedal M. Benelmekki                                                                                                         #
# =================================================================================================================================== #



# =================================================================================================================================== #
# --------------------------------------------------------- EXTERNAL IMPORTS -------------------------------------------------------- #
import os                                              # Operating system interfaces.                                                #
from src.parse_ads        import ADSListParser         # ADS Data Display "List" .csv parser.                                        #
from src.plot_comparison  import ComparisonPlotter      # Pierce-vs-negative-resistance overlay plotting class.                       #
# =================================================================================================================================== #



# =================================================================================================================================== #
# --------------------------------------------------------------- PATHS ------------------------------------------------------------- #
_ROOT          = os.path.dirname(os.path.abspath(__file__))
PIERCE_DIR     = os.path.join(_ROOT, 'data', '2026-09-22', 'simulations', 'ADS', 'Pierce-5V')
NEGRES_DIR     = os.path.join(_ROOT, 'data', '2026-09-22', 'simulations', 'ADS', 'Negative-Resistance-5V')
OUTPUT_DIR     = os.path.join(_ROOT, 'output', '2026-09-22', 'simulations', 'ADS', 'Comparison-5V')
LABEL_PIERCE   = 'Pierce'
LABEL_NEGRES   = 'Neg.\\ Resistance'
# =================================================================================================================================== #



# =================================================================================================================================== #
# ------------------------------------------------------------- DESIGN CONSTANTS ---------------------------------------------------- #
# Pierce (see main_oscillator_5v.py for how these were determined).
PIERCE_F_MARK_OLG_GHZ     = 2.972    # Randall-Hock open-loop gain: solved zero-phase crossing.                                       #
PIERCE_F_MARK_NYQUIST_GHZ = 2.972    # OscTest Nyquist: solved 0 deg crossing.                                                        #
PIERCE_CARRIER_GHZ        = 2.970    # Oscillator carrier frequency (transient FFT peak).                                             #
PIERCE_ZO_OSCTEST_OHM     = 21.5     # OscTest reference (probe) impedance.                                                           #

# Negative resistance (see main_oscillator_negative_resistance_5v.py for how these were determined).
NEGRES_F_MARK_OLG_GHZ     = 2.960    # Nominal marker only - Gc probe data is numerically degenerate (see that script's NOTE).        #
NEGRES_F_MARK_NYQUIST_GHZ = 3.047    # OscTest Nyquist: loop phase wraps through +/-180 deg here, not 0.                              #
NEGRES_CARRIER_GHZ        = 2.960    # Oscillator carrier frequency (transient FFT peak).                                             #
NEGRES_ZO_OSCTEST_OHM     = 21.5     # OscTest reference (probe) impedance - reused from Pierce; confirm against the NR schematic.    #

VCC_NOMINAL       = 5.0
PN_MARK_OFFSET_HZ = 1e5
# =================================================================================================================================== #



# =================================================================================================================================== #
# ------------------------------------------------------------ PARSE ADS EXPORTS ---------------------------------------------------- #
pp = lambda name: ADSListParser(os.path.join(PIERCE_DIR, name))
pn = lambda name: ADSListParser(os.path.join(NEGRES_DIR, name))

pierce_olg_mag     = pp('open_gain_loop_magnitude.csv')
pierce_olg_phase   = pp('open_gain_loop_phase.csv')
pierce_nq_mag      = pp('nyquist_magnitude.csv')
pierce_nq_phase    = pp('nyquist_phase.csv')
pierce_tran_time   = pp('tran_vout.csv')
pierce_tran_fft    = pp('tran_vout_fft.csv')
pierce_hb_wave     = pp('HB_waveform.csv')
pierce_hb_spec     = pp('HB_spectrum.csv')
pierce_pn          = pp('phase_noise.csv').rename({
    'indep(__d, 1)':             'offset_Hz',
    'plot_vs(pnmx, noisefreq)':  'L_dBcHz',
})

negres_olg_mag     = pn('open_loop_gain_magnitude.csv')
negres_olg_phase   = pn('open_loop_gain_phase.csv')
negres_nq_mag      = pn('nyquist_s11_db.csv')
negres_nq_phase    = pn('nyquist_s11_phase.csv')
negres_tran_time   = pn('tran_vout_time.csv')
negres_tran_fft    = pn('tran_vout_fft.csv')
negres_hb_wave     = pn('hb_waveform.csv')
negres_hb_spec     = pn('hb_spectrum.csv')
negres_pn          = pn('phase_noise.csv').rename({
    'indep(__d, 1)':             'offset_Hz',
    'plot_vs(pnmx, noisefreq)':  'L_dBcHz',
})
# =================================================================================================================================== #



# =================================================================================================================================== #
# ------------------------------------------------------------------ PLOT ----------------------------------------------------------- #
plotter = ComparisonPlotter(OUTPUT_DIR)

# ── Randall-Hock open-loop gain ─────────────────────────────────────────────────────────────────────────────────────────────────── #
plotter.plot_open_loop_gain_comparison(
    pierce_olg_mag, pierce_olg_phase, negres_olg_mag, negres_olg_phase,
    label_a=LABEL_PIERCE, label_b=LABEL_NEGRES,
    f_mark_a_ghz=PIERCE_F_MARK_OLG_GHZ, f_mark_b_ghz=NEGRES_F_MARK_OLG_GHZ,
    nominal_a=False, nominal_b=True,
    filename='open_loop_gain_comparison',
)

# ── OscTest Nyquist criterion ────────────────────────────────────────────────────────────────────────────────────────────────────── #
plotter.plot_osctest_nyquist_comparison(
    pierce_nq_mag, pierce_nq_phase, negres_nq_mag, negres_nq_phase,
    label_a=LABEL_PIERCE, label_b=LABEL_NEGRES,
    f_mark_a_ghz=PIERCE_F_MARK_NYQUIST_GHZ, f_mark_b_ghz=NEGRES_F_MARK_NYQUIST_GHZ,
    phase_target_a_deg=0.0, phase_target_b_deg=180.0,
    Zo_a=PIERCE_ZO_OSCTEST_OHM, Zo_b=NEGRES_ZO_OSCTEST_OHM,
    filename='osctest_nyquist_comparison',
)

# ── Transient output + steady-state spectrum ────────────────────────────────────────────────────────────────────────────────────── #
plotter.plot_transient_comparison(
    pierce_tran_time, pierce_tran_fft, negres_tran_time, negres_tran_fft,
    label_a=LABEL_PIERCE, label_b=LABEL_NEGRES,
    filename='transient_output_comparison',
)

# ── Harmonic-balance steady state: single operating point ──────────────────────────────────────────────────────────────────────── #
plotter.plot_hb_operating_point_comparison(
    pierce_hb_wave, pierce_hb_spec, negres_hb_wave, negres_hb_spec,
    label_a=LABEL_PIERCE, label_b=LABEL_NEGRES, vcc=VCC_NOMINAL,
    filename='hb_operating_point_comparison',
)

# ── Phase noise ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────── #
plotter.plot_phase_noise_comparison(
    pierce_pn, negres_pn, label_a=LABEL_PIERCE, label_b=LABEL_NEGRES,
    offset_col='offset_Hz', pn_col='L_dBcHz',
    carrier_a_ghz=PIERCE_CARRIER_GHZ, carrier_b_ghz=NEGRES_CARRIER_GHZ,
    mark_offset_hz=PN_MARK_OFFSET_HZ,
    filename='phase_noise_comparison',
)
# =================================================================================================================================== #

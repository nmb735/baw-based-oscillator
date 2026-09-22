# =================================================================================================================================== #
# ----------------------------------------------------------- DESCRIPTION ----------------------------------------------------------- #
# EXP-000 / 2026-07-28 — Multiplexed Polarization RFID Patch Antenna Array.                                                           #
# Compares the FEKO simulation of the single two-feed antenna unit against the pair-wise measurement of the 4-port subarray.          #
#                                                                                                                                     #
# Port convention (Block 3.C):  1 = Ant1-S   2 = Ant1-C   3 = Ant2-S   4 = Ant2-C                                                     #
# Simulation is a single unit: FEKO MicrostripPort1 -> S feed, MicrostripPort2 -> C feed.                                             #
#   sim S11 is therefore compared against measured S11 (Ant1-S) and S33 (Ant2-S);                                                     #
#   sim S22 against measured S22 (Ant1-C) and S44 (Ant2-C).  S44 is recovered from the supplementary pair42 acquisition (port 4       #
#   driven on VNA P1), so the full 4x4 reflection set S11..S44 is now available.                                                      #
# Author: Nedal M. Benelmekki                                                                                                         #
# =================================================================================================================================== #



# =================================================================================================================================== #
# --------------------------------------------------------- EXTERNAL IMPORTS -------------------------------------------------------- #
import os                                                # Operating system interfaces.                                               #
import matplotlib.pyplot        as plt                   # Plotting (repeatability figure only).                                      #
import style.style              as style                 # "Belt DEE" global rcParams (applied on import).                            #
from   style.style              import (                 # Typography and geometry tokens.                                            #
    LABEL_SIZE, LEGEND_SIZE, TICK_SIZE, TITLE_SIZE, FIG_WIDTH, FIG_HEIGHT, COLOR_CYCLE, MARKER_CYCLE, apply_minor_grid,
    ETSI_UHF_BAND_GHZ, ETSI_UHF_CAPTION,
)
from   src.parse_feko           import FekoParser        # FEKO .s2p simulation parser.                                               #
from   src.parse_subarray       import SubarrayParser    # Pair-wise 4-port subarray measurement parser.                              #
from   src.plot_measurements    import MeasurementPlotter# Measurement-only plotting class.                                            #
from   src.final_plot           import FinalPlotter      # Simulation-vs-measurement overlay class.                                    #
# =================================================================================================================================== #



# =================================================================================================================================== #
# --------------------------------------------------------------- PATHS ------------------------------------------------------------- #
_ROOT     = os.path.dirname(os.path.abspath(__file__))                        # Project root (where this script lives).               #
CAMPAIGN  = '2026-07-28'                                                      # Experiment folder / EXP-000.                          #
SIM_DIR   = os.path.join(_ROOT, 'data',   CAMPAIGN, 'simulations')            # FEKO .s2p export.                                     #
MEAS_DIR  = os.path.join(_ROOT, 'data',   CAMPAIGN, 'measurements')           # Six pair-wise NanoVNA .s2p files.                     #
MEAS_OUT  = os.path.join(_ROOT, 'output', CAMPAIGN, 'measurements')           # Measurement-only figures.                             #
FINAL_OUT = os.path.join(_ROOT, 'output', CAMPAIGN, 'final')                  # Simulation-vs-measurement figures.                    #
# =================================================================================================================================== #



# =================================================================================================================================== #
# ------------------------------------------------------------ PLOT WINDOW ---------------------------------------------------------- #
# The FEKO export spans 0.80-1.00 GHz on a coarse 14-point grid; the NanoVNA sweeps 0.75-1.00 GHz on 1601 points.                     #
# Overlay figures are therefore restricted to the common 0.80-1.00 GHz band; measurement-only figures use the full sweep.             #
XLIM_COMMON = (0.80, 1.00)                                                                                                            #
XLIM_MEAS   = (0.75, 1.00)                                                                                                            #
XLIM_RFID   = (0.80, 0.92)  # UHF RFID window (lower band 865-868 MHz, upper band 915-921 MHz) for the decluttered marker figure.    #
YLIM_MAG    = (-25.0, 0.0)                                                                                                            #
YLIM_RFID   = (-20.0, 0.0)   # Tighter magnitude window for the decluttered UHF figure.                                              #
LW          = 2.5                                                                                                                     #
# =================================================================================================================================== #



# =================================================================================================================================== #
# ---------------------------------------------------------------- PARSE ------------------------------------------------------------ #
feko = FekoParser(os.path.join(SIM_DIR, 'cumv5_SParameter.s2p'))
meas = SubarrayParser(MEAS_DIR, stem='p3_subarray_L30')

print(feko)
print(meas)

# ── resonance summary ─────────────────────────────────────────────────────────
# Both S-feed ports are dual-resonant, so the global minimum alone is not a fair summary: every notch is listed.                      #
def _fmt(notches: list) -> str:
    """Render a resonance list as 'f GHz (depth dB)' entries."""
    return '  |  '.join(f'{f:.4f} GHz ({d:6.2f} dB)' for f, d in notches) or '-- none --'

print('\nMeasured reflections (installed-with-cables, cable bank NOT de-embedded):')
for param in meas.reflections:
    port = int(param[1])
    print(f'  {param}  Port {port} ({SubarrayParser.PORT_LABELS[port]:8s}) '
          f'[{len(meas.repeats(port))} acq.]  {_fmt(meas.resonances(param))}')
if meas.missing:
    print(f'  NOT ACQUIRED: {", ".join(meas.missing)} — '
          f'port 4 never sits on VNA P1 in the six-pair schedule.')

print('\nSimulated reflections (single unit, 14-point grid — notch depth is grid-limited):')
for param, feed in (('S11', 'S feed'), ('S22', 'C feed')):
    df = feko.get_df(param)
    print(f'  {param}  {feed:8s}                {" " * 10}'
          f'{_fmt(SubarrayParser.find_resonances(df["freq_GHz"].values, df["mag_dB"].values))}')
# =================================================================================================================================== #



# =================================================================================================================================== #
# ------------------------------------------------------ MEASUREMENT-ONLY PLOTS ----------------------------------------------------- #
meas_plotter = MeasurementPlotter(MEAS_OUT)

# All four measured reflections on one axis: Ant1-S, Ant1-C, Ant2-S, Ant2-C.
# meas.reflections now resolves to [S11, S22, S33, S44]; S44 comes from the pair42 file.
meas_plotter.plot_s_parameters(
    meas,
    params=meas.reflections,
    display='magnitude',
    show_minimum=True,
    linewidth=LW,
    linestyles=['-', '--', '-.', ':'],
    xlim=XLIM_MEAS,
    ylim_mag=YLIM_MAG,
    title=r'Measured reflections --- $S_{11}$ Ant1-S, $S_{22}$ Ant1-C, $S_{33}$ Ant2-S, $S_{44}$ Ant2-C',
    filename='Measured_reflections.pdf',
)

# ── decluttered UHF-RFID publication figure ───────────────────────────────────
# Four traces on one axis read as clutter once each carries a minimum guide-line and an f/depth text box, so this variant strips all
# annotations, drops the title, and crops the sweep to the UHF RFID band.  The raw NanoVNA ripple is smoothed with a Savitzky-Golay
# filter so the traces read cleanly, and each carries a fixed count of IEEE colour-matched markers (circle / square / triangle /
# diamond), evenly spaced and phase-staggered.  The European UHF RFID band (ETSI EN 302 208, 865.7-867.5 MHz) is washed in behind
# the traces as a faint grey stripe so the in-band match reads at a glance; it carries no legend entry, only a caption note.  The
# resonance table is exported to a .txt for the figure caption.
_n_refl       = len(meas.reflections)
_REFL_MARKERS = MARKER_CYCLE[:_n_refl]                                          # 'o', 's', '^', 'D' — one glyph per S_ii.            #
_SMOOTH_WIN   = 41                                                              # ~6.4 MHz Savitzky-Golay window on the 1601-pt sweep.#
_N_MARKERS    = 9                                                               # Identical marker count on every trace.              #

meas_plotter.plot_s_parameters(
    meas,
    params=meas.reflections,
    display='magnitude',
    show_minimum=False,                                                         # No maximum-resonance guide lines / annotations.     #
    linewidth=LW,
    linestyles=['-', '--', '-.', ':'],
    colors=COLOR_CYCLE[:_n_refl],
    markers=_REFL_MARKERS,
    n_markers=_N_MARKERS,
    markersize=9,
    smooth=_SMOOTH_WIN,
    xlim=XLIM_RFID,
    band=ETSI_UHF_BAND_GHZ,                                                     # Faint grey ETSI UHF RFID stripe, no legend entry.   #
    ylim_mag=YLIM_RFID,
    title=None,                                                                 # No title — the caption carries the detail.          #
    filename='Measured_reflections_UHF.pdf',
)

# ── resonance table for the figure caption ────────────────────────────────────
_MARKER_NAMES = {'o': 'circle', 's': 'square', '^': 'triangle', 'D': 'diamond',
                 'v': 'down-triangle', 'P': 'plus', '*': 'star', 'X': 'cross'}
os.makedirs(MEAS_OUT, exist_ok=True)
_caption_path = os.path.join(MEAS_OUT, 'Measured_reflections_UHF_caption.txt')
with open(_caption_path, 'w', encoding='utf-8') as fh:
    fh.write('Measured reflection coefficients of the 4-port subarray over '
             f'{XLIM_RFID[0]:.2f}-{XLIM_RFID[1]:.2f} GHz, covering the UHF RFID lower (865-868 MHz) and '
             'upper (915-921 MHz) bands, installed with cables (cable bank not de-embedded). Traces are '
             'Savitzky-Golay smoothed and distinguished by colour-matched markers. Resonances are read '
             'from the raw sweep. ' + ETSI_UHF_CAPTION + '\n\n')
    fh.write(f'{"Trace":6s}{"Port":10s}{"Marker":11s}Resonances [f (GHz), depth (dB)]\n')
    for param, marker in zip(meas.reflections, _REFL_MARKERS):
        port = int(param[1])
        fh.write(f'{param:6s}{SubarrayParser.PORT_LABELS[port]:10s}'
                 f'{_MARKER_NAMES.get(marker, marker):11s}{_fmt(meas.resonances(param))}\n')
print(f'Saved -> {_caption_path}')
# =================================================================================================================================== #



# =================================================================================================================================== #
# --------------------------------------------------- SIMULATION VS. MEASUREMENT ---------------------------------------------------- #
final = FinalPlotter(FINAL_OUT)

# ── S feed: simulated port 1 against both measured S-feeds (Ant1 and Ant2) ────
final.plot_s_parameters(
    params=['S11', 'S33'],
    sim_parsers={'S11': feko['S11']},
    meas_s2p=meas,
    display='magnitude',
    linewidth=LW,
    sim_colors=[COLOR_CYCLE[0]],
    meas_colors=[COLOR_CYCLE[0], COLOR_CYCLE[1]],
    sim_linestyles=['-'],
    meas_linestyles=['--', '-.'],
    # Minimum annotation is deliberately off here: the S-feed ports are dual-resonant and the annotator marks only the global
    # minimum, which falls on a different notch for S11 than for S33 and reads as a detuning that is not there.  The full notch
    # list is printed to stdout instead.
    show_minimum_sim=False,
    show_minimum_meas=False,
    xlim=XLIM_COMMON,
    ylim_mag=YLIM_MAG,
    title=r'S feed --- simulation vs.\ measurement ($S_{11}$ Ant1-S, $S_{33}$ Ant2-S)',
    filename='sim_vs_meas_S_feed.pdf',
)

# ── C feed: simulated port 2 against both measured C feeds (Ant1 and Ant2) ────
# S44 (Ant2-C) is now recovered from the pair42 acquisition, so the C feed mirrors
# the S feed: one simulated curve compared against the two physical C ports.
final.plot_s_parameters(
    params=['S22', 'S44'],
    sim_parsers={'S22': feko['S22']},
    meas_s2p=meas,
    display='magnitude',
    linewidth=LW,
    sim_colors=[COLOR_CYCLE[1]],
    meas_colors=[COLOR_CYCLE[1], COLOR_CYCLE[3]],
    sim_linestyles=['-'],
    meas_linestyles=['--', '-.'],
    # Minimum annotation off for parity with the S-feed plot: two measured ports
    # plus a simulated curve would stack three boxes; the notch list is printed to stdout.
    show_minimum_sim=False,
    show_minimum_meas=False,
    xlim=XLIM_COMMON,
    ylim_mag=YLIM_MAG,
    title=r'C feed --- simulation vs.\ measurement ($S_{22}$ Ant1-C, $S_{44}$ Ant2-C)',
    filename='sim_vs_meas_C_feed.pdf',
)

# ── All four reflections on one axis ──────────────────────────────────────────
# The two simulated feeds (S feed -> S11, C feed -> S22) overlaid on the full
# measured reflection set.  S33/S44 are the second unit's S/C ports and share the
# simulated feed of their polarisation, so they are drawn measurement-only.
final.plot_s_parameters(
    params=['S11', 'S22', 'S33', 'S44'],
    sim_parsers={'S11': feko['S11'], 'S22': feko['S22']},
    meas_s2p=meas,
    display='magnitude',
    linewidth=LW,
    sim_colors=[COLOR_CYCLE[0], COLOR_CYCLE[1], COLOR_CYCLE[2], COLOR_CYCLE[3]],
    meas_colors=[COLOR_CYCLE[0], COLOR_CYCLE[1], COLOR_CYCLE[2], COLOR_CYCLE[3]],
    sim_linestyles=['-', '-', '-', '-'],
    meas_linestyles=['--', '--', '-.', ':'],
    xlim=XLIM_COMMON,
    ylim_mag=YLIM_MAG,
    title=r'Reflections $S_{11}$--$S_{44}$ --- simulation vs.\ measurement',
    filename='sim_vs_meas_reflections.pdf',
)
# =================================================================================================================================== #



# =================================================================================================================================== #
# ------------------------------------------------------ MATE REPEATABILITY --------------------------------------------------------- #
# Ports 1 and 2 sit on VNA P1 in more than one pair, so their reflection is measured more than once.  The spread between those        #
# repeats bounds the cable--antenna mate repeatability and sets the floor below which sim-vs-meas disagreement cannot be read.        #
fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), constrained_layout=True)

for i, port in enumerate(meas.measured_ports):                                  # Ports whose reflection was acquired.               #
    color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
    for j, (pair, df) in enumerate(sorted(meas.repeats(port).items())):
        ax.plot(
            df['freq_GHz'], df['mag_dB'],
            color=color,
            linestyle=['-', '--', '-.', ':'][j % 4],
            linewidth=LW * 0.9,
            label=rf'Port {port} (\textrm{{{SubarrayParser.PORT_LABELS[port]}}}) --- pair {pair}',
        )

ax.set_xlabel(r'Frequency [GHz]', fontsize=LABEL_SIZE)
ax.set_ylabel(r'Magnitude [dB]',  fontsize=LABEL_SIZE)
ax.set_title(r'Reflection repeatability across pair acquisitions', fontsize=TITLE_SIZE)
ax.tick_params(labelsize=TICK_SIZE)
ax.set_xlim(XLIM_MEAS)
ax.set_ylim(YLIM_MAG)
ax.legend(fontsize=LEGEND_SIZE * 0.8, loc='lower left')
apply_minor_grid(ax)

os.makedirs(MEAS_OUT, exist_ok=True)
_path = os.path.join(MEAS_OUT, 'Measured_repeatability.pdf')
fig.savefig(_path, dpi=600, bbox_inches='tight')
plt.close(fig)
print(f'Saved -> {_path}')
# =================================================================================================================================== #

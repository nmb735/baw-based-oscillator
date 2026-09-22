# =================================================================================================================================== #
# ----------------------------------------------------------- DESCRIPTION ----------------------------------------------------------- #
# Example script: parse CST S-parameter files and produce a variety of publication-quality plots.                                     #
# Author: Nedal M. Benelmekki                                                                                                         #
# =================================================================================================================================== #



# =================================================================================================================================== #
# --------------------------------------------------------- EXTERNAL IMPORTS -------------------------------------------------------- #
import os                                              # Operating system interfaces.                                                 #                        
from src.parse_cst          import CSTParser           # CST S-parameter parser for CST simulation output files.                      #
from src.parse_measurements import S2PParser           # Touchstone .s2p parser for non-full-two-port NanoVNA measurements.           #
from src.plot_simulation    import SimulationPlotter   # CST simulation plotting class.                                               #
from src.parse_feko         import FekoParser          # FEKO S-parameter parser for FEKO simulation output files.                    #
from src.plot_measurements  import MeasurementPlotter  # Measurement plotting class.                                                  #
from src.final_plot         import FinalPlotter        # Final plotting class.                                                        #
# =================================================================================================================================== #



# =================================================================================================================================== #
# --------------------------------------------------------------- PATHS ------------------------------------------------------------- #
_ROOT       = os.path.dirname(os.path.abspath(__file__))                   # Project root directory (where main.py is located).       #
DATA_DIR    = os.path.join(_ROOT, 'data', '2026-05-14', 'simulations')     # CST simulation data directory (raw .txt files from CST). #
OUTPUT_DIR  = os.path.join(_ROOT, 'output', '2026-05-14', 'simulations')   # CST simulation output directory (plots, etc.).           #
MEAS_DIR    = os.path.join(_ROOT, 'data', '2026-05-14', 'measurements')    # Measurement data directory (Touchstone .s2p files, etc.).#
MEAS_OUT    = os.path.join(_ROOT, 'output', '2026-05-14', 'measurements')  # Measurement output directory (plots, etc.).              #
# =================================================================================================================================== #



# =================================================================================================================================== #
# -------------------------------------------------------------- PARSE CST ---------------------------------------------------------- #
CST_DATA_DIR = os.path.join(DATA_DIR, 'CST')
s11_cst = CSTParser(os.path.join(CST_DATA_DIR, 'S11.txt'))
s21_cst = CSTParser(os.path.join(CST_DATA_DIR, 'S21.txt'))
s12_cst = CSTParser(os.path.join(CST_DATA_DIR, 'S12.txt'))
s22_cst = CSTParser(os.path.join(CST_DATA_DIR, 'S22.txt'))

for parser in (s11_cst, s21_cst, s12_cst, s22_cst):
    print(parser)
# =================================================================================================================================== #



# =================================================================================================================================== #
# -------------------------------------------------------------- PARSE FEKO ---------------------------------------------------------- #
FEKO_DATA_DIR = os.path.join(DATA_DIR, 'FEKO')
feko = FekoParser(os.path.join(FEKO_DATA_DIR, 's_parameters.s2p'))
# =================================================================================================================================== #



# =================================================================================================================================== #
# ------------------------------------------------------- PLOT CST SIMULATION ------------------------------------------------------- #
plotter = SimulationPlotter(OUTPUT_DIR)
# S11 & S22
plotter.plot_s_parameters(
    {'S11': s11_cst, 'S22': s22_cst},
    display='magnitude',
    linestyles=['-', '--', '-.', ':'],
    linewidth=3.0,
    xlim=(0.75, 1.0),
    ylim_mag=(-35, 0),
    title='Simulated S-Parameters (CST)',
    filename='Simulated_S_parameters_CST.pdf',
)

# With SimulationPlotter (sim-only)
plotter.plot_s_parameters(
    parsers=feko.get_parsers(['S11', 'S21']), display='magnitude',     linestyles=['-', '--', '-.', ':'],
    linewidth=3.0,
    xlim=(0.75, 1.0),
    ylim_mag=(-35, 0),
    title='Simulated S-Parameters',
    filename='Simulated_S_parameters_FEKO.pdf',
)
# =================================================================================================================================== #


# =================================================================================================================================== #
# --------------------------------------------------- PARSE MEASUREMENTS ------------------------------------------------------------ #
s2p = S2PParser(
    file_fwd=os.path.join(MEAS_DIR, 's11-s21.s2p'),   # port-1 sweep  → S11, S21
    file_rev=os.path.join(MEAS_DIR, 's12-s22.s2p'),   # port-2 sweep  → S22, S12
)
print(s2p)
# =================================================================================================================================== #


# =================================================================================================================================== #
# --------------------------------------------------- PLOT MEASUREMENTS ------------------------------------------------------------- #
meas = MeasurementPlotter(MEAS_OUT)

# S11 magnitude with minimum annotation
meas.plot_s_parameters(
    s2p,
    params=['S11', 'S22'],
    display='magnitude',
    show_minimum=True,
    linewidth=2.5,
    xlim=(0.75, 1.0),
    ylim_mag=(-35, 0),
    title=r'Measured S-Parameters',
    filename='Measured_S_parameters.pdf',
)
# =================================================================================================================================== #


# =================================================================================================================================== #
# --------------------------------------------------- SIMULATION VS. MEASUREMENT ---------------------------------------------------- #
final = FinalPlotter(os.path.join(_ROOT, 'output', '2026-05-14', 'final'))

# S11 — magnitude, sim + meas overlay, annotate measurement minimum
final.plot_s_parameters(
    params=['S11', 'S22'],
    sim_parsers={'S11': s11_cst, 'S22': s22_cst},
    meas_s2p=s2p,
    display='magnitude',
    linewidth=2.5,
    sim_colors=['#1f77b4', '#d62728'],   # S11 sim blue, S22 sim red
    meas_colors=['#aec7e8', '#f4a3a3'],  # S11 meas light-blue, S22 meas light-red
    sim_linestyles=['-', '-'],
    meas_linestyles=['--', ':'],
    show_minimum_meas=True,
    xlim=(0.75, 1.0),
    ylim_mag=(-35, 0),
    title=r'Simulation vs.\ Measurement (CST)',
    filename='sim_vs_meas_magnitude_cst.pdf',
)

# S11 — magnitude, sim + meas overlay, annotate measurement minimum
final.plot_s_parameters(
    params=['S11', 'S22'],
    sim_parsers=feko.get_parsers(['S11', 'S21']),
    meas_s2p=s2p,
    display='magnitude',
    linewidth=2.5,
    sim_colors=['#1f77b4', '#d62728'],   # S11 sim blue, S22 sim red
    meas_colors=['#aec7e8', '#f4a3a3'],  # S11 meas light-blue, S22 meas light-red
    sim_linestyles=['-', '-'],
    meas_linestyles=['--', ':'],
    show_minimum_meas=True,
    xlim=(0.75, 1.0),
    ylim_mag=(-35, 0),
    title=r'Simulation vs.\ Measurement (FEKO)',
    filename='sim_vs_meas_magnitude_feko.pdf',
)
# =================================================================================================================================== #

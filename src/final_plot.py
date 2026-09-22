# =================================================================================================================================== #
# ----------------------------------------------------------- DESCRIPTION ----------------------------------------------------------- #
# FinalPlotter: overlay CST simulation and NanoVNA measurement S-parameters on the same axes.                                         #
# Implements the standard sim-vs-meas comparison: same colour per parameter, solid = simulation, dashed = measurement.                #
# All figures are saved as PDF at 600 dpi to a configurable output directory.                                                         #
# Author: Nedal M. Benelmekki                                                                                                         #
# =================================================================================================================================== #



# =================================================================================================================================== #
# --------------------------------------------------------- EXTERNAL IMPORTS -------------------------------------------------------- #
import os                        # Operating system interfaces.                                                                       #
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
    TITLE_SIZE, LABEL_SIZE, LEGEND_SIZE, TICK_SIZE, ANNOTATION_SIZE,                                                                  #
    FIG_WIDTH, FIG_HEIGHT, LINE_WIDTH,                                                                                                #
    apply_minor_grid,                                                                                                                 #
)                                                                                                                                     #
from src.parse_cst import CSTParser                                                                                                   #
from src.parse_measurements import S2PParser                                                                                          #
# =================================================================================================================================== #



# =================================================================================================================================== #
# ------------------------------------------------------- MODULE-LEVEL HELPERS ------------------------------------------------------ #
def _to_tex(param: str) -> str:
    """
    Convert 'S11', 'S21', etc. to LaTeX notation 'S_{11}', 'S_{21}'.

    Parameters:
        - param [str] : S-parameter name, e.g. 'S11', 'S21', etc.

    Returns:
        - str : LaTeX-formatted parameter name, e.g. 'S_{11}', 'S_{21}'.
    """
    if param.startswith('S') and len(param) == 3:
        return rf'S_{{{param[1]}{param[2]}}}'
    return param


def _pass_number(mesh_label: str, fallback: str = '?') -> str:
    """
    Extract the pass number string from a mesh-pass curve label.
    
    Parameters:
        - mesh_label [str] : The label string from which to extract the pass number.
        - fallback   [str] : The string to return if the pass number cannot be extracted.

    Returns:
        - str : The extracted pass number, or the fallback if extraction fails.
    """
    if 'Mesh Pass=' in mesh_label:
        return mesh_label.split('Mesh Pass=')[-1].rstrip(')')
    return fallback
# =================================================================================================================================== #



# =================================================================================================================================== #
# ------------------------------------------------------- FINAL PLOTTER CLASS ------------------------------------------------------- #
class FinalPlotter:
    """
    Overlay CST simulation and NanoVNA measurement S-parameters on shared axes
    for publication-ready sim-vs-meas comparison plots.

    Convention
    ----------
    * Same colour per parameter across simulation and measurement curves.
    * Simulation  → ``sim_linestyle``  (default: solid ``'-'``).
    * Measurement → ``meas_linestyle`` (default: dashed ``'--'``).
    * Mesh-pass simulation curves are rendered as lighter background lines.

    Either ``sim_parsers`` or ``meas_s2p`` (or both) may be ``None``, allowing
    the class to produce simulation-only or measurement-only plots through the
    same interface.

    Parameters
    ----------
    output_dir : str
        Directory where all figures are written (created automatically if absent).
    """

    _DEFAULT_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e',
                       '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    _MESH_ALPHA     = [0.35, 0.55]
    _MESH_LS        = ['--', '-.']

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ──────────────────────────────────────────────────── private helpers ─────
    def _save(self, fig: plt.Figure, filename: str) -> None:
        """
        Save figure as a 600 dpi PDF and close it.
        
        Parameters:
            - fig      [plt.Figure] : The Matplotlib figure to save.
            - filename [str]        : The name of the output file (relative to self.output_dir).
        
        Returns:
            - None
        """
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=600, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved -> {path}')

    @staticmethod
    def _annotate_minimum(
        ax:     plt.Axes,
        freq:   np.ndarray,
        values: np.ndarray,
        color:  str,
    ) -> None:
        """
        Mark the minimum of values with a dotted line, a marker, and a text box.

        Parameters:
            - ax     [plt.Axes]   : The axes on which to draw the annotation.
            - freq   [np.ndarray] : Array of frequency values corresponding to the values.
            - values [np.ndarray] : Array of values (e.g. magnitude in dB) from which to find the minimum.
            - color  [str]        : Color to use for the annotation elements (line, marker, text box border).

        Returns:
            - None
        """
        idx   = int(np.argmin(values))
        f_min = float(freq[idx])
        v_min = float(values[idx])
        ax.axvline(f_min, color=color, linestyle=':', linewidth=1.2, alpha=0.7)
        ax.plot(f_min, v_min, marker='v', markersize=8, color=color, zorder=5,
                markeredgewidth=1.0)
        ax.annotate(
            rf'{v_min:.2f}\,\textrm{{dB}}' + '\n' + rf'$f={f_min:.4f}$\,GHz',
            xy=(f_min, v_min),
            xytext=(10, 12),
            textcoords='offset points',
            fontsize=ANNOTATION_SIZE,
            color=color,
            ha='left',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=color, alpha=0.75),
        )

    def _style_ax(
        self,
        ax:     plt.Axes,
        xlabel: str,
        ylabel: str,
        title:  str | None = None,
        xlim                = None,
        ylim                = None,
    ) -> None:
        """
        Apply uniform axis styling.
        
        Parameters:
            - ax     [plt.Axes]   : The axes to style.
            - xlabel [str]        : Label for the x-axis.
            - ylabel [str]        : Label for the y-axis.
            - title  [str|None]   : Title for the axes (optional).
            - xlim   [tuple|None] : Limits for the x-axis as (min, max) (optional).
            - ylim   [tuple|None] : Limits for the y-axis as (min, max) (optional).
        
        Returns:
            - None
        """
        ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
        if title:
            ax.set_title(title, fontsize=TITLE_SIZE)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.legend(fontsize=LEGEND_SIZE, loc='best')
        apply_minor_grid(ax)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

    # ─────────────────────────────────────────────── public plotting API ─────
    def plot_s_parameters(
        self,
        params:            list,
        sim_parsers:       dict | None      = None,
        meas_s2p:          S2PParser | None = None,
        display:           str              = 'magnitude',
        show_mesh:         bool             = False,
        linewidth:         float            = LINE_WIDTH,
        sim_linestyle:     str              = '-',
        meas_linestyle:    str              = '--',
        colors:            list | None      = None,
        sim_colors:        list | None      = None,
        meas_colors:       list | None      = None,
        sim_linestyles:    list | None      = None,
        meas_linestyles:   list | None      = None,
        show_minimum_sim:  bool             = False,
        show_minimum_meas: bool             = False,
        xlim                               = None,
        ylim_mag                           = None,
        ylim_phase                         = None,
        title:             str | None       = None,
        filename:          str              = 'sim_vs_meas.pdf',
    ) -> None:
        """
        Overlay simulation and measurement S-parameter curves on shared axes.

        Parameters:
            - params             : S-parameters to include, e.g. ``['S11', 'S21']``. Determines colour assignment — index in *params* maps to a colour regardless of whether sim, meas, or both are available for that parameter.
            - sim_parsers        : ``{'S11': CSTParser, ...}`` — simulation data. Pass ``None`` to omit simulation curves entirely.
            - meas_s2p           : ``S2PParser`` — measurement data. Pass ``None`` to omit measurement curves entirely.
            - display            : ``'magnitude'``, ``'phase'``, or ``'both'``.
            - show_mesh          : Draw mesh-adaptation-pass curves from simulation as lighter background lines when True.
            - linewidth          : Base line width for all final (converged) curves.
            - sim_linestyle      : Matplotlib line style applied to simulation curves. Default is solid ``'-'``.
            - meas_linestyle     : Matplotlib line style applied to measurement curves. Default is dashed ``'--'``.
            - colors             : List of colours, one per entry in *params*. Used as fallback when *sim_colors* / *meas_colors* are not given.
            - sim_colors         : Per-param colours for simulation curves. Overrides *colors* for sim when provided.
            - meas_colors        : Per-param colours for measurement curves. Overrides *colors* for meas when provided.
            - sim_linestyles     : Per-param line styles for simulation curves (list). Overrides the scalar *sim_linestyle* when provided.
            - meas_linestyles    : Per-param line styles for measurement curves (list). Overrides the scalar *meas_linestyle* when provided.
            - show_minimum_sim   : Annotate the deepest point of each simulation  magnitude curve with a frequency and dB label.
            - show_minimum_meas  : Annotate the deepest point of each measurement magnitude curve.
            - xlim               : ``(f_min, f_max)`` frequency axis limits in GHz.
            - ylim_mag           : ``(y_min, y_max)`` magnitude axis limits in dB.
            - ylim_phase         : ``(y_min, y_max)`` phase axis limits in degrees.
            - title              : Figure suptitle (``'both'``) or single-axes title.
            - filename           : Output file name relative to ``self.output_dir``.
        """
        if display not in ('magnitude', 'phase', 'both'):
            raise ValueError("display must be 'magnitude', 'phase', or 'both'")

        colors_use = colors or self._DEFAULT_COLORS

        def _pick(lst, scalar, idx):
            if lst is not None:
                return lst[idx % len(lst)]
            return scalar

        # ── figure layout ─────────────────────────────────────────────────────
        if display == 'both':
            fig, (ax_mag, ax_ph) = plt.subplots(
                2, 1,
                figsize=(FIG_WIDTH, FIG_HEIGHT * 1.6),
                sharex=True,
                constrained_layout=True,
            )
        else:
            fig, ax_single = plt.subplots(
                figsize=(FIG_WIDTH, FIG_HEIGHT),
                constrained_layout=True,
            )
            ax_mag = ax_single if display == 'magnitude' else None
            ax_ph  = ax_single if display == 'phase'     else None

        # ── draw one parameter at a time ──────────────────────────────────────
        for i, param in enumerate(params):
            base_color  = colors_use[i % len(colors_use)]
            s_color     = _pick(sim_colors,  base_color,      i)
            m_color     = _pick(meas_colors, base_color,      i)
            s_linestyle = _pick(sim_linestyles,  sim_linestyle,  i)
            m_linestyle = _pick(meas_linestyles, meas_linestyle, i)
            tex         = _to_tex(param)

            # ── simulation curves ──────────────────────────────────────────────
            if sim_parsers is not None and param in sim_parsers:
                parser = sim_parsers[param]

                if show_mesh:
                    for j, lbl in enumerate(parser.mesh_labels):
                        slot = j % len(self._MESH_ALPHA)
                        kw   = dict(
                            color=s_color,
                            linestyle=self._MESH_LS[slot],
                            linewidth=linewidth * 0.65,
                            alpha=self._MESH_ALPHA[slot],
                        )
                        mesh_label = (
                            f'${tex}$ (Sim, pass {_pass_number(lbl, str(j + 1))})'
                        )
                        df = parser.get_df(lbl)
                        if df.empty:
                            continue
                        if ax_mag is not None:
                            ax_mag.plot(df['freq_GHz'], df['mag_dB'],
                                        label=mesh_label, **kw)
                        if ax_ph is not None:
                            ax_ph.plot(df['freq_GHz'], df['phase_deg'],
                                       label=mesh_label, **kw)

                sim_kw = dict(color=s_color, linestyle=s_linestyle,
                              linewidth=linewidth, alpha=1.0)
                for lbl in parser.final_labels:
                    df = parser.get_df(lbl)
                    if df.empty:
                        continue
                    sim_curve_label = f'${tex}$ (Sim)'
                    if ax_mag is not None:
                        ax_mag.plot(df['freq_GHz'], df['mag_dB'],
                                    label=sim_curve_label, **sim_kw)
                        if show_minimum_sim:
                            self._annotate_minimum(
                                ax_mag,
                                df['freq_GHz'].values,
                                df['mag_dB'].values,
                                s_color,
                            )
                    if ax_ph is not None:
                        ax_ph.plot(df['freq_GHz'], df['phase_deg'],
                                   label=sim_curve_label, **sim_kw)

            # ── measurement curve ──────────────────────────────────────────────
            if meas_s2p is not None and param in meas_s2p.params:
                df      = meas_s2p.get_df(param)
                meas_kw = dict(color=m_color, linestyle=m_linestyle,
                               linewidth=linewidth, alpha=1.0)
                if not df.empty:
                    meas_label = f'${tex}$ (Meas)'
                    if ax_mag is not None:
                        ax_mag.plot(df['freq_GHz'], df['mag_dB'],
                                    label=meas_label, **meas_kw)
                        if show_minimum_meas:
                            self._annotate_minimum(
                                ax_mag,
                                df['freq_GHz'].values,
                                df['mag_dB'].values,
                                m_color,
                            )
                    if ax_ph is not None:
                        ax_ph.plot(df['freq_GHz'], df['phase_deg'],
                                   label=meas_label, **meas_kw)

        # ── axis styling ──────────────────────────────────────────────────────
        if ax_mag is not None:
            self._style_ax(
                ax_mag,
                xlabel=r'Frequency [GHz]',
                ylabel=r'Magnitude [dB]',
                title=title if display == 'magnitude' else None,
                xlim=xlim, ylim=ylim_mag,
            )

        if ax_ph is not None:
            self._style_ax(
                ax_ph,
                xlabel=r'Frequency [GHz]',
                ylabel=r'Phase [$^\circ$]',
                title=title if display == 'phase' else None,
                xlim=xlim, ylim=ylim_phase,
            )

        if display == 'both' and title:
            fig.suptitle(title, fontsize=TITLE_SIZE + 2)

        self._save(fig, filename)

    # ─────────────────────────────────────────────────────── Smith chart ─────
    def plot_smith_chart(
        self,
        params:          list,
        sim_parsers:     dict | None      = None,
        meas_s2p:        S2PParser | None = None,
        show_mesh:       bool             = False,
        linewidth:       float            = LINE_WIDTH,
        sim_linestyle:   str              = '-',
        meas_linestyle:  str              = '--',
        colors:          list | None      = None,
        sim_colors:      list | None      = None,
        meas_colors:     list | None      = None,
        sim_linestyles:  list | None      = None,
        meas_linestyles: list | None      = None,
        title:           str | None       = None,
        filename:        str              = 'smith_chart_sim_vs_meas.pdf',
    ) -> None:
        """
        Overlay simulation and measurement S-parameters on a Smith chart.

        Parameters:
            - params          : S-parameters to include, e.g. ``['S11']``.
            - sim_parsers     : ``{'S11': CSTParser, ...}`` — simulation data.
            - meas_s2p        : ``S2PParser`` — measurement data.
            - show_mesh       : Include simulation mesh-pass curves when True.
            - linewidth       : Line width for all final curves.
            - sim_linestyle   : Line style for simulation curves.
            - meas_linestyle  : Line style for measurement curves.
            - colors          : List of colours, one per entry in *params*.
            - title           : Figure title.
            - filename        : Output file name relative to ``self.output_dir``.

        Returns:
            - None
        """
        colors_use = colors or self._DEFAULT_COLORS

        def _pick(lst, scalar, idx):
            if lst is not None:
                return lst[idx % len(lst)]
            return scalar

        fig, ax    = plt.subplots(
            figsize=(FIG_WIDTH, FIG_WIDTH),
            constrained_layout=True,
        )

        for i, param in enumerate(params):
            base_color  = colors_use[i % len(colors_use)]
            s_color     = _pick(sim_colors,  base_color,     i)
            m_color     = _pick(meas_colors, base_color,     i)
            s_linestyle = _pick(sim_linestyles,  sim_linestyle,  i)
            m_linestyle = _pick(meas_linestyles, meas_linestyle, i)
            tex         = _to_tex(param)

            # ── simulation curves ──────────────────────────────────────────────
            if sim_parsers is not None and param in sim_parsers:
                parser = sim_parsers[param]

                if show_mesh:
                    for j, lbl in enumerate(parser.mesh_labels):
                        slot = j % len(self._MESH_ALPHA)
                        kw   = dict(
                            color=s_color,
                            linestyle=self._MESH_LS[slot],
                            linewidth=linewidth * 0.65,
                            alpha=self._MESH_ALPHA[slot],
                        )
                        df = parser.get_df(lbl)
                        if df.empty:
                            continue
                        freq_hz = df['freq_GHz'].values * 1e9
                        s_cpx   = (df['real'].values + 1j * df['imag'].values).reshape(-1, 1, 1)
                        ntwk    = rf.Network(
                            frequency=rf.Frequency.from_f(freq_hz, unit='hz'),
                            s=s_cpx,
                        )
                        ntwk.plot_s_smith(
                            ax=ax, m=0, n=0,
                            label=f'${tex}$ (Sim, pass {_pass_number(lbl, str(j + 1))})',
                            **kw,
                        )

                sim_kw = dict(color=s_color, linestyle=s_linestyle,
                              linewidth=linewidth, alpha=1.0)
                for lbl in parser.final_labels:
                    df = parser.get_df(lbl)
                    if df.empty:
                        continue
                    freq_hz = df['freq_GHz'].values * 1e9
                    s_cpx   = (df['real'].values + 1j * df['imag'].values).reshape(-1, 1, 1)
                    ntwk    = rf.Network(
                        frequency=rf.Frequency.from_f(freq_hz, unit='hz'),
                        s=s_cpx,
                    )
                    ntwk.plot_s_smith(ax=ax, m=0, n=0,
                                      label=f'${tex}$ (Sim)', **sim_kw)

            # ── measurement curve ──────────────────────────────────────────────
            if meas_s2p is not None and param in meas_s2p.params:
                df = meas_s2p.get_df(param)
                if not df.empty:
                    freq_hz  = df['freq_GHz'].values * 1e9
                    s_cpx    = (df['real'].values + 1j * df['imag'].values).reshape(-1, 1, 1)
                    ntwk     = rf.Network(
                        frequency=rf.Frequency.from_f(freq_hz, unit='hz'),
                        s=s_cpx,
                    )
                    meas_kw  = dict(color=m_color, linestyle=m_linestyle,
                                    linewidth=linewidth, alpha=1.0)
                    ntwk.plot_s_smith(ax=ax, m=0, n=0,
                                      label=f'${tex}$ (Meas)', **meas_kw)

        if title:
            ax.set_title(title, fontsize=TITLE_SIZE)
        ax.legend(fontsize=LEGEND_SIZE, loc='best')
        self._save(fig, filename)
# =================================================================================================================================== #

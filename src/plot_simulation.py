# =================================================================================================================================== #
# ----------------------------------------------------------- DESCRIPTION ----------------------------------------------------------- #
# SimulationPlotter: publication-quality S-parameter plots from CSTParser data.                                                       #
# Outputs: 1-D magnitude / phase plots, combined magnitude+phase, and Smith chart.                                                    #
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
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)                  # Project root = parent of src/.                                        #
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
# =================================================================================================================================== #



# =================================================================================================================================== #
# ------------------------------------------------------- MODULE-LEVEL HELPERS ------------------------------------------------------ #
def _to_tex(param: str) -> str:
    """
    Convert 'S11', 'S21', etc. to LaTeX notation 'S_{11}', 'S_{21}'.
    
    Parameters:
        - param [str] : S-parameter label, e.g. 'S11', 'S21', etc.
    
    Returns:
        - str : LaTeX-formatted string, e.g. 'S_{11}', 'S_{21}', etc.
    """
    if param.startswith('S') and len(param) == 3:
        return rf'S_{{{param[1]}{param[2]}}}'
    return param


def _pass_number(mesh_label: str, fallback: str = '?') -> str:
    """
    Extract the pass number string from a mesh-pass curve label.

    Parameters:
        - mesh_label [str] : The label of the mesh-pass curve, expected to contain 'Mesh Pass='.
        - fallback   [str] : The string to return if 'Mesh Pass=' is not found in the label.

    Returns:
        - str : The extracted pass number as a string, or the fallback if not found.
    """
    if 'Mesh Pass=' in mesh_label:
        return mesh_label.split('Mesh Pass=')[-1].rstrip(')')
    return fallback
# =================================================================================================================================== #



# =================================================================================================================================== #
# --------------------------------------------------- SIMULATION PLOTTER CLASS ------------------------------------------------------ #
class SimulationPlotter:
    """
    Produce publication-quality plots from parsed CST S-parameter data.

    Parameters:
        - output_dir [str] : Directory where all figures are written (created automatically if absent).
    """

    _DEFAULT_COLORS     = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e',
                            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    _DEFAULT_LINESTYLES = ['-', '--', '-.', ':']

    # ── mesh-pass curve defaults (two slots; cycles if more passes exist) ─────
    _MESH_ALPHA    = [0.35, 0.55]
    _MESH_LS       = ['--', '-.']

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ──────────────────────────────────────────────────── private helpers ─────
    def _save(self, fig: plt.Figure, filename: str) -> None:
        """
        Save figure as a 600 dpi PDF and close it.
        
        Parameters:
            - fig      [Figure] : The Matplotlib figure to save.
            - filename [str]    : The name of the output file relative to self.output_dir.

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
        Mark the minimum of values with a dashed vertical line, a marker, and a text box.

        Parameters:
            - ax     [Axes]     : The Matplotlib axes to annotate.
            - freq   [ndarray]  : Array of frequency values corresponding to the input values.
            - values  [ndarray] : Array of values (e.g., magnitude in dB) from which to find the minimum.
            - color   [str]     : Color for the annotation elements.
        
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
            - ax     [Axes]       : The Matplotlib axes to style.
            - xlabel [str]        : Label for the x-axis.
            - ylabel [str]        : Label for the y-axis.
            - title  [str|None]   : Title for the axes (optional).
            - xlim   (tuple|None) : Limits for the x-axis (optional).
            - ylim   (tuple|None) : Limits for the y-axis (optional).

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
        parsers:      dict,
        display:      str         = 'magnitude',
        show_mesh:    bool        = False,
        linewidth:    float       = LINE_WIDTH,
        linestyles:   list | None = None,
        colors:       list | None = None,
        show_minimum: bool        = False,
        xlim                      = None,
        ylim_mag                  = None,
        ylim_phase                = None,
        title:        str | None  = None,
        filename:     str         = 'S_parameters.pdf',
    ) -> None:
        """
        Plot S-parameter magnitude (dB), phase (degrees), or both.

        Parameters:
            - parsers      : ``{'S11': CSTParser, 'S21': CSTParser, ...}``. One entry per S-parameter to include in the figure.
            - display      : ``'magnitude'``, ``'phase'``, or ``'both'``. ``'both'`` creates two vertically stacked subplots with a shared frequency axis.
            - show_mesh    : When True, mesh-adaptation-pass curves are drawn behind the final (converged) curve as lighter dashed lines.
            - linewidth    : Line width applied to all converged curves.
            - linestyles   : List of line styles, one per entry in *parsers*. Cycles back to the start when the list is exhausted. Defaults to ``['-', '--', '-.', ':']``.
            - colors       : List of colours, one per entry in *parsers*.  Cycles if shorter than the number of parameters.
            - show_minimum : When True, annotate the deepest point of each converged magnitude curve with a marker and a dB / frequency label.
            - xlim         : ``(f_min, f_max)`` frequency axis limits in GHz, or ``None`` for auto-scaling.
            - ylim_mag     : ``(y_min, y_max)`` magnitude axis limits in dB.
            - ylim_phase   : ``(y_min, y_max)`` phase axis limits in degrees.
            - title        : Figure suptitle (``'both'``) or single-axes title.
            - filename     : Output file name relative to ``self.output_dir``.

        Returns:
            - None
        """
        if display not in ('magnitude', 'phase', 'both'):
            raise ValueError("display must be 'magnitude', 'phase', or 'both'")

        colors_use     = colors    or self._DEFAULT_COLORS
        linestyles_use = linestyles or self._DEFAULT_LINESTYLES

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

        # ── draw curves ───────────────────────────────────────────────────────
        for i, (param, parser) in enumerate(parsers.items()):
            color    = colors_use[i % len(colors_use)]
            ls_final = linestyles_use[i % len(linestyles_use)]
            tex      = _to_tex(param)
            final_kw = dict(linestyle=ls_final, linewidth=linewidth,
                            alpha=1.0, color=color)

            # Mesh passes first so they render below the final curve.
            curve_groups: list[tuple] = []
            if show_mesh:
                for j, lbl in enumerate(parser.mesh_labels):
                    slot = j % len(self._MESH_ALPHA)
                    kw   = dict(
                        color=color,
                        linestyle=self._MESH_LS[slot],
                        linewidth=linewidth * 0.65,
                        alpha=self._MESH_ALPHA[slot],
                    )
                    curve_groups.append(
                        (lbl, parser.get_df(lbl), kw,
                         f'${tex}$ (pass {_pass_number(lbl, str(j + 1))})')
                    )

            for lbl in parser.final_labels:
                curve_groups.append((lbl, parser.get_df(lbl), final_kw, f'${tex}$'))

            for lbl, df, kw, curve_label in curve_groups:
                if df.empty:
                    continue
                is_final = 'Mesh Pass' not in lbl

                if ax_mag is not None:
                    ax_mag.plot(df['freq_GHz'], df['mag_dB'],
                                label=curve_label, **kw)
                    if show_minimum and is_final:
                        self._annotate_minimum(
                            ax_mag,
                            df['freq_GHz'].values,
                            df['mag_dB'].values,
                            color,
                        )

                if ax_ph is not None:
                    ax_ph.plot(df['freq_GHz'], df['phase_deg'],
                               label=curve_label, **kw)

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
        parsers:   dict,
        show_mesh: bool        = False,
        colors:    list | None = None,
        linewidth: float       = LINE_WIDTH,
        title:     str | None  = None,
        filename:  str         = 'smith_chart.pdf',
    ) -> None:
        """
        Plot one or more S-parameters on a Smith chart using scikit-rf.

        The chart is normalised to 50 Ohm by default (scikit-rf convention).

        Parameters:
            - parsers   : ``{'S11': CSTParser, ...}``
            - show_mesh : Include mesh-pass curves when True.
            - colors    : List of colours, one per entry in *parsers*.
            - linewidth : Line width for converged curves.
            - title     : Figure title.
            - filename  : Output file name relative to ``self.output_dir``.

        Returns:
            - None
        """
        colors_use = colors or self._DEFAULT_COLORS
        fig, ax    = plt.subplots(
            figsize=(FIG_WIDTH, FIG_WIDTH),   # square canvas for Smith chart
            constrained_layout=True,
        )

        for i, (param, parser) in enumerate(parsers.items()):
            color = colors_use[i % len(colors_use)]
            tex   = _to_tex(param)

            curve_groups: list[tuple] = []
            if show_mesh:
                for j, lbl in enumerate(parser.mesh_labels):
                    slot = j % len(self._MESH_ALPHA)
                    kw   = dict(
                        color=color,
                        linestyle=self._MESH_LS[slot],
                        linewidth=linewidth * 0.65,
                        alpha=self._MESH_ALPHA[slot],
                    )
                    curve_groups.append(
                        (lbl, parser.get_df(lbl), kw,
                         f'${tex}$ (pass {_pass_number(lbl, str(j + 1))})')
                    )

            for lbl in parser.final_labels:
                curve_groups.append(
                    (lbl, parser.get_df(lbl),
                     dict(color=color, linewidth=linewidth, alpha=1.0),
                     f'${tex}$')
                )

            for lbl, df, kw, curve_label in curve_groups:
                if df.empty:
                    continue
                freq_hz = df['freq_GHz'].values * 1e9
                s_cpx   = (df['real'].values + 1j * df['imag'].values).reshape(-1, 1, 1)
                freq    = rf.Frequency.from_f(freq_hz, unit='hz')
                ntwk    = rf.Network(frequency=freq, s=s_cpx)
                ntwk.plot_s_smith(ax=ax, m=0, n=0, label=curve_label, **kw)

        if title:
            ax.set_title(title, fontsize=TITLE_SIZE)
        ax.legend(fontsize=LEGEND_SIZE, loc='best')
        self._save(fig, filename)
# =================================================================================================================================== #

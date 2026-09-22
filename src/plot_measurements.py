# =================================================================================================================================== #
# ----------------------------------------------------------- DESCRIPTION ----------------------------------------------------------- #
# MeasurementPlotter: publication-quality S-parameter plots from S2PParser data.                                                      #
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
from scipy.signal        import savgol_filter  # Savitzky-Golay smoothing of noisy sweeps.                                            #
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
    FIG_WIDTH, FIG_HEIGHT, LINE_WIDTH, MARKER_SIZE,                                                                                   #
    apply_minor_grid, shade_band,                                                                                                     #
)                                                                                                                                     #
from src.parse_measurements import S2PParser                                                                                          #
# =================================================================================================================================== #



# =================================================================================================================================== #
# ------------------------------------------------------- MODULE-LEVEL HELPERS ------------------------------------------------------ #
def _to_tex(param: str) -> str:
    """
    Convert 'S11', 'S21', etc. to LaTeX notation 'S_{11}', 'S_{21}'.
    
    Parameters:
        - param [str] : S-parameter name, e.g. 'S11', 'S21', 'S12', 'S22'.

    Returns:
        - str : LaTeX-formatted parameter name, e.g. 'S_{11}', 'S_{21}', 'S_{12}', 'S_{22}'. 
    """
    if param.startswith('S') and len(param) == 3:
        return rf'S_{{{param[1]}{param[2]}}}'
    return param
# =================================================================================================================================== #



# =================================================================================================================================== #
# -------------------------------------------------- MEASUREMENT PLOTTER CLASS ------------------------------------------------------ #
class MeasurementPlotter:
    """
    Produce publication-quality plots from NanoVNA S-parameter measurement data.

    Parameters:
        - output_dir [str] : Directory where all figures are written (created automatically if absent).
    """

    _DEFAULT_COLORS     = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e',
                            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    _DEFAULT_LINESTYLES = ['-', '--', '-.', ':']

    def __init__(self, output_dir: str) -> None:
        """
        Class constructor. Creates the output directory if it does not exist.

        Parameters:
            - output_dir [str] : Directory where all figures are written (created automatically if absent).

        Returns:
            - None
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ──────────────────────────────────────────────────── private helpers ─────
    def _save(self, fig: plt.Figure, filename: str) -> None:
        """
        Save a figure as a 600 dpi PDF and close it.

        Parameters:
            - fig [plt.Figure] : The figure to save.
            - filename [str]   : Output file name relative to self.output_dir.

        Returns:
            - None
        """
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=600, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved -> {path}')

    @staticmethod
    def _annotate_minimum(ax: plt.Axes, freq: np.ndarray, values: np.ndarray, color: str,) -> None:
        """
        Mark the minimum of values with a dashed vertical line, a marker, and a text box.
        
        Parameters:
            - ax     [plt.Axes]   : The axes to annotate.
            - freq   [np.ndarray] : Frequency array corresponding to the values.
            - values [np.ndarray] : Values array (e.g. magnitude in dB) to find the minimum of.
            - color  [str]        : Color for the annotation elements.

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

    @staticmethod
    def _smooth(y: np.ndarray, window: int | None) -> np.ndarray:
        """
        Apply a Savitzky-Golay low-pass to a noisy trace, preserving notch shape.

        The window is coerced to an odd integer no larger than the series; a
        cubic polynomial is fitted in each window.  ``window`` of ``None`` or
        < 3 returns the input untouched.

        Parameters:
            - y      [np.ndarray] : Values to smooth (e.g. magnitude in dB).
            - window [int | None] : Filter length in samples.

        Returns:
            - np.ndarray : Smoothed values (same length as ``y``).
        """
        if not window:
            return y
        w = int(window)
        if w % 2 == 0:
            w += 1
        if w > len(y):
            w = len(y) if len(y) % 2 == 1 else len(y) - 1
        if w < 3:
            return y
        return savgol_filter(y, w, min(3, w - 1))

    @staticmethod
    def _marker_indices(
        freq:     np.ndarray,
        xlim,
        n_markers: int,
        trace:     int,
        n_traces:  int,
    ) -> list | None:
        """
        Choose ``n_markers`` sample indices evenly spaced across the visible
        band, phase-staggered per trace so glyphs of overlapping curves do not
        coincide.

        Every trace receives exactly the same number of markers regardless of
        how the sweep is decimated, which keeps a multi-trace figure legible.

        Parameters:
            - freq      [np.ndarray] : Frequency axis of the trace.
            - xlim                   : ``(f_min, f_max)`` visible window, or ``None`` for the full sweep.
            - n_markers [int]        : Number of markers to place on the trace.
            - trace     [int]        : Index of this trace among the plotted set.
            - n_traces  [int]        : Total number of traces (for the stagger offset).

        Returns:
            - list | None : Sample indices for ``markevery``, or ``None`` if the window is empty.
        """
        lo, hi = xlim if xlim is not None else (float(freq.min()), float(freq.max()))
        inwin  = np.where((freq >= lo) & (freq <= hi))[0]
        if inwin.size == 0:
            return None
        offset = (trace + 0.5) / max(n_traces, 1)                # Stagger marker phase between traces.               #
        frac   = (np.arange(n_markers) + offset) / n_markers     # Evenly spaced fractional positions in the window.  #
        sel    = np.clip((frac * (inwin.size - 1)).astype(int), 0, inwin.size - 1)
        return inwin[sel].tolist()

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
            - xlabel [str]        : X-axis label.
            - ylabel [str]        : Y-axis label.
            - title  [str|None]   : Optional title for the axes.
            - xlim   [tuple|None] : Optional (min, max) limits for the x-axis.
            - ylim   [tuple|None] : Optional (min, max) limits for the y-axis.

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
        s2p:          S2PParser,
        params:       list       = ('S11', 'S21', 'S12', 'S22'),
        display:      str        = 'magnitude',
        linewidth:    float      = LINE_WIDTH,
        linestyles:   list | None = None,
        colors:       list | None = None,
        markers:      list | None = None,
        markevery                 = None,
        n_markers:    int | None  = None,
        markersize:   float        = MARKER_SIZE,
        smooth:       int | None  = None,
        show_minimum: bool        = False,
        xlim                      = None,
        band                      = None,
        ylim_mag                  = None,
        ylim_phase                = None,
        title:        str | None  = None,
        filename:     str         = 'S_parameters.pdf',
    ) -> None:
        """
        Plot S-parameter magnitude (dB), phase (degrees), or both.

        Parameters:
            - s2p          [S2PParser]   : Parsed measurement data from ``S2PParser``.
            - params       [list]        : Which S-parameters to include, e.g. ``['S11', 'S21']``.
            - display      [str]         : ``'magnitude'``, ``'phase'``, or ``'both'``. ``'both'`` creates two vertically stacked subplots with a shared frequency axis.
            - linewidth    [float]       : Line width for all curves.
            - linestyles   [list | None] : List of line styles, one per entry in params.  Cycles back to the start when the list is exhausted. Defaults to ``['-', '--', '-.', ':']``.
            - colors       [list | None] : List of colours, one per entry in params.  Cycles if shorter than the number of parameters.
            - markers      [list | None] : List of Matplotlib marker glyphs, one per entry in params (e.g. ``['o', 's', '^', 'D']``).  ``None`` disables markers (the default for dense sweeps).  Marker face and edge take the trace colour.
            - markevery    : Marker decimation passed to ``plot``.  An ``int`` or ``(start, step)`` tuple applies to every trace; a ``list`` applies one spec per parameter (index-matched).  Ignored when *n_markers* is given.
            - n_markers    [int | None]  : Place exactly this many markers per trace, evenly spaced across the visible band (*xlim*) and phase-staggered between traces so overlapping curves stay legible.  Overrides *markevery* when set.
            - markersize   [float]       : Marker size in points.
            - smooth       [int | None]  : Savitzky-Golay window (samples) applied to each magnitude / phase trace to suppress measurement ripple before plotting.  ``None`` leaves the raw sweep untouched.
            - show_minimum [bool]: When True, annotate the deepest point of each magnitude frequency label.
            - xlim         : ``(f_min, f_max)`` frequency axis limits in GHz.
            - band         : ``(f_lo, f_hi)`` in GHz — shade this interval with a faint grey wash behind the traces to mark an operating band (e.g. ``ETSI_UHF_BAND_GHZ``).  Excluded from the legend; declare it in the caption instead.  ``None`` draws nothing.
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

        # ── operating-band wash (drawn first; sits at zorder 0, under grid and traces) ──
        for _ax in (ax_mag, ax_ph):
            if _ax is not None:
                shade_band(_ax, band)

        # ── draw curves ───────────────────────────────────────────────────────
        n_traces = len(params)
        for i, param in enumerate(params):
            df    = s2p.get_df(param)
            if df.empty:
                continue
            color = colors_use[i % len(colors_use)]
            ls    = linestyles_use[i % len(linestyles_use)]
            tex   = _to_tex(param)
            label = f'${tex}$'

            freq  = df['freq_GHz'].values
            mag   = self._smooth(df['mag_dB'].values,    smooth)   # Raw sweep when smooth is None.                    #
            phse  = self._smooth(df['phase_deg'].values, smooth)

            kw = dict(linestyle=ls, linewidth=linewidth, alpha=1.0, color=color)

            # ── optional IEEE-style markers, coloured like the trace ───────────
            if markers is not None:
                if n_markers:
                    mev = self._marker_indices(freq, xlim, n_markers, i, n_traces)
                else:
                    mev = markevery[i % len(markevery)] if isinstance(markevery, list) else markevery
                kw.update(
                    marker=markers[i % len(markers)],
                    markevery=mev,
                    markersize=markersize,
                    markerfacecolor=color,
                    markeredgecolor=color,
                )

            if ax_mag is not None:
                ax_mag.plot(freq, mag, label=label, **kw)
                if show_minimum:
                    self._annotate_minimum(ax_mag, freq, mag, color)

            if ax_ph is not None:
                ax_ph.plot(freq, phse, label=label, **kw)

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

        # ── fix origin tick-label collision (-20 dB vs 0.80 GHz) ──────────────
        for ax in (ax_mag, ax_ph):
            if ax is not None:
                # Shift the first x-tick label (0.80) to align left of the tick mark
                xticks = ax.xaxis.get_major_ticks()
                if xticks:
                    xticks[0].label1.set_horizontalalignment('left')
                
                # Optional: Add a touch of extra padding to push labels clear of axes
                ax.tick_params(axis='both', pad=5)

        if display == 'both' and title:
            fig.suptitle(title, fontsize=TITLE_SIZE + 2)

        self._save(fig, filename)

    # ─────────────────────────────────────────────────────── Smith chart ─────
    def plot_smith_chart(
        self,
        s2p:       S2PParser,
        params:    list        = ('S11',),
        colors:    list | None = None,
        linewidth: float       = LINE_WIDTH,
        title:     str | None  = None,
        filename:  str         = 'smith_chart.pdf',
    ) -> None:
        """
        Plot one or more S-parameters on a Smith chart using scikit-rf.

        Parameters:
            - s2p       [S2PParser]   : Parsed measurement data from ``S2PParser``.
            - params    [list]        : Which S-parameters to plot, e.g. ``['S11', 'S22']``.
            - colors    [list | None] : List of colours, one per entry in *params*.
            - linewidth [float]       : Line width.
            - title     [str | None]  : Figure title.
            - filename  [str]         : Output file name relative to ``self.output_dir``.

        Returns:
            - None
        """
        colors_use = colors or self._DEFAULT_COLORS
        fig, ax    = plt.subplots(
            figsize=(FIG_WIDTH, FIG_WIDTH),
            constrained_layout=True,
        )

        for i, param in enumerate(params):
            df    = s2p.get_df(param)
            color = colors_use[i % len(colors_use)]
            tex   = _to_tex(param)

            if df.empty:
                continue

            freq_hz = df['freq_GHz'].values * 1e9
            s_cpx   = (df['real'].values + 1j * df['imag'].values).reshape(-1, 1, 1)
            freq    = rf.Frequency.from_f(freq_hz, unit='hz')
            ntwk    = rf.Network(frequency=freq, s=s_cpx)
            ntwk.plot_s_smith(
                ax=ax, m=0, n=0,
                label=f'${tex}$',
                color=color,
                linewidth=linewidth,
                alpha=1.0,
            )

        if title:
            ax.set_title(title, fontsize=TITLE_SIZE)
        ax.legend(fontsize=LEGEND_SIZE, loc='best')
        self._save(fig, filename)
# =================================================================================================================================== #

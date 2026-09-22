# =================================================================================================================================== #
# ----------------------------------------------------------- DESCRIPTION ----------------------------------------------------------- #
# OscillatorPlotter: publication-quality figures for the BAW oscillator design, built from ADSListParser data.                        #
# Covers: transistor IV / gain, resonator reflection (mBVD vs. measured), Randall-Hock open-loop gain, OscTest Nyquist               #
# criterion, transient output + spectrum, harmonic-balance steady state, and phase noise.                                            #
# Every figure is saved as both PDF and PNG at >= 600 dpi. Colour/linestyle/marker cycling is reserved for figures that overlay      #
# multiple traces; a lone trace is drawn as a plain solid line. Large figure-level titles are omitted - context (Zo, carrier          #
# frequency, marked points) is carried by small formal annotations instead, kept clear of the data with a leader line.               #
# Author: Nedal M. Benelmekki                                                                                                         #
# =================================================================================================================================== #



# =================================================================================================================================== #
# --------------------------------------------------------- EXTERNAL IMPORTS -------------------------------------------------------- #
import os                        # Operating system interfaces.                                                                       #
import sys                       # System-specific parameters and functions.                                                          #
import numpy             as np   # Numerical computing.                                                                               #
import matplotlib.pyplot as plt  # Plotting.                                                                                          #
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
    LABEL_SIZE, LEGEND_SIZE, TICK_SIZE, ANNOTATION_SIZE, SMALL_SIZE,                                                                  #
    FIG_WIDTH, FIG_HEIGHT, LINE_WIDTH, MARKER_SIZE,                                                                                   #
    COLOR_CYCLE, LINESTYLE_CYCLE, MARKER_CYCLE,                                                                                       #
    apply_minor_grid, zoom_rules,                                                                                                     #
)                                                                                                                                     #
# =================================================================================================================================== #



# =================================================================================================================================== #
# ------------------------------------------------------- MODULE-LEVEL HELPERS ------------------------------------------------------ #
def _markevery(n_points: int, n_markers: int = 11) -> int:
    """
    Compute a ``markevery`` stride so a dense curve carries roughly ``n_markers`` markers, evenly spaced
    along its length regardless of how many points were exported (accessibility: markers stay legible in
    black-and-white print / under colour-vision deficiency, without cluttering a 30k-point sweep).

    Parameters:
        - n_points  [int] : Number of points in the curve.
        - n_markers [int] : Target number of markers along the curve.

    Returns:
        - int : Stride to pass as ``markevery=``. Always >= 1.
    """
    return max(1, round(n_points / max(n_markers, 1)))


def _nearest_row(df, col: str, value: float):
    """
    Return the row of ``df`` whose ``col`` value is closest to ``value`` (for snapping a requested
    marker frequency/offset onto an actually-exported sample point).

    Parameters:
        - df    [pd.DataFrame] : Table to search.
        - col   [str]          : Column to match against.
        - value [float]        : Target value.

    Returns:
        - pd.Series : The closest row.
    """
    idx = (df[col] - value).abs().idxmin()
    return df.loc[idx]


def _find_zero_crossing(df, x_col: str, y_col: str, x_approx: float, window: float, wrap_jump: float = 300.0):
    """
    Locate the genuine zero-crossing of ``y_col`` nearest to ``x_approx`` (e.g. a phase trace crossing
    0 degrees), by linear interpolation between the two bracketing samples within +/- ``window`` of
    ``x_approx``. A phase trace also jumps from +180 to -180 at its wrap points, which is a sign change
    but not a real crossing; any step larger than ``wrap_jump`` is treated as a wrap and skipped.

    Parameters:
        - df        [pd.DataFrame] : Table to search (columns ``x_col``, ``y_col``).
        - x_col     [str]          : Independent-variable column (e.g. 'freq', in Hz).
        - y_col     [str]          : Dependent-variable column to zero-cross (e.g. phase, in degrees).
        - x_approx  [float]        : Approximate location to search around, same units as ``x_col``.
        - window    [float]        : Half-width of the search window around ``x_approx``, same units as ``x_col``.
        - wrap_jump [float]        : Step size above which a sign change is treated as a wrap-around, not
                                      a genuine crossing.

    Returns:
        - tuple[float, float] : (x, y) of the interpolated crossing (y ~= 0). Falls back to the nearest
                                 sample to ``x_approx`` within the window if no genuine crossing is found.
    """
    sub = df[(df[x_col] >= x_approx - window) & (df[x_col] <= x_approx + window)].sort_values(x_col)
    x = sub[x_col].to_numpy()
    y = sub[y_col].to_numpy()

    candidates = []
    for i in range(len(y) - 1):
        y0, y1 = y[i], y[i + 1]
        if np.isnan(y0) or np.isnan(y1) or abs(y1 - y0) > wrap_jump:
            continue
        if y0 == 0:
            candidates.append((x[i], 0.0))
        elif y0 * y1 < 0:
            t = -y0 / (y1 - y0)
            candidates.append((x[i] + t * (x[i + 1] - x[i]), 0.0))

    if not candidates:
        row = _nearest_row(sub, x_col, x_approx) if len(sub) else _nearest_row(df, x_col, x_approx)
        return float(row[x_col]), float(row[y_col])

    return min(candidates, key=lambda c: abs(c[0] - x_approx))


# Colour, linestyle, and marker are cycled with DIFFERENT periods (10 / 4 / 9) rather than all mod
# len(COLOR_CYCLE). If all three shared one 10-entry period, trace #11 would silently reuse the exact
# (colour, linestyle, marker) triple of trace #1 - two curves rendered identically. Distinct, mutually
# co-prime-ish periods push that collision out to LCM(10, 4, 9) = 180 traces, well past any plot here.
# Only used where multiple traces are actually overlaid; a lone trace uses a single solid colour instead.
def _color(i: int) -> str:
    """Colour for trace ``i``, cycling every 10 traces."""
    return COLOR_CYCLE[i % len(COLOR_CYCLE)]


def _linestyle(i: int) -> str:
    """Linestyle for trace ``i``, cycling every 4 traces (solid / dashed / dash-dot / dotted)."""
    return LINESTYLE_CYCLE[i % 4]


def _marker(i: int) -> str:
    """Marker for trace ``i``, cycling every 9 traces (one fewer than the colour period)."""
    return MARKER_CYCLE[i % 9]
# =================================================================================================================================== #



# =================================================================================================================================== #
# -------------------------------------------------------- OSCILLATOR PLOTTER CLASS -------------------------------------------------- #
class OscillatorPlotter:
    """
    Produce publication-quality plots from parsed ADSListParser data for the BAW oscillator design.

    Parameters:
        - output_dir [str] : Directory where all figures are written (created automatically if absent).
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ──────────────────────────────────────────────────── private helpers ─────
    def _save(self, fig: plt.Figure, stem: str) -> None:
        """
        Save a figure as both PDF and PNG at >= 600 dpi and close it.

        Parameters:
            - fig  [Figure] : The Matplotlib figure to save.
            - stem [str]    : Output file name, without extension, relative to self.output_dir.

        Returns:
            - None
        """
        for ext in ('pdf', 'png'):
            path = os.path.join(self.output_dir, f'{stem}.{ext}')
            fig.savefig(path, dpi=600, bbox_inches='tight')
            print(f'Saved -> {path}')
        plt.close(fig)

    @staticmethod
    def _style_ax(
        ax:     plt.Axes,
        xlabel: str,
        ylabel: str,
        title:  str | None = None,
        xlim                = None,
        ylim                = None,
        legend: bool        = True,
        legend_kw: dict | None = None,
    ) -> None:
        """
        Apply uniform axis styling shared by every figure in this module.

        No figure carries a large top-level title; ``title`` here is only ever the small per-panel
        label of a multi-panel figure, set in LABEL_SIZE rather than the old title-sized font.

        When ``xlim`` is not given, the x-axis is tightened to the data range (``margins(x=0.01)``)
        instead of Matplotlib's default ~5% padding - large figures were reading as mostly white
        margin on the sides.

        Parameters:
            - ax        [Axes]     : The Matplotlib axes to style.
            - xlabel    [str]      : Label for the x-axis.
            - ylabel    [str]      : Label for the y-axis.
            - title     [str|None] : Small panel label (optional).
            - xlim      (tuple|None) : Limits for the x-axis (optional).
            - ylim      (tuple|None) : Limits for the y-axis (optional).
            - legend    [bool]     : Draw a legend if True.
            - legend_kw [dict|None]: Extra kwargs forwarded to ax.legend().

        Returns:
            - None
        """
        ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
        if title:
            ax.set_title(title, fontsize=LABEL_SIZE)
        ax.tick_params(labelsize=TICK_SIZE)
        if legend:
            kw = dict(fontsize=LEGEND_SIZE, loc='best')
            kw.update(legend_kw or {})
            ax.legend(**kw)
        apply_minor_grid(ax)
        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            ax.margins(x=0.01)
        if ylim is not None:
            ax.set_ylim(ylim)

    @staticmethod
    def _annotate_marker(
        ax:     plt.Axes,
        x:      float,
        y:      float,
        label:  str,
        xytext: tuple = (0.8, 0.85),
    ) -> None:
        """
        Mark a single point of interest with a small neutral-coloured dot and a formal callout box
        (white fill, thin grey/black border, connected to the point by a plain leader line) parked
        in open axes-fraction space rather than sitting directly on top of the trace in the trace's
        own colour - avoids the callout reading as a stray, informally-coloured legend.

        Parameters:
            - ax     [Axes]  : Axes to annotate.
            - x, y   [float] : Data coordinates of the point to mark.
            - label  [str]   : Callout text (LaTeX allowed).
            - xytext (tuple) : Callout box position, in axes-fraction coordinates.

        Returns:
            - None
        """
        ax.plot(
            [x], [y], marker='o', markersize=MARKER_SIZE * 0.7,
            markerfacecolor=style.SPINE_COLOR, markeredgecolor='white', markeredgewidth=1.0,
            linestyle='None', zorder=6,
        )
        ax.annotate(
            label, xy=(x, y), xycoords='data', xytext=xytext, textcoords='axes fraction',
            fontsize=ANNOTATION_SIZE, color=style.SPINE_COLOR, ha='center', va='center',
            arrowprops=dict(arrowstyle='-', color=style.SPINE_COLOR, linewidth=1.0, shrinkA=2, shrinkB=6),
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec=style.SPINE_COLOR, linewidth=0.9, alpha=0.95),
            zorder=7,
        )

    @staticmethod
    def _corner_note(ax: plt.Axes, text: str, loc: tuple = (0.97, 0.95), ha: str = 'right', va: str = 'top') -> None:
        """
        Small formal info box (e.g. a reference impedance or carrier frequency) parked in a figure
        corner - the reduced-clutter replacement for what used to be carried in an axes title.

        Parameters:
            - ax   [Axes]  : Axes to annotate.
            - text [str]   : Note text (LaTeX allowed).
            - loc  (tuple) : Position in axes-fraction coordinates.
            - ha, va [str] : Horizontal/vertical alignment.

        Returns:
            - None
        """
        ax.text(
            loc[0], loc[1], text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=ANNOTATION_SIZE, color=style.SPINE_COLOR,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=style.GRID_MAJOR_COLOR, linewidth=0.8, alpha=0.9),
            zorder=6,
        )

    # ═══════════════════════════════════════════════════════ TRANSISTOR ═══════
    def plot_transistor_iv(self, parser, filename: str = 'transistor_iv') -> None:
        """
        Plot collector current |I_C| vs. V_CE for each base-current sweep (I_B), one curve per I_B.
        |I_C| is plotted (not the signed ADS value) because the stage is an inverting common-emitter
        amplifier, so I_C.i is exported negative in the active region. Colour/linestyle/marker all vary
        per curve since this figure overlays 11 I_B traces.

        Parameters:
            - parser   [ADSListParser] : Parsed transistor_iv.csv (columns 'IB', 'VCE', 'IC.i', SI units A/V).
            - filename [str]           : Output file stem (no extension).

        Returns:
            - None
        """
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), constrained_layout=True)

        for i, (ib, g) in enumerate(parser.groups('IB')):
            ax.plot(
                g['VCE'], np.abs(g['IC.i']) * 1e3,
                color=_color(i), linestyle=_linestyle(i),
                marker=_marker(i), markevery=_markevery(len(g)),
                markersize=MARKER_SIZE * 0.75, linewidth=LINE_WIDTH,
                label=rf'$I_B = {ib * 1e6:.0f}\,\mathrm{{\mu A}}$',
            )

        self._style_ax(
            ax, xlabel=r'$V_{CE}$ [V]', ylabel=r'$|I_C|$ [mA]',
            legend_kw=dict(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=SMALL_SIZE,
                            ncol=1, borderaxespad=0.0, title=r'$I_B$ sweep'),
        )
        self._save(fig, filename)

    def plot_transistor_gain(self, parser, filename: str = 'transistor_hfe') -> None:
        """
        Plot |h_FE| = |I_C / I_B| vs. V_CE for each base-current sweep (I_B). The absolute value is
        plotted because the inverting common-emitter stage exports h_FE negative in the active region.
        Colour/linestyle/marker all vary per curve since this figure overlays 11 I_B traces.

        Parameters:
            - parser   [ADSListParser] : Parsed transistor_gain.csv (columns 'IB', 'VCE', 'hFE').
            - filename [str]           : Output file stem (no extension).

        Returns:
            - None
        """
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), constrained_layout=True)

        for i, (ib, g) in enumerate(parser.groups('IB')):
            ax.plot(
                g['VCE'], np.abs(g['hFE']),
                color=_color(i), linestyle=_linestyle(i),
                marker=_marker(i), markevery=_markevery(len(g)),
                markersize=MARKER_SIZE * 0.75, linewidth=LINE_WIDTH,
                label=rf'$I_B = {ib * 1e6:.0f}\,\mathrm{{\mu A}}$',
            )

        self._style_ax(
            ax, xlabel=r'$V_{CE}$ [V]', ylabel=r'$|h_{FE}|$',
            legend_kw=dict(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=SMALL_SIZE,
                            ncol=1, borderaxespad=0.0, title=r'$I_B$ sweep'),
        )
        self._save(fig, filename)

    # ═══════════════════════════════════════════════════════ RESONATOR ════════
    def plot_resonator_reflection(
        self,
        parser,
        col_measured: str,
        col_model:    str,
        filename: str = 'resonator_reflection',
        xlim           = None,
    ) -> None:
        """
        Plot |Z11| (dB) of the measured resonator against the equivalent mBVD circuit model vs. frequency.
        Colour, linestyle, and marker both vary since two traces are overlaid.

        Parameters:
            - parser       [ADSListParser] : Parsed resonator S-parameter/impedance export (column 'freq' in Hz).
            - col_measured [str]           : Column name holding the measured curve (dB).
            - col_model    [str]           : Column name holding the mBVD equivalent-circuit curve (dB).
            - filename     [str]           : Output file stem (no extension).
            - xlim         (tuple|None)    : (f_min, f_max) in GHz, or None for the full exported sweep.

        Returns:
            - None
        """
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), constrained_layout=True)
        freq_ghz = parser.df['freq'] / 1e9
        n        = len(parser.df)

        for i, (col, label) in enumerate([(col_measured, 'Measured'), (col_model, 'mBVD model')]):
            ax.plot(
                freq_ghz, parser.df[col],
                color=_color(i), linestyle=_linestyle(i),
                marker=_marker(i), markevery=_markevery(n),
                markersize=MARKER_SIZE * 0.7, linewidth=LINE_WIDTH, label=label,
            )

        self._style_ax(ax, xlabel='Frequency [GHz]', ylabel=r'$|Z_{11}|$ [dB]', xlim=xlim)
        self._save(fig, filename)

    # ═══════════════════════════════════════════════════════ OPEN-LOOP GAIN ════
    def plot_open_loop_gain(
        self,
        parser_mag,
        parser_phase,
        f_mark_mag_ghz:   float,
        f_mark_phase_ghz: float,
        filename: str = 'open_loop_gain',
        xlim           = None,
        phase_search_window_ghz: float = 0.01,
    ) -> None:
        """
        Plot the Randall-Hock corrected open-loop gain magnitude (dB) and phase (deg) vs. frequency, each
        as a single plain trace. The magnitude panel is marked at ``f_mark_mag_ghz`` directly (nearest
        exported sample). The phase panel is marked at the genuine zero-phase crossing found within
        +/- ``phase_search_window_ghz`` of ``f_mark_phase_ghz`` - not just the nearest sample to that
        frequency, which need not itself sit at 0 degrees.

        Parameters:
            - parser_mag        [ADSListParser] : Parsed open_loop_gain_magnitude.csv ('freq' Hz, gain column dB).
            - parser_phase      [ADSListParser] : Parsed open_loop_gain_phase.csv ('freq' Hz, phase column deg).
            - f_mark_mag_ghz    [float]         : Frequency to mark on the magnitude panel, in GHz.
            - f_mark_phase_ghz  [float]         : Approximate zero-crossing frequency to search around, in GHz.
            - filename          [str]           : Output file stem (no extension).
            - xlim              (tuple|None)    : (f_min, f_max) in GHz, or None for the full exported sweep.
            - phase_search_window_ghz [float]   : Half-width, in GHz, of the window searched for the crossing.

        Returns:
            - None
        """
        mag_col   = [c for c in parser_mag.df.columns   if c != 'freq'][0]
        phase_col = [c for c in parser_phase.df.columns if c != 'freq'][0]

        mag   = parser_mag.df[['freq', mag_col]].dropna().sort_values('freq', kind='mergesort')
        phase = parser_phase.df[['freq', phase_col]].dropna().sort_values('freq', kind='mergesort')

        fig, (ax_mag, ax_ph) = plt.subplots(
            2, 1, figsize=(FIG_WIDTH, FIG_HEIGHT * 1.6), sharex=True, constrained_layout=True,
        )

        color = COLOR_CYCLE[0]
        ax_mag.plot(mag['freq'] / 1e9, mag[mag_col], color=color, linewidth=LINE_WIDTH)
        ax_ph.plot(phase['freq'] / 1e9, phase[phase_col], color=color, linewidth=LINE_WIDTH)
        ax_ph.axhline(0.0, color=style.GRID_MAJOR_COLOR, linewidth=0.8, zorder=0)

        mag_pt = _nearest_row(mag, 'freq', f_mark_mag_ghz * 1e9)
        self._annotate_marker(
            ax_mag, mag_pt['freq'] / 1e9, mag_pt[mag_col],
            rf'$f={f_mark_mag_ghz:.3f}$\,GHz' + '\n' + rf'${mag_pt[mag_col]:.2f}$\,dB',
            xytext=(0.80, 0.90),
        )
        f_cross, y_cross = _find_zero_crossing(
            phase, 'freq', phase_col, f_mark_phase_ghz * 1e9, phase_search_window_ghz * 1e9,
        )
        self._annotate_marker(
            ax_ph, f_cross / 1e9, y_cross,
            rf'$f={f_cross / 1e9:.4f}$\,GHz' + '\n' + rf'${y_cross:.1f}^\circ$',
            xytext=(0.80, 0.15),
        )

        self._style_ax(ax_mag, xlabel='', ylabel=r'$|G_c|$ [dB]', xlim=xlim, legend=False)
        self._style_ax(ax_ph, xlabel='Frequency [GHz]', ylabel=r'$\angle G_c$ [$^\circ$]', xlim=xlim, legend=False)
        self._save(fig, filename)

    # ═══════════════════════════════════════════════════════ OSCTEST NYQUIST ═══
    def plot_osctest_nyquist(
        self,
        parser_mag,
        parser_phase,
        f_mark_ghz: float,
        Zo:         float = 21.5,
        filename:   str   = 'osctest_nyquist',
        xlim                     = None,
        search_window_ghz: float = 0.01,
    ) -> None:
        """
        Plot the magnitude (dB) and phase (deg) of the loop reflection product vs. frequency from an
        OscTest / OscPort analysis as continuous solid traces, for evaluating the Nyquist stability
        criterion. The genuine zero-phase crossing is located within +/- ``search_window_ghz`` of
        ``f_mark_ghz`` and that exact frequency is then marked on both panels (the magnitude value
        there is interpolated from the magnitude trace, since the two exports don't share a sample grid).

        Parameters:
            - parser_mag   [ADSListParser] : Parsed nyquist_s11_dB.csv ('freq' Hz, magnitude column dB).
            - parser_phase [ADSListParser] : Parsed nyquist_s11_phase.csv ('freq' Hz, phase column deg).
            - f_mark_ghz   [float]         : Approximate zero-crossing frequency to search around, in GHz.
            - Zo           [float]         : OscTest reference (probe) impedance in Ohm, noted on the figure.
            - filename     [str]           : Output file stem (no extension).
            - xlim         (tuple|None)    : (f_min, f_max) in GHz, or None for the full exported sweep.
            - search_window_ghz [float]    : Half-width, in GHz, of the window searched for the crossing.

        Returns:
            - None
        """
        mag_col   = [c for c in parser_mag.df.columns   if c != 'freq'][0]
        phase_col = [c for c in parser_phase.df.columns if c != 'freq'][0]
        mag   = parser_mag.df.dropna(subset=[mag_col]).sort_values('freq')
        phase = parser_phase.df.dropna(subset=[phase_col]).sort_values('freq')

        fig, (ax_mag, ax_ph) = plt.subplots(
            2, 1, figsize=(FIG_WIDTH, FIG_HEIGHT * 1.6), sharex=True, constrained_layout=True,
        )

        color = COLOR_CYCLE[1]
        ax_mag.plot(mag['freq'] / 1e9, mag[mag_col], color=color, linewidth=LINE_WIDTH)
        ax_ph.plot(phase['freq'] / 1e9, phase[phase_col], color=color, linewidth=LINE_WIDTH)
        ax_mag.axhline(0.0, color=style.GRID_MAJOR_COLOR, linewidth=0.8, zorder=0)

        f_cross, phase_cross = _find_zero_crossing(
            phase, 'freq', phase_col, f_mark_ghz * 1e9, search_window_ghz * 1e9,
        )
        mag_cross = float(np.interp(f_cross, mag['freq'], mag[mag_col]))

        self._annotate_marker(
            ax_mag, f_cross / 1e9, mag_cross,
            rf'$f={f_cross / 1e9:.4f}$\,GHz' + '\n' + rf'${mag_cross:.2f}$\,dB',
            xytext=(0.78, 0.85),
        )
        self._annotate_marker(
            ax_ph, f_cross / 1e9, phase_cross,
            rf'$f={f_cross / 1e9:.4f}$\,GHz' + '\n' + rf'${phase_cross:.1f}^\circ$',
            xytext=(0.78, 0.85),
        )

        self._corner_note(ax_mag, rf'$Z_o = {Zo:g}\,\Omega$', loc=(0.02, 0.95), ha='left')
        self._style_ax(ax_mag, xlabel='', ylabel=r'$|\Gamma_{loop}|$ [dB]', xlim=xlim, legend=False)
        self._style_ax(ax_ph, xlabel='Frequency [GHz]', ylabel=r'$\angle\Gamma_{loop}$ [$^\circ$]',
                        xlim=xlim, legend=False)
        self._save(fig, filename)

    # ═══════════════════════════════════════════════════════ TRANSIENT + FFT ═══
    def plot_transient(
        self,
        parser_time,
        parser_fft,
        filename:     str   = 'transient_output',
        t_startup_ns: tuple = (0.0, 60.0),
    ) -> None:
        """
        Plot the transient output voltage (power-up to steady state) and its spectrum (FFT over the
        steady-state window), each as a single plain trace. The time-domain panel carries one inset,
        zoomed on the power-up region, since the full 1 us span is too dense to show individual cycles
        there; the spectrum panel is shown at full width with no inset.

        Parameters:
            - parser_time  [ADSListParser] : Parsed tran_vout_time.csv ('time' s, voltage column V).
            - parser_fft   [ADSListParser] : Parsed tran_vout_time_fft.csv ('freq' Hz, spectrum column dBm).
            - filename     [str]           : Output file stem (no extension).
            - t_startup_ns (tuple)         : (t_min, t_max) in ns for the power-up inset.

        Returns:
            - None
        """
        v_col = [c for c in parser_time.df.columns if c != 'time'][0]
        f_col = [c for c in parser_fft.df.columns   if c != 'freq'][0]

        t_ns = parser_time.df['time'].values * 1e9
        v_mv = parser_time.df[v_col].values * 1e3
        f_ghz = parser_fft.df['freq'].values / 1e9
        spec  = parser_fft.df[f_col].values

        fig, (ax_t, ax_f) = plt.subplots(
            2, 1, figsize=(FIG_WIDTH, FIG_HEIGHT * 1.7), constrained_layout=True,
        )

        # ── time domain: full trace + power-up inset ────────────────────────────
        ax_t.plot(t_ns, v_mv, color=COLOR_CYCLE[0], linewidth=LINE_WIDTH * 0.9)
        self._style_ax(ax_t, xlabel='Time [ns]', ylabel=r'$V_{out}$ [mV]', legend=False)

        zoom_rules(ax_t, t_startup_ns)

        axi_start = ax_t.inset_axes([0.06, 0.55, 0.38, 0.42])
        m_start   = (t_ns >= t_startup_ns[0]) & (t_ns <= t_startup_ns[1])
        axi_start.plot(t_ns[m_start], v_mv[m_start], color=COLOR_CYCLE[0], linewidth=LINE_WIDTH * 0.8)
        axi_start.set_title('Start-Up', fontsize=SMALL_SIZE)
        axi_start.tick_params(labelsize=SMALL_SIZE * 0.8)
        axi_start.margins(x=0.02)

        # ── frequency domain: full spectrum, no inset ───────────────────────────
        ax_f.plot(f_ghz, spec, color=COLOR_CYCLE[1], linewidth=LINE_WIDTH)
        self._style_ax(ax_f, xlabel='Frequency [GHz]', ylabel=r'$V_{out}$ [dBm]', legend=False)

        self._save(fig, filename)

    # ═══════════════════════════════════════════════════════ HARMONIC BALANCE ══
    def plot_hb_operating_point(
        self,
        parser_wave,
        parser_spec,
        vcc:      float,
        filename: str = 'hb_operating_point',
    ) -> None:
        """
        Plot the harmonic-balance steady-state time-domain waveform and harmonic spectrum at a single
        (nominal) bias point. The waveform carries extra markers, evenly spaced in time, purely for
        visual rhythm (it is a single trace, so they are not needed to distinguish it from anything -
        just to break up an otherwise bare line). The spectrum is drawn as discrete stems ("deltas")
        rather than a joined line, since a harmonic index is not a continuous quantity; the stems are
        drawn "mirrored" - growing up from a floor near the lowest harmonic level rather than down from
        0 dBm - so a taller stem reads as a stronger harmonic (dBm values are all negative here, so
        stems grown from 0 would run longest for the *weakest* harmonic, backwards from the intuitive
        reading).

        Parameters:
            - parser_wave [ADSListParser] : Parsed hb_waveform.csv ('Vcc', 'time' s, voltage column V).
            - parser_spec [ADSListParser] : Parsed hb_spectrum.csv ('Vcc', 'harmindex', spectrum column dBm).
            - vcc         [float]         : Bias voltage to select (matched to the nearest exported value).
            - filename    [str]           : Output file stem (no extension).

        Returns:
            - None
        """
        v_col = [c for c in parser_wave.df.columns if c not in ('Vcc', 'time')][0]
        s_col = [c for c in parser_spec.df.columns if c not in ('Vcc', 'harmindex')][0]

        vcc_wave = min(parser_wave.df['Vcc'].unique(), key=lambda v: abs(v - vcc))
        wave     = parser_wave.df[parser_wave.df['Vcc'] == vcc_wave].sort_values('time')
        spec     = parser_spec.df[parser_spec.df['Vcc'] == vcc_wave].dropna(subset=[s_col]).sort_values('harmindex')

        fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(FIG_WIDTH * 1.6, FIG_HEIGHT), constrained_layout=True)

        ax_t.plot(wave['time'] * 1e12, wave[v_col] * 1e3,
                  color=COLOR_CYCLE[0], marker=MARKER_CYCLE[0], markevery=_markevery(len(wave), n_markers=20),
                  markersize=MARKER_SIZE * 0.6, linewidth=LINE_WIDTH)
        self._corner_note(ax_t, rf'$V_{{CC}}={vcc_wave:g}$\,V', loc=(0.97, 0.06))
        self._style_ax(ax_t, xlabel='Time [ps]', ylabel=r'$V_{out}$ [mV]', title='Waveform', legend=False)

        floor = spec[s_col].min() - 5.0
        markerline, stemlines, _ = ax_s.stem(spec['harmindex'], spec[s_col], bottom=floor, basefmt=' ')
        plt.setp(markerline, color=COLOR_CYCLE[1], marker=MARKER_CYCLE[1], markersize=MARKER_SIZE)
        plt.setp(stemlines,  color=COLOR_CYCLE[1], linewidth=LINE_WIDTH)
        ax_s.set_xticks(spec['harmindex'])
        ax_s.set_ylim(floor, spec[s_col].max() + 5.0)
        self._style_ax(ax_s, xlabel='Harmonic index', ylabel=r'$V_{out}$ [dBm]', title='Spectrum', legend=False)

        self._save(fig, filename)

    def plot_hb_family(
        self,
        parser_wave,
        parser_spec,
        filename: str = 'hb_family_vs_vcc',
    ) -> None:
        """
        Plot the harmonic-balance steady-state waveform and harmonic spectrum as a family of curves
        across the full V_CC sweep, one colour/linestyle/marker per bias point (both panels overlay
        7 traces, so the full style cycle applies). The spectrum uses grouped "delta" stems, offset
        slightly within each harmonic index so the 7 bias points stay distinguishable, mirrored the
        same way as the single-operating-point spectrum so taller = stronger.

        Parameters:
            - parser_wave [ADSListParser] : Parsed hb_waveform.csv ('Vcc', 'time' s, voltage column V).
            - parser_spec [ADSListParser] : Parsed hb_spectrum.csv ('Vcc', 'harmindex', spectrum column dBm).
            - filename    [str]           : Output file stem (no extension).

        Returns:
            - None
        """
        v_col = [c for c in parser_wave.df.columns if c not in ('Vcc', 'time')][0]
        s_col = [c for c in parser_spec.df.columns if c not in ('Vcc', 'harmindex')][0]

        fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(FIG_WIDTH * 1.6, FIG_HEIGHT), constrained_layout=True)

        wave_groups = parser_wave.groups('Vcc')
        for i, (vcc, g) in enumerate(wave_groups):
            g = g.sort_values('time')
            ax_t.plot(
                g['time'] * 1e12, g[v_col] * 1e3,
                color=_color(i), linestyle=_linestyle(i),
                marker=_marker(i), markevery=_markevery(len(g), n_markers=14),
                markersize=MARKER_SIZE * 0.55, linewidth=LINE_WIDTH * 0.9,
                label=rf'$V_{{CC}}={vcc:g}$\,V',
            )
        self._style_ax(ax_t, xlabel='Time [ps]', ylabel=r'$V_{out}$ [mV]', title='Waveform',
                        legend_kw=dict(fontsize=SMALL_SIZE, loc='lower left'))

        spec_groups = parser_spec.groups('Vcc')
        floor = parser_spec.df[s_col].min() - 5.0
        n_g   = len(spec_groups)
        width = 0.7 / n_g
        for i, (vcc, g) in enumerate(spec_groups):
            g = g.dropna(subset=[s_col]).sort_values('harmindex')
            x_off = g['harmindex'] + (i - (n_g - 1) / 2) * width
            markerline, stemlines, _ = ax_s.stem(x_off, g[s_col], bottom=floor, basefmt=' ')
            plt.setp(markerline, color=_color(i), marker=_marker(i), markersize=MARKER_SIZE * 0.65,
                     label=rf'$V_{{CC}}={vcc:g}$\,V')
            plt.setp(stemlines, color=_color(i), linewidth=LINE_WIDTH * 0.8)

        ax_s.set_xticks(sorted(parser_spec.df['harmindex'].unique()))
        ax_s.set_ylim(floor, parser_spec.df[s_col].max() + 5.0)
        self._style_ax(ax_s, xlabel='Harmonic index', ylabel=r'$V_{out}$ [dBm]', title='Spectrum',
                        legend_kw=dict(fontsize=SMALL_SIZE, loc='upper right'))

        self._save(fig, filename)

    # ═══════════════════════════════════════════════════════ PHASE NOISE ═══════
    def plot_phase_noise(
        self,
        parser,
        offset_col:   str,
        pn_col:       str,
        carrier_ghz:  float,
        mark_offset_hz: float = 1e5,
        filename:     str     = 'phase_noise',
    ) -> None:
        """
        Plot simulated phase noise L(delta-f) vs. offset frequency from the carrier, on a logarithmic
        frequency axis, as a single plain trace with a formal marker at the requested offset.

        Parameters:
            - parser         [ADSListParser] : Parsed phase_noise.csv.
            - offset_col     [str]           : Column name holding the offset frequency (Hz).
            - pn_col         [str]           : Column name holding L(delta-f) (dBc/Hz).
            - carrier_ghz    [float]         : Carrier frequency in GHz, noted on the figure.
            - mark_offset_hz [float]         : Offset frequency to mark, in Hz.
            - filename       [str]           : Output file stem (no extension).

        Returns:
            - None
        """
        df = parser.df.dropna(subset=[offset_col, pn_col]).sort_values(offset_col)

        fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), constrained_layout=True)
        ax.plot(df[offset_col], df[pn_col], color=COLOR_CYCLE[4], linewidth=LINE_WIDTH)
        ax.set_xscale('log')

        pt = _nearest_row(df, offset_col, mark_offset_hz)
        offset_label = f'{mark_offset_hz / 1e3:g}\\,kHz' if mark_offset_hz < 1e6 else f'{mark_offset_hz / 1e6:g}\\,MHz'
        self._annotate_marker(
            ax, pt[offset_col], pt[pn_col],
            rf'${pt[pn_col]:.1f}$\,dBc/Hz' + '\n' + rf'@ {offset_label}',
            xytext=(0.72, 0.85),
        )
        self._corner_note(ax, rf'$f_c={carrier_ghz:g}$\,GHz', loc=(0.97, 0.06))

        self._style_ax(
            ax, xlabel=r'Offset frequency $\Delta f$ [Hz]', ylabel=r'$\mathcal{L}(\Delta f)$ [dBc/Hz]',
            legend=False,
        )
        self._save(fig, filename)
# =================================================================================================================================== #

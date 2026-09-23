# =================================================================================================================================== #
# ----------------------------------------------------------- DESCRIPTION ----------------------------------------------------------- #
# ComparisonPlotter: two-topology overlay figures (Pierce vs. negative-resistance) built from the same ADSListParser data consumed    #
# by OscillatorPlotter. Subclasses OscillatorPlotter purely for its styling/annotation helpers (_style_ax, _mark_point,               #
# _corner_note, _save) and its module-level helpers (_nearest_row, _find_zero_crossing, _markevery) - every figure here overlays      #
# exactly two traces (one per topology), each drawn in that topology's own colour throughout every comparison figure. Each marked     #
# point's crosshair guide lines use a topology-specific linestyle (dashed for A, dotted for B) so two markers sharing one axes stay   #
# visually distinct even when their crossings sit close together.                                                                     #
# Author: Nedal M. Benelmekki                                                                                                         #
# =================================================================================================================================== #



# =================================================================================================================================== #
# --------------------------------------------------------- EXTERNAL IMPORTS -------------------------------------------------------- #
import numpy             as np   # Numerical computing.                                                                               #
import matplotlib.pyplot as plt  # Plotting.                                                                                          #
# =================================================================================================================================== #



# =================================================================================================================================== #
# ------------------------------------------------------- STYLE CONFIGURATION ------------------------------------------------------- #
import style.style as style                                                                                                           #
from style.style import (                                                                                                             #
    LABEL_SIZE, LEGEND_SIZE, SMALL_SIZE, FIG_WIDTH, FIG_HEIGHT, LINE_WIDTH, MARKER_SIZE, COLOR_CYCLE, MARKER_CYCLE,                  #
)                                                                                                                                      #
from src.plot_oscillator import OscillatorPlotter, _nearest_row, _find_zero_crossing, _markevery                                      #
# =================================================================================================================================== #



# =================================================================================================================================== #
# ------------------------------------------------------ COMPARISON PLOTTER CLASS --------------------------------------------------- #
class ComparisonPlotter(OscillatorPlotter):
    """
    Produce Pierce-vs-negative-resistance overlay figures at a shared bias point, reusing OscillatorPlotter's
    styling/annotation helpers. Topology A is always drawn in COLOR_CYCLE[0] (Pierce), topology B in
    COLOR_CYCLE[1] (negative resistance) - kept consistent across every figure in this module so a reader
    only has to learn the colour mapping once.

    Parameters:
        - output_dir [str] : Directory where all figures are written (created automatically if absent).
    """

    COLOR_A = COLOR_CYCLE[0]
    COLOR_B = COLOR_CYCLE[1]

    # ═══════════════════════════════════════════════════════ OPEN-LOOP GAIN ════
    def plot_open_loop_gain_comparison(
        self,
        mag_a, phase_a, mag_b, phase_b,
        label_a: str, label_b: str,
        f_mark_a_ghz: float, f_mark_b_ghz: float,
        nominal_a: bool = False, nominal_b: bool = False,
        filename: str = 'open_loop_gain_comparison',
        xlim = None,
        phase_search_window_ghz: float = 0.01,
    ) -> None:
        """
        Overlay the Randall-Hock open-loop gain magnitude (dB) and phase (deg) of two topologies vs.
        frequency. Each topology's phase panel is marked either at a solved zero-phase crossing
        (``nominal_*=False``, searched within +/- ``phase_search_window_ghz`` of its ``f_mark_*_ghz``)
        or at the nearest exported sample to ``f_mark_*_ghz`` directly (``nominal_*=True``, for data too
        numerically degenerate to trust a solved crossing).

        Parameters:
            - mag_a, phase_a   [ADSListParser] : Topology A's open_loop_gain_magnitude / _phase.csv.
            - mag_b, phase_b   [ADSListParser] : Topology B's open_loop_gain_magnitude / _phase.csv.
            - label_a, label_b [str]           : Legend labels for each topology.
            - f_mark_a_ghz, f_mark_b_ghz [float]: Marker frequency (GHz) for each topology's phase panel.
            - nominal_a, nominal_b [bool]      : If True, skip the zero-crossing search for that topology.
            - filename         [str]           : Output file stem (no extension).
            - xlim             (tuple|None)    : (f_min, f_max) in GHz, or None for the full exported sweep.
            - phase_search_window_ghz [float]  : Half-width, in GHz, of the window searched for a crossing.

        Returns:
            - None
        """
        fig, (ax_mag, ax_ph) = plt.subplots(
            2, 1, figsize=(FIG_WIDTH, FIG_HEIGHT * 1.6), sharex=True, constrained_layout=True,
        )

        for mag, phase, label, color, f_mark, nominal, guide_style in (
            (mag_a, phase_a, label_a, self.COLOR_A, f_mark_a_ghz, nominal_a, '--'),
            (mag_b, phase_b, label_b, self.COLOR_B, f_mark_b_ghz, nominal_b, ':'),
        ):
            mag_col   = [c for c in mag.df.columns   if c != 'freq'][0]
            phase_col = [c for c in phase.df.columns if c != 'freq'][0]
            mag_df    = mag.df[['freq', mag_col]].dropna().sort_values('freq', kind='mergesort')
            phase_df  = phase.df[['freq', phase_col]].dropna().sort_values('freq', kind='mergesort')

            ax_mag.plot(mag_df['freq'] / 1e9, mag_df[mag_col], color=color, linewidth=LINE_WIDTH, label=label)
            ax_ph.plot(phase_df['freq'] / 1e9, phase_df[phase_col], color=color, linewidth=LINE_WIDTH, label=label)

            if nominal:
                pt = _nearest_row(phase_df, 'freq', f_mark * 1e9)
                f_cross, y_cross = pt['freq'], pt[phase_col]
                value_label = rf'{label}: $f={f_cross / 1e9:.4f}$\,GHz, ${y_cross:.1f}^\circ$ (nominal)'
            else:
                f_cross, y_cross = _find_zero_crossing(
                    phase_df, 'freq', phase_col, f_mark * 1e9, phase_search_window_ghz * 1e9,
                )
                value_label = rf'{label}: $f={f_cross / 1e9:.4f}$\,GHz, ${y_cross:.1f}^\circ$'
            self._mark_point(ax_ph, f_cross / 1e9, y_cross, value_label, color=color, guide_style=guide_style)

        ax_ph.axhline(0.0, color=style.GRID_MAJOR_COLOR, linewidth=0.8, zorder=0)
        self._style_ax(ax_mag, xlabel='', ylabel=r'$|G_c|$ [dB]', xlim=xlim,
                        legend_kw=dict(fontsize=SMALL_SIZE))
        self._style_ax(ax_ph, xlabel='Frequency [GHz]', ylabel=r'$\angle G_c$ [$^\circ$]', xlim=xlim,
                        legend_kw=dict(fontsize=SMALL_SIZE))
        self._save(fig, filename)

    # ═══════════════════════════════════════════════════════ OSCTEST NYQUIST ═══
    def plot_osctest_nyquist_comparison(
        self,
        mag_a, phase_a, mag_b, phase_b,
        label_a: str, label_b: str,
        f_mark_a_ghz: float, f_mark_b_ghz: float,
        phase_target_a_deg: float = 0.0,
        phase_target_b_deg: float = 0.0,
        Zo_a: float = 21.5, Zo_b: float = 21.5,
        filename: str = 'osctest_nyquist_comparison',
        xlim = None,
        search_window_ghz: float = 0.01,
    ) -> None:
        """
        Overlay the magnitude (dB) and phase (deg) of the OscTest loop reflection product of two
        topologies vs. frequency. Each topology's genuine phase crossing through its own
        ``phase_target_*_deg`` is located within +/- ``search_window_ghz`` of its ``f_mark_*_ghz`` and
        marked on both panels. The two topologies' loop-phase references need not agree (e.g. one crosses
        0 deg, the other wraps through +/-180 deg, depending on how each loop was broken/probed in its
        schematic) - that is why each gets its own target, not a shared one.

        Parameters:
            - mag_a, phase_a   [ADSListParser] : Topology A's nyquist_s11_dB / _phase.csv.
            - mag_b, phase_b   [ADSListParser] : Topology B's nyquist_s11_dB / _phase.csv.
            - label_a, label_b [str]           : Legend labels for each topology.
            - f_mark_a_ghz, f_mark_b_ghz [float]: Approximate crossing frequency (GHz) to search around.
            - phase_target_a_deg, phase_target_b_deg [float] : Phase value each topology's loop crosses.
            - Zo_a, Zo_b       [float]         : OscTest reference (probe) impedance in Ohm, per topology.
            - filename         [str]           : Output file stem (no extension).
            - xlim             (tuple|None)    : (f_min, f_max) in GHz, or None for the full exported sweep.
            - search_window_ghz [float]        : Half-width, in GHz, of the window searched for the crossing.

        Returns:
            - None
        """
        fig, (ax_mag, ax_ph) = plt.subplots(
            2, 1, figsize=(FIG_WIDTH, FIG_HEIGHT * 1.6), sharex=True, constrained_layout=True,
        )

        notes = []
        for mag, phase, label, color, f_mark, target, Zo, guide_style in (
            (mag_a, phase_a, label_a, self.COLOR_A, f_mark_a_ghz, phase_target_a_deg, Zo_a, '--'),
            (mag_b, phase_b, label_b, self.COLOR_B, f_mark_b_ghz, phase_target_b_deg, Zo_b, ':'),
        ):
            mag_col   = [c for c in mag.df.columns   if c != 'freq'][0]
            phase_col = [c for c in phase.df.columns if c != 'freq'][0]
            mag_df    = mag.df.dropna(subset=[mag_col]).sort_values('freq')
            phase_df  = phase.df.dropna(subset=[phase_col]).sort_values('freq')

            ax_mag.plot(mag_df['freq'] / 1e9, mag_df[mag_col], color=color, linewidth=LINE_WIDTH, label=label)
            ax_ph.plot(phase_df['freq'] / 1e9, phase_df[phase_col], color=color, linewidth=LINE_WIDTH, label=label)

            f_cross, phase_cross = _find_zero_crossing(
                phase_df, 'freq', phase_col, f_mark * 1e9, search_window_ghz * 1e9, target=target,
            )
            mag_cross = float(np.interp(f_cross, mag_df['freq'], mag_df[mag_col]))
            self._mark_point(
                ax_mag, f_cross / 1e9, mag_cross,
                rf'{label}: $f={f_cross / 1e9:.4f}$\,GHz, ${mag_cross:.2f}$\,dB',
                color=color, guide_style=guide_style,
            )
            self._mark_point(
                ax_ph, f_cross / 1e9, phase_cross,
                rf'{label}: $f={f_cross / 1e9:.4f}$\,GHz, ${phase_cross:.1f}^\circ$',
                color=color, guide_style=guide_style,
            )
            notes.append(rf'$Z_o$ ({label}) $= {Zo:g}\,\Omega$')

        ax_mag.axhline(0.0, color=style.GRID_MAJOR_COLOR, linewidth=0.8, zorder=0)
        self._corner_note(ax_mag, '\n'.join(notes), loc=(0.02, 0.95), ha='left')
        self._style_ax(ax_mag, xlabel='', ylabel=r'$|\Gamma_{loop}|$ [dB]', xlim=xlim,
                        legend_kw=dict(fontsize=SMALL_SIZE))
        self._style_ax(ax_ph, xlabel='Frequency [GHz]', ylabel=r'$\angle\Gamma_{loop}$ [$^\circ$]',
                        xlim=xlim, legend_kw=dict(fontsize=SMALL_SIZE))
        self._save(fig, filename)

    # ═══════════════════════════════════════════════════════ TRANSIENT + FFT ═══
    def plot_transient_comparison(
        self,
        time_a, fft_a, time_b, fft_b,
        label_a: str, label_b: str,
        filename: str = 'transient_output_comparison',
        t_startup_ns: tuple = (0.0, 60.0),
    ) -> None:
        """
        Overlay the transient output voltage (power-up to steady state) and its spectrum (FFT over the
        steady-state window) of two topologies. The time-domain panel carries one inset, zoomed on the
        power-up region, for both traces at once.

        Parameters:
            - time_a, fft_a  [ADSListParser] : Topology A's tran_vout_time / _fft.csv.
            - time_b, fft_b  [ADSListParser] : Topology B's tran_vout_time / _fft.csv.
            - label_a, label_b [str]         : Legend labels for each topology.
            - filename       [str]           : Output file stem (no extension).
            - t_startup_ns   (tuple)         : (t_min, t_max) in ns for the power-up inset.

        Returns:
            - None
        """
        fig, (ax_t, ax_f) = plt.subplots(
            2, 1, figsize=(FIG_WIDTH, FIG_HEIGHT * 1.7), constrained_layout=True,
        )
        axi_start = ax_t.inset_axes([0.06, 0.55, 0.38, 0.42])
        axi_start.set_title('Start-Up', fontsize=SMALL_SIZE)
        axi_start.tick_params(labelsize=SMALL_SIZE * 0.8)
        axi_start.margins(x=0.02)

        for time_p, fft_p, label, color in (
            (time_a, fft_a, label_a, self.COLOR_A),
            (time_b, fft_b, label_b, self.COLOR_B),
        ):
            v_col = [c for c in time_p.df.columns if c != 'time'][0]
            f_col = [c for c in fft_p.df.columns   if c != 'freq'][0]
            t_ns  = time_p.df['time'].values * 1e9
            v_mv  = time_p.df[v_col].values * 1e3
            f_ghz = fft_p.df['freq'].values / 1e9
            spec  = fft_p.df[f_col].values

            ax_t.plot(t_ns, v_mv, color=color, linewidth=LINE_WIDTH * 0.9, label=label)
            m_start = (t_ns >= t_startup_ns[0]) & (t_ns <= t_startup_ns[1])
            axi_start.plot(t_ns[m_start], v_mv[m_start], color=color, linewidth=LINE_WIDTH * 0.8)
            ax_f.plot(f_ghz, spec, color=color, linewidth=LINE_WIDTH, label=label)

        style.zoom_rules(ax_t, t_startup_ns)
        self._style_ax(ax_t, xlabel='Time [ns]', ylabel=r'$V_{out}$ [mV]')
        self._style_ax(ax_f, xlabel='Frequency [GHz]', ylabel=r'$V_{out}$ [dBm]')
        self._save(fig, filename)

    # ═══════════════════════════════════════════════════════ HARMONIC BALANCE ══
    def plot_hb_operating_point_comparison(
        self,
        wave_a, spec_a, wave_b, spec_b,
        label_a: str, label_b: str,
        vcc: float,
        filename: str = 'hb_operating_point_comparison',
    ) -> None:
        """
        Overlay the harmonic-balance steady-state time-domain waveform and harmonic spectrum of two
        topologies at a single (nominal) bias point. The spectrum uses grouped "delta" stems, offset
        within each harmonic index so both topologies stay distinguishable, mirrored (growing up from a
        floor near the lowest level) so a taller stem reads as a stronger harmonic.

        Parameters:
            - wave_a, spec_a [ADSListParser] : Topology A's hb_waveform / hb_spectrum.csv.
            - wave_b, spec_b [ADSListParser] : Topology B's hb_waveform / hb_spectrum.csv.
            - label_a, label_b [str]         : Legend labels for each topology.
            - vcc            [float]         : Bias voltage to select (matched to the nearest exported value).
            - filename       [str]           : Output file stem (no extension).

        Returns:
            - None
        """
        fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(FIG_WIDTH * 1.6, FIG_HEIGHT), constrained_layout=True)

        specs, floors, s_cols = [], [], []
        for wave, spec, label, color, marker in (
            (wave_a, spec_a, label_a, self.COLOR_A, MARKER_CYCLE[0]),
            (wave_b, spec_b, label_b, self.COLOR_B, MARKER_CYCLE[1]),
        ):
            v_col = [c for c in wave.df.columns if c not in ('Vcc', 'time')][0]
            s_col = [c for c in spec.df.columns if c not in ('Vcc', 'harmindex')][0]

            vcc_wave = min(wave.df['Vcc'].unique(), key=lambda v: abs(v - vcc))
            w = wave.df[wave.df['Vcc'] == vcc_wave].sort_values('time')
            s = spec.df[spec.df['Vcc'] == vcc_wave].dropna(subset=[s_col]).sort_values('harmindex')

            ax_t.plot(
                w['time'] * 1e12, w[v_col] * 1e3, color=color, marker=marker,
                markevery=_markevery(len(w), n_markers=20), markersize=MARKER_SIZE * 0.6,
                linewidth=LINE_WIDTH, label=rf'{label} ($V_{{CC}}={vcc_wave:g}$\,V)',
            )
            specs.append(s); floors.append(s[s_col].min()); s_cols.append(s_col)

        floor = min(floors) - 5.0
        n_g   = len(specs)
        width = 0.6 / n_g
        y_max = max(s[c].max() for s, c in zip(specs, s_cols))
        for i, (s, s_col, color, marker, label) in enumerate(zip(
            specs, s_cols, (self.COLOR_A, self.COLOR_B), (MARKER_CYCLE[0], MARKER_CYCLE[1]), (label_a, label_b),
        )):
            x_off = s['harmindex'] + (i - (n_g - 1) / 2) * width
            markerline, stemlines, _ = ax_s.stem(x_off, s[s_col], bottom=floor, basefmt=' ')
            plt.setp(markerline, color=color, marker=marker, markersize=MARKER_SIZE * 0.65, label=label)
            plt.setp(stemlines, color=color, linewidth=LINE_WIDTH * 0.9)

        all_idx = sorted(set(specs[0]['harmindex']).union(specs[1]['harmindex']))
        ax_s.set_xticks(all_idx)
        ax_s.set_ylim(floor, y_max + 5.0)

        self._style_ax(ax_t, xlabel='Time [ps]', ylabel=r'$V_{out}$ [mV]', title='Waveform',
                        legend_kw=dict(fontsize=SMALL_SIZE, loc='lower left'))
        self._style_ax(ax_s, xlabel='Harmonic index', ylabel=r'$V_{out}$ [dBm]', title='Spectrum',
                        legend_kw=dict(fontsize=SMALL_SIZE, loc='upper right'))
        self._save(fig, filename)

    # ═══════════════════════════════════════════════════════ PHASE NOISE ═══════
    def plot_phase_noise_comparison(
        self,
        parser_a, parser_b,
        label_a: str, label_b: str,
        offset_col: str, pn_col: str,
        carrier_a_ghz: float, carrier_b_ghz: float,
        mark_offset_hz: float = 1e5,
        filename: str = 'phase_noise_comparison',
    ) -> None:
        """
        Overlay simulated phase noise L(delta-f) vs. offset frequency from the carrier of two
        topologies, on a logarithmic frequency axis, each with a formal marker at the requested offset.

        Parameters:
            - parser_a, parser_b [ADSListParser] : Each topology's parsed phase_noise.csv.
            - label_a, label_b   [str]           : Legend labels for each topology.
            - offset_col         [str]           : Column name holding the offset frequency (Hz).
            - pn_col             [str]           : Column name holding L(delta-f) (dBc/Hz).
            - carrier_a_ghz, carrier_b_ghz [float]: Carrier frequency in GHz per topology, noted on the figure.
            - mark_offset_hz     [float]         : Offset frequency to mark, in Hz.
            - filename           [str]           : Output file stem (no extension).

        Returns:
            - None
        """
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), constrained_layout=True)

        notes = []
        for parser, label, color, carrier, guide_style in (
            (parser_a, label_a, self.COLOR_A, carrier_a_ghz, '--'),
            (parser_b, label_b, self.COLOR_B, carrier_b_ghz, ':'),
        ):
            df = parser.df.dropna(subset=[offset_col, pn_col]).sort_values(offset_col)
            ax.plot(df[offset_col], df[pn_col], color=color, linewidth=LINE_WIDTH, label=label)

            pt = _nearest_row(df, offset_col, mark_offset_hz)
            offset_label = f'{mark_offset_hz / 1e3:g}\\,kHz' if mark_offset_hz < 1e6 else f'{mark_offset_hz / 1e6:g}\\,MHz'
            self._mark_point(
                ax, pt[offset_col], pt[pn_col],
                rf'{label}: ${pt[pn_col]:.1f}$\,dBc/Hz @ {offset_label}',
                color=color, guide_style=guide_style,
            )
            notes.append(rf'$f_c$ ({label}) $={carrier:g}$\,GHz')

        ax.set_xscale('log')
        self._corner_note(ax, '\n'.join(notes), loc=(0.97, 0.06))
        self._style_ax(
            ax, xlabel=r'Offset frequency $\Delta f$ [Hz]', ylabel=r'$\mathcal{L}(\Delta f)$ [dBc/Hz]',
        )
        self._save(fig, filename)
# =================================================================================================================================== #

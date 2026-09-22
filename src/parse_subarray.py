# =================================================================================================================================== #
# ----------------------------------------------------------- DESCRIPTION ----------------------------------------------------------- #
# Pair-wise 4-port subarray parser (Block 3.C acquisition protocol).                                                                  #
# SubarrayParser assembles the reflection / coupling set of a 4-port antenna network from the six pair-wise .s2p files.               #
# The NanoVNA is a 1.5-port instrument: each file carries valid data only in the S11 and S21 columns (S12 / S22 are written as        #
# zeros).  Consequently the reflection of a port is recoverable only from a file in which that port sits on VNA P1.                   #
# Port 4 sits on VNA P1 only in the supplementary ``pair42`` acquisition; absent that file S44 is reported through ``missing``.       #
# The public surface (``params`` / ``get_df``) matches S2PParser, so instances drop straight into MeasurementPlotter / FinalPlotter.  #
# Author: Nedal M. Benelmekki                                                                                                         #
# =================================================================================================================================== #



# =================================================================================================================================== #
# --------------------------------------------------------- EXTERNAL IMPORTS -------------------------------------------------------- #
import os                        # Operating system interfaces.                                                                       #
import sys                       # System-specific parameters and functions.                                                          #
import glob                      # Filename pattern expansion.                                                                        #
import numpy  as np              # Numerical computing.                                                                               #
import pandas as pd              # Data manipulation and analysis.                                                                    #
# =================================================================================================================================== #



# =================================================================================================================================== #
# ----------------------------------------------------- DIRECTORY CONFIGURATION ----------------------------------------------------- #
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))   # Directory of this script (src/).                                       #
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)                 # Project root = parent of src/.                                         #
sys.path.insert(0, _PROJECT_ROOT)                            # Add project root to sys.path for absolute imports.                     #
# =================================================================================================================================== #



# =================================================================================================================================== #
# ------------------------------------------------------- SUBARRAY PARSER CLASS ----------------------------------------------------- #
class SubarrayParser:
    """
    Assemble the S-parameter set of a 4-port antenna network from the six
    pair-wise two-port acquisitions of Block 3.C.

    Acquisition schedule
    --------------------
    ==== =================== =================== =====================
    Pair VNA P1 connects to   VNA P2 connects to  Terminated in 50 ohm
    ==== =================== =================== =====================
    1    Port 1 (Ant1, S)    Port 2 (Ant1, C)    Ports 3, 4
    2    Port 1 (Ant1, S)    Port 3 (Ant2, S)    Ports 2, 4
    3    Port 1 (Ant1, S)    Port 4 (Ant2, C)    Ports 2, 3
    4    Port 2 (Ant1, C)    Port 3 (Ant2, S)    Ports 1, 4
    5    Port 2 (Ant1, C)    Port 4 (Ant2, C)    Ports 1, 3
    6    Port 3 (Ant2, S)    Port 4 (Ant2, C)    Ports 1, 2
    ==== =================== =================== =====================

    Instrument limitation
    ---------------------
    The NanoVNA drives port 1 only, so each Touchstone file holds valid data
    in the ``S11`` and ``S21`` columns alone; the ``S12`` / ``S22`` columns are
    written as exact zeros and are discarded here.  A port's reflection is
    therefore observable only in the files where that port is on VNA P1:

        Port 1 -> pairs 12, 13, 14   (three repeats)
        Port 2 -> pairs 23, 24       (two repeats)
        Port 3 -> pair  34           (one acquisition)
        Port 4 -> pair  42           (one acquisition; supplementary file)

    The repeats are exposed through ``repeats()`` and provide a direct handle
    on cable--antenna mate repeatability, the dominant non-stationarity of the
    block.

    Assembled parameters
    --------------------
    Reflections  : ``S11``, ``S22``, ``S33``, ``S44``  (``S44`` present only when
                   the supplementary ``pair42`` file is supplied; otherwise it is
                   reported through ``missing``)
    Transmissions: ``S21``, ``S31``, ``S41``, ``S32``, ``S42``, ``S43``, ``S24``

    All curves are the *installed-with-cables* response; the cable bank is not
    de-embedded here (that is Block 3.E).  Magnitudes therefore carry the
    one-way cable loss and phases carry the cable delay.

    DataFrame columns (as in S2PParser / FekoParser):
        freq_GHz  -- frequency in GHz
        real      -- Re(S)
        imag      -- Im(S)
        mag_dB    -- 20 * log10(|S|)
        phase_deg -- arg(S) in degrees

    Parameters:
        - meas_dir [str] : Directory holding the six pair-wise .s2p files.
        - stem     [str] : Filename stem preceding the ``pairXY`` token, e.g. ``'p3_subarray_L30'``.
        - combine  [str] : ``'first'`` keeps the earliest acquisition of a repeated reflection; ``'mean'`` averages the repeats in the complex plane.
        - files    [dict | None] : Explicit ``{'12': path, ...}`` mapping that bypasses filename discovery.
    """

    # ── acquisition schedule: pair token -> (port on VNA P1, port on VNA P2) ──
    # The first six pairs are the Block 3.C schedule.  ``42`` is the supplementary
    # acquisition that finally places port 4 on VNA P1 (P1 -> port 4, P2 -> port 2),
    # recovering the S44 reflection the six-pair schedule could not deliver.  It is
    # discovered like any other pair, so campaigns without that file simply omit it
    # and continue to report S44 through ``missing``.
    _SCHEDULE = {
        '12': (1, 2),
        '13': (1, 3),
        '14': (1, 4),
        '23': (2, 3),
        '24': (2, 4),
        '34': (3, 4),
        '42': (4, 2),
    }

    # ── port index -> physical identity, for labelling ────────────────────────
    PORT_LABELS = {
        1: 'Ant1, S',
        2: 'Ant1, C',
        3: 'Ant2, S',
        4: 'Ant2, C',
    }

    def __init__(
        self,
        meas_dir: str,
        stem:     str         = 'p3_subarray_L30',
        combine:  str         = 'first',
        files:    dict | None = None,
    ) -> None:
        """
        Class constructor. Discovers the six pair files, reads them, and
        assembles the 4-port parameter set.

        Parameters:
            - meas_dir [str]         : Directory holding the six pair-wise .s2p files.
            - stem     [str]         : Filename stem preceding the ``pairXY`` token.
            - combine  [str]         : ``'first'`` or ``'mean'`` (see class docstring).
            - files    [dict | None] : Explicit ``{'12': path, ...}`` mapping overriding discovery.

        Returns:
            - None
        """
        if combine not in ('first', 'mean'):
            raise ValueError("combine must be 'first' or 'mean'")

        self.meas_dir = meas_dir
        self.stem     = stem
        self.combine  = combine

        self.files:    dict[str, str]          = files if files is not None else self._discover(meas_dir, stem)
        self._raw:     dict[str, dict]         = {}    # pair token -> {'freq_GHz', 'refl', 'trans'}
        self.data:     dict[str, pd.DataFrame] = {}    # 'S11' -> df, ...
        self._repeats: dict[int, dict]         = {}    # port  -> {pair token: df}

        self._read_pairs()
        self._assemble()

    # ──────────────────────────────────────────────────────── static helpers ──
    @staticmethod
    def _discover(meas_dir: str, stem: str) -> dict[str, str]:
        """
        Locate the six pair files inside meas_dir.

        Parameters:
            - meas_dir [str] : Directory to search.
            - stem     [str] : Filename stem preceding the ``pairXY`` token.

        Returns:
            - dict[str, str] : ``{'12': path, '13': path, ...}`` for every pair found.
        """
        found: dict[str, str] = {}
        for pair in SubarrayParser._SCHEDULE:
            matches = sorted(glob.glob(os.path.join(meas_dir, f'{stem}_pair{pair}_*.s2p')))
            if matches:
                found[pair] = matches[0]
        if not found:
            raise FileNotFoundError(
                f'No files matching "{stem}_pair<XY>_*.s2p" found in {meas_dir!r}'
            )
        return found

    @staticmethod
    def _read_s2p(filepath: str) -> tuple[str, np.ndarray]:
        """
        Read one Touchstone .s2p file, tolerating comma decimal separators.

        Parameters:
            - filepath [str] : Path to the .s2p file.

        Returns:
            - freq_unit [str]     : Lowercase frequency unit token from the option line.
            - data      [ndarray] : Shape (N, 9): freq, S11_re, S11_im, S21_re, S21_im, S12_re, S12_im, S22_re, S22_im.
        """
        freq_unit     = 'hz'
        records: list = []

        with open(filepath, encoding='utf-8', errors='replace') as fh:
            for line in fh:
                s = line.strip()
                if not s or s.startswith('!'):
                    continue
                if s.startswith('#'):
                    tokens = s[1:].split()
                    if tokens:
                        freq_unit = tokens[0].lower()
                    continue
                parts = s.split()
                if len(parts) >= 5:
                    try:
                        records.append([float(p.replace(',', '.')) for p in parts[:9]])
                    except ValueError:
                        continue

        arr = np.array(records) if records else np.empty((0, 9))
        return freq_unit, arr

    @staticmethod
    def _to_ghz(freq: np.ndarray, unit: str) -> np.ndarray:
        """
        Convert a frequency array to GHz according to units.

        Parameters:
            - freq [ndarray] : Frequency values.
            - unit [str]     : Frequency unit token ('hz', 'khz', 'mhz', 'ghz').

        Returns:
            - ndarray : Frequency values in GHz.
        """
        factors = {'hz': 1e-9, 'khz': 1e-6, 'mhz': 1e-3, 'ghz': 1.0}
        return freq * factors.get(unit.lower(), 1e-9)

    @staticmethod
    def _build_df(freq_ghz: np.ndarray, s_cpx: np.ndarray) -> pd.DataFrame:
        """
        Build a tidy DataFrame with derived magnitude and phase columns.

        Parameters:
            - freq_ghz [ndarray] : Frequency in GHz.
            - s_cpx    [ndarray] : Complex S-parameter values.

        Returns:
            - pd.DataFrame : Columns freq_GHz, real, imag, mag_dB, phase_deg.
        """
        df              = pd.DataFrame({'freq_GHz': freq_ghz,
                                        'real':     np.real(s_cpx),
                                        'imag':     np.imag(s_cpx)})
        magnitude       = np.abs(s_cpx)
        df['mag_dB']    = 20.0 * np.log10(np.clip(magnitude, 1e-15, None))
        df['phase_deg'] = np.degrees(np.angle(s_cpx))
        return df

    # ──────────────────────────────────────────────────────────── parsing ─────
    def _read_pairs(self) -> None:
        """
        Read every discovered pair file and keep only the two physically valid
        columns: the VNA-P1 reflection and the forward transmission.

        Parameters:
            - None

        Returns:
            - None
        """
        for pair, path in self.files.items():
            unit, arr = self._read_s2p(path)
            if arr.size == 0:
                continue
            self._raw[pair] = {
                'freq_GHz': self._to_ghz(arr[:, 0], unit),
                'refl':     arr[:, 1] + 1j * arr[:, 2],   # reflection at the port on VNA P1.
                'trans':    arr[:, 3] + 1j * arr[:, 4],   # transmission P1 -> P2.
                'path':     path,
            }

    def _assemble(self) -> None:
        """
        Populate ``self.data`` with the reflection and transmission parameters,
        and ``self._repeats`` with every individual reflection acquisition.

        Parameters:
            - None

        Returns:
            - None
        """
        # ── reflections: group the acquisitions by the port sitting on VNA P1 ──
        by_port: dict[int, list[str]] = {}
        for pair in sorted(self._raw):
            p1, _ = self._SCHEDULE[pair]
            by_port.setdefault(p1, []).append(pair)

        for port, pairs in by_port.items():
            self._repeats[port] = {
                pair: self._build_df(self._raw[pair]['freq_GHz'], self._raw[pair]['refl'])
                for pair in pairs
            }
            freq = self._raw[pairs[0]]['freq_GHz']
            if self.combine == 'mean' and len(pairs) > 1:
                stack = np.vstack([self._raw[p]['refl'] for p in pairs])
                s_cpx = stack.mean(axis=0)
            else:
                s_cpx = self._raw[pairs[0]]['refl']
            self.data[f'S{port}{port}'] = self._build_df(freq, s_cpx)

        # ── transmissions: one per pair, indexed S<p2><p1> (P1 driven) ─────────
        for pair in sorted(self._raw):
            p1, p2 = self._SCHEDULE[pair]
            self.data[f'S{p2}{p1}'] = self._build_df(
                self._raw[pair]['freq_GHz'], self._raw[pair]['trans']
            )

    # ──────────────────────────────────────────────────────── data access ─────
    @property
    def params(self) -> list[str]:
        """
        All assembled S-parameter keys.

        Parameters:
            - None

        Returns:
            - list[str] : e.g. ['S11', 'S21', 'S22', 'S31', ...].
        """
        return sorted(self.data.keys())

    @property
    def reflections(self) -> list[str]:
        """
        Assembled reflection keys only.

        Parameters:
            - None

        Returns:
            - list[str] : e.g. ['S11', 'S22', 'S33'].
        """
        return [p for p in self.params if p[1] == p[2]]

    @property
    def measured_ports(self) -> list[int]:
        """
        Ports whose reflection was acquired, i.e. those sitting on VNA P1 at
        least once.

        Parameters:
            - None

        Returns:
            - list[int] : e.g. [1, 2, 3].
        """
        return sorted(self._repeats)

    @property
    def missing(self) -> list[str]:
        """
        Reflection parameters the six-pair schedule cannot deliver.

        Parameters:
            - None

        Returns:
            - list[str] : Reflection keys absent from ``self.data``, e.g. ['S44'].
        """
        return [f'S{p}{p}' for p in (1, 2, 3, 4) if f'S{p}{p}' not in self.data]

    def get_df(self, param: str) -> pd.DataFrame:
        """
        Return the DataFrame for param (e.g. 'S11', 'S31').

        Parameters:
            - param [str] : S-parameter key.

        Returns:
            - pd.DataFrame : DataFrame for the requested S-parameter.
        """
        if param not in self.data:
            raise KeyError(
                f'Unknown parameter {param!r}. Available: {self.params}. '
                f'Not acquirable with this schedule: {self.missing}'
            )
        return self.data[param]

    def repeats(self, port: int) -> dict[str, pd.DataFrame]:
        """
        Return every individual reflection acquisition of a port, keyed by the
        pair token it came from.

        Parameters:
            - port [int] : Antenna-network port index (1-4).

        Returns:
            - dict[str, pd.DataFrame] : ``{'12': df, '13': df, '14': df}`` for port 1, and so on.
        """
        if port not in self._repeats:
            raise KeyError(
                f'Port {port} never sits on VNA P1 in this schedule; '
                f'its reflection was not acquired.'
            )
        return self._repeats[port]

    def resonance(self, param: str) -> tuple[float, float]:
        """
        Locate the deepest point of a parameter's magnitude response.

        Note that the ports of this structure are multi-resonant, so the global
        minimum alone is not a faithful summary; use ``resonances`` instead when
        comparing ports or comparing against simulation.

        Parameters:
            - param [str] : S-parameter key.

        Returns:
            - (f_GHz, mag_dB) [tuple[float, float]] : Frequency and depth of the minimum.
        """
        df  = self.get_df(param)
        idx = int(np.argmin(df['mag_dB'].values))
        return float(df['freq_GHz'].values[idx]), float(df['mag_dB'].values[idx])

    @staticmethod
    def find_resonances(
        freq_ghz:      np.ndarray,
        mag_db:        np.ndarray,
        threshold_db:  float = -6.0,
        prominence_db: float = 3.0,
    ) -> list[tuple[float, float]]:
        """
        Extract every resonant notch of a magnitude response, ordered by
        frequency.

        A candidate is a local minimum deeper than threshold_db.  Candidates are
        accepted deepest-first and a new one is kept only if, between it and
        every already accepted notch, the response rises at least prominence_db
        above the shallower of the two.  That is the standard topographic
        prominence test and it suppresses the noise ripple that would otherwise
        split one notch into several.

        Parameters:
            - freq_ghz      [ndarray] : Frequency axis in GHz.
            - mag_db        [ndarray] : Magnitude response in dB.
            - threshold_db  [float]   : Only minima below this level count as resonances.
            - prominence_db [float]   : Minimum rise required between two distinct notches.

        Returns:
            - list[tuple[float, float]] : ``[(f_GHz, depth_dB), ...]`` sorted by frequency.
        """
        d = np.asarray(mag_db, dtype=float)
        f = np.asarray(freq_ghz, dtype=float)
        if d.size < 3:
            return []

        interior  = np.arange(1, d.size - 1)
        is_min    = (d[interior] <= d[interior - 1]) & (d[interior] <= d[interior + 1])
        candidate = interior[is_min & (d[interior] < threshold_db)]
        if candidate.size == 0:
            return []

        accepted: list[int] = []
        for idx in candidate[np.argsort(d[candidate])]:          # Deepest candidate first.
            distinct = True
            for kept in accepted:
                lo, hi = sorted((int(idx), int(kept)))
                if d[lo:hi + 1].max() < max(d[idx], d[kept]) + prominence_db:
                    distinct = False
                    break
            if distinct:
                accepted.append(int(idx))

        return [(float(f[i]), float(d[i])) for i in sorted(accepted)]

    def resonances(
        self,
        param:         str,
        threshold_db:  float = -6.0,
        prominence_db: float = 3.0,
    ) -> list[tuple[float, float]]:
        """
        Extract every resonant notch of an assembled parameter.

        Parameters:
            - param         [str]   : S-parameter key.
            - threshold_db  [float] : Only minima below this level count as resonances.
            - prominence_db [float] : Minimum rise required between two distinct notches.

        Returns:
            - list[tuple[float, float]] : ``[(f_GHz, depth_dB), ...]`` sorted by frequency.
        """
        df = self.get_df(param)
        return self.find_resonances(
            df['freq_GHz'].values, df['mag_dB'].values, threshold_db, prominence_db
        )

    def __repr__(self) -> str:
        """
        String representation showing the pairs loaded and the parameters assembled.

        Parameters:
            - None

        Returns:
            - str : String representation of the instance.
        """
        n = len(next(iter(self.data.values()))) if self.data else 0
        return (
            f'SubarrayParser(dir="{os.path.basename(self.meas_dir)}", '
            f'pairs={sorted(self._raw)}, refl={self.reflections}, '
            f'missing={self.missing}, combine="{self.combine}", n_pts={n})'
        )
# =================================================================================================================================== #

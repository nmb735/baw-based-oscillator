# =================================================================================================================================== #
# ----------------------------------------------------------- DESCRIPTION ----------------------------------------------------------- #
# FEKO S2P parser.                                                                                                                    #
# FekoParser loads a single Touchstone .s2p file exported from FEKO and exposes frequency and S-parameter data as tidy DataFrames.   #
# Per-parameter views are CSTParser-compatible, allowing direct use with SimulationPlotter and FinalPlotter.                          #
# Author: Nedal M. Benelmekki                                                                                                         #
# =================================================================================================================================== #



# =================================================================================================================================== #
# --------------------------------------------------------- EXTERNAL IMPORTS -------------------------------------------------------- #
import os                        # Operating system interfaces.                                                                       #
import sys                       # System-specific parameters and functions.                                                          #
import numpy  as np              # Numerical computing.                                                                               #
import pandas as pd              # Data manipulation and analysis.                                                                    #
# =================================================================================================================================== #



# =================================================================================================================================== #
# ----------------------------------------------------- DIRECTORY CONFIGURATION ----------------------------------------------------- #
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))   # Directory of this script (src/).                                       #
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)                 # Project root = parent of src/.                                         #
sys.path.insert(0, _PROJECT_ROOT)                            # Add project root to sys.path for imports.                              #
# =================================================================================================================================== #



# =================================================================================================================================== #
# -------------------------------------------------------- FEKO PARSER CLASS -------------------------------------------------------- #
class FekoParser:
    """
    Parse a FEKO-exported Touchstone .s2p file.

    The file may encode data in RI (Real/Imaginary), MA (Magnitude/Angle), or
    DB (dB/Angle) format as declared on the option line.  After construction
    ``self.data`` holds one DataFrame per S-parameter::

        { 'S11': df, 'S21': df, 'S12': df, 'S22': df }

    DataFrame columns:
        - freq_GHz  : frequency in GHz
        - real      : Re(S)
        - imag      : Im(S)
        - mag_dB    : 20 * log10(|S|)
        - phase_deg : arg(S) in degrees

    Each S-parameter can be accessed as a CSTParser-compatible view via
    ``feko['S11']`` or ``feko.get_parsers()``, making it directly usable
    with ``SimulationPlotter`` and ``FinalPlotter``.
    """

    # ── CSTParser-compatible per-parameter view ───────────────────────────────
    class _ParamView:
        """
        Thin CSTParser-compatible wrapper over a single S-parameter DataFrame.

        Exposes ``mesh_labels``, ``final_labels``, ``get_df``, ``get_final_df``,
        and ``get_meta`` so that ``SimulationPlotter`` and ``FinalPlotter`` can
        treat a FEKO parameter identically to a CST curve.
        """

        def __init__(self, label: str, df: pd.DataFrame) -> None:
            self._label = label
            self._df    = df

        @property
        def labels(self) -> list[str]:
            """All curve labels (only the parameter name for FEKO)."""
            return [self._label]

        @property
        def final_labels(self) -> list[str]:
            """Labels of converged curves (all FEKO curves are final)."""
            return [self._label]

        @property
        def mesh_labels(self) -> list[str]:
            """Labels of mesh-adaptation-pass curves (always empty for FEKO)."""
            return []

        def get_df(self, label: str) -> pd.DataFrame:
            """
            Return the DataFrame (label is ignored; present for CSTParser compatibility).

            Parameters:
                - label [str] : Curve label (unused).

            Returns:
                - pd.DataFrame : DataFrame for this S-parameter.
            """
            return self._df

        def get_final_df(self) -> pd.DataFrame:
            """
            Return the final (only) DataFrame.

            Parameters:
                - None

            Returns:
                - pd.DataFrame : DataFrame for this S-parameter.
            """
            return self._df

        def get_meta(self, label: str) -> dict:
            """
            Return metadata (always empty for FEKO; present for CSTParser compatibility).

            Parameters:
                - label [str] : Curve label (unused).

            Returns:
                - dict : Empty dictionary.
            """
            return {}

    # ── constructor ───────────────────────────────────────────────────────────
    def __init__(self, file_path: str) -> None:
        """
        Class constructor: parse the given FEKO-exported .s2p file.

        Parameters:
            - file_path [str] : Absolute or relative path to the FEKO .s2p file.

        Returns:
            - None
        """
        self.file_path = file_path
        self.data: dict[str, pd.DataFrame] = {}
        self._meta: dict = {}
        self._parse()

    # ── static helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _to_ghz(freq: np.ndarray, unit: str) -> np.ndarray:
        """
        Convert a frequency array to GHz.

        Parameters:
            - freq [ndarray] : Frequency values in the original unit.
            - unit [str]     : Frequency unit token ('hz', 'khz', 'mhz', 'ghz').

        Returns:
            - ndarray : Frequency values in GHz.
        """
        factors = {'hz': 1e-9, 'khz': 1e-6, 'mhz': 1e-3, 'ghz': 1.0}
        return freq * factors.get(unit.lower(), 1e-9)

    @staticmethod
    def _build_df(freq_ghz: np.ndarray, real: np.ndarray, imag: np.ndarray) -> pd.DataFrame:
        """
        Build a tidy DataFrame with derived magnitude and phase columns.

        Parameters:
            - freq_ghz [ndarray] : Frequency in GHz.
            - real     [ndarray] : Real part of the S-parameter.
            - imag     [ndarray] : Imaginary part of the S-parameter.

        Returns:
            - pd.DataFrame : DataFrame with columns: freq_GHz, real, imag, mag_dB, phase_deg.
        """
        df              = pd.DataFrame({'freq_GHz': freq_ghz, 'real': real, 'imag': imag})
        magnitude       = np.hypot(df['real'], df['imag'])
        df['mag_dB']    = 20.0 * np.log10(magnitude.clip(lower=1e-15))
        df['phase_deg'] = np.degrees(np.arctan2(df['imag'], df['real']))
        return df

    # ── parsing ───────────────────────────────────────────────────────────────
    def _parse(self) -> None:
        """
        Read ``self.file_path`` and populate ``self.data`` and ``self._meta``.

        Handles RI (Real/Imaginary), MA (Magnitude/Angle), and DB (dB/Angle)
        data formats as declared on the Touchstone option line.

        Parameters:
            - None

        Returns:
            - None
        """
        freq_unit = 'hz'
        fmt       = 'ri'
        records   = []
        meta      = {}

        with open(self.file_path, encoding='utf-8', errors='replace') as fh:
            for line in fh:
                s = line.strip()
                if not s:
                    continue

                if s.startswith('!'):
                    content = s[1:].strip()
                    if ':' in content:
                        k, _, v = content.partition(':')
                        meta[k.strip()] = v.strip()
                    continue

                if s.startswith('#'):
                    tokens    = s[1:].split()
                    freq_unit = tokens[0].lower() if len(tokens) > 0 else 'hz'
                    fmt       = tokens[2].lower() if len(tokens) > 2 else 'ri'
                    continue

                parts = s.split()
                if len(parts) >= 9:
                    try:
                        records.append([float(p.replace(',', '.')) for p in parts[:9]])
                    except ValueError:
                        continue

        self._meta = meta

        if not records:
            for param in ('S11', 'S21', 'S12', 'S22'):
                self.data[param] = pd.DataFrame()
            return

        arr      = np.array(records)
        freq_ghz = self._to_ghz(arr[:, 0], freq_unit)

        def _to_ri(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            if fmt == 'ri':
                return a, b
            if fmt == 'ma':
                return a * np.cos(np.radians(b)), a * np.sin(np.radians(b))
            if fmt == 'db':
                mag = 10.0 ** (a / 20.0)
                return mag * np.cos(np.radians(b)), mag * np.sin(np.radians(b))
            return a, b

        # Touchstone 2-port column order: S11, S21, S12, S22 (pairs of A, B)
        for param, col_a, col_b in [('S11', 1, 2), ('S21', 3, 4), ('S12', 5, 6), ('S22', 7, 8)]:
            real, imag         = _to_ri(arr[:, col_a], arr[:, col_b])
            self.data[param]   = self._build_df(freq_ghz, real, imag)

    # ── data access ───────────────────────────────────────────────────────────
    @property
    def params(self) -> list[str]:
        """
        All available S-parameter keys.

        Parameters:
            - None

        Returns:
            - list[str] : S-parameter keys, e.g. ['S11', 'S21', 'S12', 'S22'].
        """
        return list(self.data.keys())

    def get_df(self, param: str) -> pd.DataFrame:
        """
        Return the DataFrame for the given S-parameter.

        Parameters:
            - param [str] : S-parameter key (e.g. 'S11', 'S21', 'S12', 'S22').

        Returns:
            - pd.DataFrame : DataFrame for the requested S-parameter.
        """
        if param not in self.data:
            raise KeyError(f'Unknown parameter {param!r}. Available: {self.params}')
        return self.data[param]

    def get_meta(self) -> dict:
        """
        Return file-level metadata parsed from FEKO comment lines.

        Parameters:
            - None

        Returns:
            - dict : Metadata extracted from ``!`` comment lines.
        """
        return dict(self._meta)

    def __getitem__(self, param: str) -> '_ParamView':
        """
        Return a CSTParser-compatible view for the given S-parameter.

        Use this to build the ``parsers`` / ``sim_parsers`` dict expected by
        ``SimulationPlotter`` and ``FinalPlotter``::

            feko = FekoParser('s_parameters.s2p')
            sim_parsers = {'S11': feko['S11'], 'S21': feko['S21']}

        Parameters:
            - param [str] : S-parameter key (e.g. 'S11').

        Returns:
            - _ParamView : CSTParser-compatible view object.
        """
        if param not in self.data:
            raise KeyError(f'Unknown parameter {param!r}. Available: {self.params}')
        return self._ParamView(param, self.data[param])

    def get_parsers(self, params: list | None = None) -> dict:
        """
        Return a dict of CSTParser-compatible views for use with
        ``SimulationPlotter`` and ``FinalPlotter``.

        Parameters:
            - params [list | None] : S-parameter keys to include.
                                     Defaults to all available parameters.

        Returns:
            - dict : ``{'S11': _ParamView, 'S21': _ParamView, ...}``
        """
        keys = params if params is not None else self.params
        return {p: self[p] for p in keys}

    def __repr__(self) -> str:
        """
        String representation of the FekoParser instance.

        Parameters:
            - None

        Returns:
            - str : String representation showing file name, parameters, and point count.
        """
        n = len(self.data.get('S11', pd.DataFrame()))
        return (
            f'FekoParser("{os.path.basename(self.file_path)}", '
            f'params={self.params}, n_pts={n})'
        )
# =================================================================================================================================== #

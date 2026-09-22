# =================================================================================================================================== #
# ----------------------------------------------------------- DESCRIPTION ----------------------------------------------------------- #
# Touchstone .s2p parser for non-full-two-port NanoVNA measurements.                                                                  #
# S2PParser combines two swept files (forward + port-flipped reverse) to reconstruct all four S-parameters.                           #
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
sys.path.insert(0, _PROJECT_ROOT)                            # Add project root to sys.path for absolute imports.                     #
# =================================================================================================================================== #



# =================================================================================================================================== #
# ---------------------------------------------------------- S2P PARSER CLASS ------------------------------------------------------- #
class S2PParser:
    """
    Parse a pair of Touchstone .s2p files from a non-full-two-port NanoVNA
    measurement session.

    Because the NanoVNA cannot simultaneously drive both ports, two sweeps are
    required:

        file_fwd – forward sweep  (port 1 driven) → S11 and S21
        file_rev – reverse sweep  (ports flipped)  → S22 and S12

    In ``file_rev`` the instrument labels its output as "S11" (= actual S22)
    and "S21" (= actual S12); this class re-maps the columns automatically.

    After construction, ``self.data`` holds one DataFrame per S-parameter::

        { 'S11': df, 'S21': df, 'S12': df, 'S22': df }

    DataFrame columns:
        freq_GHz  -- frequency in GHz
        real      -- Re(S)
        imag      -- Im(S)
        mag_dB    -- 20 * log10(|S|)
        phase_deg -- arg(S) in degrees
    """

    def __init__(self, file_fwd: str, file_rev: str) -> None:
        """
        Class constructor.

        Parameters:
            - file_fwd [str] : Path to the forward-sweep .s2p file  (contains S11, S21).
            - file_rev [str] : Path to the port-flipped .s2p file   (contains S22, S12).

        Returns:
            - None
        """
        self.file_fwd = file_fwd
        self.file_rev = file_rev
        self.data: dict[str, pd.DataFrame] = {}
        self._parse()

    # ──────────────────────────────────────────────────────── static helpers ──
    @staticmethod
    def _read_s2p(filepath: str) -> tuple[str, np.ndarray]:
        """
        Read one Touchstone .s2p file.

        Parameters:
            - filepath [str] : Path to the .s2p file.

        Returns:
            - freq_unit [str]     : Lowercase frequency unit token from the option line (e.g. 'hz').
            - data      [ndarray] :  of shape (N, 9). Columns: freq  S11_re  S11_im  S21_re  S21_im  S12_re  S12_im  S22_re  S22_im
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
                if len(parts) >= 9:
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
            - unit [str]      : Frequency unit token (e.g. 'hz', 'khz', 'mhz', 'ghz').
        
        Returns:
            - freq_ghz [ndarray] : Frequency values converted to GHz.
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
            - df [DataFrame] : DataFrame with columns: freq_GHz, real, imag, mag_dB, phase_deg.
        """
        df = pd.DataFrame({'freq_GHz': freq_ghz, 'real': real, 'imag': imag})
        magnitude       = np.hypot(df['real'], df['imag'])
        df['mag_dB']    = 20.0 * np.log10(magnitude.clip(lower=1e-15))
        df['phase_deg'] = np.degrees(np.arctan2(df['imag'], df['real']))
        return df

    # ──────────────────────────────────────────────────────────── parsing ─────
    def _parse(self) -> None:
        """
        Read both files and populate ``self.data``.
        
        Parameters:
            - None
        
        Returns:
            - None
        """
        unit_fwd, arr_fwd = self._read_s2p(self.file_fwd)
        unit_rev, arr_rev = self._read_s2p(self.file_rev)

        f_fwd = self._to_ghz(arr_fwd[:, 0], unit_fwd)
        f_rev = self._to_ghz(arr_rev[:, 0], unit_rev)

        # Forward file — columns at index 1-4 carry S11 and S21
        self.data['S11'] = self._build_df(f_fwd, arr_fwd[:, 1], arr_fwd[:, 2])
        self.data['S21'] = self._build_df(f_fwd, arr_fwd[:, 3], arr_fwd[:, 4])

        # Reverse file — VNA "S11" col = actual S22; VNA "S21" col = actual S12
        self.data['S22'] = self._build_df(f_rev, arr_rev[:, 1], arr_rev[:, 2])
        self.data['S12'] = self._build_df(f_rev, arr_rev[:, 3], arr_rev[:, 4])

    # ──────────────────────────────────────────────────────── data access ─────
    @property
    def params(self) -> list[str]:
        """
        All available S-parameter keys: ['S11', 'S21', 'S22', 'S12'].
        
        Parameters:
            - None

        Returns:
            - params [list[str]] : List of available S-parameter keys.
        """
        return list(self.data.keys())

    def get_df(self, param: str) -> pd.DataFrame:
        """
        Return the DataFrame for param (e.g. 'S11').

        Parameters:
            - param [str] : S-parameter key (e.g. 'S11', 'S21', 'S12', 'S22').

        Returns:
            - df [DataFrame] : DataFrame for the requested S-parameter.
        """
        if param not in self.data:
            raise KeyError(f'Unknown parameter {param!r}. Available: {self.params}')
        return self.data[param]

    def __repr__(self) -> str:
        """
        String representation of the S2PParser instance, showing file names and point counts.

        Parameters:
            - None

        Returns:
            - repr_str [str] : String representation of the instance.
        """
        n_fwd = len(self.data.get('S11', pd.DataFrame()))
        n_rev = len(self.data.get('S22', pd.DataFrame()))
        return (
            f'S2PParser('
            f'fwd="{os.path.basename(self.file_fwd)}" [{n_fwd} pts], '
            f'rev="{os.path.basename(self.file_rev)}" [{n_rev} pts])'
        )
# =================================================================================================================================== #

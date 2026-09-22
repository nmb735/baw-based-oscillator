# =================================================================================================================================== #
# ----------------------------------------------------------- DESCRIPTION ----------------------------------------------------------- #
# CST XY Data Exchange Format V2 parser.                                                                                              #
# CSTParser loads a single .txt file exported from CST Studio Suite and exposes frequency and S-parameter data as tidy DataFrames.    #
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
# --------------------------------------------------------- CST PARSER CLASS -------------------------------------------------------- #
class CSTParser:
    """
    Parse a CST XY Data Exchange Format V2 .txt file.

    Each file may contain multiple curve blocks: one per mesh-adaptation pass
    plus the final converged result.  After construction ``self.curves`` holds
    every block found::

        { curve_label: { 'meta': dict, 'df': pd.DataFrame } }

    DataFrame columns:
        - freq_GHz  : frequency in GHz
        - real      : Re(S)
        - imag      : Im(S)
        - _extra0   : port impedance (optional CST-specific column)
        - mag_dB    : 20 * log10(|S|)
        - phase_deg : arg(S) in degrees
    """

    def __init__(self, file_path: str) -> None:
        """
        Class initialization: parse the given CST-exported .txt file and populate ``self.curves``.

        Parameters:
            - file_path [str] : Absolute or relative path to the CST-exported .txt file.

        Returns:
            - None
        """
        self.file_path = file_path
        self.curves: dict = {}
        self._parse()

    # ──────────────────────────────────────────────────────── static helpers ──
    @staticmethod
    def _is_number(token: str) -> bool:
        """
        Static helper method, to determine if an input represents a float. Handles comma decimal separators.

        Parameters:
            - token [str] : Input string to check.

        Returns:
            - bool : True if *token* can be parsed as a float, False otherwise.
        """
        try:
            float(token.replace(',', '.'))
            return True
        except ValueError:
            return False

    @staticmethod
    def _build_df(raw_rows: list) -> pd.DataFrame:
        """
        Convert raw whitespace-separated text rows into a tidy DataFrame.
        CST export columns : freq_GHz  real  imag  [port_impedance]  [0]
        Derived columns    : mag_dB (20 log10|S|),  phase_deg (degrees)

        Parameters:
            - raw_rows [list] : List of strings, each representing a line of raw data from the CST export.
        
        Returns:
            - pd.DataFrame : DataFrame with columns for frequency, real/imaginary parts, and derived magnitude/phase.
        """
        records = []
        for row in raw_rows:
            parts = row.split()
            if len(parts) >= 3:
                try:
                    records.append([float(p.replace(',', '.')) for p in parts])
                except ValueError:
                    continue

        if not records:
            return pd.DataFrame()

        n         = len(records[0])
        col_names = ['freq_GHz', 'real', 'imag'] + [f'_extra{i}' for i in range(n - 3)]
        df        = pd.DataFrame(records, columns=col_names)

        magnitude       = np.hypot(df['real'], df['imag'])
        df['mag_dB']    = 20.0 * np.log10(magnitude.clip(lower=1e-15))
        df['phase_deg'] = np.degrees(np.arctan2(df['imag'], df['real']))

        return df

    # ──────────────────────────────────────────────────────────── parsing ─────
    def _parse(self) -> None:
        """
        Read ``self.file_path`` and populate ``self.curves``.

        Parameters:
            - None
        
        Returns:
            - None
        """
        curves:  dict       = {}
        label:   str | None = None
        meta:    dict       = {}
        rows:    list       = []
        in_data: bool       = False

        def _flush() -> None:
            if label is not None and rows:
                curves[label] = {'meta': dict(meta), 'df': self._build_df(rows)}

        with open(self.file_path, encoding='utf-8', errors='replace') as fh:
            for raw_line in fh:
                s = raw_line.strip()

                if s.startswith('Curvelabel'):
                    _flush()
                    label   = s.split('=', 1)[1].strip()
                    meta    = {}
                    rows    = []
                    in_data = False

                elif label is not None:
                    first = s.split()[0] if s else ''

                    if in_data:
                        if self._is_number(first):
                            rows.append(s)
                    else:
                        if self._is_number(first):
                            in_data = True
                            rows.append(s)
                        elif '=' in s:
                            k, _, v = s.partition('=')
                            meta[k.strip()] = v.strip()

        _flush()
        self.curves = curves

    # ──────────────────────────────────────────────────────── data access ─────
    @property
    def labels(self) -> list[str]:
        """
        All curve labels present in the file.

        Parameters:
            - None
        
        Returns:
            - list [str] : List of all curve labels found in the CST export file.
        """
        return list(self.curves.keys())

    @property
    def final_labels(self) -> list[str]:
        """
        Labels of converged (non-mesh-pass) curves.

        Parameters:
            - None

        Returns:
            - list [str] : List of curve labels that do not contain 'Mesh Pass', indicating final results.
        """
        return [lbl for lbl in self.curves if 'Mesh Pass' not in lbl]

    @property
    def mesh_labels(self) -> list[str]:
        """
        Labels of mesh-adaptation-pass curves, sorted by pass number.

        Parameters:
            - None
        
        Returns:
            - list [str] : List of curve labels containing 'Mesh Pass', sorted in order of appearance (assumed pass number).
        """
        return sorted(lbl for lbl in self.curves if 'Mesh Pass' in lbl)

    def get_df(self, label: str) -> pd.DataFrame:
        """
        Return the DataFrame for the given curve label.

        Parameters:
            - label [str] : Curve label for which to retrieve the DataFrame.

        Returns:
            - pd.DataFrame : DataFrame corresponding to the specified curve label.
        """
        return self.curves[label]['df']

    def get_final_df(self) -> pd.DataFrame:
        """
        Return the DataFrame of the first converged (final) curve.

        Parameters:
            - None
        
        Returns:
            - pd.DataFrame : DataFrame of the first final curve found in the CST export file.
        """
        fl = self.final_labels
        if not fl:
            raise ValueError(f'No final curve found in {self.file_path!r}')
        return self.curves[fl[0]]['df']

    def get_meta(self, label: str) -> dict:
        """
        Return the metadata dictionary for the given curve label.

        Parameters:
            - label [str] : Curve label for which to retrieve metadata.
        
        Returns:
            - dict : Metadata dictionary corresponding to the specified curve label.
        """
        return self.curves[label]['meta']

    def __repr__(self) -> str:
        """
        String representation of the CSTParser instance, showing the file name and available curve labels.

        Parameters:
            - None

        Returns:
            - str : String representation of the CSTParser instance.
        """
        return (
            f'CSTParser("{os.path.basename(self.file_path)}", '
            f'curves={self.labels})'
        )
# =================================================================================================================================== #

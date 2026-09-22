# =================================================================================================================================== #
# ----------------------------------------------------------- DESCRIPTION ----------------------------------------------------------- #
# Keysight ADS "List" export parser.                                                                                                  #
# ADSListParser loads a .csv file exported from an ADS Data Display List (File > Export) and exposes its columns as a tidy            #
# DataFrame, with every engineering-notation value (e.g. "-1.035 mA", "100.0 MHz", "6.100 psec") converted to plain SI units.         #
# Author: Nedal M. Benelmekki                                                                                                         #
# =================================================================================================================================== #



# =================================================================================================================================== #
# --------------------------------------------------------- EXTERNAL IMPORTS -------------------------------------------------------- #
import os                        # Operating system interfaces.                                                                       #
import sys                       # System-specific parameters and functions.                                                          #
import csv                       # RFC-4180 CSV reading (handles the multi-line quoted ADS header row).                               #
import re                        # Regular expressions for value/unit splitting.                                                      #
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
# ------------------------------------------------------- ENGINEERING NOTATION ------------------------------------------------------ #
#                                                                                                                                      #
#  ADS Lists print every scalar in "value unit" form, auto-picking whichever SI prefix keeps the mantissa readable — the SAME         #
#  column can mix "21.49 uA" and "-1.035 mA" row to row.  ``_VALUE_UNIT_RE`` splits the mantissa from the unit string; the unit is     #
#  then resolved to a multiplier so every row lands in one consistent SI base unit (A, V, Hz, s, ...).                                #
#                                                                                                                                      #
_VALUE_UNIT_RE = re.compile(r'^([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*([a-zA-Zµ]*)$')                                               #
#                                                                                                                                      #
_SI_PREFIX = {                                                                                                                        #
    'f': 1e-15, 'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'µ': 1e-6,                                                                     #
    'm': 1e-3,  'k': 1e3,   'K': 1e3,  'M': 1e6,   'G': 1e9,  'T': 1e12,                                                              #
}                                                                                                                                      #
_BASE_UNITS = {'Hz', 'sec', 's', 'V', 'A', 'W', 'F', 'H', 'Ohm'}                                                                       #
#                                                                                                                                      #
INVALID_TOKENS = {'<invalid>', 'invalid', 'nan', ''}   # ADS marks failed solution points this way (e.g. near a pole).                #
# =================================================================================================================================== #



# =================================================================================================================================== #
# ------------------------------------------------------- MODULE-LEVEL HELPERS ------------------------------------------------------ #
def _unit_multiplier(unit: str) -> float | None:
    """
    Resolve an ADS unit string (e.g. 'mA', 'GHz', 'fsec') to its SI multiplier.

    Parameters:
        - unit [str] : Unit string with an optional single-letter SI prefix (e.g. 'm', 'G', 'f').

    Returns:
        - float | None : Multiplier to reach the SI base unit, or None if the unit is not recognised
                          (caller then leaves the value unscaled, e.g. a plain dB or ratio column).
    """
    if unit in _BASE_UNITS:
        return 1.0
    if len(unit) >= 2 and unit[0] in _SI_PREFIX and unit[1:] in _BASE_UNITS:
        return _SI_PREFIX[unit[0]]
    return None


def _split_value_unit(token: str) -> tuple[float, str]:
    """
    Split one ADS scalar token into its numeric value (scaled to SI base units) and the raw unit string.

    Parameters:
        - token [str] : Raw cell text, e.g. '-1.035 mA', '100.0 MHz', '62.205', '<invalid>'.

    Returns:
        - tuple[float, str] : (SI-scaled value, raw unit string). Value is NaN for '<invalid>' / unparsable tokens.
    """
    token = token.strip()
    if token in INVALID_TOKENS:
        return np.nan, ''

    m = _VALUE_UNIT_RE.match(token)
    if not m:
        return np.nan, ''

    num_str, unit = m.groups()
    value = float(num_str)
    if not unit:
        return value, ''

    mult = _unit_multiplier(unit)
    if mult is None:
        # Unrecognised unit (shouldn't happen for ADS exports) - keep the raw magnitude untouched.
        return value, unit
    return value * mult, unit
# =================================================================================================================================== #



# =================================================================================================================================== #
# --------------------------------------------------------- ADS LIST PARSER --------------------------------------------------------- #
class ADSListParser:
    """
    Parse a Keysight ADS Data Display "List" export (.csv).

    ADS writes one logical CSV row as the header: each field is itself a quoted, multi-line block::

        "freq
        References : ['dB(S(1,1))']
        Dependency : [freq]
        Num. Points : [32001]
        Matrix Size : scalar
        Type : Real"

    Only the first line of each header block (the trace/variable name) is kept as the column name.
    Every data value is parsed with ``_split_value_unit`` and converted to SI base units (Hz, s, V, A, ...);
    columns with no unit (dB, ratios, hFE, bare sweep values, harmonic index, ...) are left as plain floats.
    '<invalid>' cells (ADS's marker for a failed solution point, e.g. near a pole) become NaN.

    Attributes:
        - file_path [str]         : Path to the parsed .csv file.
        - columns   [list[str]]   : Column names, in file order.
        - units     [dict]        : column_name -> most common raw unit string seen for that column ('' if none/mixed-free).
        - df        [pd.DataFrame]: Parsed data, one row per exported point, values in SI base units.
    """

    def __init__(self, file_path: str) -> None:
        """
        Class initialization: parse the given ADS List .csv export and populate ``self.df``.

        Parameters:
            - file_path [str] : Absolute or relative path to the ADS-exported .csv file.

        Returns:
            - None
        """
        self.file_path = file_path
        self.columns: list[str] = []
        self.units:   dict      = {}
        self.df = pd.DataFrame()
        self._parse()

    # ──────────────────────────────────────────────────────────── parsing ─────
    def _parse(self) -> None:
        """
        Read ``self.file_path`` and populate ``self.columns``, ``self.units``, and ``self.df``.

        Parameters:
            - None

        Returns:
            - None
        """
        with open(self.file_path, newline='', encoding='utf-8-sig', errors='replace') as fh:
            reader = csv.reader(fh)
            try:
                header_row = next(reader)
            except StopIteration:
                return
            names = [cell.split('\n')[0].strip().rstrip('\r') for cell in header_row]

            raw_rows = [row for row in reader if any(c.strip() for c in row)]

        self.columns = names
        if not raw_rows:
            self.df = pd.DataFrame(columns=names)
            return

        raw = pd.DataFrame(raw_rows, columns=names)

        parsed  = {}
        units   = {}
        for name in names:
            values, cell_units = zip(*(_split_value_unit(tok) for tok in raw[name]))
            parsed[name] = values
            seen = [u for u in cell_units if u]
            units[name] = max(set(seen), key=seen.count) if seen else ''

        self.df    = pd.DataFrame(parsed, columns=names)
        self.units = units

    # ──────────────────────────────────────────────────────── data access ─────
    def groups(self, by: str):
        """
        Split the parsed data into sorted sub-DataFrames on an independent-variable column.

        Used for swept ADS exports where one column repeats block-wise (e.g. 11 'IB' values x 301 'VCE'
        points in a transistor IV sweep, or 7 'Vcc' bias points in a harmonic-balance sweep).

        Parameters:
            - by [str] : Column name to group by (e.g. 'IB', 'Vcc').

        Returns:
            - list[tuple[float, pd.DataFrame]] : (group value, sub-DataFrame) pairs, sorted by group value,
                                                   each sub-DataFrame re-indexed from 0.
        """
        return [
            (val, g.reset_index(drop=True))
            for val, g in sorted(self.df.groupby(by), key=lambda kv: kv[0])
        ]

    def rename(self, mapping: dict) -> 'ADSListParser':
        """
        Rename columns in place (ADS gives unhelpful names to some derived traces, e.g.
        'indep(__d, 1)' / 'plot_vs(pnmx, noisefreq)' for a phase-noise plot_vs() trace).

        Parameters:
            - mapping [dict] : {old_name: new_name}.

        Returns:
            - ADSListParser : self, for chaining.
        """
        self.df = self.df.rename(columns=mapping)
        self.columns = list(self.df.columns)
        self.units   = {mapping.get(k, k): v for k, v in self.units.items()}
        return self

    def __repr__(self) -> str:
        """
        String representation of the ADSListParser instance, showing the file name and parsed columns.

        Parameters:
            - None

        Returns:
            - str : String representation of the ADSListParser instance.
        """
        return (
            f'ADSListParser("{os.path.basename(self.file_path)}", '
            f'columns={self.columns}, rows={len(self.df)})'
        )
# =================================================================================================================================== #

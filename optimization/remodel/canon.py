"""
Utility canonicalization helpers shared across the remodel package.

Provides `canon(col, v)` to normalize dataset values (strings, None, areas)
and `qscore(x)` for quality ordering.
"""
from typing import Any
import math

QUAL_ORDER = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}


def qscore(x: Any) -> int:
    try:
        return QUAL_ORDER.get(str(x).strip(), 3)
    except Exception:
        return 3


def canon(col: str, v: Any):
    """Normalize common dataset columns.

    Behaviour consolidated from existing ad-hoc implementations in the repo:
    - central air normalization maps Y/N/YES/NO -> Yes/No
    - certain categorical nulls map to 'None'
    - empty Exterior 2nd -> 'NA'
    - area fields return floats (safe cast)
    """
    # Treat numeric NaN (pandas) as missing, and normalize to empty string
    if isinstance(v, float) and math.isnan(v):
        s = ""
    else:
        s = "" if v is None else str(v).strip()

    # Central Air: accept Y/N/YES/NO (any case)
    if col == "Central Air":
        su = s.upper()
        return {"Y": "Yes", "N": "No", "YES": "Yes", "NO": "No"}.get(su, s)

    # Fields that treat empty/NA as a semantic None
    if col in ("Mas Vnr Type", "Pool QC", "Alley", "Misc Feature", "Fence"):
        s_upper = s.upper()
        # treat pandas' 'nan' and other variants as missing
        return "None" if s_upper in ("", "NA", "NO APLICA", "NONE", "NAN") else s

    # Exterior 2nd: represent missing separately as 'NA'
    if col == "Exterior 2nd":
        return "NA" if s == "" else s

    # Areas: ensure floats
    if col in ["BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF", "Pool Area"]:
        try:
            return float(v or 0.0)
        except Exception:
            return 0.0

    # Default: return value unchanged (preserve original types when possible)
    return v

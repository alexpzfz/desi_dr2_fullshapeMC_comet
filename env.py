"""Shared environment bootstrap for scripts in this repo.

Adds this repo's ``src/`` directory and the sibling ``full-shape_wrap``
package to ``sys.path``, and exposes shared output-directory constants.
Import this after inserting the repo root onto ``sys.path``, e.g.:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import env  # noqa: F401
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
FULL_SHAPE_WRAP_DIR = Path("/global/u2/a/alexpzfz/full-shape_wrap")
CHAINS_DIR = REPO_ROOT / "outputs" / "chains"
PLOTS_DIR_CUTSKY_ABACUSHF = REPO_ROOT / "outputs" / "plots" / "cutsky_abacushf"

for _p in (SRC_DIR, FULL_SHAPE_WRAP_DIR):
    _p_str = str(_p)
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)

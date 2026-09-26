"""Shared environment bootstrap for scripts in this repo.

Adds this repo's ``src/`` directory and the ``full-shape_wrap`` (and, if
present, ``comet-emu``) checkouts to ``sys.path``, and exposes shared path
constants. Import this after inserting the repo root onto ``sys.path``, e.g.:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import env  # noqa: F401

Nothing here is machine specific. External locations default to paths
relative to the repo and can be overridden with environment variables:

- ``DESI_MC_DATA_DIR``: cached data vectors/covariances/windows (default:
  ``<repo>/data``, containing ``cutsky/`` and ``cubic/``). On NERSC, make
  ``<repo>/data`` a symlink to the shared copy; elsewhere, put a copy there.
- ``FULL_SHAPE_WRAP_DIR``: the full-shape_wrap checkout (default: a sibling
  ``full-shape_wrap`` directory next to this repo).
- ``COMET_EMU_DIR``: a comet-emu checkout to use instead of the installed
  ``comet`` package (default: a sibling ``comet-emu`` directory; skipped if it
  does not exist).
"""
import os
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"

# True on NERSC machines; the raw (uncached) measurements only live there.
ON_NERSC = "NERSC_HOST" in os.environ

DATA_DIR = Path(os.environ.get("DESI_MC_DATA_DIR", REPO_ROOT / "data"))
DATA_DIR_CUTSKY = DATA_DIR / "cutsky"
DATA_DIR_CUBIC = DATA_DIR / "cubic"

FULL_SHAPE_WRAP_DIR = Path(os.environ.get("FULL_SHAPE_WRAP_DIR", REPO_ROOT.parent / "full-shape_wrap"))
COMET_EMU_DIR = Path(os.environ.get("COMET_EMU_DIR", REPO_ROOT.parent / "comet-emu"))

CHAINS_DIR = REPO_ROOT / "outputs" / "chains"
PLOTS_DIR_CUTSKY_ABACUSHF = REPO_ROOT / "outputs" / "plots" / "cutsky_abacushf"
PLOTS_DIR_SCALE_CONFIG = REPO_ROOT / "outputs" / "plots" / "scale_config"

if not FULL_SHAPE_WRAP_DIR.is_dir():
    warnings.warn(f"full-shape_wrap not found at {FULL_SHAPE_WRAP_DIR}; clone it there "
                  "or set $FULL_SHAPE_WRAP_DIR.")

for _p in (COMET_EMU_DIR, SRC_DIR, FULL_SHAPE_WRAP_DIR):
    _p_str = str(_p)
    if _p.is_dir() and _p_str not in sys.path:
        sys.path.insert(0, _p_str)

# desi_dr2_fullshapeMC_comet

Fitting and plotting scripts for the DESI full-shape mock challenge analysis
(cubic-box and cutsky mocks, AbacusSummit / second-gen mocks), fit with the
COMET emulator.

## Layout

- `env.py` — shared bootstrap: adds `src/` and the sibling `full-shape_wrap`
  package to `sys.path`, and defines `CHAINS_DIR`. Imported by every script
  below via the standard preamble:
  ```python
  import sys
  from pathlib import Path
  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
  import env  # noqa: F401
  ```
- `src/` — analysis code:
  - `fit_*.py` — likelihood fits for the different mock sets (cubic box,
    cutsky AbacusHF, cutsky second-gen, Abacus secondary).
  - `read_data*.py`, `priors_mc.py` — data loading and prior definitions
    shared by the fit scripts.
  - `plot.py`, `plot_utils.py` — plotting utilities and scripts.
- `postprocessing/` (formerly `format_chains/`) — chain post-processing;
  writes to `postprocessing/out/`.
- `submit/` — SLURM job submission scripts. Each resolves its own repo root
  from `${BASH_SOURCE[0]}`, so they can be submitted (`sbatch`) from any
  working directory.
- `comparisons/giosue/` (formerly `compare_giosue/`) — comparison against
  Giosue's results.
- `outputs/` — generated artifacts:
  - `outputs/chains/` — symlink to chain output on `$PSCRATCH` (not tracked
    in git).
  - `outputs/plots/{cubic,cutsky_abacushf,cutsky_abacus2ndgen}/` — output
    plots.
  - `outputs/savedata/` — saved intermediate data.
  - `outputs/logs/` — job logs.
- `nb/` — exploratory notebooks and scratch scripts, left as-is; most content
  here is gitignored (see below) and its internal paths are not maintained.

## Notebooks

Jupyter notebooks (`*.ipynb`) are intentionally excluded from version control
(see `.gitignore`) to keep the repository lightweight and diff-friendly.

## Dependencies

Scripts import from the `full-shape_wrap` package (`observables`, `params`,
`likelihood`, `samplers`, `theory`, etc.). `env.py` hardcodes its location as
`/global/u2/a/alexpzfz/full-shape_wrap` — update that path if the package
moves.

# mock_challenge

Fitting and plotting scripts for the DESI full-shape mock challenge analysis
(cubic-box and cutsky mocks, AbacusSummit / second-gen mocks).

## Layout

- `fit_*.py` — likelihood fits for the different mock sets (cubic box, cutsky
  AbacusHF, cutsky second-gen, Abacus secondary).
- `read_data*.py`, `priors_mc.py` — data loading and prior definitions shared
  by the fit scripts.
- `plot.py`, `plot_utils.py` — plotting utilities and scripts.
- `submit_*.sh` — job submission scripts for NERSC.
- `format_chains/` — chain post-processing.
- `compare_giosue/` — comparison against Giosue's results.
- `plots_cubic/`, `plots_cutsky_abacushf/`, `plots_cutsky_abacus2ndgen/` — output plots.
- `savedata/`, `log/` — saved intermediate data and logs.
- `chains/` — symlink to chain output on `$PSCRATCH` (not tracked in git).
- `nb/` — exploratory notebooks (not tracked in git, see below).

## Notebooks

Jupyter notebooks (`*.ipynb`) are intentionally excluded from version control
(see `.gitignore`) to keep the repository lightweight and diff-friendly.

## Dependencies

These scripts import from the `full-shape_wrap` package (`observables`,
`params`, `likelihood`, `samplers`, `theory`, etc.) via a `sys.path` insert
relative to the script location. Since this repo was split out of
`full-shape_wrap/tmp/mock_challenge`, that relative path (`parents[2]`) no
longer resolves correctly and will need to be updated (e.g. to an explicit
path or an installed package import) for the fit/plot scripts to run.

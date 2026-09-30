# desi_dr2_fullshapeMC_comet

Fitting and plotting scripts for the DESI full-shape mock challenge analysis
(cubic-box and cutsky mocks, AbacusSummit / second-gen mocks), fit with the
COMET emulator.

## Layout

- `env.py` — shared bootstrap: adds `src/` and the sibling `full-shape_wrap`
  (and `comet-emu`, if present) checkouts to `sys.path`, and defines the path
  constants (`DATA_DIR`, `CHAINS_DIR`, ...). Imported by every script
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
- `submit/` — SLURM job submission scripts. The `run_*.sh` wrappers resolve
  the repo root from `${BASH_SOURCE[0]}`; the sbatch'd scripts ask Slurm
  (`scontrol`) for their original path, since Slurm runs a spool copy. Their
  `#SBATCH --output` is relative, so either submit from the repo root or pass
  `--output` (the wrappers do).
- `comparisons/giosue/` (formerly `compare_giosue/`) — comparison against
  Giosue's results.
- `data/` — cached data vectors, covariances and windows (`cutsky/`,
  `cubic/`). Not tracked in git; see [Setup](#setup).
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

## Setup

The repo has no machine-specific paths. After cloning:

1. **Cached data.** Put the cached data at `data/` in the repo root (it must
   contain `cutsky/` and `cubic/`). On NERSC, symlink the shared copy:
   ```bash
   ln -s /global/cfs/cdirs/desicollab/users/alexpzfz/DR2_2pt3pt/data_for_mock_challenge data
   ```
   Elsewhere, get it with `./sync_data.sh pull` (see
   [Syncing data and chains](#syncing-data-and-chains)). To keep it somewhere
   else, set `DESI_MC_DATA_DIR`.
2. **full-shape_wrap.** Clone it next to this repo
   (`../full-shape_wrap`), or set `FULL_SHAPE_WRAP_DIR`.
3. **comet.** Install `comet-emu` in your environment, or clone it next to
   this repo (`../comet-emu`) or set `COMET_EMU_DIR`, in which case that
   checkout takes precedence over the installed package.
4. **Chains output.** `outputs/chains/` is created on first use. On NERSC you
   may want it to be a symlink to `$PSCRATCH`.

## Syncing data and chains

`sync_data.sh` copies the cached data and the chains between NERSC and
another machine with rsync over ssh. Run it on the other machine. NERSC is
the source of truth for the data; chains can be run on either machine and
are synced both ways:

```bash
./sync_data.sh pull                 # data/ from NERSC (add/update only)
./sync_data.sh pull data --delete   # also remove local files gone from NERSC
./sync_data.sh pull chains          # outputs/chains/ from NERSC
./sync_data.sh pull all             # both
./sync_data.sh push chains          # local outputs/chains/ to NERSC
```

Extra arguments are passed to rsync (e.g. `--dry-run`). Chains are never
deleted on either side, and a file is not overwritten if the copy being
replaced is newer. Nautilus snapshots (`*_snap.hdf5`, `*_nautilus.hdf5`) are
skipped, as they are only checkpoints and make up most of the volume. Run the
script on a login node; compute nodes often have no internet access.

One-time setup on the other machine:

1. Get NERSC's `sshproxy.sh` (see the NERSC docs on MFA / sshproxy) and add
   the data transfer node to `~/.ssh/config`:
   ```
   Host dtn01.nersc.gov
       User <nersc-username>
       IdentityFile ~/.ssh/nersc
   ```
2. On NERSC, make sure `outputs/chains` exists in the repo (a symlink to
   `$PSCRATCH` is fine).

Then, once a day before syncing, run `./sshproxy.sh -u <nersc-username>` to
get a fresh 24-hour key.

Set `DESI_MC_NERSC_HOST` to use another ssh host, and `DESI_MC_NERSC_REPO`
if the repo is not at `/global/homes/a/alexpzfz/desi_dr2_fullshapeMC_comet`
on NERSC.

## Dependencies

Scripts import from the `full-shape_wrap` package (`observables`, `params`,
`likelihood`, `samplers`, `theory`, etc.) and from `comet`, plus the usual
scientific stack (numpy, scipy, jax, numba, matplotlib, getdist, nautilus,
iminuit, mpi4py for postprocessing).

**NERSC only (cosmodesi stack):** `cosmoprimo` is only needed for
`add_sigma8(engine='class'/'camb')` in the postprocessing (`engine='comet'`
does not need it). The AbacusSummit reference values (fiducial parameters,
true Omega_m/sigma8) are tabulated in `src/abacus_cosmologies.py`.
`lsstypes` and `clustering_statistics` (desi-clustering) are
needed only to read the raw measurements, i.e. `cached=False` in
`src/read_data.py` and running `src/read_data.py` as a script to rebuild the
cache. They are imported lazily, so the default cached path works without
them. `src/fit_cutsky_secondgen.py` always reads raw measurements and so
only runs on NERSC. The `cosmodesi_environment.sh` lines in `submit/` are
only sourced when `$NERSC_HOST` is set.

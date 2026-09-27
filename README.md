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
   [Syncing the cached data](#syncing-the-cached-data)). To keep it somewhere
   else, set `DESI_MC_DATA_DIR`.
2. **full-shape_wrap.** Clone it next to this repo
   (`../full-shape_wrap`), or set `FULL_SHAPE_WRAP_DIR`.
3. **comet.** Install `comet-emu` in your environment, or clone it next to
   this repo (`../comet-emu`) or set `COMET_EMU_DIR`, in which case that
   checkout takes precedence over the installed package.
4. **Chains output.** `outputs/chains/` is created on first use. On NERSC you
   may want it to be a symlink to `$PSCRATCH`.

## Syncing the cached data

The cache is built on NERSC (`python src/read_data.py`) and shared with
other clusters through a cloud remote (e.g. Nextcloud) using
[rclone](https://rclone.org). NERSC is the source of truth:

```bash
./sync_data.sh push            # on NERSC, after rebuilding the cache
./sync_data.sh pull            # elsewhere: add/update files in data/
./sync_data.sh pull --delete   # elsewhere: also remove files gone from the remote
```

Extra arguments are passed to rclone (e.g. `--dry-run`). `push` refuses to run
off NERSC, or if `data/cutsky` or `data/cubic` is empty, since it mirrors
deletions to the remote. Run `pull` on a login node; compute nodes often have
no internet access.

One-time setup on each machine:

1. Install rclone into your `PATH` (no root needed):
   ```bash
   curl -LO https://downloads.rclone.org/rclone-current-linux-amd64.zip
   unzip rclone-current-linux-amd64.zip && cp rclone-*-linux-amd64/rclone ~/.local/bin/
   ```
2. Run `rclone config` and create a remote named `nextcloud`: type `webdav`,
   URL `https://<your-nextcloud>/remote.php/dav/files/<username>/`, vendor
   `nextcloud`, and a Nextcloud app password (Settings → Security).

The default remote path is `nextcloud:desi_mc_data`; set
`DESI_MC_RCLONE_REMOTE` to use a different remote or folder.

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

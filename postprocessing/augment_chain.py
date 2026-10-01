"""Add derived cosmological parameters (sigma8, Omega_m) to a chain .h5 file.

Works on any chain produced by the fitting pipeline (see src/samplers.py) --
no hardcoded filenames or naming conventions. The chain is read and written
directly with h5py (no getdist): new parameters are appended as extra columns
of `points` (with matching `names` / `latex_names` entries), and every other
dataset and HDF5 attr (e.g. tracer, de_model, units -- see the `metadata`
passed to NautilusSampler.save / MinuitMinimizer.save) is carried over to
the output file untouched.

By default the file is augmented in place: the original is left untouched
until the new parameters have been fully computed, then safely overwritten.
Pass --output to write the augmented chain to a different path instead and
leave the input file alone.

Safety of the in-place default: before touching anything, a `.bak` copy of
the input file is made. The augmented chain is then written to a temporary
file and only moved over the original (os.replace, atomic on the same
filesystem) once that write has fully succeeded. The backup is deleted only
after the replace succeeds; if anything raises partway through, the original
file is untouched (or, in the unlikely case the replace itself is
interrupted, still recoverable from the `.bak` copy) and is never left
truncated.

sigma8 is the slow part (one CLASS/CAMB/emulator call per sample) and
should be run under MPI for any non-trivial chain:
    mpirun -n <N> python augment_chain.py <chain.h5> [chain2.h5 ...] [options]
From Python, augment_chain(..., pool=<multiprocessing.Pool>) parallelizes over
the pool's workers instead, from a single (non-MPI) process; this is what
src/fit_cutsky_abacushf.py --augment_chain uses right after a nautilus run.
The COMET emulator is only loaded when --engine comet is used.

Derived parameters are only computed for samples with non-zero weight. The
nautilus chains also store rejected points (log_weight = -inf, e.g. those
failing the w0 + wa < 0 conditional prior), which CLASS may not be able to
evaluate; those rows get NaN. For class/camb, samples on which the engine fails
are retried with fallbacks (see postprocess.SIGMA8_FALLBACKS); if all of them
fail the sample gets NaN and a warning is printed.

Already-present parameters are skipped by default (pass --force to
recompute them anyway). When sigma8 is (re)computed, the engine used
(comet/class/camb) is recorded as the 'sigma8_engine' attr on the output
file.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import env  # noqa: F401
import h5py
import numpy as np
from postprocess import COMM_WORLD, compute_Omega_m, compute_sigma8

comm = COMM_WORLD  # serial stand-in if mpi4py is not installed
rank = comm.Get_rank()

# name -> (latex label, fn(points, names, engine, pool) returning the column on rank 0)
DERIVED_PARAMS = {
    'sigma8': (r'\sigma_8', lambda points, names, engine, pool: compute_sigma8(points, names, comm=comm,
                                                                               engine=engine, pool=pool)),
    'Omega_m': (r'\Omega_m', lambda points, names, engine, pool: compute_Omega_m(points, names)),
}
CHAIN_DATASETS = ('points', 'names', 'latex_names')


def read_chain(chain_file):
    """Return (points, log_weights, names, labels, attrs) from a chain .h5 file."""
    with h5py.File(chain_file, 'r') as f:
        points = f['points'][:]
        log_weights = f['log_weights'][:]
        names = list(f['names'].asstr()[:])
        labels = list(f['latex_names'].asstr()[:])
        attrs = dict(f.attrs)
    return points, log_weights, names, labels, attrs


def write_chain(src_file, dst_file, points, names, labels, attrs):
    """Write `dst_file` as a copy of `src_file` with points/names/latex_names
    and attrs replaced; all other datasets are copied over as-is."""
    str_dtype = h5py.string_dtype(encoding='utf-8')
    with h5py.File(src_file, 'r') as src, h5py.File(dst_file, 'w') as dst:
        for key in src:
            if key not in CHAIN_DATASETS:
                src.copy(src[key], dst, name=key)
        dst.create_dataset('points', data=points)
        dst.create_dataset('names', data=names, dtype=str_dtype)
        dst.create_dataset('latex_names', data=labels, dtype=str_dtype)
        for key, value in attrs.items():
            dst.attrs[key] = value


def augment_chain(chain_file, params=('sigma8', 'Omega_m'), engine='comet', output=None,
                   force=False, keep_backup=False, pool=None):
    """Load `chain_file`, add any of `params` not already present as derived
    parameters, and write the result to `output` (default: overwrite
    `chain_file` in place, safely -- see module docstring).

    Without `pool`, all ranks of MPI.COMM_WORLD must call this together (the
    derived-param computation is parallelized across them); only rank 0 writes
    to disk. With `pool` (e.g. a multiprocessing.Pool), the computation is
    spread over its workers instead and this process does all the I/O.
    """
    is_root = pool is not None or rank == 0
    chain_file = Path(chain_file)
    in_place = output is None
    target = chain_file if in_place else Path(output)

    unknown = [p for p in params if p not in DERIVED_PARAMS]
    if unknown:
        raise ValueError(f"Unknown derived parameter(s) {unknown}. Available: {sorted(DERIVED_PARAMS)}")

    points, log_weights, names, labels, attrs = read_chain(chain_file)
    valid = np.isfinite(log_weights)
    if is_root and not valid.all():
        print(f"{chain_file.name}: {(~valid).sum()}/{valid.size} samples have zero weight, "
              "setting their derived parameters to NaN.")

    to_add = [p for p in params if force or p not in names]
    skipped = [p for p in params if p not in to_add]
    if is_root and skipped:
        print(f"{chain_file.name}: {skipped} already present, skipping (pass --force to recompute).")

    if not to_add:
        if is_root and not in_place:
            shutil.copy2(chain_file, target)
        return

    backup = None
    if is_root and in_place:
        backup = chain_file.with_name(chain_file.name + '.bak')
        shutil.copy2(chain_file, backup)

    try:
        for name in to_add:
            label, compute = DERIVED_PARAMS[name]
            valid_values = compute(points[valid], names, engine, pool)
            if not is_root:
                continue
            values = np.full(len(points), np.nan)
            values[valid] = valid_values
            n_nan = np.isnan(valid_values).sum()
            if n_nan:
                print(f"WARNING: {chain_file.name}: {name} could not be computed for {n_nan} "
                      f"sample(s) with non-zero weight; they are NaN in the output.")
            if name in names:  # --force: overwrite the existing column
                points[:, names.index(name)] = values
            else:
                points = np.column_stack([points, values])
                names.append(name)
                labels.append(label)
        if 'sigma8' in to_add:
            attrs = {**attrs, 'sigma8_engine': engine}

        if is_root:
            os.makedirs(target.parent, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(dir=target.parent, suffix='.h5')
            os.close(fd)
            try:
                write_chain(chain_file, tmp_path, points, names, labels, attrs)
                os.replace(tmp_path, target)
            except BaseException:
                os.remove(tmp_path)
                raise
            print(f"Saved augmented chain ({', '.join(to_add)}) to {target}")
    finally:
        if is_root and backup is not None:
            if keep_backup:
                print(f"Kept backup at {backup}")
            else:
                os.remove(backup)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('chain_files', type=str, nargs='+', help="Chain .h5 file(s) to augment.")
    parser.add_argument('--params', type=str, nargs='+', default=['sigma8', 'Omega_m'],
                         choices=sorted(DERIVED_PARAMS), help="Derived parameters to add.")
    parser.add_argument('--engine', type=str, default='comet', choices=['comet', 'class', 'camb'],
                         help="Engine used to compute sigma8 (see postprocess.add_sigma8).")
    parser.add_argument('--output', type=str, default=None,
                         help="Where to write the augmented chain(s). With a single input file, "
                              "a file path. With multiple input files, a directory (each chain is "
                              "written there under its original filename). Default: overwrite each "
                              "input file in place (safely, see module docstring).")
    parser.add_argument('--force', action='store_true',
                         help="Recompute --params even if already present in the chain.")
    parser.add_argument('--keep_backup', action='store_true',
                         help="Keep the '.bak' safety copy made before an in-place overwrite "
                              "instead of deleting it once the overwrite succeeds.")
    args = parser.parse_args()

    if args.output is not None and len(args.chain_files) > 1:
        output_dir = Path(args.output)
        if rank == 0:
            os.makedirs(output_dir, exist_ok=True)
        comm.Barrier()

    for chain_file in args.chain_files:
        chain_file = Path(chain_file)
        if args.output is None:
            output = None
        elif len(args.chain_files) > 1:
            output = Path(args.output) / chain_file.name
        else:
            output = args.output

        if rank == 0:
            print(f"Processing {chain_file}")
        augment_chain(chain_file, params=args.params, engine=args.engine, output=output,
                       force=args.force, keep_backup=args.keep_backup)

"""Add derived cosmological parameters (sigma8, Omega_m) to a chain .h5 file.

Works on any chain produced by the fitting pipeline (see src/samplers.py) --
no hardcoded filenames or naming conventions. The chain's HDF5 attrs (e.g.
tracer, de_model, units -- see the `metadata` passed to NautilusSampler.save
/ MinuitMinimizer.save) are read alongside the samples and carried over to
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
import plot_utils as pu
from postprocess import add_Omega_m, add_sigma8
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

DERIVED_PARAMS = {
    'sigma8': lambda samples, engine: add_sigma8(samples, comm=comm, engine=engine),
    'Omega_m': lambda samples, engine: add_Omega_m(samples, comm=comm),
}


def augment_chain(chain_file, params=('sigma8', 'Omega_m'), engine='comet', output=None,
                   force=False, keep_backup=False):
    """Load `chain_file`, add any of `params` not already present as derived
    parameters, and write the result to `output` (default: overwrite
    `chain_file` in place, safely -- see module docstring).

    All ranks of MPI.COMM_WORLD must call this together (the derived-param
    computation is parallelized across them); only rank 0 touches disk.
    Returns the augmented MCSamples object.
    """
    chain_file = Path(chain_file)
    in_place = output is None
    target = chain_file if in_place else Path(output)

    unknown = [p for p in params if p not in DERIVED_PARAMS]
    if unknown:
        raise ValueError(f"Unknown derived parameter(s) {unknown}. Available: {sorted(DERIVED_PARAMS)}")

    samples, attrs = pu.get_samples(str(chain_file), return_attrs=True)

    existing = set(samples.getParamNames().list())
    to_add = [p for p in params if force or p not in existing]
    skipped = [p for p in params if p not in to_add]
    if rank == 0 and skipped:
        print(f"{chain_file.name}: {skipped} already present, skipping (pass --force to recompute).")

    if not to_add:
        if rank == 0 and not in_place:
            shutil.copy2(chain_file, target)
        return samples

    backup = None
    if rank == 0 and in_place:
        backup = chain_file.with_name(chain_file.name + '.bak')
        shutil.copy2(chain_file, backup)

    try:
        for name in to_add:
            samples = DERIVED_PARAMS[name](samples, engine)
        if 'sigma8' in to_add:
            attrs = {**attrs, 'sigma8_engine': engine}

        if rank == 0:
            os.makedirs(target.parent, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(dir=target.parent, suffix='.h5')
            os.close(fd)
            try:
                pu.save_samples(samples, tmp_path, attrs=attrs)
                os.replace(tmp_path, target)
            except BaseException:
                os.remove(tmp_path)
                raise
            print(f"Saved augmented chain ({', '.join(to_add)}) to {target}")
    finally:
        if rank == 0 and backup is not None:
            if keep_backup:
                print(f"Kept backup at {backup}")
            else:
                os.remove(backup)

    return samples


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

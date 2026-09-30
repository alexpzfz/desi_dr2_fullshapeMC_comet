"""Postprocess the Mpc vs Mpc/h 'scale_config' chains.

For every (tracer, de_model) pair found in outputs/chains/scale_config:
  - loads the Mpc and Mpc/h chains,
  - adds sigma8 and Omega_m as derived parameters to each, in place, unless
    they're already present (see augment_chain.py; pass --force to bypass
    this cache and recompute anyway),
  - overplots the two unit conventions on a single triangle plot of the
    sampled cosmological parameters.

It then also produces, per de_model, figure-of-bias/figure-of-merit plots
comparing the two unit conventions across tracers, in two 3D parameter
spaces: {h, wc, As} and {h, Omega_m, sigma8}.

The sigma8/Omega_m computation is the slow part (one CLASS/CAMB/emulator
call per sample) and should be run under MPI (see
submit/submit_post_scale_config.sh), not on a login node. Once the chains
already carry sigma8/Omega_m, rerunning this script to tweak a plot is
cheap and can be done directly on a login node.
"""
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import env  # noqa: F401
import plot_utils as pu
from augment_chain import augment_chain
from fit_cutsky_abacushf import tracer_label_dict
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

TRACER_ORDER = list(tracer_label_dict.keys())

FILENAME_RE = re.compile(
    r'^Abacus-hf-dr2-v2-altmtl_(?P<tracer>.+?)_(?P<region>GCcomb)_(?P<freedom>\w+?)freedom_'
    r'(?P<de_model>lambda|w0wa|w0)_pk_dk(?P<dkP>[0-9.]+)_kmax(?P<kmaxP>[0-9.\-]+)_fullreparam'
    r'(?:_bk_dk(?P<dkB>[0-9.]+)_kmax(?P<kmaxB>[0-9.\-]+))?'
    r'(?P<mpch>_Mpch)?$'
)

FOB_FOM_PARAM_SETS = {
    'h_wc_As': ['h', 'wc', 'As'],
    'h_Om_s8': ['h', 'Omega_m', 'sigma8'],
}


def get_fob_fom_param_sets(de_model):
    """FoB/FoM parameter sets for this de_model: the base 3D sets, extended
    with w0 and wa (5D) when de_model == 'w0wa'."""
    sets = {name: list(params) for name, params in FOB_FOM_PARAM_SETS.items()}
    if de_model == 'w0wa':
        for params in sets.values():
            params += ['w0', 'wa']
    return sets


def discover_chains(chain_dir):
    """Group chain .h5 files in chain_dir by (tracer, de_model), pairing the
    Mpc and Mpc/h unit variants of the same run."""
    groups = defaultdict(dict)
    for fn in sorted(chain_dir.glob('*.h5')):
        if fn.stem.endswith('_derived'):
            # A previously-saved augmented chain (see load_and_augment); not
            # a raw chain to discover on its own.
            continue
        m = FILENAME_RE.match(fn.stem)
        if m is None:
            print(f"Skipping unrecognized chain filename: {fn.name}")
            continue
        key = (m['tracer'], m['de_model'])
        unit = 'mpch' if m['mpch'] else 'mpc'
        groups[key][unit] = fn
    return groups


def get_cosmo_params_to_plot(de_model):
    params = ['wb', 'wc', 'h', 'ns', 'log10As']
    if de_model in ('w0', 'w0wa'):
        params.append('w0')
    if de_model == 'w0wa':
        params.append('wa')
    return params


def load_and_augment(files, engine='comet', output_dir=None, force=False):
    """Load the Mpc/Mpc-h chains for one (tracer, de_model) group, adding the
    sigma8 and Omega_m derived parameters to each via augment_chain.

    By default this augments each chain file in place (safely -- see
    augment_chain.augment_chain), so a chain that already carries
    sigma8/Omega_m is loaded as-is and left untouched; pass force=True to
    recompute anyway. Pass output_dir to instead write the augmented chains
    there, leaving the original chain files untouched."""
    samples = {}
    for unit, fn in files.items():
        if not fn.exists():
            print(f"File not found: {fn}")
            continue
        output = (output_dir / fn.name) if output_dir is not None else None
        samples[unit] = augment_chain(fn, engine=engine, output=output, force=force)
    return samples


def plot_units_contour(tracer, de_model, samples, plot_dir):
    if 'mpc' not in samples or 'mpch' not in samples:
        return
    if rank != 0:
        return
    params = get_cosmo_params_to_plot(de_model) + ['sigma8', 'Omega_m']
    g = pu.plot_triangle([samples['mpc'], samples['mpch']], params,
                          labels=['Mpc', 'Mpc/h'], filled=[True, False], contour_colors=['orange', 'C0'], 
                          contour_lws=[2, 2],
                          contour_ls=['-', '--'])
    g.fig.suptitle(f'{tracer}', fontsize=20, y=1.02)
    os.makedirs(plot_dir, exist_ok=True)
    out_fn = plot_dir / f'{tracer}_{de_model}_units_contours.png'
    g.fig.savefig(out_fn, dpi=150, bbox_inches='tight')
    plt.close(g.fig)
    print(f"Saved contour plot to {out_fn}")


def plot_units_fob_fom(de_model, tracer_samples, plot_dir):
    tracers = [t for t in TRACER_ORDER if t in tracer_samples
               and 'mpc' in tracer_samples[t] and 'mpch' in tracer_samples[t]]
    if not tracers or rank != 0:
        return
    os.makedirs(plot_dir, exist_ok=True)
    for set_name, params in get_fob_fom_param_sets(de_model).items():
        mpc_list = [tracer_samples[t]['mpc'] for t in tracers]
        mpch_list = [tracer_samples[t]['mpch'] for t in tracers]
        fig, axes = pu.plot_fob_fom([mpc_list, mpch_list], params, xlabels=tracers,
                                     samples_labels=['Mpc', 'Mpc/h'], cosmo_true='c000')
        # No fig.suptitle here: plot_fob_fom's own per-panel titles already
        # state the parameter space, and a suptitle would collide with its
        # figure-level legend.
        out_fn = plot_dir / f'fobfom_{de_model}_{set_name}.png'
        fig.savefig(out_fn, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved FoM/FoB plot to {out_fn}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--chain_dir', type=str, default=str(env.CHAINS_DIR / 'scale_config'))
    parser.add_argument('--plot_dir', type=str, default=str(env.PLOTS_DIR_SCALE_CONFIG))
    parser.add_argument('--output_dir', type=str, default=None,
                         help="Where to write the chains augmented with sigma8/Omega_m. "
                              "Default: augment each chain file in --chain_dir in place, "
                              "safely (see augment_chain.py), instead of writing a new file.")
    parser.add_argument('--engine', type=str, default='comet', choices=['comet', 'class', 'camb'],
                         help="Engine used to compute sigma8 (see postprocess.add_sigma8).")
    parser.add_argument('--force', action='store_true',
                         help="Recompute sigma8/Omega_m even if a chain already has them.")
    args = parser.parse_args()

    chain_dir = Path(args.chain_dir)
    plot_dir = Path(args.plot_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None

    groups = discover_chains(chain_dir)
    if not groups:
        print(f"No recognized chain files found in {chain_dir}")

    # all_samples[de_model][tracer] = {'mpc': samples, 'mpch': samples}
    all_samples = defaultdict(dict)
    for (tracer, de_model), files in sorted(groups.items()):
        if rank == 0:
            print(f"Processing {tracer}, {de_model}: {sorted(files)}")
        samples = load_and_augment(files, engine=args.engine, output_dir=output_dir, force=args.force)
        all_samples[de_model][tracer] = samples
        plot_units_contour(tracer, de_model, samples, plot_dir)

    for de_model, tracer_samples in all_samples.items():
        plot_units_fob_fom(de_model, tracer_samples, plot_dir)

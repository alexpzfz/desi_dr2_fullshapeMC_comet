"""Compare 'bispec_ext' scale_config chains against their non-'bispec_ext'
counterpart.

For every chain .h5 file in outputs/chains/scale_config whose name contains
'bispec_ext', this looks for the counterpart file with that suffix removed
(same tracer/de_model/units). Where both exist, it overplots their sampled
cosmological parameters on a single triangle plot, labelling each with the
wall-clock time it took to run (the 'elapsed_minutes' attribute saved by
fit_cutsky_abacushf.py into the chain's .h5 attrs).

This is a plain, single-process script -- unlike post_scale_config.py it does
not compute any derived parameters, so it doesn't need MPI and can be run
directly on a login node.
"""
import re
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import env  # noqa: F401
import plot_utils as pu

FILENAME_RE = re.compile(
    r'^Abacus-hf-dr2-v2-altmtl_(?P<tracer>.+?)_(?P<region>GCcomb)_(?P<freedom>\w+?)freedom_'
    r'(?P<de_model>lambda|w0wa|w0)_pk_dk(?P<dkP>[0-9.]+)_kmax(?P<kmaxP>[0-9.\-]+)_fullreparam'
    r'(?:_bk_dk(?P<dkB>[0-9.]+)_kmax(?P<kmaxB>[0-9.\-]+))?'
    r'(?P<mpch>_Mpch)?'
    r'(?P<ext>_bispec_ext)?$'
)


def get_cosmo_params_to_plot(de_model):
    params = ['wb', 'wc', 'h', 'ns', 'log10As']
    if de_model in ('w0', 'w0wa'):
        params.append('w0')
    if de_model == 'w0wa':
        params.append('wa')
    return params


def get_elapsed_minutes(fn):
    with h5py.File(fn, 'r') as f:
        return float(f.attrs['elapsed_minutes'])


def find_pairs(chain_dir):
    """Pair every '*_bispec_ext.h5' chain in chain_dir with its counterpart
    file (same name, '_bispec_ext' suffix removed), skipping any for which
    that counterpart doesn't exist."""
    pairs = []
    for fn in sorted(chain_dir.glob('*_bispec_ext.h5')):
        m = FILENAME_RE.match(fn.stem)
        if m is None:
            print(f"Skipping unrecognized chain filename: {fn.name}")
            continue
        counterpart = fn.with_name(fn.name.replace('_bispec_ext.h5', '.h5'))
        if not counterpart.exists():
            print(f"No non-bispec_ext counterpart for {fn.name}; skipping.")
            continue
        pairs.append((fn, counterpart, m['tracer'], m['de_model'], bool(m['mpch'])))
    return pairs


def plot_bispec_ext_comparison(bispec_ext_fn, counterpart_fn, tracer, de_model, mpch, plot_dir):
    samples_ext = pu.get_samples(str(bispec_ext_fn))
    samples_base = pu.get_samples(str(counterpart_fn))
    t_ext = get_elapsed_minutes(bispec_ext_fn)
    t_base = get_elapsed_minutes(counterpart_fn)

    params = get_cosmo_params_to_plot(de_model)
    labels = [f'bispec\\_ext ({t_ext:.1f} min)', f'baseline ({t_base:.1f} min)']
    g = pu.plot_triangle([samples_ext, samples_base], params, labels=labels,
                          filled=[True, False], contour_colors=['orange', 'C0'],
                          contour_lws=[2, 2], contour_ls=['-', '--'])
    g.fig.suptitle(f'{tracer}', fontsize=20, y=1.02)

    plot_dir.mkdir(parents=True, exist_ok=True)
    suffix = '_Mpch' if mpch else ''
    out_fn = plot_dir / f'{tracer}_{de_model}{suffix}_bispec_ext_contours.png'
    g.fig.savefig(out_fn, dpi=150, bbox_inches='tight')
    plt.close(g.fig)
    print(f"Saved comparison plot to {out_fn}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--chain_dir', type=str, default=str(env.CHAINS_DIR / 'scale_config'))
    parser.add_argument('--plot_dir', type=str, default=str(env.PLOTS_DIR_SCALE_CONFIG))
    args = parser.parse_args()

    chain_dir = Path(args.chain_dir)
    plot_dir = Path(args.plot_dir)

    pairs = find_pairs(chain_dir)
    if not pairs:
        print(f"No bispec_ext/counterpart pairs found in {chain_dir}")

    for bispec_ext_fn, counterpart_fn, tracer, de_model, mpch in pairs:
        print(f"Plotting {tracer}, {de_model}{' (Mpc/h)' if mpch else ''}: "
              f"{bispec_ext_fn.name} vs {counterpart_fn.name}")
        plot_bispec_ext_comparison(bispec_ext_fn, counterpart_fn, tracer, de_model, mpch, plot_dir)

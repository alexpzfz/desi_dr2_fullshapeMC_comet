"""Compare N sets of chains case by case.

Each set is a directory of chains (.h5) plus a legend label, given with
--set DIR LABEL. Every chain filename is turned into a match key by applying
that set's (optional) rules, in this order:
  --filter REGEX     only keep chains whose stem matches REGEX (re.search),
  --strip SUFFIX     remove SUFFIX from the end of the stem,
  --sub REGEX REPL   re.sub(REGEX, REPL, stem); can be given several times.
Those options apply to the --set given immediately before them. Chains whose
key is present in every set are matched up; the others are skipped.

For every matched case this script:
  - overplots the N chains on triangle plots of {h, wc, log10As} and
    {h, Omega_m, sigma8} (plus w0, wa for w0wa chains),
  - overplots them on a separate triangle plot of the nuisance parameters.

It then also produces, per configuration (i.e. everything in the key except
the tracer: de_model, units, scale cuts, ...), figure-of-bias/figure-of-merit
plots comparing the N sets across all tracers available for that
configuration, in the same parameter spaces.

Examples:
    # corrected vs original scale_config chains
    python post_compare_chains.py \\
        --set scale_config Original \\
        --set scale_config_corrected Corrected --strip _corrected

    # P-only vs P+B, only for zgeom_Mpch chains
    python post_compare_chains.py \\
        --set pk_only 'P' \\
        --set scale_config 'P+B' --sub '_bk_dk[0-9.]+_kmax[0-9.\\-]+' '' --filter zgeom_Mpch

The chains are expected to already carry sigma8/Omega_m (see
augment_chain.py), so no derived parameters are computed here and this can
be run directly on a login node.
"""
import argparse
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import env  # noqa: F401
import plot_utils as pu
from fit_cutsky_abacushf import tracer_label_dict

TRACER_ORDER = list(tracer_label_dict.keys())

# Splits a match key into the tracer and the rest of the configuration.
FILENAME_RE = re.compile(r'^(?P<prefix>[^_]+)_(?P<tracer>.+?)_(?P<config>GCcomb_.+)$')
DE_MODEL_RE = re.compile(r'_(?P<de_model>lambda|w0wa|w0)_')
CONFIG_RE = re.compile(
    r'^GCcomb_(?P<freedom>\w+?)freedom_(?P<de_model>lambda|w0wa|w0)'
    r'_pk_dk(?P<dkP>[0-9.]+)_kmax(?P<kmaxP>[0-9.\-]+)_(?P<prior>[a-z]+)'
    r'(?:_bk_dk(?P<dkB>[0-9.]+)_kmax(?P<kmaxB>[0-9.\-]+))?'
    r'_(?P<zeff>z\w+?)(?P<mpch>_Mpch)?(?:_(?P<extra>.+))?$'
)
DE_MODEL_LABELS = {'lambda': r'$\Lambda$CDM', 'w0': r'$w_0$CDM', 'w0wa': r'$w_0w_a$CDM'}

COSMO_PARAM_SETS = {
    'h_wc_log10As': ['h', 'wc', 'log10As'],
    'h_Om_s8': ['h', 'Omega_m', 'sigma8'],
}
NUISANCE_PARAMS = ['NP0_r', 'NB0_r', 'MB0_r', 'avir']

LINESTYLES = ['-', '--', ':', '-.']


def get_colors(n):
    return [f'C{i}' for i in range(n)]


def get_cosmo_param_sets(de_model):
    """Contour/FoB/FoM parameter sets for this de_model: the base 3D sets,
    extended with w0 and wa (5D) when de_model == 'w0wa'."""
    sets = {name: list(params) for name, params in COSMO_PARAM_SETS.items()}
    if de_model == 'w0wa':
        for params in sets.values():
            params += ['w0', 'wa']
    return sets


def resolve_dir(d):
    """A set directory is taken as given if it exists, otherwise relative to
    env.CHAINS_DIR."""
    p = Path(d)
    return p if p.is_dir() else env.CHAINS_DIR / d


def match_key(stem, chain_set):
    if chain_set['filter'] and not re.search(chain_set['filter'], stem):
        return None
    strip = chain_set['strip']
    if strip:
        if not stem.endswith(strip):
            return None
        stem = stem[:-len(strip)]
    for pattern, repl in chain_set['subs']:
        stem = re.sub(pattern, repl, stem)
    return stem


def collect_set(chain_set):
    """Return {key: path} for every chain of a set."""
    chain_dir = resolve_dir(chain_set['dir'])
    if not chain_dir.is_dir():
        sys.exit(f"Chain directory not found: {chain_set['dir']}")
    files = {}
    for fn in sorted(chain_dir.glob('*.h5')):
        key = match_key(fn.stem, chain_set)
        if key is None:
            continue
        if key in files:
            print(f"[{chain_set['label']}] {fn.name} and {files[key].name} both map to key {key}, "
                  f"keeping the latter.")
            continue
        files[key] = fn
    print(f"[{chain_set['label']}] {len(files)} chains in {chain_dir}")
    return files


def discover_matches(chain_sets):
    """Return matches[config][tracer] = [path for each set] for every key
    present in all sets."""
    files_per_set = [collect_set(s) for s in chain_sets]
    common = set.intersection(*(set(f) for f in files_per_set))
    for s, files in zip(chain_sets, files_per_set):
        n_unmatched = len(set(files) - common)
        if n_unmatched:
            print(f"[{s['label']}] {n_unmatched} chains without a match in every other set, skipping them.")
    matches = defaultdict(dict)
    for key in sorted(common):
        m = FILENAME_RE.match(key)
        if m is None:
            print(f"Skipping unrecognized chain key: {key}")
            continue
        matches[m['config']][m['tracer']] = [files[key] for files in files_per_set]
    return matches


def get_de_model(config):
    m = DE_MODEL_RE.search(config)
    return m['de_model'] if m else 'lambda'


def get_config_title(config):
    """Human-readable description of a configuration (the key minus the
    tracer), e.g. 'w0waCDM, Mpc/h, zgeom, interm freedom, fullreparam' on the
    first line and the P/B scale cuts on the second."""
    m = CONFIG_RE.match(config)
    if m is None:
        return config.replace('_', r'\_')
    first = [DE_MODEL_LABELS.get(m['de_model'], m['de_model']),
             r'Mpc$/h$' if m['mpch'] else 'Mpc',
             m['zeff'],
             f"{m['freedom']} freedom",
             m['prior']]
    second = [rf"$P$: $k_{{\rm max}}={m['kmaxP'].replace('-', ', ')}$"]
    if m['kmaxB']:
        second.append(rf"$B$: $k_{{\rm max}}={m['kmaxB'].replace('-', ', ')}$")
    if m['extra']:
        second.append(m['extra'].replace('_', r'\_'))
    return ', '.join(first) + '\n' + ', '.join(second)


def plot_triangle(samples, labels, params, title, out_fn):
    missing = [p for p in params for s in samples if p not in s.getParamNames().list()]
    if missing:
        print(f"Skipping {out_fn.name}: missing parameters {sorted(set(missing))}")
        return
    n = len(samples)
    g = pu.plot_triangle(samples, params, labels=labels, filled=[i == 0 for i in range(n)],
                          contour_colors=get_colors(n), contour_lws=[2] * n,
                          contour_ls=[LINESTYLES[i % len(LINESTYLES)] for i in range(n)],
                          legend_loc='upper right')
    g.fig.suptitle(title, fontsize=20, y=1.05)
    g.fig.savefig(out_fn, dpi=150, bbox_inches='tight')
    plt.close(g.fig)
    print(f"Saved contour plot to {out_fn}")


def plot_contours(tracer, config, samples, labels, plot_dir):
    title = f'{tracer}: {get_config_title(config)}'
    for set_name, params in get_cosmo_param_sets(get_de_model(config)).items():
        plot_triangle(samples, labels, params, title, plot_dir / f'{tracer}_{config}_{set_name}_contours.png')
    plot_triangle(samples, labels, NUISANCE_PARAMS, title, plot_dir / f'{tracer}_{config}_nuisance_contours.png')


def plot_config_fob_fom(config, tracer_samples, labels, plot_dir):
    tracers = [t for t in TRACER_ORDER if t in tracer_samples]
    tracers += sorted(t for t in tracer_samples if t not in TRACER_ORDER)
    if not tracers:
        return
    for set_name, params in get_cosmo_param_sets(get_de_model(config)).items():
        missing = [p for p in params for t in tracers for s in tracer_samples[t]
                   if p not in s.getParamNames().list()]
        if missing:
            print(f"Skipping FoB/FoM {config} {set_name}: missing parameters {sorted(set(missing))}")
            continue
        samples_lists = [[tracer_samples[t][i] for t in tracers] for i in range(len(labels))]
        fig, axes = pu.plot_fob_fom(samples_lists, params, xlabels=tracers, samples_labels=labels,
                                     colors=get_colors(len(labels)), cosmo_true='c000')
        # A fig.suptitle would collide with plot_fob_fom's figure-level
        # legend, so the configuration goes in as the legend's title instead.
        fig.legends[0].set_title(get_config_title(config), prop={'size': 10})
        out_fn = plot_dir / f'fobfom_{config}_{set_name}.png'
        fig.savefig(out_fn, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved FoM/FoB plot to {out_fn}")


class AddSet(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        sets = getattr(namespace, self.dest) or []
        sets.append({'dir': values[0], 'label': values[1], 'strip': None, 'subs': [], 'filter': None})
        setattr(namespace, self.dest, sets)


class SetOption(argparse.Action):
    """Attach an option to the most recently given --set."""
    def __call__(self, parser, namespace, values, option_string=None):
        if not namespace.sets:
            parser.error(f'{option_string} must come after a --set')
        chain_set = namespace.sets[-1]
        if self.dest == 'subs':
            chain_set['subs'].append(tuple(values))
        else:
            chain_set[self.dest] = values


def default_plot_dir(labels):
    tag = '_vs_'.join(re.sub(r'[^\w.+-]+', '-', label).strip('-') for label in labels)
    return env.PLOTS_DIR_SCALE_CONFIG.parent / 'comparisons' / tag


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--set', dest='sets', nargs=2, metavar=('DIR', 'LABEL'), action=AddSet, default=None,
                        help="Chain directory (absolute, or relative to outputs/chains) and its legend label. "
                             "Give it once per set; the first set is drawn filled.")
    parser.add_argument('--strip', dest='strip', metavar='SUFFIX', action=SetOption,
                        help="Suffix removed from the stems of the preceding set (chains without it are ignored).")
    parser.add_argument('--sub', dest='subs', nargs=2, metavar=('REGEX', 'REPL'), action=SetOption,
                        help="re.sub applied to the stems of the preceding set; repeatable.")
    parser.add_argument('--filter', dest='filter', metavar='REGEX', action=SetOption,
                        help="Only keep chains of the preceding set whose stem matches REGEX.")
    parser.add_argument('--plot_dir', type=str, default=None,
                        help="Default: outputs/plots/comparisons/<label1>_vs_<label2>...")
    parser.add_argument('--no_contours', action='store_true', help="Only make the FoB/FoM plots.")
    args = parser.parse_args()

    if not args.sets or len(args.sets) < 2:
        parser.error('give at least two --set')
    labels = [s['label'] for s in args.sets]

    plot_dir = Path(args.plot_dir) if args.plot_dir else default_plot_dir(labels)
    os.makedirs(plot_dir, exist_ok=True)

    matches = discover_matches(args.sets)
    if not matches:
        print("No chains matched across all sets.")

    # all_samples[config][tracer] = [samples for each set]
    all_samples = defaultdict(dict)
    for config, tracer_files in sorted(matches.items()):
        for tracer, files in sorted(tracer_files.items()):
            print(f"Processing {tracer}, {config}")
            samples = [pu.get_samples(fn) for fn in files]
            all_samples[config][tracer] = samples
            if not args.no_contours:
                plot_contours(tracer, config, samples, labels, plot_dir)

    for config, tracer_samples in all_samples.items():
        plot_config_fob_fom(config, tracer_samples, labels, plot_dir)

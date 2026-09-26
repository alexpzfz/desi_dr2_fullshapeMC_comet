import os
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import env  # noqa: F401

import plot_utils as pu
from postprocess import export_to_text
from abacus_cosmologies import get_abacus_cosmology
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()


plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def get_true_markers(cosmo_name):
    """Return extra reference markers for the chosen AbacusSummit cosmology."""
    abacus = get_abacus_cosmology(cosmo_name)
    return {
        'Omega_m': abacus['Omega_m'],
        'Omega_b': abacus['Omega_b'],
        'sigma8': abacus['sigma8_cb'],
    }


def _parse_chain_stem(stem):
    match = re.match(
        r'^Abacus-hf-cubic_(?P<tracer>[^_]+)_(?P<hod>[^_]+)hod_(?P<cosmo>c\d{3})_(?P<freedom>[^_]+)freedom_(?P<de_model>lambda|w0wa)_pk_kmax(?P<kmaxP>[0-9.\-]+?)(?P<reparam>_fullreparam|_hybridreparam|_noreparam)?(?P<bispec>_bk(?:_kmax[0-9.\-]+)?)?(?P<extra>_.+)?$',
        stem,
    )
    if match is None:
        return None
    return match.groupdict()


def process_chain(chain_file):
    parsed = _parse_chain_stem(chain_file.stem)
    if parsed is None:
        print(f'Skipping unrecognized chain filename: {chain_file.name}')
        return

    tracer = parsed['tracer']
    hod = parsed['hod']
    cosmo_name = parsed['cosmo']
    de_model = parsed['de_model']
    markers = get_true_markers(cosmo_name)

    if '_bk' in chain_file.stem:
        suffix = 'pkbk'
    else:
        suffix = 'pk'

    try:
        samples = pu.get_samples(str(chain_file))
    except FileNotFoundError:
        print(f'File not found: {chain_file}')
        return

    print(f'Loaded samples for {tracer}, {hod}, {cosmo_name}, {de_model}, {suffix}')

    os.makedirs('./out', exist_ok=True)
    out_fn = f'./out/{chain_file.stem}.txt'

    samples = export_to_text(samples, out_fn, engine='comet', comm=comm)

    if rank == 0:
        params = ['Omega_m', 'h', 'sigma8']
        if de_model == 'w0wa':
            params += ['w0', 'wa']
        g = pu.plot_triangle(samples, params_to_plot=params, cosmo_true=cosmo_name, extra_markers=markers)
        g.fig.savefig(f'{chain_file.stem}_triangle.png')


if __name__ == '__main__':
    out_dir = env.CHAINS_DIR
    chain_files = sorted(out_dir.glob('Abacus-hf-cubic_*.h5'))

    if not chain_files:
        print(f'No cubic chain files found in {out_dir}')

    for chain_file in chain_files:
        process_chain(chain_file)
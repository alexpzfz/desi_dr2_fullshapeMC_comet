import os
import sys

# Must be set before `import jax`.
#
# JAX's CPU (XLA) backend does NOT expose a thread-count knob through
# XLA_FLAGS in this jaxlib build -- empirically verified: neither
# --xla_cpu_multi_thread_eigen nor --xla_cpu_use_xnnpack changes anything
# (measured cpu_time/wall_time ratio ~32 either way on this node). What
# actually sizes its execution thread pool is os.sched_getaffinity(0): the
# ratio dropped from ~32 to ~1.0/2.9/5.8 when restricting the process to
# 1/4/8 cores via taskset. So the real lever against nautilus's pool
# processes all fighting over every core is CPU affinity (see
# _pin_to_n_cores below and its use at the pool/sampler construction and
# in --time_likelihood), not any XLA_FLAGS setting.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
# The cosmodesiconda env ships the xla_cuda12 PJRT plugin alongside the CPU
# one. JAX unconditionally probes every registered plugin at backend
# discovery time (before JAX_PLATFORMS is even consulted), so on a
# --constraint=cpu node with no GPU driver the cuda12 plugin's cuInit()
# check always fails. That failure is caught internally by JAX
# (xla_bridge.discover_pjrt_plugins uses a bare `except:` around
# plugin_module.initialize()) and is harmless -- it just logs a scary
# traceback via logging.exception and moves on. Silence that one logger
# so it stops cluttering the slurm log; JAX_PLATFORMS=cpu below still
# ensures JAX only *uses* the cpu backend either way.
import logging
logging.getLogger('jax._src.xla_bridge').setLevel(logging.CRITICAL)
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import json
import hashlib
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

import lsstypes as types
from clustering_statistics.tools import get_stats_fn
import matplotlib.pyplot as plt
from lsstypes import ObservableTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import env  # noqa: F401
sys.path.insert(0, "/global/homes/a/alexpzfz/comet-emu")
from observables import PowerSpectrumMultipoles, BispectrumSugiyamaMultipoles, JointObservable
from params import Params
from likelihood import Likelihood
from samplers import NautilusSampler, MinuitMinimizer
from theory import COMET
from read_data import get_obs_pk, get_obs_pk_bk
from priors_mc import get_pars
import plot_utils as pu



tracer_label_dict = {'BGS': {'tracer': 'BGS', 'zrange': (0.1, 0.4)},
                     'LRG1': {'tracer': 'LRG', 'zrange': (0.4, 0.6)},
                     'LRG2': {'tracer': 'LRG', 'zrange': (0.6, 0.8)},
                     'LRG3': {'tracer': 'LRG', 'zrange': (0.8, 1.1)},
                     'ELG1': {'tracer': 'ELG', 'zrange': (0.8, 1.1)},
                     'ELG2': {'tracer': 'ELG', 'zrange': (1.1, 1.6)},
                     'QSO': {'tracer': 'QSO', 'zrange': (0.8, 2.1)}}

zsnap_dict = {'BGS': {(0.1, 0.4): 0.300},
              'LRG': {(0.4, 0.6): 0.500, (0.6, 0.8): 0.725, (0.8, 1.1): 0.950},
              'ELG': {(0.8, 1.1): 0.950, (1.1, 1.6): 1.475},
              'QSO': {(0.8, 2.1): 1.550}}


def _fmt_float(x):
    return f"{float(x):.3f}".rstrip('0').rstrip('.')


def get_fn(tracer_label, region, freedom, dkP, kmaxP, bispec=False, dkB=None, kmaxB=None,
           de_model='lambda', reparam_option='full', free_Mnu=False, outdir=str(env.CHAINS_DIR), extra=None,
           counterterm_basis='DESIct', avirB_free=False, use_Mpc=True, zeff_choice='zsnap'):

    if not isinstance(tracer_label, list):
        tracer_label = [tracer_label] 

    if isinstance(kmaxP, list):
        kmaxP_str = '-'.join([_fmt_float(k) for k in kmaxP])
    else:
        kmaxP_str = _fmt_float(kmaxP)

    tracer_str = '-'.join(tracer_label) if len(tracer_label) > 1 else tracer_label[0]
    
    fn = f'{outdir}/Abacus-hf-dr2-v2-altmtl_{tracer_str}_{region}_{freedom}freedom_{de_model}_pk_dk{_fmt_float(dkP)}_kmax{kmaxP_str}'
    if reparam_option is not None:
        fn += f'_{reparam_option}reparam'

    if free_Mnu:
        fn += '_freeMnu'
    if bispec:
        fn += '_bk'
        if isinstance(kmaxB, list):
            kmaxB_str = '-'.join([_fmt_float(k) for k in kmaxB])
        else:
            kmaxB_str = _fmt_float(kmaxB)
        if dkB is not None:
            fn += f'_dk{_fmt_float(dkB)}'
        if kmaxB is not None:
            fn += f'_kmax{kmaxB_str}'
        if avirB_free:
            fn += '_avirBfree'
    if counterterm_basis != 'DESIct':
        fn += f'_ct{counterterm_basis}'

    fn += f'_{zeff_choice}'
    if not use_Mpc:
        fn += '_Mpch'

    if extra is not None:
        fn += f'_{extra}'
    return fn

def _tracer_key(tracer):
    return 'ELG_LOP' if 'ELG' in tracer else tracer

def get_prior_refs(tracer_list, zrange_list):
    if not isinstance(tracer_list, list):
        tracer_list = [tracer_list]
    if not isinstance(zrange_list, (list, np.ndarray)):
        zrange_list = [zrange_list]
    sigmaR_ref_dict = {
        'BGS': {(0.1, 0.4): 0.694},
        'LRG': {(0.4, 0.6): 0.62469875, (0.6, 0.8): 0.56705006, (0.8, 1.1): 0.51262079},
        'ELG_LOP': {(0.8, 1.1): 0.5050584, (1.1, 1.6): 0.43297989},
        'QSO': {(0.8, 2.1): 0.40714599},
    }
    b1_ref_dict = {
        'BGS': {(0.1, 0.4): 1.541},
        'LRG': {(0.4, 0.6): 1.883, (0.6, 0.8): 2.031, (0.8, 1.1): 2.216},
        'ELG_LOP': {(0.8, 1.1): 1.187, (1.1, 1.6): 1.429},
        'QSO': {(0.8, 2.1): 2.266},
    }
    fsat_dict = {'BGS': 0.13, 'LRG': 0.13, 'ELG_LOP': 0.06, 'QSO': 0.2}

    b1_ref = []
    sigmaR_ref = []
    fsat = []
    sigma1_eff = []
    for tracer, zr in zip(tracer_list, zrange_list):
        key = _tracer_key(tracer)
        b1_ref.append(b1_ref_dict[key][tuple(zr)])
        sigmaR_ref.append(sigmaR_ref_dict[key][tuple(zr)])
        fsat.append(fsat_dict[key])
        if key == 'BGS':
            sigma1_eff.append(150/70 * 10**(1/3) * np.sqrt(1 + 0.2))
        elif key == 'LRG':
            sigma1_eff.append(150/70 * 10**(1/3) * np.sqrt(1 + 0.8))
        elif key == 'ELG_LOP':
            sigma1_eff.append(150/70 * 2.1 ** (1/2))
        else:
            sigma1_eff.append(150/70 * 10**(0.7/3) * 2.4 ** (1/2))

    return np.array(b1_ref, dtype=float), np.array(sigmaR_ref, dtype=float), np.array(sigma1_eff, dtype=float), np.array(fsat, dtype=float)


def build_pars(args, b1_ref, sigmaR_ref, sigma1_eff, fsat, z_array):
    """Construct a fresh Params object (with its own COMET emulator instance)."""
    pars = get_pars(bispec=args.bispec, de_model=args.de_model, reparam_option=args.reparam,
                     freedom=args.freedom, free_Mnu=args.free_Mnu,
                     b1_ref=b1_ref, sigmaR_ref=sigmaR_ref, sigma1_eff=sigma1_eff, fsat=fsat,
                     z_array=z_array, counterterm_basis=args.counterterm_basis,
                     avirB_free=args.avirB_free, use_Mpc=not args.mpc_h)
    if args.bispec:
        pars.emu.bispec_kwargs['sugiyama']['quad_deg'] = (7, 16, 5)
        pars.emu.bispec_kwargs['sugiyama']['mu12_transform'] = 'k3'
        pars.emu.BispNum.backend = 'jax'
    pars.emu.use_interp_kwin = True
    return pars


def get_cosmo_params_to_plot(args):
    """Names of the sampled/exported cosmological parameters for this fit,
    in the order they should appear in the triangle plot."""
    params = ['wb', 'wc', 'h', 'ns', 'log10As']
    if args.de_model in ('w0', 'w0wa'):
        params.append('w0')
    if args.de_model == 'w0wa':
        params.append('wa')
    if args.free_Mnu:
        params.append('Mnu')
    return params


def _parse_tuple_ell(ell_str):
    """Convert string like '000' or '0,0,0' to tuple of ints like (0, 0, 0)"""
    if ',' in ell_str:
        return tuple(int(x) for x in ell_str.split(','))
    else:
        return tuple(int(x) for x in ell_str)


def _load_scale_map(json_str, arg_name, valid_labels):
    """Parse a --*_map JSON string of the form '{"LRG2": ..., "QSO": ...}'
    into a dict, validating that every key is a tracer label that was
    actually requested via --tracer_label."""
    if json_str is None:
        return {}
    try:
        mapping = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse --{arg_name} as JSON: {e}")
    if not isinstance(mapping, dict):
        raise ValueError(f"--{arg_name} must be a JSON object mapping tracer_label to an override value, "
                         f"e.g. '{{\"QSO\": 0.25}}'")
    unknown = set(mapping) - set(valid_labels)
    if unknown:
        raise ValueError(f"--{arg_name} references unknown tracer label(s) {sorted(unknown)}; "
                         f"must be a subset of --tracer_label {list(valid_labels)}")
    return mapping


def _map_ell_value(val, bispec=False):
    """Normalize an ell override value coming from a JSON map: a list of ints
    for pk (e.g. [0, 2]), or a list of 3-int lists for bk (e.g. [[0, 0, 0], [2, 0, 2]])."""
    if bispec:
        return [tuple(int(x) for x in ell) for ell in val]
    return [int(x) for x in val]


def _scale_maps_suffix(maps_dict):
    """Short, deterministic filename suffix summarizing any per-tracer scale-cut
    overrides, so runs with different --*_map settings don't collide on disk."""
    active = {name: m for name, m in maps_dict.items() if m}
    if not active:
        return ''
    canonical = json.dumps(active, sort_keys=True)
    h = hashlib.sha1(canonical.encode()).hexdigest()[:8]
    return f'_pertracer{h}'


def _w0wa_conditional_prior(params):
    # Module-level (not a closure inside __main__) so it can be pickled by
    # reference to the pool workers under the 'spawn' start method.
    return params['w0'] + params['wa'] < 0


def _load_shm_array(path):
    # Copy-on-write memory map: all workers share the same physical pages of
    # the file in /dev/shm until (if ever) one writes to its view.
    return np.load(path, mmap_mode='c')


def _dump_likelihood_shared(obj, directory, min_bytes=1_000_000):
    """Pickle obj to directory/likelihood.pkl, but store every large plain
    numpy array as its own .npy file in `directory` (meant to be /dev/shm) and
    pickle just a reference to it. Workers then memory-map those files instead
    of each holding a private copy (the likelihood's data arrays are ~all of
    its ~675 MB, which x256 workers was a big part of the OOM)."""
    import pickle

    class _ShmPickler(pickle.Pickler):
        n_shared = 0
        shared_bytes = 0

        def reducer_override(self, o):
            if type(o) is np.ndarray and o.nbytes >= min_bytes and not o.dtype.hasobject:
                path = os.path.join(directory, f'arr{self.n_shared:04d}.npy')
                np.save(path, o, allow_pickle=False)
                self.n_shared += 1
                self.shared_bytes += o.nbytes
                return _load_shm_array, (path,)
            return NotImplemented

    path = os.path.join(directory, 'likelihood.pkl')
    with open(path, 'wb') as f:
        p = _ShmPickler(f, protocol=pickle.HIGHEST_PROTOCOL)
        p.dump(obj)
    return path, p.n_shared, p.shared_bytes


def _pickle_size_breakdown(obj, top=8):
    """Tally the array payload inside obj's pickle by kind (numpy vs jax vs
    other) and list the largest arrays. Tells us how much of the pickle could
    be shared between workers via memory-mapped files."""
    import io
    import pickle
    from collections import Counter
    sizes, counts, big = Counter(), Counter(), []

    class _Tally(pickle.Pickler):
        def reducer_override(self, o):
            kind = None
            if isinstance(o, np.ndarray):
                kind = 'numpy.ndarray'
            elif type(o).__module__.startswith(('jax', 'jaxlib')) and hasattr(o, 'nbytes'):
                kind = 'jax array'
            if kind:
                sizes[kind] += o.nbytes
                counts[kind] += 1
                big.append((o.nbytes, kind, tuple(o.shape), str(o.dtype)))
            return NotImplemented

    buf = io.BytesIO()
    _Tally(buf, protocol=pickle.HIGHEST_PROTOCOL).dump(obj)
    total = buf.tell()
    print(f"pickle total: {total / 1e6:.0f} MB")
    for kind in sizes:
        print(f"  {kind}: {sizes[kind] / 1e6:.0f} MB in {counts[kind]} arrays")
    print(f"  everything else: {(total - sum(sizes.values())) / 1e6:.0f} MB")
    print(f"largest {top} arrays:")
    for nbytes, kind, shape, dtype in sorted(big, reverse=True)[:top]:
        print(f"  {nbytes / 1e6:8.1f} MB  {kind}  shape={shape}  dtype={dtype}")


def _find_unpicklable(obj, path='likelihood', _seen=None, _out=None):
    """Recursively walk obj's attributes/containers and return a list of
    (path, type, error) for the *leaf* objects that fail to pickle. Used by
    --check_pickle, since 'spawn' needs everything sent to workers picklable."""
    import pickle
    _seen = set() if _seen is None else _seen
    _out = [] if _out is None else _out
    if id(obj) in _seen:
        return _out
    _seen.add(id(obj))
    try:
        pickle.dumps(obj)
        return _out
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
    if isinstance(obj, dict):
        children = [(f"{path}[{k!r}]", v) for k, v in obj.items()]
    elif isinstance(obj, (list, tuple, set)):
        children = [(f"{path}[{i}]", v) for i, v in enumerate(obj)]
    elif hasattr(obj, '__dict__'):
        children = [(f"{path}.{k}", v) for k, v in vars(obj).items()]
    else:
        children = []
    n_before = len(_out)
    for p, v in children:
        _find_unpicklable(v, p, _seen, _out)
    if len(_out) == n_before:  # no failing child: obj itself is the culprit
        _out.append((path, type(obj).__name__, err))
    return _out


def _pin_to_n_cores(n_cores, worker_index=0):
    """Restrict the calling process to `n_cores` distinct CPUs, taking the
    worker_index'th disjoint slice of the process's currently-allowed set
    (from os.sched_getaffinity, which is what SLURM's --cpu-bind sets up).

    JAX's CPU thread pool sizes itself off this affinity mask, not off any
    XLA_FLAGS setting (verified empirically: cpu_time/wall_time ratio for a
    single JAX call tracked the taskset core count almost exactly, and was
    unaffected by --xla_cpu_multi_thread_eigen or --xla_cpu_use_xnnpack).
    So pinning each nautilus pool worker to its own slice of cores is the
    actual way to cap how many cores a single JAX call can use, and to stop
    N pool workers from all contending for every core at once.
    """
    available = sorted(os.sched_getaffinity(0))
    start = worker_index * n_cores
    my_cores = available[start:start + n_cores]
    if len(my_cores) < n_cores:
        raise ValueError(f"Not enough CPUs in affinity mask ({len(available)}) to give "
                          f"worker_index={worker_index} its {n_cores} core(s).")
    os.sched_setaffinity(0, my_cores)
    return my_cores


class _WorkerLikelihood:
    """Picklable stand-in for NautilusSampler.likelihood_wrapper (same logic),
    without the bound-method link to the sampler and its pool. Sent to each
    worker once via the pool initializer instead of with every map() chunk."""

    def __init__(self, likelihood):
        self.likelihood = likelihood
        self.params = likelihood.params
        self.require_blobs = len(self.params.exported_derived_names) > 0
        self._n_calls = 0

    def _log_memory(self):
        # Report resident memory of a few workers at a few call counts, to see
        # the real per-worker footprint (static data + JAX/numba runtime state).
        import multiprocessing as mp
        ident = mp.current_process()._identity
        if not ident or ident[0] > 4:
            return
        # RssAnon = private memory; RssFile/RssShmem = mapped shared pages (the
        # memory-mapped likelihood arrays), which are counted in every worker's
        # VmRSS but only once in the job's real memory usage.
        keys = ('VmRSS', 'VmHWM', 'RssAnon', 'RssFile', 'RssShmem')
        with open('/proc/self/status') as f:
            mem = {k: v.strip() for k, v in (line.split(':', 1) for line in f if line.startswith(keys))}
        print(f"[pool worker {ident[0] - 1}] call {self._n_calls}: " + ", ".join(f"{k}={mem.get(k)}" for k in keys), flush=True)

    def __call__(self, param_dict):
        self._n_calls += 1
        if self._n_calls in (1, 50, 1000):
            self._log_memory()
        full_dict = self.params.get_full_dict(param_dict)
        loglike = self.likelihood.get_loglike(full_dict)
        if self.require_blobs:
            return loglike, [full_dict[name] for name in self.params.exported_derived_names]
        return loglike


def _pool_worker_init(threads_per_worker, likelihood_path):
    """multiprocessing.Pool initializer: pin this worker to its own slice of
    cores based on its 1-indexed pool slot (see _pin_to_n_cores), and install
    the likelihood in nautilus's worker global (what nautilus itself does when
    given an int pool), so map() only ships the small parameter dicts.

    The likelihood is loaded from a pickle file (on node-local /dev/shm)
    rather than passed via initargs: with 'spawn', Pool starts workers one at
    a time and pipes initargs to each child before starting the next, so a
    649 MB payload made startup serial (~19 s/worker, 40 min for 128 workers).
    Passing just a path lets all workers boot and load in parallel."""
    import multiprocessing as mp
    import pickle
    from nautilus.pool import initialize_worker
    worker_index = mp.current_process()._identity[0] - 1
    cores = _pin_to_n_cores(threads_per_worker, worker_index)
    with open(likelihood_path, 'rb') as f:
        initialize_worker(pickle.load(f))
    print(f"[pool worker {worker_index}] pinned to cores {cores}, likelihood loaded", flush=True)


if __name__ == "__main__":
    import argparse
    import time
    import multiprocessing as mp
    # Force 'spawn' instead of the Linux default 'fork' for the nautilus
    # worker pool below. Forking a process that has already imported jax
    # duplicates its (multi-threaded) XLA runtime into each child, which
    # JAX explicitly warns is unsafe and defeats the single-thread-per-call
    # setup above. 'spawn' starts each worker fresh so it re-applies the
    # XLA_FLAGS/OMP_NUM_THREADS env vars set at the top of this file.
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracer_label', type=str, default='LRG', nargs='*', help="Tracer label(s) to fit. Should be one of LRG1, LRG2, LRG3, ELG1, ELG2, QSO. If multiple are provided, they will be fit simultaneously with shared cosmological parameters but independent nuisance parameters.")
    parser.add_argument('--region', type=str, default='GCcomb')
    parser.add_argument('--mocktype', type=str, default=None, help="Override the data mock type. "
                        "Default: per-tracer (abacus-hf-dr2-v2-altmtl, or abacus-2ndgen-dr2-altmtl for BGS). "
                        "Set it to fit an alternative data set, e.g. a blinded one.")
    parser.add_argument('--mocktype_cov', type=str, default=None, help="Override the mock type of the "
                        "covariance. Default: per-tracer (holi-v3-altmtl, or holi-bgs-altmtl for BGS).")
    parser.add_argument('--data_dir', type=str, default=None, help="Directory holding the cached data, "
                        "covariance and window files. Default: the location read_data.py points at.")
    parser.add_argument('--ellP', type=int, nargs='*', default=[0, 2])
    parser.add_argument('--kminP', type=float, default=0.02, nargs='*')
    parser.add_argument('--kmaxP', type=float, default=0.3, nargs='*')
    parser.add_argument('--ellP_map', type=str, default=None, help="JSON object overriding --ellP for specific "
                        "tracer labels, e.g. '{\"QSO\": [0, 2, 4]}'. Tracers not listed use --ellP.")
    parser.add_argument('--kminP_map', type=str, default=None, help="JSON object overriding --kminP for specific "
                        "tracer labels (scalar or per-ell list), e.g. '{\"LRG2\": 0.03, \"QSO\": [0.03, 0.04]}'. "
                        "Tracers not listed use --kminP.")
    parser.add_argument('--kmaxP_map', type=str, default=None, help="JSON object overriding --kmaxP for specific "
                        "tracer labels (scalar or per-ell list), e.g. '{\"QSO\": 0.25}'. Tracers not listed use --kmaxP.")
    parser.add_argument('--ellwinP', type=int, nargs='*', default=[0, 2, 4])
    parser.add_argument('--kwinminP', type=float, default=0.0015, nargs='*')
    parser.add_argument('--kwinmaxP', type=float, default=0.5, nargs='*')
    parser.add_argument('--dkP', type=float, default=0.005)
    parser.add_argument('--bispec', action='store_true')
    parser.add_argument('--avirB_free', action='store_true', help="Sample avirB as a free parameter instead of tying it to avir (the default when --bispec is set).")
    parser.add_argument('--ellB', type=_parse_tuple_ell, nargs='*', default=[(0, 0, 0), (2, 0, 2)])
    parser.add_argument('--kminB', type=float, default=0.02, nargs='*')
    parser.add_argument('--kmaxB', type=float, default=0.2, nargs='*')
    parser.add_argument('--ellB_map', type=str, default=None, help="JSON object overriding --ellB for specific "
                        "tracer labels, e.g. '{\"QSO\": [[0, 0, 0]]}'. Tracers not listed use --ellB.")
    parser.add_argument('--kminB_map', type=str, default=None, help="JSON object overriding --kminB for specific "
                        "tracer labels (scalar or per-ell list), e.g. '{\"QSO\": 0.03}'. Tracers not listed use --kminB.")
    parser.add_argument('--kmaxB_map', type=str, default=None, help="JSON object overriding --kmaxB for specific "
                        "tracer labels (scalar or per-ell list), e.g. '{\"QSO\": 0.15}'. Tracers not listed use --kmaxB.")
    parser.add_argument('--ellwinB', type=_parse_tuple_ell, nargs='*', default=[(0, 0, 0), (0, 2, 2)])
    parser.add_argument('--kwinminB', type=float, default=0.0015, nargs='*')
    parser.add_argument('--kwinmaxB', type=float, default=0.3, nargs='*')
    parser.add_argument('--dkB', type=float, default=0.005)
    parser.add_argument('--reparam', type=str, default='full', choices=['full', 'hybrid', 'none'])
    parser.add_argument('--de_model', type=str, default='lambda', choices=['lambda', 'w0wa', 'w0'])
    parser.add_argument('--counterterm_basis', type=str, default='DESIct', choices=['DESIct', 'Comet'])
    parser.add_argument('--freedom', type=str, default='interm', choices=['min', 'max', 'interm'])
    parser.add_argument('--free_Mnu', action='store_true')
    parser.add_argument('--mpc_h', action='store_true', help="Run the fit in Mpc/h units instead of Mpc. Switches the reparametrization to use sigma8 instead of sigma12.")
    parser.add_argument('--zeff_choice', type=str, default='zsnap', choices=['zsnap', 'zgeom'],
                        help="Which effective redshift to evaluate the theory at: 'zsnap' (default) uses the "
                        "fixed AbacusSummit snapshot redshift from zsnap_dict; 'geometry' uses the effective "
                        "redshift computed from the survey window/n(z) geometry instead. Reflected in the output filename.")
    parser.add_argument('--outdir', type=str, default=str(env.CHAINS_DIR))
    parser.add_argument('--n_live', type=int, default=3000)
    parser.add_argument('--extra', type=str, default=None, help="Extra string to add to output filename for uniqueness (e.g. to distinguish different sampler settings).")
    parser.add_argument('--minimize', action='store_true', help="Run an iMinuit MIGRAD minimization instead of Nautilus nested sampling.")
    parser.add_argument('--hesse', action='store_true', help="Run HESSE after MIGRAD to get the covariance matrix. Only used with --minimize.")
    parser.add_argument('--seed_init', type=int, default=None, help="Random seed for drawing the Minuit starting point from the priors. Only used with --minimize.")
    parser.add_argument('--plot_contours', action='store_true', help="After the Nautilus chain finishes, plot the triangle/contour plot for the cosmological parameters and save it to --plot_dir. Not used with --minimize.")
    parser.add_argument('--plot_dir', type=str, default=str(env.PLOTS_DIR_CUTSKY_ABACUSHF), help="Directory to store the contour plot in, when --plot_contours is set.")
    parser.add_argument('--check_pickle', action='store_true', help="Build the likelihood, report every attribute "
                        "that cannot be pickled (required by the 'spawn' worker pool), and exit.")
    parser.add_argument('--time_likelihood', type=int, default=None, help="Instead of minimizing/sampling, time this many "
                        "calls to likelihood.get_loglike() at the YAML-default fiducial value of each free parameter, "
                        "then exit. Useful for benchmarking single-call cost under different core-pinning settings "
                        "(see COMET_THREADS_PER_WORKER / _pin_to_n_cores at the top of this file).")

    args = parser.parse_args()

    print(f"Fitting tracer(s) {args.tracer_label} in region {args.region}")
    if args.mocktype or args.mocktype_cov or args.data_dir:
        print(f"Data overrides: mocktype={args.mocktype}, mocktype_cov={args.mocktype_cov}, data_dir={args.data_dir}")
    print(f"Power spectrum settings: ellP={args.ellP}, kminP={args.kminP}, kmaxP={args.kmaxP}, ellwinP={args.ellwinP}, kwinminP={args.kwinminP}, kwinmaxP={args.kwinmaxP}, dkP={args.dkP}")
    if args.bispec:
        print(f"Bispectrum settings: ellB={args.ellB}, kminB={args.kminB}, kmaxB={args.kmaxB}, ellwinB={args.ellwinB}, kwinminB={args.kwinminB}, kwinmaxB={args.kwinmaxB}, dkB={args.dkB}, avirB_free={args.avirB_free}")
    print(f"DE model: {args.de_model}, reparametrization: {args.reparam}, freedom: {args.freedom}, free_Mnu: {args.free_Mnu}, counterterm_basis: {args.counterterm_basis}, units: {'Mpc/h' if args.mpc_h else 'Mpc'}, zeff_choice: {args.zeff_choice}")


    tracer_list = []
    zrange_list = []
    for label in args.tracer_label:
        if label not in tracer_label_dict:
            raise ValueError(f"Invalid tracer label {label}. Should be one of {list(tracer_label_dict.keys())}")
        info = tracer_label_dict[label]
        tracer_list.append(info['tracer'])
        zrange_list.append(info['zrange'])

    # Per-tracer overrides of ell/kmin/kmax for pk and bk (for a joint fit
    # where different tracers need different scale cuts and/or multipoles).
    # Tracers not present in a given map fall back to the shared --ellP/--kminP/... value.
    ellP_map = _load_scale_map(args.ellP_map, 'ellP_map', args.tracer_label)
    kminP_map = _load_scale_map(args.kminP_map, 'kminP_map', args.tracer_label)
    kmaxP_map = _load_scale_map(args.kmaxP_map, 'kmaxP_map', args.tracer_label)
    ellB_map = _load_scale_map(args.ellB_map, 'ellB_map', args.tracer_label)
    kminB_map = _load_scale_map(args.kminB_map, 'kminB_map', args.tracer_label)
    kmaxB_map = _load_scale_map(args.kmaxB_map, 'kmaxB_map', args.tracer_label)
    scale_maps_suffix = _scale_maps_suffix({'ellP_map': ellP_map, 'kminP_map': kminP_map, 'kmaxP_map': kmaxP_map,
                                            'ellB_map': ellB_map, 'kminB_map': kminB_map, 'kmaxB_map': kmaxB_map})
    if scale_maps_suffix:
        print(f"Per-tracer scale-cut overrides: ellP_map={ellP_map}, kminP_map={kminP_map}, kmaxP_map={kmaxP_map}, "
              f"ellB_map={ellB_map}, kminB_map={kminB_map}, kmaxB_map={kmaxB_map}")

    # Build one observable per z-bin for simultaneous fit
    observables = []
    mocktypes_used = []
    mocktypes_cov_used = []
    data_dir_kw = {} if args.data_dir is None else {'outdir': args.data_dir}
    for label, tracer_i, zr in zip(args.tracer_label, tracer_list, zrange_list):
        ellP_i = _map_ell_value(ellP_map[label], bispec=False) if label in ellP_map else args.ellP
        kminP_i = kminP_map.get(label, args.kminP)
        kmaxP_i = kmaxP_map.get(label, args.kmaxP)
        ellB_i = _map_ell_value(ellB_map[label], bispec=True) if label in ellB_map else args.ellB
        kminB_i = kminB_map.get(label, args.kminB)
        kmaxB_i = kmaxB_map.get(label, args.kmaxB)
        if tracer_i == 'BGS':
            mocktype = 'abacus-2ndgen-dr2-altmtl'
            mocktype_cov = 'holi-bgs-altmtl'
        else:
            mocktype = 'abacus-hf-dr2-v2-altmtl'
            mocktype_cov = 'holi-v3-altmtl'
        # --mocktype / --mocktype_cov override the per-tracer defaults above
        mocktype = args.mocktype or mocktype
        mocktype_cov = args.mocktype_cov or mocktype_cov
        mocktypes_used.append(mocktype)
        mocktypes_cov_used.append(mocktype_cov)
        if label in ellP_map or label in kminP_map or label in kmaxP_map or label in ellB_map or label in kminB_map or label in kmaxB_map:
            print(f"  [{label}] using ellP={ellP_i}, kminP={kminP_i}, kmaxP={kmaxP_i}"
                  + (f", ellB={ellB_i}, kminB={kminB_i}, kmaxB={kmaxB_i}" if args.bispec else ""))
        if not args.bispec:
            obs_i = get_obs_pk(tracer=tracer_i, zrange=zr, region=args.region, mocktype=mocktype, mocktype_cov=mocktype_cov,
                               ell=ellP_i, kmin=kminP_i, kmax=kmaxP_i, ellwin=args.ellwinP,
                               kwinmin=args.kwinminP, kwinmax=args.kwinmaxP, dk=args.dkP, use_Mpc=not args.mpc_h,
                               **data_dir_kw)
        else:
            obs_i = get_obs_pk_bk(tracer=tracer_i, zrange=zr, region=args.region, mocktype=mocktype, mocktype_cov=mocktype_cov,
                                        ellP=ellP_i, kminP=kminP_i, kmaxP=kmaxP_i, ellwinP=args.ellwinP,
                                        kwinminP=args.kwinminP, kwinmaxP=args.kwinmaxP, dkP=args.dkP,
                                        ellB=ellB_i, kminB=kminB_i, kmaxB=kmaxB_i, ellwinB=args.ellwinB,
                                        kwinminB=args.kwinminB, kwinmaxB=args.kwinmaxB, dkB=args.dkB, slice_winB_theory=2,
                                        use_Mpc=not args.mpc_h, **data_dir_kw)
        print(f"Effective redshift from geometry: {obs_i.cosmo_fid['z']:.3f}, from zrange: {zr}")
        if args.zeff_choice == 'zsnap':
            obs_i.cosmo_fid['z'] = zsnap_dict[tracer_i][zr]
            print(f'Switched effective redshift to zsnap: {obs_i.cosmo_fid["z"]:.3f}')
        else:
            print(f"Using effective redshift from geometry: {obs_i.cosmo_fid['z']:.3f}")
        observables.append(obs_i)


    b1_ref, sigmaR_ref, sigma1_eff, fsat = get_prior_refs(tracer_list, zrange_list)
    z_array = [obs.cosmo_fid['z'] for obs in observables]
    # sort z_array
    z_array = np.array(z_array)
    sort_idx = np.argsort(z_array)
    z_array = z_array[sort_idx]
    observables = [observables[i] for i in sort_idx]

    if args.counterterm_basis == 'DESIct':
        am_params = ['btd_r', 'a0_r', 'a2_r', 'NP0_r', 'NP20_r', 'NP22_r']
    elif args.counterterm_basis == 'Comet':
        am_params = ['btd_r', 'c0_r', 'c2_r', 'NP0_r', 'NP20_r', 'NP22_r']
    if args.bispec:
        am_params += ['NB0_r', 'MB0_r']

    pars = build_pars(args, b1_ref, sigmaR_ref, sigma1_eff, fsat, z_array)

    conditional_prior_fn = _w0wa_conditional_prior if args.de_model == 'w0wa' else None

    likelihood = Likelihood(observables, pars, am_params=am_params, conditional_prior=conditional_prior_fn)

    if args.check_pickle:
        import pickle
        bad = _find_unpicklable(likelihood)
        if not bad:
            print("likelihood is picklable.")
            _pickle_size_breakdown(likelihood)
        else:
            print(f"{len(bad)} unpicklable object(s) reachable from the likelihood:")
            for p, t, e in bad:
                print(f"  {p}  [{t}]  -> {e}")
        sys.exit(0)

    if args.time_likelihood:
        # Pin this process the same way a real pool worker would be pinned
        # below, so the benchmark reflects the actual per-worker core budget
        # instead of running unpinned across everything --cpus-per-task grants.
        threads_per_worker = int(os.environ.get('COMET_THREADS_PER_WORKER', '1'))
        my_cores = _pin_to_n_cores(threads_per_worker)
        print(f"Pinned this process to {len(my_cores)} core(s): {my_cores}")

        # Same point nautilus/Minuit would evaluate: the YAML-default 'value'
        # for each free parameter, filled out to the full dict get_loglike expects.
        fiducial = {name: pars.parameters[name].value for name in pars.sampled_param_names}
        full_dict = pars.get_full_dict(fiducial)

        loglike = likelihood.get_loglike(full_dict)  # warm-up: triggers JAX tracing/compilation, excluded from timing
        print(f"Warm-up call: loglike={loglike:.3f}")

        n_calls = args.time_likelihood
        times = np.empty(n_calls)
        cpu_times = np.empty(n_calls)
        for i in range(n_calls):
            t0 = time.perf_counter()
            c0 = time.process_time()
            likelihood.get_loglike(full_dict)
            times[i] = time.perf_counter() - t0
            cpu_times[i] = time.process_time() - c0
        print(f"get_loglike x{n_calls}: wall mean={times.mean()*1e3:.2f} ms  median={np.median(times)*1e3:.2f} ms  "
              f"std={times.std()*1e3:.2f} ms  min={times.min()*1e3:.2f} ms  max={times.max()*1e3:.2f} ms")
        # time.process_time() sums CPU time across all threads of this process.
        # ratio ~= 1 means the call ran on effectively one core no matter how
        # many OS threads exist; ratio ~= N means it actually used N cores'
        # worth of CPU time per call, regardless of what any XLA flag claims.
        ratio = cpu_times.sum() / times.sum()
        print(f"get_loglike x{n_calls}: cpu mean={cpu_times.mean()*1e3:.2f} ms  "
              f"cpu_time/wall_time ratio={ratio:.2f} (~1 = single-threaded, ~N = using N cores/call)")
        sys.exit(0)

    os.makedirs(args.outdir, exist_ok=True)
    extra = (args.extra or '') + scale_maps_suffix or None
    fn = get_fn(tracer_label=args.tracer_label, region=args.region,  freedom=args.freedom, dkP=args.dkP, kmaxP=args.kmaxP,
                bispec=args.bispec, dkB=args.dkB, kmaxB=args.kmaxB,
                de_model=args.de_model, reparam_option=args.reparam, free_Mnu=args.free_Mnu, outdir=args.outdir, extra=extra,
                counterterm_basis=args.counterterm_basis, avirB_free=args.avirB_free, use_Mpc=not args.mpc_h,
                zeff_choice=args.zeff_choice)
    if args.minimize:
        # Stage 1: minimize with analytical marginalisation (AM) of the
        # linear nuisance parameters, and save the resulting MAP.
        minimizer = MinuitMinimizer(likelihood, seed_init=args.seed_init, verbose=True)

        t0 = time.time()
        print("Starting minimization with analytical marginalisation...")

        minimizer.run(hesse=args.hesse, verbose=True)
        t1 = time.time()
        print(f"AM minimization finished in {(t1-t0)/60:.2f} minutes.")

        best_fit_am, uncertainties_am = minimizer.get_map(return_am=True)

        fn_minuit_am = fn + '_minuit_am'
        run_metadata_am = {**vars(args), 'mocktypes_used': mocktypes_used, 'mocktypes_cov_used': mocktypes_cov_used,
                           'elapsed_minutes': (t1 - t0) / 60, 'stage': 'analytical_marginalisation'}
        minimizer.save(fn_minuit_am, best_fit=best_fit_am, uncertainties=uncertainties_am, metadata=run_metadata_am)
        print(f"AM best-fit result saved to {fn_minuit_am}")

        # Stage 2: use the AM MAP (including the conditional MAP of the
        # marginalised parameters) as the starting point for a minimization
        # of the full likelihood, where those parameters are sampled directly
        # instead of analytically marginalised.
        pars_full = build_pars(args, b1_ref, sigmaR_ref, sigma1_eff, fsat, z_array)
        likelihood_full = Likelihood(observables, pars_full, am_params=None, conditional_prior=conditional_prior_fn)

        minimizer_full = MinuitMinimizer(likelihood_full, seed_init=args.seed_init, verbose=True)
        minimizer_full.set_starting_point(best_fit_am, errors_dict=uncertainties_am, reset_errors=True)

        t0 = time.time()
        print("Starting full-likelihood minimization from the AM MAP...")

        minimizer_full.run(hesse=args.hesse, verbose=True)
        t1 = time.time()
        print(f"Full-likelihood minimization finished in {(t1-t0)/60:.2f} minutes.")

        fn_minuit = fn + '_minuit_full'
        run_metadata = {**vars(args), 'mocktypes_used': mocktypes_used, 'mocktypes_cov_used': mocktypes_cov_used,
                         'elapsed_minutes': (t1 - t0) / 60, 'stage': 'full_likelihood',
                         'am_map_file': fn_minuit_am}
        minimizer_full.save(fn_minuit, metadata=run_metadata)
        print(f"Full-likelihood best-fit result saved to {fn_minuit}")
    else:
        fn_snap = fn + '_snap.hdf5'

        # Actual CPU budget for this job, as granted by SLURM (--cpu-bind sets
        # this affinity mask); this is what _pin_to_n_cores slices up below.
        n_threads = len(os.sched_getaffinity(0))
        # Cores per pool worker. Default 1: one process per core, each JAX
        # call single-threaded -- the setup recommended after benchmarking
        # (see COMET_THREADS_PER_WORKER in the --time_likelihood benchmark).
        # Set e.g. COMET_THREADS_PER_WORKER=4 to instead run fewer, wider
        # workers (pool=n_threads//4, each pinned to 4 cores).
        threads_per_worker = int(os.environ.get('COMET_THREADS_PER_WORKER', '1'))
        n_workers = max(1, n_threads // threads_per_worker)
        print(f"{n_threads} CPUs available. Using {n_workers} pool worker(s) x "
              f"{threads_per_worker} core(s)/worker ({n_workers * threads_per_worker} of {n_threads} cores used).")

        import shutil
        import tempfile
        # Dump the likelihood once to node-local RAM (/dev/shm); its big numpy
        # arrays are stored as separate files that workers memory-map (shared
        # physical pages), the rest is a small pickle every worker loads.
        t_dump = time.time()
        shm_dir = tempfile.mkdtemp(prefix='comet_likelihood_', dir='/dev/shm' if os.path.isdir('/dev/shm') else None)
        likelihood_path, n_shared, shared_bytes = _dump_likelihood_shared(_WorkerLikelihood(likelihood), shm_dir)
        print(f"Dumped likelihood to {shm_dir}: {n_shared} arrays ({shared_bytes/1e6:.0f} MB) shared via mmap, "
              f"{os.path.getsize(likelihood_path)/1e6:.1f} MB pickled per worker, in {time.time() - t_dump:.1f} s", flush=True)

        try:
            t_pool = time.time()
            pool = mp.get_context('spawn').Pool(n_workers, initializer=_pool_worker_init,
                                                initargs=(threads_per_worker, likelihood_path))
            print(f"Pool of {n_workers} workers launched in {time.time() - t_pool:.1f} s "
                  f"(workers finish loading in the background)", flush=True)
            sampler = NautilusSampler(likelihood, n_live=args.n_live, filepath=fn_snap, pool=pool)
            # nautilus only swaps in likelihood_worker for int pools; do it by hand
            # so map() calls the copy installed in each worker instead of pickling
            # the whole likelihood with every chunk.
            from nautilus.pool import likelihood_worker
            sampler.sampler.likelihood = likelihood_worker

            # run sampler
            t0 = time.time()
            print("Starting sampler...")

            sampler.sample(verbose=True, discard_exploration=True)
        finally:
            shutil.rmtree(shm_dir, ignore_errors=True)
        t1 = time.time()
        print(f"Sampler finished in {(t1-t0)/60:.2f} minutes.")
        run_metadata = {**vars(args), 'mocktypes_used': mocktypes_used, 'mocktypes_cov_used': mocktypes_cov_used,
                         'n_threads': n_threads, 'pool_size': n_workers, 'threads_per_worker': threads_per_worker,
                         'discard_exploration': True, 'elapsed_minutes': (t1 - t0) / 60}
        sampler.save(fn, metadata=run_metadata)
        print(f"Chain saved to {fn}")

        if args.plot_contours:
            print("Plotting contours for the cosmological parameters...")
            try:
                samples = pu.get_samples(fn + '.h5')
                cosmo_params = get_cosmo_params_to_plot(args)
                try:
                    g = pu.plot_triangle(samples, params_to_plot=cosmo_params, filled=False)
                except (RuntimeError, FileNotFoundError):
                    # plot_utils sets rcParams['text.usetex'] = True at import
                    # time, which needs a working LaTeX install; fall back to
                    # matplotlib's own mathtext if that's not available here.
                    plt.rc('text', usetex=False)
                    g = pu.plot_triangle(samples, params_to_plot=cosmo_params, filled=False)
                os.makedirs(args.plot_dir, exist_ok=True)
                plot_fn = os.path.join(args.plot_dir, f"{Path(fn).name}_contours.png")
                g.fig.savefig(plot_fn, dpi=150, bbox_inches='tight')
                plt.close(g.fig)
                print(f"Contour plot saved to {plot_fn}")
            except Exception as e:
                # The chain itself is already saved at this point; don't let a
                # plotting failure look like the whole run failed.
                print(f"Warning: failed to plot contours ({e}). Chain is still saved at {fn}.")

        # clean up
        os.remove(fn_snap)


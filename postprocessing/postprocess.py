import numpy as np
from functools import partial
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import env  # noqa: F401
from abacus_cosmologies import get_abacus_cosmology


class _SerialComm:
    """Single-process stand-in for an mpi4py communicator (the subset used
    here), for environments without mpi4py (e.g. the raven 'fs' env)."""
    def Get_rank(self): return 0
    def Get_size(self): return 1
    def gather(self, obj, root=0): return [obj]
    def allgather(self, obj): return [obj]
    def Gatherv(self, sendbuf, recvbuf, root=0): recvbuf[0][:] = sendbuf
    def Barrier(self): pass
    def Abort(self, errorcode=0): sys.exit(errorcode)


try:
    from mpi4py import MPI
    COMM_WORLD = MPI.COMM_WORLD
except ImportError:
    COMM_WORLD = _SerialComm()

COSMOPRIMO_MAPPING = {
    'wb': 'omega_b',
    'wc': 'omega_cdm',
    'h': 'h',
    'n_s': 'n_s',
    'ns': 'n_s',
    'log10As': 'logA',
    'w0': 'w0_fld',
    'wa': 'wa_fld',
    'Mnu': 'm_ncdm',
}

cosmo_fid = get_abacus_cosmology('c000')
# Fiducial values used for cosmological params not sampled in the chain.
FIDUCIAL = {
    'omega_b': cosmo_fid['omega_b'],
    'omega_cdm': cosmo_fid['omega_cdm'],
    'h': cosmo_fid['h'],
    'n_s': cosmo_fid['n_s'],
    'logA': cosmo_fid['logA'],
    'm_ncdm': cosmo_fid['m_ncdm'],
}

def _map_logAs_to_As(logAs):
    #logAs = ln(10^10 * As) => As = exp(logAs) / 10^10
    As = np.exp(logAs) / 1e10
    As *= 1e9 # Convert to 1e-9 units
    return As

FIDUCIAL_COMET = {
    'wb' : FIDUCIAL['omega_b'],
    'wc' : FIDUCIAL['omega_cdm'],
    'h' : FIDUCIAL['h'],
    'ns' : FIDUCIAL['n_s'],
    'As' : _map_logAs_to_As(FIDUCIAL['logA']),
    'Mnu' : FIDUCIAL['m_ncdm'],
    'z': 0.0} 



def map_to_cosmoprimo(names):
    return [COSMOPRIMO_MAPPING[name] for name in names]


def _compute_sigma8_cosmoprimo(params, engine, which='cb', **extra):
    # cosmoprimo (with CLASS/CAMB) is only needed for engine='class'/'camb';
    # engine='comet' works without it.
    from cosmoprimo import Cosmology
    cosmo = Cosmology(**FIDUCIAL | params, engine=engine, **extra)
    return _sigma8_from_cosmo(cosmo, engine, which)


# Per-sample fallbacks, tried in order when the previous one raises a cosmoprimo
# computation/input error. CLASS's perturbation evolver fails ("Step size too
# small") for some strongly phantom w0wa samples (w0 + wa <~ -3.5, high h); a
# looser integration tolerance or CAMB fixes it, agreeing to ~0.3%.
SIGMA8_FALLBACKS = {
    'class': [('class', {}), ('class', {'tol_perturbations_integration': 1e-4}), ('camb', {})],
    'camb': [('camb', {}), ('class', {}), ('class', {'tol_perturbations_integration': 1e-4})],
}


def _fallback_label(engine, extra):
    return engine + ''.join(f' {k}={v}' for k, v in extra.items())


def _compute_sigma8_with_fallbacks(params, engine, which='cb'):
    """Return (sigma8, label of the method that worked); (nan, 'failed') if all fail."""
    from cosmoprimo.cosmology import CosmologyComputationError, CosmologyInputError
    for eng, extra in SIGMA8_FALLBACKS[engine]:
        try:
            return _compute_sigma8_cosmoprimo(params, eng, which, **extra), _fallback_label(eng, extra)
        except (CosmologyComputationError, CosmologyInputError):
            continue
    return np.nan, 'all failed (NaN)'


def _sigma8_from_cosmo(cosmo, engine, which):
    if which == 'm':
        return cosmo.sigma8_m
    if engine == 'camb':
        # cosmoprimo's CAMB engine has no sigma8_cb property; compute it from the delta_cb power spectrum.
        return cosmo.get_fourier().sigma8_z(0., of='delta_cb')
    return cosmo.sigma8_cb # we want the sigma8 of the CDM+baryons (no neutrinos) ??

_emu = None
def _get_emu():
    # Built lazily so that engine='class'/'camb' never loads the emulator.
    global _emu
    if _emu is None:
        from comet import comet
        _emu = comet(model='VDG_infty', use_Mpc=False)
    return _emu

def _compute_sigma8_comet(params):
    de_model = 'lambda' if 'wa' not in params else 'w0wa'
    if 'log10As' in params:
        params['As'] = _map_logAs_to_As(params.pop('log10As'))
    scale = 8.0 # Mpc/h
    # scale_Mpc = scale / params['h'] # convert to Mpc
    return _get_emu().sigmaR(scale, FIDUCIAL_COMET | params, de_model=de_model)


def _compute_sigma8_row(row, cosmo_names, cp_names, engine, which):
    """sigma8 for one sample `row` (columns `cosmo_names`, cosmoprimo names
    `cp_names`). Returns (sigma8, label of the method used or None for comet)."""
    if engine == 'comet':
        return _compute_sigma8_comet(dict(zip(cosmo_names, row))), None
    return _compute_sigma8_with_fallbacks(dict(zip(cp_names, row)), engine, which)


def _compute_sigma8_rows(points, cosmo_names, cp_names, engine, which, pool=None):
    """sigma8 for each row of `points`, over `pool` if given. Returns
    (sigma8 array, {method label: count})."""
    row_fn = partial(_compute_sigma8_row, cosmo_names=cosmo_names, cp_names=cp_names, engine=engine, which=which)
    results = pool.map(row_fn, points) if pool is not None else [row_fn(row) for row in points]
    methods = {}
    for _, method in results:
        if method is not None:
            methods[method] = methods.get(method, 0) + 1
    return np.array([s for s, _ in results], dtype=float), methods


def _report_sigma8_methods(all_methods, engine, n):
    if engine not in ['class', 'camb']:
        return
    methods = {}
    for m in all_methods:
        for k, v in m.items():
            methods[k] = methods.get(k, 0) + v
    default = _fallback_label(*SIGMA8_FALLBACKS[engine][0])
    if set(methods) - {default}:
        print(f"sigma8 methods used over {n} samples -- " + ', '.join(f"{k}: {v}" for k, v in methods.items()))


def compute_sigma8(points, names, comm=None, engine='class', which='cb', pool=None):
    """Compute sigma8 for each row of `points` (columns named by `names`).

    By default this is MPI-collective: the rows are split across the ranks of
    ``comm``, and the full sigma8 array is returned on rank 0 (``None`` on the
    other ranks).

    If ``pool`` is given (e.g. a ``multiprocessing.Pool``; anything with a
    ``map``), the rows are instead split across its workers from this single process, ``comm`` is ignored, and the full
    array is always returned. Use this from a non-MPI job (e.g. right after a
    nautilus run, see src/fit_cutsky_abacushf.py --augment_chain).
    """
    _engine_options = ['class', 'camb', 'comet']
    if engine not in _engine_options:
        raise ValueError(f"Invalid engine '{engine}'. Valid options are: {_engine_options}")

    names = list(names)
    cosmo_names = [n for n in names if n in COSMOPRIMO_MAPPING]
    cp_names = map_to_cosmoprimo(cosmo_names)
    idx = [names.index(n) for n in cosmo_names]
    points = np.asarray(points)[:, idx]
    n = points.shape[0]

    if pool is not None:
        sigma8, methods = _compute_sigma8_rows(points, cosmo_names, cp_names, engine, which, pool=pool)
        _report_sigma8_methods([methods], engine, n)
        return sigma8

    if comm is None:
        comm = COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_idx = np.array_split(np.arange(n), size)[rank]
    try:
        local_sigma8, local_methods = _compute_sigma8_rows(points[local_idx], cosmo_names, cp_names, engine, which)
    except BaseException:
        # Any other error on one rank would leave the others blocked in the
        # collectives below; abort the whole job instead of hanging.
        if size > 1:
            import traceback
            traceback.print_exc()
            sys.stderr.flush()
            comm.Abort(1)
        raise

    all_methods = comm.gather(local_methods, root=0)
    if rank == 0:
        _report_sigma8_methods(all_methods, engine, n)

    counts = comm.allgather(local_sigma8.size)
    sigma8 = np.empty(n, dtype=local_sigma8.dtype) if rank == 0 else None
    comm.Gatherv(local_sigma8, (sigma8, counts) if rank == 0 else None, root=0)
    return sigma8


def add_sigma8(samples, comm=None, engine='class', which='cb'):
    if comm is None:
        comm = COMM_WORLD
    sigma8 = compute_sigma8(samples.samples, samples.getParamNames().list(),
                            comm=comm, engine=engine, which=which)
    if comm.Get_rank() == 0:
        samples.addDerived(sigma8, name='sigma8', label=r'\sigma_8')
        samples.updateBaseStatistics()
    return samples


def compute_Omega_m(points, names):
    """
    Computes Omega_m = (omega_cdm + omega_b + omega_ncdm) / h^2 for each row of
    `points` (columns named by `names`), using fiducial values for any parameter
    not in the chain.
    """
    names = list(names)
    points = np.asarray(points)
    n = points.shape[0]

    def col(*candidates, default):
        for name in candidates:
            if name in names:
                return points[:, names.index(name)]
        return default

    # Default to fiducial values if parameters are not in the chain
    wc = col('wc', 'omega_cdm', default=cosmo_fid['omega_cdm'])
    wb = col('wb', 'omega_b', default=cosmo_fid['omega_b'])
    h = col('h', default=cosmo_fid['h'])
    omega_nu = cosmo_fid['omega_ncdm']
    # mnu = col('Mnu', 'm_ncdm', default=...)
    # omega_nu = mnu / 93.14

    Omega_m = (wc + wb + omega_nu) / h**2
    if np.isscalar(Omega_m):
        Omega_m = np.full(n, Omega_m)
    return Omega_m


def add_Omega_m(samples, comm=None):
    """
    Adds Omega_m (see :func:`compute_Omega_m`) to the GetDist samples.

    Symmetric with :func:`add_sigma8`: only rank 0 of ``comm`` mutates ``samples``;
    other ranks return ``samples`` unchanged.
    """
    if comm is None:
        comm = COMM_WORLD
    rank = comm.Get_rank()

    if 'Omega_m' in samples.index:
        return samples

    if rank != 0:
        return samples

    Omega_m = compute_Omega_m(samples.samples, samples.getParamNames().list())
    samples.addDerived(Omega_m, name='Omega_m', label=r'\Omega_m')
    samples.updateBaseStatistics()

    return samples


def export_to_text(samples, out_fn, comm=None, engine='class', which='cb'):
    """
    Exports a GetDist samples object to a text file containing only specific columns:
    weights, logAs, Omega_m, h, Omega_b

    MPI-collective: must be called by all ranks in ``comm``. Only rank 0 writes
    the output file and returns the data array; other ranks return ``None``.
    """
    if comm is None:
        comm = COMM_WORLD
    rank = comm.Get_rank()

    # both are collective; only rank 0 receives the updated samples
    samples = add_Omega_m(samples, comm=comm)
    samples = add_sigma8(samples, comm=comm, engine=engine, which=which)

    if rank != 0:
        return None

    n = samples.samples.shape[0]
    weights = samples.weights if hasattr(samples, 'weights') and samples.weights is not None else np.ones(n)
    like = samples.loglikes if hasattr(samples, 'loglikes') and samples.loglikes is not None else np.zeros(n)
    
    # get logAs (checking common names)
    if 'log10As' in samples.index:
        logAs = samples.samples[:, samples.index['log10As']]
    elif 'logA' in samples.index:
        logAs = samples.samples[:, samples.index['logA']]
    # elif 'As' in samples.index:
    #     logAs = samples.samples[:, samples.index['As']]
    else:
        raise ValueError("Could not find logAs, log10As, logA, or As in samples")
        
    # get Omega_m (we just added it)
    Omega_m = samples.samples[:, samples.index['Omega_m']]
    
    # get h
    if 'h' in samples.index:
        h = samples.samples[:, samples.index['h']]
    else:
        h = np.full(n, cosmo_fid['h'])

    # # get physical baryon fraction Omega_b = omega_b / h^2
    # if 'Omega_b' in samples.index:
    #     Omega_b = samples.samples[:, samples.index['Omega_b']]
    # else:
    #     if 'wb' in samples.index:
    #         wb = samples.samples[:, samples.index['wb']]
    #     elif 'omega_b' in samples.index:
    #         wb = samples.samples[:, samples.index['omega_b']]
    #     else:
    #         wb = np.full(n, cosmo_fid['omega_b'])
    #     Omega_b = wb / h**2
    
    # get sigma8
    if 'sigma8' in samples.index:
        sigma8 = samples.samples[:, samples.index['sigma8']]
    else:
        sigma8 = np.full(n, cosmo_fid['sigma8_cb'])

    w0wa_in_chain = False    
    if 'w0' in samples.index:
        w0 = samples.samples[:, samples.index['w0']]
        wa = samples.samples[:, samples.index['wa']]
        w0wa_in_chain = True


    # stack columns
    #data = np.column_stack([weights, logAs, Omega_m, h, Omega_b])
    if not w0wa_in_chain:
        data = np.column_stack([like, weights, Omega_m, sigma8, h, logAs])
        header = "loglike weights Omega_m sigma8 h Omega_b logAs"
    else:
        data = np.column_stack([like, weights, Omega_m, sigma8, h, w0, wa, logAs])
        header = "loglike weights Omega_m sigma8 h w0 wa logAs"
    np.savetxt(out_fn, data, header=header, comments='# ')
    
    return samples

if __name__ == "__main__":
    #test
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import env  # noqa: F401
    import plot_utils as pu
    fn = str(env.CHAINS_DIR / 'Abacus-hf-dr2-v2-altmtl_ELG2_GCcomb_intermfreedom_lambda_pk_dk0.01_kmax0.35-0.25_fullreparam_bk_dk0.01_kmax0.1-0.1.h5')
    samples = pu.get_samples(fn)

    out_fn = 'test_output.txt'
    data = export_to_text(samples, out_fn, engine='comet')
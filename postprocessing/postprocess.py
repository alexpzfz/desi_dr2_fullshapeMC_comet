import numpy as np
from cosmoprimo import Cosmology, fiducial
from mpi4py import MPI
import sys
from comet import comet

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

cosmo_fid = fiducial.AbacusSummit(name='000')
# Fiducial values used for cosmological params not sampled in the chain.
FIDUCIAL = {
    'omega_b': cosmo_fid.get('omega_b'),
    'omega_cdm': cosmo_fid.get('omega_cdm'),
    'h': cosmo_fid.get('h'),
    'n_s': cosmo_fid.get('n_s'),
    'logA': cosmo_fid.get('logA'),
    'm_ncdm': cosmo_fid.get('m_ncdm'),
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


def _compute_sigma8_cosmoprimo(params, engine, which='cb'):
    cosmo = Cosmology(**FIDUCIAL | params, engine=engine)
    if which == 'm':
        return cosmo.sigma8_m
    return cosmo.sigma8_cb # we want the sigma8 of the CDM+baryons (no neutrinos) ??

emu = comet(model='VDG_infty', use_Mpc=False)
def _compute_sigma8_comet(params):
    de_model = 'lambda' if 'wa' not in params else 'w0wa'
    if 'log10As' in params:
        params['As'] = _map_logAs_to_As(params.pop('log10As'))
    scale = 8.0 # Mpc/h
    # scale_Mpc = scale / params['h'] # convert to Mpc
    return emu.sigmaR(scale, FIDUCIAL_COMET | params, de_model=de_model)


def add_sigma8(samples, comm=None, engine='class', which='cb'):
    _engine_options = ['class', 'camb', 'comet']
    if engine not in _engine_options:
        raise ValueError(f"Invalid engine '{engine}'. Valid options are: {_engine_options}")
    if comm is None:
        comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    cosmo_names = [n for n in samples.getParamNames().list() if n in COSMOPRIMO_MAPPING]
    cp_names = map_to_cosmoprimo(cosmo_names)
    idx = [samples.index[n] for n in cosmo_names]
    points = samples.samples[:, idx]
    n = points.shape[0]

    local_idx = np.array_split(np.arange(n), size)[rank]
    if engine in ['class', 'camb']:
        local_sigma8 = np.array(
            [_compute_sigma8_cosmoprimo(dict(zip(cp_names, points[i])), engine=engine, which=which) for i in local_idx]
        )
    elif engine == 'comet':
        local_sigma8 = np.array(
            [_compute_sigma8_comet(dict(zip(cosmo_names, points[i]))) for i in local_idx]
        )

    counts = comm.allgather(local_sigma8.size)
    sigma8 = np.empty(n, dtype=local_sigma8.dtype) if rank == 0 else None
    comm.Gatherv(local_sigma8, (sigma8, counts) if rank == 0 else None, root=0)

    if rank == 0:
        samples.addDerived(sigma8, name='sigma8', label=r'\sigma_8')
        samples.updateBaseStatistics()
    return samples


def add_Omega_m(samples, comm=None):
    """
    Computes Omega_m = (omega_cdm + omega_b + m_ncdm/93.14) / h^2 and adds it to the GetDist samples.

    Symmetric with :func:`add_sigma8`: only rank 0 of ``comm`` mutates ``samples``;
    other ranks return ``samples`` unchanged.
    """
    if comm is None:
        comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if 'Omega_m' in samples.index:
        return samples

    if rank != 0:
        return samples

    n = samples.samples.shape[0]
    
    # Default to fiducial values if parameters are not in the chain
    wc = cosmo_fid.get('omega_cdm')
    wb = cosmo_fid.get('omega_b')
    h = cosmo_fid.get('h')
    
    # Get neutrino mass (handle if it's an array/tuple in some cosmoprimo versions)
    mnu = cosmo_fid.get('m_ncdm')
    omega_nu = cosmo_fid.get('omega_ncdm')
    if isinstance(mnu, (list, tuple, np.ndarray)):
        mnu = np.sum(mnu)
    
    if 'wc' in samples.index:
        wc = samples.samples[:, samples.index['wc']]
    elif 'omega_cdm' in samples.index:
        wc = samples.samples[:, samples.index['omega_cdm']]
        
    if 'wb' in samples.index:
        wb = samples.samples[:, samples.index['wb']]
    elif 'omega_b' in samples.index:
        wb = samples.samples[:, samples.index['omega_b']]
        
    if 'h' in samples.index:
        h = samples.samples[:, samples.index['h']]

    # if 'Mnu' in samples.index:
    #     mnu = samples.samples[:, samples.index['Mnu']]
    # elif 'm_ncdm' in samples.index:
    #     mnu = samples.samples[:, samples.index['m_ncdm']]
        
    # omega_nu = mnu / 93.14
    Omega_m = (wc + wb + omega_nu) / h**2
    if np.isscalar(Omega_m):
        Omega_m = np.full(n, Omega_m)
        
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
        comm = MPI.COMM_WORLD
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
        h = np.full(n, cosmo_fid.get('h'))

    # # get physical baryon fraction Omega_b = omega_b / h^2
    # if 'Omega_b' in samples.index:
    #     Omega_b = samples.samples[:, samples.index['Omega_b']]
    # else:
    #     if 'wb' in samples.index:
    #         wb = samples.samples[:, samples.index['wb']]
    #     elif 'omega_b' in samples.index:
    #         wb = samples.samples[:, samples.index['omega_b']]
    #     else:
    #         wb = np.full(n, cosmo_fid.get('omega_b'))
    #     Omega_b = wb / h**2
    
    # get sigma8
    if 'sigma8' in samples.index:
        sigma8 = samples.samples[:, samples.index['sigma8']]
    else:
        sigma8 = np.full(n, cosmo_fid.get('sigma8'))

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
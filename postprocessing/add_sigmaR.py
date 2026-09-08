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

emu = comet(model='VDG_infty', use_Mpc=True)
def _compute_sigma8_comet(params):
    de_model = 'lambda' if 'wa' not in params else 'w0wa'
    if 'log10As' in params:
        params['As'] = _map_logAs_to_As(params.pop('log10As'))
    scale = 8.0 # Mpc/h
    scale_Mpc = scale / params['h'] # convert to Mpc
    return emu.sigmaR(scale_Mpc, FIDUCIAL_COMET | params, de_model=de_model)


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




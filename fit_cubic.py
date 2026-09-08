import os
import numpy as np
from pathlib import Path
#import lsstypes as types
#from clustering_statistics.tools import get_stats_fn
import matplotlib.pyplot as plt
#from lsstypes import ObservableTree
import sys
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from observables import PowerSpectrumMultipoles, BispectrumSugiyamaMultipoles, JointObservable
from params import Params
from likelihood import Likelihood
from samplers import NautilusSampler
from theory import COMET
from read_data_boxes import get_obs_pk, get_obs_pk_bk
from priors_mc import get_pars



def _fmt_float(x):
    return f"{float(x):.3f}".rstrip('0').rstrip('.')


def get_fn(tracer, freedom, kmaxP, bispec=False, kmaxB=None, hod='fiducial', cosmo='c000',
           de_model='lambda', reparam_option='full', free_Mnu=False, outdir='./chains', extra=None,
           counterterm_basis='DESIct', gausscov=False):

    if isinstance(kmaxP, list):
        kmaxP_str = '-'.join([_fmt_float(k) for k in kmaxP])
    else:
        kmaxP_str = _fmt_float(kmaxP)

    fn = f'{outdir}/Abacus-hf-cubic_{tracer}_{hod}hod_{cosmo}_{freedom}freedom_{de_model}_pk_kmax{kmaxP_str}'    
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
        if kmaxB is not None:
            fn += f'_kmax{kmaxB_str}'
    if counterterm_basis != 'DESIct':
        fn += f'_ct{counterterm_basis}'
    if extra is not None:
        fn += f'_{extra}'
    if gausscov:
        fn += '_gausscov'
    return fn


def get_prior_refs(tracer_list):
    if not isinstance(tracer_list, list):
        tracer_list = [tracer_list]

    #sigmaR_ref_dict = {'LRG': 0.559, 'ELG': 0.504, 'QSO': 0.418}
    sigmaR_ref_dict = {'LRG': 0.559,
                       'ELG': 0.504,
                       'QSO': 0.418}
    b1_ref_dict = {'LRG': 2.047, 'ELG': 1.184, 'QSO': 2.162}
    fsat_dict = {'LRG': 0.13, 'ELG': 0.06, 'QSO': 0.2}
    sigma1_eff_dict = {'LRG': 150/70 * 10**(1/3) * np.sqrt(1 + 0.8),
                       'ELG': 150/70 * 2.1 ** (1/2),
                       'QSO': 150/70 * 10**(0.7/3) * 2.4 ** (1/2)}

    b1_ref = []
    sigmaR_ref = []
    fsat = []
    sigma1_eff = []
    for tracer in tracer_list:
        b1_ref.append(b1_ref_dict[tracer])
        sigmaR_ref.append(sigmaR_ref_dict[tracer])
        fsat.append(fsat_dict[tracer])
        sigma1_eff.append(sigma1_eff_dict[tracer])

    return np.array(b1_ref, dtype=float), np.array(sigmaR_ref, dtype=float), np.array(sigma1_eff, dtype=float), np.array(fsat, dtype=float)


def _parse_tuple_ell(ell_str):
    """Convert string like '000' or '0,0,0' to tuple of ints like (0, 0, 0)"""
    if ',' in ell_str:
        return tuple(int(x) for x in ell_str.split(','))
    else:
        return tuple(int(x) for x in ell_str)


if __name__ == "__main__":
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracer', type=str, default='LRG', help="Tracer label(s) to fit. Should be one of LRG, ELG, QSO.")
    parser.add_argument('--hod', type=str, default='fiducial', choices=['fiducial', 'alternative'], help="HOD option to fit. Should be one of 'fiducial' or 'alternative'.")
    parser.add_argument('--cosmo', type=str, default='c000', help="Cosmology to fit. Should be one of 'c000', 'c001', 'c002', 'c004'. Only 'fiducial' HOD is available for cosmology variations.")
    parser.add_argument('--ellP', type=int, nargs='*', default=[0, 2])
    parser.add_argument('--kminP', type=float, default=0.02, nargs='*')
    parser.add_argument('--kmaxP', type=float, default=0.3, nargs='*')
    parser.add_argument('--bispec', action='store_true')
    parser.add_argument('--ellB', type=_parse_tuple_ell, nargs='*', default=[(0, 0, 0), (2, 0, 2)])
    parser.add_argument('--kminB', type=float, default=0.02, nargs='*')
    parser.add_argument('--kmaxB', type=float, default=0.2, nargs='*')
    parser.add_argument('--reparam', type=str, default='full', choices=['full', 'hybrid', 'none'])
    parser.add_argument('--de_model', type=str, default='lambda', choices=['lambda', 'w0wa', 'w0'])
    parser.add_argument('--counterterm_basis', type=str, default='DESIct', choices=['DESIct', 'Comet'])
    parser.add_argument('--freedom', type=str, default='max', choices=['min', 'max', 'interm'])
    parser.add_argument('--free_Mnu', action='store_true')
    parser.add_argument('--outdir', type=str, default='./chains')
    parser.add_argument('--n_live', type=int, default=2000)
    parser.add_argument('--gausscov', action='store_true', help="Whether to use Gaussian covariance instead of the provided covariance.")
    parser.add_argument('--extra', type=str, default=None, help="Extra string to add to output filename for uniqueness (e.g. to distinguish different sampler settings).")

    args = parser.parse_args()

    os.environ['OMP_NUM_THREADS'] = '1'  # to avoid numpy multithreading issues with multiprocessing

    print(f"Fitting tracer {args.tracer} with HOD {args.hod} and cosmology {args.cosmo}")
    print(f"Power spectrum settings: ellP={args.ellP}, kminP={args.kminP}, kmaxP={args.kmaxP}") 
    if args.bispec:
        print(f"Bispectrum settings: ellB={args.ellB}, kminB={args.kminB}, kmaxB={args.kmaxB}")
    print(f"DE model: {args.de_model}, reparametrization: {args.reparam}, freedom: {args.freedom}, free_Mnu: {args.free_Mnu}, counterterm_basis: {args.counterterm_basis}")


    if not args.bispec:
        obs = get_obs_pk(tracer=args.tracer, hod_opt=args.hod, kmin=args.kminP, kmax=args.kmaxP, ell=args.ellP, nocov=False, cosmo=args.cosmo,
                         gausscov=args.gausscov)
    else:
        obs = get_obs_pk_bk(tracer=args.tracer, hod_opt=args.hod, kminP=args.kminP, kmaxP=args.kmaxP, ellP=args.ellP,
                             kminB=args.kminB, kmaxB=args.kmaxB, ellB=args.ellB, cosmo=args.cosmo)


    b1_ref, sigmaR_ref, sigma1_eff, fsat = get_prior_refs(args.tracer)
    z_array = [obs.cosmo_fid['z']]

    pars = get_pars(bispec=args.bispec, de_model=args.de_model, reparam_option=args.reparam,
                    freedom=args.freedom, free_Mnu=args.free_Mnu,
                    b1_ref=b1_ref, sigmaR_ref=sigmaR_ref, sigma1_eff=sigma1_eff, fsat=fsat,
                    z_array=z_array, counterterm_basis=args.counterterm_basis)

    if args.counterterm_basis == 'DESIct': 
        am_params = ['btd', 'a0', 'a2', 'NP0', 'NP20', 'NP22']
    elif args.counterterm_basis == 'Comet':
        am_params = ['btd', 'c0', 'c2', 'c4', 'NP0', 'NP20', 'NP22']
    if args.bispec:
        am_params += ['NB0', 'MB0']
    
    if args.reparam != 'none':
        am_params = [p + '_r' for p in am_params]

    conditional_prior_fn = None
    if args.de_model == 'w0wa':
        def conditional_prior_fn(params):
            w0 = params['w0']
            wa = params['wa']
            return w0 + wa < 0
    
    likelihood = Likelihood(obs, pars, am_params=am_params, conditional_prior=conditional_prior_fn)

    os.makedirs(args.outdir, exist_ok=True)
    fn = get_fn(tracer=args.tracer, hod=args.hod, cosmo=args.cosmo, freedom=args.freedom, de_model=args.de_model,
                reparam_option=args.reparam, free_Mnu=args.free_Mnu, outdir=args.outdir, extra=args.extra,
                counterterm_basis=args.counterterm_basis, bispec=args.bispec, kmaxP=args.kmaxP, kmaxB=args.kmaxB,
                gausscov=args.gausscov)
    fn_snap = fn + '_nautilus.hdf5'


    # get available cores depending on environment (e.g. SLURM_CPUS_PER_TASK for slurm, or default to os.cpu_count())
    n_threads = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count())) 
    print(f"Using {n_threads} threads for sampling.")
    print(f"Pool size for Nautilus sampler: {n_threads//2}")

    sampler = NautilusSampler(likelihood, n_live=args.n_live, filepath=fn_snap, pool=n_threads//2)

    # run sampler
    t0 = time.time()
    print("Starting sampler...")

    sampler.sample(verbose=True, discard_exploration=True)
    t1 = time.time()
    print(f"Sampler finished in {(t1-t0)/60:.2f} minutes.")
    run_metadata = {**vars(args), 'n_threads': n_threads, 'pool_size': n_threads // 2,
                     'discard_exploration': True, 'elapsed_minutes': (t1 - t0) / 60}
    sampler.save(fn, metadata=run_metadata)
    print(f"Chain saved to {fn}")

    # clean up
    os.remove(fn_snap)


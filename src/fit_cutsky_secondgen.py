import os
import sys
from pathlib import Path
import numpy as np
import lsstypes as types
from clustering_statistics.tools import get_stats_fn
import matplotlib.pyplot as plt
from lsstypes import ObservableTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import env  # noqa: F401

from observables import PowerSpectrumMultipoles, BispectrumSugiyamaMultipoles, JointObservable
from params import Params
from likelihood import Likelihood
from samplers import NautilusSampler
from theory import COMET
from priors_mc import get_pars


stats_dir = Path('/global/cfs/cdirs/desi/mocks/cai/LSS/DA2/mocks/desipipe')

def get_pk(tracer, zrange, region, mocktype, imock, dk=0.005):
    fn = get_stats_fn(stats_dir=stats_dir, kind='mesh2_spectrum', 
                      version=mocktype, tracer=tracer, zrange=zrange, region=region,
                      weight='default-FKP', imock=imock)
    pspectrum = types.read(fn)
    if dk == 0.005:
        pspectrum = pspectrum.select(k=slice(0, None, 5))
    elif dk == 0.01:
        pspectrum = pspectrum.select(k=slice(0, None, 10))
    return pspectrum

def get_bk(tracer, zrange, region, mocktype, imock, dk=0.005):
    fn = get_stats_fn(stats_dir=stats_dir, kind='mesh3_spectrum', 
                      version=mocktype, tracer=tracer, zrange=zrange, region=region,
                      weight='default-FKP', basis='sugiyama-diagonal',imock=imock)
    bk = types.read(fn)
    if dk == 0.005:
        bk = bk.select(k=slice(0, None, 1))
    elif dk == 0.01:
        bk = bk.select(k=slice(0, None, 2))
    return bk

def get_mean_pk(tracer, zrange, region, mocktype, dk=0.005):
    pks = []
    for imock in range(25):
        pks.append(get_pk(tracer, zrange, region, mocktype, imock, dk=dk))
    return ObservableTree.mean(pks)

def get_mean_bk(tracer, zrange, region, mocktype, dk=0.005):
    bks = []
    for imock in range(25):
        bks.append(get_bk(tracer, zrange, region, mocktype, imock, dk=dk))
    return ObservableTree.mean(bks)

def get_cov_pk(tracer, zrange, region, mocktype, rang=(0, 1000), dk=0.005):
    pks = []
    for imock in range(*rang):
        try:
            pks.append(get_pk(tracer, zrange, region, mocktype, imock, dk=dk))
        except:
            continue
    cov = ObservableTree.cov(pks)
    cov.attrs['n_mocks'] = len(pks)
    return cov

def get_cov_bk(tracer, zrange, region, mocktype, rang=(0, 1000), dk=0.005):
    bks = []
    for imock in range(*rang):
        try:
            bks.append(get_bk(tracer, zrange, region, mocktype, imock, dk=dk))
        except:
            continue
    cov = ObservableTree.cov(bks)
    cov.attrs['n_mocks'] = len(bks)
    return cov

def get_cov_pk_bk(tracer, zrange, region, mocktype, rang=(0, 1000), dkP=0.005, dkB=0.005,
                  kminP=0.01, kmaxP=0.4, kminB=0.01, kmaxB=0.2, ellP=[0, 2], ellB=['000', '202']):
    observables = []
    if not isinstance(kminP, list):
        kminP = [kminP] * len(ellP)
    if not isinstance(kmaxP, list):
        kmaxP = [kmaxP] * len(ellP)
    if not isinstance(kminB, list):
        kminB = [kminB] * len(ellB)
    if not isinstance(kmaxB, list):
        kmaxB = [kmaxB] * len(ellB)
    
    ellB_ = [tuple(int(ll) for ll in ell) for ell in ellB]
    for imock in range(*rang):
        try:
            pk = get_pk(tracer, zrange, region, mocktype, imock, dk=dkP)
            bk = get_bk(tracer, zrange, region, mocktype, imock, dk=dkB)

            pk = pk.get(ells=ellP)
            bk = bk.get(ells=ellB_)

            for i, ell in enumerate(ellP):
                pk = pk.at(ells=ell).select(k=(kminP[i], kmaxP[i]))
            for i, ell in enumerate(ellB_):
                bk = bk.at(ells=ell).select(k=(kminB[i], kmaxB[i]))
            
            tree = ObservableTree([pk, bk], observables=['spectrum2', 'spectrum3'])
            observables.append(tree)
        except:
            continue
    cov = types.cov(observables)
    cov.attrs['n_mocks'] = len(observables)
    return cov

def get_window_pk(tracer, zrange, region, mocktype):
    fn = get_stats_fn(stats_dir=stats_dir, kind='window_mesh2_spectrum', 
                      version=mocktype, tracer=tracer, zrange=zrange, region=region,
                      weight='default-FKP', imock=0)
    window = types.read(fn)
    return window

def get_window_bk(tracer, zrange, region, mocktype):
    fn = get_stats_fn(stats_dir=stats_dir, kind='window_mesh3_spectrum', 
                      version=mocktype, tracer=tracer, zrange=zrange, region=region,
                      weight='default-FKP', basis='sugiyama-diagonal', imock=0)
    window = types.read(fn)
    return window

cosmo_fid = {'wb': 0.02237, 'wc': 0.1200, 'h': 0.6736, 'As': 2.0830, 'ns': 0.9649, 'Mnu': 0.06, 'z': 0.725}
# From this point on, everything follows my own conventions!
def get_obs_pk(tracer, zrange, region, mocktype, kmin=0.01, kmax=0.3,
               kwinmin=0.0, kwinmax=0.5, ellwin=[0, 2], nocov=False, mocktype_cov='holi-v1-altmtl', dk=0.005,
               force_nmocks=None):
    pk = get_mean_pk(tracer, zrange, region, mocktype, dk=dk)
    pk = pk.get(ells=[0, 2])
    p0 = pk.get(ells=0).value()
    p2 = pk.get(ells=2).value()
    k = pk.get(ells=0).coords('k')
    nbar = 1/pk.get(ells=0).values('shotnoise')[0]
    #nbar = 1 / pk.get(ells=0).values('num_shotnoise')[0]
    cov = None
    if not nocov:
        tracer_cov_str = tracer if not 'ELG' in tracer else 'ELG_LOPnotqso'
        cov_ = get_cov_pk(tracer_cov_str, zrange, region, mocktype_cov, rang=(0, 1000), dk=dk)
        cov_ = cov_.at.observable.match(pk)
        cov = cov_.value()
 
    nmocks = cov_.attrs['n_mocks'] if cov is not None else None
    if force_nmocks is not None:
        nmocks = force_nmocks
        print(f"Warning: forcing number of mocks to {nmocks}, but actual number in covariance is {cov_.attrs['n_mocks']}")

    window_ = get_window_pk(tracer, zrange, region, mocktype)
    if not isinstance(kwinmin, list):
        kwinmin = [kwinmin] * len(ellwin)
    if not isinstance(kwinmax, list):
        kwinmax = [kwinmax] * len(ellwin)

    window_ = window_.at.theory.get(ells=ellwin)
    for i, ell in enumerate(ellwin):
        window_ = window_.at.theory.at(ells=ell).select(k=(kwinmin[i], kwinmax[i]))
    
    zeff = window_.observable.get(ells=0).attrs['zeff']
    window_ = window_.at.observable.match(pk)
    win = window_.value()
    kwin = []
    for i, ell in enumerate(ellwin):
        kwin.append(window_.theory.get(ells=ell).coords('k'))
    

    obs = PowerSpectrumMultipoles(k, [p0, p2], ell=[0, 2], cov=cov, nbar=nbar,
                                  cosmo_fid=cosmo_fid | {'z': zeff}, kmin=kmin, kmax=kmax,
                                  wmat=win, kwin=kwin, ellwin=ellwin, nmocks_cov=nmocks)
    return obs

def get_obs_bk(tracer, zrange, region, mocktype, kmin=0.01, kmax=0.3,
               kwinmin=0.0, kwinmax=0.5, ellwin=[(0, 0, 0), (0, 2, 2), (1, 1, 0), (1, 1, 2), (2, 2, 0)], nocov=False,
               mocktype_cov='holi-v1-altmtl', dk=0.005, slice_win_theory=2, force_nmocks=None):
    bk = get_mean_bk(tracer, zrange, region, mocktype, dk=dk)
    bk = bk.get(ells=[(0, 0, 0), (2, 0, 2)])
    b000 = bk.get(ells=(0, 0, 0)).value()
    b202 = bk.get(ells=(2, 0, 2)).value()
    k1k2 = bk.get(ells=(0, 0, 0)).coords('k') # shape (n, 2)
    cov = None
    if not nocov:
        tracer_cov_str = tracer if not 'ELG' in tracer else 'ELG_LOPnotqso'
        cov_ = get_cov_bk(tracer_cov_str, zrange, region, mocktype_cov, rang=(0, 1000), dk=dk)
        cov_ = cov_.at.observable.match(bk)
        cov = cov_.value()
    nmocks = cov_.attrs['n_mocks'] if cov is not None else None
    if force_nmocks is not None:
        nmocks = force_nmocks
        print(f"Warning: forcing number of mocks to {nmocks}, but actual number in covariance is {cov_.attrs['n_mocks']}")
    
    window_ = get_window_bk(tracer, zrange, region, mocktype)
    ellwin_ = [tuple(int(ll) for ll in ell) for ell in ellwin]
    window_ = window_.at.theory.get(ells=ellwin_)
    window_ = window_.at.theory.select(k=slice(0, None, slice_win_theory))
    if not isinstance(kwinmin, list):
        kwinmin = [kwinmin] * len(ellwin)
    if not isinstance(kwinmax, list):
        kwinmax = [kwinmax] * len(ellwin)
    window_ = window_.at.theory.get(ells=ellwin_)
    for i, ell in enumerate(ellwin_):
        window_ = window_.at.theory.at(ells=ell).select(k=(kwinmin[i], kwinmax[i]))

    zeff = window_.observable.get(ells=(0, 0, 0)).attrs['zeff']
    window_ = window_.at.observable.match(bk)
    win = window_.value()
    pairwin = []
    for i, ell in enumerate(ellwin_):
        pairwin.append(window_.theory.get(ells=ell).coords('k'))
    obs = BispectrumSugiyamaMultipoles(k1k2, [b000, b202], ell=[(0, 0, 0), (2, 0, 2)], cov=cov,
                                       cosmo_fid=cosmo_fid | {'z': zeff}, kmin=kmin, kmax=kmax,
                                       wmat=win, pairwin=pairwin, ellwin=ellwin, nmocks_cov=nmocks)
    return obs

def get_obs_joint_pk_bk(tracer, zrange, region, mocktype, kminP=0.01, kmaxP=0.3,
                kwinminP=0.0, kwinmaxP=0.5, ellwinP=[0, 2, 4], kminB=0.01, kmaxB=0.2,
                kwinminB=0.0, kwinmaxB=0.5, ellwinB=[(0, 0, 0), (0, 2, 2), (1, 1, 0), (1, 1, 2), (2, 2, 0)],
                mocktype_cov='holi-v1-altmtl', dkP=0.005, dkB=0.005, slice_winB_theory=2, force_nmocks=None):
    obs_pk = get_obs_pk(tracer, zrange, region, mocktype, kmin=kminP, kmax=kmaxP,
                            kwinmin=kwinminP, kwinmax=kwinmaxP, ellwin=ellwinP,
                            nocov=True, mocktype_cov=mocktype_cov, dk=dkP,
                            force_nmocks=None)
    obs_bk = get_obs_bk(tracer, zrange, region, mocktype, kmin=kminB, kmax=kmaxB,
                            kwinmin=kwinminB, kwinmax=kwinmaxB, ellwin=ellwinB,
                            nocov=True, mocktype_cov=mocktype_cov, dk=dkB,
                            slice_win_theory=slice_winB_theory,
                            force_nmocks=None)

    tracer_cov_str = tracer if not 'ELG' in tracer else 'ELG_LOPnotqso' 
    cov_ = get_cov_pk_bk(tracer_cov_str, zrange, region, mocktype_cov, rang=(0, 1000), dkP=dkP, dkB=dkB, 
                         kminP=kminP, kmaxP=kmaxP, kminB=kminB, kmaxB=kmaxB)
    nmocks = cov_.attrs['n_mocks']
    cov = cov_.value()
    if force_nmocks is not None:
        nmocks = force_nmocks
        print(f"Warning: forcing number of mocks to {nmocks}, but actual number in covariance is {cov_.attrs['n_mocks']}")
    obs = JointObservable(obs_pk, obs_bk, cov=cov, nmocks_cov=nmocks)
    return obs
    


tracer_label_dict = {'LRG1': {'tracer': 'LRG', 'zrange': (0.4, 0.6)},
                    'LRG2': {'tracer': 'LRG', 'zrange': (0.6, 0.8)},
                    'LRG3': {'tracer': 'LRG', 'zrange': (0.8, 1.1)},
                    'ELG1': {'tracer': 'ELG_LOP', 'zrange': (0.8, 1.1)},
                    'ELG2': {'tracer': 'ELG_LOP', 'zrange': (1.1, 1.6)},
                    'QSO': {'tracer': 'QSO', 'zrange': (0.8, 2.1)}}


def _fmt_float(x):
    return f"{float(x):.3f}".rstrip('0').rstrip('.')


def get_fn(tracer_label, region, freedom, dkP, kmaxP, bispec=False, dkB=None, kmaxB=None,
           de_model='lambda', reparam_option='full', free_Mnu=False, outdir=str(env.CHAINS_DIR), extra=None):

    if not isinstance(tracer_label, list):
        tracer_label = [tracer_label] 

    if isinstance(kmaxP, list):
        kmaxP_str = '-'.join([_fmt_float(k) for k in kmaxP])
    else:
        kmaxP_str = _fmt_float(kmaxP)

    tracer_str = '-'.join(tracer_label) if len(tracer_label) > 1 else tracer_label[0]
    
    fn = f'{outdir}/Abacus_2ndgen_complete_{tracer_str}_{region}_{freedom}freedom_{de_model}_pk_dk{_fmt_float(dkP)}_kmax{kmaxP_str}'
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
        'LRG': {(0.4, 0.6): 0.621, (0.6, 0.8): 0.564, (0.8, 1.1): 0.510},
        'ELG_LOP': {(0.8, 1.1): 0.503, (1.1, 1.6): 0.431},
        'QSO': {(0.8, 2.1): 0.405},
    }
    b1_ref_dict = {
        'LRG': {(0.4, 0.6): 1.883, (0.6, 0.8): 2.031, (0.8, 1.1): 2.216},
        'ELG_LOP': {(0.8, 1.1): 1.187, (1.1, 1.6): 1.429},
        'QSO': {(0.8, 2.1): 2.266},
    }
    fsat_dict = {'LRG': 0.13, 'ELG_LOP': 0.06, 'QSO': 0.2}

    b1_ref = []
    sigmaR_ref = []
    fsat = []
    sigma1_eff = []
    for tracer, zr in zip(tracer_list, zrange_list):
        key = _tracer_key(tracer)
        b1_ref.append(b1_ref_dict[key][tuple(zr)])
        sigmaR_ref.append(sigmaR_ref_dict[key][tuple(zr)])
        fsat.append(fsat_dict[key])
        if key == 'LRG':
            sigma1_eff.append(150/70 * 10**(1/3) * np.sqrt(1 + 0.8))
        elif key == 'ELG_LOP':
            sigma1_eff.append(150/70 * 2.1 ** (1/2))
        else:
            sigma1_eff.append(150/70 * 10**(0.7/3) * 2.4 ** (1/2))

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
    parser.add_argument('--tracer_label', type=str, default='LRG', nargs='*', help="Tracer label(s) to fit. Should be one of LRG1, LRG2, LRG3, ELG1, ELG2, QSO. If multiple are provided, they will be fit simultaneously with shared cosmological parameters but independent nuisance parameters.")
    parser.add_argument('--region', type=str, default='GCcomb')
    parser.add_argument('--mocktype', type=str, default='abacus-2ndgen-complete')
    parser.add_argument('--kminP', type=float, default=0.01, nargs='*')
    parser.add_argument('--kmaxP', type=float, default=0.3, nargs='*')
    parser.add_argument('--ellwinP', type=int, nargs='*', default=[0, 2, 4])
    parser.add_argument('--kwinminP', type=float, default=0.0, nargs='*')
    parser.add_argument('--kwinmaxP', type=float, default=0.5, nargs='*')
    parser.add_argument('--dkP', type=float, default=0.01)
    parser.add_argument('--bispec', action='store_true')
    parser.add_argument('--kminB', type=float, default=0.01, nargs='*')
    parser.add_argument('--kmaxB', type=float, default=0.2, nargs='*')
    parser.add_argument('--ellwinB', type=_parse_tuple_ell, nargs='*', default=[(0, 0, 0), (0, 2, 2)])
    parser.add_argument('--kwinminB', type=float, default=0.005, nargs='*')
    parser.add_argument('--kwinmaxB', type=float, default=0.3, nargs='*')
    parser.add_argument('--dkB', type=float, default=0.01)
    parser.add_argument('--reparam', type=str, default='full', choices=['full', 'hybrid', 'none'])
    parser.add_argument('--de_model', type=str, default='lambda', choices=['lambda', 'w0wa', 'w0'])
    parser.add_argument('--freedom', type=str, default='max', choices=['min', 'max', 'adhoc'])
    parser.add_argument('--free_Mnu', action='store_true')
    parser.add_argument('--force_nmocks', type=int, default=None, help="Force the number of mocks used for covariance rescaling. Use with caution!")
    parser.add_argument('--outdir', type=str, default=str(env.CHAINS_DIR))

    args = parser.parse_args()

    os.environ['OMP_NUM_THREADS'] = '1'  # to avoid numpy multithreading issues with multiprocessing

    print(f"Fitting tracer(s) {args.tracer_label} in region {args.region} with mock type {args.mocktype}")
    print(f"Power spectrum settings: kminP={args.kminP}, kmaxP={args.kmaxP}, ellwinP={args.ellwinP}, kwinminP={args.kwinminP}, kwinmaxP={args.kwinmaxP}, dkP={args.dkP}")
    if args.bispec:
        print(f"Bispectrum settings: kminB={args.kminB}, kmaxB={args.kmaxB}, ellwinB={args.ellwinB}, kwinminB={args.kwinminB}, kwinmaxB={args.kwinmaxB}, dkB={args.dkB}")
    print(f"DE model: {args.de_model}, reparametrization: {args.reparam}, freedom: {args.freedom}, free_Mnu: {args.free_Mnu}")


    tracer_list = []
    zrange_list = []
    for label in args.tracer_label:
        if label not in tracer_label_dict:
            raise ValueError(f"Invalid tracer label {label}. Should be one of {list(tracer_label_dict.keys())}")
        info = tracer_label_dict[label]
        tracer_list.append(info['tracer'])
        zrange_list.append(info['zrange'])

    # Build one observable per z-bin for simultaneous fit
    observables = []
    for tracer_i, zr in zip(tracer_list, zrange_list):
        if not args.bispec:
            obs_i = get_obs_pk(tracer=tracer_i, zrange=zr, region=args.region, mocktype=args.mocktype,
                               kmin=args.kminP, kmax=args.kmaxP, ellwin=args.ellwinP,
                               kwinmin=args.kwinminP, kwinmax=args.kwinmaxP, dk=args.dkP,
                               force_nmocks=args.force_nmocks)
        else:
            obs_i = get_obs_joint_pk_bk(tracer=tracer_i, zrange=zr, region=args.region, mocktype=args.mocktype,
                                        kminP=args.kminP, kmaxP=args.kmaxP, ellwinP=args.ellwinP,
                                        kwinminP=args.kwinminP, kwinmaxP=args.kwinmaxP, dkP=args.dkP,
                                        kminB=args.kminB, kmaxB=args.kmaxB, ellwinB=args.ellwinB,
                                        kwinminB=args.kwinminB, kwinmaxB=args.kwinmaxB, dkB=args.dkB,
                                        force_nmocks=args.force_nmocks)
        observables.append(obs_i)


    b1_ref, sigmaR_ref, sigma1_eff, fsat = get_prior_refs(tracer_list, zrange_list)
    z_array = [obs.cosmo_fid['z'] for obs in observables]
    # sort z_array
    z_array = np.array(z_array)
    sort_idx = np.argsort(z_array)
    z_array = z_array[sort_idx]
    observables = [observables[i] for i in sort_idx]

    pars = get_pars(bispec=args.bispec, de_model=args.de_model, reparam_option=args.reparam,
                              freedom=args.freedom, free_Mnu=args.free_Mnu,
                              b1_ref=b1_ref, sigmaR_ref=sigmaR_ref, sigma1_eff=sigma1_eff, fsat=fsat,
                              z_array=z_array)
    
    am_params = ['btd_r', 'a0_r', 'a2_r', 'NP0_r', 'NP20_r', 'NP22_r']
    if args.bispec:
        am_params += ['NB0_r', 'MB0_r']
        pars.emu.bispec_kwargs['sugiyama']['use_pdw_interp'] = True


    conditional_prior_fn = None
    if args.de_model == 'w0wa':
        def conditional_prior_fn(params):
            w0 = params['w0']
            wa = params['wa']
            return w0 + wa <=0
    
    likelihood = Likelihood(observables, pars, am_params=am_params, conditional_prior=conditional_prior_fn)

    os.makedirs(args.outdir, exist_ok=True)
    fn = get_fn(tracer_label=args.tracer_label, region=args.region,  freedom=args.freedom, dkP=args.dkP, kmaxP=args.kmaxP,
                bispec=args.bispec, dkB=args.dkB, kmaxB=args.kmaxB,
                de_model=args.de_model, reparam_option=args.reparam, free_Mnu=args.free_Mnu, outdir=args.outdir)
    if args.force_nmocks is not None:
        fn += f'_forcednmocks{args.force_nmocks}'
    fn_snap = fn + '_nautilus.hdf5'


    # get available cores depending on environment (e.g. SLURM_CPUS_PER_TASK for slurm, or default to os.cpu_count())
    n_threads = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count())) 
    print(f"Using {n_threads} threads for sampling.")

    sampler = NautilusSampler(likelihood, filepath=fn_snap, pool=n_threads)

    # run sampler
    t0 = time.time()
    print("Starting sampler...")

    sampler.sample(verbose=True)
    t1 = time.time()
    print(f"Sampler finished in {(t1-t0)/60:.2f} minutes.")
    run_metadata = {**vars(args), 'n_threads': n_threads, 'pool_size': n_threads,
                     'elapsed_minutes': (t1 - t0) / 60}
    sampler.save(fn, metadata=run_metadata)
    print(f"Chain saved to {fn}")

    # clean up
    os.remove(fn_snap)


import os
import numpy as np
from pathlib import Path
import lsstypes as types
from clustering_statistics.tools import get_stats_fn
from clustering_statistics.box_tools import get_box_stats_fn
import matplotlib.pyplot as plt
from lsstypes import ObservableTree
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import env  # noqa: F401
from observables import PowerSpectrumMultipoles, BispectrumSugiyamaMultipoles, JointObservable
from utils import cut_cov, cut_window


stats_dir = Path('/global/cfs/cdirs/desi/science/cai/desi-clustering/dr2/summary_statistics/')
#stats_dir_bgs = Path('/global/cfs/cdirs/desi/science/cai/desi-clustering/dr2/summary_statistics/full_shape/fiber_assignment_systematics/abacus-2ndgen-dr2-altmtl/')
project = 'full_shape/base'
project_bgs = 'full_shape/fiber_assignment_systematics'
stats_dir_box = Path('/global/cfs/cdirs/desi/science/cai/desi-clustering/dr2/summary_statistics/mock_challenge/')
stats_dir_ezmocks = Path('/global/cfs/cdirs/desi/science/gqc/y3_fits/mockchallenge_abacus/measurements/EZmocks_lsstypes/')

def count_available_mocks(tracer, zrange, region, mocktype, kind='mesh2_spectrum', stats_dir=stats_dir, project=project):
    count = 0
    for imock in range(1000):
        fn = get_stats_fn(stats_dir=stats_dir, project=project, kind=kind, 
                          version=mocktype, tracer=tracer, zrange=zrange, region=region,
                          weight='default-FKP', imock=imock)
        if os.path.exists(fn):
            count += 1
        else:
            continue
    return count


def get_pk(tracer, zrange=None, region=None, mocktype=None, imock=None,
           zsnap=None, cosmo=None, hod=None, dk=0.005, stats_dir=stats_dir, project=project):
    if not 'cubic' in mocktype:
        fn = get_stats_fn(stats_dir=stats_dir, project=project, kind='mesh2_spectrum', 
                        version=mocktype, tracer=tracer, zrange=zrange, region=region,
                        weight='default-FKP', imock=imock)
    elif mocktype == 'abacushf_cubic':
        fn = get_box_stats_fn(stats_dir=stats_dir_box, tracer=tracer, version=f'v2/{tracer}', project='abacushf_MC/', 
                              zsnap=zsnap, cosmo=cosmo, hod=hod, kind='mesh2_spectrum', imock=imock)
    elif mocktype == 'ezmocks_cubic':
        fn = get_box_stats_fn(stats_dir=stats_dir_ezmocks, tracer=tracer, zsnap=zsnap, kind='mesh2_spectrum', imock=imock)
        fn = Path(str(fn).replace('los-z_', ''))

    pspectrum = types.read(fn)
    if dk == 0.005:
        pspectrum = pspectrum.select(k=slice(0, None, 5))
    elif dk == 0.01:
        pspectrum = pspectrum.select(k=slice(0, None, 10))
    return pspectrum

def get_bk(tracer, zrange=None, region=None, mocktype=None, imock=None,
           zsnap=None, cosmo=None, hod=None, dk=0.005, stats_dir=stats_dir, project=project):
    if not 'cubic' in mocktype:
        fn = get_stats_fn(stats_dir=stats_dir, project=project, kind='mesh3_spectrum', 
                          version=mocktype, tracer=tracer, zrange=zrange, region=region,
                          weight='default-FKP', basis='sugiyama-diagonal',imock=imock)
    elif mocktype == 'abacushf_cubic':
        fn = get_box_stats_fn(stats_dir=stats_dir_box, tracer=tracer, version=f'v2/{tracer}', project='abacushf_MC/', 
                              zsnap=zsnap, cosmo=cosmo, hod=hod, kind='mesh3_spectrum', basis='sugiyama-diagonal', imock=imock)
    elif mocktype == 'ezmocks_cubic':
        fn = get_box_stats_fn(stats_dir=stats_dir_ezmocks, tracer=tracer, zsnap=zsnap, kind='mesh3_spectrum', basis='sugiyama-diagonal', imock=imock)
        fn = Path(str(fn).replace('los-z_', ''))
    bk = types.read(fn)
    if dk == 0.005:
        bk = bk.select(k=slice(0, None, 1))
    elif dk == 0.01:
        bk = bk.select(k=slice(0, None, 2))
    return bk

def get_mean_pk(tracer, zrange=None, region=None, mocktype=None, zsnap=None, cosmo=None, hod=None, dk=0.005, stats_dir=stats_dir, project=project):
    pks = []
    for imock in range(25):
        pks.append(get_pk(tracer, zrange, region, mocktype, imock, zsnap=zsnap, cosmo=cosmo, hod=hod, dk=dk, stats_dir=stats_dir, project=project))
    return ObservableTree.mean(pks)

def get_mean_bk(tracer, zrange=None, region=None, mocktype=None, zsnap=None, cosmo=None, hod=None, dk=0.005, stats_dir=stats_dir, project=project):
    bks = []
    for imock in range(25):
        bks.append(get_bk(tracer, zrange, region, mocktype, imock, zsnap=zsnap, cosmo=cosmo, hod=hod, dk=dk, stats_dir=stats_dir, project=project))
    return ObservableTree.mean(bks)

def get_cov_pk(tracer, zrange=None, region=None, mocktype=None, zsnap=None, cosmo=None, hod=None, rang=(0, 1000), dk=0.005, stats_dir=stats_dir, project=project):
    pks = []
    for imock in range(*rang):
        try:
            pks.append(get_pk(tracer, zrange, region, mocktype, imock, zsnap=zsnap, cosmo=cosmo, hod=hod, dk=dk, stats_dir=stats_dir, project=project))
        except:
            continue
    cov = ObservableTree.cov(pks)
    cov.attrs['n_mocks'] = len(pks)
    return cov

def get_cov_bk(tracer, zrange=None, region=None, mocktype=None, zsnap=None, cosmo=None, hod=None, rang=(0, 1000), dk=0.005, stats_dir=stats_dir, project=project):
    bks = []
    for imock in range(*rang):
        try:
            bks.append(get_bk(tracer, zrange, region, mocktype, imock, zsnap=zsnap, cosmo=cosmo, hod=hod, dk=dk, stats_dir=stats_dir, project=project))
        except:
            continue
    cov = ObservableTree.cov(bks)
    cov.attrs['n_mocks'] = len(bks)
    return cov

def get_cov_pk_bk_preprocessed(tracer, zrange=None, region=None, mocktype=None, zsnap=None, cosmo=None, hod=None, rang=(0, 1000), dkP=0.005, dkB=0.005,
                  kminP=0.01, kmaxP=0.4, kminB=0.01, kmaxB=0.2, ellP=[0, 2], ellB=['000', '202'], stats_dir=stats_dir, project=project):
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
            pk = get_pk(tracer, zrange, region, mocktype, imock, zsnap=zsnap, cosmo=cosmo, hod=hod, dk=dkP, stats_dir=stats_dir, project=project)
            bk = get_bk(tracer, zrange, region, mocktype, imock, zsnap=zsnap, cosmo=cosmo, hod=hod, dk=dkB, stats_dir=stats_dir, project=project)

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

def get_cov_pk_bk(tracer, zrange=None, region=None, mocktype=None, zsnap=None, cosmo=None, hod=None, rang=(0, 1000), dkP=0.005, dkB=0.005, stats_dir=stats_dir, project=project):
    observables = []
    for imock in range(*rang):
        try:
            pk = get_pk(tracer, zrange, region, mocktype, imock, zsnap=zsnap, cosmo=cosmo, hod=hod, dk=dkP, stats_dir=stats_dir, project=project)
            bk = get_bk(tracer, zrange, region, mocktype, imock, zsnap=zsnap, cosmo=cosmo, hod=hod, dk=dkB, stats_dir=stats_dir, project=project)
            tree = ObservableTree([pk, bk], observables=['spectrum2', 'spectrum3'])
            observables.append(tree)
        except:
            continue
    cov = types.cov(observables)
    cov.attrs['n_mocks'] = len(observables)
    return cov

def get_window_pk(tracer, zrange, region, mocktype, stats_dir=stats_dir, project=project):
    fn = get_stats_fn(stats_dir=stats_dir, project=project, kind='window_mesh2_spectrum', 
                      version=mocktype, tracer=tracer, zrange=zrange, region=region,
                      weight='default-FKP', imock=0)
    window = types.read(fn)
    return window

def get_window_bk(tracer, zrange, region, mocktype, stats_dir=stats_dir, project=project):
    fn = get_stats_fn(stats_dir=stats_dir, project=project, kind='window_mesh3_spectrum', 
                      version=mocktype, tracer=tracer, zrange=zrange, region=region,
                      weight='default-FKP', basis='sugiyama-diagonal', imock=0)
    window = types.read(fn)
    return window


tracers = ['BGS', 'LRG', 'ELG', 'QSO']
zranges = {'BGS': [(0.1, 0.4)],
           'LRG': [(0.4, 0.6), (0.6, 0.8), (0.8, 1.1)],
           'ELG': [(0.8, 1.1), (1.1, 1.6)],
           'QSO': [(0.8, 2.1)]}
tracer_labels = {'BGS': {(0.1, 0.4): 'BGS'},
                 'LRG': {(0.4, 0.6): 'LRG1', (0.6, 0.8): 'LRG2', (0.8, 1.1): 'LRG3'},
                 'ELG': {(0.8, 1.1): 'ELG1', (1.1, 1.6): 'ELG2'},
                 'QSO': {(0.8, 2.1): 'QSO'}}


def get_fn(outdir, kind, mocktype, label, region, dkP, dkB=None, slice_winB=None, n_mocks_cov=None):
    if kind == 'pk':
        return Path.joinpath(outdir, f'mean_pk_{mocktype}_{label}_{region}_dk{dkP}.txt')
    elif kind == 'cov_pk':
        return Path.joinpath(outdir, f'cov_pk_{mocktype}_{label}_{region}_dk{dkP}_nmocks{n_mocks_cov}.txt')
    elif kind == 'window_pk':
        return Path.joinpath(outdir, f'window_pk_{mocktype}_{label}_{region}_dk{dkP}.txt')
    elif kind == 'window_pk_k':
        return Path.joinpath(outdir, f'window_pk_kth_{mocktype}_{label}_{region}_dk{dkP}.txt')
    elif kind == 'bk':
        return Path.joinpath(outdir, f'mean_bk_sugiyama_diag_{mocktype}_{label}_{region}_dk{dkB}.txt')
    elif kind == 'cov_pk_bk':
        return Path.joinpath(outdir, f'cov_pk_bk_sugiyama_diag_{mocktype}_{label}_{region}_dkP{dkP}_dkB{dkB}_nmocks{n_mocks_cov}.txt')
    elif kind == 'window_bk':
        if slice_winB is None:
            return Path.joinpath(outdir, f'window_bk_sugiyama_diag_{mocktype}_{label}_{region}_dk{dkB}.txt')
        else:
            return Path.joinpath(outdir, f'window_bk_sugiyama_diag_{mocktype}_{label}_{region}_dk{dkB}_slice{slice_winB}.txt')
    elif kind == 'window_bk_k':
        if slice_winB is None:
            return Path.joinpath(outdir, f'window_bk_sugiyama_diag_kth_{mocktype}_{label}_{region}_dk{dkB}.txt')
        else:
            return Path.joinpath(outdir, f'window_bk_sugiyama_diag_kth_{mocktype}_{label}_{region}_dk{dkB}_slice{slice_winB}.txt')



def get_fn_box(outdir, kind, mock_type, tracer, zsnap, cosmo, hod, dkP, dkB=None, n_mocks_cov=None):
    if kind == 'pk':
        return Path.joinpath(outdir, f'mean_pk_{mock_type}_{tracer}_zsnap{zsnap}_cosmo{cosmo}_hod{hod}_dk{dkP}.txt')
    elif kind == 'cov_pk':
        return Path.joinpath(outdir, f'cov_pk_{mock_type}_{tracer}_zsnap{zsnap}_cosmo{cosmo}_hod{hod}_dk{dkP}_nmocks{n_mocks_cov}.txt')
    elif kind == 'bk':
        return Path.joinpath(outdir, f'mean_bk_sugiyama_diag_{mock_type}_{tracer}_zsnap{zsnap}_cosmo{cosmo}_hod{hod}_dk{dkB}.txt')
    elif kind == 'cov_pk_bk':
        return Path.joinpath(outdir, f'cov_pk_bk_sugiyama_diag_{mock_type}_{tracer}_zsnap{zsnap}_cosmo{cosmo}_hod{hod}_dkP{dkP}_dkB{dkB}_nmocks{n_mocks_cov}.txt')
        
    
cosmo_fid = {'wb': 0.02237, 'wc': 0.1200, 'h': 0.6736, 'As': 2.0830, 'ns': 0.9649, 'Mnu': 0.06}
def get_obs_pk_raw(tracer, zrange, region, mocktype, kmin=0.02, kmax=0.3,
                   kwinmin=0.0, kwinmax=0.5, ell=[0, 2], ellwin=[0, 2, 4], nocov=False, mocktype_cov='holi-v3-altmtl', dk=0.005,
                   use_Mpc=True):

    tracer_str = tracer if not 'ELG' in tracer else 'ELG_LOPnotqso'
    pk = get_mean_pk(tracer_str, zrange, region, mocktype, dk=dk)
    pk = pk.get(ells=ell)
    pell_list = [pk.get(ells=ll).value() for ll in ell]
    k = pk.get(ells=0).coords('k')
    nbar = 1/pk.get(ells=0).values('shotnoise')[0]
    cov = None
    if not nocov:
        cov_ = get_cov_pk(tracer_str, zrange, region, mocktype_cov, rang=(0, 1000), dk=dk)
        cov_ = cov_.at.observable.match(pk)
        cov = cov_.value()
 
    nmocks = cov_.attrs['n_mocks'] if cov is not None else None
    window_ = get_window_pk(tracer_str, zrange, region, mocktype)
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
    

    obs = PowerSpectrumMultipoles(k, pell_list, ell=ell, cov=cov, nbar=nbar,
                                  cosmo_fid=cosmo_fid | {'z': zeff}, kmin=kmin, kmax=kmax,
                                  wmat=win, kwin=kwin, ellwin=ellwin, nmocks_cov=nmocks, save_Mpc_units=use_Mpc)
    return obs

def get_obs_pk_cached(tracer, zrange, region, mocktype, kmin=0.02, kmax=0.3,
                     kwinmin=0.0, kwinmax=0.5, ell=[0, 2], ellwin=[0, 2, 4], nocov=False, mocktype_cov='holi-v3-altmtl', dk=0.005,
                     outdir='/global/cfs/cdirs/desicollab/users/alexpzfz/DR2_2pt3pt/data_for_mock_challenge/cutsky/',
                     use_Mpc=True):
    label = tracer_labels[tracer][zrange]
    outdir = Path(outdir)
    fn_mean = get_fn(outdir, 'pk', mocktype, label, region, dkP=dk)
    # read header to get zeff and nbar
    with open(fn_mean, 'r') as f:
        header = f.readlines()[:2]
    zeff = float(header[0].split('=')[1])
    nbar = float(header[1].split('=')[1])
    k, p0, p2, p4 = np.loadtxt(fn_mean, unpack=True)
    pell_dict = {0: p0, 2: p2, 4: p4} 
    pell_list = [pell_dict[ll] for ll in ell]
    cov = None
    n_mocks_cov = None
    if not nocov:
        fn_cov_ = get_fn(outdir, 'cov_pk', mocktype_cov, label, region, dkP=dk, n_mocks_cov='*')
        # as we don't know the number of mocks in advance, we need to find the file that matches the pattern
        fn_cov_base = fn_cov_.name
        fn_cov_list = list(outdir.glob(fn_cov_base))
        if len(fn_cov_list) == 0:
            raise FileNotFoundError(f"No covariance file found for pattern {fn_cov_}")
        elif len(fn_cov_list) > 1:
            print(f"Multiple covariance files found for pattern {fn_cov_}: {fn_cov_list}")
            print('Selecting the file with the highest number of mocks.')
            fn_cov = max(fn_cov_list, key=lambda x: int(x.stem.split('nmocks')[1]))
        elif len(fn_cov_list) == 1:
            fn_cov = fn_cov_list[0]
        # read number of mocks from the filename
        print(f"Using covariance file: {fn_cov}")
        n_mocks_cov = int(fn_cov.stem.split('nmocks')[1])
        cov = np.loadtxt(fn_cov)
        cov = cut_cov(cov, k, ell=[0, 2, 4], ell_select=ell)
    
    fn_window = get_fn(outdir, 'window_pk', mocktype, label, region, dkP=dk)
    fn_window_k = get_fn(outdir, 'window_pk_k', mocktype, label, region, dkP=dk)
    win = np.loadtxt(fn_window)
    kwin = np.loadtxt(fn_window_k)
    win = cut_window(win, x=k, ell=[0, 2, 4], xwin=kwin, ellwin=[0, 2, 4], ell_select=ell, ellwin_select=ellwin)

    obs = PowerSpectrumMultipoles(k, pell_list, ell=ell, cov=cov, nbar=nbar,
                                  cosmo_fid=cosmo_fid | {'z': zeff}, kmin=kmin, kmax=kmax,
                                  wmat=win, kwin=kwin, ellwin=ellwin, kwinmin=kwinmin, kwinmax=kwinmax, nmocks_cov=n_mocks_cov,
                                  save_Mpc_units=use_Mpc)
    return obs

def get_obs_pk(tracer, zrange, region, mocktype, kmin=0.02, kmax=0.3,
               kwinmin=0.0, kwinmax=0.5, ell=[0, 2], ellwin=[0, 2, 4], nocov=False, mocktype_cov='holi-v3-altmtl', dk=0.005,
               cached=True, outdir='/global/cfs/cdirs/desicollab/users/alexpzfz/DR2_2pt3pt/data_for_mock_challenge/cutsky/',
               use_Mpc=True):
    if cached:
        return get_obs_pk_cached(tracer, zrange, region, mocktype, kmin=kmin, kmax=kmax,
                                 kwinmin=kwinmin, kwinmax=kwinmax, ell=ell, ellwin=ellwin,
                                 nocov=nocov, mocktype_cov=mocktype_cov, dk=dk, outdir=outdir, use_Mpc=use_Mpc)
    else:
        return get_obs_pk_raw(tracer, zrange, region, mocktype, kmin=kmin, kmax=kmax,
                              kwinmin=kwinmin, kwinmax=kwinmax, ell=ell, ellwin=ellwin,
                              nocov=nocov, mocktype_cov=mocktype_cov, dk=dk, use_Mpc=use_Mpc)

def get_obs_bk_cached(tracer, zrange, region, mocktype, kmin=0.02, kmax=0.3,
                     kwinmin=0.0, kwinmax=0.5, ell=[(0, 0, 0), (2, 0, 2)],
                     ellwin=[(0, 0, 0), (0, 2, 2), (1, 1, 0), (1, 1, 2), (2, 2, 0), (2, 2, 2)],
                     nocov=False, mocktype_cov='holi-v3-altmtl', dk=0.005, slice_winB=None,
                     outdir='/global/cfs/cdirs/desicollab/users/alexpzfz/DR2_2pt3pt/data_for_mock_challenge/cutsky/',
                     use_Mpc=True):
    label = tracer_labels[tracer][zrange]
    outdir = Path(outdir)
    fn_mean = get_fn(outdir, 'bk', mocktype, label, region, dkP=None, dkB=dk)
    # read header to get zeff and nbar
    k1, k2, b000, b202 = np.loadtxt(fn_mean, unpack=True)
    k1k2 = np.column_stack((k1, k2)) # shape (15129, 2)
    bell_dict = {(0, 0, 0): b000, (2, 0, 2): b202}
    bell_list = [bell_dict[ll] for ll in ell]
    cov = None
    if not nocov:
        raise NotImplementedError("Covariance for bispectrum multipoles not implemented yet. Set nocov=True to skip.")
 
    
    fn_window = get_fn(outdir, 'window_bk', mocktype, label, region, dkP=None, dkB=dk, slice_winB=slice_winB)
    fn_window_k = get_fn(outdir, 'window_bk_k', mocktype, label, region, dkP=None, dkB=dk, slice_winB=slice_winB)
    win = np.loadtxt(fn_window)
    kwin = np.loadtxt(fn_window_k)
    win = cut_window(win, x=k1k2, ell=[(0, 0, 0), (2, 0, 2)], xwin=kwin,
                     ellwin=[(0, 0, 0), (0, 2, 2), (1, 1, 0), (1, 1, 2), (2, 2, 0), (2, 2, 2)],
                     ell_select=ell, ellwin_select=ellwin)
    zeff = None

    obs = BispectrumSugiyamaMultipoles(k1k2, bell_list, ell=ell, cov=cov,
                                       cosmo_fid= cosmo_fid | {'z': zeff}, kmin=kmin, kmax=kmax,
                                       wmat=win, pairwin=kwin, ellwin=ellwin, kwinmin=kwinmin, kwinmax=kwinmax,
                                       save_Mpc_units=use_Mpc)
    return obs


def get_obs_bk_raw(tracer, zrange, region, mocktype, kmin=0.02, kmax=0.3,
                   kwinmin=0.0, kwinmax=0.5, ell=[(0, 0, 0), (2, 0, 2)],
                   ellwin=[(0, 0, 0), (0, 2, 2), (1, 1, 0), (1, 1, 2), (2, 2, 0), (2, 2, 2)],
                   nocov=False, mocktype_cov='holi-v3-altmtl', dk=0.005, slice_winB=None, use_Mpc=True):
    tracer_str = tracer if not 'ELG' in tracer else 'ELG_LOPnotqso'
    bk = get_mean_bk(tracer_str, zrange, region, mocktype, dk=dk)
    bk = bk.get(ells=ell)
    bell_list = [bk.get(ells=ll).value() for ll in ell]
    k1k2 = bk.get(ells=ell[0]).coords('k')
    cov = None
    if not nocov:
        cov_ = get_cov_bk(tracer_str, zrange, region, mocktype_cov, rang=(0, 1000), dk=dk)
        cov_ = cov_.at.observable.match(bk)
        cov = cov_.value()

    nmocks = cov_.attrs['n_mocks'] if cov is not None else None

    window_ = get_window_bk(tracer_str, zrange, region, mocktype)
    window_ = window_.at.theory.get(ells=ellwin)
    if slice_winB is not None:
        window_ = window_.at.theory.select(k=slice(0, None, slice_winB))
    window_ = window_.at.observable.match(bk)
    win = window_.value()
    kwin = []
    for ll in ellwin:
        kwin.append(window_.theory.get(ells=ll).coords('k'))
    zeff = None

    obs = BispectrumSugiyamaMultipoles(k1k2, bell_list, ell=ell, cov=cov,
                                       cosmo_fid=cosmo_fid | {'z': zeff}, kmin=kmin, kmax=kmax,
                                       wmat=win, pairwin=kwin, ellwin=ellwin, kwinmin=kwinmin, kwinmax=kwinmax,
                                       nmocks_cov=nmocks, save_Mpc_units=use_Mpc)
    return obs


def get_obs_pk_bk_cached(tracer, zrange, region, mocktype, kminP=0.01, kmaxP=0.3, ellP=[0, 2], ellB=[(0, 0, 0), (2, 0, 2)],
                            kwinminP=0.0, kwinmaxP=0.5, ellwinP=[0, 2, 4], kminB=0.01, kmaxB=0.2,
                            kwinminB=0.0, kwinmaxB=0.5, ellwinB=[(0, 0, 0), (0, 2, 2), (1, 1, 0), (1, 1, 2), (2, 2, 0), (2, 2, 2)],
                            mocktype_cov='holi-v3-altmtl', dkP=0.005, dkB=0.005, slice_winB_theory=2,
                            outdir='/global/cfs/cdirs/desicollab/users/alexpzfz/DR2_2pt3pt/data_for_mock_challenge/cutsky/',
                            use_Mpc=True):
    obs_pk = get_obs_pk_cached(tracer, zrange, region, mocktype, kmin=kminP, kmax=kmaxP,
                            kwinmin=kwinminP, kwinmax=kwinmaxP, ell=ellP,ellwin=ellwinP,
                            nocov=True, mocktype_cov=mocktype_cov, dk=dkP, outdir=outdir, use_Mpc=use_Mpc)
    obs_bk = get_obs_bk_cached(tracer, zrange, region, mocktype, kmin=kminB, kmax=kmaxB,
                            kwinmin=kwinminB, kwinmax=kwinmaxB, ell=ellB, ellwin=ellwinB,
                            nocov=True, mocktype_cov=mocktype_cov, dk=dkB,
                            slice_winB=slice_winB_theory, outdir=outdir, use_Mpc=use_Mpc)

    outdir = Path(outdir)
    fn_cov_ = get_fn(outdir, 'cov_pk_bk', mocktype_cov, tracer_labels[tracer][zrange], region, dkP=dkP, dkB=dkB, n_mocks_cov='*')
    fn_cov_base = fn_cov_.name
    fn_cov_list = list(outdir.glob(fn_cov_base))
    if len(fn_cov_list) == 0:
        raise FileNotFoundError(f"No covariance file found for {tracer} {zrange} with dkP={dkP} and dkB={dkB}. Searched for {fn_cov_base} in {outdir}")
    elif len(fn_cov_list) > 1:
        print(f"Warning: multiple covariance files found for {tracer} {zrange} with dkP={dkP} and dkB={dkB}")
        print("Using the file with the highest number of mocks.")
        fn_cov = max(fn_cov_list, key=lambda x: int(x.stem.split('nmocks')[1]))
    elif len(fn_cov_list) == 1:
        fn_cov = fn_cov_list[0]
    print(f"Using covariance file: {fn_cov}")
    n_mocks_cov = int(fn_cov.stem.split('nmocks')[1])
    cov = np.loadtxt(fn_cov)

    # need to load pk and bk unfortunately, for the k and k1k2 values to cut the covariance
    fn_pk = get_fn(outdir, 'pk', mocktype, tracer_labels[tracer][zrange], region, dkP=dkP, dkB=None)
    k_pk, p0, p2, p4 = np.loadtxt(fn_pk, unpack=True)
    fn_bk = get_fn(outdir, 'bk', mocktype, tracer_labels[tracer][zrange], region, dkP=None, dkB=dkB)
    k1, k2, b000, b202 = np.loadtxt(fn_bk, unpack=True)
    k1k2 = np.column_stack((k1, k2))
    x = [k_pk, k_pk, k_pk, k1k2, k1k2]
    ell = [0, 2, 4, (0, 0, 0), (2, 0, 2)]
    ellselect = ellP + ellB
    if not isinstance(kminP, list):
        kminP = [kminP] * len(ellP)
    if not isinstance(kminB, list):
        kminB = [kminB] * len(ellB)
    if not isinstance(kmaxP, list):
        kmaxP = [kmaxP] * len(ellP)
    if not isinstance(kmaxB, list):
        kmaxB = [kmaxB] * len(ellB)

    xmin = kminP + kminB
    xmax = kmaxP + kmaxB

    cov = cut_cov(cov, x=x, ell=ell, ell_select=ellselect, xmin=xmin, xmax=xmax)

    # The raw covariance loaded from disk is always in h-units; whether it gets
    # converted to Mpc units is now decided by JointObservable itself, based on
    # obs_pk/obs_bk's own Mpc_units setting (i.e. use_Mpc above).
    obs = JointObservable(obs_pk, obs_bk, cov=cov, nmocks_cov=n_mocks_cov, cov_input_Mpc_units=False)
    return obs

def get_obs_pk_bk_raw(tracer, zrange, region, mocktype, kminP=0.01, kmaxP=0.3, ellP=[0, 2], ellB=[(0, 0, 0), (2, 0, 2)],
                        kwinminP=0.0, kwinmaxP=0.5, ellwinP=[0, 2, 4], kminB=0.01, kmaxB=0.2,
                        kwinminB=0.0, kwinmaxB=0.5, ellwinB=[(0, 0, 0), (0, 2, 2), (1, 1, 0), (1, 1, 2), (2, 2, 0), (2, 2, 2)],
                        mocktype_cov='holi-v3-altmtl', dkP=0.005, dkB=0.005, slice_winB_theory=2,
                        use_Mpc=True):
    if not isinstance(kminP, list):
        kminP = [kminP] * len(ellP)
    if not isinstance(kmaxP, list):
        kmaxP = [kmaxP] * len(ellP)
    if not isinstance(kminB, list):
        kminB = [kminB] * len(ellB)
    if not isinstance(kmaxB, list):
        kmaxB = [kmaxB] * len(ellB)

    obs_pk = get_obs_pk_raw(tracer, zrange, region, mocktype, kmin=kminP, kmax=kmaxP,
                            kwinmin=kwinminP, kwinmax=kwinmaxP, ell=ellP, ellwin=ellwinP,
                            nocov=True, mocktype_cov=mocktype_cov, dk=dkP, use_Mpc=use_Mpc)
    obs_bk = get_obs_bk_raw(tracer, zrange, region, mocktype, kmin=kminB, kmax=kmaxB,
                            kwinmin=kwinminB, kwinmax=kwinmaxB, ell=ellB, ellwin=ellwinB,
                            nocov=True, dk=dkB, slice_winB=slice_winB_theory, use_Mpc=use_Mpc)

    tracer_str = tracer if not 'ELG' in tracer else 'ELG_LOPnotqso'
    # get_cov_pk_bk_preprocessed applies the ellP/ellB selection and the
    # kminP/kmaxP/kminB/kmaxB scale cuts per-mock before stacking, so the
    # resulting covariance is already aligned with obs_pk/obs_bk -- no need
    # to separately fetch the full (uncut) k grids and cut_cov() it after.
    cov_ = get_cov_pk_bk_preprocessed(tracer_str, zrange, region, mocktype_cov, rang=(0, 1000),
                                       dkP=dkP, dkB=dkB, kminP=kminP, kmaxP=kmaxP, kminB=kminB, kmaxB=kmaxB,
                                       ellP=ellP, ellB=ellB)
    cov = cov_.value()
    n_mocks_cov = cov_.attrs['n_mocks']

    obs = JointObservable(obs_pk, obs_bk, cov=cov, nmocks_cov=n_mocks_cov, cov_input_Mpc_units=False)
    return obs

def get_obs_pk_bk(tracer, zrange, region, mocktype, kminP=0.01, kmaxP=0.3, ellP=[0, 2], ellB=[(0, 0, 0), (2, 0, 2)],
                    kwinminP=0.0, kwinmaxP=0.5, ellwinP=[0, 2, 4], kminB=0.01, kmaxB=0.2,
                    kwinminB=0.0, kwinmaxB=0.5, ellwinB=[(0, 0, 0), (0, 2, 2), (1, 1, 0), (1, 1, 2), (2, 2, 0), (2, 2, 2)],
                    mocktype_cov='holi-v3-altmtl', dkP=0.005, dkB=0.005, slice_winB_theory=2,
                    cached=True, outdir='/global/cfs/cdirs/desicollab/users/alexpzfz/DR2_2pt3pt/data_for_mock_challenge/cutsky/',
                    use_Mpc=True):
    if cached:
        return get_obs_pk_bk_cached(tracer, zrange, region, mocktype, kminP=kminP, kmaxP=kmaxP,
                                    ellP=ellP, ellB=ellB,
                                    kwinminP=kwinminP, kwinmaxP=kwinmaxP, ellwinP=ellwinP,
                                    kminB=kminB, kmaxB=kmaxB,
                                    kwinminB=kwinminB, kwinmaxB=kwinmaxB, ellwinB=ellwinB,
                                    mocktype_cov=mocktype_cov, dkP=dkP, dkB=dkB,
                                    slice_winB_theory=slice_winB_theory,
                                    outdir=outdir, use_Mpc=use_Mpc)
    else:
        return get_obs_pk_bk_raw(tracer, zrange, region, mocktype, kminP=kminP, kmaxP=kmaxP,
                                 ellP=ellP, ellB=ellB,
                                 kwinminP=kwinminP, kwinmaxP=kwinmaxP, ellwinP=ellwinP,
                                 kminB=kminB, kmaxB=kmaxB,
                                 kwinminB=kwinminB, kwinmaxB=kwinmaxB, ellwinB=ellwinB,
                                 mocktype_cov=mocktype_cov, dkP=dkP, dkB=dkB,
                                 slice_winB_theory=slice_winB_theory, use_Mpc=use_Mpc)
    

if __name__ == "__main__":
    """
    If run as main script, this will read the data and covariance and save them as txt files.
    """
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--region', type=str, choices=['NGC', 'SGC', 'GCcomb'], default='GCcomb')
    parser.add_argument('--mocktype', type=str, default='abacus-hf-dr2-v2-altmtl')
    parser.add_argument('--mocktype_cov', type=str, default='holi-v3-altmtl')
    parser.add_argument('--dkP', type=float, default=0.005)
    parser.add_argument('--bispec', action='store_true')
    parser.add_argument('--dkB', type=float, default=0.005)
    parser.add_argument('--slice_winB', type=int, default=None)

    parser.add_argument('--outdir', type=str, default='/global/cfs/cdirs/desicollab/users/alexpzfz/DR2_2pt3pt/data_for_mock_challenge/cutsky/')
    parser.add_argument('--overwrite', action='store_true')

    args = parser.parse_args()

    if 'cubic' not in args.mocktype:
        for tracer in tracers:
            for zrange in zranges[tracer]:
                label = tracer_labels[tracer][zrange]
                print(f"Processing {label}...")
                outdir = Path(args.outdir)
                outdir.mkdir(parents=True, exist_ok=True)

                tracer_name = tracer
                tracer_cov_name = tracer
                mocktype = args.mocktype
                mocktype_cov = args.mocktype_cov
                project_use = project
                project_cov = project
        
                if 'ELG' in tracer:
                    tracer_name = 'ELG_LOPnotqso'
                    tracer_cov_name = 'ELG_LOPnotqso'
                if 'BGS' in tracer:
                    tracer_name = 'BGS_BRIGHT-02'
                    tracer_cov_name = 'BGS_BRIGHT-21.35'
                    project_use = project_bgs
                    if mocktype == 'abacus-hf-dr2-v2-altmtl':
                        print(f'No {mocktype} available for tracer {tracer}, using abacus-2ndgen-dr2-altmtl instead')
                        mocktype = 'abacus-2ndgen-dr2-altmtl'
                    if mocktype_cov == 'holi-v3-altmtl':
                        print(f'No {mocktype_cov} available for tracer {tracer}, using holi-bgs-altmtl instead')
                        mocktype_cov = 'holi-bgs-altmtl'

                fn_mean = get_fn(outdir, 'pk', mocktype, label, args.region, args.dkP, args.dkB)
                fn_window = get_fn(outdir, 'window_pk', mocktype, label, args.region, args.dkP, args.dkB)
                fn_window_k = get_fn(outdir, 'window_pk_k', mocktype, label, args.region, args.dkP, args.dkB)

                # get avail mocks for covariance
                n_mocks_cov = count_available_mocks(tracer_cov_name, zrange, args.region, mocktype_cov, project=project_cov)
                fn_cov = get_fn(outdir, 'cov_pk', mocktype_cov, label, args.region, args.dkP, args.dkB, n_mocks_cov=n_mocks_cov)

                mean_pk = None
                cov_pk = None
                window_pk = None
                mean_bk = None
                cov_pk_bk = None
                window_bk = None
            
                if not fn_mean.exists() or args.overwrite:
                    mean_pk = get_mean_pk(tracer_name, zrange, args.region, mocktype, dk=args.dkP, project=project_use)
                    k = mean_pk.get(ells=0).coords('k')
                    p0 = mean_pk.get(ells=0).value()
                    p2 = mean_pk.get(ells=2).value()
                    p4 = mean_pk.get(ells=4).value()
                    nbar = 1 / mean_pk.get(ells=0).values('shotnoise')[0]

                    #add nbar to the header of the txt file
                    # need to load window to get zeff
                    window_pk = get_window_pk(tracer_name, zrange, args.region, mocktype, project=project_use)
                    zeff = window_pk.observable.get(ells=0).attrs['zeff']

                    header = f'zeff = {zeff:.10f}\n'
                    header += f'nbar = {nbar:.10e}\n'
                    header += 'k\tP0\tP2\tP4\n'
                    np.savetxt(fn_mean, np.vstack([k, p0, p2, p4]).T, header=header)
                else:
                    print(f"{fn_mean} already exists, skipping...")


                if not fn_cov.exists() or args.overwrite:
                    # verify that the number of mocks in the covariance matches the available mocks
                    cov_pk = get_cov_pk(tracer_cov_name, zrange, args.region, mocktype_cov, dk=args.dkP, project=project_cov)
                    assert cov_pk.attrs['n_mocks'] == n_mocks_cov, f"Number of mocks in covariance ({cov_pk.attrs['n_mocks']}) does not match available mocks ({n_mocks_cov})"
                    np.savetxt(fn_cov, cov_pk.value())
                else:
                    print(f"{fn_cov} already exists, skipping...")
                
                if not fn_window.exists() or args.overwrite:
                    if window_pk is None:
                        window_pk = get_window_pk(tracer_name, zrange, args.region, mocktype, project=project_use)
                    if mean_pk is None:
                        mean_pk = get_mean_pk(tracer_name, zrange, args.region, mocktype, dk=args.dkP, project=project_use)
                    window_pk = window_pk.at.observable.match(mean_pk)
                    k_window = window_pk.theory.get(ells=0).coords('k')

                    np.savetxt(fn_window, window_pk.value())
                    np.savetxt(fn_window_k, k_window)
                else:
                    print(f"{fn_window} already exists, skipping...")
                

                if args.bispec:
                    fn_mean_bk = get_fn(outdir, 'bk', mocktype, label, args.region, args.dkP, args.dkB)
                    fn_cov_pk_bk = get_fn(outdir, 'cov_pk_bk', mocktype_cov, label, args.region, args.dkP, args.dkB, n_mocks_cov=n_mocks_cov)

                    if args.slice_winB is None: 
                        fn_window_bk = get_fn(outdir, 'window_bk', mocktype, label, args.region, args.dkP, args.dkB)
                        fn_window_bk_k = get_fn(outdir, 'window_bk_k', mocktype, label, args.region, args.dkP, args.dkB)
                    else:
                        fn_window_bk = get_fn(outdir, 'window_bk', mocktype, label, args.region, args.dkP, args.dkB, slice_winB=args.slice_winB)
                        fn_window_bk_k = get_fn(outdir, 'window_bk_k', mocktype, label, args.region, args.dkP, args.dkB, slice_winB=args.slice_winB)

                    if not fn_mean_bk.exists() or args.overwrite:
                        mean_bk = get_mean_bk(tracer_name, zrange, args.region, mocktype, dk=args.dkB, project=project_use)
                        k1k2 = mean_bk.get(ells=(0, 0, 0)).coords('k') # shape (n, 2)
                        k1 = k1k2[:, 0]
                        k2 = k1k2[:, 1]
                        b000 = mean_bk.get(ells=(0, 0, 0)).value()
                        b202 = mean_bk.get(ells=(2, 0, 2)).value()

                        header = 'k1\tk2\tB000\tB202\n'
                        np.savetxt(fn_mean_bk, np.vstack([k1, k2, b000, b202]).T, header=header)
                    else:
                        print(f"{fn_mean_bk} already exists, skipping...")

                    if not fn_cov_pk_bk.exists() or args.overwrite:
                        cov_pk_bk = get_cov_pk_bk(tracer_cov_name, zrange, args.region, mocktype_cov, dkP=args.dkP, dkB=args.dkB, project=project_cov)
                        assert cov_pk_bk.attrs['n_mocks'] == n_mocks_cov, f"Number of mocks in covariance ({cov_pk_bk.attrs['n_mocks']}) does not match available mocks ({n_mocks_cov})"
                        np.savetxt(fn_cov_pk_bk, cov_pk_bk.value())
                    else:
                        print(f"{fn_cov_pk_bk} already exists, skipping...")
                    
                    if not fn_window_bk.exists() or args.overwrite:
                        if mean_bk is None:
                            mean_bk = get_mean_bk(tracer_name, zrange, args.region, mocktype, dk=args.dkB, project=project_use)
                        window_bk = get_window_bk(tracer_name, zrange, args.region, mocktype, project=project_use)
                        window_bk = window_bk.at.observable.match(mean_bk)

                        if args.slice_winB is not None:
                            window_bk = window_bk.at.theory.select(k=slice(0, None, args.slice_winB))
                        k1k2_window = window_bk.theory.get(ells=(0, 0, 0)).coords('k') # shape (n, 2)

                        np.savetxt(fn_window_bk, window_bk.value())
                        np.savetxt(fn_window_bk_k, k1k2_window)
                    else:
                        print(f"{fn_window_bk} already exists, skipping...")

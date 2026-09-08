import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
#ROOT_DIR = Path(__file__).resolve().parents[2]
# if str(ROOT_DIR) not in sys.path:
#     sys.path.insert(0, str(ROOT_DIR))
sys.path.append('../../')
from observables import PowerSpectrumMultipoles, BispectrumSugiyamaMultipoles, JointObservable
from utils import cut_cov, cut_window

 
#cosmo_fid = {'wb': 0.02237, 'wc': 0.1200, 'h': 0.6736, 'As': 2.0830, 'ns': 0.9649, 'Mnu': 0.06}
cosmologies = {'c000': {'wb': 0.02237, 'wc': 0.1200, 'h': 0.6736, 'As': 2.083, 'ns': 0.9649, 'Mnu': 0.06, 'w0': -1.0, 'wa': 0.0},
              'c001': {'wb': 0.02242, 'wc': 0.1134, 'h': 0.7030, 'As': 2.037, 'ns': 0.9638, 'Mnu': 0.06, 'w0': -1.0, 'wa': 0.0},
              'c002': {'wb': 0.02237, 'wc': 0.1200, 'h': 0.6278, 'As': 2.314, 'ns': 0.9649, 'Mnu': 0.06, 'w0': -0.7, 'wa': -0.5},
              'c003': {'wb': 0.02260, 'wc': 0.1291, 'h': 0.7160, 'As': 2.2438, 'ns': 0.9876, 'Mnu': 0.06, 'w0': -1.0, 'wa': 0.0},
              'c004': {'wb': 0.02237, 'wc': 0.1200, 'h': 0.6736, 'As': 1.7949, 'ns': 0.9649, 'Mnu': 0.06, 'w0': -1.0, 'wa': 0.0},}

outdir = Path('/global/cfs/cdirs/desicollab/users/alexpzfz/DR2_2pt3pt/data_for_mock_challenge/cubic')

names_dict = {'LRG': 'LRG2', 'ELG': 'ELG1', 'QSO': 'QSO'}
hod_options = ['fiducial', 'alternative']
labels_dict = {(tracer, hod_opt): f'{names_dict[tracer]}_{hod_opt}_hod' for tracer in names_dict for hod_opt in hod_options}
zsnap_dict = {'LRG': 0.725, 'ELG': 0.95, 'QSO': 1.4}
zsnap_ezmock_dict = {'LRG': 0.8, 'ELG': 0.95, 'QSO': 1.4}

def get_fn_box(outdir, kind, mock_type, label, zsnap, nmocks_cov=None):
    if kind == 'pk':
        return Path.joinpath(outdir, f'mean_pk_{mock_type}_{label}_zsnap{zsnap}.txt')
    elif kind == 'cov_pk':
        return Path.joinpath(outdir, f'cov_pk_{mock_type}_{label}_zsnap{zsnap}_nmocks{nmocks_cov}.txt')
    elif kind == 'bk':
        return Path.joinpath(outdir, f'mean_bk_sugiyama_diag_{mock_type}_{label}_zsnap{zsnap}.txt')
    elif kind == 'cov_pk_bk':
        return Path.joinpath(outdir, f'cov_pk_bk_sugiyama_diag_{mock_type}_{label}_zsnap{zsnap}_nmocks{nmocks_cov}.txt')

def get_fn_cv(outdir, kind, tracer, zsnap, cosmo):
    if kind == 'pk':
        return Path.joinpath(outdir, f'Pk_{tracer}_zsnap_{zsnap}_{cosmo}.txt')
    elif kind == 'bk':
        return Path.joinpath(outdir, f'Bk_{tracer}_zsnap_{zsnap}_{cosmo}.txt')

def get_obs_pk(tracer, hod_opt='fiducial', kmin=0.02, kmax=0.3, ell=[0, 2], nocov=False, cosmo='c000',
               outdir='/global/cfs/cdirs/desicollab/users/alexpzfz/DR2_2pt3pt/data_for_mock_challenge/cubic/',
               gausscov=False):
    outdir = Path(outdir)
    label = labels_dict[(tracer, hod_opt)]
    zsnap = zsnap_dict[tracer]
    if cosmo == 'c000':
        fn_mean = get_fn_box(outdir, 'pk', 'abacus_hf', label, zsnap)
    else:
        assert hod_opt == 'fiducial', "Only fiducial HOD is available for cosmology variations"
        fn_mean = get_fn_cv(outdir / 'cosmology_variations', 'pk', tracer, zsnap, cosmo)
    # read header to get zeff and nbar
    with open(fn_mean, 'r') as f:
        header = f.readlines()[:2]
    # nbar = float(header[0].split('=')[1])
    # look for the pattern nbar= or nbar = in the header
    nbar = None
    for line in header:
        if 'nbar' in line:
            nbar = float(line.split('=')[1])
            break
        elif 'nmean' in line:
            nbar = float(line.split('=')[1])
            break
    zeff = zsnap
    k, p0, p2, p4 = np.loadtxt(fn_mean, unpack=True)
    pell_dict = {0: p0, 2: p2, 4: p4} 
    pell_list = [pell_dict[ll] for ll in ell]
    cov = None
    n_mocks_cov = None
    zsnap_ezmock = zsnap_ezmock_dict[tracer]
    if not nocov:
        if not gausscov:
            fn_cov_ = get_fn_box(outdir, 'cov_pk', 'ezmocks', label, zsnap_ezmock, nmocks_cov='*')
            # as we don't know the number of mocks in advance, we need to find the file that matches the pattern
            fn_cov_base = fn_cov_.name
            fn_cov_list = list(outdir.glob(fn_cov_base))
            if len(fn_cov_list) == 0:
                raise FileNotFoundError(f"No covariance file found for pattern {fn_cov_}")
            elif len(fn_cov_list) > 1:
                raise ValueError(f"Multiple covariance files found for pattern {fn_cov_}: {fn_cov_list}")
            fn_cov = fn_cov_list[0]
            # read number of mocks from the filename
            n_mocks_cov = int(fn_cov.stem.split('nmocks')[1])
            cov = np.loadtxt(fn_cov)
            cov = cut_cov(cov, k, ell=[0, 2, 4], ell_select=ell)
        else:
            fn_cov = outdir / f'pk_cov_{tracer}_gauss.txt'
            cov = np.loadtxt(fn_cov)
            cov = cut_cov(cov, k, ell=[0, 2, 4], ell_select=ell)
            n_mocks_cov = None
    
    cosmo_fid = cosmologies[cosmo].copy() 
    obs = PowerSpectrumMultipoles(k, pell_list, ell=ell, cov=cov, nbar=nbar,
                                  cosmo_fid=cosmo_fid | {'z': zeff}, kmin=kmin, kmax=kmax,
                                  nmocks_cov=n_mocks_cov)
    return obs

def get_obs_bk(tracer, hod_opt='fiducial', kmin=0.02, kmax=0.3,
               ell=[(0, 0, 0), (2, 0, 2)], nocov=False, cosmo='c000',
               outdir='/global/cfs/cdirs/desicollab/users/alexpzfz/DR2_2pt3pt/data_for_mock_challenge/cubic/'):
    outdir = Path(outdir)
    label = labels_dict[(tracer, hod_opt)]
    if cosmo == 'c000':
        fn_mean = get_fn_box(outdir, 'bk', 'abacus_hf', label, zsnap_dict[tracer])
    else:
        assert hod_opt == 'fiducial', "Only fiducial HOD is available for cosmology variations"
        fn_mean = get_fn_cv(outdir / 'cosmology_variations', 'bk', tracer, zsnap_dict[tracer], cosmo)
    k1, k2, b000, b202 = np.loadtxt(fn_mean, unpack=True)
    k1k2 = np.column_stack((k1, k2)) 
    if cosmo != 'c000':
        mask = (k1 <= 0.2) & (k2 <= 0.2)
        k1k2 = k1k2[mask]
        b000 = b000[mask]
        b202 = b202[mask]
    bell_dict = {(0, 0, 0): b000, (2, 0, 2): b202}
    bell_list = [bell_dict[ll] for ll in ell]
    cov = None
    if not nocov:
        raise NotImplementedError("Covariance for bispectrum multipoles not implemented yet. Set nocov=True to skip.")
  
    zeff = zsnap_dict[tracer]

    cosmo_fid = cosmologies[cosmo].copy()
    obs = BispectrumSugiyamaMultipoles(k1k2, bell_list, ell=ell, cov=cov,
                                       cosmo_fid= cosmo_fid | {'z': zeff}, kmin=kmin, kmax=kmax)
    return obs


def get_obs_pk_bk(tracer, hod_opt='fiducial', kminP=0.01, kmaxP=0.3, ellP=[0, 2],
                  kminB=0.01, kmaxB=0.2, ellB=[(0, 0, 0), (2, 0, 2)], cosmo='c000',
                  outdir='/global/cfs/cdirs/desicollab/users/alexpzfz/DR2_2pt3pt/data_for_mock_challenge/cubic/'):
    obs_pk = get_obs_pk(tracer, hod_opt, kmin=kminP, kmax=kmaxP, ell=ellP, nocov=True, cosmo=cosmo, outdir=outdir)
    obs_bk = get_obs_bk(tracer, hod_opt, kmin=kminB, kmax=kmaxB, ell=ellB, nocov=True, cosmo=cosmo, outdir=outdir)
    outdir = Path(outdir)
    fn_cov_ = get_fn_box(outdir, 'cov_pk_bk', 'ezmocks', labels_dict[(tracer, hod_opt)], zsnap_ezmock_dict[tracer], nmocks_cov='*')
    fn_cov_base = fn_cov_.name
    fn_cov_list = list(outdir.glob(fn_cov_base))
    if len(fn_cov_list) == 0:
        raise FileNotFoundError(f"No covariance file found for pattern {fn_cov_}")
    elif len(fn_cov_list) > 1:
        print(f"Multiple covariance files found for pattern {fn_cov_}: {fn_cov_list}")
    fn_cov = fn_cov_list[0]
    n_mocks_cov = int(fn_cov.stem.split('nmocks')[1])
    cov = np.loadtxt(fn_cov)
    if cosmo == 'c000':
        fn_pk = get_fn_box(outdir, 'pk', 'abacus_hf', labels_dict[(tracer, hod_opt)], zsnap_dict[tracer])
        fn_bk = get_fn_box(outdir, 'bk', 'abacus_hf', labels_dict[(tracer, hod_opt)], zsnap_dict[tracer])
    else:
        assert hod_opt == 'fiducial', "Only fiducial HOD is available for cosmology variations"
        fn_pk = get_fn_cv(outdir / 'cosmology_variations', 'pk', tracer, zsnap_dict[tracer], cosmo)
        fn_bk = get_fn_cv(outdir / 'cosmology_variations', 'bk', tracer, zsnap_dict[tracer], cosmo)
    k_pk, _, _, _ = np.loadtxt(fn_pk, unpack=True)
    k1, k2, _, _ = np.loadtxt(fn_bk, unpack=True)
    if cosmo != 'c000':
        k1 = k1[k1 <= 0.2]
        k2 = k2[k2 <= 0.2]
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
    obs = JointObservable(obs_pk, obs_bk, cov=cov, nmocks_cov=n_mocks_cov)
    return obs

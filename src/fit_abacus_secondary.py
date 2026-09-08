import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import env  # noqa: F401
import numpy as np
from likelihood import Likelihood
from observables import PowerSpectrumMultipoles
from params import Params
from samplers import NautilusSampler
#from comet import comet
from theory import COMET



cosmo_fid = {'wb': 0.02237, 'wc': 0.1200, 'h': 0.6736, 'As': 2.0830, 'ns': 0.9649, 'Mnu': 0.06, 'z': 0.725}
cosmo_true = {'c000': {'wb': 0.02237, 'wc': 0.1200, 'h': 0.6736, 'As': 2.083, 'ns': 0.9649, 'Mnu': 0.06},
              'c001': {'wb': 0.02242, 'wc': 0.1134, 'h': 0.7030, 'As': 2.037, 'ns': 0.9638, 'Mnu': 0.06},
              'c002': {'wb': 0.02237, 'wc': 0.1200, 'h': 0.6278, 'As': 2.314, 'ns': 0.9649, 'Mnu': 0.06, 'w0': -0.7, 'wa': -0.5},
              'c004': {'wb': 0.02237, 'wc': 0.1200, 'h': 0.6736, 'As': 1.7949, 'ns': 0.9649, 'Mnu': 0.06}}
nbar_values = {'c000': 6.99e-4, 'c001': 6.74e-4, 'c002': 8.25e-4, 'c003': 6.74e-4, 'c004': 6.99e-4}

def get_obs(cosmo):
    data_fn = f'/global/homes/a/alexpzfz/pycompass/tmp/mock_challenge/data_config_files/DESI_Abacus_CubicBox_LRG_z0.725_{cosmo}_mean.dat' 
    cov_fn = f'/global/homes/a/alexpzfz/pycompass/tmp/mock_challenge/data_config_files/DESI_Abacus_CubicBox_LRG_z0.725_{cosmo}_covmat.dat'  

    k, p0, err0, p2, err2 = np.loadtxt(data_fn, unpack=True)
    cov = np.loadtxt(cov_fn)

    khmin, khmax = 0.01, 0.3
    nbar = nbar_values[cosmo]
    obs = PowerSpectrumMultipoles(k, [p0, p2], ell=[0, 2], cov=cov, nbar=nbar, 
                                  cosmo_fid=cosmo_fid, Mpc_units=False,
                                  kmin=khmin, kmax=khmax, nmocks_cov=2000)

    return obs

def get_emu_pars(cosmo='c000', bias_basis='DesJeoSch', counterterm_basis='DESI', reparam_option='full'):
    emu = COMET(model='VDG_infty', use_Mpc=True, bias_basis=bias_basis, counterterm_basis=counterterm_basis)
    pars = Params(emu)
    pars.update_parameter('wb', 0.02237, prior=(0.02237, 0.00055), prior_type='gaussian', fixed=False)
    pars.update_parameter('wc', 0.1200, prior=(0.085, 0.155), prior_type='uniform', fixed=False)
    #pars.update_parameter('As', 2.0830, prior=(1.055, 3.17), prior_type='uniform', fixed=False)
    pars.update_parameter('h', 0.6736, prior=(0.5, 1.0), prior_type='uniform', fixed=False)
    pars.update_parameter('ns', 0.9649, prior=(0.9649, 0.042/5), prior_type='gaussian', fixed=False)
    if cosmo == 'c002':
        pars.update_parameter('w0', -1., prior=(-3., 1.), prior_type='uniform', fixed=False)
        pars.update_parameter('wa', 0., prior=(-3., 2.), prior_type='uniform', fixed=False)
    pars.add_sampled_param("log10As", 3.04, prior=(2.31, 3.5), latex=r"\ln(10^{10} A_s)")
    pars.set_derived_param("As", lambda d: np.exp(d["log10As"]) / 1e10 * 1e9, exported=True)     
    
    pars.set_and_fix_param('Mnu', 0.06)
    
    b1ref = 2.047
    sigmaR_ref = 0.559 # need to calculate sigma_12_ref
    fsat = 0.13
    sigma1_eff = 150/70 * 10**(1/3) * (1 + 0.8)**(1/2)
    bK2ref = -2./7.*(b1ref - 1.)
    btdref = 23./42.*(b1ref - 1.)
    stoch_scale = 500 / (cosmo_fid['h']**2)

    stochastic_mode = 'ap'
    counterterms_mode = 'ap+sigma_12'
    if reparam_option == 'full':
        bias_mode = 'ap+sigma_12'
    elif reparam_option == 'hybrid':
        bias_mode = 'sigma_12'



    pars.use_reparametrization(bias_mode=bias_mode, counterterms_mode=counterterms_mode,
                                stochastic_mode=stochastic_mode, sigmaR_ref=sigmaR_ref)

    pars.update_parameter('b1_r', 2., prior=(0.1, 8), prior_type='uniform', fixed=False)
    pars.update_parameter('b2t_r', 0., prior=(0, 20), prior_type='gaussian', fixed=False)
    pars.update_parameter('bK2_r', bK2ref, prior=(bK2ref, 20), prior_type='gaussian', fixed=False)
    pars.update_parameter('btd_r', btdref, prior=(btdref, 80), prior_type='gaussian', fixed=False)
    pars.update_parameter('avir', 5., prior=(0, 30), prior_type='uniform', fixed=False)
    pars.update_parameter('a0_r', 0., prior=(0, stoch_scale), prior_type='gaussian', fixed=False)
    pars.update_parameter('a2_r', 0., prior=(0, stoch_scale), prior_type='gaussian', fixed=False)
    pars.update_parameter('NP0_r', 0., prior=(0, 2.), prior_type='gaussian', fixed=False) 
    pars.update_parameter('NP20_r', 0., prior=(0, 2.), prior_type='gaussian', fixed=False)
    pars.update_parameter('NP22_r', 0., prior=(0, 5. * fsat * sigma1_eff**2), prior_type='gaussian', fixed=False)
    pars.set_and_fix_param('a4_r', 0.)
    pars.set_and_fix_param('a4', 0.)
    # pars.set_reference_sigmaR(sigmaR_ref)

    return emu, pars


if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--cosmo', type=str, default='c000', choices=['c000', 'c001', 'c002', 'c003', 'c004'])
    parser.add_argument('--reparam_option', type=str, default='full', choices=['full', 'hybrid'])
    args = parser.parse_args()
    cosmo = args.cosmo

    emu, pars = get_emu_pars(cosmo=cosmo, reparam_option=args.reparam_option)
    obs = get_obs(cosmo)

    am_params = ['btd_r', 'a0_r', 'a2_r', 'NP0_r', 'NP20_r', 'NP22_r']
    like = Likelihood(obs, emu, pars, am_params=am_params)
    fn = f'{env.CHAINS_DIR}/Abacus_LRG_z0.725_{cosmo}_maxfree_p0p2_kmax0.3_reparam-{args.reparam_option}_test_new'
    fn_snap = fn + '_nautilus.hdf5'
    sampler = NautilusSampler(pars, like, filepath=fn_snap, pool=8)
    sampler.sample(verbose=True)
    sampler.save(fn, metadata={**vars(args), 'pool_size': 8})
    # clean up the snapshot file
    os.remove(fn_snap)


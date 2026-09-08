import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import env  # noqa: F401
import numpy as np
from params import Params
from theory import COMET

cosmo_fid = {'wb': 0.02237, 'wc': 0.1200, 'h': 0.6736, 'As': 2.0830, 'ns': 0.9649, 'Mnu': 0.06, 'z': 0.725}
cosmo_abacus = {'c000': {'wb': 0.02237, 'wc': 0.1200, 'h': 0.6736, 'As': 2.083, 'ns': 0.9649, 'Mnu': 0.06},
                'c001': {'wb': 0.02242, 'wc': 0.1134, 'h': 0.7030, 'As': 2.037, 'ns': 0.9638, 'Mnu': 0.06},
                'c002': {'wb': 0.02237, 'wc': 0.1200, 'h': 0.6278, 'As': 2.314, 'ns': 0.9649, 'Mnu': 0.06, 'w0': -0.7, 'wa': -0.5},
                'c004': {'wb': 0.02237, 'wc': 0.1200, 'h': 0.6736, 'As': 1.7949, 'ns': 0.9649, 'Mnu': 0.06}}

def get_pars(bias_basis='DESI', counterterm_basis='DESIct', reparam_option=None,
             de_model='lambda', freedom='max', b1_ref=2.109, sigmaR_ref=0.539, sigma1_eff=150/70 * 10**(1/3) * (1 + 0.8)**(1/2), fsat=0.13,
             bispec=False, free_Mnu=False, z_array=None, ns_times_Planck=10):
    
    if isinstance(reparam_option, str) and reparam_option.lower() == 'none':
        reparam_option = None

    freedom_options = ['min', 'max', 'interm']
    if freedom not in freedom_options:
        raise ValueError(f"Invalid freedom option: {freedom}. Must be one of {freedom_options}.")

    if z_array is None:
        z_array = [cosmo_fid['z']]
    z_array = np.atleast_1d(np.array(z_array, dtype=float))
    nz = len(z_array)

    def _to_nz_array(value, name):
        arr = np.atleast_1d(np.array(value, dtype=float))
        if arr.size == 1:
            return np.full(nz, arr.item(), dtype=float)
        if arr.size != nz:
            raise ValueError(f"{name} must be scalar or length {nz}; got length {arr.size}")
        return arr

    b1_ref = _to_nz_array(b1_ref, 'b1_ref')
    sigmaR_ref = _to_nz_array(sigmaR_ref, 'sigmaR_ref')
    sigma1_eff = _to_nz_array(sigma1_eff, 'sigma1_eff')
    fsat = _to_nz_array(fsat, 'fsat')

    emu = COMET(model='VDG_infty', use_Mpc=True, bias_basis=bias_basis, counterterm_basis=counterterm_basis)
    coev_params = ['bK2', 'btd'] if freedom == 'min' else []
    pars = Params(emu, coev_params=coev_params, z_array=z_array)
    #pars = Params(emu, coev_params=coev_params)
    pars.update_parameter('wb', 0.02237, prior=(0.02237, 0.00055), prior_type='gaussian', fixed=False)
    if freedom == 'interm':
        pars.update_parameter('wb', 0.02237, prior=(0.02237, 0.00037), prior_type='gaussian', fixed=False)
    #pars.update_parameter('wc', 0.1200, prior=(0.085, 0.155), prior_type='uniform', fixed=False)
    pars.update_parameter('wc', 0.1200, prior=(0.08, 0.16), prior_type='uniform', fixed=False)
    #pars.update_parameter('As', 2.0830, prior=(1.055, 3.17), prior_type='uniform', fixed=False)
    pars.update_parameter('h', 0.6736, prior=(0.5, 1.0), prior_type='uniform', fixed=False)
    pars.update_parameter('ns', 0.9649, prior=(0.9649, ns_times_Planck * 0.0042), prior_type='gaussian', fixed=False)
    if freedom == 'interm':
        pars.set_and_fix_param('ns', 0.9649)
    if de_model == 'w0wa':
        pars.update_parameter('w0', -1., prior=(-3., 1.), prior_type='uniform', fixed=False)
        pars.update_parameter('wa', 0., prior=(-3., 2.), prior_type='uniform', fixed=False)
    elif de_model == 'w0':
        pars.update_parameter('w0', -1., prior=(-3., 1.), prior_type='uniform', fixed=False)
    elif de_model != 'lambda':
        raise ValueError(f"Invalid de_model: {de_model}. Must be one of 'lambda', 'w0', or 'w0wa'.")
    # pars.add_sampled_param("log10As", 3.04, prior=(2.31, 3.5), latex=r"\ln(10^{10} A_s)")
    pars.add_sampled_param("log10As", 3.04, prior=(2., 4.), latex=r"\ln(10^{10} A_s)")
    pars.set_derived_param("As", lambda d: np.exp(d["log10As"]) / 1e10 * 1e9, exported=True)     
    
    if free_Mnu:
        pars.update_parameter('Mnu', 0.06, prior=(0.0, 5.0), prior_type='uniform', fixed=False)
    else:
        pars.set_and_fix_param('Mnu', 0.06)

    bK2ref = -2./7.*(b1_ref - 1.)
    btdref = 23./42.*(b1_ref - 1.)

    h = cosmo_fid['h']
    stoch_scales = {'max': 500., 'min': 50., 'interm': 50.}
    stoch_scale = stoch_scales[freedom] / (h**2)
    

    stochastic_mode = 'ap'
    counterterms_mode = 'ap+sigma_12'
    if reparam_option == 'full':
        bias_mode = 'ap+sigma_12'
    elif reparam_option == 'hybrid':
        bias_mode = 'sigma_12'

    
    gauss_scales = {
        'max': {'NP0_r': np.full(nz, 20., dtype=float), 'NP20_r': 50. * fsat * sigma1_eff**2, 'NP22_r': 50. * fsat * sigma1_eff**2},
        'min': {'NP0_r': np.full(nz, 2., dtype=float), 'NP20_r': 5. * fsat * sigma1_eff**2, 'NP22_r': 5. * fsat * sigma1_eff**2},
        'interm': {'NP0_r': np.full(nz, 2., dtype=float), 'NP20_r': 5. * fsat * sigma1_eff**2, 'NP22_r': 5. * fsat * sigma1_eff**2},
    }
    
    gs = gauss_scales[freedom]

    if reparam_option is not None:
        pars.use_reparametrization(bias_mode=bias_mode, counterterms_mode=counterterms_mode,
                                    stochastic_mode=stochastic_mode, sigmaR_ref=sigmaR_ref)

    sub_rep = '_r' if reparam_option is not None else ''

    for iz in range(nz):
        sub_idz = f"_{iz}" if nz > 1 else ""
        subscript = sub_rep + sub_idz
        pars.update_parameter(f'b1{subscript}', 2., prior=(0.1, 8), prior_type='uniform', fixed=False)
        pars.update_parameter(f'b2d{subscript}', 0., prior=(0, 20), prior_type='gaussian', fixed=False)
        if freedom == 'interm':
            pars.update_parameter(f'b1{subscript}', 2., prior=(0.1, 4), prior_type='uniform', fixed=False)
            pars.update_parameter(f'b2d{subscript}', 0., prior=(0, 5), prior_type='gaussian', fixed=False)
        if freedom == 'max':
            pars.update_parameter(f'bk2{subscript}', bK2ref[iz], prior=(bK2ref[iz], 20), prior_type='gaussian', fixed=False)
            pars.update_parameter(f'btd{subscript}', btdref[iz], prior=(btdref[iz], 80), prior_type='gaussian', fixed=False)
        elif freedom == 'interm':
            pars.update_parameter(f'bk2{subscript}', bK2ref[iz], prior=(bK2ref[iz], 5), prior_type='gaussian', fixed=False)
            pars.update_parameter(f'btd{subscript}', btdref[iz], prior=(btdref[iz], 1.), prior_type='gaussian', fixed=False)

        pars.update_parameter(f'avir{sub_idz}', 5., prior=(0, 30.), prior_type='uniform', fixed=False)
        pars.update_parameter(f'NP0{subscript}', 0., prior=(0, gs['NP0_r'][iz]), prior_type='gaussian', fixed=False)
        pars.update_parameter(f'NP20{subscript}', 0., prior=(0, gs['NP20_r'][iz]/h**2), prior_type='gaussian', fixed=False)
        pars.update_parameter(f'NP22{subscript}', 0., prior=(0, gs['NP22_r'][iz]/h**2), prior_type='gaussian', fixed=False)

        if counterterm_basis == 'DESIct':
            pars.update_parameter(f'a0{subscript}', 0., prior=(0, stoch_scale), prior_type='gaussian', fixed=False)
            pars.update_parameter(f'a2{subscript}', 0., prior=(0, stoch_scale), prior_type='gaussian', fixed=False)
            pars.update_parameter(f'a4{subscript}', 0., prior=(0, stoch_scale), prior_type='gaussian', fixed=False)
            #pars.set_and_fix_param(f'a4{subscript}', 0.)
            #if reparam_option is not None:
            #    pars.set_and_fix_param(f'a4{sub_idz}', 0.)
        elif counterterm_basis == 'Comet':
            pars.update_parameter(f'c0{subscript}', 0., prior=(0, stoch_scale), prior_type='gaussian', fixed=False)
            pars.update_parameter(f'c2{subscript}', 0., prior=(0, stoch_scale), prior_type='gaussian', fixed=False)
            pars.update_parameter(f'c4{subscript}', 0., prior=(0, stoch_scale), prior_type='gaussian', fixed=False)

        if bispec:
            pars.update_parameter(f'NB0{subscript}', 0., prior=(0, 1.), prior_type='gaussian', fixed=False)
            pars.update_parameter(f'MB0{subscript}', 0., prior=(0, 3.), prior_type='gaussian', fixed=False)
            pars.update_parameter(f'avirB{sub_idz}', 0., prior=(0, 30.), prior_type='uniform', fixed=False)
        else:
            pars.set_and_fix_param(f'NB0{subscript}', 0.)
            pars.set_and_fix_param(f'MB0{subscript}', 0.)
            pars.set_and_fix_param(f'avirB{sub_idz}', 0.)

            if reparam_option is not None:
                pars.set_and_fix_param(f'NB0{sub_idz}', 0.)
                pars.set_and_fix_param(f'MB0{sub_idz}', 0.)

    return pars
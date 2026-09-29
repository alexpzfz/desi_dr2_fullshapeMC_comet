import sys
from pathlib import Path
from functools import partial
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

def _As_from_log10As(d):
    """Module-level (picklable) derived-parameter function: As in units of 1e-9."""
    return np.exp(d["log10As"]) / 1e10 * 1e9

def _w0_from_w0pwa_w0mwa(d):
    """Derived-parameter function that computes w0 from w0pwa and w0mwa."""
    w0pwa = d["w0pwa"]
    w0mwa = d["w0mwa"]
    w0 = 0.5 * (w0pwa + w0mwa)
    return w0

def _wa_from_w0pwa_w0mwa(d):
    """Derived-parameter function that computes wa from w0pwa and w0mwa."""
    w0pwa = d["w0pwa"]
    w0mwa = d["w0mwa"]
    wa = 0.5 * (w0pwa - w0mwa)
    return wa

def _copy_param(p, source_name):
    """Derived-parameter function that ties a parameter to the value of another one."""
    return p[source_name]


def _compute_sigma8_ref(emu, z_array):
    """Reference sigma8 at the fiducial cosmology, evaluated at each z in z_array.

    Mirrors the sigma8 computation in postprocessing/add_sigmaR.py, using
    R=8/h Mpc when the emulator works in Mpc units, or R=8 Mpc/h when it
    works in Mpc/h units, so that it is always directly comparable to the
    'sigma_8' derived parameter (see Params.get_sigma_8).
    """
    h = cosmo_fid['h']
    R = 8.0 / h if emu.use_Mpc else 8.0
    comet_dict = {'wb': cosmo_fid['wb'], 'wc': cosmo_fid['wc'], 'h': h, 'ns': cosmo_fid['ns'],
                  'As': cosmo_fid['As'], 'Mnu': cosmo_fid['Mnu'], 'z': np.array(z_array, dtype=float)}
    return np.atleast_1d(emu.sigmaR(R, comet_dict, de_model='lambda'))


def get_pars(bias_basis='DESI', counterterm_basis='DESIct', reparam_option=None,
             de_model='lambda', freedom='max', b1_ref=2.109, sigmaR_ref=0.539, sigma1_eff=150/70 * 10**(1/3) * (1 + 0.8)**(1/2), fsat=0.13,
             bispec=False, free_Mnu=False, z_array=None, ns_times_Planck=10,
             use_Mpc=True, avirB_free=False, sigma_kind=None, rotatew0wa=False, model='VDG', cnlo_free=False):
    # model:
    #   'VDG' - VDG_infty model: virial damping parameters avir (and avirB for
    #           the bispectrum)
    #   'EFT' - EFT model: no damping, with the k^4 counterterm cnlo fixed
    #           to zero, or free with cnlo_free=True. Power spectrum only (the
    #           EFT bispectrum is not supported yet)
    # reparam_option:
    #   'full'     - all nuisance parameters reparametrised (bias with ap+sigma)
    #   'hybrid'   - as 'full', but bias parameters rescaled by sigma only (no ap)
    #   'jeffreys' - only the bias parameters entering the model non-linearly
    #                (b1, b2d, bk2) are reparametrised (ap+sigma); the linear
    #                ones keep their original form, to be analytically
    #                marginalised with a Jeffreys prior (which is invariant
    #                under their reparametrisation anyway)
    #   None/'none' - no reparametrisation

    if isinstance(reparam_option, str) and reparam_option.lower() == 'none':
        reparam_option = None
    reparam_options = ['full', 'hybrid', 'jeffreys']
    if reparam_option is not None and reparam_option not in reparam_options:
        raise ValueError(f"Invalid reparam_option: {reparam_option}. Must be one of {reparam_options} or None.")
    # whether the linear nuisance parameters are reparametrised too
    reparam_linear = reparam_option in ('full', 'hybrid')

    model_options = {'VDG': 'VDG_infty', 'EFT': 'EFT'}
    if model not in model_options:
        raise ValueError(f"Invalid model: {model}. Must be one of {list(model_options)}.")
    if model == 'EFT' and bispec:
        raise ValueError("The EFT model is only supported for the power spectrum (bispec=False).")
    if cnlo_free and model != 'EFT':
        raise ValueError("cnlo_free=True requires model='EFT' (the VDG model has no cnlo).")

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
    sigma1_eff = _to_nz_array(sigma1_eff, 'sigma1_eff')
    fsat = _to_nz_array(fsat, 'fsat')

    emu = COMET(model=model_options[model], use_Mpc=use_Mpc, bias_basis=bias_basis, counterterm_basis=counterterm_basis)

    # The reparametrization needs a reference sigma_R matching whichever sigma
    # (sigma_12 or sigma_8) it is normalized against, which in turn depends on
    # the working units (Mpc uses sigma_12, Mpc/h uses sigma_8).
    if use_Mpc:
        sigma_kind_ = 'sigma_12'
        active_sigmaR_ref = _to_nz_array(sigmaR_ref, 'sigmaR_ref')
    else:
        sigma_kind_ = 'sigma_8'
        # if sigma8_ref is not None:
        #     active_sigmaR_ref = _to_nz_array(sigma8_ref, 'sigma8_ref')
        # else:
        #     active_sigmaR_ref = _compute_sigma8_ref(emu, z_array)
        active_sigmaR_ref = _to_nz_array(sigmaR_ref, 'sigmaR_ref')
    if sigma_kind is None:
        sigma_kind = sigma_kind_
        print(f"Using default sigma_kind={sigma_kind} based on use_Mpc={use_Mpc}")
    else:
        if sigma_kind not in ['sigma_8', 'sigma_12']:
            raise ValueError(f"Invalid sigma_kind: {sigma_kind}. Must be 'sigma_8' or 'sigma_12'.")
        print(f"Using user-specified sigma_kind={sigma_kind} (use_Mpc={use_Mpc})")

    coev_params = ['bK2', 'btd'] if freedom == 'min' else []
    pars = Params(emu, coev_params=coev_params, z_array=z_array)
    #pars = Params(emu, coev_params=coev_params)
    pars.update_parameter('wb', 0.02237, prior=(0.02237, 0.00055), prior_type='gaussian', fixed=False)
    #if freedom == 'interm':
    #    pars.update_parameter('wb', 0.02237, prior=(0.02237, 0.00037), prior_type='gaussian', fixed=False)
    #pars.update_parameter('wc', 0.1200, prior=(0.085, 0.155), prior_type='uniform', fixed=False)
    pars.update_parameter('wc', 0.1200, prior=(0.08, 0.16), prior_type='uniform', fixed=False)
    #pars.update_parameter('As', 2.0830, prior=(1.055, 3.17), prior_type='uniform', fixed=False)
    pars.update_parameter('h', 0.6736, prior=(0.5, 1.0), prior_type='uniform', fixed=False)
    pars.update_parameter('ns', 0.9649, prior=(0.9649, ns_times_Planck * 0.0042), prior_type='gaussian', fixed=False)
    #if freedom == 'interm':
    #    pars.set_and_fix_param('ns', 0.9649)
    if de_model == 'w0wa':
        pars.update_parameter('w0', -1., prior=(-3., 1.), prior_type='uniform', fixed=False)
        pars.update_parameter('wa', 0., prior=(-3., 2.), prior_type='uniform', fixed=False)
        if rotatew0wa:
            pars.add_sampled_param('w0pwa', -1, prior=(-6., 0.), prior_type='uniform', latex=r'w_0 + w_a')
            pars.add_sampled_param('w0mwa', -1, prior=(-5., 4.), prior_type='uniform', latex=r'w_0 - w_a')

            pars.set_derived_param('w0', _w0_from_w0pwa_w0mwa, exported=False)
            pars.set_derived_param('wa', _wa_from_w0pwa_w0mwa, exported=False)

    elif de_model == 'w0':
        pars.update_parameter('w0', -1., prior=(-3., 1.), prior_type='uniform', fixed=False)
    elif de_model != 'lambda':
        raise ValueError(f"Invalid de_model: {de_model}. Must be one of 'lambda', 'w0', or 'w0wa'.")
    # pars.add_sampled_param("log10As", 3.04, prior=(2.31, 3.5), latex=r"\ln(10^{10} A_s)")
    pars.add_sampled_param("log10As", 3.04, prior=(1.61, 3.91), latex=r"\ln(10^{10} A_s)")
    pars.set_derived_param("As", _As_from_log10As, exported=False)
    
    if free_Mnu:
        pars.update_parameter('Mnu', 0.06, prior=(0.0, 5.0), prior_type='uniform', fixed=False)
    else:
        pars.set_and_fix_param('Mnu', 0.06)

    bK2ref = -2./7.*(b1_ref - 1.)
    btdref = 23./42.*(b1_ref - 1.)

    h = cosmo_fid['h']
    # The Gaussian widths below (stoch_scale, NP20_r, NP22_r) are defined as
    # "natural" scales in Mpc/h units; when working in Mpc they must be
    # converted down by the relevant power of h, but when already working in
    # Mpc/h no conversion is needed.
    hconv = h if use_Mpc else 1.0
    stoch_scales = {'max': 500., 'min': 50., 'interm': 50.}
    stoch_scale = stoch_scales[freedom] / (hconv**2)
    cnlo_scale = 500. / (hconv**4)


    stochastic_mode = 'ap'
    counterterms_mode = f'ap+{sigma_kind}'
    if reparam_option in ('full', 'jeffreys'):
        bias_mode = f'ap+{sigma_kind}'
    elif reparam_option == 'hybrid':
        bias_mode = sigma_kind


    gauss_scales = {
        'max': {'NP0_r': np.full(nz, 20., dtype=float), 'NP20_r': 50. * fsat * sigma1_eff**2, 'NP22_r': 50. * fsat * sigma1_eff**2},
        'min': {'NP0_r': np.full(nz, 2., dtype=float), 'NP20_r': 5. * fsat * sigma1_eff**2, 'NP22_r': 5. * fsat * sigma1_eff**2},
        'interm': {'NP0_r': np.full(nz, 2., dtype=float), 'NP20_r': 5. * fsat * sigma1_eff**2, 'NP22_r': 5. * fsat * sigma1_eff**2},
    }
    
    gs = gauss_scales[freedom]

    if reparam_option is not None:
        pars.use_reparametrization(bias_mode=bias_mode, counterterms_mode=counterterms_mode,
                                    stochastic_mode=stochastic_mode, sigmaR_ref=active_sigmaR_ref,
                                    skip_linear_params=not reparam_linear)

    sub_rep = '_r' if reparam_option is not None else ''

    for iz in range(nz):
        sub_idz = f"_{iz}" if nz > 1 else ""
        subscript = sub_rep + sub_idz
        subscript_linear = subscript if reparam_linear else sub_idz
        pars.update_parameter(f'b1{subscript}', 2., prior=(0.1, 8), prior_type='uniform', fixed=False)
        pars.update_parameter(f'b2d{subscript}', 0., prior=(0, 20), prior_type='gaussian', fixed=False)
        # if freedom == 'interm':
        #     pars.update_parameter(f'b1{subscript}', 2., prior=(0.1, 4), prior_type='uniform', fixed=False)
        #     pars.update_parameter(f'b2d{subscript}', 0., prior=(0, 5), prior_type='gaussian', fixed=False)
        if freedom == 'max':
            pars.update_parameter(f'bk2{subscript}', bK2ref[iz], prior=(bK2ref[iz], 20), prior_type='gaussian', fixed=False)
            pars.update_parameter(f'btd{subscript_linear}', btdref[iz], prior=(btdref[iz], 80), prior_type='gaussian', fixed=False)
        elif freedom == 'interm':
            pars.update_parameter(f'bk2{subscript}', bK2ref[iz], prior=(bK2ref[iz], 20.), prior_type='gaussian', fixed=False)
            pars.update_parameter(f'btd{subscript_linear}', btdref[iz], prior=(btdref[iz], 1.), prior_type='gaussian', fixed=False)

        if model == 'VDG':
            pars.update_parameter(f'avir{sub_idz}', 5., prior=(0, 20./hconv), prior_type='uniform', fixed=False)
        elif cnlo_free:
            # k^4 counterterm: enters the model linearly, like a0/a2, and is
            # reparametrised the same way. Width as in the COMET examples,
            # in (Mpc/h)^4.
            pars.update_parameter(f'cnlo{subscript_linear}', 0., prior=(0, cnlo_scale), prior_type='gaussian', fixed=False)
        else:
            pars.set_and_fix_param(f'cnlo{subscript_linear}', 0.)
            if reparam_linear:
                pars.set_and_fix_param(f'cnlo{sub_idz}', 0.)
        pars.update_parameter(f'NP0{subscript_linear}', 0., prior=(0, gs['NP0_r'][iz]), prior_type='gaussian', fixed=False)
        pars.update_parameter(f'NP20{subscript_linear}', 0., prior=(0, gs['NP20_r'][iz]/hconv**2), prior_type='gaussian', fixed=False)
        pars.update_parameter(f'NP22{subscript_linear}', 0., prior=(0, gs['NP22_r'][iz]/hconv**2), prior_type='gaussian', fixed=False)

        if counterterm_basis == 'DESIct':
            pars.update_parameter(f'a0{subscript_linear}', 0., prior=(0, stoch_scale), prior_type='gaussian', fixed=False)
            pars.update_parameter(f'a2{subscript_linear}', 0., prior=(0, stoch_scale), prior_type='gaussian', fixed=False)
            #pars.update_parameter(f'a4{subscript}', 0., prior=(0, stoch_scale), prior_type='gaussian', fixed=False)
            pars.set_and_fix_param(f'a4{subscript_linear}', 0.)
            if reparam_linear:
                pars.set_and_fix_param(f'a4{sub_idz}', 0.)
        elif counterterm_basis == 'Comet':
            pars.update_parameter(f'c0{subscript_linear}', 0., prior=(0, stoch_scale), prior_type='gaussian', fixed=False)
            pars.update_parameter(f'c2{subscript_linear}', 0., prior=(0, stoch_scale), prior_type='gaussian', fixed=False)
            #pars.update_parameter(f'c4{subscript}', 0., prior=(0, stoch_scale), prior_type='gaussian', fixed=False)
            pars.set_and_fix_param(f'c4{subscript_linear}', 0.)
            if reparam_linear:
                pars.set_and_fix_param(f'c4{sub_idz}', 0.)

        if bispec:
            pars.update_parameter(f'NB0{subscript_linear}', 0., prior=(0, 2.), prior_type='gaussian', fixed=False)
            pars.update_parameter(f'MB0{subscript_linear}', 0., prior=(0, 1.), prior_type='gaussian', fixed=False)
            if avirB_free:
                pars.update_parameter(f'avirB{sub_idz}', 0., prior=(0, 20./hconv), prior_type='uniform', fixed=False)
            else:
                # Default: tie avirB to avir instead of sampling it independently.
                pars.set_derived_param(f'avirB{sub_idz}', partial(_copy_param, source_name=f'avir{sub_idz}'), exported=False)
        else:
            pars.set_and_fix_param(f'NB0{subscript_linear}', 0.)
            pars.set_and_fix_param(f'MB0{subscript_linear}', 0.)
            if model == 'VDG':
                pars.set_and_fix_param(f'avirB{sub_idz}', 0.)

            if reparam_linear:
                pars.set_and_fix_param(f'NB0{sub_idz}', 0.)
                pars.set_and_fix_param(f'MB0{sub_idz}', 0.)

    return pars

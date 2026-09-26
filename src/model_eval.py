"""Rebuild the observables/likelihood behind a saved fit_cutsky_abacushf.py
chain (from the run metadata stored as .h5 attrs), evaluate the model at a
given getdist MCSamples' posterior mean, and plot it against the data.
"""
import numpy as np
import matplotlib.pyplot as plt

import fit_cutsky_abacushf as fc  # also wires up sys.path for read_data/likelihood/observables
from read_data import get_obs_pk, get_obs_pk_bk
from likelihood import Likelihood


def build_likelihood_from_attrs(attrs):
    """Rebuild the Likelihood (observables + Params/COMET emulator) that
    produced a chain, from the run metadata `fit_cutsky_abacushf.py` saves
    as .h5 attrs (as returned by `plot_utils.get_samples(fn, return_attrs=True)`)."""
    from types import SimpleNamespace

    args = SimpleNamespace(
        tracer_label=list(attrs['tracer_label']),
        region=str(attrs['region']),
        bispec=bool(attrs['bispec']),
        de_model=str(attrs['de_model']),
        reparam=str(attrs['reparam']),
        freedom=str(attrs['freedom']),
        free_Mnu=bool(attrs['free_Mnu']),
        counterterm_basis=str(attrs['counterterm_basis']),
        avirB_free=bool(attrs['avirB_free']),
        mpc_h=bool(attrs['mpc_h']),
        zeff_choice=str(attrs.get('zeff_choice', 'zsnap')),
    )

    data_dir = attrs.get('data_dir', 'None')
    data_dir_kw = {} if data_dir in (None, 'None') else {'outdir': str(data_dir)}

    tracer_list, zrange_list = [], []
    for label in args.tracer_label:
        info = fc.tracer_label_dict[label]
        tracer_list.append(info['tracer'])
        zrange_list.append(info['zrange'])

    mocktypes_used = list(attrs['mocktypes_used'])
    mocktypes_cov_used = list(attrs['mocktypes_cov_used'])

    observables = []
    for i, (tracer_i, zr) in enumerate(zip(tracer_list, zrange_list)):
        mocktype = mocktypes_used[i]
        mocktype_cov = mocktypes_cov_used[i]
        if not args.bispec:
            obs_i = get_obs_pk(
                tracer=tracer_i, zrange=zr, region=args.region, mocktype=mocktype, mocktype_cov=mocktype_cov,
                ell=[int(x) for x in attrs['ellP']], kmin=list(attrs['kminP']), kmax=list(attrs['kmaxP']),
                ellwin=[int(x) for x in attrs['ellwinP']], kwinmin=list(attrs['kwinminP']), kwinmax=list(attrs['kwinmaxP']),
                dk=float(attrs['dkP']), use_Mpc=not args.mpc_h, **data_dir_kw)
        else:
            obs_i = get_obs_pk_bk(
                tracer=tracer_i, zrange=zr, region=args.region, mocktype=mocktype, mocktype_cov=mocktype_cov,
                ellP=[int(x) for x in attrs['ellP']], kminP=list(attrs['kminP']), kmaxP=list(attrs['kmaxP']),
                ellwinP=[int(x) for x in attrs['ellwinP']], kwinminP=list(attrs['kwinminP']), kwinmaxP=list(attrs['kwinmaxP']),
                dkP=float(attrs['dkP']),
                ellB=[tuple(int(x) for x in row) for row in attrs['ellB']], kminB=list(attrs['kminB']), kmaxB=list(attrs['kmaxB']),
                ellwinB=[tuple(int(x) for x in row) for row in attrs['ellwinB']], kwinminB=list(attrs['kwinminB']), kwinmaxB=list(attrs['kwinmaxB']),
                dkB=float(attrs['dkB']), slice_winB_theory=2, use_Mpc=not args.mpc_h, **data_dir_kw)
        if args.zeff_choice == 'zsnap':
            obs_i.cosmo_fid['z'] = fc.zsnap_dict[tracer_i][zr]
        observables.append(obs_i)

    b1_ref, sigmaR_ref, sigma1_eff, fsat = fc.get_prior_refs(tracer_list, zrange_list)
    z_array = np.array([obs.cosmo_fid['z'] for obs in observables])
    sort_idx = np.argsort(z_array)
    z_array = z_array[sort_idx]
    observables = [observables[i] for i in sort_idx]

    am_params = fc.get_am_params(args.counterterm_basis, args.bispec, args.reparam)

    pars = fc.build_pars(args, b1_ref, sigmaR_ref, sigma1_eff, fsat, z_array)

    conditional_prior_fn = None
    if args.de_model == 'w0wa':
        def conditional_prior_fn(params):
            return params['w0'] + params['wa'] < 0

    return Likelihood(observables, pars, am_params=am_params, jeffreys=args.reparam == 'jeffreys',
                      conditional_prior=conditional_prior_fn)


def evaluate_model_at_means(likelihood, samples):
    """Evaluate the model at the posterior mean of `samples`, including the
    linear analytically-marginalised (AM) contribution added back in via the
    likelihood's design matrix (see Likelihood.get_chi2/marg_chi2).

    Returns one flattened array per observable in `likelihood.observables`,
    in the same order as `obs.get_flatten()`.
    """
    names = samples.getParamNames().list()
    means = dict(zip(names, samples.getMeans()))

    pars = likelihood.params
    free_dict = {name: means[name] for name in pars.sampled_param_names}
    full_dict = pars.get_full_dict(free_dict)

    am_params = likelihood.am_params if likelihood.do_am else [[] for _ in likelihood.observables]
    # Only overwrites the '_r'-suffixed AM keys (e.g. 'a0_r'), not their
    # physical counterparts (e.g. 'a0'), which Likelihood.__init__ pinned to
    # 0 and get_comet_dict reads instead -- so this does not feed into
    # `preds` below, and the design-matrix contribution isn't double-counted.
    for am_names_iz in am_params:
        for name in am_names_iz:
            full_dict[name] = means[name]

    comet_params = pars.get_comet_dict(full_dict)
    preds = likelihood.emu.predict(likelihood.observables, comet_params, de_model=likelihood.de_model)

    cache = {}
    models = []
    for iz, am_names_iz in enumerate(am_params):
        if am_names_iz:
            dm = likelihood.get_design_matrix(full_dict, cache, iz)
            am_vals = np.array([full_dict[name] for name in am_names_iz])
            models.append(preds[iz] + dm @ am_vals)
        else:
            models.append(preds[iz])
    return models


def _split_flat(y_list, flat):
    """Split a flattened 1D array into segments matching the lengths of
    y_list (e.g. an Observable's `.y`, one array per multipole)."""
    segs = []
    offset = 0
    for y in y_list:
        n = len(y)
        segs.append(flat[offset:offset + n])
        offset += n
    return segs


def plot_model_over_data(obs, model_flat, h_units=True, ax=None):
    """Plot `obs`'s data (via its own `.plot()`) and overlay `model_flat`
    (as returned per-observable by `evaluate_model_at_means`) in matching
    colors. Handles JointObservable by recursing into its two sub-observables
    on a pair of side-by-side axes."""
    if obs.__class__.__name__ == 'JointObservable':
        obs1, obs2 = obs.observables
        model1, model2 = model_flat[:obs1.n_data], model_flat[obs1.n_data:]
        if ax is None:
            _, ax = plt.subplots(1, 2, figsize=(11, 4))
        plot_model_over_data(obs1, model1, h_units=h_units, ax=ax[0])
        plot_model_over_data(obs2, model2, h_units=h_units, ax=ax[1])
        return ax

    ax = obs.plot(ax=ax, h_units=h_units)
    factor = obs.h_fid if (h_units and obs.h_fid is not None) else 1.0
    model_segs = _split_flat(obs.y, model_flat)

    for i, container in enumerate(ax.containers):
        color = container.lines[0].get_color()
        if obs.__class__.__name__ == 'PowerSpectrumMultipoles':
            x = obs.x[i]
            ax.plot(x / factor, x * model_segs[i] * factor**2, '-', color=color)
        elif obs.__class__.__name__ == 'BispectrumSugiyamaMultipoles':
            k = obs.x[i][:, 0]
            ax.plot(k / factor, k**2 * model_segs[i] * factor**4, '-', color=color)
        else:
            raise TypeError(f"Unsupported observable type for plotting: {type(obs)}")
    return ax

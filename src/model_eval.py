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
        sigma_kind=None,
        rotatew0wa=False,
        model='VDG',
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
    """Evaluate the model at the posterior mean of `samples` (see `evaluate_model`)."""
    names = samples.getParamNames().list()
    return evaluate_model(likelihood, dict(zip(names, samples.getMeans())))


def evaluate_model(likelihood, values):
    """Evaluate the model at the parameter values `values` (a dict holding
    the sampled parameters and, if the likelihood uses AM, the linear
    analytically-marginalised ones, e.g. a MinuitMinimizer.get_map() best
    fit), including the AM contribution added back in via the likelihood's
    design matrix (see Likelihood.get_chi2/marg_chi2).

    Returns one flattened array per observable in `likelihood.observables`,
    in the same order as `obs.get_flatten()`.
    """
    pars = likelihood.params
    free_dict = {name: values[name] for name in pars.sampled_param_names}
    full_dict = pars.get_full_dict(free_dict)

    am_params = likelihood.am_params if likelihood.do_am else [[] for _ in likelihood.observables]
    # Only overwrites the '_r'-suffixed AM keys (e.g. 'a0_r'), not their
    # physical counterparts (e.g. 'a0'), which Likelihood.__init__ pinned to
    # 0 and get_comet_dict reads instead -- so this does not feed into
    # `preds` below, and the design-matrix contribution isn't double-counted.
    for am_names_iz in am_params:
        for name in am_names_iz:
            full_dict[name] = values[name]

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
    factor, _ = obs.plot_units(h_units)
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


def _plot_x(obs, i, factor):
    """x values (in plotting units) of multipole `i` of `obs`, as used by `obs.plot()`."""
    if obs.__class__.__name__ == 'BispectrumSugiyamaMultipoles':
        return obs.x[i][:, 0] / factor
    return obs.x[i] / factor


def plot_model_over_data_residuals(obs, model_flat, h_units=True, fig=None, n_sigma=2):
    """Same as `plot_model_over_data`, plus one panel per multipole below the
    main one showing (data - model) / sigma, with a +-`n_sigma` band. Handles
    JointObservable by placing its two sub-observables in side-by-side columns.
    Returns the figure."""
    if fig is None:
        n_cols = 2 if obs.__class__.__name__ == 'JointObservable' else 1
        n_res = max(o.n_obs for o in obs.observables) if n_cols == 2 else obs.n_obs
        fig = plt.figure(figsize=(5.5 * n_cols, 4 + 1.2 * n_res))

    if obs.__class__.__name__ == 'JointObservable':
        obs1, obs2 = obs.observables
        model1, model2 = model_flat[:obs1.n_data], model_flat[obs1.n_data:]
        subfig1, subfig2 = fig.subfigures(1, 2)
        plot_model_over_data_residuals(obs1, model1, h_units=h_units, fig=subfig1, n_sigma=n_sigma)
        plot_model_over_data_residuals(obs2, model2, h_units=h_units, fig=subfig2, n_sigma=n_sigma)
        return fig

    axes = fig.subplots(1 + obs.n_obs, 1, sharex=True, height_ratios=[3] + [1] * obs.n_obs,
                        gridspec_kw={'hspace': 0.05})
    ax_main, ax_res = axes[0], axes[1:]
    plot_model_over_data(obs, model_flat, h_units=h_units, ax=ax_main)
    xlabel = ax_main.get_xlabel()
    ax_main.set_xlabel('')

    factor, _ = obs.plot_units(h_units)
    model_segs = _split_flat(obs.y, model_flat)
    err_segs = _split_flat(obs.y, np.sqrt(np.diag(obs.cov)))
    for i, (ax, container) in enumerate(zip(ax_res, ax_main.containers)):
        color = container.lines[0].get_color()
        x = _plot_x(obs, i, factor)
        ax.axhspan(-n_sigma, n_sigma, color='gray', alpha=0.2, lw=0)
        ax.axhline(0, color='k', lw=0.8)
        ax.plot(x, (obs.y[i] - model_segs[i]) / err_segs[i], 'o', color=color, ms=4)
        ax.set_ylabel(r'$\Delta / \sigma$')
        ax.text(0.02, 0.9, container.get_label(), transform=ax.transAxes, ha='left', va='top', fontsize='small')
    ax_res[-1].set_xlabel(xlabel)
    return fig

if __name__ == '__main__':
    import argparse
    from plot_utils import get_samples

    parser = argparse.ArgumentParser()
    parser.add_argument('chain_file', type=str, help="Path to a getdist .h5 chain file")
    parser.add_argument('--h_units', action='store_true', help="Plot in h Mpc^-1 units")
    parser.add_argument('--plot_fn', type=str, default=None, help="If given, save the plot to this filename instead of showing it interactively")
    args = parser.parse_args()

    samples, attrs = get_samples(args.chain_file, return_attrs=True)
    likelihood = build_likelihood_from_attrs(attrs)
    model_flat = evaluate_model_at_means(likelihood, samples)

    for obs, model in zip(likelihood.observables, model_flat):
        plot_model_over_data(obs, model, h_units=args.h_units)
    if args.plot_fn:
        plt.savefig(args.plot_fn, bbox_inches='tight')
    else:
        plt.show()
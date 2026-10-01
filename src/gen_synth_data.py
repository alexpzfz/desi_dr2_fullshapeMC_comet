"""Build a noiseless synthetic data vector from the posterior mean of a
fit_cutsky_abacushf.py Nautilus chain, or from a Minuit best fit
(fit_cutsky_abacushf.py --minimize), detected from the file contents.

Several files can be given, e.g. the single-tracer best fits of each tracer:
each file is evaluated on its own (with its own run settings and parameters)
and its tracers are written under the same `--mocktype_out`, so they can be
fitted jointly. The files must not share tracers and must agree on region,
statistic (pk / pk+bk), k binning and zeff choice.

The run settings (tracers, units, reparametrisation, model, zeff choice,
windows, ...) are read from each file's .h5 attrs. The model is evaluated on
the full k range of the cached data files (all multipoles, no scale cuts)
unless --kmin*/--kmax* are given, so the synthetic data can be fitted with
different scale cuts than the fit it came from.

With --abacus_cosmo, the cosmological parameters are set to an AbacusSummit
cosmology (default c000) and only the nuisance parameters are taken from the
chain / best fit. --nuisance_basis controls which nuisance values are held
fixed when the cosmology changes: the physical ones (b1, a0, ...), or the
sampled ones (b1_r, a0_r, ...), whose physical values then follow the new
sigma_R / q_iso.

With --no_window (P(k) only), the model is the theory at the data k, with no
window convolution, and identity window matrices are written for the fit.
With --gaussian_cov (requires --no_window), the covariances are COMET's
Gaussian ones at the true parameters, with the data nbar as (uniform) density.
The volume of each tracer is set by --cov_volume:
  match_mocks (default): the P0 variance matches the mock covariance over
      --cov_match_krange (median ratio of the diagonals);
  geometric: the comoving volume of the DESI DR2 footprint (DESI_DR2_AREA_DEG2)
      over the tracer's redshift range, in the fiducial cosmology of the data;
  desi_dr2_veff: V = V_eff (1 + 1/(nP))^2 from the DESI DR2 effective volumes
      (DESI_DR2_VEFF_GPC3), kept for reference.
The two DESI-based ones give errors larger than the mocks' (about 1.2-1.5x for
LRG3), which the mocks' scatter is below. The covariances are written with
nmocks0, which read_data.py takes as an analytic covariance (no
Hartlap/Percival correction).

The output directory mirrors the cached-data layout read by read_data.py:
new mean_pk/mean_bk files under mocktype `--mocktype_out`, plus symlinks to the
original windows (renamed to that mocktype) and covariances, or the identity
windows / Gaussian covariances. Fit it with

    python fit_cutsky_abacushf.py --tracer_label BGS LRG1 LRG2 LRG3 ELG2 \
        --data_dir <outdir> --mocktype <mocktype_out> ...

adding --mocktype_cov <mocktype_out> with --gaussian_cov.
"""
import os
import re
import sys
import json
import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt

import fit_cutsky_abacushf as fc  # also sets thread env vars and wires up sys.path
import env
from read_data import get_obs_pk, get_obs_pk_bk, get_fn
from likelihood import Likelihood
import plot_utils as pu

ELLP_ALL = [0, 2, 4]
ELLB_ALL = [(0, 0, 0), (2, 0, 2)]

# DESI DR2 effective volumes (full footprint, i.e. GCcomb) in Gpc^3, without
# h, defined as V_eff = int dV [n P / (1 + n P)]^2 with P = P(k, mu) at the
# reference point below
DESI_DR2_VEFF_GPC3 = {'BGS': 3.8, 'LRG1': 4.9, 'LRG2': 7.6, 'LRG3': 9.8, 'ELG1': 5.8, 'ELG2': 8.3, 'QSO': 2.7}
VEFF_KREF, VEFF_MUREF = 0.14, 0.6  # h/Mpc
# DESI DR2 footprint areas (full footprint, i.e. GCcomb)
DESI_DR2_AREA_DEG2 = {'BGS': 12355, 'LRG1': 10031, 'LRG2': 10031, 'LRG3': 10031, 'ELG1': 10352, 'ELG2': 10352, 'QSO': 11181}
FULL_SKY_DEG2 = 4 * np.pi * (180 / np.pi)**2


def _attr(attrs, name, default=None):
    """Read a run-metadata attr, mapping the 'None' string h5 stores for None
    (and missing attrs of older chains) to `default`."""
    val = attrs.get(name, default)
    if isinstance(val, str) and val == 'None':
        return default
    return val


def _per_ell(val, n):
    """Scalar or per-ell list -> list of length n."""
    val = np.atleast_1d(np.asarray(val, dtype=float)).tolist()
    return val * n if len(val) == 1 else val


def get_run_args(attrs):
    """Namespace with the fit_cutsky_abacushf.py options that build_pars needs."""
    return SimpleNamespace(
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
        sigma_kind=_attr(attrs, 'sigma_kind'),
        rotatew0wa=bool(_attr(attrs, 'rotatew0wa', False)),
        model=str(_attr(attrs, 'model', 'VDG')),
        zeff_choice=str(_attr(attrs, 'zeff_choice', 'zsnap')),
        fix_cosmo=_attr(attrs, 'fix_cosmo'),
        free_cnlo=bool(_attr(attrs, 'free_cnlo', False)),
    )


def get_data_dir(args, attrs):
    """Cached data a fit was run on: --data_dir, else the fit's own, else the default."""
    return Path(args.data_dir or _attr(attrs, 'data_dir') or env.DATA_DIR_CUTSKY)


def get_shared_settings(attrs):
    """Run settings that must agree between the files written under one mocktype."""
    bispec = bool(attrs['bispec'])
    return {'region': str(attrs['region']), 'bispec': bispec, 'dkP': float(attrs['dkP']),
            'dkB': float(attrs['dkB']) if bispec else None,
            'zeff_choice': str(_attr(attrs, 'zeff_choice', 'zsnap'))}


def load_reference(fn):
    """Reference parameter values and run attrs of a fit output: the posterior
    mean of a Nautilus chain, or the best fit of a Minuit file (which also
    holds the conditional MAP of any analytically-marginalised parameters).
    Returns (values, attrs, kind) with kind 'chain' or 'bestfit'."""
    import h5py
    with h5py.File(fn, 'r') as f:
        is_bestfit = 'best_fit' in f
        if is_bestfit:
            values = dict(zip(f['names'].asstr()[:], f['best_fit'][:].astype(float)))
            attrs = dict(f.attrs)
    if is_bestfit:
        if not attrs.get('valid', True):
            print(f"WARNING: {fn} did not fully converge (valid=False, edm={attrs.get('edm', np.nan):.3e}).")
        return values, attrs, 'bestfit'
    samples, attrs = pu.get_samples(fn, return_attrs=True)
    return dict(zip(samples.getParamNames().list(), samples.getMeans())), attrs, 'chain'


def set_identity_window(obs):
    """Replace the window of a P(k) observable by the identity on its own k
    grid: the theory evaluated at the data k, without convolution. This is
    what the fit gets from the files of write_identity_window."""
    obs.wmat = np.eye(obs.n_data)
    obs.xwin = obs.kwin = [x.copy() for x in obs.x]
    obs.ellwin = list(obs.ell)
    obs.nobswin = obs.n_obs


def build_full_range_observables(args, attrs, data_dir, no_window=False):
    """One observable per tracer with all multipoles on the full k grid of the
    cached files, using the chain's data/windows/covariances (or no window),
    sorted by redshift as in the fit. Returns (observables, labels) in that order."""
    mocktypes = list(attrs['mocktypes_used'])
    mocktypes_cov = list(attrs['mocktypes_cov_used'])
    data_dir_kw = {} if data_dir is None else {'outdir': data_dir}

    observables = []
    for label, mocktype, mocktype_cov in zip(args.tracer_label, mocktypes, mocktypes_cov):
        tracer, zr = fc.tracer_label_dict[label]['tracer'], fc.tracer_label_dict[label]['zrange']
        common = dict(tracer=tracer, zrange=zr, region=args.region, mocktype=mocktype, mocktype_cov=mocktype_cov,
                      use_Mpc=not args.mpc_h, **data_dir_kw)
        winP = dict(ellwinP=[int(x) for x in attrs['ellwinP']], kwinminP=np.asarray(attrs['kwinminP']).tolist(),
                    kwinmaxP=np.asarray(attrs['kwinmaxP']).tolist(), dkP=float(attrs['dkP']))
        if not args.bispec:
            obs = get_obs_pk(ell=ELLP_ALL, kmin=0., kmax=np.inf,
                             ellwin=winP['ellwinP'], kwinmin=winP['kwinminP'], kwinmax=winP['kwinmaxP'],
                             dk=winP['dkP'], **common)
        else:
            obs = get_obs_pk_bk(ellP=ELLP_ALL, kminP=_per_ell(0., 3), kmaxP=_per_ell(np.inf, 3),
                                ellB=ELLB_ALL, kminB=_per_ell(0., 2), kmaxB=_per_ell(np.inf, 2),
                                ellwinB=[tuple(int(x) for x in row) for row in attrs['ellwinB']],
                                kwinminB=np.asarray(attrs['kwinminB']).tolist(), kwinmaxB=np.asarray(attrs['kwinmaxB']).tolist(),
                                dkB=float(attrs['dkB']), slice_winB_theory=2, **winP, **common)
        if no_window:
            set_identity_window(obs)
        if args.zeff_choice == 'zsnap':
            obs.cosmo_fid['z'] = fc.zsnap_dict[tracer][zr]
        observables.append(obs)

    order = np.argsort([obs.cosmo_fid['z'] for obs in observables])
    return [observables[i] for i in order], [args.tracer_label[i] for i in order]


def nuisance_names(pars):
    """Physical (emulator-facing) nuisance parameter names, e.g. b1_0, a0_2."""
    return [name for group in (pars.bias_params, pars.counterterm_params, pars.stochastic_params, pars.extra_params)
            for name in group]


def get_true_params(pars, means, abacus_cosmo=None, nuisance_basis='physical'):
    """Full parameter dict for the synthetic data: the reference values (chain
    posterior mean or best fit), or, with `abacus_cosmo`, the Abacus cosmology
    plus their nuisance parameters in the requested basis."""
    missing = [p for p in pars.sampled_param_names if p not in means]
    if missing:
        raise KeyError(f"Sampled parameters not found in the chain / best fit: {missing}")
    free_chain = {p: means[p] for p in pars.sampled_param_names}
    full_chain = pars.get_full_dict(free_chain)
    if abacus_cosmo is None:
        return full_chain

    cosmo_true = fc.abacus_sampled_cosmo(abacus_cosmo, pars)
    # A --fix_cosmo fit has no sampled cosmology: override the fixed values
    # instead (get_full_dict lets the given values take precedence)
    values = fc.abacus_cosmo_values(abacus_cosmo)
    cosmo_true |= {p: values[p] for p in getattr(pars, 'fixed_cosmo_names', [])}
    full_true = pars.get_full_dict(free_chain | cosmo_true)
    if nuisance_basis == 'physical':
        # The physical nuisance values derived at the reference cosmology;
        # this also overrides the co-evolution relations, which only depend
        # on the (unchanged) physical biases.
        for name in nuisance_names(pars):
            if name in full_chain:
                full_true[name] = full_chain[name]
                # keep the reparametrised value (e.g. b1_r) consistent with
                # it at the new cosmology, as the value a fit would recover
                func = pars.parameters[name].derived_func
                r_name = getattr(func, 'keywords', {}).get('name')
                if getattr(func, 'func', None) == pars._compute_reparam and r_name is not None:
                    full_true[r_name] = full_true[name] / pars.get_reparam_factor(full_true, r_name)
    return full_true


# Counterterms (both bases, plus the EFT k^4 one) and the scale-dependent
# stochastic terms, zeroed with --zero_ct_np2
ZERO_CT_NP2 = ('a0', 'a2', 'a4', 'c0', 'c2', 'c4', 'cnlo', 'NP20', 'NP22')


def zero_params(full_dict, bases):
    """Set the parameters with these base names to zero, in place: the physical
    and the reparametrised (_r) ones, of every z bin (_<i>)."""
    pattern = re.compile(rf"^({'|'.join(map(re.escape, bases))})(_r)?(_\d+)?$")
    for name in full_dict:
        if pattern.match(name):
            full_dict[name] = 0.
    return full_dict


def to_h_units(obs, model_flat):
    """Split one observable's flattened model into P(k) and B(k1, k2)
    multipoles in (Mpc/h) units, as stored in the cached data files."""
    if obs.__class__.__name__ == 'JointObservable':
        obs_pk, obs_bk = obs.observables
        pk, _ = to_h_units(obs_pk, model_flat[:obs_pk.n_data])
        _, bk = to_h_units(obs_bk, model_flat[obs_pk.n_data:])
        return pk, bk
    h3 = obs.h_fid**3 if obs.Mpc_units else 1.
    segs = np.split(model_flat, np.cumsum([len(y) for y in obs.y])[:-1])
    if obs.__class__.__name__ == 'PowerSpectrumMultipoles':
        return [seg * h3 for seg in segs], None
    return None, [seg * h3**2 for seg in segs]


def gaussian_cov_unit_volume(emu, obs, model_flat, dkP):
    """COMET Gaussian covariance of the P0, P2, P4 of `obs` (full k grid,
    observable units) for a unit volume, with `model_flat` the noiseless
    multipoles and the Poisson shot noise 1/nbar added to P0, as in
    comet's Pell_covariance."""
    k = obs.x[0]
    dk = dkP * obs.h_fid if obs.Mpc_units else dkP
    p0, p2, p4 = np.split(model_flat, 3)
    Pell = {'ell0': p0 + 1. / obs.nbar, 'ell2': p2, 'ell4': p4}
    n = len(k)
    cov = np.zeros((3 * n, 3 * n))
    for i, l1 in enumerate(ELLP_ALL):
        for j, l2 in enumerate(ELLP_ALL[i:], start=i):
            block = np.diag(emu._Gaussian_covariance(l1, l2, k, dk, Pell, volume=1.))
            cov[i * n:(i + 1) * n, j * n:(j + 1) * n] = block
            cov[j * n:(j + 1) * n, i * n:(i + 1) * n] = block
    return cov


def volume_from_veff(veff_gpc3, obs, model_flat):
    """Volume (observable units) to use in the Gaussian covariance, which
    includes the shot noise, 2 (P + 1/n)^2 / N_modes(V), so that it equals the
    one implied by an effective volume, 2 P^2 / N_modes(V_eff), at the V_eff
    reference point: V = V_eff (1 + 1 / (n P))^2, with P(VEFF_KREF, VEFF_MUREF)
    from the model and n the data nbar (uniform-n approximation).
    Returns (V, n P)."""
    from scipy.special import eval_legendre
    k = obs.x[0]
    kref = VEFF_KREF * obs.h_fid if obs.Mpc_units else VEFF_KREF
    pref = sum(np.interp(kref, k, p) * eval_legendre(ell, VEFF_MUREF)
               for ell, p in zip(ELLP_ALL, np.split(model_flat, 3)))
    nP = obs.nbar * pref
    # Gpc^3 -> Mpc^3, or (Mpc/h)^3 with the fiducial h
    veff = veff_gpc3 * 1e9 * (1. if obs.Mpc_units else obs.h_fid**3)
    return veff * (1. + 1. / nP)**2, float(nP)


def geometric_volume(area_deg2, zrange, obs):
    """Comoving volume (observable units) of `area_deg2` between zrange[0] and
    zrange[1], in the fiducial flat LCDM cosmology of the data (`obs.cosmo_fid`)."""
    c = obs.cosmo_fid
    om = (c['wb'] + c['wc'] + c.get('Mnu', 0.) / 93.14) / c['h']**2
    z = np.linspace(0., zrange[1], 20001)
    inv_e = 1. / np.sqrt(om * (1 + z)**3 + 1 - om)
    chi = np.concatenate([[0.], np.cumsum((inv_e[1:] + inv_e[:-1]) / 2 * np.diff(z))]) * 2997.92458  # Mpc/h
    chi_min, chi_max = np.interp(zrange, z, chi)
    volume = area_deg2 / FULL_SKY_DEG2 * 4 * np.pi / 3 * (chi_max**3 - chi_min**3)
    return volume / obs.h_fid**3 if obs.Mpc_units else volume


def matched_volume(cov_unit, cov_mocks, k_h, krange):
    """Volume (observable units) for which the Gaussian P0 variance matches
    the mock one: the median ratio of the P0 diagonals over `krange` (h/Mpc)."""
    n = len(k_h)
    sel = (k_h >= krange[0]) & (k_h <= krange[1])
    if not sel.any():
        raise ValueError(f"No k bins in --cov_match_krange {krange}.")
    return float(np.median(np.diag(cov_unit)[:n][sel] / np.diag(cov_mocks)[:n][sel]))


def write_synthetic(outdir, mocktype_out, label, mocktype, region, dkP, pk, kminP, kmaxP,
                    data_dir, dkB=None, bk=None, kminB=None, kmaxB=None):
    """Write the model, given on the full k grid of the original file, in the
    cached-file format (NaN outside [kmin, kmax]), keeping its zeff/nbar header."""
    fn_in = get_fn(data_dir, 'pk', mocktype, label, region, dkP=dkP)
    with open(fn_in) as f:
        header = ''.join(line[2:] for line in f.readlines()[:3])
    k = np.loadtxt(fn_in)[:, 0]
    cols = [k]
    for i, seg in enumerate(pk):
        cols.append(np.where((k >= kminP[i]) & (k <= kmaxP[i]), seg, np.nan))
    np.savetxt(get_fn(outdir, 'pk', mocktype_out, label, region, dkP=dkP), np.column_stack(cols), header=header.rstrip('\n'))

    if bk is not None:
        fn_in = get_fn(data_dir, 'bk', mocktype, label, region, dkP=None, dkB=dkB)
        k1k2 = np.loadtxt(fn_in)[:, :2]
        cols = [k1k2[:, 0], k1k2[:, 1]]
        for i, seg in enumerate(bk):
            cols.append(np.where(np.all((k1k2 >= kminB[i]) & (k1k2 <= kmaxB[i]), axis=1), seg, np.nan))
        np.savetxt(get_fn(outdir, 'bk', mocktype_out, label, region, dkP=None, dkB=dkB), np.column_stack(cols),
                   header='k1\tk2\tB000\tB202')


def write_identity_window(outdir, mocktype_out, label, region, dkP, k):
    """Identity P(k) window on the data k grid `k` (h/Mpc) for all of P0, P2,
    P4, so that the fit evaluates the theory at the data k, unconvolved."""
    np.savetxt(get_fn(outdir, 'window_pk', mocktype_out, label, region, dkP=dkP), np.eye(len(ELLP_ALL) * len(k)))
    np.savetxt(get_fn(outdir, 'window_pk_k', mocktype_out, label, region, dkP=dkP), k)


def link_windows_and_covs(outdir, data_dir, mocktype_out, label, mocktype, mocktype_cov, region, dkP, dkB=None,
                          windows=True, covs=True):
    """Symlink this tracer's windows (renamed to `mocktype_out`) and/or
    covariances, so that `outdir` can be passed as --data_dir to the fit."""
    links = {}
    if windows:
        for kind in ('window_pk', 'window_pk_k'):
            links[get_fn(data_dir, kind, mocktype, label, region, dkP=dkP)] = get_fn(outdir, kind, mocktype_out, label, region, dkP=dkP)
    cov_patterns = [get_fn(data_dir, 'cov_pk', mocktype_cov, label, region, dkP=dkP, n_mocks_cov='*')] if covs else []
    if dkB is not None:
        for kind in ('window_bk', 'window_bk_k'):
            for slice_winB in (None, 2):
                links[get_fn(data_dir, kind, mocktype, label, region, dkP=None, dkB=dkB, slice_winB=slice_winB)] = \
                    get_fn(outdir, kind, mocktype_out, label, region, dkP=None, dkB=dkB, slice_winB=slice_winB)
        cov_patterns.append(get_fn(data_dir, 'cov_pk_bk', mocktype_cov, label, region, dkP=dkP, dkB=dkB, n_mocks_cov='*'))
    for pattern in cov_patterns:
        for src in data_dir.glob(pattern.name):
            links[src] = outdir / src.name

    for src, dst in links.items():
        # dst.resolve() == src.resolve() also when dst *is* src (outdir ==
        # data_dir): never unlink the original file
        if not src.exists() or dst.resolve() == src.resolve():
            continue
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        # relative, so the link still works when data/ is copied elsewhere
        dst.symlink_to(os.path.relpath(src.resolve(), dst.parent.resolve()))


def generate(fn, reference, args, outdir, mocktype_out):
    """Write the synthetic data of the tracers of one chain / best-fit file,
    given its load_reference output. Returns its manifest entry and
    (labels, observables, models) for plotting."""
    means, attrs, kind = reference
    run = get_run_args(attrs)
    data_dir = get_data_dir(args, attrs)

    observables, labels = build_full_range_observables(run, attrs, data_dir, no_window=args.no_window)
    b1_ref, sigmaR_ref, sigma1_eff, fsat = fc.get_prior_refs(
        [fc.tracer_label_dict[l]['tracer'] for l in labels], [fc.tracer_label_dict[l]['zrange'] for l in labels])
    z_array = np.array([obs.cosmo_fid['z'] for obs in observables])
    pars = fc.build_pars(run, b1_ref, sigmaR_ref, sigma1_eff, fsat, z_array)
    # No analytic marginalisation: the linear parameters are set to their
    # reference values (the conditional draws stored in the chain, or the
    # conditional MAP stored in the best fit) like the rest.
    likelihood = Likelihood(observables, pars, am_params=None)

    full_dict = get_true_params(pars, means, abacus_cosmo=args.abacus_cosmo, nuisance_basis=args.nuisance_basis)
    if args.zero_ct_np2:
        zero_params(full_dict, ZERO_CT_NP2)
    models = likelihood.emu.predict(likelihood.observables, pars.get_comet_dict(full_dict), de_model=likelihood.de_model)

    mocktypes = dict(zip(run.tracer_label, attrs['mocktypes_used']))
    mocktypes_cov = dict(zip(run.tracer_label, attrs['mocktypes_cov_used']))
    dkP = float(attrs['dkP'])
    dkB = float(attrs['dkB']) if run.bispec else None
    kminP, kmaxP = _per_ell(args.kminP, 3), _per_ell(args.kmaxP, 3)
    kminB, kmaxB = _per_ell(args.kminB, 2), _per_ell(args.kmaxB, 2)
    volumes = {}
    for label, obs, model in zip(labels, likelihood.observables, models):
        if not np.all(np.isfinite(model)):
            raise RuntimeError(f"Non-finite model for {label}.")
        pk, bk = to_h_units(obs, model)
        write_synthetic(outdir, mocktype_out, label, mocktypes[label], run.region, dkP, pk, kminP, kmaxP,
                        data_dir, dkB=dkB, bk=bk, kminB=kminB, kmaxB=kmaxB)
        h3 = obs.h_fid**3 if obs.Mpc_units else 1.
        k_h = obs.x[0] / obs.h_fid if obs.Mpc_units else obs.x[0]
        if args.no_window:
            write_identity_window(outdir, mocktype_out, label, run.region, dkP, k_h)
        if args.gaussian_cov:
            cov_unit = gaussian_cov_unit_volume(likelihood.emu, obs, model, dkP)
            if args.cov_volume == 'desi_dr2_veff':
                volume, nP = volume_from_veff(DESI_DR2_VEFF_GPC3[label], obs, model)
                info = f"V_eff = {DESI_DR2_VEFF_GPC3[label]} Gpc^3, nP(k={VEFF_KREF}, mu={VEFF_MUREF}) = {nP:.3f}, "
            elif args.cov_volume == 'geometric':
                volume, nP = geometric_volume(DESI_DR2_AREA_DEG2[label], fc.tracer_label_dict[label]['zrange'], obs), None
                info = f"geometric, {DESI_DR2_AREA_DEG2[label]} deg^2, "
            else:
                volume, nP = matched_volume(cov_unit, obs.cov, k_h, args.cov_match_krange), None
                info = "matched to the mocks, "
            # shown with the Gaussian errors in the --plot_fn plots
            obs.cov = cov_unit / volume
            np.savetxt(get_fn(outdir, 'cov_pk', mocktype_out, label, run.region, dkP=dkP, n_mocks_cov=0),
                       obs.cov * h3**2)
            volumes[label] = {'V_Gpch3': volume * h3 / 1e9}
            if args.cov_volume == 'desi_dr2_veff':
                volumes[label] |= {'Veff_Gpc3': DESI_DR2_VEFF_GPC3[label], 'nP_ref': nP}
            elif args.cov_volume == 'geometric':
                volumes[label]['area_deg2'] = DESI_DR2_AREA_DEG2[label]
            print(f"[{label}] Gaussian covariance: {info}V = {volumes[label]['V_Gpch3']:.3f} (Gpc/h)^3")
        link_windows_and_covs(outdir, data_dir, mocktype_out, label, mocktypes[label], mocktypes_cov[label], run.region,
                              dkP, dkB=dkB, windows=not args.no_window, covs=not args.gaussian_cov)
        print(f"[{label}] z={obs.cosmo_fid['z']:.3f}: wrote synthetic {'pk+bk' if run.bispec else 'pk'} for mocktype {mocktype_out}")

    true_params = {p: float(full_dict[p]) for p in pars.sampled_param_names + nuisance_names(pars) if p in full_dict}
    cosmo_true = {p: float(full_dict[p]) for p in fc.get_cosmo_params_to_plot(SimpleNamespace(**{**vars(run), 'fix_cosmo': None}))
                  if p in full_dict}
    source = {
        'file': str(Path(fn).resolve()),
        'reference': 'posterior mean of the chain' if kind == 'chain' else 'Minuit best fit',
        'tracers': labels,
        'source_mocktypes': mocktypes,
        'data_dir': str(data_dir),
        'fit_units': 'Mpc/h' if run.mpc_h else 'Mpc',
        'model': run.model, 'de_model': run.de_model, 'reparam': run.reparam, 'zeff_choice': run.zeff_choice,
        'fix_cosmo': run.fix_cosmo,
        'z': [float(z) for z in z_array],
        'cosmo': cosmo_true,
        # parameter names follow the source fit: no z-bin index for a
        # single-tracer fit, '_<i>' (i-th tracer by redshift) for a joint one
        'true_params': true_params,
    }
    if args.gaussian_cov:
        source['gaussian_cov_volumes'] = volumes
    return source, (labels, likelihood.observables, models)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', type=str, nargs='+',
                        help="fit_cutsky_abacushf.py output(s): Nautilus .h5 chains and/or Minuit best-fit .h5 files, "
                             "e.g. one single-tracer best fit per tracer.")
    parser.add_argument('--outdir', type=str, required=True, help="Directory for the synthetic data (usable as --data_dir).")
    parser.add_argument('--mocktype_out', type=str, default=None,
                        help="Mocktype name of the synthetic data files (the fit's --mocktype). "
                             "Default: 'synth-chainmean' (chains), 'synth-bestfit' (best fits) or 'synth-mixed', "
                             "or 'synth-abacus<cosmo>-<nuisance_basis>' with --abacus_cosmo.")
    parser.add_argument('--abacus_cosmo', type=str, nargs='?', const='c000', default=None,
                        help="Use this AbacusSummit cosmology (default when given without a value: c000) as the true "
                             "cosmology, taking only the nuisance parameters from the chain / best fit. For a "
                             "--fix_cosmo fit, it replaces the fixed cosmology.")
    parser.add_argument('--nuisance_basis', type=str, default='physical', choices=['physical', 'sampled'],
                        help="Only used with --abacus_cosmo. 'physical': keep the physical nuisance parameters (b1, a0, ...) "
                             "of the reference point. 'sampled': keep the sampled, possibly reparametrised ones (b1_r, a0_r, ...), "
                             "so the physical values are rescaled by the new sigma_R/q_iso.")
    parser.add_argument('--zero_ct_np2', action='store_true',
                        help="Set the counterterms (a0, a2, a4 / c0, c2, c4, and cnlo) and the NP20, NP22 stochastic "
                             "terms to zero, keeping the rest of the reference parameters.")
    parser.add_argument('--kminP', type=float, nargs='*', default=[0.], help="Scalar or per-ell (0, 2, 4). Default: full range.")
    parser.add_argument('--kmaxP', type=float, nargs='*', default=[np.inf], help="Scalar or per-ell (0, 2, 4). Default: full range.")
    parser.add_argument('--kminB', type=float, nargs='*', default=[0.], help="Scalar or per-ell (000, 202). Default: full range.")
    parser.add_argument('--kmaxB', type=float, nargs='*', default=[np.inf], help="Scalar or per-ell (000, 202). Default: full range.")
    parser.add_argument('--data_dir', type=str, default=None, help="Cached data the fits were run on. Default: each file's data_dir, "
                        "else env.DATA_DIR_CUTSKY.")
    parser.add_argument('--no_window', action='store_true',
                        help="P(k) only. No window: the model is the theory at the data k, and identity window "
                             "matrices are written instead of linking the data ones.")
    parser.add_argument('--gaussian_cov', action='store_true',
                        help="Requires --no_window. Write COMET Gaussian covariances (at the true parameters) "
                             "instead of linking the mock ones; fit with --mocktype_cov <mocktype_out>.")
    parser.add_argument('--cov_volume', type=str, default='match_mocks', choices=['match_mocks', 'geometric', 'desi_dr2_veff'],
                        help="Only used with --gaussian_cov. 'match_mocks': volume for which the P0 variance matches "
                             "the mock covariance over --cov_match_krange. 'geometric': comoving volume of the DESI DR2 "
                             "footprint over the tracer's redshift range (GCcomb only). 'desi_dr2_veff': from the DESI "
                             "DR2 effective volumes (GCcomb only).")
    parser.add_argument('--cov_match_krange', type=float, nargs=2, default=[0.02, 0.2],
                        help="k range (h/Mpc) over which the Gaussian P0 variance is matched to the mock one "
                             "to set each tracer's volume. Only used with --cov_volume match_mocks.")
    parser.add_argument('--plot_fn', type=str, default=None, help="If given, plot the synthetic data over the original data, "
                        "one file per tracer, named <plot_fn stem>_<tracer><ext>.")
    args = parser.parse_args()
    for name, n in (('kminP', 3), ('kmaxP', 3), ('kminB', 2), ('kmaxB', 2)):
        if len(getattr(args, name)) not in (1, n):
            parser.error(f"--{name} takes 1 (all multipoles) or {n} (one per multipole) values, "
                         f"got {len(getattr(args, name))}.")
    if args.gaussian_cov and not args.no_window:
        parser.error("--gaussian_cov requires --no_window (the Gaussian covariance ignores the window).")

    # Check the files are compatible before writing anything
    references = {fn: load_reference(fn) for fn in args.files}
    outdir = Path(args.outdir)
    seen, shared = {}, None
    for fn, (_, attrs, _) in references.items():
        for label in attrs['tracer_label']:
            if label in seen:
                parser.error(f"Tracer {label} is in both {seen[label]} and {fn}.")
            seen[label] = fn
        shared_i = get_shared_settings(attrs)
        if shared is None:
            shared = shared_i
        elif shared_i != shared:
            parser.error(f"{fn} has settings {shared_i}, incompatible with the previous file(s) {shared}.")
        if outdir.resolve() == get_data_dir(args, attrs).resolve():
            parser.error(f"--outdir is the data directory {fn} was fitted on; the synthetic data and its links "
                         "would overwrite the original files. Use a separate directory.")
    if args.gaussian_cov and args.cov_volume != 'match_mocks' and shared['region'] != 'GCcomb':
        parser.error(f"--cov_volume {args.cov_volume} uses full-footprint DESI DR2 numbers, so it needs GCcomb fits; "
                     "use --cov_volume match_mocks for NGC/SGC.")
    if shared['bispec'] and (args.no_window or args.gaussian_cov):
        parser.error("--no_window and --gaussian_cov are only supported for P(k) fits (no --bispec).")

    mocktype_out = args.mocktype_out
    if mocktype_out is None:
        if args.abacus_cosmo is not None:
            mocktype_out = f'synth-abacus{args.abacus_cosmo}-{args.nuisance_basis}'
        else:
            kinds = {kind for _, _, kind in references.values()}
            mocktype_out = {frozenset({'chain'}): 'synth-chainmean', frozenset({'bestfit'}): 'synth-bestfit'}.get(
                frozenset(kinds), 'synth-mixed')
        if args.zero_ct_np2:
            mocktype_out += '-noctnp2'
        if args.no_window:
            mocktype_out += '-nowin'
        if args.gaussian_cov:
            mocktype_out += {'match_mocks': '-gausscov', 'geometric': '-gausscovgeom', 'desi_dr2_veff': '-gausscovveff'}[args.cov_volume]
    outdir.mkdir(parents=True, exist_ok=True)

    sources, to_plot = [], []
    for fn in args.files:
        print(f"Processing {fn}")
        source, plot_data = generate(fn, references[fn], args, outdir, mocktype_out)
        sources.append(source)
        to_plot.append(plot_data)

    cosmos = [s['cosmo'] for s in sources]
    if any(c.keys() != cosmos[0].keys() or not np.allclose(list(c.values()), list(cosmos[0].values()), rtol=1e-6, atol=0)
           for c in cosmos[1:]):
        print("WARNING: the files have different reference cosmologies, so the synthetic data of the tracers "
              "are not generated at a common cosmology. Use --abacus_cosmo to impose one.")

    manifest = {
        'mocktype': mocktype_out,
        'tracers': [label for s in sources for label in s['tracers']],
        'region': shared['region'],
        'statistic': 'pk+bk' if shared['bispec'] else 'pk',
        'dkP': shared['dkP'], 'kminP': _per_ell(args.kminP, 3), 'kmaxP': _per_ell(args.kmaxP, 3),
        **({'dkB': shared['dkB'], 'kminB': _per_ell(args.kminB, 2), 'kmaxB': _per_ell(args.kmaxB, 2)} if shared['bispec'] else {}),
        'units': 'Mpc/h (as the cached data)',
        'abacus_cosmo': args.abacus_cosmo,
        'nuisance_basis': args.nuisance_basis if args.abacus_cosmo else None,
        'zeroed_params': list(ZERO_CT_NP2) if args.zero_ct_np2 else [],
        'window': 'none (identity)' if args.no_window else 'data',
        'covariance': 'mocks' if not args.gaussian_cov else {
            'match_mocks': f"COMET Gaussian, volume matched to the mocks' P0 variance over k in {args.cov_match_krange} h/Mpc",
            'geometric': "COMET Gaussian, comoving volume of the DESI DR2 footprint",
            'desi_dr2_veff': f"COMET Gaussian, V = V_eff (1 + 1/(nP))^2 from the DESI DR2 V_eff, P(k={VEFF_KREF}, mu={VEFF_MUREF})",
        }[args.cov_volume],
        'sources': sources,
        'created': datetime.date.today().isoformat(),
    }
    fn_manifest = outdir / f'manifest_{mocktype_out}.json'
    with open(fn_manifest, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {fn_manifest}")

    if args.plot_fn:
        import model_eval as me
        stem, ext = os.path.splitext(args.plot_fn)
        for labels, observables, models in to_plot:
            for label, obs, model in zip(labels, observables, models):
                me.plot_model_over_data(obs, model, h_units=True)
                plt.gcf().suptitle(label)
                plt.savefig(f'{stem}_{label}{ext or ".png"}', bbox_inches='tight')
                plt.close()
        print(f"Plots saved to {stem}_<tracer>{ext or '.png'}")

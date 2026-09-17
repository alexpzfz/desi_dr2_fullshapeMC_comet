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
sys.path.insert(0, "/global/homes/a/alexpzfz/comet-emu")
from observables import PowerSpectrumMultipoles, BispectrumSugiyamaMultipoles, JointObservable
from params import Params
from likelihood import Likelihood
from samplers import NautilusSampler, MinuitMinimizer
from theory import COMET
from read_data import get_obs_pk, get_obs_pk_bk
from priors_mc import get_pars
import plot_utils as pu




tracer_label_dict = {'BGS': {'tracer': 'BGS', 'zrange': (0.1, 0.4)},
                     'LRG1': {'tracer': 'LRG', 'zrange': (0.4, 0.6)},
                     'LRG2': {'tracer': 'LRG', 'zrange': (0.6, 0.8)},
                     'LRG3': {'tracer': 'LRG', 'zrange': (0.8, 1.1)},
                     'ELG1': {'tracer': 'ELG', 'zrange': (0.8, 1.1)},
                     'ELG2': {'tracer': 'ELG', 'zrange': (1.1, 1.6)},
                     'QSO': {'tracer': 'QSO', 'zrange': (0.8, 2.1)}}

zsnap_dict = {'BGS': {(0.1, 0.4): 0.300},
              'LRG': {(0.4, 0.6): 0.500, (0.6, 0.8): 0.725, (0.8, 1.1): 0.950},
              'ELG': {(0.8, 1.1): 0.950, (1.1, 1.6): 1.475},
              'QSO': {(0.8, 2.1): 1.550}}


def _fmt_float(x):
    return f"{float(x):.3f}".rstrip('0').rstrip('.')


def get_fn(tracer_label, region, freedom, dkP, kmaxP, bispec=False, dkB=None, kmaxB=None,
           de_model='lambda', reparam_option='full', free_Mnu=False, outdir=str(env.CHAINS_DIR), extra=None,
           counterterm_basis='DESIct', avirB_free=False, use_Mpc=True):

    if not isinstance(tracer_label, list):
        tracer_label = [tracer_label] 

    if isinstance(kmaxP, list):
        kmaxP_str = '-'.join([_fmt_float(k) for k in kmaxP])
    else:
        kmaxP_str = _fmt_float(kmaxP)

    tracer_str = '-'.join(tracer_label) if len(tracer_label) > 1 else tracer_label[0]
    
    fn = f'{outdir}/Abacus-hf-dr2-v2-altmtl_{tracer_str}_{region}_{freedom}freedom_{de_model}_pk_dk{_fmt_float(dkP)}_kmax{kmaxP_str}'
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
        if avirB_free:
            fn += '_avirBfree'
    if counterterm_basis != 'DESIct':
        fn += f'_ct{counterterm_basis}'
    if not use_Mpc:
        fn += '_Mpch'
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
        'BGS': {(0.1, 0.4): 0.694},
        'LRG': {(0.4, 0.6): 0.62469875, (0.6, 0.8): 0.56705006, (0.8, 1.1): 0.51262079},
        'ELG_LOP': {(0.8, 1.1): 0.5050584, (1.1, 1.6): 0.43297989},
        'QSO': {(0.8, 2.1): 0.40714599},
    }
    b1_ref_dict = {
        'BGS': {(0.1, 0.4): 1.541},
        'LRG': {(0.4, 0.6): 1.883, (0.6, 0.8): 2.031, (0.8, 1.1): 2.216},
        'ELG_LOP': {(0.8, 1.1): 1.187, (1.1, 1.6): 1.429},
        'QSO': {(0.8, 2.1): 2.266},
    }
    fsat_dict = {'BGS': 0.13, 'LRG': 0.13, 'ELG_LOP': 0.06, 'QSO': 0.2}

    b1_ref = []
    sigmaR_ref = []
    fsat = []
    sigma1_eff = []
    for tracer, zr in zip(tracer_list, zrange_list):
        key = _tracer_key(tracer)
        b1_ref.append(b1_ref_dict[key][tuple(zr)])
        sigmaR_ref.append(sigmaR_ref_dict[key][tuple(zr)])
        fsat.append(fsat_dict[key])
        if key == 'BGS':
            sigma1_eff.append(150/70 * 10**(1/3) * np.sqrt(1 + 0.2))
        elif key == 'LRG':
            sigma1_eff.append(150/70 * 10**(1/3) * np.sqrt(1 + 0.8))
        elif key == 'ELG_LOP':
            sigma1_eff.append(150/70 * 2.1 ** (1/2))
        else:
            sigma1_eff.append(150/70 * 10**(0.7/3) * 2.4 ** (1/2))

    return np.array(b1_ref, dtype=float), np.array(sigmaR_ref, dtype=float), np.array(sigma1_eff, dtype=float), np.array(fsat, dtype=float)


def build_pars(args, b1_ref, sigmaR_ref, sigma1_eff, fsat, z_array):
    """Construct a fresh Params object (with its own COMET emulator instance)."""
    pars = get_pars(bispec=args.bispec, de_model=args.de_model, reparam_option=args.reparam,
                     freedom=args.freedom, free_Mnu=args.free_Mnu,
                     b1_ref=b1_ref, sigmaR_ref=sigmaR_ref, sigma1_eff=sigma1_eff, fsat=fsat,
                     z_array=z_array, counterterm_basis=args.counterterm_basis,
                     avirB_free=args.avirB_free, use_Mpc=not args.mpc_h)
    if args.bispec:
        pars.emu.bispec_kwargs['sugiyama']['quad_deg'] = (7, 16, 5)
        pars.emu.bispec_kwargs['sugiyama']['mu12_transform'] = 'k3'
        pars.emu.BispNum.backend = 'jax'
    pars.emu.use_interp_kwin = True
    return pars


def get_cosmo_params_to_plot(args):
    """Names of the sampled/exported cosmological parameters for this fit,
    in the order they should appear in the triangle plot."""
    params = ['wb', 'wc', 'h', 'ns', 'log10As']
    if args.de_model in ('w0', 'w0wa'):
        params.append('w0')
    if args.de_model == 'w0wa':
        params.append('wa')
    if args.free_Mnu:
        params.append('Mnu')
    return params


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
    parser.add_argument('--mocktype', type=str, default=None, help="Override the data mock type. "
                        "Default: per-tracer (abacus-hf-dr2-v2-altmtl, or abacus-2ndgen-dr2-altmtl for BGS). "
                        "Set it to fit an alternative data set, e.g. a blinded one.")
    parser.add_argument('--mocktype_cov', type=str, default=None, help="Override the mock type of the "
                        "covariance. Default: per-tracer (holi-v3-altmtl, or holi-bgs-altmtl for BGS).")
    parser.add_argument('--data_dir', type=str, default=None, help="Directory holding the cached data, "
                        "covariance and window files. Default: the location read_data.py points at.")
    parser.add_argument('--ellP', type=int, nargs='*', default=[0, 2])
    parser.add_argument('--kminP', type=float, default=0.02, nargs='*')
    parser.add_argument('--kmaxP', type=float, default=0.3, nargs='*')
    parser.add_argument('--ellwinP', type=int, nargs='*', default=[0, 2, 4])
    parser.add_argument('--kwinminP', type=float, default=0.0015, nargs='*')
    parser.add_argument('--kwinmaxP', type=float, default=0.5, nargs='*')
    parser.add_argument('--dkP', type=float, default=0.005)
    parser.add_argument('--bispec', action='store_true')
    parser.add_argument('--avirB_free', action='store_true', help="Sample avirB as a free parameter instead of tying it to avir (the default when --bispec is set).")
    parser.add_argument('--ellB', type=_parse_tuple_ell, nargs='*', default=[(0, 0, 0), (2, 0, 2)])
    parser.add_argument('--kminB', type=float, default=0.02, nargs='*')
    parser.add_argument('--kmaxB', type=float, default=0.2, nargs='*')
    parser.add_argument('--ellwinB', type=_parse_tuple_ell, nargs='*', default=[(0, 0, 0), (0, 2, 2)])
    parser.add_argument('--kwinminB', type=float, default=0.0015, nargs='*')
    parser.add_argument('--kwinmaxB', type=float, default=0.3, nargs='*')
    parser.add_argument('--dkB', type=float, default=0.005)
    parser.add_argument('--reparam', type=str, default='full', choices=['full', 'hybrid', 'none'])
    parser.add_argument('--de_model', type=str, default='lambda', choices=['lambda', 'w0wa', 'w0'])
    parser.add_argument('--counterterm_basis', type=str, default='DESIct', choices=['DESIct', 'Comet'])
    parser.add_argument('--freedom', type=str, default='interm', choices=['min', 'max', 'interm'])
    parser.add_argument('--free_Mnu', action='store_true')
    parser.add_argument('--mpc_h', action='store_true', help="Run the fit in Mpc/h units instead of Mpc. Switches the reparametrization to use sigma8 instead of sigma12.")
    parser.add_argument('--outdir', type=str, default=str(env.CHAINS_DIR))
    parser.add_argument('--n_live', type=int, default=2000)
    parser.add_argument('--extra', type=str, default=None, help="Extra string to add to output filename for uniqueness (e.g. to distinguish different sampler settings).")
    parser.add_argument('--minimize', action='store_true', help="Run an iMinuit MIGRAD minimization instead of Nautilus nested sampling.")
    parser.add_argument('--hesse', action='store_true', help="Run HESSE after MIGRAD to get the covariance matrix. Only used with --minimize.")
    parser.add_argument('--seed_init', type=int, default=None, help="Random seed for drawing the Minuit starting point from the priors. Only used with --minimize.")
    parser.add_argument('--plot_contours', action='store_true', help="After the Nautilus chain finishes, plot the triangle/contour plot for the cosmological parameters and save it to --plot_dir. Not used with --minimize.")
    parser.add_argument('--plot_dir', type=str, default=str(env.PLOTS_DIR_CUTSKY_ABACUSHF), help="Directory to store the contour plot in, when --plot_contours is set.")

    args = parser.parse_args()

    os.environ['OMP_NUM_THREADS'] = '1'  # to avoid numpy multithreading issues with multiprocessing

    print(f"Fitting tracer(s) {args.tracer_label} in region {args.region}")
    if args.mocktype or args.mocktype_cov or args.data_dir:
        print(f"Data overrides: mocktype={args.mocktype}, mocktype_cov={args.mocktype_cov}, data_dir={args.data_dir}")
    print(f"Power spectrum settings: ellP={args.ellP}, kminP={args.kminP}, kmaxP={args.kmaxP}, ellwinP={args.ellwinP}, kwinminP={args.kwinminP}, kwinmaxP={args.kwinmaxP}, dkP={args.dkP}")
    if args.bispec:
        print(f"Bispectrum settings: ellB={args.ellB}, kminB={args.kminB}, kmaxB={args.kmaxB}, ellwinB={args.ellwinB}, kwinminB={args.kwinminB}, kwinmaxB={args.kwinmaxB}, dkB={args.dkB}, avirB_free={args.avirB_free}")
    print(f"DE model: {args.de_model}, reparametrization: {args.reparam}, freedom: {args.freedom}, free_Mnu: {args.free_Mnu}, counterterm_basis: {args.counterterm_basis}, units: {'Mpc/h' if args.mpc_h else 'Mpc'}")


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
    mocktypes_used = []
    mocktypes_cov_used = []
    data_dir_kw = {} if args.data_dir is None else {'outdir': args.data_dir}
    for tracer_i, zr in zip(tracer_list, zrange_list):
        if tracer_i == 'BGS':
            mocktype = 'abacus-2ndgen-dr2-altmtl'
            mocktype_cov = 'holi-bgs-altmtl'
        else:
            mocktype = 'abacus-hf-dr2-v2-altmtl'
            mocktype_cov = 'holi-v3-altmtl'
        # --mocktype / --mocktype_cov override the per-tracer defaults above
        mocktype = args.mocktype or mocktype
        mocktype_cov = args.mocktype_cov or mocktype_cov
        mocktypes_used.append(mocktype)
        mocktypes_cov_used.append(mocktype_cov)
        if not args.bispec:
            obs_i = get_obs_pk(tracer=tracer_i, zrange=zr, region=args.region, mocktype=mocktype, mocktype_cov=mocktype_cov,
                               ell=args.ellP, kmin=args.kminP, kmax=args.kmaxP, ellwin=args.ellwinP,
                               kwinmin=args.kwinminP, kwinmax=args.kwinmaxP, dk=args.dkP, use_Mpc=not args.mpc_h,
                               **data_dir_kw)
        else:
            obs_i = get_obs_pk_bk(tracer=tracer_i, zrange=zr, region=args.region, mocktype=mocktype, mocktype_cov=mocktype_cov,
                                        ellP=args.ellP, kminP=args.kminP, kmaxP=args.kmaxP, ellwinP=args.ellwinP,
                                        kwinminP=args.kwinminP, kwinmaxP=args.kwinmaxP, dkP=args.dkP,
                                        ellB=args.ellB, kminB=args.kminB, kmaxB=args.kmaxB, ellwinB=args.ellwinB,
                                        kwinminB=args.kwinminB, kwinmaxB=args.kwinmaxB, dkB=args.dkB, slice_winB_theory=2,
                                        use_Mpc=not args.mpc_h, **data_dir_kw)
        print(f"Effective redshift from geometry: {obs_i.cosmo_fid['z']:.3f}, from zrange: {zr}")
        obs_i.cosmo_fid['z'] = zsnap_dict[tracer_i][zr]
        print(f'Switched effective redshift to zsnap: {obs_i.cosmo_fid["z"]:.3f}')
        observables.append(obs_i)


    b1_ref, sigmaR_ref, sigma1_eff, fsat = get_prior_refs(tracer_list, zrange_list)
    z_array = [obs.cosmo_fid['z'] for obs in observables]
    # sort z_array
    z_array = np.array(z_array)
    sort_idx = np.argsort(z_array)
    z_array = z_array[sort_idx]
    observables = [observables[i] for i in sort_idx]

    if args.counterterm_basis == 'DESIct':
        am_params = ['btd_r', 'a0_r', 'a2_r', 'NP0_r', 'NP20_r', 'NP22_r']
    elif args.counterterm_basis == 'Comet':
        am_params = ['btd_r', 'c0_r', 'c2_r', 'NP0_r', 'NP20_r', 'NP22_r']
    if args.bispec:
        am_params += ['NB0_r', 'MB0_r']

    pars = build_pars(args, b1_ref, sigmaR_ref, sigma1_eff, fsat, z_array)

    conditional_prior_fn = None
    if args.de_model == 'w0wa':
        def conditional_prior_fn(params):
            w0 = params['w0']
            wa = params['wa']
            return w0 + wa < 0
    
    likelihood = Likelihood(observables, pars, am_params=am_params, conditional_prior=conditional_prior_fn)

    os.makedirs(args.outdir, exist_ok=True)
    fn = get_fn(tracer_label=args.tracer_label, region=args.region,  freedom=args.freedom, dkP=args.dkP, kmaxP=args.kmaxP,
                bispec=args.bispec, dkB=args.dkB, kmaxB=args.kmaxB,
                de_model=args.de_model, reparam_option=args.reparam, free_Mnu=args.free_Mnu, outdir=args.outdir, extra=args.extra,
                counterterm_basis=args.counterterm_basis, avirB_free=args.avirB_free, use_Mpc=not args.mpc_h)
    if args.minimize:
        # Stage 1: minimize with analytical marginalisation (AM) of the
        # linear nuisance parameters, and save the resulting MAP.
        minimizer = MinuitMinimizer(likelihood, seed_init=args.seed_init, verbose=True)

        t0 = time.time()
        print("Starting minimization with analytical marginalisation...")

        minimizer.run(hesse=args.hesse, verbose=True)
        t1 = time.time()
        print(f"AM minimization finished in {(t1-t0)/60:.2f} minutes.")

        best_fit_am, uncertainties_am = minimizer.get_map(return_am=True)

        fn_minuit_am = fn + '_minuit_am'
        run_metadata_am = {**vars(args), 'mocktypes_used': mocktypes_used, 'mocktypes_cov_used': mocktypes_cov_used,
                           'elapsed_minutes': (t1 - t0) / 60, 'stage': 'analytical_marginalisation'}
        minimizer.save(fn_minuit_am, best_fit=best_fit_am, uncertainties=uncertainties_am, metadata=run_metadata_am)
        print(f"AM best-fit result saved to {fn_minuit_am}")

        # Stage 2: use the AM MAP (including the conditional MAP of the
        # marginalised parameters) as the starting point for a minimization
        # of the full likelihood, where those parameters are sampled directly
        # instead of analytically marginalised.
        pars_full = build_pars(args, b1_ref, sigmaR_ref, sigma1_eff, fsat, z_array)
        likelihood_full = Likelihood(observables, pars_full, am_params=None, conditional_prior=conditional_prior_fn)

        minimizer_full = MinuitMinimizer(likelihood_full, seed_init=args.seed_init, verbose=True)
        minimizer_full.set_starting_point(best_fit_am, errors_dict=uncertainties_am, reset_errors=True)

        t0 = time.time()
        print("Starting full-likelihood minimization from the AM MAP...")

        minimizer_full.run(hesse=args.hesse, verbose=True)
        t1 = time.time()
        print(f"Full-likelihood minimization finished in {(t1-t0)/60:.2f} minutes.")

        fn_minuit = fn + '_minuit_full'
        run_metadata = {**vars(args), 'mocktypes_used': mocktypes_used, 'mocktypes_cov_used': mocktypes_cov_used,
                         'elapsed_minutes': (t1 - t0) / 60, 'stage': 'full_likelihood',
                         'am_map_file': fn_minuit_am}
        minimizer_full.save(fn_minuit, metadata=run_metadata)
        print(f"Full-likelihood best-fit result saved to {fn_minuit}")
    else:
        fn_snap = fn + '_snap.hdf5'

        # get available cores depending on environment (e.g. SLURM_CPUS_PER_TASK for slurm, or default to os.cpu_count())
        n_threads = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))
        print(f"Using {n_threads} threads for sampling.")
        print(f"Pool size for Nautilus sampler: {n_threads}")

        sampler = NautilusSampler(likelihood, n_live=args.n_live, filepath=fn_snap, pool=n_threads)

        # run sampler
        t0 = time.time()
        print("Starting sampler...")

        sampler.sample(verbose=True, discard_exploration=True)
        t1 = time.time()
        print(f"Sampler finished in {(t1-t0)/60:.2f} minutes.")
        run_metadata = {**vars(args), 'mocktypes_used': mocktypes_used, 'mocktypes_cov_used': mocktypes_cov_used,
                         'n_threads': n_threads, 'pool_size': n_threads,
                         'discard_exploration': True, 'elapsed_minutes': (t1 - t0) / 60}
        sampler.save(fn, metadata=run_metadata)
        print(f"Chain saved to {fn}")

        if args.plot_contours:
            print("Plotting contours for the cosmological parameters...")
            try:
                samples = pu.get_samples(fn + '.h5')
                cosmo_params = get_cosmo_params_to_plot(args)
                try:
                    g = pu.plot_triangle(samples, params_to_plot=cosmo_params, filled=False)
                except (RuntimeError, FileNotFoundError):
                    # plot_utils sets rcParams['text.usetex'] = True at import
                    # time, which needs a working LaTeX install; fall back to
                    # matplotlib's own mathtext if that's not available here.
                    plt.rc('text', usetex=False)
                    g = pu.plot_triangle(samples, params_to_plot=cosmo_params, filled=False)
                os.makedirs(args.plot_dir, exist_ok=True)
                plot_fn = os.path.join(args.plot_dir, f"{Path(fn).name}_contours.png")
                g.fig.savefig(plot_fn, dpi=150, bbox_inches='tight')
                plt.close(g.fig)
                print(f"Contour plot saved to {plot_fn}")
            except Exception as e:
                # The chain itself is already saved at this point; don't let a
                # plotting failure look like the whole run failed.
                print(f"Warning: failed to plot contours ({e}). Chain is still saved at {fn}.")

        # clean up
        os.remove(fn_snap)


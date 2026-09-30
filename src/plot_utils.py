import numpy as np
import matplotlib.pyplot as plt
from getdist import plots, MCSamples
import scipy.stats as stats
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=12)

cosmo_dict = {'c000': {'wb': 0.02237, 'wc': 0.1200, 'h': 0.6736, 'As': 2.083, 'ns': 0.9649, 'Mnu': 0.06, 'w0': -1.0, 'wa': 0.0},
              'c001': {'wb': 0.02242, 'wc': 0.1134, 'h': 0.7030, 'As': 2.037, 'ns': 0.9638, 'Mnu': 0.06, 'w0': -1.0, 'wa': 0.0},
              'c002': {'wb': 0.02237, 'wc': 0.1200, 'h': 0.6278, 'As': 2.314, 'ns': 0.9649, 'Mnu': 0.06, 'w0': -0.7, 'wa': -0.5},
              'c004': {'wb': 0.02237, 'wc': 0.1200, 'h': 0.6736, 'As': 1.7949, 'ns': 0.9649, 'Mnu': 0.06, 'w0': -1.0, 'wa': 0.0}}
cosmo_list = ['c000', 'c001', 'c002', 'c004']
for c in cosmo_list:
    cosmo_dict[c]['log10As'] = np.log(1e10 * cosmo_dict[c]['As'] * 1e-9)

# True Omega_m/sigma8 for each fiducial cosmology, so that plot_fob_fom (which
# looks these up from cosmo_dict) also works for the {h, Omega_m, sigma8}
# parameter space, not just the directly-sampled cosmological parameters.
from abacus_cosmologies import get_abacus_cosmology
for c in cosmo_list:
    _abacus = get_abacus_cosmology(c)
    cosmo_dict[c]['Omega_m'] = _abacus['Omega_m']
    cosmo_dict[c]['sigma8'] = _abacus['sigma8_cb']

colors = {'c000': 'blue', 'c001': 'orange', 'c002': 'green', 'c004': 'red'}
tracer_colors = {('BGS_ANY-02', (0.1, 0.4)):'yellowgreen','BGS': 'yellowgreen', 'BGS_BRIGHT-21.5': 'yellowgreen', ('BGS_BRIGHT-21.5', (0.1, 0.4)): 'yellowgreen', ('BGS_BRIGHT-21.35', (0.1, 0.4)): 'yellowgreen',
'LRG': 'red', ('LRG', (0.4, 0.6)): 'orange', ('LRG', (0.6, 0.8)): 'orangered', ('LRG', (0.8, 1.1)): 'firebrick', 
'LRG+ELG': 'slateblue', 'LRGplusELG': 'slateblue', ('LRG+ELG_LOPnotqso', (0.8, 1.1)): 'slateblue', 'ELG': 'blue', ('ELG', (0.8, 1.1)): 'skyblue', ('ELG', (1.1, 1.6)): 'steelblue', 'QSO': 'seagreen', ('QSO', (0.8, 2.1)): 'seagreen', 'Lya': 'purple', ('Lya', (0.8, 3.5)): 'purple', ('Lya', (1.8, 4.2)): 'purple'}

tracer_labels = {'BGS_BRIGHT-21.35': {(0.1, 0.4): r'$\tt BGS$'},
                 'LRG': {(0.4, 0.6): r'$\tt LRG1$', (0.6, 0.8): r'$\tt LRG2$', (0.8, 1.1): r'$\tt LRG3$'},
                 'LRG+ELG_LOPnotqso': {(0.8, 1.1): r'$\tt LRG3+ELG1$'},
                 'ELG': {(0.8, 1.1): r'$\tt ELG1$', (1.1, 1.6): r'$\tt ELG2$'},
                 'QSO': {(0.8, 2.1): r'$\tt QSO$'}}


def correct_labels(names, labels):
    for i, name in enumerate(names):
        if name == 'NP0':
            label = r'N^P_0'
        elif name == 'NP20':
            label = r'N^P_{2, 0}'
        elif name == 'NP22':
            label = r'N^P_{2, 2}'
        elif name == 'NP0_r':
            label = r'\tilde{N}^P_0'
        elif name == 'NP20_r':
            label = r'\tilde{N}^P_{2, 0}'
        elif name == 'NP22_r':
            label = r'\tilde{N}^P_{2, 2}'
        elif name == 'b1_r':
            label = r'\tilde{b}_1'
        elif name == 'b2_r':
            label = r'\tilde{b}_2'
        elif name == 'bK2_r':
            label = r'\tilde{b}_{K^2}'
        elif name == 'btd_r':
            label = r'\tilde{b}_{td}'
        elif name == 'a0_r':
            label = r'\tilde{a}_0'
        elif name == 'a2_r':
            label = r'\tilde{a}_2'
        elif name == 'a4_r':
            label = r'\tilde{a}_4'
        else:
            label = labels[i]
        labels[i] = label
    return labels


def get_fob_threshold(n_sigma, k_dimensions):
    # Probability within n-sigma for a 1D Gaussian
    p = 2 * stats.norm.cdf(n_sigma) - 1
    # Find the corresponding chi-squared threshold
    return np.sqrt(stats.chi2.ppf(p, k_dimensions))

        
def get_samples(fn, drop_w0pwag0=False, return_attrs=False):
    import h5py
    with h5py.File(fn, 'r') as f:
        samples_ = f['points'][:]
        loglikes = f['log_likelihoods'][:]
        weights = np.exp(f['log_weights'][:])
        names = f['names'].asstr()[:]
        labels = f['latex_names'].asstr()[:]
        attrs = dict(f.attrs)
    names_list = names.tolist()
    preferred_order = ['wc', 'h', 'log10As', 'w0', 'wa', 'wb', 'ns']
    sorted_indices = np.argsort([preferred_order.index(name) if name in preferred_order else len(preferred_order)+names_list.index(name) for name in names])
    samples_ = samples_[:, sorted_indices]
    if drop_w0pwag0:
        w0_index = np.where(names == 'w0')[0][0]
        wa_index = np.where(names == 'wa')[0][0]
        mask = (samples_[:, w0_index] + samples_[:, wa_index]) <= 0
        samples_ = samples_[mask]
        loglikes = loglikes[mask]
        weights = weights[mask]
    names = names[sorted_indices]
    labels = labels[sorted_indices]
    labels = correct_labels(names, labels)

    samples = MCSamples(samples=samples_, loglikes=loglikes, weights=weights, names=names, labels=labels)
    if return_attrs:
        return samples, attrs

    return samples

def save_samples(samples, fn, attrs=None):
    """Write an MCSamples object to the same .h5 schema read by get_samples
    (points/log_weights/log_likelihoods/names/latex_names), so derived
    parameters added via samples.addDerived (e.g. sigma8, Omega_m) are
    persisted and the file can be reloaded with get_samples.

    If `attrs` is given (e.g. the dict returned by get_samples(..., return_attrs=True)),
    it is written back as HDF5 attrs on the file, so metadata recorded by the
    run that produced the chain (tracer, de_model, units, ...) survives a
    resave."""
    import h5py
    names = samples.getParamNames().list()
    labels = samples.getParamNames().labels()
    str_dtype = h5py.string_dtype(encoding='utf-8')
    with h5py.File(fn, 'w') as f:
        f.create_dataset('points', data=samples.samples)
        f.create_dataset('log_weights', data=np.log(samples.weights))
        f.create_dataset('log_likelihoods', data=samples.loglikes)
        f.create_dataset('names', data=names, dtype=str_dtype)
        f.create_dataset('latex_names', data=labels, dtype=str_dtype)
        for key, value in (attrs or {}).items():
            f.attrs[key] = value


def plot_triangle(samples_list, params_to_plot=None, labels=None, width_inch=14, cmap=None, settings_dict=None,
                   legend_fontsize=18, filled=False, cosmo_true='c000', extra_markers=None, **kwargs):
    g = plots.get_subplot_plotter(width_inch=width_inch)
    g.settings.figure_legend_frame = False
    g.settings.legend_fontsize = legend_fontsize
    g.settings.axes_fontsize = 15
    g.settings.axes_labelsize = 24
    # g.settings.line_styles = 'Set2'
    # g.settings.solid_colors = 'Set2'
    if settings_dict is not None:
        for key, value in settings_dict.items():
            setattr(g.settings, key, value)
    if cmap is not None:
        g.settings.line_styles = cmap
        g.settings.solid_colors = cmap
    g.settings.legend_frame = False
    markers = {k: v for k, v in cosmo_dict[cosmo_true].items()}
    if extra_markers:
        markers.update(extra_markers)
    g.triangle_plot(samples_list, params_to_plot, markers=markers, filled=filled, 
                    legend_labels=labels, **kwargs)
    return g

def _shade_sigma_regions(ax, sigma_levels, k, symmetric=False):
    """Shade nested n-sigma regions (chi-squared thresholds with k degrees
    of freedom) from widest/lightest to narrowest/darkest, so the smallest
    region is drawn last and stays visually distinct on top."""
    sigma_levels = sorted(sigma_levels)
    alphas = np.linspace(0.45, 0.12, len(sigma_levels))
    for n_sigma, alpha in sorted(zip(sigma_levels, alphas), key=lambda t: -t[0]):
        thr = get_fob_threshold(n_sigma, k)
        if symmetric:
            ax.axhspan(-thr, thr, color='grey', alpha=alpha, zorder=0)
        else:
            ax.axhspan(0, thr, color='grey', alpha=alpha, zorder=0)


def _param_label(samples, param):
    return samples.getParamNames().labels()[samples.index[param]]


def plot_fob_fom(samples_lists, params_to_plot, xlabels, samples_labels=None, colors=None, cosmo_true='c000',
                  sigma_levels=(0.5, 1.0, 2.0), pull_params=None, bar_width=0.8):
    """
    Bar-chart comparison, per tracer (xlabels), of:
      - FoB and FoM jointly defined on params_to_plot, with the parameters
        they are defined on shown in the axis label (e.g. FoB(h, Omega_m,
        sigma_8)); one bar group per entry of samples_lists (e.g. one per
        unit convention),
      - the individual pull (theta - theta_true) / sigma_theta for each
        parameter in pull_params (default: params_to_plot), one panel per
        parameter.

    samples_lists is a list of groups; each group is a list of MCSamples
    aligned with xlabels (one sample set per tracer). All shaded regions
    are the n-sigma thresholds from sigma_levels, using the chi-squared
    distribution with the relevant number of degrees of freedom (len(
    params_to_plot) for FoB, 1 for each individual pull panel).
    """
    if not isinstance(samples_lists[0], list):
        samples_lists = [samples_lists]
    if pull_params is None:
        pull_params = params_to_plot

    n_groups = len(samples_lists)
    n_x = len(xlabels)
    n_rows = 2 + len(pull_params)

    fig, axes = plt.subplots(n_rows, 1, figsize=(6, 1.5 * n_rows), sharex=True, layout='constrained')

    x = np.arange(n_x)
    width = bar_width / n_groups
    offsets = (np.arange(n_groups) - (n_groups - 1) / 2) * width

    k = len(params_to_plot)
    _shade_sigma_regions(axes[0], sigma_levels, k)

    ref_samples = samples_lists[0][0]
    joint_label = ', '.join(_param_label(ref_samples, p) for p in params_to_plot)

    for i, samples_list in enumerate(samples_lists):
        fob_arr = []
        fom_arr = []
        for samples in samples_list:
            param_indices = [samples.index[param] for param in params_to_plot]
            means = samples.getMeans()[param_indices]
            cov = samples.getCov()[np.ix_(param_indices, param_indices)]
            diff = means - np.array([cosmo_dict[cosmo_true][param] for param in params_to_plot])
            fob = np.sqrt(diff @ np.linalg.inv(cov) @ diff)
            fom = 1 / np.sqrt(np.linalg.det(cov))
            fob_arr.append(fob)
            fom_arr.append(fom)
        color = colors[i] if colors else None
        label = samples_labels[i] if samples_labels else None
        axes[0].bar(x + offsets[i], fob_arr, width, color=color, label=label, zorder=2)
        axes[1].bar(x + offsets[i], fom_arr, width, color=color, label=label, zorder=2)

    # A rotated y-label can't comfortably fit the full parameter list once
    # there are more than a couple of them (e.g. the 5D w0wa case), so show
    # it as a compact horizontal title instead and keep the y-label short.
    axes[0].set_ylabel('FoB')
    axes[1].set_ylabel('FoM')
    axes[0].set_title(r'$\mathrm{FoB}(' + joint_label + r')$', fontsize=13)
    axes[1].set_title(r'$\mathrm{FoM}(' + joint_label + r')$', fontsize=13)

    for row, param in enumerate(pull_params, start=2):
        ax = axes[row]
        _shade_sigma_regions(ax, sigma_levels, k=1)
        for i, samples_list in enumerate(samples_lists):
            pulls = []
            for samples in samples_list:
                idx = samples.index[param]
                mean = samples.getMeans()[idx]
                sigma = np.sqrt(samples.getCov()[idx, idx])
                true = cosmo_dict[cosmo_true][param]
                pulls.append(abs((mean - true) / sigma))
            color = colors[i] if colors else None
            ax.bar(x + offsets[i], pulls, width, color=color, zorder=2)
        plabel = _param_label(ref_samples, param)
        ax.set_ylabel(r'$|\Delta ' + plabel + r'|/\sigma_{' + plabel + r'}$', fontsize=11)

    # labels are the ticks for the x-axis
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(xlabels, rotation=35, ha='right')

    if samples_labels:
        # 'outside upper center' reserves its own margin under a constrained
        # layout, instead of overlapping axes[0]'s title/ylabel like an
        # axes-anchored legend would.
        handles, _ = axes[0].get_legend_handles_labels()
        fig.legend(handles, samples_labels, ncol=len(samples_labels), loc='outside upper center')
    return fig, axes

def plot_relative_uncertainties(samples_lists, params_to_plot, xlabels=None, samples_labels=None, colors=None):
    fig, axes = plt.subplots(len(params_to_plot), 1, figsize=(6, 5), sharex=True, layout='constrained')
    if not isinstance(samples_lists[0], list): 
        samples_lists = [samples_lists]

    for j, samples_list in enumerate(samples_lists):
        rel_uncertainties_list = []

        for samples in samples_list:
            param_indices = [samples.index[param] for param in params_to_plot]
            params_labels = [samples.getParamNames().labels()[i] for i in param_indices]
            means = samples.getMeans()[param_indices]
            cov = samples.getCov()[np.ix_(param_indices, param_indices)]
            rel_uncertainties = np.sqrt(np.diag(cov)) / means * 100.
            rel_uncertainties_list.append(rel_uncertainties)
        for i, param in enumerate(params_to_plot):
            x = np.arange(len(samples_list))
            y = [rel_uncertainties[i] for rel_uncertainties in rel_uncertainties_list]
            axes[i].plot(x, y, marker='o', color=colors[j] if colors is not None else None)
            axes[i].set_ylabel(rf'$\sigma({params_labels[i]})$ [\%]')
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(xlabels if xlabels is not None else [f'Sample {i}' for i in x], rotation=35, ha='right')
    if samples_labels is not None:
        axes[0].legend(samples_labels, ncol=len(samples_labels)//2, loc='upper left', bbox_to_anchor=(0.2, 1.6))
    return fig, axes
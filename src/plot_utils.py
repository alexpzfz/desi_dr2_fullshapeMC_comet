import numpy as np
import matplotlib.pyplot as plt
from getdist import plots, MCSamples
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
        
def get_samples(fn, drop_w0pwag0=False):
    import h5py
    with h5py.File(fn, 'r') as f:
        samples_ = f['points'][:]
        loglikes = f['log_likelihoods'][:]
        weights = np.exp(f['log_weights'][:])
        names = f['names'].asstr()[:]
        labels = f['latex_names'].asstr()[:]
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

    return samples

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

def plot_fob_fom(samples_lists, params_to_plot, xlabels, samples_labels=None, colors=None, cosmo_true='c000'):
    if not isinstance(samples_lists[0], list):
        samples_lists = [samples_lists]
    
    fig, axes = plt.subplots(2, 1, figsize=(6, 4), sharex=True, layout='constrained')
    # add shaded region for FoB
    axes[0].axhspan(0, 1.88, color='grey', alpha=0.4)
    axes[0].axhspan(0, 2.83, color='grey', alpha=0.2)

    for i, samples_list in enumerate(samples_lists):
        fob_arr = []
        fom_arr = []
        for samples in samples_list:
            param_indices = [samples.index[param] for param in params_to_plot]
            means = samples.getMeans()[param_indices]
            cov = samples.getCov()[np.ix_(param_indices, param_indices)]
            diff = means - np.array([cosmo_dict[cosmo_true][param] for param in params_to_plot])
            fob = np.sqrt(diff @ np.linalg.inv(cov) @ diff)
            fom = 1 /np.sqrt(np.linalg.det(cov))
            fob_arr.append(fob)
            fom_arr.append(fom)
        axes[0].plot(fob_arr, marker='o', color=colors[i] if colors else None, label=samples_labels[i] if samples_labels else None)
        axes[1].plot(fom_arr, marker='o', color=colors[i] if colors else None, label=samples_labels[i] if samples_labels else None)
    
    axes[0].set_ylabel('FoB')
    axes[1].set_ylabel('FoM')
    
    # labels are the ticks for the x-axis
    axes[1].set_xticks(range(len(xlabels)))
    axes[1].set_xticklabels(xlabels, rotation=35, ha='right')

    if samples_labels:
        axes[0].legend(ncol=len(samples_labels)//2, loc='upper left', bbox_to_anchor=(0.2, 1.6))
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
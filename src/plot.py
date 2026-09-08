import numpy as np
import matplotlib.pyplot as plt
from getdist import plots, MCSamples
from fit_abacus_secondary import cosmo_fid, cosmo_true

cosmo_list = ['c000', 'c001', 'c002', 'c004']
for c in cosmo_list:
    cosmo_true[c]['log10As'] = np.log(1e10 * cosmo_true[c]['As'] * 1e-9)

colors = {'c000': 'blue', 'c001': 'orange', 'c002': 'green', 'c004': 'red'}


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


def plot_contours(samples, cosmo=None, pars_to_plot=None):
    g = plots.get_subplot_plotter()
    g.settings.axes_fontsize = 18
    g.settings.axes_labelsize = 30
    if cosmo is not None:
        markers = cosmo_true[cosmo]
        g.triangle_plot(samples, pars_to_plot, filled=True, contour_colors=[colors[cosmo]], markers=markers)
    else:
        g.triangle_plot(samples, pars_to_plot, filled=True)

    return g


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fn', type=str, required=True, help='Path to the .h5 file containing the samples')
    parser.add_argument('--cosmo', type=str, help='Cosmology identifier (e.g., c000, c001, c002, c004)', default=None)
    parser.add_argument('--output', type=str, help='Path to save the plot (e.g., contour_plot.png)', default=None)
    parser.add_argument('--cosmo_only', action='store_true', help='Plot only the cosmological parameters')
    args = parser.parse_args()
    
    fn = args.fn
    if args.cosmo is None:
        #try to infer the cosmology from the filename if not provided
        for c in cosmo_list:
            if c in fn:
                args.cosmo = c
                break
    if args.output is None:
        if not args.cosmo_only:
            args.output = f'{fn.replace(".h5", "")}_contours_full.pdf'
        else:
            args.output = f'{fn.replace(".h5", "")}_contours_cosmo.pdf'

    samples = get_samples(fn)
    params_to_plot = None
    if args.cosmo_only:
        params_to_plot = ['wc', 'h', 'log10As', 'w0', 'wa', 'wb', 'ns']
        # verify that these parameters are in the samples
        params_to_plot = [p for p in params_to_plot if p in samples.getParamNames().list()]


    g = plot_contours(samples, cosmo=args.cosmo, pars_to_plot=params_to_plot)
    g.fig.savefig(args.output, bbox_inches='tight')

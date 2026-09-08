import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
sys.path.append('../../../')
import plot_utils as pu
from getdist import MCSamples


basedir = '/global/homes/a/alexpzfz/full-shape_wrap/tmp/mock_challenge/chains/'

tracers = ['LRG', 'ELG', 'QSO']
de_cases = ['lambda', 'w0wa']
cosmo_cases = ['c000', 'c001', 'c002', 'c004']
hod_cases = ['fiducial', 'alternative']
kmax_cases = [[0.35, 0.3], [0.4, 0.35]]


def get_fn(tracer, de_case, kmax, cosmo='c000', hod='fiducial'):
    return basedir + f'Abacus-hf-cubic_{tracer}_{hod}hod_{cosmo}_intermfreedom_{de_case}_pk_kmax{kmax[0]}-{kmax[1]}_fullreparam.h5'

var_cosmo_dict = {}
var_hod_dict = {}

for tracer in tracers:
    for de_case in de_cases:
        for cosmo_case in cosmo_cases:
            for kmax in kmax_cases:
                fn = get_fn(tracer, de_case, kmax, cosmo=cosmo_case)
                samples = pu.get_samples(fn)
                var_cosmo_dict[(tracer, de_case, cosmo_case, tuple(kmax))] = samples

                if cosmo_case == 'c000':
                    for hod_case in hod_cases:
                        fn = get_fn(tracer, de_case, kmax, cosmo=cosmo_case, hod=hod_case)
                        samples = pu.get_samples(fn)
                        var_hod_dict[(tracer, de_case, hod_case, tuple(kmax))] = samples


def get_samples_giosue(fn, joint=False, w0wa=False):
    data = np.load(fn, allow_pickle=True)
    names = ['wc', 'wb', 'h', 'log10As', 'b1_r', 'b2d_r', 'bk2_r',  'avir']
    labels = ['\\omega_c', '\\omega_b', 'h', '\\ln(10^{10} A_s)', '\\tilde{b}_1', '\\tilde{b}_{2}', '\\tilde{b}_{K^2}', 'a_{vir}']
    if w0wa:
        names = ['wc', 'wb', 'w0', 'wa', 'h', 'log10As', 'b1_r', 'b2d_r', 'bk2_r',  'avir']
        labels = ['\\omega_c', '\\omega_b', 'w_0', 'w_a', 'h', '\\ln(10^{10} A_s)', '\\tilde{b}_1', '\\tilde{b}_{2}', '\\tilde{b}_{K^2}', 'a_{vir}']
    if joint:
        names = ['wc', 'wb', 'h', 'log10As'] + [None] * 24
        labels = ['\\omega_c', '\\omega_b', 'h', '\\ln(10^{10} A_s)'] + [None] * 24
        if w0wa:
            names = ['wc', 'wb', 'w0', 'wa', 'h', 'log10As'] + [None] * 24
            labels = ['\\omega_c', '\\omega_b', 'w_0', 'w_a', 'h', '\\ln(10^{10} A_s)'] + [None] * 24
    samples = MCSamples(samples=data[:, 2:], names=names, weights=np.exp(data[:, 0]), loglikes=data[:, 1], labels=labels)
    return samples

def get_fn_giosue(tracer, de_case, cosmo='c000', hod='fiducial'):
    fn = basedir + 'Chains_cubic_boxes/' 
    if de_case == 'lambda':
        fn += 'LCDM_'
    elif de_case == 'w0wa':
        fn += 'w0wa_'

    if tracer == 'ELG':
        fn += 'ELG1_'
    else:
        raise NotImplementedError(f"Tracer {tracer} not implemented for Giosue's chains.")
    if de_case == 'lambda':
        fn += 'ELG1_'

    fn += f'{hod}_hod'
    fn += f'_{cosmo}.npy'
    return fn

# load Giosue's chains
var_cosmo_dict_giosue = {}
var_hod_dict_giosue = {}
for tracer in ['ELG']:
    for de_case in de_cases:
        for cosmo_case in cosmo_cases:
            fn = get_fn_giosue(tracer, de_case, cosmo=cosmo_case)
            samples = get_samples_giosue(fn, joint=False , w0wa=(de_case=='w0wa'))
            var_cosmo_dict_giosue[(tracer, de_case, cosmo_case)] = samples
            if cosmo_case == 'c000':
                for hod_case in hod_cases:
                    fn = get_fn_giosue(tracer, de_case, cosmo=cosmo_case, hod=hod_case)
                    samples = get_samples_giosue(fn, joint=False , w0wa=(de_case=='w0wa'))
                    var_hod_dict_giosue[(tracer, de_case, hod_case)] = samples


# Compare chains for ELG between Giosue's and mine
# params_to_plot=['wc', 'h', 'log10As', 'avir',  'wb', 'b1_r', 'b2d_r', 'bk2_r']
# labels = ['Giosue', 'Alejandro']
# for tracer in ['ELG']:
#     for de_case in de_cases:
#         params_to_plot = ['wc', 'h', 'log10As']
#         if de_case == 'w0wa':
#             params_to_plot += ['w0', 'wa']
#         params_to_plot += ['avir',  'wb', 'b1_r', 'b2d_r', 'bk2_r']
#         for cosmo_case in cosmo_cases:
#             samples1 = var_cosmo_dict_giosue[(tracer, de_case, cosmo_case)]
#             samples2 = var_cosmo_dict[(tracer, de_case, cosmo_case, (0.35, 0.3))]
#             g = pu.plot_triangle([samples1, samples2], params_to_plot=params_to_plot, filled=False,
#                                  labels=labels, legend_loc='upper right', width_inch=8, legend_fontsize=12, cosmo_true=cosmo_case)
#             g.fig.suptitle(f'{tracer} {de_case} {cosmo_case}', fontsize=16)
#             g.fig.savefig(f'/global/homes/a/alexpzfz/full-shape_wrap/tmp/mock_challenge/plots_cubic/compare_giosue_alejandro_{tracer}_{de_case}_{cosmo_case}.png', dpi=200)
        
# labels = ['Giosue', 'Alejandro']
# for tracer in ['ELG']:
#     for de_case in de_cases:
#         params_to_plot = ['wc', 'h', 'log10As']
#         if de_case == 'w0wa':
#             params_to_plot += ['w0', 'wa']
#         params_to_plot += ['avir',  'wb', 'b1_r', 'b2d_r', 'bk2_r']
#         for hod_case in hod_cases:
#             samples1 = var_hod_dict_giosue[(tracer, de_case, hod_case)]
#             samples2 = var_hod_dict[(tracer, de_case, hod_case, (0.35, 0.3))]
#             g = pu.plot_triangle([samples1, samples2], params_to_plot=params_to_plot, filled=False,
#                                  labels=labels, legend_loc='upper right', width_inch=8, legend_fontsize=12)
#             g.fig.suptitle(f'{tracer} {de_case} {hod_case}', fontsize=16)
#             g.fig.savefig(f'/global/homes/a/alexpzfz/full-shape_wrap/tmp/mock_challenge/plots_cubic/compare_giosue_alejandro_{tracer}_{de_case}_{hod_case}.png', dpi=200)

for tracer in tracers:
    for de_case in de_cases:
        params_to_plot = ['wc', 'h', 'log10As']
        if de_case == 'w0wa':
            params_to_plot += ['w0', 'wa']
        params_to_plot += ['avir',  'wb', 'b1_r', 'b2d_r', 'bk2_r']
        for cosmo_case in cosmo_cases:
            samples1 = var_cosmo_dict[(tracer, de_case, cosmo_case, (0.35, 0.3))]
            samples2 = var_cosmo_dict[(tracer, de_case, cosmo_case, (0.4, 0.35))]
            labels = [f'kmax={0.35, 0.3}', f'kmax={0.4, 0.35}']
            g = pu.plot_triangle([samples1, samples2], params_to_plot=params_to_plot, filled=False,
                                 labels=labels, legend_loc='upper right', width_inch=8, legend_fontsize=12, cosmo_true=cosmo_case)
            g.fig.suptitle(f'{tracer} {de_case} {cosmo_case}', fontsize=16)
            g.fig.savefig(f'/global/homes/a/alexpzfz/full-shape_wrap/tmp/mock_challenge/plots_cubic/triangle_{tracer}_{de_case}_{cosmo_case}.png', dpi=200)

        for hod_case in hod_cases:
            samples1 = var_hod_dict[(tracer, de_case, hod_case, (0.35, 0.3))]
            samples2 = var_hod_dict[(tracer, de_case, hod_case, (0.4, 0.35))]
            labels = [f'kmax={0.35, 0.3}', f'kmax={0.4, 0.35}']
            g = pu.plot_triangle([samples1, samples2], params_to_plot=params_to_plot, filled=False,
                                 labels=labels, legend_loc='upper right', width_inch=8, legend_fontsize=12)
            g.fig.suptitle(f'{tracer} {de_case} {hod_case}', fontsize=16)
            g.fig.savefig(f'/global/homes/a/alexpzfz/full-shape_wrap/tmp/mock_challenge/plots_cubic/triangle_{tracer}_{de_case}_{hod_case}.png', dpi=200)
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
sys.path.append('../../../')
import plot_utils as pu
from fit_cutsky_abacushf import get_fn
from postprocess import export_to_text
from cosmoprimo.fiducial import AbacusSummit
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


cosmo = AbacusSummit(name=0)
markers = {'Omega_m': cosmo.get('Omega_m'),
           'Omega_b': cosmo.get('Omega_b'),
           'sigma8': cosmo.sigma8_cb,}

out_dir = '/global/u2/a/alexpzfz/full-shape_wrap/tmp/mock_challenge/chains'

tracers = ['LRG', 'ELG', 'QSO']
zranges = {'LRG': [(0.4, 0.6), (0.6, 0.8), (0.8, 1.1)],
           'ELG': [(0.8, 1.1), (1.1, 1.6)],
           'QSO': [(0.8, 2.1)]}
tracer_names = {'LRG': {(0.4, 0.6): 'LRG1', (0.6, 0.8): 'LRG2', (0.8, 1.1): 'LRG3'},
                'ELG': {(0.8, 1.1): 'ELG1', (1.1, 1.6): 'ELG2'},
                'QSO': {(0.8, 2.1): 'QSO'}}

region = 'GCcomb'
# samples_dict = {tracer: {zrange: None for zrange in zranges[tracer]} for tracer in tracers}
for tracer in tracers:
    for zrange in zranges[tracer]:
        tracer_name = tracer_names[tracer][zrange]
        fn = get_fn(tracer_name, region, freedom='interm', dkP=0.01, kmaxP=[0.35, 0.25],  bispec=True, dkB=0.01,
                    kmaxB=[0.2, 0.15], de_model='lambda', outdir=out_dir)
        fn += '.h5'
        try:
            samples = pu.get_samples(fn) 
        except FileNotFoundError:
            print(f"File not found: {fn}")
            continue
        print(f"Loaded samples for {tracer_name}")

        out_fn = f"./out/{tracer_name}_{region}_lambda_kmaxP0.35-0.25_kmaxB0.2-0.15.txt"
        samples = export_to_text(samples, out_fn, engine='comet', comm=comm)
        if rank == 0:
            g = pu.plot_triangle(samples, params_to_plot=['Omega_m', 'h', 'sigma8'],cosmo_true='c000', extra_markers=markers)
            g.fig.savefig(f"{tracer_name}_{region}_lambda_kmaxP0.35-0.25_kmaxB0.2-0.15_triangle.png")
        
        if tracer == 'QSO':
            #try 000 only
            fn = get_fn(tracer_name, region, freedom='interm', dkP=0.01, kmaxP=[0.35, 0.25],  bispec=True, dkB=0.01,
                    kmaxB=[0.2], de_model='lambda', outdir=out_dir)
            fn += '.h5'
            try:
                samples = pu.get_samples(fn)
            except FileNotFoundError:
                print(f"File not found: {fn}")
                continue
            print(f"Loaded samples for {tracer_name} with monopole only")
            out_fn = f"./out/{tracer_name}_{region}_lambda_kmaxP0.35-0.25_kmaxB0.2-0.0.txt"
            samples = export_to_text(samples, out_fn, engine='comet', comm=comm)
            if rank == 0:
                g = pu.plot_triangle(samples, params_to_plot=['Omega_m', 'h', 'sigma8'],cosmo_true='c000', extra_markers=markers)
                g.fig.savefig(f"{tracer_name}_{region}_lambda_kmaxP0.35-0.25_kmaxB0.2-0.0_triangle.png")

# do the same for w0wa
for tracer in tracers:
    for zrange in zranges[tracer]:
        tracer_name = tracer_names[tracer][zrange]
        fn = get_fn(tracer_name, region, freedom='interm', dkP=0.01, kmaxP=[0.35, 0.25],  bispec=True, dkB=0.01,
                    kmaxB=[0.2, 0.15], de_model='w0wa', outdir=out_dir)
        fn += '.h5'
        try:
            samples = pu.get_samples(fn) 
        except FileNotFoundError:
            print(f"File not found: {fn}")
            continue
        print(f"Loaded samples for {tracer_name}")

        out_fn = f"./out/{tracer_name}_{region}_w0wa_kmaxP0.35-0.25_kmaxB0.2-0.15.txt"
        samples = export_to_text(samples, out_fn, engine='comet', comm=comm)
        if rank == 0:
            g = pu.plot_triangle(samples, params_to_plot=['Omega_m', 'h', 'sigma8', 'w0', 'wa'],cosmo_true='c000', extra_markers=markers)
            g.fig.savefig(f"{tracer_name}_{region}_w0wa_kmaxP0.35-0.25_kmaxB0.2-0.15_triangle.png")
        
        if tracer == 'QSO':
            #try 000 only
            fn = get_fn(tracer_name, region, freedom='interm', dkP=0.01, kmaxP=[0.35, 0.25],  bispec=True, dkB=0.01,
                    kmaxB=[0.2], de_model='w0wa', outdir=out_dir)
            fn += '.h5'
            try:
                samples = pu.get_samples(fn)
            except FileNotFoundError:
                print(f"File not found: {fn}")
                continue
            print(f"Loaded samples for {tracer_name} with monopole only")
            out_fn = f"./out/{tracer_name}_{region}_w0wa_kmaxP0.35-0.25_kmaxB0.2-0.0.txt"
            samples = export_to_text(samples, out_fn, engine='comet', comm=comm)
            if rank == 0:
                g = pu.plot_triangle(samples, params_to_plot=['Omega_m', 'h', 'sigma8', 'w0', 'wa'],cosmo_true='c000', extra_markers=markers)
                g.fig.savefig(f"{tracer_name}_{region}_w0wa_kmaxP0.35-0.25_kmaxB0.2-0.0_triangle.png")


# now do the joint
samples = pu.get_samples(get_fn([tracer_names[tracer][zrange] for tracer in tracers for zrange in zranges[tracer]], region, freedom='interm', 
                      dkP=0.01, kmaxP=[0.35, 0.25],  
                      bispec=True, dkB=0.01, kmaxB=[0.2, 0.15], de_model='lambda', extra='new', outdir=out_dir) + '.h5')
out_fn = f"./out/ALL_{region}_lambda_kmaxP0.35-0.25_kmaxB0.2-0.15.txt"
samples = export_to_text(samples, out_fn, engine='comet', comm=comm)
if rank == 0:
    g = pu.plot_triangle(samples, params_to_plot=['Omega_m', 'h', 'sigma8'],cosmo_true='c000', extra_markers=markers)
    g.fig.savefig(f"ALL_{region}_lambda_kmaxP0.35-0.25_kmaxB0.15_triangle.png")

samples = pu.get_samples(get_fn([tracer_names[tracer][zrange] for tracer in tracers for zrange in zranges[tracer]], region, freedom='interm', 
                      dkP=0.01, kmaxP=[0.35, 0.25],  
                      bispec=True, dkB=0.01, kmaxB=[0.2, 0.15], de_model='w0wa', extra='new', outdir=out_dir) + '.h5')
out_fn = f"./out/ALL_{region}_w0wa_kmaxP0.35-0.25_kmaxB0.2-0.15.txt"
samples = export_to_text(samples, out_fn, engine='comet', comm=comm)
if rank == 0:
    g = pu.plot_triangle(samples, params_to_plot=['Omega_m', 'h', 'sigma8', 'w0', 'wa'],cosmo_true='c000', extra_markers=markers)
    g.fig.savefig(f"ALL_{region}_w0wa_kmaxP0.35-0.25_kmaxB0.15_triangle.png")
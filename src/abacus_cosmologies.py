"""AbacusSummit reference cosmologies, tabulated so that cosmoprimo is not needed.

Parameter names follow cosmoprimo. Values were generated with cosmoprimo
(cosmodesiconda 20260321-1.0.0) as:

    from cosmoprimo.fiducial import AbacusSummit
    c = AbacusSummit(name=name)
    c.get(param)  # sigma8_cb / sigma8_m from c.sigma8_cb / c.sigma8_m

The sigma8 values depend on the Boltzmann engine and cosmoprimo version at the
~1e-4 level; rerun the snippet above if exact agreement matters.
"""

ABACUS_COSMOLOGIES = {
    'c000': {'omega_b': 0.02237, 'omega_cdm': 0.12, 'h': 0.6736, 'n_s': 0.9649,
             'logA': 3.0363942552728806, 'm_ncdm': 0.059999919304831895, 'omega_ncdm': 0.0006442000000000073,
             'w0_fld': -1.0, 'wa_fld': 0.0,
             'Omega_m': 0.3151917236644108, 'Omega_b': 0.049301692328524445,
             'sigma8_cb': 0.8110979270298851, 'sigma8_m': 0.8076353990239834},
    'c001': {'omega_b': 0.02242, 'omega_cdm': 0.1134, 'h': 0.703, 'n_s': 0.9638,
             'logA': 3.0143577376771558, 'm_ncdm': 0.059999919304831895, 'omega_ncdm': 0.0006442000000000073,
             'w0_fld': -1.0, 'wa_fld': 0.0,
             'Omega_m': 0.2761263645798636, 'Omega_b': 0.045365422321325594,
             'sigma8_cb': 0.7799683504928396, 'sigma8_m': 0.7764681446801683},
    'c002': {'omega_b': 0.02237, 'omega_cdm': 0.12, 'h': 0.6278, 'n_s': 0.9649,
             'logA': 3.1415627217655304, 'm_ncdm': 0.059999919304831895, 'omega_ncdm': 0.0006442000000000072,
             'w0_fld': -0.7, 'wa_fld': -0.5,
             'Omega_m': 0.3628576966909172, 'Omega_b': 0.056757513532746974,
             'sigma8_cb': 0.8113228211079818, 'sigma8_m': 0.8078737568507639},
    'c003': {'omega_b': 0.0226, 'omega_cdm': 0.1291, 'h': 0.716, 'n_s': 0.9876,
             'logA': 3.110755950122773, 'm_ncdm': 0.059999919304831895, 'omega_ncdm': 0.0006442000000000072,
             'w0_fld': -1.0, 'wa_fld': 0.0,
             'Omega_m': 0.2971663932786309, 'Omega_b': 0.04408414219281545,
             'sigma8_cb': 0.858328757084177, 'sigma8_m': 0.8548785737580564},
    'c004': {'omega_b': 0.02237, 'omega_cdm': 0.12, 'h': 0.6736, 'n_s': 0.9649,
             'logA': 2.8875344030760046, 'm_ncdm': 0.059999919304831895, 'omega_ncdm': 0.0006442000000000073,
             'w0_fld': -1.0, 'wa_fld': 0.0,
             'Omega_m': 0.3151917236644108, 'Omega_b': 0.049301692328524445,
             'sigma8_cb': 0.7529199162504742, 'sigma8_m': 0.7497057466547449},
}


def get_abacus_cosmology(name):
    """Return the parameter dict for an AbacusSummit cosmology, e.g. 'c000', '000' or 0."""
    if isinstance(name, int):
        name = f'{name:03d}'
    name = str(name)
    if not name.startswith('c'):
        name = 'c' + name
    return ABACUS_COSMOLOGIES[name]

import pickle
from tqdm import tqdm
import numpy as np
import healpy as hp

from enterprise.signals import signal_base
from enterprise_extensions import deterministic as det
from pptadr3 import PPTADR3_Pulsar_dist


def get_value(param, name, low, high):
    """fix or random params"""
    return np.random.uniform(low, high) if param is None else param

def get_pos(theta, phi):
    x1 = np.cos(theta) * np.cos(phi)
    x2 = np.cos(theta) * np.sin(phi)
    x3 = np.sin(theta)
    return x1, x2, x3

def get_G(psr): 

    M = psr.Mmat
    n = M.shape[1]
    U, _, _ = np.linalg.svd(M)
    G = U[:, n:]
    return n, G
    
def cw_residual(num, psr, 
                cos_gwtheta = None,
                gwphi       = None,
                cos_inc     = None,
                log10_mc    = None,
                log10_fgw   = None,
                log10_dist  = None,
                phase0      = None,
                psi         = None):
    """
    params note:
    :num: number of inject cw
    :psr: a enterprise pulsar object

    return:
    a list of cw residual
    """

    psrdist = PPTADR3_Pulsar_dist[psr.name]
    rrlst = []
    for _ in range(num):
        current_cos_gwtheta = get_value(cos_gwtheta, "cos_gwtheta", -1, 1)
        current_gwphi       = get_value(gwphi, "gwphi", 0, 2 * np.pi)
        current_cos_inc     = get_value(cos_inc, "cos_inc", -1, 1)
        current_log10_mc    = get_value(log10_mc, "log10_mc", 7, 10)
        current_log10_fgw   = get_value(log10_fgw, "log10_fgw", -9, -7.7)
        current_log10_dist  = get_value(log10_dist, "log10_dist", 1.5, 2.5)
        current_phase0      = get_value(phase0, "phase0", 0, 2 * np.pi)
        current_psi         = get_value(psi, "psi", 0, np.pi)

        params_dir = {
            "cos_gwtheta": current_cos_gwtheta,
            "gwphi": current_gwphi,
            "cos_inc": current_cos_inc,
            "log10_mc": current_log10_mc,
            "log10_fgw": current_log10_fgw,
            "log10_dist": current_log10_dist,
            "phase0": current_phase0,
            "psi": current_psi
        }

        """cw residual"""
        rr = det.cw_delay(psr.toas, psr.pos, psr.pdist,
                          psrTerm=False, p_dist=psrdist, p_phase=None,
                          evolve=False, phase_approx=False, check=False,
                          tref=psr.toas.min(), **params_dir)
        rrlst.append(rr)

    return rrlst

def ecc_cw_residual(num, psr,
                    cos_gwtheta = None,
                    gwphi       = None,
                    log10_mc    = None,
                    log10_dist  = None,
                    log10_h     = None,
                    log10_F     = None,
                    cos_inc     = None,
                    psi         = None,
                    gamma0      = None,
                    e0          = None,
                    l0          = None,
                    q           = None):
    
    psrdist = PPTADR3_Pulsar_dist[psr.name]
    # print(psrdist)
    rrlst = []
    for _ in range(num):
        current_cos_gwtheta = get_value(cos_gwtheta, "cos_gwtheta", -1, 1)
        current_gwphi       = get_value(gwphi, "gwphi", 0, 2 * np.pi)
        current_cos_inc     = get_value(cos_inc, "cos_inc", -1, 1)
        current_log10_mc    = get_value(log10_mc, "log10_mc", 6, 9.5)
        current_log10_F     = get_value(log10_F, "log10_F", -9, -7)
        current_log10_dist  = get_value(log10_dist, "log10_dist", -2, 4)
        current_gamma0      = get_value(gamma0, "gamma0", 0, np.pi)
        current_psi         = get_value(psi, "psi", 0, np.pi)
        current_log10_h     = get_value(log10_h, 'log10_h', -17, -12)
        current_e0          = get_value(e0, 'e0', 0.01, 0.8)
        current_l0          = get_value(l0, 'l0', 0.01, 2 * np.pi)
        current_q           = get_value(q, 'q', 0.01, 1)

        params_dir = {
            "cos_gwtheta": current_cos_gwtheta,
            "gwphi": current_gwphi,
            "cos_inc": current_cos_inc,
            "log10_mc": current_log10_mc,
            "log10_F": current_log10_F,
            "log10_dist": current_log10_dist,
            "gamma0": current_gamma0,
            "psi": current_psi,
            "log10_h": current_log10_h,
            "e0": current_e0,
            "l0": current_l0,
            "q": current_q
        }
        try:
            rr = det.compute_eccentric_residuals(psr.toas, psr.theta, psr.phi, 
                                                pdist=psrdist, psrTerm=False, tref=psr.toas.min(),
                                                check=True, **params_dir)
        except Exception as e:
            print(e)
            continue

        rrlst.append(rr)

    return rrlst

def memory_residual(num, psr,
                    cos_gwtheta = None,
                    gwphi       = None,
                    gwpol       = None,
                    t0          = None,
                    log10_h     = None):
    
    psrdist = PPTADR3_Pulsar_dist[psr.name]

    rrlst = []
    for _ in range(num):
        current_cos_gwtheta = get_value(cos_gwtheta, "cos_gwtheta", -1, 1)
        current_gwphi       = get_value(gwphi, "gwphi", 0, 2 * np.pi)
        current_log10_h     = get_value(log10_h, 'log10_h', -17, -12)
        current_gwpol       = get_value(gwpol, 'gwpol', 0., np.pi)
        current_t0          = get_value(t0, 't0', psr.toas.min()/86400, psr.toas.max()/86400)

        params_dir = {
            "cos_gwtheta": current_cos_gwtheta,
            "gwphi": current_gwphi,
            "log10_h": current_log10_h,
            "gwpol": current_gwpol,
            "t0": current_t0
        }
        try:
            rr = det.bwm_delay(psr.toas, psr.pos, **params_dir)
        except:
            continue

        rrlst.append(rr)

    return rrlst

def fdm_residual(num, psr,
                 log10_A = None, 
                 log10_f = None,
                 phase_e = None,
                 phase_p = None):
    
    psrdist = PPTADR3_Pulsar_dist[psr.name]

    rrlst = []
    for _ in range(num):
        current_log10_A = get_value(log10_A, "log10_A", -18, -11)
        current_log10_f = get_value(log10_f, "log10_f", -9, -7)
        current_phase_e = get_value(phase_e, 'phase_e', 0, 2 * np.pi)
        current_phase_p = get_value(phase_p, 'phase_p', 0, 2 * np.pi)

        params_dir = {
            "log10_A": current_log10_A,
            "log10_f": current_log10_f,
            "phase_e": current_phase_e,
            "phase_p": current_phase_p
        }
        try:
            rr = det.fdm_delay(psr.toas, **params_dir)
        except:
            continue

        rrlst.append(rr)

    return rrlst

signals = {'cw':  cw_residual,
           'ecc': ecc_cw_residual,
           'mem': memory_residual,
           'dem': fdm_residual}

def cache_GGTlst(psrs, models, noisedict, output='./GGTlst.pkl', save=False):

    assert len(psrs) == len(models)
    ptalst = []
    GGTlst = []
    for ii, psr in tqdm(enumerate(psrs)):
        n, G = get_G(psr)
        pta = signal_base.PTA(models[ii](psr))
        N    = np.diag(pta.get_ndiag(noisedict)[0]) 
        B    = np.diag(pta.get_phi(noisedict)[0][n: ]) 
        T    = pta.get_basis()[0][:, n:]

        C    = N + np.dot(np.dot(T, B), T.T) 
        tmp = np.dot(G.T, np.dot(C, G))
        G_GT = np.dot(G, np.dot(np.linalg.inv(tmp), G.T))
        ptalst.append(pta)
        GGTlst.append(G_GT)

    if save:
        with open(output, 'wb') as f:
            pickle.dump(GGTlst, f)
    
    return GGTlst


def get_snr(num_inject, psrs, GGTlst, signal_name='cw', mean_snr=True, **params):

    # signal_name: one of ['cw', 'ecc', 'mem', 'fdm']

    if mean_snr:
        SNR = {psr.name: 0. for psr in psrs}
    else:
        SNR = {psr.name: [] for psr in psrs}

    residual = signals[signal_name]
    for p, G_GT in tqdm(zip(psrs, GGTlst)):

        res_lst = residual(num_inject, p, **params)

        for ii, res in enumerate(res_lst):
            snr = np.dot(np.dot(res.T, G_GT), res) 
            if mean_snr:
                SNR[p.name] += (snr / num_inject)
            else:
                SNR[p.name].append(snr)
        
    return SNR

def snr_skymap(psrs, GGTlst, signal_name='cw', **fix_params):

    thetas, phis = hp.pix2ang(8, range(768))
    residual = signals[signal_name]
    SNR_sky = {}
    for i, (theta, phi) in enumerate(zip(thetas, phis)):
        snr_pta = np.zeros(100)
        for iii, (psr, GCG) in enumerate(zip(psrs, GGTlst)):
            res_lst = residual(100, psr, cos_gwtheta=np.cos(theta), gwphi=phi, **fix_params)
            for ii, res in enumerate(res_lst):
                snr_pta[ii] += np.dot(res, np.dot(GCG, res))
        SNR_sky[i] = snr_pta
    
    return SNR_sky
#!/usr/bin/env python
# coding: utf-8

import os
import sys
import glob
import json
import numpy as np
import subprocess
from enterprise import constants as const
from enterprise.pulsar import Pulsar
from enterprise.signals import signal_base
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import parameter
from enterprise.signals import selections
from enterprise.signals import gp_priors
from enterprise.signals import utils
from enterprise.signals import deterministic_signals
from enterprise.signals import gp_bases
from enterprise_extensions.blocks import common_red_noise_block
# from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
from enterprise_extensions import hypermodel, deterministic

from ppta_dr3_models import *
from ppta_dr3_utils import *
    

def dr3models(psrs, marg_tm=True, crn=None, model="signal_models"):

    psr_groupnoiselist_dict = {psr: None for psr in psrnames}
    psr_groupnoiselist_dict.update(psr_groupnoise_dict_dict["all"])
    psr_groupnoiselist_dict = get_groups_in_toas(psrs, psr_groupnoiselist_dict)
    by_group_dict = {key: selections.Selection(sel_by_group_factory(item)._sel_by_group) for key, item in psr_groupnoiselist_dict.items()}

    # Set up group-selected ecorrs
    psr_groupecorrlist_dict = {psr: None for psr in psrnames}
    psr_groupecorrlist_dict.update(psr_groupecorr_dict_dict["all"])
    psr_groupecorrlist_dict = get_groups_in_toas(psrs, psr_groupecorrlist_dict) 
    by_group_ecorr_dict = {key: selections.Selection(sel_by_group_factory(item)._sel_by_group) for key, item in psr_groupecorrlist_dict.items()}

    tspan, fundamental_freq = get_tspan_fundamental_freq(psrs)
    print('TSpan is {:.3f} days'.format(tspan / 86400.0))
    print('Fundamental frequency is {:.3g} Hz'.format(fundamental_freq))

    if marg_tm:
        tm = gp_signals.MarginalizingTimingModel(use_svd=True)
    else:
        tm = gp_signals.TimingModel(use_svd=True)

    maxobs = max([psr.toas.max() for psr in psrs])
    minobs = min([psr.toas.min() for psr in psrs])
    maxTspan = maxobs - minobs
    print(f"the max Tspan of selection pulsar is {maxTspan/365.25/24/3600} yr")

    # tref = 53000 * 86400
    max_freq = 1 / 240 / 86400 # 1 / 240 days

    """
    Define white noise model
    """
    # EFAC "MeasurementNoise" can add equad, but only t2equad - we want the tnequad
    efac_prior = parameter.Constant()
    wn = white_signals.MeasurementNoise(
        efac=efac_prior,
        selection=by_backend)

    # EQUAD - TempoNest definition: sigma = sqrt((efac*sigma_0)**2 + (tnequad)**2)
    log10_equad_prior = parameter.Constant()
    wn += white_signals.TNEquadNoise(
        log10_tnequad=log10_equad_prior,
        selection=by_backend)

    # ECORR - we will swap to "white_signals.EcorrKernelNoise" later
    log10_ecorr_prior = parameter.Constant()
    wn += gp_signals.EcorrBasisModel(
        log10_ecorr=log10_ecorr_prior,
        selection=ecorr_selection)

    wn += gp_signals.EcorrBasisModel(
        log10_ecorr=log10_ecorr_prior,
        selection=global_ecorr_selection, name='basis_ecorr_all')

    signal_models = []
    pta_models = []
    priors = {}

    components_dict = {key: [] for key in ['red', 'dm', 'band', 'chrom', 'hf']}

    for psr in psrs:
        s = tm + wn

        if crn:
            crn_model_dict = get_crn_model_dict(tspan)
            crn_model = crn_model_dict["pl_nocorr_freegam"]
            s += crn_model

        """
        Define red noise model
        """
        rn_model, rn_lgA_prior, rn_gam_prior, priors = get_informed_rednoise_priors(psr, 'red_noise', noisedict_p0015, noisedict_p9985, priors)

        Tspan = psr.toas.max() - psr.toas.min()  # seconds
        max_cadence = 240.0  # days
        red_components = int(Tspan / (max_cadence*86400))
        components_dict['red'].append(red_components)
        print("Using {} red noise components".format(red_components))
        rn = gp_signals.FourierBasisGP(rn_model, components=red_components,
                                    selection=no_selection, name='red_noise')



        hf_model, hf_lgA_prior, hf_gam_prior, priors = get_informed_rednoise_priors(psr, 'hf_noise', noisedict_p0015, noisedict_p9985, priors)
        
        max_cadence = 30  # days
        hf_components = int(Tspan / (max_cadence*86400))
        components_dict['hf'].append(hf_components)
        print("Using {} hf achromatic noise components".format(hf_components))
        hf = gp_signals.FourierBasisGP(hf_model, components=hf_components,
                                    selection=no_selection, name='hf_noise')


        band_model, band_lgA_prior, band_gam_prior, priors = get_informed_rednoise_priors(psr, 'band_noise_low', noisedict_p0015, noisedict_p9985, priors)

        bandmid_model, bandmid_lgA_prior, bandmid_gam_prior, priors = get_informed_rednoise_priors(psr, 'band_noise_mid', noisedict_p0015, noisedict_p9985, priors)
        bandhigh_model, bandhigh_lgA_prior, bandhigh_gam_prior, priors = get_informed_rednoise_priors(psr, 'band_noise_high', noisedict_p0015, noisedict_p9985, priors)
        
        max_cadence = 60  # days
        band_components = int(Tspan / (max_cadence*86400))
        components_dict['band'].append(band_components)
        bn = gp_signals.FourierBasisGP(band_model, components=band_components,
                                    selection=low_freq, name='band_noise_low')
        
        bn_mid = gp_signals.FourierBasisGP(bandmid_model, components=band_components,
                                        selection=mid_freq, name='band_noise_mid')
        
        bn_high = gp_signals.FourierBasisGP(bandhigh_model, components=band_components,
                                            selection=high_freq, name='band_noise_high')
        
        """
        Define system noise model
        """
        if not dir == 'ppta15':
            max_cadence = 30  # days
            gn_components = int(Tspan / (max_cadence * 86400.0))
            gn_lgA_prior_min = np.inf
            gn_lgA_prior_max = -np.inf
            gn_gamma_prior_min = np.inf
            gn_gamma_prior_max = -np.inf
            #setting up group noise priors using the supmax and supmin of all group noise bounds
            if psr_groupnoiselist_dict[psr.name] is None:
                gn = None
            else:
                for group in psr_groupnoiselist_dict[psr.name]:

                    gn_model__, gn_lgA_prior, gn_gam_prior, gn_lgA_prior_min_, gn_lgA_prior_max_, gn_gamma_prior_min_, gn_gamma_prior_max_, priors = get_informed_rednoise_priors(psr, 'group_noise_' + group, noisedict_p0015, noisedict_p9985, priors, return_priorvals = True)
                    if gn_lgA_prior_min_ < gn_lgA_prior_min:
                        gn_lgA_prior_min = gn_lgA_prior_min_
                    if gn_lgA_prior_max_ > gn_lgA_prior_max:
                        gn_lgA_prior_max = gn_lgA_prior_max_
                    if gn_gamma_prior_min_ < gn_gamma_prior_min:
                        gn_gamma_prior_min = gn_gamma_prior_min_
                    if gn_gamma_prior_max_ > gn_gamma_prior_max:
                        gn_gamma_prior_max = gn_gamma_prior_max_
                    

                gn_lgA_prior_mins = [-18, gn_lgA_prior_min]
                gn_lgA_prior_maxs = [-11, gn_lgA_prior_max]
                gn_gamma_prior_mins = [0, gn_gamma_prior_min]
                gn_gamma_prior_maxs = [7, gn_gamma_prior_max]
                #for group noise, we look over all priors and set a global one based on the overall max and mins of the individual priors
                gn_lgA_prior = parameter.Uniform(np.max(gn_lgA_prior_mins), np.min(gn_lgA_prior_maxs))
                gn_gamma_prior = parameter.Uniform(np.max(gn_gamma_prior_mins), np.min(gn_gamma_prior_maxs))
                gn_model = gp_priors.powerlaw(log10_A=gn_lgA_prior,
                                            gamma=gn_gamma_prior)

                gn = FourierBasisGP_ppta(gn_model, fmax=1/(30*86400),
                                        selection = by_group_dict[psr.name],
                                        name='group_noise')

        
        """
        Define DM noise model
        """

        dm_model, dm_lgA_prior, dm_gam_prior, priors = get_informed_rednoise_priors(psr, 'dm_gp', noisedict_p0015, noisedict_p9985, priors)

        Tspan = psr.toas.max() - psr.toas.min()  # seconds
        max_cadence = 60  # days
        dm_components = int(Tspan / (max_cadence*86400))
        components_dict['dm'].append(dm_components)
        print("Using {} DM components".format(dm_components))
        dm_basis = gp_bases.createfourierdesignmatrix_dm(nmodes=dm_components)
        dm = gp_signals.BasisGP(dm_model, dm_basis, name='dm_gp')
        
        """
        Define chromatic noise model
        """

        chrom_model, chrom_lgA_prior, chrom_gam_prior, priors = get_informed_rednoise_priors(psr, 'chrom_gp', noisedict_p0015, noisedict_p9985, priors)

        idx = 4  # Define freq^-idx scaling
        max_cadence = 240  # days
        chrom_components = int(Tspan / (max_cadence*86400))
        components_dict['chrom'].append(chrom_components)
        print("Using {} Chrom components".format(chrom_components))
        chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=chrom_components,
                                                                idx=idx)
        chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')

        """achromatic quadratic"""
        #coefficient of quadratic term
        log10_a_quad = parameter.Uniform(-24, -16)
        sgn_a = parameter.Uniform(-1.0, 1.0)
        #coefficient of linear term
        log10_b_quad = parameter.Uniform(-16, -9)
        sgn_b = parameter.Uniform(-1.0, 1.0)
        #coefficient of DC term
        log10_c_quad = parameter.Uniform(-12, -8)
        sgn_c = parameter.Uniform(-1.0, 1.0)

        wf = achrom_tm_quadratic(a=log10_a_quad, b=log10_b_quad, c=log10_c_quad, sgn_a=sgn_a, sgn_b=sgn_b, sgn_c=sgn_c)
        tm_quad = deterministic_signals.Deterministic(wf, name="tm_quad")
        
        
        """
        DM annual
        """
        log10_Amp_dm1yr = parameter.Uniform(-10, -2)
        phase_dm1yr = parameter.Uniform(0, 2*np.pi)
        
        wf = chrom_yearly_sinusoid(log10_Amp=log10_Amp_dm1yr,
                                phase=phase_dm1yr, idx=2)
        
        dm1yr = deterministic_signals.Deterministic(wf, name="dm1yr")

        """
        define solar wind model
        """

        sw, priors = get_informed_nearth_priors(psr, noisedict_p0015, noisedict_p9985, priors)

        """
        DM Gaussian
        """
        log10_Amp = parameter.Uniform(-10, -2)
        log10_sigma_gauss = parameter.Uniform(0, 3)
        epoch_gauss = parameter.Uniform(53800, 54000)
        
        wf = dm_gaussian(log10_Amp=log10_Amp, epoch=epoch_gauss, log10_sigma=log10_sigma_gauss, idx=2)
        
        dmgauss = deterministic_signals.Deterministic(wf, name="dmgauss") 

        """
        Gaussian 20cm - J1600-3053
        """
        epoch_gauss_20cm = parameter.Uniform(57385, 57785)
        wf = gaussian_20cm(log10_Amp=log10_Amp, epoch=epoch_gauss_20cm, log10_sigma=log10_sigma_gauss, nu1=1000, nu2=2000)
        gauss_20cm = deterministic_signals.Deterministic(wf, name="gauss_20cm")

        """
        Chromatic-Gaussian Gaussian - J1600-3053
        """
        nu0_chrom_gaussian = parameter.Uniform(1000, 2000)
        log10_sigma_nu_gauss = parameter.Uniform(-1, 6)
        wf = gaussian_chrom_gaussian(log10_Amp=log10_Amp, epoch=epoch_gauss_20cm, log10_sigma=log10_sigma_gauss, nu_0=nu0_chrom_gaussian, log10_sigma_nu=log10_sigma_nu_gauss)
        gauss_chrom_gauss = deterministic_signals.Deterministic(wf, name="gauss_chrom_gauss")
        
        """
        Define total model by summing all components
        """
        # define model, add DM variations
        s += rn  + dm + sw
        
        """
        Add special noise model components for some pulsars
        """       
        # Define exponential dip parameters for 0437, 1643, and 1713
        if psr.name == 'J1713+0747' and psr.toas.min() < 57500*86400:
            expdip = True
            num_dips = 2
            idx = [parameter.Uniform(1.0, 3.0), parameter.Uniform(0.0, 2.0)]
            tmin = [54650, 57400]  # centred 54750 and 57510
            tmax = [54850, 57600]
        elif psr.name == 'J0437-4715' and psr.toas.min() < 57100*86400:
            expdip = True
            num_dips = 1
            idx = [parameter.Uniform(-1.0, 2.0)]
            tmin = [57000]
            tmax = [57200]
        elif psr.name == 'J1643-1224' and psr.toas.min() < 57100*86400:
            expdip = True
            num_dips = 1
            idx = [parameter.Uniform(-2.0, 0.0)]
            tmin = [57000]
            tmax = [57200]
        elif psr.name == 'J2145-0750' and psr.toas.min() < 56450*86400  and psr.toas.max() > 56300*86400:
            expdip = True
            num_dips = 1
            idx = [parameter.Uniform(-2.0, 2.0)]
            tmin = [56250]#, psr.toas.min()/86400]
            tmax = [56450]#, psr.toas.max()/86400]
        else:
            expdip = False
        
        # Add exponential dips to model
        if expdip:
            name = ['dmexp_{0}'.format(ii+1) for ii in range(num_dips)]
        
            for idip in range(num_dips):
                t0_exp = parameter.Uniform(tmin[idip], tmax[idip])
                log10_Amp_exp = parameter.Uniform(-10, -2)
                log10_tau_exp = parameter.Uniform(0, 2.5)
                # Define chromatic exponential decay waveform
                wf = chrom_exp_decay(log10_Amp=log10_Amp_exp,
                                    t0=t0_exp, log10_tau=log10_tau_exp,
                                    sign_param=-1.0, idx=idx[idip])
                expdip = deterministic_signals.Deterministic(wf, name=name[idip])
                s += expdip
        
        # Annual DM for J0613
        if psr.name == 'J0613-0200':
            s += dm1yr

        # Gaussian DM for J1603
        if psr.name == 'J1603-7202' and psr.toas.min() < 57500*86400:
            s += dmgauss

        # 20CM Gaussian bump for J1600:
        bump_1600 = True
        if psr.name == "J1600-3053" and psr.toas.min() < 57585*86400 and bump_1600:
            s += gauss_20cm

        # chromatic-gaussian Gaussian bump for J1600:
        bump_1600_gauss = True
        if psr.name == "J1600-3053" and psr.toas.min() < 57585*86400 and bump_1600_gauss:
            s += gauss_chrom_gauss
        
        # Chromatic noise for several pulsars in Goncharov+ or Lentati+ (1600 and 1643)
        if psr.name in ['J0437-4715', 'J0613-0200', 'J1017-7156', 'J1045-4509', 'J1600-3053', 'J1643-1224', 'J1939+2134']:
            s += chrom
    
        # Excess low-frequency band noise for several pulsars in Goncharov+ or Lentati+
        if psr.name in ['J0437-4715', 'J0613-0200', 'J1017-7156', 'J1045-4509', 'J1600-3053', 'J1643-1224', 'J1713+0747', 'J1909-3744', 'J1939+2134']:
            s += bn 
        
        if psr.name in ['J0437-4715']:
            s += bn_mid
        
        # add in group noise
        if  psr_groupnoiselist_dict[psr.name] is not  None:
            print(f"{psr.name} IN GROUP NOISE DICT")
            s += gn
        
        # Add some high-frequency (up to 1/30 days) achromatic process
        if psr.name in ['J0437-4715', 'J1017-7156', 'J1022+1001', 'J1600-3053', 'J1713+0747', 'J1744-1134', 'J1909-3744', 'J2241-5236']:
            s += hf
        
        if psr.name in ['J0437-4715']:
            s += bn_high
            
        pta_models.append(s(psr))
        signal_models.append(s)

    if model == "signal_models":
        return signal_models

    elif model == "pta_models":
        return pta_models
    
    else:
        raise ValueError

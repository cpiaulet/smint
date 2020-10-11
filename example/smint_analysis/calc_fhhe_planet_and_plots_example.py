# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 11:52:50 2020

@author: caroline

Estimate gas-to-core mass ratio based on the Lopez & Fortney 2014
models based on planet mass, radius, insolation and system age

Example script
"""

# Import modules ---------- 
from __future__ import division,print_function
import numpy as np
from smint import fit_fhhe
import astropy.io as aio

#%% Inputs for fit

print('\nReading in the inputs...')

params = dict()

# path to folder containing Lopez & Fortney models
params["path_folder_models"] = '../smint_models/'

# planet params (Mass and Radius in Earth masses)
params["Mp_earth"] = 25.7
params["err_Mp_earth"] = 2.9

params["Rp_earth"] = 4.73
params["err_Rp_earth"] = 0.16

# incident flux at the planets in units of the solar constant
params["Sinc_earth"] = 100.8
params["err_Sinc_earth"] = 1.8

# system params
# for a flat prior on the age (used if flat_age==True)
params["age_Gyr_inf"] = 1. # lower bound
params["age_Gyr_sup"] = 10. # upper bound

# alternatively, for a gaussian prior (used if flat_age==False)
params["age_Gyr"] = 5. # median of gaussian prior
params["err_age_Gyr"] = 3. # std of gaussian prior

# params for the MCMC
params["flat_age"] = True # if True, use flat prior on star's age
params["log_fenv_prior"] = False # if True, prior on fenv uniform on log10

# option for extrapolation
# if True, uses extrapolated table (fenv up to 100%, mass up to 40 Mearth)
# if False, uses original Lopez & Fortney table (fenv up to 20%, mass up to 20 Mearth)
params["extrap"] = True 

params["nsteps"] = 1000 # number of MCMC steps [1000 for testing, use much more]
params["ndim"] = 4 # number of fitted params
params["nwalkers"] = 100 # number of MCMC walkers

params["run_fit"] = True # if True, runs the MCMC
params["postprocess_oldfit"] = False # if True, no MCMC is run and old chains are loaded
params["frac_burnin"] = 0.6 # fraction of the chains to be discarded as burn-in

# saving paths (OR path to chains if postprocess_oldfit==True)
params["save"] = True # if True, save chains to npy files
params["outputdir"] = '../smint_results/'
params["fname"] = 'test' # identifier for this fit (used for saving)

# plotting and printing options

# colors for corner and histograms
params["met1_color"] = 'b' # color in histograms for metallicity = 1*solar
params["met50_color"] = 'g' # color in histograms for metallicity = 50*solar

params["corner_indiv"] = True # if True, plot individual corner plots for each fit
params["corner_both"] = True # if True, plot both corner plots superimposed

#%% End of user input

#%% Setting up the fit

print('\nSetting up the fit...')
params = fit_fhhe.setup_priors(params)

params = fit_fhhe.ini_fit(params)

#%% Run the fit
if params["run_fit"]==True and params["postprocess_oldfit"]==False:
    
    print('\nGenerating the interpolator...')
    path_file = params["path_folder_models"] + 'master_table_LF14_add_fenv_100_M_40.csv'
    t = aio.ascii.read(path_file)
    R_array = np.load(params["path_folder_models"] + 'LF14_add_fenv_100_M_40.npy')
    interpolator = fit_fhhe.make_interpolator_LF14(t, R_array, log_fenv_prior=params["log_fenv_prior"])

    print('\nRunning the fit...')  
    
    print('\nmet = 1*solar:')
    sampler_met1 = fit_fhhe.run_fit(params, interpolator, met=1.)
    
    print('\nmet = 50*solar:')
    sampler_met50 = fit_fhhe.run_fit(params, interpolator, met=50.)

    print('\nExtracting samples...')
    # extract samples 
    samples_met1 = sampler_met1.chain[:, int(params["frac_burnin"]*params["nsteps"]):, :].reshape((-1, params["ndim"]))
    samples_met50 = sampler_met50.chain[:, int(params["frac_burnin"]*params["nsteps"]):, :].reshape((-1, params["ndim"]))


#%% If loading from an old fit
if params["postprocess_oldfit"]:
    print('\nLoading chains from previous fit...')
    samples_met1 = np.load(params["outputdir"]+params["fname"]+'_chains_met1.npy')
    samples_met50 = np.load(params["outputdir"]+params["fname"]+'_chains_met50.npy')
    samples_met1 = samples_met1[:, int(params["frac_burnin"]*samples_met1.shape[1]):, :].reshape((-1, params["ndim"]))
    samples_met50 = samples_met50[:, int(params["frac_burnin"]*samples_met50.shape[1]):, :].reshape((-1, params["ndim"]))


#%% corner plot for each 
if params["corner_indiv"]:
    print('\nPlotting individual corner plots...')
    fig_met1 = fit_fhhe.plot_corner(samples_met1, params, which="met1")
    fig_met50 = fit_fhhe.plot_corner(samples_met50, params, which="met50")
    fig_met1.savefig(params['outputdir']+params["fname"]+'_corner_met1.png')
    fig_met50.savefig(params['outputdir']+params["fname"]+'_corner_met1.png')

if params["corner_both"]:
    print('\nPlotting corner plot with both metallicities...')
    rg = [[0.,100.], [0., 40.], [0., 10.], [0., 1000.]] # None
    fig_both = fit_fhhe.plot_corner([samples_met1,samples_met50], params, which="both", rg=rg)
    fig_both.savefig(params['outputdir']+params["fname"]+'_corner_both.png')
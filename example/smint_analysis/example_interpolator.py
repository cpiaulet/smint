#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 10:12:42 2020

@author: caroline

Make simple interpolator from the Lopez & Fortney (2014) or the
Zeng (2016) models to get the HHe or water mass fraction
given some input params
"""

import numpy as np
from smint import fit_fh2o
import pickle
import astropy.io as aio 

#%% Inputs for fit

print('\nReading in the inputs...')

params = dict()

# path to folder containing Lopez & Fortney models
params["path_folder_models"] = '../smint_models/'

# planet params (Mass and Radius in Earth masses)
params["Mp_earth"] = 6.2
params["err_Mp_earth"] = 0.2

params["Rp_earth"] = 2.19
params["err_Rp_earth"] = 0.09

# parameters for the fit

params["nsteps"] = 1000 # number of MCMC steps [for testing, use much more e.g. 10000]
params["ndim"] = 2 # number of fitted params
params["nwalkers"] = 100 # number of MCMC walkers

params["run_fit"] = True # if True, runs the MCMC
params["postprocess_oldfit"] = (params["run_fit"]==False) # if True, no MCMC is run and old chains are loaded
params["frac_burnin"] = 0.6 # fraction of the chains to be discarded as burn-in

# saving paths (OR path to chains if postprocess_oldfit==True)
params["save"] = (params["run_fit"]==True) # if True, save chains to npy files
params["outputdir"] = '../smint_results/'
params["fname"] = 'test_fh2o' # identifier for this fit (used for saving)

# plotting and printing options

# colors for corner and histograms
params["hist_color"] = 'b' # color in histograms
params["plot_corner"] = True # if True, generate corner plot

#%% End of user input 

#%% Setting up the fit

print('\nSetting up the fit...')

params["labels"] = [r"$f_{H_2O}$ [%]", r"M$_p$ [M$_\oplus$]"]

# set initial walker positions
params["pos0"] = [np.array([50., params["Mp_earth"]]) \
                 + np.array([40., params["err_Mp_earth"]]) \
                     * np.random.randn(params["ndim"]) for i in range(params["nwalkers"])]


if params["save"]:
    # save params dictionary
    f = open(params["outputdir"]+params["fname"]+"_params"+".pkl","wb")
    pickle.dump(params, f)
    f.close()

#%% Run the fit

if params["run_fit"]==True and params["postprocess_oldfit"]==False:
    
    print('\nGenerating the interpolator...')
    t_rock_h2o = aio.ascii.read(params["path_folder_models"]+"t_rock_h2o_Zeng2016.csv")
    interpolator = fit_fh2o.make_interpolator_fh2o(t_rock_h2o)

    print('\nRunning the fit...')      
    sampler = fit_fh2o.run_fit(params, interpolator)
    
    print('\nExtracting samples...')
    samples = sampler.chain[:, int(params["frac_burnin"]*params["nsteps"]):, :].reshape((-1, params["ndim"]))


#%% If loading from an old fit

if params["postprocess_oldfit"]:
    print('\nLoading chains from previous fit...')
    samples = np.load(params["outputdir"]+params["fname"]+'_chains.npy')
    samples = samples[:, int(params["frac_burnin"]*samples.shape[1]):, :].reshape((-1, params["ndim"]))

#%% corner plot for each 
if params["plot_corner"]:
    print('\nGenerating corner plot...')
    fig = fit_fh2o.plot_corner(samples, params)
    fig.savefig(params['outputdir']+params["fname"]+'_corner.png')

# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 11:52:50 2020

@author: caroline

Estimate gas-to-core mass ratio based on the Lopez & Fortney 2014
models based on planet mass, radius, insolation and system age

Example script
"""

# Import modules ---------- 
import numpy as np
from smint import fit_fhhe
# import astropy.io as aio
from astropy.io import ascii as aioascii

import pickle
import os
import configparser
import argparse
from copy import deepcopy
import pdb
import sys

#%% The main code starts here

def main(argv): 
    
    '''
    Example:
	python calc_hhe_planet_and_plots_example.py template_ini.ini
    '''
    
    if len(argv)>1:
        iniFile=argv[1]
    else:
        iniFile='../smint_analysis/template_ini_fhhe.ini'
            
    if not os.path.exists(iniFile):
        print('USER ERROR: iniFile does not exist.')
        raise
    config = configparser.ConfigParser()
    config.read(iniFile)


    #%% Inputs for fit
    
    print('\nReading in the inputs...')

    parser = argparse.ArgumentParser(description='Inputs for the code.')
    
    parser.add_argument('-path_folder_models', help='path to folder containing Lopez & Fortney + Zeng models', default=config.get('paths','path_folder_models'))
    parser.add_argument('-outputdir', help='saving path (OR path to chains if run_fit==False)', default=config.get('paths','outputdir'))
    parser.add_argument('-fname', help='identifier for this fit (used for saving)', default=config.get('paths','fname'))

    parser.add_argument('-Mp_earth', help='planet mass (in Mearth)', default=config.getfloat('physical params','Mp_earth'))
    parser.add_argument('-err_Mp_earth', help='planet mass uncertainty (in Mearth)', default=config.getfloat('physical params','err_Mp_earth'))
    parser.add_argument('-Rp_earth', help='planet radius (in Rearth)', default=config.getfloat('physical params','Rp_earth'))
    parser.add_argument('-err_Rp_earth', help='planet radius uncertainty (in Rearth)', default=config.getfloat('physical params','err_Rp_earth'))
    parser.add_argument('-Sinc_earth', help='incident flux at the planet in units of the solar constant', default=config.getfloat('physical params','Sinc_earth'))
    parser.add_argument('-err_Sinc_earth', help='uncertainty on incident flux at the planet in units of the solar constant', default=config.getfloat('physical params','err_Sinc_earth'))
    
    parser.add_argument('-age_Gyr_inf', help='lower bound on system age in Gyr for flat age prior', default=config.getfloat('physical params','age_Gyr_inf'))
    parser.add_argument('-age_Gyr_sup', help='upper bound on system age in Gyr for flat age prior', default=config.getfloat('physical params','age_Gyr_sup'))
    parser.add_argument('-age_Gyr', help='mean of gaussian prior on system age in Gyr', default=config.getfloat('physical params','age_Gyr'))
    parser.add_argument('-err_age_Gyr', help='std of gaussian prior on system age in Gyr', default=config.getfloat('physical params','err_age_Gyr'))

    parser.add_argument('-flat_age', help='bool. if True, use flat prior on stellar age', default=config.getboolean('MCMC params','flat_age'))
    parser.add_argument('-log_fenv_prior', help='bool. if True, prior on fenv uniform on log10', default=config.getboolean('MCMC params','log_fenv_prior'))
    parser.add_argument('-nsteps', help='number of MCMC steps', default=config.getint('MCMC params','nsteps'))
    parser.add_argument('-ndim', help='number of fitted params', default=config.getint('MCMC params','ndim'))
    parser.add_argument('-nwalkers', help='number of MCMC walkers', default=config.getint('MCMC params','nwalkers'))
    parser.add_argument('-run_fit', help='bool. if True, runs the MCMC; otherwise, postprocess an existing fit', default=config.getboolean('MCMC params','run_fit'))
    parser.add_argument('-frac_burnin', help='fraction of the chains to be discarded as burn-in [range 0--1]', default=config.getfloat('MCMC params','frac_burnin'))
    
    parser.add_argument('-met1_color', help='color in histograms for metallicity = 1*solar', default=config.get('plotting','met1_color'))
    parser.add_argument('-met50_color', help='color in histograms for metallicity = 50*solar', default=config.get('plotting','met50_color'))
    parser.add_argument('-corner_indiv', help='bool. if True, plot individual corner plots for each fit', default=config.getboolean('plotting','corner_indiv'))
    parser.add_argument('-corner_both', help='bool. if True, plot both corner plots superimposed', default=config.getboolean('plotting','corner_both'))

    args, unknown = parser.parse_known_args()

    # make the params dict from the parser object

    params = deepcopy(args.__dict__)
    
    params["postprocess_oldfit"] = (params["run_fit"]==False) # if True, no MCMC is run and old chains are loaded    
    params["save"] = (params["run_fit"]==True) # if True, save chains to npy files


    #%% End of user input
    
    #%% Setting up the fit
    
    print('\nSetting up the fit...')
    params = fit_fhhe.setup_priors(params)
    
    params = fit_fhhe.ini_fit(params)
    
    if params["save"]:
        # save params dictionary
        f = open(params["outputdir"]+params["fname"]+"_params"+".pkl","wb")
        pickle.dump(params, f)
        f.close()
    
    #%% Run the fit
    if params["run_fit"]==True and params["postprocess_oldfit"]==False:
        
        print('\nGenerating the interpolator...')
        path_file = params["path_folder_models"] + 'master_table_LF14_20201014.csv'
        t = aioascii.read(path_file)
        R_array = np.load(params["path_folder_models"] + 'LF14_20201014.npy')
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
        fig_met50.savefig(params['outputdir']+params["fname"]+'_corner_met50.png')
    
    if params["corner_both"]:
        print('\nPlotting corner plot with both metallicities...')
        rg = [[5.,20.], [0., 15.], [1., 10.], [27., 37.]] # None
        fig_both = fit_fhhe.plot_corner([samples_met1,samples_met50], params, which="both", rg=rg)
        fig_both.savefig(params['outputdir']+params["fname"]+'_corner_both.png')
    
#%%

if __name__ == "__main__":
    main(sys.argv)
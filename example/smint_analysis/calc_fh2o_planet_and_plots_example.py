#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 08:36:48 2020

@author: caroline

Estimate water mass fraction from the mass and radius of a planet
assuming a rock+water mix, using the Zeng et al. 2016 grid

Example script
"""

# Import modules ---------- 
import numpy as np
from smint import fit_fh2o
# import astropy.io as aio
from astropy.io import ascii as aioascii

import pickle
import os
import configparser
import argparse
from copy import deepcopy
import pdb
import sys
from astropy import table


#%% The main code starts here

def main(argv):
    '''
    Example:
	python calc_hhe_planet_and_plots_example.py template_ini.ini
    '''
        
    if len(argv)>1:
        iniFile=argv[1]
    else:
        iniFile='template_ini_h2o.ini'
            
    if not os.path.exists(iniFile):
        raise FileNotFoundError('USER ERROR: iniFile does not exist.')
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

    parser.add_argument('-nsteps', help='number of MCMC steps', default=config.getint('MCMC params','nsteps'))
    parser.add_argument('-ndim', help='number of fitted params', default=config.getint('MCMC params','ndim'))
    parser.add_argument('-nwalkers', help='number of MCMC walkers', default=config.getint('MCMC params','nwalkers'))
    parser.add_argument('-run_fit', help='bool. if True, runs the MCMC; otherwise, postprocess an existing fit', default=config.getboolean('MCMC params','run_fit'))
    parser.add_argument('-frac_burnin', help='fraction of the chains to be discarded as burn-in [range 0--1]', default=config.getfloat('MCMC params','frac_burnin'))
    
    parser.add_argument('-hist_color', help='color in histograms and corner', default=config.get('plotting','hist_color'))
    parser.add_argument('-plot_corner', help='bool. if True, generate corner plot', default=config.getboolean('plotting','plot_corner'))

    args, unknown = parser.parse_known_args()

    # make the params dict from the parser object

    params = deepcopy(args.__dict__)
    
    params["postprocess_oldfit"] = (params["run_fit"]==False) # if True, no MCMC is run and old chains are loaded    
    params["save"] = (params["run_fit"]==True) # if True, save chains to npy files
    print(params)
    
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
        t_rock_h2o = aioascii.read(params["path_folder_models"]+"t_rock_h2o_Zeng2016.csv")
        interpolator = fit_fh2o.make_interpolator_fh2o(t_rock_h2o)
    
        print('\nRunning the fit...')      
        sampler = fit_fh2o.run_fit(params, interpolator)
        
        print('\nExtracting samples...')
        samples = sampler.chain[:, int(params["frac_burnin"]*params["nsteps"]):, :].reshape((-1, params["ndim"]))
    
    #%% Save and print median +/- sigma
    if params["run_fit"]==True and params["postprocess_oldfit"]==False:
        print("\nCalculating parameter constraints...")
        fit_fh2o.calc_constraints(samples, params)
    
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

#%%
if __name__ == "__main__":
    main(sys.argv)

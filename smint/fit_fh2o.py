#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 08:37:47 2020

@author: caroline

Estimate water mass fraction from the mass and radius of a planet
assuming a rock+water mix, using the Zeng et al. 2016 grid

Utilities functions
"""

# Import modules ---------- 
from __future__ import division,print_function
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import emcee
import corner
from copy import deepcopy
import astropy.io as aio
from astropy import table
import pandas as pd

#%% utilities for interpolation

def read_zeng_table(fname, prints=False):
    '''
    Read the table containing Li Zeng's MR relations
    '''
    if prints:
        print('\nReading in file: ',fname)
    df = pd.read_csv(fname, sep='\t')
    t = table.Table.from_pandas(df)
    
    # set properly column names
    cnames = t.colnames
    t[cnames[0]].name = 'Mass'
    for c in t.colnames:
        t[c].unit = t[c][0]
    t.remove_row(0) # contained the units

    df2 = t.to_pandas()
    for c in df2.columns:
        try:
            df2[c] = pd.to_numeric(df2[c], downcast="float")
        except:
            continue
    t2 = table.Table.from_pandas(df2)
    
    if prints:
        print('\n')
        print(t2)
        print('\n')
        print(t2.info)
    return t2

def find_radius_fh2o_table(t=None, interp=None, mass=10., fh2o=10.):
    """
    Given a planet mass (in Earth masses) and a h2o mass fraction (%),
    get the best-matching planet radius (in Earth radii)
    """
    if interp is None:
        interp = make_interpolator_fh2o(t)
    r_earth = interp((fh2o,mass), method='linear')
    return r_earth

def prep_table_interp_fh2o(params):
    """
    Takes as input the fit params and prepares the rock-h2o
    table for the interpolator
    """
    t_large_zeng = read_zeng_table(params["path_folder_models"] + 'large_table_Zeng2016.txt')
    
    # isolate the rock-h2o compositions
    col_rock_h2o = ['Mass']
    for c in t_large_zeng.columns:
        if 'o' in c:
            if c in ['cold_h2/he', 'max_coll_strip']:
                continue
            col_rock_h2o.append(c)
    t_rock_h2o = deepcopy(t_large_zeng[col_rock_h2o])
    t_rock_h2o['rocky '].name = '0%h2o'
    
    return t_rock_h2o

def make_interpolator_fh2o(params, t=None):
    """
    Takes as an input t the table of rock-h2o mix
    make an interpolator for the planet radius as a function of 
    ['Mass', 'fh2o']
    """
    if t is None:
        t = prep_table_interp_fh2o(params)
    
    # make array of f_h2o_pc and planet radius values
    f_h2o_pc = []
    R_list = []
    for c in t.columns[1:]:
        s = c.split('%')
        f_h2o_pc.append(float(s[0]))
        R_list.append(np.array(t[c]))
    f_h2o_pc = np.array(f_h2o_pc)
    
    Mass_oplus = np.array(t['Mass'])
    
    R_array = np.array(R_list)
    interpolator = RegularGridInterpolator((f_h2o_pc, Mass_oplus,), 
                                            R_array, bounds_error=False)
    return interpolator



#%% emcee functions

def lnlike(theta, true_rad, true_rad_err, interp):
    """
    Log-likelihood function for emcee fit
    """
    
    fh2o, mass = theta 
    
    # estimate interpolated radius for these params
    radius = find_radius_fh2o_table(t=None, interp=interp, mass=mass, fh2o=fh2o)
    return -0.5*(((true_rad-radius)/true_rad_err)**2)

def lnprior(theta, mass_mu, mass_err):
    """
    Log-prior (using known mass distri and uniform in fh2o)
    """
    fh2o, mass = theta

    if (fh2o < 0.) or (fh2o > 100.):
        return -np.inf
    if (mass < 0.0625) + (mass > 32.):
        return -np.inf

    return -0.5 *(((mass_mu - mass)/mass_err)**2.)

def lnprob(theta, true_rad, true_rad_err, interp, mass_mu, mass_err):
    """
    Log-probability function
    """
    lp = lnprior(theta, mass_mu, mass_err) 
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike(theta, true_rad, true_rad_err, interp)
    

#%% run interpolator MCMC

def run_fit(params, interpolator):
    """
    Run the emcee fit using the previously-set up priors and params
    Interpolator: generated using make_interpolator_fh2o()
    returns the emcee sampler
    """
    
    print("\nSetting up the sampler...")
    sampler = emcee.EnsembleSampler(params["nwalkers"], params["ndim"], lnprob,
                                    args=(params["Rp_earth"],params["err_Rp_earth"],
                                          interpolator, params["Mp_earth"],
                                          params["err_Mp_earth"]))
    
    print("\nRunning the emcee fit...")
    sampler.run_mcmc(params["pos0"], params["nsteps"])
    
    if params["save"]:
        print("\nSaving the results...")
        np.save(params["outputdir"]+params["fname"]+'_chains.npy', sampler.chain)
    
    return sampler


#%% post-processing 
    
def plot_corner(samples, params, plot_datapoints=False, smooth=1.,
                        quantiles=[0.16, 0.5, 0.84], title_kwargs={'fontsize':14},
                        hist_kwargs={"linewidth":3}, rg=None, 
                        show_titles=True, **kwargs):
    """
    Corner plot for an emcee fit of the water mass fraction that matches
    the observed planet params
    
    samples: generated by emcee sampler
    params: fit params
    other args: args for the corner function
    
    Returns the figure with the corner plot 
    """
    hist_kwargs["color"] = params["hist_color"]
    fig = corner.corner(samples, labels=params["labels"], 
                        plot_datapoints=plot_datapoints, smooth=smooth,
                        show_titles=show_titles, quantiles=quantiles,
                        title_kwargs=title_kwargs, color=color,
                        hist_kwargs=hist_kwargs, range=rg, **kwargs)
    return fig



# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 11:52:50 2020

@author: caroline

Estimate gas-to-core mass ratio based on the Lopez & Fortney 2014
models based on planet mass, radius, insolation and system age

Utilities functions
"""

# Import modules ---------- 
from __future__ import division,print_function
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import emcee
import corner

#%% utilities for interpolation

def find_radius_LF14_table(t=None, interp=None, met=1., age=1., finc=1.,
                           mass=10., fenv=10.):
    """
    Given a metallicity, an age, an incident flux and a planet mass and envelope
    mass fraction (%),
    get the best-matching planet radius
    """
    if interp is None:
        interp = make_interpolator_LF14(t)
    f_env_pc = interp((met,age,finc,mass,fenv), method='linear')
    return f_env_pc

def make_interpolator_LF14(t, R_array):
    """
    make an interpolator for the planet radius as a function of 
    ['metallicity_solar','age_Gyr', 'F_inc_oplus','Mass_oplus','f_env_pc']
    """
    
    # make array of f_env_pc values
    metallicity_solar = np.unique(np.array(t['metallicity_solar']))
    age_Gyr = np.unique(np.array(t['age_Gyr']))
    F_inc_oplus = np.unique(np.array(t['F_inc_oplus']))
    Mass_oplus = np.unique(np.array(t['Mass_oplus']))
    f_env_pc = np.unique(np.array(t['f_env_pc']))
    
                    
    interpolator = RegularGridInterpolator((metallicity_solar, age_Gyr,
                                            F_inc_oplus,Mass_oplus,
                                            f_env_pc,), 
                                            R_array, bounds_error=False)
    return interpolator

#%% emcee functions

def lnlike(theta, true_rad, true_rad_err, interp, met, log_fenv_prior):
    """
    Log-likelihood function for emcee fit
    """
    if log_fenv_prior:
        log10_fenv, mass, age, finc = theta
        fenv = 10**log10_fenv
    else:
        fenv, mass, age, finc = theta
    # estimate interpolated radius for these params
    radius = find_radius_LF14_table(t=None, interp=interp, met=met, age=age, 
                                    finc=finc, mass=mass, fenv=fenv)
    return -0.5*(((true_rad-radius)/true_rad_err)**2)

def lnprior(theta, mu, icovmat, flat_age, age_min, age_max, log_fenv_prior, extrap):
    """
    prior (using known age distri, Finc distri, mass distri and flat in fenv
    or log fenv)
    """
    if log_fenv_prior:
        log10_fenv, mass, age, finc = theta
        fenv = 10**log10_fenv
    else:
        fenv, mass, age, finc = theta

    if extrap:
        fenv_max = 100.
        mass_max = 40.
    else:
        fenv_max = 20.
        mass_max = 20.
        
    if (fenv < 0.01) or (fenv > fenv_max):
        return -np.inf
    if (mass < 1.0) + (mass > mass_max):
        return -np.inf
    if (age < 0.1) + (age > 10.):
        return -np.inf
    if flat_age:
        if (age < age_min) + (age > age_max):
            return -np.inf
    if (finc < 0.1) + (finc > 1000.):
        return -np.inf
    else:
        if flat_age:
            arr = np.array([mass, finc])
        else:
            arr = np.array([mass, age, finc])
        diff = arr - mu
        return -np.dot(diff, np.dot(icovmat, diff)) / 2.0

def lnprob(theta, true_rad, true_rad_err, interp, met, mu, icovmat, 
           flat_age=True, age_min=0.1, age_max=10., log_fenv_prior=True, extrap=False):
    """
    Log-probability function
    """
    lp = lnprior(theta, mu, icovmat, flat_age, age_min, age_max, log_fenv_prior, extrap)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike(theta, true_rad, true_rad_err, interp, met, log_fenv_prior)
    

#%% setup and run interpolator
        
def setup_priors(params):
    """
    input: params of the fit
    output: mean and covariance matrix for gaussian priors,
    lower and upper bound on the flat prior on the age if not a gaussian prior
    """
    if params["flat_age"]:
        mu = np.array([params["Mp_earth"], params["Sinc_earth"]])
        covmat = np.zeros((2,2))
        covmat[0,0] = params["err_Mp_earth"]**2.
        covmat[1,1] = params["err_Sinc_earth"]**2.
        age_min = params["age_Gyr_inf"]
        age_max = params["age_Gyr_sup"]
    else:
        mu = np.array([params["Mp_earth"], params["age_Gyr"], params["Sinc_earth"]])
        covmat = np.zeros((3,3))
        covmat[0,0] = params["err_Mp_earth"]**2.
        covmat[1,1] = params["err_age_Gyr"]**2.
        covmat[2,2] = params["err_Sinc_earth"]**2.
        age_min = 0.1
        age_max = 10.
    
    params["icovmat"] = np.linalg.inv(covmat)
    params["mu"] = mu
    params["covmat"] = covmat
    params["age_min"] = age_min
    params["age_max"] = age_max
    
    return params

def ini_fit(params):
    """
    input: params of the fit
    output: initial positions of the walkers and labels for the fitted para
    """

    if params["log_fenv_prior"]:
        fenv_ini = 0.
        fenv_unc = 1.
        fenv_label = r"$\log_{10}$ f$_{HHe}$ [%]"
    else:
        fenv_ini = 75.
        fenv_unc = 10.
        fenv_label = r"f$_{HHe}$ [%]"
    x0 = np.array([fenv_ini, params["Mp_earth"], params["age_Gyr"], params["Sinc_earth"]])
    
    params["labels"] = [fenv_label, r"M$_p$ [M$_\oplus$]", "Age [Gyr]", r"S$_{inc}$ [S$_\oplus$]"]
    
    params["pos0"] = [x0 + np.array([fenv_unc, 2.*params["Mp_earth"], 1., 10.])\
                     * np.random.randn(params["ndim"]) for i in range(params["nwalkers"])]

    return params


def run_fit(params, interpolator, met=1.):
    """
    Run the emcee fit using the previously-set up priors and params
    Interpolator: generated using make_interpolator_LF14()
    returns the emcee samplers for met=1*solar or met=50*soar
    """
    
    if met !=1 and met != 50:
        raise ValueError("Metallicity has to be 1 or 50 * solar!")
    
    print("\nSetting up the sampler...")
    sampler = emcee.EnsembleSampler(params["nwalkers"], params["ndim"], lnprob,
                                    args=(params["Rp_earth"],params["err_Rp_earth"],
                                          interpolator, met, params["mu"],
                                          params["icovmat"], params["flat_age"],
                                          params["age_min"], params["age_max"], 
                                          params["log_fenv_prior"], params['extrap']))
    
    print("\nRunning the emcee fit...")
    sampler.run_mcmc(params["pos0"], params["nsteps"])
    
    if params["save"]:
        print("\nSaving the results...")
        np.save(params["outputdir"]+params["fname"]+'chains_met'+str(int(met))+'.npy', sampler.chain)
    
    return sampler


#%% post-processing 
    
def plot_corner(samples, params, which="met1", 
                        plot_datapoints=False, smooth=1.,
                        quantiles=[0.16, 0.5, 0.84], title_kwargs={'fontsize':14},
                        hist_kwargs={"linewidth":3}, rg=None, **kwargs):
    """
    Corner plot for an emcee fit of the envelope mass fraction that matches
    the observed planet and system params
    
    samples: generated by emcee sampler
    params: fit params
    which: "met1", "met50", "both" depending on what we want to show
    other args: args for the corner function
    
    Returns the figure with the corner plot 
    """
    print("\n** Plotting corner for", which)
    if "met" in which:
        if which == "met1":
            color = params["met1_color"]
            hist_kwargs["color"] = params["met1_color"]
        elif which == "met50":
            color = params["met50_color"]
            hist_kwargs["color"] = params["met50_color"]

        fig = corner.corner(samples, labels=params["labels"], 
                            plot_datapoints=plot_datapoints, smooth=smooth,
                            show_titles=True, quantiles=quantiles,
                            title_kwargs=title_kwargs, color=color,
                            hist_kwargs=hist_kwargs, range=rg, **kwargs)
    if which == "both":
        hist_kwargs["color"] = params["met50_color"]
        fig = corner.corner(samples[1], labels=params["labels"], 
                            plot_datapoints=plot_datapoints, smooth=smooth,
                            show_titles=False, title_kwargs=title_kwargs,
                            color=params["met50_color"], hist_kwargs=hist_kwargs,
                            range=rg, **kwargs)
        hist_kwargs["color"] = params["met1_color"]
        corner.corner(samples[0], fig=fig, labels=params["labels"], 
                            plot_datapoints=plot_datapoints, smooth=smooth,
                            show_titles=False, title_kwargs=title_kwargs,
                            color=params["met1_color"], hist_kwargs=hist_kwargs,
                            range=rg, **kwargs)
        
    return fig



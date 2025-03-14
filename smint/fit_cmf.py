# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 11:52:50 2020

@author: caroline

Estimate water mass fraction based on Aguichine+2021 models
for irradiated water worlds

Utilities functions
"""

# Import modules ---------- 
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import emcee
import corner

#%% utilities for interpolation

def find_radius_from_comp(path_models=None, interp_r=None, fcore_in_interior=1., 
                          Tirr=1., fh2o=0., log10_mass=1.):
    """
    Given a planet internal structure (mass fraction of the core in the
                                       core+mantle, h2o mass fraction),
    and a planet mass, get the best-matching planet radius
    interp_r: interpolator to get radii from parameters
    """
    if interp_r is None:
        interp_r = make_interpolator_A21(path_models)
    r_earth = interp_r((fcore_in_interior,Tirr,fh2o,log10_mass), method='linear')
    return r_earth

def make_interpolator_A21(path_models, which_quantity="r"):
    """
    make an interpolator for quantity which_quantity as a function of 
    ['fcore_in_interior','Tirr', 'fh2o','Mass_oplus']
    using the Aguichine et al. 2021 grid
    the interpolation is linear with the log10 of the planet mass
    which_quantity="r", "r_err", "validity"
    """
    
    # make array of f_env_pc values
    fcore_in_interior = np.load(path_models + "aguichine_xcoreprimes.npy")
    Tirr = np.load(path_models + "aguichine_tirrs.npy")
    fh2o = np.load(path_models + "aguichine_xh2os.npy")
    log10_Mass_oplus = np.load(path_models + "log10masses_earth_grid_aguichine.npy")
    
    if which_quantity == "r":
        y = np.load(path_models + "radii_earth_grid_aguichine.npy")
 
    elif which_quantity == "r_err":
        y = np.load(path_models + "radii_earth_err_grid_aguichine.npy")

    elif which_quantity == "validity":
        y = np.load(path_models + "validity_grid_aguichine.npy")
        
                        
    interpolator = RegularGridInterpolator((fcore_in_interior, Tirr,
                                            fh2o, log10_Mass_oplus,), 
                                            y, 
                                            bounds_error=False)
    return interpolator

#%% emcee functions

def lnlike(theta, params, interp_r, interp_valid):
    """
    Log-likelihood function for emcee fit
    """
    
    fcore_in_interior, mass = theta
    # for CMF fitting
    fh2o=0.
    Tirr=400.

    true_rad = params["Rp_earth"]
    true_rad_err = params["err_Rp_earth"]
    
    # using the result for Tirr=400K at lower temperatures
    if Tirr < 400.:
        Tirr = 400.
        
    # estimate interpolated radius for these params
    radius = find_radius_from_comp(path_models=None, interp_r=interp_r, fcore_in_interior=fcore_in_interior, 
                          Tirr=Tirr, fh2o=fh2o, log10_mass=np.log10(mass))
    valid = interp_valid((fcore_in_interior,Tirr,fh2o,np.log10(mass)), method='linear')

    if valid == 2.:
        lnlk = -np.inf
    else:
        lnlk = -0.5*(((true_rad-radius)/true_rad_err)**2)

    return lnlk

def lnprior(theta, params):
    """
    prior (using known Tirr distri, mass distri and flat in fcore_in_interior
    and fh2o)
    """
    # for CMF fitting

    fcore_in_interior, mass = theta

    fh2o=0.
    Tirr=400.

    use_kde = params["use_KDE_for_Mp_prior"]
    
    if use_kde:
        kde_points = params["kde_Mp_points"]
        kde_density = params["kde_Mp_density"]    
        mu_Tirr = params["Tirr"]
        sig_Tirr = params["err_Tirr"]

    else:
        mu = params["mu"]
        icovmat = params["icovmat"]
    
    grid_lim = params["grid_lim"]
    
    
            
    if (fcore_in_interior < grid_lim['fcore_in_interior'][0]) or (fcore_in_interior > grid_lim['fcore_in_interior'][1]):
        return -np.inf
    if (mass < grid_lim['mass'][0]) or (mass > grid_lim['mass'][1]):
        return -np.inf
    if (fh2o < grid_lim['fh2o'][0]) or (fh2o > grid_lim['fh2o'][1]):
        return -np.inf
    if (Tirr < grid_lim['Tirr'][0]) or (Tirr > grid_lim['Tirr'][1]):
        return -np.inf

    else:
        if use_kde:
            lp_Tirr = -0.5*(((Tirr-mu_Tirr)/sig_Tirr)**2)
            lp_mass = np.log(np.interp(mass, kde_points, kde_density))
            return lp_Tirr + lp_mass
        else:
            arr = np.array([mass, Tirr])
            diff = arr - mu
            return -np.dot(diff, np.dot(icovmat, diff)) / 2.0

def lnprob(theta, params, interp_r, interp_valid):
    """
    Log-probability function
    """
    lp = lnprior(theta, params)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike(theta, params, interp_r, interp_valid)
    

#%% setup and run interpolator
        
def setup_priors(params):
    """
    input: params of the fit
    output: mean and covariance matrix for gaussian priors,
    lower and upper bound on the flat prior on the age if not a gaussian prior
    """
    if params["use_KDE_for_Mp_prior"]:
        params["kde_Mp_points"] = np.load(params["path_file_kde_points"])
        params["kde_Mp_density"] = np.load(params["path_file_kde_density"])        

        params["icovmat"] = None
        params["mu"] = None
        params["covmat"] = None
    
    else:
        mu = np.array([params["Mp_earth"], 500.]) # dummy Tirr value for CMF fit
        covmat = np.zeros((2,2))
        covmat[0,0] = params["err_Mp_earth"]**2.
        covmat[1,1] = 100.**2. # dummy value in CMF fit
        
        params["icovmat"] = np.linalg.inv(covmat)
        params["mu"] = mu
        params["covmat"] = covmat
        
        params["kde_Mp_points"] = None
        params["kde_Mp_density"] = None
    
    return params

def ini_fit(params, grid_lim=None):
    """
    input: params of the fit
    grid_lim: dict with the lower and upper bounds on the grid params
    if None, uses the bounds from the Aguichine et al. (2021) grid
    output: initial positions of the walkers and labels for the fitted para
    """

    x0 = np.array([0.33, params["Mp_earth"]])
    
    params["labels"] = [r"f$_\mathrm{core}'$", r"M$_p$ [M$_\oplus$]"]
    
    params["pos0"] = [x0 + np.array([0.2, params["err_Mp_earth"]])\
                     * np.random.randn(params["ndim"]) for i in range(params["nwalkers"])]

    if grid_lim is None:
        grid_lim = dict()
        grid_lim['fcore_in_interior'] = [0., 0.9]
        grid_lim['mass'] = [0.2, 20.]
        grid_lim['fh2o'] = [0.0, 1.0]
        grid_lim['Tirr'] = [400, 1300.]

    params["grid_lim"] = grid_lim
        
    return params


def run_fit(params, interp_r, interp_validity):
    """
    Run the emcee fit using the previously-set up priors and params
    Interpolators: generated using make_interpolator_A21()
    returns the emcee sampler
    """
    
    
    print("\nSetting up the sampler...")
    sampler = emcee.EnsembleSampler(params["nwalkers"], params["ndim"], lnprob,
                                    args=(params, interp_r, interp_validity))
    
    print("\nRunning the emcee fit...")
    sampler.run_mcmc(params["pos0"], params["nsteps"], progress=True)
    
    if params["save"]:
        print("\nSaving the results...")
        np.save(params["outputdir"]+params["fname"]+'_chains.npy', sampler.chain)
    
    return sampler


#%% post-processing 
    
def plot_corner(samples, params, plot_datapoints=False, smooth=1.,
                        quantiles=[0.16, 0.5, 0.84], title_kwargs={'fontsize':14},
                        hist_kwargs={"linewidth":3}, rg=None, 
                        show_titles=True, levels=(0.393,0.865,0.989), **kwargs):
    """
    Corner plot for an emcee fit of the water mass fraction that matches
    the observed planet params
    
    samples: generated by emcee sampler
    params: fit params
    other args: args for the corner function
    
    Returns the figure with the corner plot 
    """
    hist_kwargs["color"] = params["hist_color"]
    color = params["hist_color"]
    fig = corner.corner(samples, labels=params["labels"], 
                        plot_datapoints=plot_datapoints, smooth=smooth,
                        show_titles=show_titles, quantiles=quantiles,
                        title_kwargs=title_kwargs, color=color,
                        hist_kwargs=hist_kwargs, range=rg, levels=levels,
                        **kwargs)
    return fig



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
import matplotlib.pyplot as plt

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
    make an interpol ator for quantity which_quantity as a function of
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
    
    fcore_in_interior, fh2o, Tirr, mass = theta 
    
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
    fcore_in_interior, fh2o, Tirr, mass = theta 
    
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
        mu = np.array([params["Mp_earth"], params["Tirr"]])
        covmat = np.zeros((2,2))
        covmat[0,0] = params["err_Mp_earth"]**2.
        covmat[1,1] = params["err_Tirr"]**2.
        
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

    x0 = np.array([0.33, 0.2, params["Tirr"], params["Mp_earth"]])
    
    params["labels"] = [r"f$_\mathrm{core}'$", r"f$_\mathrm{H_2O}$", r"T$_\mathrm{irr}$ [K]", r"M$_p$ [M$_\oplus$]"]
    
    params["pos0"] = [x0 + np.array([0.2, 0.1, params["err_Tirr"], params["err_Mp_earth"]])\
                     * np.random.randn(params["ndim"]) for i in range(params["nwalkers"])]

    if grid_lim is None:
        grid_lim = dict()
        grid_lim['fcore_in_interior'] = [0., 0.9]
        grid_lim['mass'] = [0.2, 20.]
        grid_lim['fh2o'] = [0.0, 1.0]
        grid_lim['Tirr'] = [params["Tirr_min"], 1300.]
    
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
    lp_array = np.array([lnprior(sample, params) for sample in samples])
    good_indices = np.where(np.isfinite(lp_array))[0]
    samples = samples[good_indices]
    fig = corner.corner(samples, labels=params["labels"], 
                        plot_datapoints=plot_datapoints, smooth=smooth,
                        show_titles=show_titles, quantiles=quantiles,
                        title_kwargs=title_kwargs, color=color,
                        hist_kwargs=hist_kwargs, range=rg, levels=levels,
                        **kwargs)
    return fig


def plot_mass_radius(samples, params, interp_r, interp_validity):

    #%% mass radius curve
    masses_to_calc = np.logspace(np.log10(0.4), np.log10(20), 1000)
    one = np.ones_like(masses_to_calc)

    #params #samples #fcore_in_interior, fh2o, Tirr, mass (theta)
    input = np.median(samples, axis=0)

    if input[2] < 400.:
        input[2] = 400.
        print("Median irradiation temperature was less than 400 K, using 400 K for plotting")

    # fcore_in_interior, Tirr, fh2o, log10_Mass_oplus (interp)
    # parameters are: fraction of core in (core+mantle) by mass; irradiation T, water mass fraction (0.1 is 10%), log10 mass in Earth masses
    param_best = np.array([one * input[0], one * input[2], one * input[1], np.log10(masses_to_calc)]).T
    radii_best = interp_r((param_best), method="linear")
    validity_best = interp_validity((param_best), method="linear")
    ind_valid_best = np.where(validity_best < 0.5)[0]

    wmf_low = np.floor(input[1] * 10) / 10
    wmf_high = np.ceil(input[1] * 10) / 10

    param_roundlow = np.array([one * input[0], one * input[2], one * wmf_low, np.log10(masses_to_calc)]).T
    radii_roundlow = interp_r((param_roundlow), method="linear")
    validity_roundlow = interp_validity((param_roundlow), method="linear")
    ind_valid_roundlow = np.where(validity_roundlow < 0.5)[0]

    param_roundhigh = np.array([one * input[0], one * input[2], one * wmf_high, np.log10(masses_to_calc)]).T
    radii_roundhigh = interp_r((param_roundhigh), method="linear")
    validity_roundhigh = interp_validity((param_roundhigh), method="linear")
    ind_valid_roundhigh = np.where(validity_roundhigh < 0.5)[0]

    fig, ax = plt.subplots(1, 1)
    ax.plot(masses_to_calc[ind_valid_best], radii_best[ind_valid_best], label="Best Fit - WMF = " + str(round(input[1]*100, 0)) + "%", color="C0", linestyle='dashed')
    ax.plot(masses_to_calc[ind_valid_roundlow], radii_roundlow[ind_valid_roundlow], label="WMF = " + str(wmf_low*100) + "%", color="C2")
    ax.plot(masses_to_calc[ind_valid_roundhigh], radii_roundhigh[ind_valid_roundhigh], label="WMF = " + str(wmf_high*100) + "%", color="C9")

    # mass and radius values from Cadieux+ 2025
    ax.errorbar(params["Mp_earth"], params["Rp_earth"], params["err_Rp_earth"], params["err_Mp_earth"], marker="*", color="white", ecolor="black", markeredgecolor="black", capsize=2, markersize=10, ls="")

    #x axis limits
    x_min = max(1, params["Mp_earth"] - 5 * params["err_Mp_earth"])
    x_max = params["Mp_earth"] + 5 * params["err_Mp_earth"]

    if x_min > 1.0:
        x_min = 1.0
    if x_max < 3.0:
        x_max = 3.0

    #x ticks
    ax.set_xscale("log")
    positions = np.arange(int(np.floor(x_min)), int(np.ceil(x_max)) + 1)
    ax.set_xticks(positions, labels = [str(int(p)) for p in positions])
    ax.spines['top'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    #setting other labels
    ax.set_xlabel(r"Mass [M$_\oplus$]")
    ax.set_ylabel(r"Radius [R$_\oplus$]")
    ax.set_xlim(x_min, x_max)
    ax.legend(loc=2)
    ax.text(0.95, 0.95, params["fname"],
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    #param_h2o_10percent = np.array([one * 0.325, one * 400., one * 0.1, np.log10(masses_to_calc)]).T

    fig.savefig(params["path_folder_models"]  + "../smint_results/" + params["fname"] + "_mass_radius_best.png")

    return fig

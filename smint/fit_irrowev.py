# Import modules ----------
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import emcee
import corner
import matplotlib.pyplot as plt
from astropy.io import ascii
import os
from datetime import datetime  # Get current date and time


# %% utilities for interpolation


def find_radius_from_comp(path_to_models=None, interp_r=None, age=1., log10_mass=1.30103,
                          Teq=400., wmf=1., which_table="M", which_quantity="R_20mbar"):
    """
    Given a planet internal structure (h2o mass fraction), planet mass, and planet age, get the best-matching planet radius
    interp_r: interpolator to get radii from parameters
    """
    if interp_r is None:
        interp_r = make_interpolator_A25(path_to_models, which_table=which_table, which_quantity=which_quantity)
    r_earth = interp_r((age, log10_mass, Teq, wmf), method='linear')
    return r_earth


def make_interpolator_A25(path_to_models, which_table="M", which_quantity="R_20mbar"):
    """
    make an interpolator for quantity which_quantity as a function of
    ['age','mass', 'Teq','wmf']
    using the Aguichine et al. 2025 grid
    the interpolation is linear with the log10 of the planet mass
    which_quantity="R_20mbar", "R_1mibar"
    """
    physical_table = ascii.read(path_to_models + which_table + '_table_aguichine2025.csv')

    if which_quantity == "R_20mbar":
        y = np.array(physical_table['R_20mbar'])


    elif which_quantity == "R_1mibar":
        y = np.array(physical_table['R_1mibar'])

    # make array of values
    unique_wmf = np.unique(physical_table['WMF'])
    unique_teq = np.unique(physical_table['T_eq'])
    unique_mp = np.log10(np.unique(physical_table['M_p']))
    unique_age = np.unique(physical_table['Age'])

    y_reshape = y.reshape((np.size(unique_wmf), np.size(unique_teq), np.size(unique_mp), np.size(unique_age))).T

    interpolator = RegularGridInterpolator((unique_age, unique_mp, unique_teq, unique_wmf,), y_reshape,
                                           bounds_error=False)
    return interpolator


# %% emcee functions


def lnlike(theta, params, interp_r):
    """
    Log-likelihood function for emcee fit
    """

    age, wmf, Teq, mass = theta

    true_rad = params["Rp_earth"]
    true_rad_err = params["err_Rp_earth"]

    # using the result for Teq=400K at lower temperatures
    if Teq < 400.:
        Teq = 400.

    # estimate interpolated radius for these params
    radius = find_radius_from_comp(path_to_models=None, interp_r=interp_r, age=age, log10_mass=np.log10(mass), Teq=Teq, wmf=wmf)

    if np.isnan(radius):
        return -np.inf

    lnlk = -0.5 * (((true_rad - radius) / true_rad_err) ** 2)
    return lnlk


def lnprior(theta, params):
    """
    prior (using known Teq distri, mass distri, age distri, and wmf)
    """
    age, wmf, Teq, mass = theta

    use_kde = params["use_KDE_for_Mp_prior"]
    flat_age = params["flat_age"]

    if use_kde:
        kde_points = params["kde_Mp_points"]
        kde_density = params["kde_Mp_density"]
        mu_Teq = params["Teq"]
        sig_Teq = params["err_Teq"]
        if flat_age == False:
            mu_age = params["age_Gyr"]
            sig_age = params["err_age_Gyr"]
    else:
        mu = params["mu"]
        icovmat = params["icovmat"]

    grid_lim = params["grid_lim"]

    age_min = params["age_Gyr_inf"]
    age_max = params["age_Gyr_sup"]

    if (age < grid_lim['age'][0]) or (age > grid_lim['age'][1]):
        return -np.inf
    if flat_age:
        if (age < age_min) or (age > age_max):
            return -np.inf
    if (wmf < grid_lim['wmf'][0]) or (wmf > grid_lim['wmf'][1]):
        return -np.inf
    if (mass < grid_lim['mass'][0]) or (mass > grid_lim['mass'][1]):
        return -np.inf
    if (Teq < grid_lim['Teq'][0]) or (Teq > grid_lim['Teq'][1]):
        return -np.inf

    else:
        if use_kde:
            lp_Teq = -0.5 * (((Teq - mu_Teq) / sig_Teq) ** 2)
            lp_mass = np.log(np.interp(mass, kde_points, kde_density))
            if flat_age == False:
                lp_age = -0.5*(((age - mu_age) / sig_age) ** 2)
            else:
                lp_age = 0
            return lp_Teq + lp_mass + lp_age
        else:
            if flat_age == False:
                arr = np.array([mass, Teq, age])
            else:
                arr = np.array([mass, Teq])
            diff = arr - mu
            return -np.dot(diff, np.dot(icovmat, diff)) / 2.0


def lnprob(theta, params, interp_r):
    """
    Log-probability function
    """
    lp = lnprior(theta, params)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike(theta, params, interp_r)


# %% setup and run interpolator

def setup_priors(params):
    """
    input: params of the fit
    output: mean and covariance matrix for gaussian priors,
    lower and upper bound on the flat prior on the age if not a gaussian prior
    """
    flat_age = params["flat_age"]

    if params["use_KDE_for_Mp_prior"]:
        params["kde_Mp_points"] = np.load(params["path_file_kde_points"])
        params["kde_Mp_density"] = np.load(params["path_file_kde_density"])

        params["icovmat"] = None
        params["mu"] = None
        params["covmat"] = None

        if flat_age == True:
            params["age_min"] = params["age_Gyr_inf"]
            params["age_max"] = params["age_Gyr_sup"]

    else:
        if flat_age == False:
            mu = np.array([params["Mp_earth"], params["Teq"], params["age_Gyr"]])
            covmat = np.zeros((3, 3))
            covmat[0, 0] = params["err_Mp_earth"] ** 2.
            covmat[1, 1] = params["err_Teq"] ** 2.
            covmat[2, 2] = params["err_age_Gyr"] ** 2.
        else:
            mu = np.array([params["Mp_earth"], params["Teq"]])
            covmat = np.zeros((2, 2))
            covmat[0, 0] = params["err_Mp_earth"] ** 2.
            covmat[1, 1] = params["err_Teq"] ** 2.
            params["age_min"] = params["age_Gyr_inf"]
            params["age_max"] = params["age_Gyr_sup"]

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
    if None, uses the bounds from the Aguichine et al. (2025) grid
    output: initial positions of the walkers and labels for the fitted para
    """
    now = datetime.now()  # Format as YYYYMMDD_HHhMMmSSs
    Datestr = now.strftime("%Y%m%d_%Hh%Mm%Ss")
    print(Datestr)  # Output: 20260715_170942s (based on current time)
    params["outputdir_fullpath"] = params["outputdir"] + "/" + params["fname"] + "_" + Datestr
    os.makedirs(params["outputdir_fullpath"], exist_ok=True)

    flat_age = params["flat_age"]
    params["ndim"] = 4

    if flat_age == False:
        initial_age = params["age_Gyr"]
        age_spread = params["err_age_Gyr"]
    else:
        initial_age = np.mean([params["age_Gyr_inf"], params["age_Gyr_sup"]])
        age_spread = 0.2

    x0 = np.array([initial_age, 0.2, params["Teq"], params["Mp_earth"]])

    params["labels"] = [r"$\mathrm{Age}$ [Gyr]", r"f$_\mathrm{H_2O}$", r"T$_\mathrm{eq}$ [K]", r"M$_p$ [M$_\oplus$]"]

    params["pos0"] = [x0 + np.array([age_spread, 0.1, params["err_Teq"], params["err_Mp_earth"]]) \
                  * np.random.randn(params["ndim"]) for i in range(params["nwalkers"])]

    if grid_lim is None:
        grid_lim = dict()
        grid_lim['age'] = [0.001, 20.]
        grid_lim['wmf'] = [0.001, 1.]
        grid_lim['mass'] = [0.2, 20.]
        grid_lim['Teq'] = [params["Teq_min"], 1500.]

    params["grid_lim"] = grid_lim

    return params


def run_fit(params, interp_r):
    """
    Run the emcee fit using the previously-set up priors and params
    Interpolators: generated using make_interpolator_A25()
    returns the emcee sampler
    """

    print("\nSetting up the sampler...")
    sampler = emcee.EnsembleSampler(params["nwalkers"], params["ndim"], lnprob,
                                    args=(params, interp_r))

    print("\nRunning the emcee fit...")
    sampler.run_mcmc(params["pos0"], params["nsteps"], progress=True)

    if params["save"]:
        print("\nSaving the results...")
        np.save(params["outputdir_fullpath"] + "/" + params["fname"] + '_chains.npy', sampler.chain)

    return sampler


# %% post-processing

def plot_corner(samples, params, plot_datapoints=False, smooth=1.,
                quantiles=[0.16, 0.5, 0.84], title_kwargs={'fontsize': 14},
                hist_kwargs={"linewidth": 3}, rg=None,
                show_titles=True, levels=(0.393, 0.865, 0.989), **kwargs):
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


def plot_mass_radius(samples, params, interp_r):
    # %% mass radius curve
    masses_to_calc = np.logspace(np.log10(0.4), np.log10(20), 1000)
    one = np.ones_like(masses_to_calc)

    #Age, WMF, Teq, mass (theta)
    input = np.median(samples, axis=0)

    if input[2] < 400.:
        input[2] = 400.
        print("Median equilibrium temperature was less than 400 K, using 400 K for plotting")

    # Age, mass, Teq, WMF (interp)
    # parameters are: Age in Gyr; log10 mass in Earth masses, equilibrium T, water mass fraction (0.1 is 10%),
    param_best = np.array([one * input[0], np.log10(masses_to_calc), one * input[2], one * input[1]]).T
    radii_best = interp_r((param_best), method="linear")

    # Creating comparison curves
    wmf_low = np.floor(input[1] * 10) / 10
    wmf_high = np.ceil(input[1] * 10) / 10

    param_roundlow = np.array([one * input[0], np.log10(masses_to_calc), one * input[2], one * wmf_low]).T
    radii_roundlow = interp_r((param_roundlow), method="linear")

    param_roundhigh = np.array([one * input[0], np.log10(masses_to_calc), one * input[2], one * wmf_high]).T
    radii_roundhigh = interp_r((param_roundhigh), method="linear")

    # Plotting the curves
    fig, ax = plt.subplots(1, 1)
    ax.plot(masses_to_calc, radii_best, label = "Best Fit - WMF = " + str(round(input[1] * 100, 0)) + "%", color = "C0", linestyle = 'dashed')
    ax.plot(masses_to_calc, radii_roundlow, label="WMF = " + str(wmf_low * 100) + "%", color="C2")
    ax.plot(masses_to_calc, radii_roundhigh, label="WMF = " + str(wmf_high * 100) + "%", color="C9")

    # mass and radius values to create error bar
    ax.errorbar(params["Mp_earth"], params["Rp_earth"], params["err_Rp_earth"], params["err_Mp_earth"], marker="*",
                color="white", ecolor="black", markeredgecolor="black", capsize=2, markersize=10, ls="")

    # x axis limits
    x_min = max(1, params["Mp_earth"] - 5 * params["err_Mp_earth"])
    x_max = params["Mp_earth"] + 5 * params["err_Mp_earth"]

    if x_min > 1.0:
        x_min = 1.0
    if x_max < 3.0:
        x_max = 3.0

    # x ticks
    ax.set_xscale("log")
    positions = np.arange(int(np.floor(x_min)), int(np.ceil(x_max)) + 1)
    ax.set_xticks(positions, labels=[str(int(p)) for p in positions])
    ax.spines['top'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    # setting other labels
    ax.set_xlabel(r"Mass [M$_\oplus$]")
    ax.set_ylabel(r"Radius [R$_\oplus$]")
    ax.set_xlim(x_min, x_max)
    ax.legend(loc=2)
    ax.text(0.95, 0.95, params["fname"],
    transform = ax.transAxes,
    verticalalignment = 'top',
    horizontalalignment = 'right',
    bbox = dict(boxstyle='round', facecolor='white', alpha=0.5))


    fig.savefig(params["outputdir_fullpath"] + "/" + params["fname"] + "_mass_radius_best.png")


    return fig

# Import modules ----------
import numpy as np
from smint import fit_irrowev
import pickle
import os
import configparser
import argparse
from copy import deepcopy
import sys


# %% The main code starts here


def main(argv):
    '''
    Example:
    python calc_irrowev_planet_and_plots_example.py template_ini_irrowev.ini
    '''

    if len(argv) > 1:
        iniFile = argv[1]
    else:
        iniFile = 'template_ini_irrowev.ini'

    if not os.path.exists(iniFile):
        raise FileNotFoundError('USER ERROR: iniFile does not exist.')
    config = configparser.ConfigParser()
    config.read(iniFile)

    # %% Inputs for fit

    print('\nReading in the inputs...')

    parser = argparse.ArgumentParser(description='Inputs for the code.')

    parser.add_argument('-path_folder_models', help='path to folder containing Aguichine models', default=config.get('paths', 'path_folder_models'))
    parser.add_argument('-outputdir', help='saving path (OR path to chains if run_fit==False)', default=config.get('paths', 'outputdir'))
    parser.add_argument('-fname', help='identifier for this fit (used for saving)', default=config.get('paths', 'fname'))

    parser.add_argument('-Mp_earth', help='planet mass (in Mearth)', default=config.getfloat('physical params', 'Mp_earth'))
    parser.add_argument('-err_Mp_earth', help='planet mass uncertainty (in Mearth)', default=config.getfloat('physical params', 'err_Mp_earth'))
    parser.add_argument('-use_KDE_for_Mp_prior', help='bool. if True, use KDE prior on the mass', default=config.getboolean('physical params', 'use_KDE_for_Mp_prior'))
    parser.add_argument('-path_file_kde_points', help='path to npy array of points where mass KDE is evaluated (in Mearth)', default=config.get('physical params', 'path_file_kde_points'))
    parser.add_argument('-path_file_kde_density', help='path to npy array of KDE evaluated at kde_points', default=config.get('physical params', 'path_file_kde_density'))
    parser.add_argument('-Rp_earth', help='planet radius (in Rearth)', default=config.getfloat('physical params', 'Rp_earth'))
    parser.add_argument('-err_Rp_earth', help='planet radius uncertainty (in Rearth)', default=config.getfloat('physical params', 'err_Rp_earth'))
    parser.add_argument('-Teq', help='planet equilibrium T (in K)', default=config.getfloat('physical params', 'Teq'))
    parser.add_argument('-err_Teq', help='planet equilibrium T uncertainty (in K)', default=config.getfloat('physical params', 'err_Teq'))
    parser.add_argument('-stellar_type', help='stellar type (M or G)', default=config.get('physical params', 'stellar_type'))
    parser.add_argument('-radius_pressure', help='for radius at 20 mbar write as r_20mbar, for radius at 1 microbar write as r_1mibar)', default=config.get('physical params', 'radius_pressure'))

    parser.add_argument('-Teq_min', help='lower bound of Teq prior (in K)', default=config.getfloat('MCMC params', 'Teq_min'))
    parser.add_argument('-age_Gyr_inf', help='lower bound on system age in Gyr for flat age prior', default=config.getfloat('physical params', 'age_Gyr_inf'))
    parser.add_argument('-age_Gyr_sup', help='upper bound on system age in Gyr for flat age prior', default=config.getfloat('physical params', 'age_Gyr_sup'))
    parser.add_argument('-age_Gyr', help='mean of gaussian prior on system age in Gyr', default=config.getfloat('physical params', 'age_Gyr'))
    parser.add_argument('-err_age_Gyr', help='std of gaussian prior on system age in Gyr', default=config.getfloat('physical params', 'err_age_Gyr'))
    parser.add_argument('-flat_age', help='bool. if True, use flat prior on stellar age', default=config.getboolean('MCMC params', 'flat_age'))
    parser.add_argument('-nsteps', help='number of MCMC steps', default=config.getint('MCMC params', 'nsteps'))
    parser.add_argument('-ndim', help='number of fitted params', default=4)
    parser.add_argument('-nwalkers', help='number of MCMC walkers', default=config.getint('MCMC params', 'nwalkers'))
    parser.add_argument('-run_fit', help='bool. if True, runs the MCMC; otherwise, postprocess an existing fit', default=config.getboolean('MCMC params', 'run_fit'))
    parser.add_argument('-frac_burnin', help='fraction of the chains to be discarded as burn-in [range 0--1]', default=config.getfloat('MCMC params', 'frac_burnin'))

    parser.add_argument('-hist_color', help='color in histograms and corner', default=config.get('plotting', 'hist_color'))
    parser.add_argument('-plot_corner', help='bool. if True, generate corner plot', default=config.getboolean('plotting', 'plot_corner'))
    parser.add_argument('-plot_mass_radius', help='bool. if True, generate mass radius plot', default=config.getboolean('plotting', 'plot_mass_radius'))

    args, unknown = parser.parse_known_args()

    # make the params dict from the parser object

    params = deepcopy(args.__dict__)

    params["postprocess_oldfit"] = (params["run_fit"] == False)  # if True, no MCMC is run and old chains are loaded
    params["save"] = (params["run_fit"] == True)  # if True, save chains to npy files
    print(params)

    # %% End of user input

    # %% Setting up the fit

    print('\nSetting up the fit...')
    params = fit_irrowev.setup_priors(params)

    params = fit_irrowev.ini_fit(params)

    print('\nGenerating interpolator for radius...')
    interp_r = fit_irrowev.make_interpolator_A25(params["path_folder_models"],
                                                 which_table=params["stellar_type"],
                                                 which_quantity=params["radius_pressure"])

    if params["save"]:
        # save params dictionary
        f = open(params["outputdir"] + params["fname"] + "_params" + ".pkl", "wb")
        pickle.dump(params, f)
        f.close()

    # %% Run the fit

    if params["run_fit"] == True and params["postprocess_oldfit"] == False:
        print('\nRunning the fit...')
        sampler = fit_irrowev.run_fit(params, interp_r)

        print('\nExtracting samples...')
        samples = sampler.chain[:, int(params["frac_burnin"] * params["nsteps"]):, :].reshape((-1, params["ndim"]))

    # %% If loading from an old fit

    if params["postprocess_oldfit"]:
        print('\nLoading chains from previous fit...')
        samples = np.load(params["outputdir"] + params["fname"] + '_chains.npy')
        samples = samples[:, int(params["frac_burnin"] * samples.shape[1]):, :].reshape((-1, params["ndim"]))

    # %% corner plot for each
    if params["plot_corner"]:
        print('\nGenerating corner plot...')
        fig = fit_irrowev.plot_corner(samples, params)
        fig.savefig(params['outputdir'] + params["fname"] + '_corner.png')

    # #%% mass radius curve
    if params["plot_mass_radius"]:
        print('\nPlotting mass radius curves...')
        fig = fit_irrowev.plot_mass_radius(samples, params, interp_r)
        fig.savefig(params["path_folder_models"] + "../smint_results/" + params["fname"] + "_mass_radius.png")


# %%


if __name__ == "__main__":
    main(sys.argv)
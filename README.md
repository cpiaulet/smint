# smint
*smint* (Structure Model INTerpolator) is a tool to obtain posterior distributions on the core mass fraction (CMF, now also implemented for rocky planets with no atmosphere!) as well as the H/He or H2O mass fraction of a planet based upon interpolation onto the Lopez & Fortney (2014), Zeng (2016) and Aguichine et al. (2021) model grids. 

If you use this code, please cite Caroline Piaulet as well as the paper describing the grid of interest: Lopez & Fortney (2014), Zeng et al. (2016) and/or Aguichine et al. (2021): 
* https://ui.adsabs.harvard.edu/abs/2021AJ....161...70P/abstract (first paper describing the code) 
* https://ui.adsabs.harvard.edu/abs/2014ApJ...792....1L/abstract
* https://ui.adsabs.harvard.edu/abs/2016ApJ...819..127Z/abstract
* https://ui.adsabs.harvard.edu/abs/2021ApJ...914...84A/abstract

Feel free to contribute to this tool by raising issues or submitting pull requests. Any questions or requests can also be addressed to caroline.piaulet@umontreal.ca.

## Installation
You can install *smint* from GitHub:

    git clone https://github.com/cpiaulet/smint.git
    cd smint
    python setup.py install

### Dependencies
The dependencies of *stctm* are *NumPy*, *scipy*, *emcee*, *corner*, *astropy*, *pickle*, *configparser* and *argparse*.

### Example
Copy the three folders in ```smint/example/``` wherever in your installation you want to run your code;

#### Standalone interpolator
You can run ```smint_analysis/example_interpolator.py``` in order to get familiar with the interpolation tool.

If instead you want to compute the distributions of iron-rich core mass fraction, water or H/He mass fractions given some planet and system parameters, you can use as reference one of the four other example scripts. 


#### MCMC fits
The example scripts in smint_analysis/ (```calc_X_planet_and_plots_example.py```) illustrate respectively how to compute the distribution of H/He (```X=fhhe```) or water mass fractions from the Zeng et al. 2016 water models (```X=fh2o```), the Aguichine et al. 2021 irradiated ocean worlds models (```X=irrow```), or the CMF of a planet with no envelope (```X=cmf```) given basic planet parameters. Parameters of the planet, star, MCMC fits and plots can be modified from the default values using a config file, examples of which are provided for each type of fit as ```template_ini_X.ini```. 

Here is an example one-liner to run a fit of the H/He mass fraction of a planet using planet parameters in the template_ini_hhe.ini config file:

    python calc_fhhe_planet_and_plots_example.py template_ini_hhe.ini

A few additional tips:
* The "ini file" (.ini) contains all the user input necessary to run the code, including planet and mcmc parameters, saving path and plotting options
* The path to an "ini file" needs to be provided to the python (.py) script (if different from the default) *or* provided as an argument if the script is run from the command line
* Any parameter in the ini file can be modified from the default using the command line (instead of modifying the file). For instance, if you want to run the same fit as above, but only modify the path to the output directory, you can do it as follows:
```
    python calc_fhhe_planet_and_plots_example.py template_ini_hhe.ini -outputdir=../smint_results/newfit/
```
* The python files can be run as long as the relative path to the models and results directories are correct;
* Each script illustrates an example of how to initialize, run and postprocess a fit using *smint*;
* The output files (pickle file recording the fit parameters, MCMC chains, csv file containing median and percentile values, corner plots) which are produced when running the example scripts as is are already in smint/example/smint_results ("test_new" files for the H/He fraction, "test_fh2o" files for the H2O mass fraction using the Zeng et al. (2016) models and "test_irrow" files for the Aguichine et al. (2021) models).

#### Post-processing

*smint* now offers postprocessing capabilities to extract statistics or make plots from already-existing chains in the standard *smint* format. To do this, the ```run_fit``` parameter in the .ini file must be set to ```False``` before running the python script.

### Planet models and implementation

#### Planet mass prior

By default, the ```use_KDE_for_Mp_prior``` parameter in the config file is set to ```False```. In this configuration, a Gaussian prior is assumed for the planet mass, with mean ```Mp_earth``` and standard deviation ```err_Mp_earth```.

As an alternative, *smint* offers the possibility of providing a kernel density estimation of the planet mass, which will be used by the code instead as a prior if  ```use_KDE_for_Mp_prior = True```. In that case, the user needs to specify the following relative paths:
* ```path_file_kde_points```: path to a npy file containing a single array of x values for the KDE, in units of Earth masses;
* ```path_file_kde_density```: path to a npy file containing a single array of KDE estimates of the probability evaluated at the x values from the "points" array.


#### H/He mass fraction fits - options for MCMC priors

If you want to explore a fit to the H/He mass fraction using a uniform prior on the base-10 log of the H/He mass fraction (the default is a uniform linear prior), set ```log_fenv_prior = True``` in the config file. Note that in this case, the interpolation within the Lopez and Fortney (2014) models grid is also linear with the base-10 log of fHHe.

Because of thermal evolution, the predicted radii for a planet of the same mass, irradiation and composition vary as a function of stellar age. The default prior on the stellar age is uniform, with bounds of ```age_Gyr_inf``` and ```age_Gyr_sup``` (1 to 10 Gyr in the example). Alternatively, if the stellar age is well constrained by observations, one can instead use a Gaussian prior with mean ```age_Gyr``` and uncertainty ```err_age_Gyr```, which will be used if the user sets ```flat_age = False```. When using a Gaussian prior, the models with ages beyond the limits of the Lopez and Fortney (2014) grid (0.1 to 10 Gyr) will still be associated with zero probability.

#### Mass-radius curves for irradiated ocean worlds

When fitting for the water mass fraction of a close-in planet, I advise for using the Aguichine et al. (2021) models ("irrow" files in *smint*) rather than the Zeng et al. (2016) models. Beyond the additional physics (careful consideration of the state of the water layer), they also have the advantage of providing models for any relative amounts of silicates vs. iron in the core. The parametrization is as follows:
* f'core tracks the amount of iron in the core (in mass fraction)
* fh2o tracks the mass fraction of water for the bulk planet
For instance, if f'core=0.5 and fh2o=0.33, the planet is composed of 1/3 by mass water, 1/3 by mass iron, 1/3 by mass silicates (iron and silicates being in equal amounts in the core).

A few noteworthy details about the implementation:
* The radii are calculated, for a given composition, mass and irradiation temperature, using scaling relations. However, the code will check the bank of actual model calculations for whether or not a given set of input parameters yielded a physical model. If not, the returned probability will be zero.
* The models only go down to irradiation temperatures of 400 K. Therefore, I recommend using 400 K (as is the default) as the lower bound on the prior on the irradiation temperature. However, if you wish to run a fit for a planet that has a lower irradiation temperature, you can lower this bound using the ```Tirr_min``` parameter in the .ini file. In practice, for Tirr below 400 K, radii will be computed as if Tirr=400 K.
* The Aguichine et al. (2021) models go down to 10% water by mass. To extend the prior range to 0% water, predicted radii are interpolated with the pure rock/iron mass-radius relations from Zeng (http://www.astrozeng.com/).


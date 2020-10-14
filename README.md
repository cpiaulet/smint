# smint
*smint* (Structure Model INTerpolator) is a tool to obtain posterior distributions on the H/He or H2O mass fraction of a planet based upon interpolation onto the Lopez & Fortney (2014) and Zeng (2016) model grids.

If you use this code, please cite Caroline Piaulet, Lopez & Fortney (2014) and Zeng et al. (2016): 
https://ui.adsabs.harvard.edu/abs/2014ApJ...792....1L/abstract
https://ui.adsabs.harvard.edu/abs/2016ApJ...819..127Z/abstract

Feel free to contribute to this tool by raising issues or submitting pull requests.

## Installation
You can install *smint* from GitHub:

    git clone https://github.com/cpiaulet/smint.git
    cd smint
    python setup.py install

### Dependencies
The dependencies of *smint* are *NumPy*, *scipy*, *emcee*, *corner* and *astropy*.

### Example
Copy the three folders in smint/example/ wherever in your installation you want to run your code;

#### Standalone interpolator
You can run smint_analysis/example_interpolator.py in order to get familiar with the interpolation tool.

If instead you want to compute the distributions of water or H/He mass fractions given some planet and system parameters, you can use as reference one of the two other example scripts

#### MCMC fits
The two other example scripts in smint_analysis/ illustrate respectively how to compute the distribution of H/He (fhhe) or water (fh2o) mass fractions given a planet mass and radius (and a few other parameters for the H/He mass fraction).

For each of these scripts:
* They can be run as long as the relative path to the models and results directories are correct;
* Each illustrates an example of how to initialize, run and postprocess a fit using *smint*;
* The output files (pickle file recording the fit parameters, MCMC chains and corner plots) which should be produced by running the example scripts as is are already in smint/example/smint_results ("test_new" files for the H/He fraction and "test_fh2o" files for the H2O mass fraction).


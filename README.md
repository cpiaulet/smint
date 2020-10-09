# smint
*smint* (Structure Models INTerpolator) is a tool to obtain posterior distributions on the envelope mass fraction of a planet based upon interpolation onto the Lopez & Fortney (2014) model grids.

If you use this code, please cite Caroline Piaulet as well as Lopez & Fortney (2014): https://ui.adsabs.harvard.edu/abs/2014ApJ...792....1L/abstract

Feel free to contribute to this tool by raising issues or submitting pull requests.

## Installation
You can install *smint* from GitHub:

    git clone https://github.com/cpiaulet/smint.git
    cd smint
    python setup.py install

### Dependencies
The dependencies of *smint* are NumPy, scipy, emcee, corner and astropy.

### Example
* Copy the three folders in smint/example/ wherever in your installation you want to run your code;
* The script in smint_analysis/ can be run as long as the relative path to the models and results directories are correct;
* It illustrates an example of how to initialize, run and postprocess a fit using *smint*;
* The output files (chains and corner plots) which should be produced by running the example script as is are already in smint/example/smint_results.


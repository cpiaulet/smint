#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 08:36:48 2020

@author: caroline

Estimate planet radius for a given set of parameters and 
HHe or water mass fraction, using Lopez&Fortney 2014 and
Zeng et al. 2016

Example script
"""

# Import modules ---------- 
import numpy as np
from smint import fit_fh2o, fit_fhhe
import astropy.io as aio 

#%% Setup path

path_folder_models = "../smint_models/"

#%% Using HHe mass fraction

path_file = path_folder_models + 'master_table_LF14_20201014.csv'
t = aio.ascii.read(path_file)
R_array = np.load(path_folder_models + 'LF14_20201014.npy')
interp_hhe = fit_fhhe.make_interpolator_LF14(t, R_array, log_fenv_prior=False)

# met in * solar, age in Gyr, finc in units of the solar constant, log_10 mass [Mearth], envelope mass fraction in %
r_interp_hhe = fit_fhhe.find_radius_LF14_table(interp=interp_hhe, met=1., age=2., log10_finc=2.5, log10_mass=0.2, fenv=12.)
print(r_interp_hhe, 'Earth radii')

#%% Using H2O mass fraction

t_rock_h2o = aio.ascii.read(path_folder_models+"t_rock_h2o_Zeng2016.csv")
interp_h2o = fit_fh2o.make_interpolator_fh2o(t_rock_h2o)

# mass in Earth masses, fh2o is the water mass fraction in %
r_interp_h2o = fit_fh2o.find_radius_fh2o(interp=interp_h2o, mass=8., fh2o=11.)
print(r_interp_h2o, 'Earth radii')


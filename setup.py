from setuptools import setup

with open("smint/version.py", "r") as f:
    exec(f.read())

setup(name='smint',
      version=__version__,
      description='Inferring HHe and H2O mass fractions using the Lopez&Fortney 2014 and Zeng 2016 model grids',
      url='http://github.com/cpiaulet/smint',
      author='Caroline Piaulet',
      author_email='caroline.piaulet@umontreal.ca',
      license='GNU GPL v3.0',
      packages=['smint'],
      install_requires=['numpy', 'scipy', 'emcee', 'corner', 'astropy'],
      zip_safe=False)
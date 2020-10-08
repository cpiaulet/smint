from setuptools import setup

with open("smint/version.py", "r") as f:
    exec(f.read())

setup(name='smint',
      version=__version__,
      description='Inferring envelope mass fractions using the Zeng and Lopez&Fortney model grids',
      url='http://github.com/cpiaulet/smint',
      author='Caroline Piaulet',
      author_email='caroline.piaulet@umontreal.ca',
      license='MIT',
      packages=['smint'],
      install_requires=['numpy', 'scipy', 'emcee', 'corner', 'astropy'],
      zip_safe=False)
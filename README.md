# bayesn-public
A place for the public BayeSN code

*Please note: This is an experimental beta version of the code, made available for testing purposes only*

Developed and maintained by: Stephen Thorp (@stevet40), Gautham Narayan (@gnarayan), and Kaisey S. Mandel (@CambridgeAstroStat) on behalf of the BayeSN Team (@bayesn)

## Introduction 
This repository contains *preliminary* code which can be used to work with the BayeSN model for the SEDs of Type Ia Supernovae. Functionality is for fitting with and simulating from the BayeSN models trained as part of the following papers:
 - Mandel K.S., Thorp S., Narayan G., Friedman A.S., Avelino A., 2022, [MNRAS](https://doi.org/10.1093/mnras/stab3496), [510, 3939](https://ui.adsabs.harvard.edu/abs/2022MNRAS.510.3939M/abstract)
 - Thorp S., Mandel K.S., Jones D.O., Ward S.M., Narayan G., 2021, [MNRAS](https://doi.org/10.1093/mnras/stab2849), [508, 4310](https://ui.adsabs.harvard.edu/abs/2021MNRAS.508.4310T/abstract)

If you make use of any code from this repository, please cite Thorp et al. (2021) and Mandel et al. (2022), as above.

This repository contains a small amount of demo data. If you make use of these data, please be sure to cite the original sources!

## Repository Structure
The repository has the following structure:
 - `bayesn-public/`:
   - `BayeSNmodel/`: Core module
     - `filters/`: Various filter functions
     - `model_files/`: Files which specify pre-trained BayeSN models
     - `stan_files/`: Stan models used for fitting
     - `templates/`: Hsiao et al. (2007) SN Ia template
     - `SNmodel_pb_obsmode_map.txt`: Map specifying passband aliases and magnitude systems
     - `bayesn_model.py`: Module for working with a BayeSN model
     - `io.py`: Module for file I/O operations
     - `passband.py`: Module for passband-related operations
     - `plotting_utils.py`: Some useful plotting functions
     - `spline_utils.py`: Functions for spline related computation
     - `utils.py`: Miscellaneous useful functions
   - `demo_lcs/`: Two light curves for demonstration purposes
   - `demo_outputs/`: Pre-computed MCMC chains for demonstration purposes
   - `fit_sn.py`: Simple script for running a single light curve fit

## Getting Started
### Install
To install the code, you just need to clone this repository:
```bash
	git clone https://github.com/bayesn/bayesn-public
```

### Dependencies
In order to use this code, you will need reasonably up to date versions of:
- `numpy`
- `scipy`
- `matplotlib`
- `astropy`
- `extinction`
- `sncosmo`
- `pysynphot`\* \(\<=0.9.14\)\*\*
- `cmdstanpy` \(\>=1.0.0\)
- `cmdstan` \(!=2.27.0\)

\* This requires some extra files to be downloaded, see below.

\*\* More up-to-date versions may work. However, these use an updated version of the Vega standard. The original analyses in Mandel et al. (2022) and Thorp et al (2021) used `pysynphot` version 0.9.12. Compatibility with other versions is not guaranteed.

### `pysynphot`
To install the code, you just need to run:
```bash
	pip install pysynphot
```
You must then download some [data files](https://ssb.stsci.edu/trds/tarfiles/). An easy way to do this from the command line is to run:
```bash
	for f in synphot{1..7}.tar.gz 
	do 
		wget https://ssb.stsci.edu/trds/tarfiles/$f
		tar xzvf $f
		rm $f
	done
```

Then, to use `pysynphot`, it needs to know how to find the files you just downloaded. It does this by referring to the environment variable `PYSYN_CDBS`. This variable needs to point to the `trds` subdirectory of the untarred file structure. To set this, if you untarred in the directory `/data/username/synphot/`, you'd run the following command in bash:
```bash
	export PYSYN_CDBS=/data/username/synphot/grp/redcat/trds/
```
Alternatively, from within python, you can use the `os` module to set this at the top of a script:
```python
	import os
	os.environ["PYSYN_CDBS"] = "/data/username/synphot/grp/redcat/trds/"
```
NOTE: This should ideally be done before all other imports. So, if you're setting this within python, these lines should be placed at the very top. `pysynphot`, and modules using it, will need the environment variable to be set *at the time of import*.

### `cmdstan`
BayeSN uses CmdStanPy as an interface to Stan. To install CmdStan, you can proceed as per the [`cmdstanpy` docs](https://mc-stan.org/cmdstanpy/installation.html). You'll need to run:
```bash
	pip install cmdstanpy
```
Then, when you come to installing CmdStan, you can use the `cmdstanpy.install_cmdstan` utility from within a Python session, like so:
```python
	import cmdstanpy
	cmdstanpy.install_cmdstan()
```

Be aware that a successful install will require a `gcc` version more modern than 4.9.3.

After installing, it is recommended that you try out the ["Hello, World"](https://mc-stan.org/cmdstanpy/hello_world.html) example provided in the `cmdstanpy` documentation. Just running the two code blocks from ["The Stan model"](https://mc-stan.org/cmdstanpy/hello_world.html#the-stan-model) to ["Fitting the model"](https://mc-stan.org/cmdstanpy/hello_world.html#fitting-the-model) should be sufficient to test the install is working.

### Checking Things Work
After you've cloned the repo and done any necessary installs, try running the following command in the top level of the repository:
```bash
  python fit_sn.py --model T21 --metafile demo_lcs/meta/T21_demo_meta.txt --filters griz --opt demo_lcs/Foundation_DR1/Foundation_DR1_ASASSN-16cs.txt .
```
This will obtain the MAP estimate from fitting the Thorp et al. (2021) BayeSN model to the light curves of ASASSN-16cs (from Foundation DR1). This fit should take a few seconds (plus compile time, the first time you run Stan), and will spit out a list of estimated parameter values if it has worked successfully.

### Using BayeSN
Most functionality is provided by the `bayesn_model.SEDmodel` class within the `BayeSNmodel` module. The important functions should have docstrings describing how everything works.

An example Python script (`fit_sn.py`) is provided for fitting a single supernova light curve. Demo light curves from Mandel et al. (2022) and Thorp et al. (2021) are also providen in `demo_lcs`, along with precomputed MCMC chains in `demo_outputs`. To fit SN2005iq using the Mandel et al. (2022) model, you would need to run:
```bash
  python fit_sn.py --model M20 --metafile demo_lcs/meta/M20_demo_meta.txt --filters BVRIYJH demo_lcs/CSP/sn2005iq__u_CSP_14_B_CSP_18_g_CSP_20_V_CSP_18_r_CSP_20_i_CSP_19_H_WIRC_2_H_RC_10_J_RC1_10_J_WIRC_2_Y_RC_21__CSP3_krisciunas17.Wstd_snana.dat .
```
This would call Stan to sample from the posterior distribution of the BayeSN model parameters, conditional on SN2005iq's light curves. This will be fairly slow, as this supernova has a lot of data. For a faster fit, you can pass the `--opt` flag, which will return a MAP estimate (as per the example in the [Checking Things Work](#checking-things-work) section), rather than sampling the full posterior.

If you are writing a script/notebook which uses BayeSN, you can do it in the top level of the repository. Alternatively, you can add the `BayeSNmodel` directory to your pythonpath. The `.gitignore` is set so any `.py` or `.ipynb` files which do not have the prefix `demo_` will not be tracked. So... if you are writing your own scripts in the top level of the cloned repo, just name them anything other than `demo_*`, and git shouldn't try to interfere with them.

## Acknowledgements
Special thanks to:
 - Sam M. Ward and Suhail Dhawan for testing the code.
 - Arturo Avelino and Andrew S. Friedman for preparing the files for the Mandel et al. (2022) training set 
 - David O. Jones and the Foundation Supernova Survey team for making the Foundation DR1 light curves available.

# -*- coding: UTF-8 -*-
"""
A minimal script which runs photometric distance fits for a single 
supernova.
"""

#Check the PYSYN_CDBS variable is set
import os
if "PYSYN_CDBS" not in os.environ:
	raise OSError("PYSYN_CDBS environment variable is not set!")

#Imports
import argparse
import numpy as np
from BayeSNmodel import bayesn_model, io

#Command line arguments
parser = argparse.ArgumentParser(description="BayeSN Photometric Distance Fitting Script")
parser.add_argument("--model", default="M20", help="Name of a predefined BayeSN model, or a path to a set of BayeSN model files. Used to intialise the SEDmodel class. Defaults to using the Mandel+20 (M20) model.")
parser.add_argument("--metafile", default=None, help="Path to a file specifying additional metadata for the supernova. Passed to BayeSNmodel.io.read_snana_lcfile. Defaults to None.")
parser.add_argument("--sampname", default=None, help="Sample name. Generally not required, but needed for Foundation_DR1 to trigger specific preprocessing. Defaults to None.")
parser.add_argument("--filters", default=None, help="Case insensitive string listing initials of photometric passbands to include. Any passband whose first initial matches a letter in the string will be included (unless this clashes with the wavelength range over which the model is defined). If not provided, fits every filter available.")
parser.add_argument("--fittmax", default=False, type=float, help="Fit for time of maximum. If not provided, time of maximum will be fixed to the value given by a supernova's SEARCH_PEAKMJD or PEAKMJD metadata entry. If provided, the supplied value will be interpreted as the +/- shift in phase from PEAKMJD or SEARCH_PEAKMJD which will be allowed when fitting for time of maximum.")
parser.add_argument("--tmax", default=None, type=float, help="Initial guess for time of B-band maximum (overrides PEAKMJD/SEARCH_PEAKMJD)")
parser.add_argument("--pbtrunc", default=0.002, type=float, help="Fraction of total transmission at which to truncate passband transmission curves. Deafult is 0.002 (0.2 percent).")
parser.add_argument("--nwarmup", default=250, type=int, help="Number of warmup iterations to use in Stan fit (HMC only). Default is 250.")
parser.add_argument("--nsampling", default=250, type=int, help="Number of post-warmup iterations to use in Stan fit (HMC only). Default is 250.")
parser.add_argument("--nchains", default=4, type=int, help="Number of MCMC chains to use in Stan fit (HMC only). Default is 4. Optimization will only use 1.")
parser.add_argument("--nworkers", default=None, type=int, help="Number of cores to use (HMC only). Default is to use 1 core per chain (multiple cores per chain is not possible at the moment).")
parser.add_argument("--nopbar", action="store_true", help="Turn off live updating progress bar (recommended if this script is being nohupped/having output piped to a file). HMC only.")
parser.add_argument("--cnsl", action="store_true", help="Show cmdstan console output.")
parser.add_argument("--writestan", action="store_true", help="If provided, cmdstan output files will be written to a directory beneath the provided savepath, along with the .npy file containing the post warmup MCMC samples. Otherwise, only the .npy file will be written, directly in the savepath.")
parser.add_argument("--writesummary", action="store_true", help="If requested, a parameter summary .txt table will be written alongside the .npy file.")
parser.add_argument("--opt", action="store_true", help="Run in optimizer mode, to find the MAP. If not provided, the default will be to run a full HMC exploration of the posterior.")
parser.add_argument("loadpath", help="Path to an SNANA format light curve file.")
parser.add_argument("savepath", help="Path to a directory where MCMC chains will be saved as .npy files.")
args = parser.parse_args()

#Check the savepath exists
if not os.path.exists(args.savepath):
	raise ValueError("Provided savepath does not exist!")

#Set number of cores to use
if args.nworkers is None or args.nworkers > args.nchains:
	n_workers = args.nchains
else:
	n_workers = args.nworkers

#Set fitting mode
if args.opt is True:
	fit_mode = "opt"
else:
	fit_mode = "hmc"

#Time of maximum to be fit for?
if args.fittmax is False:
	fix_tmax = True
	phase_shift = 0
else:
	fix_tmax = False
	phase_shift = args.fittmax

#Load light curve
sn, lc = io.read_snana_lcfile(args.loadpath, sampname=args.sampname, metafile=args.metafile)

#Discard any unwanted filters
if args.filters is not None:
	flt_mask = np.zeros((len(lc),), dtype=bool)
	for flt in args.filters:
		#Case insensitive match
		flt_mask += np.char.upper(lc["flt"].astype('<U1')) == flt.upper()
	lc = lc[flt_mask]

print("==========\n{}\n==========".format(sn))
print("Passbands included:", np.unique(np.array(lc["flt"])))

#Initialise model and compile Stan code
b = bayesn_model.SEDmodel(model=args.model, compile=True, fix_tmax=fix_tmax)
if args.pbtrunc != 0.002:
	b.load_passbands(cutoff_factor=args.pbtrunc)

#Create a subdirectory, if stan output has been requested
if args.writestan:
	savepath = os.path.join(args.savepath, sn)
	cmdstan_output_dir = savepath
	if not os.path.exists(savepath):
		os.mkdir(savepath)
else:
	savepath = args.savepath
	cmdstan_output_dir = None

#Fit the light curves
fit, samples, summary = b.fit_supernova(lc, tmax=args.tmax, fix_tmax=fix_tmax, phase_shift=phase_shift, n_warmup=args.nwarmup, n_sampling=args.nsampling, n_chains=args.nchains, n_workers=n_workers, show_progress=(not args.nopbar), show_console=args.cnsl, cmdstan_output_dir=cmdstan_output_dir, fit_mode=fit_mode)

#Save the chains for postprocessing
if fit_mode == "hmc":
	np.save("{}/{}_chains".format(savepath, sn), samples)
else:
	np.save("{}/{}_result".format(savepath, sn), samples)

#Save a summary, if requested
if args.writesummary:
	with open("{}/{}_summary.txt".format(savepath, sn), "w") as sf:
		print(summary.to_string(), file=sf)
# -*- coding: UTF-8 -*-
"""
BayeSN Utility Functions. Defines a few utility functions which carry out
standard operations requied by BayeSN.
"""

import numpy as np
import extinction
from scipy.optimize import minimize

def interpolate_passband(wave_int, wave, thru, z):
	"""
	Interpolates a passband onto a provided (rest-frame) wavelength grid.

	Given a throughput curve which is known at particular observer frame
	wavelengths, and a grid of rest frame wavelengths corresponding to a
	particular redshift, z, computes the rest frame wavelengths which fall
	within this filter at observation time, and interpolates the trhoughput onto
	these.

	Parameters
	----------
	wave_int : :py:class:`numpy.array`
		Grid of rest frame wavelengths to interpolate onto.
	wave : :py:class:`numpy.array`
		Observer frame wavelengths at which filter transmission is known
	thru : :py:class:`numpy.array`
		Filter throughput at wavelengths in wave
	z : float
		Redshift which rest frame wavelengths correspond to. Set this to 0
		if you wish to input wave_int already in observer frame.
	
	Returns
	-------
	l_r : :py:class:`numpy.array`
		Subset of rest frame wavelengths from the input grid which fall
		within the passband at observation time.
	tp : :py:class:`numpy.array`
		Filter throughput at l_r        
	"""
	wave_obs = wave_int*(1+z)
	mask = (wave_obs < max(wave))*(wave_obs > min(wave))
		
	return wave_int[mask], np.interp(wave_obs[mask], wave, thru)

def interpolate_hsiao(t_int, l_int, t_hsiao, l_hsiao, f_hsiao):
	"""
	Interpolates the Hsiao template to a particular time.

	Returns a slice of the Hsiao template, evaluated at an
	arbitrary time and grid of wavelengths.

	Parameters
	----------
	t_int : float
		Time at which to evaluate template
	l_int : :py:class:`numpy.array`
		Grid of rest frame wavelengths at which to return slice
	t_hsiao : :py:class:`numpy.array`
		Vector of times at which the template is known
	l_hsiao : :py:class:`numpy.array`
		Vector of wavelengths at which the template is known
	f_hsiao : :py:class:`numpy.array`
		Grid of fluxes coresponding to the provided time and 
		wavelength vectors.
	
	Returns
	-------
	f_int : :py:class:`numpy.array`
		Vector of fluxes defining the Hsiao template at wavelegths
		l_int at time t_int.     
	"""
	dt = t_hsiao[1] - t_hsiao[0]
	mask0 = (t_hsiao <= t_int)*(t_hsiao > t_int - dt)
	mask1 = (t_hsiao > t_int)*(t_hsiao <= t_int + dt)
	t0 = t_hsiao[mask0]
	t1 = t_hsiao[mask1]

	f_int = (f_hsiao[mask0, :]*(t1 - t_int) + f_hsiao[mask1, :]*(t_int - t0))/dt

	f_int = np.interp(l_int, l_hsiao, f_int.flatten())

	return f_int

def extinction_mw(wave, EBV, RV=3.1):
	"""
	Convenience function returning Milky Way extinction.

	Parameters
	----------
	wave : :py:class:`numpy.array`
		Grid of observed frame wavelengths to evaluate at.
	EBV : float
		Known Milky Way reddening value.
	RV : float
		Milky Way RV value - defaults to 3.1.
	
	Returns
	-------
	R : :py:class:`numpy.array`
		Milky Way extinction in flux units at the input wavelengths.
		Computed using Fitzpatrick99 law.
	"""
	RV = 3.1
	AV = EBV*RV
	return 10**(-0.4*extinction.fitzpatrick99(wave, AV, RV))

def distmod_err(z_cmb, z_cmb_err, sigma_pec=150):
	"""
	Function for computing external distance error.

	Calculates uncertainty in redshift derived distance modulus
	propagated from redshift and peculiar velocity uncertainties.
	See Mandel+20 eq. 26, Avelino+19 eq. 8.

	Parameters
	----------
	z_cmb : float
		Estimated CMB frame redshift
	z_cmb_err : float
		Uncertainty on CMB frame redshift
	sigma_pec : float, optional
		Peculiar velocity uncertainty. Default is 150 km/s.
	
	Returns
	-------
	mu_err : float
		Distance modulus uncertainty.
	"""

	return (5.0/(z_cmb*np.log(10)))*np.sqrt(z_cmb_err**2 + (sigma_pec/3e5)**2)

def neg_log_lkhd_sigma_pv(sigma_pv, mu_phot, mu_ext, mu_ext_err):
	"""
	Negative log likelihood for sigma_{-pv}.

	Objective function used by :py:func:`BayeSNmodel.utils.calc_sigma_pv`.
	See	Mandel+20, eq. 32.

	Parameters
	----------
	sigma_pv : float
		Value of sigma_{-pv} at which to calculate log likelihood.
	mu_phot : :py:class:`numpy.array`
		Numpy array of photometric distance estimates
	mu_ext : :py:class:`numpy.array`
		Numpy array of external distance estimates. Hubble residuals
		are given by `mu_phot - mu_ext`.
	mu_ext_err : :py:class:`numpy.array`
		Numpy array of external distance estimate uncertainties.
	
	Returns
	-------
	neg_log_lkhd : float
		Negative log likelihood.
	"""
	if sigma_pv < 0:
		return np.inf
	else:
		var = sigma_pv**2 + mu_ext_err**2
		return np.sum(0.5*np.log(2*np.pi*var) + 0.5*((mu_phot-mu_ext)**2)/var)

def calc_sigma_pv(mu_phot, mu_ext, mu_ext_err):
	"""
	Computes sigma_{-pv}.

	Calculates the level of dispersion in a set of Hubble residuals
	which is not accounted for by uncertainties in external distance
	estimates (mainly due to peculiar velocity uncertainties). See
	Mandel+20, eq. 32.

	Parameters
	----------
	mu_phot : :py:class:`numpy.array`
		Numpy array of photometric distance estimates
	mu_ext : :py:class:`numpy.array`
		Numpy array of external distance estimates. Hubble residuals
		are given by `mu_phot - mu_ext`.
	mu_ext_err : :py:class:`numpy.array`
		Numpy array of external distance estimate uncertainties.
	
	Returns
	-------
	sigma_pv : float
		Hubble residual scatter not accounted for by uncertainties in
		external distance estimates.
	"""
	guess = np.std(mu_phot-mu_ext)
	result = minimize(neg_log_lkhd_sigma_pv, guess, args=(mu_phot, mu_ext, mu_ext_err))
	return result.x[0]

def calc_rms(mu_phot, mu_ext):
	"""
	Computes raw RMS of Hubble residuals.

	Parameters
	----------
	mu_phot : :py:class:`numpy.array`
		Numpy array of photometric distance estimates
	mu_ext : :py:class:`numpy.array`
		Numpy array of external distance estimates. Hubble residuals
		are given by `mu_phot - mu_ext`.
	
	Returns
	-------
	rms : float
		Total raw RMS of Hubble residuals.
	"""
	return np.sqrt(np.mean((mu_phot - mu_ext)**2))
# -*- coding: UTF-8 -*-
"""
Instrumental throughput models and calibration and synthetic photometry
routines
"""
from __future__ import absolute_import
from __future__ import unicode_literals
import warnings
warnings.simplefilter('once')
import numpy as np
import astropy.table as at
import pysynphot as S
import scipy.interpolate as scinterp
import sncosmo
from . import io
from collections import OrderedDict

def synflux(spec, pb):
    """
    Compute the synthetic flux of spectrum ``spec`` through passband ``pb``

    Parameters
    ----------
    spec : Table
        The spectrum.
        Must have ``dtype=[('wave', '<f8'), ('flux', '<f8')]``
    pb : Table
        The passband transmission.
        Must have ``dtype=[('wave', '<f8'), ('throughput', '<f8')]``
        Must have been interpolated so that spectrum['wave'].shape == passband['wave'].shape.

    Returns
    -------
    flux : float
        The normalized flux of the spectrum through the passband

    Notes
    -----
        The passband is assumed to be dimensionless photon transmission
        efficiency.
    """
    if len(spec) != len(pb):
        pbinterp = scinterp.interp1d(pb['wave'], pb['throughput'],\
            kind='cubic', bounds_error=False, fill_value=(0., 0.))
        itput = pbinterp(spec['wave'])
    else:
        itput = pb['throughput']

    n = np.trapz(spec['flux']*spec['wave']*itput, spec['wave'])
    d = np.trapz(spec['wave']*itput, spec['wave'])
    out = n/d
    return out


def synphot(spec, pb, zp=0.):
    """
    Compute the synthetic magnitude of spectrum ``spec`` through passband ``pb``

    Parameters
    ----------
    spec : Table
        The spectrum.
        Must have ``dtype=[('wave', '<f8'), ('flux', '<f8')]``
    pb : Table
        The passband transmission.
        Must have ``dtype=[('wave', '<f8'), ('throughput', '<f8')]``
        Must have been interpolated so that spectrum['wave'].shape == passband['wave'].shape.
    zp : float, optional
        The zeropoint to apply to the synthetic flux

    Returns
    -------
    mag : float
        The synthetic magnitude of the spectrum through the passband

    See Also
    --------
    :py:func:`BayeSNmodel.passband.synflux`
    """
    flux = synflux(spec, pb)
    m = -2.5*np.log10(flux) + zp
    return m


def chop_syn_spec_pb(spec, model_mag, pb, cutoff_factor=0.002):
    """
    Trims the pysynphot bandpass pb to non-zero throughput, computes the
    zeropoint of the passband given the SED spec, and model magnitude of spec
    in the passband

    Parameters
    ----------
    spec : :py:class:`numpy.recarray`
        The spectrum. Typically a standard which has a known ``model_mag``.
        This can be a real source such as Vega, BD+174708, or one of the three
        CALSPEC standards, or an idealized synthetic source such as AB.
        Must have ``dtype=[('wave', '<f8'), ('flux', '<f8')]``
    model_mag : float
        The apparent magnitude of the spectrum through the passband.  The
        difference between the apparent magnitude and the synthetic magnitude
        is the synthetic zeropoint.
    pb : :py:class:`numpy.recarray`
        The passband transmission.
        Must have ``dtype=[('wave', '<f8'), ('throughput', '<f8')]``
    cutoff_factor : float, optional
        Fraction of max throughput to truncate at. Passband is truncated
        beyond the last wavelength where the throughput is > `cutoff_factor`
        times the maximum. Default is 0.002 (0.2% of max throughput).

    Returns
    -------
    outpb : :py:class:`numpy.recarray`
        The passband transmission with zero throughput entries trimmed.
        Has ``dtype=[('wave', '<f8'), ('throughput', '<f8')]``
    outzp : float
        The synthetic zeropoint of the passband ``pb`` such that the source
        with spectrum ``spec`` will have apparent magnitude ``model_mag``
        through ``pb``. With the synthetic zeropoint computed, the synthetic
        magnitude of any source can be converted into an apparent magnitude and
        can be passed to :py:func:`BayeSNmodel.passband.synphot`.

    See Also
    --------
    :py:func:`BayeSNmodel.passband.synphot`
    """
    # interpolate the passband onto the spectrum - use cubic by default
    pbinterp = scinterp.interp1d(pb['wave'], pb['throughput'],\
            kind='cubic', bounds_error=False, fill_value=(0., 0.))
    itput = pbinterp(spec['wave'])
    ind = itput < 0.
    itput[ind] = 0.

    outpb = at.Table((spec['wave'], itput), names=['wave','throughput'])

    max_ind = outpb['throughput'].argmax()
    max_wave = outpb['wave'][max_ind]
    max_pb = outpb['throughput'][max_ind]
    threshold = cutoff_factor*max_pb
    ind = outpb['throughput'] >= threshold
    min_wave = outpb['wave'][ind].min()
    max_wave = outpb['wave'][ind].max()
    ind = (outpb['wave'] >= min_wave) & (outpb['wave'] <= max_wave)
    outpb = outpb[ind]

    ind = (spec['wave'] >= min_wave) & (spec['wave'] <= max_wave)
    spec = spec[ind]
    itput = itput[ind]

    # compute the zeropoint
    outzp = model_mag - synphot(spec, outpb)
    d = np.trapz(spec['wave']*itput, spec['wave'])
    outpb['norm_throughput'] = outpb['throughput']*(10.**(-0.4*outzp))/d
    return outpb, outzp


def get_pbnames(lcs):
    """
    Gets the set of unique passband names ``pbnames`` from the light curves
    ``lcs``

    Parameters
    ----------
    lcs : dictionary
        Dictionary containing the light curves such as that returned by
        :py:func:`BayeSNmodel.io.read_sn_sample` or :py:func:`BayeSNmodel.io.read_sn_sample_file`
        Each entry in the dictionary must be a :py:class:`astropy.Table`

    Returns
    -------
    pbnames : list
        List of unique passband names
    """
    pbnames = set()
    for sn, lc in lcs.items():
        pbnames = pbnames | set(lc['flt'])
    pbnames = sorted(list(pbnames))
    return pbnames


def get_pbmodel(pbnames, pbfile=None, mag_type=None, mag_zero=0., cutoff_factor=0.002, verbose=False, onerror=False):
    """
    Converts passband names ``pbnames`` into passband models based on the
    mapping of name to ``pysynphot`` ``obsmode`` strings in ``pbfile``.

    Parameters
    ----------
    pbnames : array-like
        List of passband names to get throughput models for Each name is
        resolved by first looking in ``pbfile`` (if provided) If an entry is
        found, that entry is treated as an ``obsmode`` for pysynphot. If the
        entry cannot be treated as an ``obsmode,`` we attempt to treat as an
        ASCII file. If neither is possible, an error is raised.
    pbfile : str, optional
        Filename containing mapping between ``pbnames`` and ``pysynphot``
        ``obsmode`` string, as well as the standard that has 0 magnitude in the
        system (either ''Vega'' or ''AB''). The ``obsmode`` may also be the
        fullpath to a file that is readable by ``pysynphot``
    mag_type : str, optional
        One of ''vegamag'' or ''abmag''
        Used to specify the standard that has 0 magnitude in the passband.
        If ``magsys`` is specified in ``pbfile,`` that overrides this option.
        Must be the same for all passbands listed in ``pbname`` that do not
        have ``magsys`` specified in ``pbfile``
        If ``pbnames`` require multiple ``mag_types``, concatentate the output.
    mag_zero : float, optional
        Magnitude of the standard in the passband
        If ``magzero`` is specified in ``pbfile,`` that overrides this option.
        Must be the same for all passbands listed in ``pbname`` that do not
        have ``magzero`` specified in ``pbfile``
        If ``pbnames`` require multiple ``mag_zero``, concatentate the output.
    cutoff_factor : float, optional
        Fraction of max throughput to truncate at. Passband is truncated
        beyond the last wavelength where the throughput is > `cutoff_factor`
        times the maximum. Default is 0.002 (0.2% of max throughput).

    Returns
    -------
    out : dict
        Output passband model dictionary. Has passband name ``pb`` from ``pbnames`` as key.

    Raises
    ------
    RuntimeError
        If a bandpass cannot be loaded

    Notes
    -----
        Each item of ``out`` is a tuple with
            * ``pb`` : (:py:class:`numpy.recarray`)
              The passband transmission with zero throughput entries trimmed.
              Has ``dtype=[('wave', '<f8'), ('throughput', '<f8')]``
            * ``transmission`` : (array-like)
              The non-zero passband transmission interpolated onto overlapping model wavelengths
            * ``ind`` : (array-like)
              Indices of model wavelength that overlap with this passband
            * ``zp`` : (float)
              mag_type zeropoint of this passband
            * ``avgwave`` : (float)
              Passband average/reference wavelength

        ``pbfile`` must be readable by :py:func:`BayeSNmodel.io.read_pbmap` and
        must return a :py:class:`numpy.recarray`
        with``dtype=[('pb', 'str'),('obsmode', 'str')]``

        If there is no entry in ``pbfile`` for a passband, then we attempt to
        use the passband name ``pb`` as ``obsmode`` string as is.

        Trims the bandpass to entries with non-zero transmission and determines
        the ``VEGAMAG/ABMAG`` zeropoint for the passband - i.e. ``zp`` that
        gives ``mag_Vega/AB=0.`` in all passbands.

    See Also
    --------
    :py:func:`BayeSNmodel.io.read_pbmap`
    :py:func:`BayeSNmodel.passband.chop_syn_spec_pb`
    """

    # figure out the mapping from passband to observation mode
    if pbfile is None:
        pbfile = 'SNmodel_pb_obsmode_map.txt'
        pbfile = io.get_pkgfile(pbfile)

    pbdata  = io.read_pbmap(pbfile)
    pbmap   = dict(list(zip(pbdata['pb'], pbdata['obsmode'])))
    sysmap  = dict(list(zip(pbdata['pb'], pbdata['magsys'])))
    zeromap = dict(list(zip(pbdata['pb'], pbdata['magzero'])))

    # setup the photometric system by defining the standard and corresponding magnitude system
    if mag_type not in ('vegamag', 'abmag', None):
        message = 'Magnitude system must be one of abmag or vegamag'
        raise RuntimeError(message)

    try:
        mag_zero = float(mag_zero)
    except ValueError as e:
        message = 'Zero magnitude must be a floating point number'
        raise RuntimeError(message)

    # define the standards
    vega = S.Vega
    vega.convert('flam')
    ab   = S.FlatSpectrum(0., waveunits='angstrom', fluxunits='abmag')
    ab.convert('flam')

    # defile the magnitude sysem
    if mag_type == 'vegamag':
        mag_type= 'vegamag'
    else:
        mag_type = 'abmag'

    out = OrderedDict()

    for pb in pbnames:

        standard = None

        # load each passband
        obsmode = pbmap.get(pb, pb)
        magsys  = sysmap.get(pb, mag_type)
        synphot_mag = zeromap.get(pb, mag_zero)

        if magsys == 'vegamag':
            standard = vega
        elif magsys == 'abmag':
            standard = ab
        else:
            message = 'Unknown standard system {} for passband {}'.format(magsys, pb)
            raise RuntimeError(message)

        # convert the standard to a Table
        standard_flux = standard.flux
        standard_wave = standard.wave
        standard = at.Table((standard_wave, standard_flux), names=['wave','flux'])

        loadedpb = False
        # treat the bandpass as a sncosmo string
        try:
            # sncosmo bandpasses are setup for interpolation
            bp = sncosmo.get_bandpass(obsmode)
            itput = bp(standard['wave'])
            bp = S.ArrayBandpass(standard['wave'], itput, name=pb)
            loadedpb = True
        except Exception as e:
            message = 'Could not load pb {} as {} from sncosmo'.format(pb, obsmode)
            if verbose:
                warnings.warn(message, RuntimeWarning)
            loadedpb = False

        # treat the passband as a obsmode string
        if not loadedpb:
            try:
                bp = S.ObsBandpass(obsmode)
                loadedpb = True
            except ValueError:
                message = 'Could not load pb {} from sncosmo or as obsmode string {}'.format(pb, obsmode)
                if verbose:
                    warnings.warn(message, RuntimeWarning)
                loadedpb = False

        # if that fails, try to load the passband interpreting obsmode as a file
        if not loadedpb:
            try:
                bandpassfile = io.get_pkgfile(obsmode)
                bp = S.FileBandpass(bandpassfile)
                loadedpb = True
            except Exception as e:
                message = 'Could not load passband {} from sncosmo/obsmode or file {}'.format(pb, obsmode)
                if verbose:
                    warnings.warn(message, RuntimeWarning)
                loadedpb = False

        if not loadedpb:
            message = 'Could not load passband {}. Giving up.'.format(pb)
            if onerror:
                raise RuntimeError(message)
            else:
                warnings.warn(message, RuntimeWarning)
                continue
        else:
            wave = bp.wave
            throughput = bp.throughput
            bp = at.Table((wave, throughput), names=['wave','throughput'])

        ind = bp['throughput'] < 0.
        bp['throughput'][ind] = 0.

        # cut the passband to non-zero values and interpolate onto overlapping standard wavelengths
        outpb, outzp = chop_syn_spec_pb(standard, synphot_mag, bp, cutoff_factor=cutoff_factor)

        # save everything we need for this passband
        out[pb] = (outpb, outzp)
    return out

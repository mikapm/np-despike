#!/usr/bin/env python3

import numpy as np
import scipy as sp
from scipy import interpolate
from pygpd import despike
from pygpd import nigp

"""
Functions for replacing removed or missing points.
"""

def interpolate_spline(data, mask, t=None):
    """
    By Susanne

    Parameters:
        data - np.array
        mask - bool array where 0 means points to interpolate over
        t - np.array; time (x) axis 

    """
    # Make time axis if not given
    if t is None:
        t = np.arange(len(data))
    good_data_indices = np.argwhere(mask==1).transpose()[0]   
    # Find the B-spline representation of the curve of data
    data_interp_func = sp.interpolate.splrep(t[good_data_indices], 
            data[good_data_indices], s=0)                                
    # Evaluate the smoothing polynomial
    data_interp = sp.interpolate.splev(t, data_interp_func, der=0)

    return data_interp


def interpolate_cubic(data, mask, t=None, fill_value='extrapolate'):
    """
    Parameters:
        data - np.array
        mask - bool array where 0 means points to interpolate over
        t - np.array; time (x) axis 

    """
    # Make time axis if not given
    if t is None:
        t = np.arange(len(data))
    good_data_indices = np.argwhere(mask==1).transpose()[0]   
    # Get interpolation object (function)
    f_interpol = sp.interpolate.interp1d(t[good_data_indices],
            data[good_data_indices], kind='cubic', fill_value=fill_value)
    # Interpolate over missing values
    data_interp = f_interpol(t)

    return data_interp


def interpolate_linear(data, mask, t=None, fill_value='extrapolate'):
    """
    Parameters:
        data - np.array
        mask - bool array where 0 means points to interpolate over
        t - np.array; time (x) axis 
        fill_value - str; If "extrapolate", then points outside the data 
                     range will be extrapolated.

    """
    # Make time axis if not given
    if t is None:
        t = np.arange(len(data))
    good_data_indices = np.argwhere(mask==1).transpose()[0]   
    # Get interpolation object (function)
    f_interpol = sp.interpolate.interp1d(t[good_data_indices],
            data[good_data_indices], fill_value=fill_value)
    # Interpolate over missing values
    data_interp = f_interpol(t)

    return data_interp


def interpolate_gp(data, mask, use_nigp=False, chunksize=200,
        l_bounds=(5.0,30.0), use_sklearn=True, return_sigma=False):
    """
    Use GP/NIGP regression to interpolate over dropouts defined
    in mask.

    Parameters:
        data - np.array
        mask - bool array where 0 means points to interpolate over
        use_nigp - Set to True to use NIGP regression
        return_sigma - bool; Set to True to return pointwise std 
                       (sqrt of diagonal entries of cov matrix).
    """
    if return_sigma:
        data_interp, _, _, sigma, _, _, _ = despike.GP_despike(data, mask,
                chunksize=chunksize,  return_aux=True,
                length_scale_bounds=l_bounds, predict_all=True, 
                use_nigp=use_nigp, use_sklearn=use_sklearn, 
                despike=False, print_kernel=True,
                )
        return data_interp, sigma
    else:
        data_interp = despike.GP_despike(data, mask, chunksize=chunksize,
                return_aux=False, length_scale_bounds=l_bounds, 
                predict_all=True, use_nigp=use_nigp,
                use_sklearn=use_sklearn, despike=False, print_kernel=True
                )
        return data_interp


def replace_data(signal, mask, t=None, method='linear', **gpargs):
    '''
    By Susanne
    
    Remove only spikes by deconvolution and interpolation,
    
    Parameters:
            mask                mask for interpolation/recovery
            **gpargs            kwargs for interpolate_gp()

    Output:
            data_rep            repaired data
    '''
    
    data = signal.copy()

    # Make time axis if not given
    if t is None:
        t = np.arange(len(data))

    if method == 'linear':
        return interpolate_linear(data, mask)
    elif method == 'spline':
        return interpolate_spline(data, mask)
    elif method == 'gp':
        return interpolate_gp(data, mask, use_nigp=False, **gpargs)
    elif method == 'nigp':
        return interpolate_gp(data, mask, use_nigp=True, use_sklearn=False, **gpargs)


#!/usr/bin/env python3
import numpy as np
from scipy.special import erfinv
from scipy.interpolate import CubicSpline
from statsmodels import robust
import matplotlib.pyplot as plt
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from itertools import chain
from astropy.stats import mad_std
import nigp
import replace
import spectra


def GP_despike(ts, dropout_mask=None, chunksize=200, overlap=None, return_aux=True,
        length_scale_bounds=(5.0, 30.0), predict_all=True, use_nigp=False, kernel=None,
        use_sklearn=True, despike=True, score_thresh=0.995, print_kernel=True):
    """
    Non-parametric despiking and spike replacement based on fitting
    a Gaussian process to the data, following 
    
    Malila et al. (2021, Jtech):
    "A nonparametric, data-driven approach to despiking ocean surface wave time
    series."  
    
    Chunks up data according to input arg
    chunksize to speed up processing, which can be slow for long datasets 
    (computational expense O(N^3)).

    Also includes option to use noisy-input GP regression (NIGP)
    (McHutchon and Rasmussen 2011) instead of standard GP regression. This option was
    mainly included for testing purposes. For optimal performance 
    (both computational efficiency and despiking accuracy), 
    it is advised that standard GP regression is used.

    Parameters:
        ts - float array; 1D time series to despike
        dropout_mask - bool array; mask for missing values (and obvious
                       spikes), where 0 means a bad value (dropout or spike)
        chunksize - int; no. of datapoints in which to chunk up the signal;
                    if the signal is longer than ~1000 points the 
                    function is quite slow without chunking.
        overlap - int; number of points to overlap the chunks, 
                  adds 2*overlap to the true chunksize
        return_aux - bool; if True, also return y_pred, sigma, y_samp
        length_scale_bounds - tuple; bounds for allowable length scales
                              for the GP kernel.
        kernel - If not None, input a specified covariance kernel of own choice 
                 (only works for use_sklearn=True)
        predict_all - bool; if set to False, will remove the data points
                      specified by dropout_mask in the prediction stage.
                      This may help with identifying outliers in cases
                      of many dropouts.
        use_nigp - bool; set to True to use NIGP regression (not advised)
        despike - bool; if False, only replace samples given in dropout_mask, 
                  i.e no spike detection is done.
        use_sklearn - bool; By default True, meaning that the scikit-learn GP regression package is
                      used (advised)
        score_thresh - float; R^2 upper threshold for despiking/no despiking
        print_kernel - bool; set to False to not print kernel parameters.

    Returns:
        y_desp - despiked time series (only spikes and dropouts replaced)
        (if return_aux=True, then also returns):
        spike_mask - bool array; mask for detected spikes
        y_pred - predicted (smooth) signal produced by GP fit
        sigma - std for each data point given by the GP fit
        y_samp - 100 samples drawn from the pdf of each dropout 
                 (+ obvious spike) point. Only works for use_sklearn=True.
        theta - dict; GP fit hyperparameters for each chunk
    """

    # Check if ts is all NaN
    ts = np.array(ts)
    if np.sum(np.isnan(ts)) == len(ts):
        print('Input signal all NaN, nothing to despike ... \n')
        # Return array of NaNs.
        return np.ones(len(ts)) * np.nan
    elif len(ts[ts==-999]) == len(ts):
        print('Input signal all NaN, nothing to despike ... \n')
        # Return array of NaNs.
        return np.ones(len(ts)) * np.nan

    n = chunksize

    if dropout_mask is not None:
        mask_do = dropout_mask
    else:
        # No dropout mask given -> don't mask anything
        mask_do = np.ones(len(ts), dtype=bool)

    # Initialize output arrays
    y_desp = ts.copy() # Output time series with spikes & dropouts replaced
    spike_mask = np.ones(len(ts), dtype=bool)
    y_pred = np.zeros(len(ts)) # Full predicted mean signal
    sigma = np.zeros_like(y_pred) # Uncertainty (std) for each data point
    y_samp = np.ones(len(ts), dtype=(float,100)) * np.nan # Samples from posterior

    # ********************************
    # Chunk up the time series for more efficient computing
    # ********************************

    chunk_range = np.arange(n, len(ts)+n, step=n)
    # Initialize hyperparameters dict theta
    theta = {
            'l': np.zeros(len(chunk_range)),
            'sigma_f': np.zeros(len(chunk_range)),
            'sigma_y': np.zeros(len(chunk_range)),
            'sigma_x': np.zeros(len(chunk_range)),
            'score': np.zeros(len(chunk_range)),
            }
    for chunk_no, chunk in enumerate(np.arange(n, len(ts)+n, step=n)):
        if print_kernel:
            print('Chunk number: ', chunk_no)
        # Define chunk interval (overlap vs. no overlap)
        if overlap is not None:
            if (chunk-n-overlap) < 0:
                # First chunk with overlap only at the end
                interval = np.arange(chunk-n, chunk+overlap, dtype=int)
                # Full range of x's for prediction
                x_pred = np.arange(chunk-n, chunk+overlap)
            elif (chunk+overlap) > len(ts):
                # Last chunk with overlap only at the start
                interval = np.arange(chunk-n-overlap, chunk, dtype=int)
                # Full range of x's for prediction
                x_pred = np.arange(chunk-n-overlap, chunk)
            else:
                # All other chunks with overlap at start & end
                interval = np.arange(chunk-n-overlap, chunk+overlap, dtype=int)
                # Full range of x's for prediction
                x_pred = np.arange(chunk-n-overlap, chunk+overlap)
        else:
            # No overlap
            interval = np.arange(chunk-n, chunk, dtype=int)
            # Full range of x's for prediction
            x_pred = np.arange(chunk-n, chunk)

        # Chunk up data and mask based on interval
        y_chunk = ts[interval] # take out test chunk
        mask_do_chunk = mask_do[interval] # same for the dropout mask
        # Cut out dropouts from y_chunk
        y_chunk_nodo = y_chunk[mask_do_chunk]

        # Make x (t) axis for GP fit to training data
        x_train = interval.copy()
        # Cut out dropouts (how the GP fit works)
        x_train = x_train[mask_do_chunk] 

        # If requested, remove points specified in dropout_mask from x_pred
        # This means that the prediction skips the data points determined by
        # dropout_mask. In the output, those skipped data points will be
        # assigned NaN.
        if not predict_all:
            x_pred = x_pred[mask_do_chunk]
            y_chunk = y_chunk[mask_do_chunk]
            # Set skipped (i.e. not predicted) data points to NaN in output
            skipped = interval[~mask_do_chunk]
            y_pred[skipped] = np.nan
            y_desp[skipped] = np.nan
            # Also remove masked points from interval, as it is used later
            interval = interval[mask_do_chunk]

        # ********************************
        # Train the data and fit GP model
        # ********************************

        yn_mean = np.nanmean(y_chunk_nodo)
        # Copy chunk for GP fitting
        y_train = y_chunk_nodo.copy()
        # Remove mean
        y_train -= yn_mean

        if use_sklearn is False:
            if not use_nigp:
                sigma_x_init = 0
            else:
                sigma_x_init = 1e-5
            # Numpy implementation of NIGP/GP
            gp = nigp.NIGPRegressor(
                    use_nigp = use_nigp,
                    l_bounds = length_scale_bounds,
                    sigma_x_init = sigma_x_init,
                    sf_bounds = (1e-5, 50.0),
                    sy_bounds = (1e-5, 50.0),
                    sx_bounds = (1e-5, 50.0),
                    )
            # Train GP model on data and optimise hyperparameters 
            # (including input noise if use_nigp=True)
            y_pred_n, sigma_n, theta_s, score = gp.train(x_pred.reshape(-1,1), 
                    x_train.reshape(-1,1), y_train.reshape(-1,1),
                    return_score=True)
            # Add score to theta_s
            theta_s['score'] = score
        else:
            if kernel is None:
                # kernel = parameterization of covariance matrix + Gaussian noise,
                # eq. 5 in Bohlinger et al.
                kernel = (ConstantKernel(1.0) * \
                        RBF(length_scale=length_scale_bounds[0], length_scale_bounds=length_scale_bounds) + \
                        WhiteKernel(noise_level=1.0))
			#1e-8 * np.eye(len(x_train) # Added noise to help with num. stability
#                        ))
                #kernel = (RBF(length_scale=1, length_scale_bounds=length_scale_bounds) +
                #kernel = (1*RBF(length_scale=1) +
            # Define GP
            gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
            # Train GP process on data
            gp.fit(np.array(x_train).reshape(-1,1), np.array(y_train).reshape(-1,1))
            gp.kernel_
            if print_kernel:
                print('GP kernel (sklearn): ', gp.kernel_)
            # Compute coefficient of determination R^2
            score = gp.score(x_train.reshape(-1,1), y_train.reshape(-1,1))
            # Make prediction based on GP (fills in missing points).
            # Remember, x_pred includes the entire range of indices in the chunk,
            # unless predict_all is set to False.
            # TODO: This might not be optimal for spike detection, as found in
            # testing with the time series from 3 Jan 2019, 00:40. A better way
            # might be to run an initial spike detection run in which dropout and
            # obvious spike locations are removed from x_pred, as this may have a
            # greater success rate at detecting spikes. Afterward, another GP fit
            # could be run with the full x_pred vector, in order to fill in all the
            # gaps.
            y_pred_n, sigma_n = gp.predict(np.array(x_pred).reshape(-1,1), 
                    return_std=True)
            # Save trained hyperparams and R^2 score
            l = gp.kernel_.k1.get_params()['k2__length_scale']
            sigma_f = np.sqrt(gp.kernel_.k1.get_params()['k1__constant_value'])
            sigma_y = np.sqrt(gp.kernel_.k2.get_params()['noise_level'])
            theta_s = {'l':l, 'sigma_f':sigma_f, 'sigma_y':sigma_y,
                    'sigma_x':0.0, 'score':score}

        if print_kernel:
            print('score: ', score)
        # Add hyperparameters to theta dict
        for key in theta.keys():
            theta[key][chunk_no] = theta_s[key]
        # Add the mean back to the prediction
        y_pred_n += yn_mean
        y_pred_n = np.squeeze(y_pred_n)
        sigma_n = np.squeeze(sigma_n)

        # ********************************
        # Update output arrays
        # ********************************

        y_pred[interval] = y_pred_n
        sigma[interval] = sigma_n
        if despike and score < score_thresh:
            # Detect outliers and replace both newly found spikes and dropouts
            mask = np.logical_and(y_chunk>y_pred_n-2*sigma_n, y_chunk<y_pred_n+2*sigma_n)
            if print_kernel:
                print('Spikes found: {} \n'.format(np.sum(~mask)))
            mask = np.logical_or(~mask, ~mask_do_chunk)
            mask = ~mask
        else:
            # Only replace points defined in dropout_mask
            mask = mask_do_chunk.copy()
        spike_mask[interval] = mask
        # Replace spikes and dropouts by corresponding y_pred values
        y_chunk[~mask] = y_pred_n[~mask]
        y_desp[interval] = y_chunk


        # ********************************
        # Sample y values for dropouts from posterior
        # (if prediction was made for those points)
        # ********************************
        if predict_all and use_sklearn:
            x_dropouts = interval[~mask_do_chunk].reshape(-1,1)
            # Sample from posterior (only implemented in sklearn for now)
            if len(x_dropouts):
                # Make matrix of samples for each dropout location
                S = gp.sample_y(x_dropouts, 100) 
                S += yn_mean # Add the mean back
                # Make samples into tuples
                y_samp_do = [list(map(tuple,i)) for i in S]
                y_samp_do = list(chain(*y_samp_do))
                x_dropouts = x_dropouts.squeeze()
                if len(np.atleast_1d(x_dropouts))==1:
                    x_dropouts = int(x_dropouts)
                # Store samples into output array
                #y_samp[x_dropouts] = y_samp_do
                y_samp[x_dropouts] = S.squeeze()

    # Don't want the initial dropouts to be included in the returned spike mask
    # -> add those back
    spike_mask[dropout_mask==0] = 1

    if return_aux is True:
        return y_desp, spike_mask, y_pred, sigma, y_samp, theta
    else:
        # Only return the despiked signal
        return y_desp


def phase_space_2d(ts, p=6, C=0.2, fmin=0.05, fmax=1.0, sg_poly=3, sg_winlen=13,
        diff_npts=5, ax=None):
    """
    2D phase space despiking following Voermans et al. (2021; JTECH):
    "Wave Anomaly Detection in Wave Measurements".

    Parameters: 


    """
    import pandas as pd
    from scipy.signal import savgol_filter
    def inside_ellipse(x, y, a, b, origin=(0,0), theta=0):
        """
        Check if point given by coordinates x,y lies inside
        the ellipse centred at origin with semi axis in x-dir.
        a and semi axis in y-dir. b, and (possibly) rotated by
        angle (in radians) theta.

        Returns True if point is inside, False otherwise.

        Borrowed from: 
        https://www.geeksforgeeks.org/check-if-a-point-is-inside-
        outside-or-on-the-ellipse/

        Equation for rotated ellipse from:
        https://math.stackexchange.com/questions/426150/what-is-
        the-general-equation-of-the-ellipse-that-is-not-in-the-
        origin-and-rotate
        """

        h = origin[0]
        k = origin[1]
        p = ( ((x-h)*np.cos(theta) + (y-k)*np.sin(theta))**2 / a**2 +
                ((x-h)*np.sin(theta) - (y-k)*np.cos(theta))**2 / b**2 )

        if p < 1:
            return True
        else:
            return False

    # Subtract the median (more robust than the mean, see Wahl (2003))
    u = ts.copy()
    ts_med = np.nanmedian(ts)
    u -= ts_med
    # Length of time series
    N = len(ts)

    # Compute spectra from noisy time series. Filter out all measurements +/- 4*MAD
    # and interpolate with cubic spline to get "noise-free" spectral estimates.
    mad_u = mad_std(u, ignore_nan=True) # MAD of u
    mask = np.logical_or(np.abs(u) > 4 * mad_u, np.isnan(u))
    u_masked = u.copy()
    if mask[0]:
        mask[0] = 0
        # Initial NaN -> set first value to 0
        u_masked[0] = 0
    if mask[-1]:
        mask[-1] = 0
        # Closing NaN -> set last value to 0
        u_masked[N-1] = 0
    u_masked[mask] = np.nan
    # Compute std of filtered u
    std_u = np.nanstd(u_masked)
    # Create new mask that takes into account original NaNs
    mask_full = np.isnan(u_masked)
    # Initialize cubic spline
    t = np.arange(N) # x axis (time array)
    cs = CubicSpline(t[~mask_full], u_masked[~mask_full], bc_type='natural')#((2,0),(2,0)))
    u_interp = cs(t) # Interpolated signal
    E, F = spectra.spec_uvz(u_interp, fs=5)
    # Truncate spectrum
    f_mask = np.logical_and(F>=fmin, F<=fmax) # Mask for low/high freqs
    E = E[f_mask] # Truncate too high/low frequencies
    F = F[f_mask]
    # Compute integrated spectral parameters nu and Tm02
    nu = spectra.spec_bandwidth(E, F)
    m0 = spectra.spec_moment(E, F, 0)
    m2 = spectra.spec_moment(E, F, 2)
    Tm02 = np.sqrt(m0 / m2)

    # Take the 2nd derivative of the time series using 5-point central difference
    # scheme
    #u_filt = savgol_filter(u_interp, sg_winlen, sg_poly)
    u_filt = u_interp.copy()
    du2 = np.zeros_like(u_interp)
    h = 0.2 # Time step
    #du2[1:-1] = (1*u_interp[:-2] - 2*u_interp[1:-1] + 1*u_interp[2:])/(1*1.0*h**2)
    #du2[1:-1] = (1*u_filt[:-2] - 2*u_filt[1:-1] + 1*u_filt[2:])/(1*1.0*h**2)
    du2[4:-4] = (
            - 9 * u_filt[:-8]
            + 128 * u_filt[1:-7]
            - 1008 * u_filt[2:-6]
            + 8064 * u_filt[3:-5]
            - 14350 * u_filt[4:-4]
            + 8064 * u_filt[5:-3]
            - 1008 * u_filt[6:-2]
            + 128 * u_filt[7:-1]
            - 9 * u_filt[8:]
            ) / (5040 * h**2)
#    du2[2:-2] = (
#            - 1 * u_filt[:-4]
#            + 16 * u_filt[1:-3]
#            - 30 * u_filt[2:-2]
#            + 16 * u_filt[3:-1]
#            - 1 * u_filt[4:]
#            ) / (12 * h**2)

    # Normalize u and du2 by std
    Su = u_interp / std_u
    std_du2 = np.nanstd(du2)
    Sdu2 = du2 / std_du2

    # Compute the semi axes and origins of the ellipses
    x1 = np.sqrt(2) * p # Major axis
#    print('x1: ', x1)
    x2 = C * np.sqrt(nu) * np.sqrt(std_du2 / std_u) * Tm02 # Minor axis
#    print('x2: ', x2)
    origin = (0,0)

    # Define rotation angle of ellipse following Voermans et al. (2021)
    theta = (-1 / 4) * np.pi # Voermans et al. (2021) angle for normalized phase space
    #theta = np.arctan2(np.nansum(u_interp * du2), np.nansum(u_interp**2, dtype=float))
#    print('theta: ', theta)

    # Check if points lie inside the ellipses. If outside -> spike
    spike_mask = np.ones(N, dtype=bool)
    # For plotting, make three different masks
    for i in range(N):
        # Check (rotated due to correlation) u-du2 phase space
        if not inside_ellipse(x=Su[i], y=Sdu2[i], a=x1, b=x2, origin=origin, theta=theta):
            spike_mask[i] = 0

    # Replace spikes if any were detected
    if spike_mask.sum() != N:
        cs = CubicSpline(t[spike_mask], u_interp[spike_mask], bc_type='natural')#((2,0),(2,0)))
        u = cs(t) # Interpolated signal
        # Replace spikes by interpolation
        #u = replace.replace_data(u, spike_mask, method=replace_single)

    # Add back median
    u = u + ts_med

    # Plot if requested
    if ax is not None:
        from matplotlib.patches import Ellipse
        # all pts
        ax.scatter(Su[spike_mask], Sdu2[spike_mask], marker='.', color='k', alpha=0.2)
        # pts outside the ellipse in red
        ax.scatter(np.ma.masked_array(Su, mask=spike_mask),
                np.ma.masked_array(Sdu2, mask=spike_mask),
                marker='*', s=40, color='r', alpha=0.8)
        # parameterised ellipse
        t = np.linspace(0, 2*np.pi, 1000)
        xp = x1 * np.cos(t) * np.cos(theta) - x2 * np.sin(t) * np.sin(theta) # + x0
        yp = x1 * np.cos(t) * np.sin(theta) + x2 * np.sin(t) * np.cos(theta) # + y0
        # patches ellipse (should be equal to parameterised ellipse)
        ell2 = Ellipse(xy=origin, width=2*x1, height=2*x2,
                angle=np.degrees(theta), edgecolor='k')
        ax.add_artist(ell2)
        ell2.set_clip_box(ax.bbox)
        ell2.set_alpha(0.2)
        ax.legend(handletextpad=0, handlelength=0, fontsize=14)
        ax.set_xlabel(r'$\eta$ [m]', fontsize=16)
        ax.set_ylabel(r'$\Delta^2 \eta$ [m]', fontsize=16)
        ax.set_xlim(1.25*np.min(xp), 1.25*np.max(xp))
        ax.set_ylim(1.25*np.min(yp), 1.25*np.max(yp))

        return u, spike_mask, Su, Sdu2, ax

    else:
        return u, spike_mask, Su, Sdu2



def phase_space_3d(ts, replace_single='linear', replace_multi='linear',
        threshold='universal', ax=None):
    """
    3D phase space method for despiking originally developed by
    Goring and Nikora (2002), and later modified by Wahl (2003) and
    Mori (2005). 
    
    This function is based on the Matlab function func_despike_phasespace3d
    by Mori (2005).

    Parameters:
        ts - time series to despike
        threshold - either 'universal' or 'chauvenet'
        ax - matplotlib.Axes3D object; optional
    """
    from mpl_toolkits.mplot3d import axes3d, Axes3D

    # Universal threshold lambda_u: the expected maximum of N independent
    # samples drawn from a standard normal distribution (Goring & Nikora, 2002)
    N = len(ts)
    if threshold == 'universal':
        lambda_u = np.sqrt(2*np.log(N))
    # Chauvenet's criterion is independent of the size of the sample (Wahl, 2003)
    elif threshold == 'chauvenet':
        p = 1 / (2*N) # rejection probability
        Z = np.sqrt(2) * erfinv(1-p)

    # Subtract the median (more robust than the mean, see Wahl (2003))
    ts_med = np.median(ts)
    u = ts.copy()
    u = u - ts_med

    # Take the 1st and 2nd derivatives of the time series
    du1 = np.gradient(u)
    du2 = np.gradient(du1)

    # Estimate the rotation angle theta of the principal axis 
    # of u versus du2 using the cross correlation. 
    # (for u vs. du1 and du1 vs. du2, theta = 0 due to symmetry)
    theta = np.arctan2(np.dot(u, du2), np.sum(u**2, dtype=float))

    # Look for outliers in 3D phase space
    # Following the func_excludeoutlier_ellipsoid3d.m Matlab script by Mori
    if theta == 0:
        x = u
        y = du1
        z = du2
    else:
        # Rotation matrix about y-axis (du1 axis)
        R = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0], 
            [-np.sin(theta), 0 , np.cos(theta)]
            ])
        x = u * R[0,0] + du1 * R[0,1] + du2 * R[0,2]
        y = u * R[1,0] + du1 * R[1,1] + du2 * R[1,2]
        z = u * R[2,0] + du1 * R[2,1] + du2 * R[2,2]
    
    # G&N02: For each pair of variables, calculate the ellipse that has max. and min.
    # from the previous computation. Use the MAD (median of absolute deviation)
    # instead of std for more robust scale estimators (Wahl, 2003).
    # Semi axes of the ellipsoid
    a = lambda_u * 1.483*robust.mad(x, c=1)
    b = lambda_u * 1.483*robust.mad(y, c=1)
    c = lambda_u * 1.483*robust.mad(z, c=1)

    # Mask for detected spikes (0=spike)
    spike_mask = np.ones(N, dtype=bool)

    # Check for outliers (points not within the ellipsoid)
    for i in range(N):
        # Data point u, du1, du2 coordinates
        x1 = x[i]
        y1 = y[i]
        z1 = z[i]
        # Point on the ellipsoid given by a, b, c
        x2 = (a*b*c) * x1 / np.sqrt((a*c*y1)**2 + b**2*(c**2*x1**2+a**2*z1**2))
        y2 = (a*b*c) * y1 / np.sqrt((a*c*y1)**2 + b**2*(c**2*x1**2+a**2*z1**2))
        zt = c**2 * (1-(x2/a)**2 - (y2/b)**2)
        if z1 < 0:
            z2 = -np.sqrt(zt)
        elif z1 > 0:
            z2 = np.sqrt(zt)
        else:
            z2 = 0
        # Check for outliers from the ellipsoid by subtracting the ellipsoid
        # corresponding to the data (x1, y1, z1) from the ellipsoid given by
        # a,b and c. If the difference is less than 0 the point lies outside
        # the ellipsoid.
        dis = (x2**2 + y2**2 + z2**2) - (x1**2 + y1**2 + z1**2)
        if dis < 0:
            spike_mask[i] = 0

    # Replace spikes if any were detected
    if spike_mask.sum() != N:
        # Replace spikes by interpolation
        u = replace.replace_data(u, spike_mask, method=replace_single)

    # Add back median
    u = u + ts_med

    if ax is not None:
        # Return axis for plot
        pu = np.ma.masked_array(u, mask=spike_mask)
        pdu1 = np.ma.masked_array(du1, mask=spike_mask)
        pdu2 = np.ma.masked_array(du2, mask=spike_mask)
        ax.scatter(pu, pdu1, pdu2, marker='*', color='r', alpha=0.8, s=40)
        ax.scatter(u[spike_mask], du1[spike_mask], du2[spike_mask], marker='.',
                color='k', alpha=0.2)
        #ax.scatter(x, y, z, 'r*')
        ax = plotEllipsoid(ax, center=[0,0,0], radii=[a,b,c], rotation=R, plotAxes=True)

        return u, spike_mask, ax
    else:
        return u, spike_mask


def plotEllipsoid(ax, center, radii, rotation, plotAxes=False, 
        cageColor='b', cageAlpha=0.2):
    """
    Plot an ellipsoid.
    
    Borrowed from:
    https://github.com/minillinim/ellipsoid/blob/master/ellipsoid.py
    """
        
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    
    # cartesian coordinates that correspond to the spherical angles:
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    # rotate accordingly
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center

    if plotAxes:
        # make some purdy axes
        axes = np.array([[radii[0],0.0,0.0],
                         [0.0,radii[1],0.0],
                         [0.0,0.0,radii[2]]])
        # rotate accordingly
        for i in range(len(axes)):
            axes[i] = np.dot(axes[i], rotation)


        # plot axes
        for p in axes:
            X3 = np.linspace(-p[0], p[0], 100) + center[0]
            Y3 = np.linspace(-p[1], p[1], 100) + center[1]
            Z3 = np.linspace(-p[2], p[2], 100) + center[2]
            ax.plot(X3, Y3, Z3, color=cageColor, alpha=0.8)

    # plot ellipsoid
    ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color=cageColor, alpha=cageAlpha)

    return ax


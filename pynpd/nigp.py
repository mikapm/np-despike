#!/usr/bin/env python3
"""
Gaussian Process regression with Input Noise (NIGP)
following McHutchon and Rasmussen (2011):
"Gaussian Process Training with Input Noise"
and the PhD Thesis of McHutchon (2014):
"Nonlinear Modelling and Control using Gaussian Processes"

GP Numpy implementation based on Martin Krasser's blog:
http://krasserm.github.io/2018/03/19/gaussian-processes/

Contains example script comparing standard GP 
regression of this code vs. the scikit-learn (sklearn) 
GP implementation.

"""

import numpy as np
import scipy as sp
from numpy.linalg import (inv, cholesky, det, lstsq)
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (ConstantKernel, RBF,
                                              WhiteKernel, Matern)
from scipy.spatial.distance import cdist


def kernel(X1, X2, l=1.0, sigma_f=1.0, kernel_type='RBF'):
    """
    Following Martin Krasser:
    http://krasserm.github.io/2018/03/19/gaussian-processes/
    and the scikit-learn library documentation:
    https://github.com/scikit-learn/scikit-learn/blob/1495f6924/
    sklearn/gaussian_process/kernels.py#L1255

    Isotropic squared exponential kernel. 
    Computes a covariance matrix from points in X1 and X2.
    Parameters: 
        X1 - Array of m points (m x d). 
        X2 - Array of n points (n x d). 
        l - float; length scale parameter
            (controls smoothness)
        sigma_f - float; signal std (vertical variation)
        kernel_type - str; 'RBF' or 'Matern' (for Matern 3/2)
    
    Returns: 
        Covariance matrix (m x n). 
    """
    if kernel_type == 'RBF':
        # Squared distance (x_i-x_j)^T(x_i-x_j) expanded.
        # The summation here is direct summation of a row
        # and a column vector, so sqdist is a matrix.
        sqdist = (np.sum(X1**2, 1).reshape(-1, 1) +\
                  np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T))

        # Return the parameterized covariance matrix 
        return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)

    elif kernel_type == 'Matern':
        # Returns Matern kernel with nu=1.5
        # Compute distance between each pair of the two 
        # collections of inputs
        s = cdist(X1,X2)
        frac = (np.sqrt(3)*s) / l
        return sigma_f**2 * (1 + frac) * np.exp(-frac)


def posterior_predictive(X_s, X_train, Y_train, l=1.0, sigma_f=1.0, 
        sigma_y=1e-8, sigma_x=1e-8, Grad_fmean=None, kernel_type='RBF'):
    """
    Original version by Martin Krasser:
    http://krasserm.github.io/2018/03/19/gaussian-processes/

    Extended to incorporate parameterization of input noise
    variance by scaling output noise variance by the square of
    the posterior mean following McHutchon and Rasmussen (2011).

    Computes the mean and covariance of the GP posterior 
    predictive distribution from m training data X_train and
    Y_train and n new inputs X_s. 
    
    Parameters: 
        X_s - New input locations (n x d). 
        X_train - Training locations (m x d).
        Y_train - Training targets (m x 1).
        l - Kernel length scale parameter. 
        sigma_f - Kernel vertical variation parameter.
        sigma_y - Output noise parameter. 
        sigma_x - Input noise parameter of McHutchon (here assumed a
                  scalar)
        Grad_fmean - Posterior mean gradient (column vector) (m x d)
        kernel_type - str; 'RBF' or 'Matern'
        
    Returns: 
        Posterior mean vector (n x d) and covariance matrix (n x n). 
    """
    if Grad_fmean is None:
        # Gradient at training points
        Grad_fmean_train = np.zeros(len(X_train)).reshape(-1,1)
        # Gradient at input points
        Grad_fmean_input = np.zeros(len(X_s)).reshape(-1,1)
    else:
        # Gradient at training points
        Grad_fmean_train = np.interp(
                X_train.ravel(), X_s.ravel(), Grad_fmean.ravel())
        Grad_fmean_train = Grad_fmean_train.reshape(-1,1)
        Grad_fmean_input = Grad_fmean.copy()
        Grad_fmean_input = Grad_fmean_input.reshape(-1,1)

    # K_y or K_n
    # Noisy covariance matrix (with added output noise scaled by the
    # square of the posterior mean gradient)
    K = (kernel(X_train, X_train, l, sigma_f, kernel_type=kernel_type) + \
        sigma_y**2 * np.eye(len(X_train)) + \
        np.diag(np.diag(np.dot(Grad_fmean_train,
            np.atleast_2d(Grad_fmean_train.ravel())))) * sigma_x**2)
    # K_{*}
    K_s = kernel(X_train, X_s, l, sigma_f, kernel_type=kernel_type)
    # K_{**}
    K_ss = kernel(X_s, X_s, l, sigma_f, kernel_type=kernel_type) + \
        sigma_y**2 * np.eye(len(X_s)) + \
        np.diag(np.diag(np.dot(Grad_fmean_input,
            np.atleast_2d(Grad_fmean_input.ravel())))) * sigma_x**2 
    # K matrix inversion following Rasmussen & Williams (2006, p. 19)
    L = sp.linalg.cholesky(K, lower=True)
    a1 = np.linalg.lstsq(L, Y_train, rcond=None)[0]
    alpha = np.linalg.lstsq(L.T, a1, rcond=None)[0]
    
    # Equation (4) of Krasser
    mu_s = K_s.T.dot(alpha)

    # Equation (5) of Krasser
    v = np.linalg.lstsq(L, K_s, rcond=None)[0]
    cov_s = K_ss - v.T.dot(v)
    
    return mu_s, cov_s


def nll_fn(X_train, Y_train, Grad_fmean=None, kernel_type='RBF'):
    """
    Original version by Martin Krasser:
    http://krasserm.github.io/2018/03/19/gaussian-processes/

    Extended to incorporate parameterization of input noise
    variance by scaling output noise variance by the square of the 
    posterior mean following McHutchon and Rasmussen (2011).

    Returns a function that computes the negative marginal log-
    likelihood for training data X_train and Y_train and given
    noise levels. 
    
    Parameters: 
        X_train - training locations (m x d). 
        Y_train - training targets (m x 1). 
        Grad_fmean - Posterior mean gradient (column vector) (m x d)
        kernel_type - str; 'RBF' or 'Matern'
    Returns: 
        Minimization objective. 
    """
    if Grad_fmean is None:
        Grad_fmean = np.zeros(len(X_train)).reshape(-1,1)

    # Numerically stable implementation of Eq. (7) as described
    # in http://www.gaussianprocess.org/gpml/chapters/RW2.pdf, Section
    # 2.2, Algorithm 2.1. Extra output variance term added following
    # McHutchon and Rasmussen (2011).
    K = kernel(X_train, X_train, l=theta[0],
            sigma_f=theta[1], kernel_type=kernel_type) + \
        theta[2]**2 * np.eye(len(X_train)) + \
        np.diag(np.diag(np.dot(Grad_fmean,
            np.atleast_2d(Grad_fmean.ravel())))) * theta[3]**2 +\
        1e-8 * np.eye(len(X_train)) # Added noise to help with num. stability

     # Cholesky decomposition
    L = sp.linalg.cholesky(K, lower=True)

    return np.sum(np.log(np.diagonal(L))) + \
           0.5 * Y_train.T.dot(lstsq(L.T, lstsq(L, Y_train,
               rcond=None)[0], rcond=None)[0]) + \
           0.5 * len(X_train) * np.log(2*np.pi)
    

class NIGPRegressor():
    """
    Main NIGP class.
    """
    def __init__(self, 
            l_init = 1.0, 
            sigma_f_init = 1.0,
            sigma_y_init = 1.0, 
            sigma_x_init = 1.0,
            l_bounds = (5.0, 30.0), 
            sf_bounds = (1e-5, 50),
            sy_bounds = (1e-5, 50),
            sx_bounds = (1e-5, 50),
            use_nigp = True,
            print_kernel = True,
            kernel = 'RBF',
            ):
        """
        Parameters:
            kernel - covariance matrix kernel
            l_init - initial guess for length scale parameter
            sigma_f_init - initial guess for measurement noise
            sigma_y_init - initial guess for output noise
            sigma_x_init - initial guess for input noise
            l_bounds - bounds for l
            sf_bounds - bounds for sigma_f
            sy_bounds - bounds for sigma_y
            sx_bounds - bounds for sigma_x
            use_nigp - bool: set to False to do standard GP regression
        """
        self.kernel = kernel
        self.sigma_f_init = sigma_f_init
        self.sigma_y_init = sigma_y_init
        self.sigma_x_init = sigma_x_init 
        self.l_bounds = l_bounds
        self.l_init = max(l_init, self.l_bounds[0])
        self.sf_bounds = sf_bounds
        self.sy_bounds = sy_bounds
        self.sx_bounds = sx_bounds
        self.use_nigp = use_nigp
        self.print_kernel = print_kernel
        if self.use_nigp is True:
            self.gp_str = 'NIGP'
        else:
            self.gp_str = 'GP'


    def train(self, X, X_train, Y_train, return_score=True):
        """
        Following the code of Martin Krasser:
        http://krasserm.github.io/2018/03/19/gaussian-processes/
        and the modified NIGP approach of McHutchon and Rasmussen (2011)

        If return_score is True, returns the coefficient of
        determination R^2 of the prediction.
        """
        # Make GP fit using initial guess hyperparameters (ignoring
        # input noise)
        mu_s_init, cov_s_init = posterior_predictive(
                X,
                X_train,
                Y_train,
                l = self.l_init,
                sigma_f = self.sigma_f_init,
                sigma_y = self.sigma_y_init,
                sigma_x = 0,
                kernel_type = self.kernel
                )

        if self.use_nigp is True:
            # Compute gradient of posterior mean
            Grad_fmean = np.gradient(mu_s_init.ravel())
            # Gradient at training points (for optimization)
            Grad_fmean_opt = np.interp(
                    X_train.ravel(), X.ravel(), Grad_fmean.ravel())
            Grad_fmean_opt = Grad_fmean_opt.reshape(-1,1)
        else:
            Grad_fmean = None
            Grad_fmean_opt = None

        # Minimize the negative log-likelihood w.r.t. hyperparameters l, sigma_f, sigma_y 
        # and sigma_x.
        l_bounds_new = (self.l_bounds[0], self.l_bounds[1])
        success = False
        while not success:
            try:
                res = minimize(
                        nll_fn(X_train, Y_train,
                               Grad_fmean=Grad_fmean_opt, 
                               kernel_type=self.kernel),
                        [self.l_init, self.sigma_f_init,
                               self.sigma_y_init, self.sigma_x_init], 
                        bounds=(l_bounds_new, self.sf_bounds,
                                   self.sy_bounds, self.sx_bounds),
                        method='L-BFGS-B',
                        )
                # Store the optimization results in global variables so that
                # we can compare it later with the results from other 
                # implementations.
                (l_opt, sigma_f_opt, sigma_y_opt, sigma_x_opt) = res.x
                # If optimization did not fail, escape while loop
                success = True
            except np.linalg.LinAlgError:
                # Singular matrix, cannot be inverted. Reduce l_min by 1 and try again.
                # Don't let new l_min go below 0.
                print('Singular matrix encountered, trying with lower min. length scale.')
                l_bounds_iter = (max(v for v in [l_bounds_new[0]-1, 0] if v >= 0),
                        self.l_bounds[1])
                l_bounds_new = (l_bounds_iter[0], l_bounds_iter[1]) # update tuple
                if l_bounds_iter[0] == 0:
                    # Use initial guesses as optimized parameters
                    l_opt = self.l_bounds[0]
                    sigma_f_opt = self.sigma_f_init
                    sigma_y_opt = self.sigma_y_init
                    sigma_x_opt = self.sigma_x_init
                    # Stop iterating if l_min == 0
                    success = True

        if self.print_kernel:
            print('{0} kernel (scipy):    {1:.3f}**2 * {2}(length_scale={3:.3f}) + '\
                    'WhiteKernel(noise_level={4:.5f}) + InputNoiseVar={5:.5f}'.format(
                        self.gp_str, sigma_f_opt, self.kernel, l_opt, 
                        sigma_y_opt**2, sigma_x_opt**2))
        # Save optimized hyperparameters
        theta = {'l': l_opt, 
                'sigma_f': sigma_f_opt,
                'sigma_y': sigma_y_opt,
                'sigma_x': sigma_x_opt
                }

        # Compute the prosterior predictive statistics with optimized
        # kernel parameters.
        mu_s, cov_s = posterior_predictive(
                X,
                X_train,
                Y_train, 
                l = l_opt,
                sigma_f = sigma_f_opt, 
                sigma_y = sigma_y_opt,
                sigma_x = sigma_x_opt,
                Grad_fmean = Grad_fmean, 
                kernel_type = self.kernel
                )

        if return_score:
            # Return the coefficient of determination R^2 of the prediction.
            ind_mu = np.in1d(X, X_train).nonzero()[0] # Common indices
            u = ((Y_train - mu_s[ind_mu])**2).sum()
            v = ((Y_train - np.mean(Y_train))**2).sum()
            score = 1 - u/v
            return mu_s, np.sqrt(np.diag(cov_s)), theta, score
        else:
            return mu_s, np.sqrt(np.diag(cov_s)), theta


if __name__ == '__main__':

    def safe_arange(start, stop, step):
        """
        Hack to fix floating point precision problem of np.arange().
        Borrowed from:
        https://stackoverflow.com/questions/47243190/numpy-arange-how-to
        -make-precise-array-of-floats
        """
        return step * np.arange(start / step, stop / step)

    # Make sample data
    X = safe_arange(-5, 5.2, 0.2).reshape(-1, 1) # Finite number of points
    # Noisy training data
    X_train = safe_arange(-3, 4, 1).reshape(-1, 1)
    Y_train = np.sin(X_train) + args.noise * np.random.randn(*X_train.shape)

    # Use NIGP_regressor with normal GP regression
    gpn = NIGPRegressor(
            use_nigp=False, 
            l_bounds=(1e-1, 30.0), 
            l_init=1.0,
            sigma_y_init=1.0,
            sigma_f_init=1.0,
            #sigma_f_init=1e-5,
            sigma_x_init=0.0,
            sf_bounds = (1e-5, None),
            sy_bounds = (1e-5, None), 
            sx_bounds = (1e-5, None),
            )
    # Train GP model on data and optimise hyperparameters 
    # (including input noise)
    mu_sn, sigma_sn, th_sn, score = gpn.train(X, X_train, Y_train)

    # Do the equivalent using the sklearn library
    rbf = ConstantKernel(1.0) * RBF(length_scale=1.0, length_scale_bounds=gpn.l_bounds)
    # Add White Noise with variance defined by noise variable.
    # This is not the same as adding arg alpha=noise**2 to GaussianProgressRegressor!!!
    kernel = rbf + WhiteKernel(noise_level=args.noise, )#noise_level_bounds='fixed')
    gpr = GaussianProcessRegressor(kernel=kernel, )#alpha=args.noise**2)
    # Reuse training data from previous 1D example
    gpr.fit(X_train, Y_train)
    gpr.kernel_
    print('GP kernel (sklearn): ', gpr.kernel_)
    # Compute posterior predictive mean and covariance
    mu_sk, sigma_sk = gpr.predict(X, return_std=True)



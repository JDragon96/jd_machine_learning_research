import numpy as np
import Statistic
import copy

def normal(mu, sigma, x):
    """ sigma = 분산 """
    return 1/np.sqrt(2 * np.pi * np.sqrt(sigma)) * np.exp(-((x - mu)**2) / (2*(sigma)))

def multivariable_gaussian(data, mu, sig):
    """
    >>> data: (N, dim)
    >>> mu  : (dim)
    >>> sig : (dim, dim)
    """
    new_data = copy.deepcopy(data)
    dim = np.shape(data)[1]
    sig_abs = sig[0][0] * sig[1][1] - sig[1][0] * sig[0][1]
    inv_sig = np.linalg.inv(sig)
    sig_tile = np.tile(inv_sig, (len(data), 1, 1))
    u = pow(2 * np.pi, dim / 2) * sig_abs
    
    # e_total shape : (4, 1, 2)
    # sig_tile shape : (4, 2, 2)
    e_total = (np.expand_dims(new_data, axis=1) - mu)
    result = np.exp(e_total @ sig_tile @ e_total.transpose(0, 2, 1) * -0.5) / u
    return result.reshape(len(data))

def normal_bayesian(data):
    data = np.array(data)
    a = -0.5 * (data)
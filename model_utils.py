import GPy
import numpy as np
import gpytorch
import warnings
import pandas as pd

def calc_k_rbf(x, lengthscale, scale, xprime=None, dim=1):
    if xprime is None:
        xprime = x

    if len(lengthscale) > 1:
        ARD = True
    else:
        ARD = False

    kernel = GPy.kern.RBF(input_dim=dim, variance=scale, lengthscale=lengthscale, ARD=ARD)

    return kernel.K(x, xprime)


def calc_k_matern(x, lengthscale, scale, xprime=None):
    
    if x.shape[-1] == 1:
        x = x.squeeze(-1)

    if xprime is None:
        xprime = x

    elif xprime.shape[-1] == 1:
        xprime = xprime.squeeze(-1)


    if len(x.shape) == 1:
        r = np.abs(x[:, None] - xprime[None, :]) / lengthscale
    else:
        r = np.linalg.norm(x[:, None, :] - xprime[None, :, :], axis=-1, ord=2) / lengthscale

    return scale * np.exp(-r)


def calc_m_linear(x, weights, bias):
    return np.matmul(x, weights) + bias

def calc_m_exp(x, weights, bias):
    return bias * (1 - np.exp(-np.divide(x, weights).squeeze(-1)))


def calc_m_sigmoid(x, weights, bias):
    return 1 / (1 + np.exp(-(np.matmul(x, weights) + bias)))


def create_mvn(m, k):    
    return gpytorch.distributions.MultivariateNormal(m, k)


def calc_posterior_k(xtrain, lengthscale, scale, sigma, xstar, kernel='rbf', dim=1, multiobs=False):
    if multiobs:
        x_rep = xtrain.copy()
        xtrain = np.unique(xtrain)[:, np.newaxis]


    if kernel == 'rbf':
        k_starstar = calc_k_rbf(xstar, lengthscale, scale, dim=dim)
        k_starx = calc_k_rbf(xtrain, lengthscale, scale, xprime=xstar, dim=dim)
        k_xx = calc_k_rbf(xtrain, lengthscale, scale, dim=dim)
    elif kernel == 'matern':
        k_starstar = calc_k_matern(xstar, lengthscale, scale)
        k_starx = calc_k_matern(xtrain, lengthscale, scale, xprime=xstar)
        k_xx = calc_k_matern(xtrain, lengthscale, scale)
    else:
        warnings.warn("You didn't specify a valid kernel type")

    if np.isscalar(sigma):
        Sigma = sigma * np.eye(k_xx.shape[0])
    else:
        idx = []
        for x in xtrain:
            idx.append(np.where(x_rep == x)[0][0])
        Sigma = np.diag(sigma[idx])

    return k_starstar - np.matmul(k_starx.T, np.linalg.solve(k_xx + Sigma, k_starx))


def calc_posterior_m(x_train, ytrain, w, b, lengthscale, scale, sigma, xstar, mean='linear', kernel='rbf', dim=1, multiobs=False):

    x_rep = x_train.copy()
    if multiobs:
        x_train = np.unique(x_train)[:, np.newaxis]
        A = np.zeros((x_rep.size, x_train.size))
        for i, tu in enumerate(x_train):
            A[x_rep.squeeze() == tu, i] = 1



    if mean == 'linear':
        m = calc_m_linear(xstar, w, b)
        mtrain = calc_m_linear(x_rep, w, b)
    elif mean == 'exponential':
        m = calc_m_exp(xstar, w, b)
        mtrain = calc_m_exp(x_rep, w, b)
    elif mean == 'sigmoid':
        m = calc_m_sigmoid(xstar, w, b)
        mtrain = calc_m_sigmoid(x_rep, w, b)
    elif mean == 'zero':
        m = np.zeros(xstar.shape[0])
        mtrain = np.zeros(x_rep.shape[0])
    else:
        warnings.warn("You didn't specify a valid mean function")

    if kernel == 'rbf':
        k_starx = calc_k_rbf(x_train, lengthscale, scale, xprime=xstar, dim=dim)
        k_xx = calc_k_rbf(x_train, lengthscale, scale, dim=dim)
    elif kernel == 'matern':
        k_starx = calc_k_matern(x_train, lengthscale, scale, xprime=xstar)
        k_xx = calc_k_matern(x_train, lengthscale, scale)

    if np.isscalar(sigma):
        Sigma = sigma * np.eye(k_xx.shape[0])
    else:
        idx = []
        for x in x_train:
            idx.append(np.where(x_rep == x)[0][0])
        Sigma = np.diag(sigma[idx])
    

    if multiobs:
        return m + np.matmul(k_starx.T, np.linalg.solve(k_xx + Sigma, 
                                                        np.matmul(A.T, ytrain - mtrain)/A.shape[0]))
    else:
        return m + np.matmul(k_starx.T, np.linalg.solve(k_xx + Sigma, ytrain - mtrain))



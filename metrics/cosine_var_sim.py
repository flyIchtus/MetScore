import numpy as np


def cosine_var_sim(Real, X):
    """
    Compute 1 - <varX . varReal> / (|varX| |varReal|)

    Inputs :
        X :  np.array with shape N x C x H x W

        Real :  np.array with shape N x C x H x W

    Returns :

        cvs :  np.array of shape C

    """

    varX = np.std(X, axis=0, ddof=1)
    varReal = np.std(Real, axis=0, ddof=1)

    normX = np.sqrt(np.sum(varX * varX, axis=(-1, -2)))
    normReal = np.sqrt(np.sum(varReal * varReal, axis=(-1, -2)))

    scal = np.sum(varX * varReal, axis=(-2, -1))

    cvs = 1.0 - scal / (normX * normReal + 1e-8)

    return cvs


def cosine_var_sim_multi_dates(Real, X):
    """
    Compute 1 - <varX . varReal> / (|varX| |varReal|)

    Inputs :
        X :  np.array with shape D x N x C x H x W

        Real :  np.array with shape D x N x C x H x W

    Returns :

        cvs :  np.array of shape C

    """

    varX = np.std(X, axis=1, ddof=1)
    varReal = np.std(Real, axis=1, ddof=1)

    normX = np.sqrt(np.sum(varX * varX, axis=(-1, -2)))
    normReal = np.sqrt(np.sum(varReal * varReal, axis=(-1, -2)))

    scal = np.sum(varX * varReal, axis=(-2, -1))

    cvs = 1.0 - scal / (normX * normReal + 1e-8)

    return cvs.mean(axis=0)

import numpy as np


def analytic_mse(ds):
    return np.linalg.pinv(ds.xs.T @ ds.xs) @ ds.xs.T @ ds.ys


def analytic_l2(ds, scale):
    return np.linalg.pinv(ds.xs.T @ ds.xs + scale * np.eye(ds.xs.shape[1])) @ ds.xs.T @ ds.ys

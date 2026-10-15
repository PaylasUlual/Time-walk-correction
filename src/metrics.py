# src/metrics.py
import numpy as np
from scipy.stats import norm

def fit_resolution(residuals, clip_ps=200):
    """Returns (mu,sigma) in ps from a Gaussian fit to the residuals. Only values within +- clip_ps of the median are used
    so that the long tails don't inflate sigma. The clip is around the median instead of zero because the residuals have a
    constant offset of about 150 ps."""
    residuals = np.asarray(residuals)
    center = np.median(residuals)
    clipped = residuals[np.abs(residuals - center) < clip_ps]
    mu, sigma = norm.fit(clipped)
    return mu, sigma
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def chapmanF2F1(h, Nmax2, hmax2, H, alpha2, Nmax1, alpha1, hmax1):
    z2 = (h - hmax2) / H
    z1 = (h - hmax1) / H

    ez2 = np.exp(-np.clip(z2, -50, 50))
    ez1 = np.exp(-np.clip(z1, -50, 50))

    term2 = alpha2 * (1 - z2 - ez2)
    term1 = alpha1 * (1 - z1 - ez1)

    term2 = np.clip(term2, -700, 700)
    term1 = np.clip(term1, -700, 700)

    exp2 = np.exp(term2)
    exp1 = np.exp(term1)

    # Final clip to prevent overflow when multiplied by Nmax
    exp2 = np.clip(exp2, 0, 1e300)
    exp1 = np.clip(exp1, 0, 1e300)

    return Nmax2 * exp2 + Nmax1 * exp1

def epstein(h, NmF2, hmF2, B2u0, k):
    z = h - hmF2
    B2u = B2u0 + k * z

    # Clip the exponential term to avoid overflow
    exp_term = np.exp(np.clip(z / B2u, -50, 50))
    N_norm = 4.0 * exp_term / ((1 + exp_term) ** 2)

    return NmF2 * N_norm

def single_chapman(h, Nmax, hmax, H, alpha):
    z = (h - hmax) / H
    ez = np.exp(-np.clip(z, -50, 50))
    term = np.clip(alpha * (1 - z - ez), -700, 700)
    return Nmax * np.exp(term)





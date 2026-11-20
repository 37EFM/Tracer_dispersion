"""Analytical results from Chapters 6 (Diffusion) and 7 (Mixing and dispersion)."""
from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid


def gaussian(x, t, D, x0=0.0, sigma0=0.0, mass=1.0, u=0.0):
    """Prototypical solution: a Gaussian of variance sigma0^2 + 2 D t centred on x0 + u t."""
    var = sigma0 ** 2 + 2 * D * t
    return mass / np.sqrt(2 * np.pi * var) * np.exp(-0.5 * (x - x0 - u * t) ** 2 / var)


def wall_bounded_profile(z, t, z0, D, H, n_modes=200, sigma0=0.0):
    """Diffusion between two walls at z = -H and z = 0 from a release at z0.

    Cosine series (no-flux walls); the n-th mode decays as exp(-n^2 pi^2 D t / H^2).
    Normalised to unit mass per unit width.  ``sigma0`` smooths the initial
    delta into a Gaussian of that width.
    """
    z = np.asarray(z, float)
    zp, z0p = z + H, z0 + H
    c = np.ones_like(z) / H
    for n in range(1, n_modes + 1):
        kn = n * np.pi / H
        c += 2 / H * np.cos(kn * zp) * np.cos(kn * z0p) * np.exp(-kn ** 2 * (D * t + sigma0 ** 2 / 2))
    return c


def mixing_time(H, D):
    """Transverse diffusion time H^2 / D (Chapter 7)."""
    return H ** 2 / D


def uniformity_time(H, D, tolerance=0.01):
    """Time after which the slowest mode of wall-bounded diffusion is below ``tolerance``."""
    return H ** 2 / (np.pi ** 2 * D) * np.log(2 / tolerance)


def taylor_K(z, u, D):
    """Taylor's effective longitudinal diffusivity, equation (7.3) of the notes,

        K = -(1/H) int u'(z) [ int_{-H}^{z} (1/D(z')) int_{-H}^{z'} u'(z'') dz'' dz' ] dz,

    for tabulated u(z) and D(z) on the same z (from -H to 0).  D may vary with z.
    """
    z = np.asarray(z, float)
    u = np.asarray(u, float)
    D = np.broadcast_to(np.asarray(D, float), z.shape)
    H = z[-1] - z[0]
    ubar = trapezoid(u, z) / H
    up = u - ubar
    inner = cumulative_trapezoid(up, z, initial=0.0)                 # int u' dz''
    # at a wall D -> 0 but so does the inner integral (no flux); the ratio is taken as 0 there
    ratio = np.divide(inner, D, out=np.zeros_like(inner), where=D > 0)
    middle = cumulative_trapezoid(ratio, z, initial=0.0)             # int (1/D) int u'
    return -trapezoid(up * middle, z) / H


def taylor_K_linear(delta_u, H, D):
    """Closed form of taylor_K for a linear profile with velocity difference delta_u (Exercise 7.1)."""
    return delta_u ** 2 * H ** 2 / (120 * D)


def shear_dispersion_scaling(delta_u, H, D):
    """Order-of-magnitude estimate K ~ (delta u)^2 H^2 / D, equation (7.1) of the notes."""
    return delta_u ** 2 * H ** 2 / D

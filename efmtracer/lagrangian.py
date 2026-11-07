r"""Lagrangian particles: the random walk equivalent to advection-diffusion.

A tracer particle in a flow u(z) with a diffusivity D(z) obeys the
stochastic differential equations (Ito form)

    dx = u(z) dt                      + sqrt(2 D_x(z)) dW_x
    dz = (dD/dz) dt                   + sqrt(2 D(z))   dW_z

where dW are increments of a Wiener process (Gaussian, zero mean, variance
dt).  The drift term dD/dz is essential whenever D varies with z: without
it, particles accumulate where D is small, which violates the *well-mixed
condition* -- a uniform distribution must stay uniform (Visser 1997).  The
concentration of a cloud of such particles satisfies the same equation as
the Eulerian solver, d/dz(D dc/dz) with D inside the derivative.

Three time-stepping schemes are provided:

``'naive'``
    Euler without the drift term.  Wrong when D varies; included to show
    what goes wrong.
``'euler'``
    Euler-Maruyama: the equations above with the coefficients frozen at
    the start of the step.  First order.
``'rk2'``  (default)
    A second-order Runge-Kutta (Heun) treatment of the deterministic part:
    the velocity u and the drift dD/dz are averaged between the start of
    the step and a predicted end point, so that a particle crossing layers
    of different speed is advected with the mean speed along its path.
    The random part uses Milstein's correction for the z-dependence of
    the noise amplitude, ``+ (1/2) (dD/dz) (dW^2 - dt)``, which is what
    keeps the well-mixed condition near the walls, where D goes to zero and
    the amplitude sqrt(2D) varies fastest.  (The plain stochastic Heun
    scheme, which converges to the Stratonovich form of the equation,
    under-fills the wall layers at any practical time step for this
    reason; see the tests.)

Walls at z = -H and z = 0 are reflecting.
"""
from __future__ import annotations

import numpy as np


def reflect(z, H):
    """Reflect positions back into [-H, 0]."""
    z = np.where(z > 0, -z, z)
    z = np.where(z < -H, -2 * H - z, z)
    return np.clip(z, -H, 0.0)       # a particle that jumped more than H is just clipped


def step(x, z, dt, prof, rng, scheme="rk2", Dx=None):
    """Advance all particles by dt. ``prof`` provides u(z), D(z), dDdz(z)."""
    H = prof.H
    n = len(x)
    dWx = rng.standard_normal(n) * np.sqrt(dt)
    dWz = rng.standard_normal(n) * np.sqrt(dt)
    Dx_fun = Dx

    def coefficients(zz):
        """Velocity, drift and noise amplitudes of both equations at height zz."""
        u, D, dDdz = prof.evaluate(zz)
        Dx = D if Dx_fun is None else Dx_fun(zz)
        return u, dDdz, np.sqrt(2 * Dx), np.sqrt(2 * D)

    u, dDdz, bx, bz = coefficients(z)
    if scheme == "naive":
        x_new = x + u * dt + bx * dWx
        z_new = z + bz * dWz
    elif scheme == "euler":
        x_new = x + u * dt + bx * dWx
        z_new = z + dDdz * dt + bz * dWz
    elif scheme in ("rk2", "heun"):
        noise_z = bz * dWz + 0.5 * dDdz * (dWz ** 2 - dt)        # Milstein: b db/dz = dD/dz
        # predictor
        z_pred = reflect(z + dDdz * dt + noise_z, H)
        u2, dDdz2, bx2, _ = coefficients(z_pred)
        # corrector: trapezoidal drift and velocity, same random increments
        z_new = z + 0.5 * (dDdz + dDdz2) * dt + noise_z
        x_new = x + 0.5 * (u + u2) * dt + 0.5 * (bx + bx2) * dWx
    else:
        raise ValueError("scheme must be 'naive', 'euler' or 'rk2'")
    return x_new, reflect(z_new, H)


def release_blob(n, x0, z0, sigma_x, sigma_z, H, rng):
    """Particles with the same Gaussian distribution as eulerian.gaussian_blob."""
    x = rng.normal(x0, sigma_x, n)
    z = rng.normal(z0, sigma_z, n)
    # resample the few that fall outside the channel
    bad = (z < -H) | (z > 0)
    while bad.any():
        z[bad] = rng.normal(z0, sigma_z, bad.sum())
        bad = (z < -H) | (z > 0)
    return x, z


def run(x, z, dt, t_end, prof, rng, scheme="rk2", save_every=None, u_frame=0.0, Dx=None):
    """Integrate the particles; returns (times, positions, moments).

    ``positions`` is a list of (x, z) arrays saved every ``save_every``
    seconds; ``moments`` holds the mean and variance of x (in the frame
    moving at ``u_frame``) and of z at every step.
    """
    x = np.array(x, float)
    z = np.array(z, float)
    nsteps = int(round(t_end / dt))
    save_every = t_end / 10 if save_every is None else save_every
    stride = max(int(round(save_every / dt)), 1)
    times, pos = [0.0], [(x.copy(), z.copy())]
    hist = dict(t=[0.0], mean_x=[x.mean()], var_x=[x.var()], mean_z=[z.mean()], var_z=[z.var()])
    for n in range(1, nsteps + 1):
        x, z = step(x, z, dt, prof, rng, scheme, Dx)
        t = n * dt
        xf = x - u_frame * t
        hist["t"].append(t); hist["mean_x"].append(xf.mean()); hist["var_x"].append(xf.var())
        hist["mean_z"].append(z.mean()); hist["var_z"].append(z.var())
        if n % stride == 0 or n == nsteps:
            times.append(t)
            pos.append((x - u_frame * t, z.copy()))
    return np.array(times), pos, {k: np.array(v) for k, v in hist.items()}


def concentration(x, z, grid, mass=1.0, L_wrap=None):
    """Bin particles on an Eulerian grid; returns c (nz x nx) normalised to ``mass``.

    ``L_wrap`` wraps x periodically onto [0, L) like the Eulerian domain.
    """
    xx = np.mod(x, L_wrap) if L_wrap else x
    xedges = np.concatenate([grid.x - grid.dx / 2, [grid.x[-1] + grid.dx / 2]])
    zedges = grid.zi
    h, _, _ = np.histogram2d(z, xx, bins=[zedges, xedges])
    return h * mass / (len(x) * grid.dx * grid.dz)

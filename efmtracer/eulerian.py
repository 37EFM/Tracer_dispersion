r"""Eulerian solver for the advection-diffusion equation in a channel.

The equation solved is

    dc/dt + u(z) dc/dx = d/dx( D_x(z) dc/dx ) + d/dz( D(z) dc/dz ),

for -H <= z <= 0 with no flux through the bottom and the surface, and
periodic in x.  The diffusivity sits *inside* the vertical derivative, which
matters when it varies with z: the vertical flux is -D(z) dc/dz, not
-D d2c/dz2.

Numerics (one time step, operator splitting):

1. **Along x** the velocity depends on z only, so each horizontal line of
   the grid is simply translated by u(z) dt and smoothed by D_x(z).  Both
   are done exactly in Fourier space: a translation is a phase shift and
   diffusion a damping of each wavenumber.  No numerical diffusion, no
   stability limit.
2. **Along z** the equation d c/dt = d/dz(D dc/dz) is discretised with
   finite volumes -- the flux across the interface between cells j and j+1
   is D_{j+1/2} (c_{j+1} - c_j)/dz -- and stepped implicitly (backward
   Euler, or Crank-Nicolson with ``theta=0.5``), which is a tridiagonal
   system for every column.  Mass is conserved exactly.

The computation is done in a frame moving at ``u_frame`` (default the
depth-averaged velocity), so that the patch stays inside the domain.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded


class Grid:
    """Cell-centred grid: x periodic on [0, L), z on [-H, 0] with nz cells."""

    def __init__(self, L, nx, H, nz):
        self.L, self.H, self.nx, self.nz = float(L), float(H), int(nx), int(nz)
        self.dx = self.L / nx
        self.dz = self.H / nz
        self.x = (np.arange(nx) + 0.5) * self.dx
        self.z = -self.H + (np.arange(nz) + 0.5) * self.dz          # cell centres
        self.zi = -self.H + np.arange(nz + 1) * self.dz             # interfaces
        self.k = 2 * np.pi * np.fft.rfftfreq(nx, self.dx)            # wavenumbers


def gaussian_blob(grid, x0, z0, sigma_x, sigma_z, mass=1.0):
    """Initial condition: a Gaussian patch of unit mass (per unit width)."""
    X, Z = np.meshgrid(grid.x, grid.z)
    c = np.exp(-0.5 * ((X - x0) / sigma_x) ** 2 - 0.5 * ((Z - z0) / sigma_z) ** 2)
    return c * mass / (c.sum() * grid.dx * grid.dz)


def _vertical_matrix(grid, D_interfaces, dt, theta):
    """Banded matrix of the implicit vertical diffusion step."""
    nz, dz = grid.nz, grid.dz
    Di = np.asarray(D_interfaces, float).copy()
    Di[0] = Di[-1] = 0.0                                             # no flux through the walls
    r = theta * dt / dz ** 2
    lower = -r * Di[1:nz]            # coefficient of c_{j-1} in row j (j = 1..nz-1)
    upper = -r * Di[1:nz]            # coefficient of c_{j+1} in row j (j = 0..nz-2)
    diag = 1 + r * (Di[1:] + Di[:-1])
    ab = np.zeros((3, nz))
    ab[0, 1:] = upper
    ab[1] = diag
    ab[2, :-1] = lower
    return ab


def _explicit_vertical(c, grid, D_interfaces, dt, theta):
    """The explicit part (1 - theta) of the vertical diffusion operator."""
    if theta >= 1.0:
        return c
    Di = np.asarray(D_interfaces, float).copy()
    Di[0] = Di[-1] = 0.0
    flux = np.zeros((grid.nz + 1, c.shape[1]))
    flux[1:-1] = Di[1:-1, None] * (c[1:] - c[:-1]) / grid.dz
    return c + (1 - theta) * dt / grid.dz * (flux[1:] - flux[:-1])


class Solver:
    """Advection-diffusion of c(z, x) on a Grid.

    Parameters
    ----------
    grid : Grid
    u : callable z -> velocity [m/s]
    D : callable z -> vertical diffusivity [m^2/s]
    Dx : callable z -> horizontal diffusivity; default the same as D
    u_frame : velocity of the moving frame; default the depth average of u
    theta : 1 for backward Euler (default, robust), 0.5 for Crank-Nicolson
    """

    def __init__(self, grid, u, D, Dx=None, u_frame=None, theta=1.0):
        self.grid = grid
        self.u_z = np.asarray(u(grid.z), float)
        self.D_i = np.asarray(D(grid.zi), float)                     # at interfaces
        self.Dx_z = np.asarray((Dx or D)(grid.z), float)
        self.u_frame = float(np.mean(self.u_z)) if u_frame is None else float(u_frame)
        self.theta = theta
        self._dt = None

    def _prepare(self, dt):
        g = self.grid
        k = g.k[None, :]
        # phase shift for the translation, damping for horizontal diffusion
        self._propagator = np.exp(-1j * k * (self.u_z - self.u_frame)[:, None] * dt
                                  - self.Dx_z[:, None] * k ** 2 * dt)
        self._ab = _vertical_matrix(g, self.D_i, dt, self.theta)
        self._dt = dt

    def step(self, c, dt):
        """Advance c (shape nz x nx) by dt."""
        if dt != self._dt:
            self._prepare(dt)
        # 1. along x, in Fourier space
        c = np.fft.irfft(np.fft.rfft(c, axis=1) * self._propagator, n=self.grid.nx, axis=1)
        # 2. along z, implicit finite volumes
        rhs = _explicit_vertical(c, self.grid, self.D_i, dt, self.theta)
        return solve_banded((1, 1), self._ab, rhs)

    def run(self, c0, dt, t_end, save_every=None, callback=None):
        """Integrate from c0 to t_end; returns (times, snapshots, moments).

        ``snapshots`` is a list of c arrays saved every ``save_every`` seconds
        (default: ten snapshots), ``moments`` a dict of time series computed at
        every step by :func:`moments` (mass, mean and variance of x in the
        moving frame, mean and variance of z).
        """
        c = np.array(c0, float)
        nsteps = int(round(t_end / dt))
        save_every = t_end / 10 if save_every is None else save_every
        save_stride = max(int(round(save_every / dt)), 1)
        times, snaps = [0.0], [c.copy()]
        hist = {k: [v] for k, v in moments(c, self.grid).items()}
        hist["t"] = [0.0]
        for n in range(1, nsteps + 1):
            c = self.step(c, dt)
            m = moments(c, self.grid)
            for k, v in m.items():
                hist[k].append(v)
            hist["t"].append(n * dt)
            if n % save_stride == 0 or n == nsteps:
                times.append(n * dt)
                snaps.append(c.copy())
            if callback is not None:
                callback(n * dt, c)
        return np.array(times), snaps, {k: np.array(v) for k, v in hist.items()}


def moments(c, grid):
    """Mass and the first two moments of a concentration field (shape nz x nx)."""
    dA = grid.dx * grid.dz
    mass = c.sum() * dA
    cx = c.sum(axis=0) * grid.dz                 # depth-integrated, function of x
    cz = c.sum(axis=1) * grid.dx                 # x-integrated, function of z
    mx = (cx * grid.x).sum() * grid.dx / mass
    vx = (cx * (grid.x - mx) ** 2).sum() * grid.dx / mass
    mz = (cz * grid.z).sum() * grid.dz / mass
    vz = (cz * (grid.z - mz) ** 2).sum() * grid.dz / mass
    return dict(mass=mass, mean_x=mx, var_x=vx, mean_z=mz, var_z=vz)


def depth_average(c, grid):
    """Depth-averaged concentration as a function of x."""
    return c.mean(axis=0)

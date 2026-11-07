"""Velocity and diffusivity profiles as smooth functions of z.

The channel occupies -H <= z <= 0 (z = 0 at the surface, as in GOTM).
"""
from __future__ import annotations

from importlib import resources
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator


class Profiles:
    """Callable profiles u(z), D(z) and dD/dz(z) on -H <= z <= 0.

    Built from tabulated values with a shape-preserving cubic interpolant
    (PCHIP), which has a continuous first derivative and no overshoot.
    Outside the table the end values are held, so the callables are safe
    for particles that touch the walls.
    """

    def __init__(self, z_u, u, z_D, D, H, name=""):
        self.H = float(H)
        self.name = name
        self.z_u, self.u_table = np.asarray(z_u, float), np.asarray(u, float)
        self.z_D, self.D_table = np.asarray(z_D, float), np.asarray(D, float)
        self._u = PchipInterpolator(self.z_u, self.u_table, extrapolate=True)
        self._D = PchipInterpolator(self.z_D, np.maximum(self.D_table, 0.0), extrapolate=True)
        self._dDdz = self._D.derivative()
        # one spline with three columns (u, D, dD/dz) on a common fine table, so that
        # the particle model gets all three with a single evaluation
        zt = np.linspace(-self.H, 0.0, 2001)
        cols = np.column_stack([self._u(zt), np.maximum(self._D(zt), 0.0), self._dDdz(zt)])
        self._all = PchipInterpolator(zt, cols, extrapolate=True)

    def _clip(self, z):
        return np.clip(np.asarray(z, float), -self.H, 0.0)

    def u(self, z):
        return self._u(self._clip(z))

    def D(self, z):
        return np.maximum(self._D(self._clip(z)), 0.0)

    def dDdz(self, z):
        return self._dDdz(self._clip(z))

    def evaluate(self, z):
        """u, D and dD/dz at z in one call (used by the particle model)."""
        v = self._all(self._clip(z))
        return v[..., 0], np.maximum(v[..., 1], 0.0), v[..., 2]

    @property
    def u_mean(self):
        """Depth-averaged velocity."""
        z = np.linspace(-self.H, 0, 2001)
        return float(np.trapezoid(self.u(z), z) / self.H)

    @property
    def D_mean(self):
        z = np.linspace(-self.H, 0, 2001)
        return float(np.trapezoid(self.D(z), z) / self.H)

    @property
    def delta_u(self):
        z = np.linspace(-self.H, 0, 2001)
        u = self.u(z)
        return float(u.max() - u.min())


def constant(H, u=0.0, D=1e-3):
    """Uniform flow and constant diffusivity between two walls."""
    z = np.array([-H, 0.0])
    return Profiles(z, [u, u], z, [D, D], H, name=f"uniform u = {u}, D = {D}")


def linear_shear(H, u_bottom, u_top, D):
    """Linear velocity profile with constant diffusivity (Exercise 7.1 of the notes)."""
    z = np.array([-H, 0.0])
    return Profiles(z, [u_bottom, u_top], z, [D, D], H, name="linear shear")


def couette(path=None, diffusivity="nuh"):
    """The turbulent Couette flow of the GOTM exercise (final profiles of a 24 h run).

    Parameters
    ----------
    path : str or Path
        A ``.npz`` written by ``save_from_gotm``; default the file shipped in ``data/``.
    diffusivity : 'nuh' or 'num'
        Use the turbulent diffusivity of scalars (``nuh``) or the eddy
        viscosity (``num``) as D(z).  They differ by the turbulent Prandtl
        number of the closure.
    """
    if path is None:
        path = resources.files("efmtracer").joinpath("../data/couette_profiles.npz")
        if not Path(str(path)).exists():
            path = Path(__file__).resolve().parent.parent / "data" / "couette_profiles.npz"
    d = np.load(str(path))
    return Profiles(d["z"], d["u"], d["zi"], d[diffusivity], d["H"], name=f"GOTM Couette ({diffusivity})")


def save_from_gotm(ds, path, time=-1):
    """Store the profiles of a GOTM output Dataset (from efmgotm.run) for later use."""
    last = ds.isel(time=time)
    np.savez(path, z=last.z.values, u=last.u.values, zi=last.zi.values, num=last.num.values,
             nuh=last.nuh.values, u_taub=float(last.u_taub), H=float(-last.zi.values[0]))

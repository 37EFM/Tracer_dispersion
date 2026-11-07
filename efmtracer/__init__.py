"""efmtracer -- advection-diffusion of a tracer in a channel, Eulerian and Lagrangian.

Course 37EFM Environmental Fluid Mechanics (TU/e).  Two ways of computing
how a patch of tracer spreads in a sheared, turbulent channel flow such as
the Couette flow of the GOTM exercises:

* :mod:`efmtracer.eulerian`   solves the advection-diffusion equation for the
  concentration c(x, z, t) with a depth-dependent diffusivity D(z) inside the
  vertical derivative;
* :mod:`efmtracer.lagrangian` follows particles whose motion is the
  stochastic differential equation equivalent to that same equation, with a
  second-order (Heun, "RK2") time stepping.

:mod:`efmtracer.profiles` loads the Couette profiles u(z) and D(z) and
:mod:`efmtracer.theory` holds the analytical results of Chapters 6 and 7 of
the lecture notes (Gaussian spreading, wall-bounded diffusion, Taylor's
effective diffusivity).
"""
from . import eulerian, lagrangian, profiles, theory

__version__ = "0.1.0"

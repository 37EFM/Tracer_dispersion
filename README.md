# Tracer_dispersion — Eulerian and Lagrangian advection–diffusion in a channel

Course material for **37EFM Environmental Fluid Mechanics** (Eindhoven
University of Technology). A patch of tracer is released in a channel
and followed in two ways: as a concentration field obeying the
advection–diffusion equation, and as a cloud of particles obeying the
equivalent stochastic equations. The flow and the turbulent diffusivity
come from the turbulent Couette simulation of the
[GOTM exercises](https://github.com/37EFM/GOTM_exercises).

| Notebook | What it does |
|---|---|
| `notebooks/01_constant_diffusion.ipynb` | A blob between two walls with constant $D$ and uniform flow: how long until it is spread over the depth, for molecular and for turbulent diffusivities. Checks against the Gaussian and the wall-bounded series solutions of Chapter 5, and the time to uniformity measured for a release at mid-depth and one near the bed. |
| `notebooks/02_couette_dispersion.ipynb` | The same blob in the turbulent Couette flow, with $u(z)$ and $D(z)$ from GOTM: shear dispersion, Taylor's effective diffusivity of Chapter 7, and the three regimes of the variance. |
| `notebooks/03_lagrangian_particles.ipynb` | Particles with a second-order stochastic time step: the well-mixed test, and a side-by-side comparison with the Eulerian solution. |

## Installation

```bash
git clone https://github.com/37EFM/Tracer_dispersion.git
cd Tracer_dispersion
conda env create -f environment.yml
conda activate efmtracer
jupyter lab notebooks
```

Only numpy, scipy and matplotlib are needed; `pip install -e .` in the
repository works as well. The environment of the GOTM exercises can
also be used (`pip install -e .` inside it).

## The equations and the methods

The concentration $c(x, z, t)$ in a channel $-H \le z \le 0$ obeys

$$\frac{\partial c}{\partial t} + u(z)\frac{\partial c}{\partial x}
= \frac{\partial}{\partial x}\Big(D_x\frac{\partial c}{\partial x}\Big)
+ \frac{\partial}{\partial z}\Big(D(z)\frac{\partial c}{\partial z}\Big),$$

with no flux through the bottom and the surface. The diffusivity sits
inside the vertical derivative, which matters when it varies with $z$.

**Eulerian** (`efmtracer/eulerian.py`). The domain is periodic in $x$
and the computation is done in the frame moving with the depth-averaged
velocity. Because $u$ depends on $z$ only, the along-$x$ step is a
translation of each horizontal line plus a damping of each wavenumber,
done exactly in Fourier space. The vertical step is a finite-volume
discretisation, with the flux $D_{j+1/2}(c_{j+1}-c_j)/\Delta z$ across
each interface, stepped implicitly (backward Euler, or Crank–Nicolson
with `theta=0.5`). Mass is conserved to round-off.

**Lagrangian** (`efmtracer/lagrangian.py`). Particles obey

$$dx = u(z)\,dt + \sqrt{2D_x}\,dW_x, \qquad
dz = \frac{dD}{dz}\,dt + \sqrt{2D(z)}\,dW_z,$$

whose cloud has the same concentration equation as above. The drift
$dD/dz$ enforces the well-mixed condition (Visser 1997). The default
time step `rk2` is a second-order Runge–Kutta treatment of the
velocity and the drift (trapezoidal between the start of the step and
a predicted end point, with the same random increments) combined with
Milstein's correction for the $z$-dependence of the noise amplitude,
which keeps the wall layers correctly filled. `euler` is
Euler–Maruyama; `naive` leaves the drift out and shows what goes wrong.
Walls reflect.

**Profiles** (`efmtracer/profiles.py`). `profiles.couette()` loads
$u(z)$ and $D(z)$ of the GOTM Couette run from
`data/couette_profiles.npz` (final profiles of the 24 h run; $D$ is the
turbulent diffusivity of scalars `nuh`, the eddy viscosity `num` can be
chosen instead). `profiles.constant` and `profiles.linear_shear` give
the analytical cases. `profiles.save_from_gotm` stores the profiles of
any other GOTM run.

**Theory** (`efmtracer/theory.py`). The Gaussian solution, the
wall-bounded cosine series, the transverse diffusion time $H^2/D$,
Taylor's effective diffusivity for arbitrary $u(z)$ and $D(z)$,

$$E = -\frac{1}{H}\int_{-H}^{0} u'(z)\left[\int_{-H}^{z}\frac{1}{D(z')}\int_{-H}^{z'}u'(z'')\,dz''\,dz'\right]dz,$$

its closed form $(\Delta u)^2 H^2/(120 D)$ for a linear profile, and the
scaling $(\Delta u)^2 H^2 / D$.

## Results to expect

For the Couette profiles ($H = 10$ m, $\Delta u = 0.86$ m/s,
$\bar D = 0.031$ m$^2$/s): $E = 3.6$ m$^2$/s, about 115 times $\bar D$,
and a transverse diffusion time of 0.9 h. The variance of the patch
grows as $t^2$ for the first minutes, then bends over to $2Et$ after
about one transverse diffusion time. Particles and field agree to a few
per cent at a 2 s particle step. The tests in `tests/` check all of
this on synthetic cases; run them with `pytest`.

## References

- Taylor, G. I. (1953). Dispersion of soluble matter in solvent flowing
  slowly through a tube. *Proc. R. Soc. Lond. A* 219, 186–203.
- Taylor, G. I. (1954). The dispersion of matter in turbulent flow
  through a pipe. *Proc. R. Soc. Lond. A* 223, 446–468.
- Visser, A. W. (1997). Using random walk models to simulate the
  vertical distribution of particles in a turbulent water column.
  *Mar. Ecol. Prog. Ser.* 158, 275–281.
- Kloeden, P. E. and Platen, E. (1992). *Numerical Solution of
  Stochastic Differential Equations*. Springer.

## Licence

MIT, see `LICENSE`.

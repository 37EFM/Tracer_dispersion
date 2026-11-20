"""Run with ``pytest``."""
import numpy as np

from efmtracer import eulerian, lagrangian, profiles, theory

H = 10.0


def test_eulerian_gaussian_and_mass():
    D = 0.01
    prof = profiles.constant(H, u=0.5, D=D)
    grid = eulerian.Grid(L=400, nx=400, H=H, nz=100)
    c0 = eulerian.gaussian_blob(grid, 200, -H / 2, 5.0, 0.5)
    solver = eulerian.Solver(grid, prof.u, prof.D)
    _, snaps, hist = solver.run(c0, dt=5.0, t_end=2000, save_every=2000)
    assert abs(hist["mass"][-1] - 1.0) < 1e-10
    assert abs(hist["var_x"][-1] - (25 + 2 * D * 2000)) < 1e-6          # exact in Fourier space
    series = theory.wall_bounded_profile(grid.z, 2000, -H / 2, D, H, sigma0=0.5)
    assert np.abs(snaps[-1].sum(axis=1) * grid.dx - series).max() < 1e-4


def test_eulerian_varying_D_conserves_mass_and_reaches_uniform():
    prof = profiles.couette()
    grid = eulerian.Grid(L=2000, nx=200, H=H, nz=50)
    c0 = eulerian.gaussian_blob(grid, 1000, -H / 2, 5.0, 0.5)
    _, snaps, hist = eulerian.Solver(grid, prof.u, prof.D).run(c0, dt=10.0, t_end=4 * theory.mixing_time(H, prof.D_mean), save_every=1e9)
    assert abs(hist["mass"][-1] - 1.0) < 1e-9
    assert abs(hist["var_z"][-1] - H ** 2 / 12) < 0.05                   # vertically uniform


def test_taylor_E_linear_profile():
    z = np.linspace(-H, 0, 2001)
    E = theory.taylor_E(z, 0.5 * (z + H) / H, 0.01)
    assert abs(E - theory.taylor_E_linear(0.5, H, 0.01)) < 1e-3


def test_shear_dispersion_matches_taylor():
    prof = profiles.linear_shear(H, 0.0, 0.5, 0.02)
    z = np.linspace(-H, 0, 2001)
    E = theory.taylor_E(z, prof.u(z), prof.D(z))
    grid = eulerian.Grid(L=8000, nx=800, H=H, nz=50)        # wide enough: sigma_x reaches ~700 m
    c0 = eulerian.gaussian_blob(grid, 4000, -H / 2, 5.0, 0.5)
    T = theory.mixing_time(H, 0.02)
    _, _, hist = eulerian.Solver(grid, prof.u, prof.D).run(c0, dt=10.0, t_end=6 * T, save_every=1e9)
    late = hist["t"] > 3 * T
    E_fit = np.polyfit(hist["t"][late], hist["var_x"][late], 1)[0] / 2
    assert abs(E_fit / E - 1) < 0.05


def test_lagrangian_constant_D_variance():
    rng = np.random.default_rng(0)
    prof = profiles.constant(H, u=0.5, D=0.01)
    x, z = lagrangian.release_blob(20000, 200, -H / 2, 5.0, 0.5, H, rng)
    _, _, hist = lagrangian.run(x, z, 5.0, 2000, prof, rng, "rk2", u_frame=0.5)
    assert abs(hist["var_x"][-1] - 65) < 3                                # 2Dt + sigma0^2, within noise
    assert abs(hist["var_z"][-1] - H ** 2 / 12) < 0.3


def test_well_mixed_condition():
    rng = np.random.default_rng(1)
    prof = profiles.couette()
    T = theory.mixing_time(H, prof.D_mean)
    for scheme, flat in [("naive", False), ("euler", True), ("rk2", True)]:
        z0 = rng.uniform(-H, 0, 40000)
        _, pos, _ = lagrangian.run(np.zeros(40000), z0, 2.0, T, prof, rng, scheme)
        h, _ = np.histogram(pos[-1][1], bins=10, range=(-H, 0))
        ratio = h / h.mean()
        assert (np.abs(ratio - 1).max() < 0.08) == flat


def test_lagrangian_matches_eulerian_in_shear():
    rng = np.random.default_rng(2)
    prof = profiles.couette()
    grid = eulerian.Grid(L=4000, nx=400, H=H, nz=50)
    c0 = eulerian.gaussian_blob(grid, 2000, -H / 2, 5.0, 0.5)
    solver = eulerian.Solver(grid, prof.u, prof.D)
    _, _, he = solver.run(c0, dt=5.0, t_end=1800, save_every=1e9)
    x, z = lagrangian.release_blob(20000, 2000, -H / 2, 5.0, 0.5, H, rng)
    _, _, hl = lagrangian.run(x, z, 1.0, 1800, prof, rng, "rk2", u_frame=solver.u_frame)
    assert abs(hl["var_x"][-1] / he["var_x"][-1] - 1) < 0.06

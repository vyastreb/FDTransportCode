"""
Analytical regression tests for Cartesian and polar transport solvers.

We verify that the discrete solvers reproduce known 1D solutions of the
Reynolds equation for:
  * constant gap fields
  * gap fields varying linearly between inlet and outlet.
"""

import sys
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse.linalg import spsolve

from fluxflow.transport import create_diffusion_matrix, face_k
from fluxflow.transport_polar import create_diffusion_matrix_polar

TOL_CART = 1e-5
TOL_POLAR = 1e-5

_REPORT_ROWS = []

def _record_result(name: str, err_abs: float, err_rel: float, tol_abs: float, tol_rel: float):
    _REPORT_ROWS.append((name, err_abs, err_rel, tol_abs, tol_rel))


@pytest.fixture(scope="session", autouse=True)
def _print_summary():
    yield
    if not _REPORT_ROWS:
        return

    tol_abs = _REPORT_ROWS[0][3]
    tol_rel = _REPORT_ROWS[0][4]
    sys.stdout.write("\n")
    sys.stdout.write(f"Tolerances: abs={tol_abs:.3e}, rel={tol_rel:.3e}\n")
    sys.stdout.write(f"{'test':<20s}{'max_abs':>14s}{'max_rel':>14s}\n")
    sys.stdout.write("-" * 48 + "\n")

    for name, err_abs, err_rel, _, _ in _REPORT_ROWS:
        line = f"{name:<20s}{err_abs:>14.3e}{err_rel:>14.3e}"
        sys.stdout.write(line + "\n")

# ---------------------------------------------------------------------------
# Analytical helpers

def _analytic_cartesian_linear(x, g_in, g_out, p_in=0.0, p_out=1.0):
    """
    Analytical pressure for 1D Reynolds equation with linear gap g(x).

    g(x) = g_in + (g_out - g_in) * x   for x in [0, 1]
    Boundary conditions: p(0) = p_in, p(1) = p_out
    """
    slope = g_out - g_in
    if abs(slope) < 1e-14:
        return p_in + (p_out - p_in) * x

    inv_g0_sq = 1.0 / (g_in**2)
    inv_gx_sq = 1.0 / (g_in + slope * x) ** 2

    numerator = (inv_g0_sq - inv_gx_sq) / (2.0 * slope)
    denominator = (inv_g0_sq - 1.0 / (g_in + slope) ** 2) / (2.0 * slope)

    return p_in + (p_out - p_in) * numerator / denominator


def _integral_linear_gap_polar(r, a, b):
    """Primitive of 1 / (r (a r + b)^3) used by the polar linear-gap solution."""
    eps = 1e-14
    r = np.asarray(r)

    if abs(a) < eps:
        if abs(b) < eps:
            raise ValueError("Degenerate gap field (a=b=0).")
        return np.log(r) / (b**3)

    if abs(b) < eps:
        # g(r) = a r  ->  integral = -1 / (3 a^3 r^3)
        return -1.0 / (3.0 * (a**3) * (r**3))

    return (
        (2.0 * a * r + 3.0 * b) / (2.0 * (b**2) * (a * r + b) ** 2)
        + (np.log(r) - np.log(r + b / a)) / (b**3)
    )


def _analytic_polar_linear(
    r,
    r_inner,
    r_outer,
    g_inner,
    g_outer,
    p_inner=1.0,
    p_outer=0.0,
):
    """
    Analytical solution for radial flow with gap linear in radius:
        g(r) = g_in + (g_out - g_in) * (r - r_in) / (r_out - r_in)
    Boundary conditions: p(r_in) = p_in, p(r_out) = p_out
    """
    a = (g_outer - g_inner) / (r_outer - r_inner)
    b = g_inner - a * r_inner

    F_r = _integral_linear_gap_polar(r, a, b)
    F_in = _integral_linear_gap_polar(r_inner, a, b)
    F_out = _integral_linear_gap_polar(r_outer, a, b)

    return p_inner + (p_outer - p_inner) * (F_r - F_in) / (F_out - F_in)


# ---------------------------------------------------------------------------
# Linear-system wrappers

def _solve_cartesian_pressure(gaps):
    """Solve the Cartesian matrix directly with SciPy for high accuracy."""
    n = gaps.shape[0]
    A, b = create_diffusion_matrix(n, gaps, penalty=None)
    pressure = spsolve(A.tocsc(), b)
    return pressure.reshape((n, n))


def _solve_polar_pressure(gaps, r_inner, r_outer, theta_extent):
    """Solve the polar matrix directly with SciPy for high accuracy."""
    A, b, _, _ = create_diffusion_matrix_polar(
        gaps,
        r_inner=r_inner,
        r_outer=r_outer,
        theta_extent=theta_extent,
        theta_bc_code=0,  # periodic
        p_inner=1.0,
        p_outer=0.0,
    )
    pressure = spsolve(A.tocsc(), b)
    return pressure.reshape(gaps.shape)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Discrete 1D helper for Cartesian configurations

def _discrete_cartesian_profile(g_line, p_west=0.0, p_east=1.0):
    """
    Closed-form solution of the 1D discrete system produced by the Cartesian solver.

    Parameters
    ----------
    g_line : array_like
        Gap values along the x-direction (length n).
    p_west : float
        Reservoir pressure at the inlet (x=0).
    p_east : float
        Reservoir pressure at the outlet (x=1).
    """
    g_line = np.asarray(g_line, dtype=float)
    n = g_line.size

    resistance = 1.0 / (g_line[0] ** 3)
    for k in range(1, n):
        resistance += 1.0 / face_k(g_line[k], g_line[k - 1])
    resistance += 1.0 / (g_line[-1] ** 3)

    flux = (p_east - p_west) / resistance

    profile = np.empty(n, dtype=float)
    profile[0] = p_west + flux / (g_line[0] ** 3)
    for i in range(1, n):
        profile[i] = profile[i - 1] + flux / face_k(g_line[i], g_line[i - 1])

    return profile


# ---------------------------------------------------------------------------
# Tests

def test_cartesian_constant_gap_matches_linear_profile():
    n = 30
    g0 = 0.5
    gaps = np.full((n, n), g0, dtype=float)

    pressure = _solve_cartesian_pressure(gaps)
    x_nodes = (np.arange(n) + 1.0) / (n + 1.0)

    numeric = pressure.mean(axis=1)
    g_line = np.full(n, g0, dtype=float)
    expected = _discrete_cartesian_profile(g_line, p_west=0.0, p_east=1.0)

    err_abs = np.max(np.abs(numeric - expected))
    denom = np.max(np.abs(expected))
    if denom < 1e-12:
        denom = 1.0
    err_rel = err_abs / denom
    _record_result("cartesian-constant", err_abs, err_rel, TOL_CART, TOL_CART)

    assert_allclose(numeric, expected, atol=TOL_CART, rtol=TOL_CART)


def test_cartesian_linear_gap_matches_analytic_solution():
    n = 64
    g_in = 0.6
    g_out = 1.1

    x_nodes = (np.arange(n) + 1.0) / (n + 1.0)
    g_line = g_in + (g_out - g_in) * x_nodes
    gaps = np.tile(g_line[:, None], (1, n))

    pressure = _solve_cartesian_pressure(gaps)
    numeric = pressure.mean(axis=1)
    expected = _discrete_cartesian_profile(g_line, p_west=0.0, p_east=1.0)

    err_abs = np.max(np.abs(numeric - expected))
    denom = np.max(np.abs(expected))
    if denom < 1e-12:
        denom = 1.0
    err_rel = err_abs / denom
    _record_result("cartesian-linear", err_abs, err_rel, TOL_CART, TOL_CART)

    assert_allclose(numeric, expected, atol=TOL_CART, rtol=TOL_CART)


def test_polar_constant_gap_matches_log_profile():
    g0 = 0.4
    r_inner = 1.0
    r_outer = 2.0
    n_r = 30 # discretize the radial direction
    n_theta = int(n_r / (r_outer - r_inner) * np.pi * (r_outer + r_inner))
    theta_extent = 2.0 * np.pi

    gaps = np.full((n_r, n_theta), g0, dtype=float)
    pressure = _solve_polar_pressure(gaps, r_inner, r_outer, theta_extent)

    r = np.linspace(r_inner, r_outer, n_r)
    numeric = pressure.mean(axis=1)
    expected = 1.0 + (0.0 - 1.0) * np.log(r / r_inner) / np.log(r_outer / r_inner)

    err_abs = np.max(np.abs(numeric - expected))
    denom = np.max(np.abs(expected))
    if denom < 1e-12:
        denom = 1.0
    err_rel = err_abs / denom
    _record_result("polar-constant", err_abs, err_rel, TOL_POLAR, TOL_POLAR)

    assert_allclose(numeric, expected, atol=TOL_POLAR, rtol=TOL_POLAR)


def test_polar_linear_gap_matches_analytic_solution():
    n_r = 96
    r_inner = 1.2
    r_outer = 2.0
    n_theta = 30
    n_theta = int(n_r / (r_outer - r_inner) * np.pi * (r_outer + r_inner))
    theta_extent = 2.0 * np.pi
    g_inner = 0.5
    g_outer = 0.9

    r = np.linspace(r_inner, r_outer, n_r)
    g_r = g_inner + (g_outer - g_inner) * (r - r_inner) / (r_outer - r_inner)
    gaps = np.tile(g_r[:, None], (1, n_theta))

    pressure = _solve_polar_pressure(gaps, r_inner, r_outer, theta_extent)
    numeric = pressure.mean(axis=1)
    expected = _analytic_polar_linear(
        r,
        r_inner=r_inner,
        r_outer=r_outer,
        g_inner=g_inner,
        g_outer=g_outer,
        p_inner=1.0,
        p_outer=0.0,
    )

    err_abs = np.max(np.abs(numeric - expected))
    denom = np.max(np.abs(expected))
    if denom < 1e-12:
        denom = 1.0
    err_rel = err_abs / denom
    _record_result("polar-linear", err_abs, err_rel, TOL_POLAR, TOL_POLAR)

    assert_allclose(numeric, expected, atol=TOL_POLAR, rtol=TOL_POLAR)



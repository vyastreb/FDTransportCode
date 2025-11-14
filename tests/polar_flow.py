"""
Manual polar solver test with annular roughness + deterministic turning pattern.

The gap field is composed of:
    * Deterministic term: A * cos(k_r * r) * sin(k_theta * theta)
    * Stochastic roughness sampled from a periodic surface of size (2 * r_e)^2

Roughness RMS is constrained to A / 3.
"""

import numpy as np

from fluxflow.transport_polar import solve_fluid_problem_polar, compute_total_flux_polar
from fluxflow import random_field as RF


# Geometry and surface parameters
A = 0.1
WAVE_AMPLITUDE = 0.02
ROUGHNESS_RMS = A / 5.0
K_THETA = 6
K_RADIAL = 10
R_INNER = 1.5
R_OUTER = 2.0
THETA_EXTENT = 2.0 * np.pi
THETA_BC = "periodic"

# Discretization
N_R = 1000
N_THETA = int(N_R / (R_OUTER - R_INNER) * np.pi * (R_OUTER + R_INNER))
N_CART = int(2 * (R_OUTER) * 2 * N_R / (R_OUTER - R_INNER))  # Cartesian grid used for roughness generation (covers (2*r_e) x (2*r_e))

print(f"Mesh size: {N_R} x {N_THETA} = {N_R * N_THETA}")

# Roughness spectrum parameters (similar to other tests)
HURST = 0.8
K_LOW = 40.0 / N_CART
K_HIGH = 800.0 / N_CART
RNG_SEED = 239

# Solver setup
SOLVER = "petsc-cg.hypre"
RTOL = 1e-9
BASE_GAP = 0.3  # Ensures positive mean gap after superposition


def generate_cartesian_roughness() -> np.ndarray:
    """Generate a periodic rough surface on a square domain (2 * r_e)^2."""
    np.random.seed(RNG_SEED)
    surface = RF.periodic_gaussian_random_field(
        dim=2,
        N=N_CART,
        Hurst=HURST,
        k_low=K_LOW,
        k_high=K_HIGH,
        plateau=True,
    )
    surface -= np.mean(surface)
    std = np.std(surface)
    if std > 0:
        surface *= ROUGHNESS_RMS / std
    else:
        surface[:] = 0.0
    return surface


def sample_roughness_to_annulus(surface: np.ndarray, R: np.ndarray, Theta: np.ndarray) -> np.ndarray:
    """Sample the Cartesian roughness onto the polar annulus using periodic bilinear interpolation."""
    N_cart = surface.shape[0]
    domain_size = 2.0 * R_OUTER  # (2 * r_e) extent in both x and y

    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)

    xi = ((X + domain_size / 2.0) / domain_size) % 1.0 * N_cart
    yi = ((Y + domain_size / 2.0) / domain_size) % 1.0 * N_cart

    i0 = np.floor(xi).astype(np.int32)
    j0 = np.floor(yi).astype(np.int32)
    tx = xi - i0
    ty = yi - j0

    i1 = (i0 + 1) % N_cart
    j1 = (j0 + 1) % N_cart

    z00 = surface[i0, j0]
    z10 = surface[i1, j0]
    z01 = surface[i0, j1]
    z11 = surface[i1, j1]

    sampled = (
        (1.0 - tx) * (1.0 - ty) * z00
        + tx * (1.0 - ty) * z10
        + (1.0 - tx) * ty * z01
        + tx * ty * z11
    )

    sampled -= np.mean(sampled)
    rms = np.sqrt(np.mean(sampled**2))
    if rms > 0:
        sampled *= ROUGHNESS_RMS / rms
    return sampled


def build_annular_gap():
    """Construct the annular gap field with deterministic and stochastic components."""
    r = np.linspace(R_INNER, R_OUTER, N_R)
    theta = np.linspace(0.0, THETA_EXTENT, N_THETA, endpoint=False)
    R, Theta = np.meshgrid(r, theta, indexing='ij')

    turned_surface = A * np.cos(2 * np.pi * K_RADIAL * (R - R_INNER) / (R_OUTER - R_INNER)) + WAVE_AMPLITUDE * np.sin(K_THETA * Theta)

    roughness_surface = generate_cartesian_roughness()
    roughness_polar = sample_roughness_to_annulus(roughness_surface, R, Theta)

    gap = BASE_GAP + turned_surface + roughness_polar
    gap = np.clip(gap, 0.0, None)
    return gap, R, Theta


def run_annulus_simulation():
    g_annulus, R, Theta = build_annular_gap()
    g_annulus -= 0.2
    g_annulus[g_annulus < 0] = 0

    gaps_filtered, pressure, flux, dr, dtheta = solve_fluid_problem_polar(
        gaps=g_annulus,
        r_inner=R_INNER,
        r_outer=R_OUTER,
        theta_extent=THETA_EXTENT,
        theta_bc=THETA_BC,
        solver=SOLVER,
        rtol=RTOL,
        p_inner=1.0,
        p_outer=0.0,
        dilation_iterations=1,
    )

    if gaps_filtered is None or flux is None or pressure is None:
        raise RuntimeError("Polar solver failed to find a percolating channel.")

    Q_total, flux_error = compute_total_flux_polar(gaps_filtered, flux, R_INNER, R_OUTER, dtheta)

    print(f"Q_total = {Q_total:.6e}")
    print(f"Flux conservation error = {flux_error:.3e}")

    return g_annulus, R, Theta, gaps_filtered, pressure, flux, dr, dtheta, Q_total, flux_error


if __name__ == "__main__":
    (
        gap_raw,
        R,
        Theta,
        gap_filtered,
        pressure,
        flux,
        dr,
        dtheta,
        Q,
        err,
    ) = run_annulus_simulation()

    import matplotlib.pyplot as plt

    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    flux_intensity = np.nan_to_num(np.linalg.norm(flux, axis=2))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), subplot_kw={"aspect": "equal"})

    pcm0 = axes[0].pcolormesh(X, Y, gap_raw, shading="auto")
    axes[0].set_title("Gap field")
    cbar0 = fig.colorbar(pcm0, ax=axes[0], shrink=0.9, pad=0.02)

    pcm1 = axes[1].pcolormesh(X, Y, pressure, shading="auto")
    axes[1].set_title("Pressure")
    cbar1 = fig.colorbar(pcm1, ax=axes[1], shrink=0.9, pad=0.02)

    log_flux_intensity = np.log10(flux_intensity)
    pcm2 = axes[2].pcolormesh(
        X, Y, log_flux_intensity, cmap="jet", shading="auto", vmin=-6, vmax=-3
    )
    axes[2].set_title("Flux intensity |q|")
    cbar2 = fig.colorbar(pcm2, ax=axes[2], shrink=0.9, pad=0.02)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        for spine in ax.spines.values():
            spine.set_visible(False)

    for cbar in (cbar0, cbar1, cbar2):
        cbar.outline.set_visible(False)

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.88, wspace=0.12)
    fig.savefig("polar_flow.png", dpi=800, bbox_inches="tight", pad_inches=0.02)

    for ax in axes:
        ax.set_aspect("equal")

    plt.show()

"""
Finite-Difference Reynolds Fluid Flow Solver in Polar Coordinates

Author: Vladislav A. Yastrebov (CNRS, Mines Paris - PSL)
AI: Cursor, Claude, ChatGPT (polar extension by ChatGPT)
Date: Nov 2025
License: BSD 3-Clause
"""

import logging
import time
from typing import Optional, Tuple

import numpy as np
from numba import jit, njit
from scipy.sparse import coo_matrix, save_npz
from skimage.measure import label

logger = logging.getLogger(__name__)


def setup_logging(level=logging.INFO):
    handler = logging.StreamHandler()
    formatter = logging.Formatter('/FS: %(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)


def set_verbosity(level: str = 'info'):
    levels = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL,
    }
    logger.setLevel(levels.get(level.lower(), logging.INFO))


THETA_BC_PERIODIC = 0
THETA_BC_SYMMETRY = 1


@njit
def face_k(a: float, b: float) -> float:
    """Harmonic mean of face conductivity; zero if either side blocked."""
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return 2.0 * (a ** 3) * (b ** 3) / (a ** 3 + b ** 3)


@njit
def _build_matrix_elements_polar(
    n_r: int,
    n_theta: int,
    gaps: np.ndarray,
    r_inner: float,
    dr: float,
    dtheta: float,
    p_inner: float,
    p_outer: float,
    theta_bc_code: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Numba-accelerated assembly of diffusion matrix in polar coordinates."""
    N = n_r * n_theta
    row_indices = np.empty(5 * N, dtype=np.int32)
    col_indices = np.empty(5 * N, dtype=np.int32)
    data = np.empty(5 * N, dtype=np.float64)
    b = np.zeros(N, dtype=np.float64)

    dtheta_sq = dtheta * dtheta if n_theta > 1 else 1.0

    nnz = 0
    for i in range(n_r):
        r_i = r_inner + i * dr
        for j in range(n_theta):
            idx = i * n_theta + j
            if gaps[i, j] <= 0.0:
                row_indices[nnz] = idx
                col_indices[nnz] = idx
                data[nnz] = 1.0
                b[idx] = 0.0
                nnz += 1
                continue

            if i == 0:
                row_indices[nnz] = idx
                col_indices[nnz] = idx
                data[nnz] = 1.0
                b[idx] = p_inner
                nnz += 1
                continue
            if i == n_r - 1:
                row_indices[nnz] = idx
                col_indices[nnz] = idx
                data[nnz] = 1.0
                b[idx] = p_outer
                nnz += 1
                continue

            diag = 0.0

            # Radial neighbors
            r_ip_half = r_i + 0.5 * dr
            r_im_half = r_i - 0.5 * dr

            g_plus = face_k(gaps[i, j], gaps[i + 1, j])
            if g_plus > 0.0:
                coeff = g_plus * r_ip_half / (r_i * dr * dr)
                if i + 1 == n_r - 1:
                    diag += coeff
                    b[idx] += coeff * p_outer
                else:
                    row_indices[nnz] = idx
                    col_indices[nnz] = (i + 1) * n_theta + j
                    data[nnz] = -coeff
                    nnz += 1
                    diag += coeff

            g_minus = face_k(gaps[i, j], gaps[i - 1, j])
            if g_minus > 0.0:
                coeff = g_minus * r_im_half / (r_i * dr * dr)
                if i - 1 == 0:
                    diag += coeff
                    b[idx] += coeff * p_inner
                else:
                    row_indices[nnz] = idx
                    col_indices[nnz] = (i - 1) * n_theta + j
                    data[nnz] = -coeff
                    nnz += 1
                    diag += coeff

            # Angular neighbors
            if n_theta > 1 and dtheta > 0.0:
                if theta_bc_code == THETA_BC_PERIODIC:
                    j_plus = (j + 1) % n_theta
                    j_minus = (j - 1 + n_theta) % n_theta

                    g_theta_plus = face_k(gaps[i, j], gaps[i, j_plus])
                    if g_theta_plus > 0.0:
                        coeff = g_theta_plus / (r_i * r_i * dtheta_sq)
                        row_indices[nnz] = idx
                        col_indices[nnz] = i * n_theta + j_plus
                        data[nnz] = -coeff
                        nnz += 1
                        diag += coeff

                    g_theta_minus = face_k(gaps[i, j], gaps[i, j_minus])
                    if g_theta_minus > 0.0:
                        coeff = g_theta_minus / (r_i * r_i * dtheta_sq)
                        row_indices[nnz] = idx
                        col_indices[nnz] = i * n_theta + j_minus
                        data[nnz] = -coeff
                        nnz += 1
                        diag += coeff
                else:
                    if j + 1 < n_theta:
                        g_theta_plus = face_k(gaps[i, j], gaps[i, j + 1])
                        if g_theta_plus > 0.0:
                            coeff = g_theta_plus / (r_i * r_i * dtheta_sq)
                            row_indices[nnz] = idx
                            col_indices[nnz] = i * n_theta + (j + 1)
                            data[nnz] = -coeff
                            nnz += 1
                            diag += coeff
                    if j - 1 >= 0:
                        g_theta_minus = face_k(gaps[i, j], gaps[i, j - 1])
                        if g_theta_minus > 0.0:
                            coeff = g_theta_minus / (r_i * r_i * dtheta_sq)
                            row_indices[nnz] = idx
                            col_indices[nnz] = i * n_theta + (j - 1)
                            data[nnz] = -coeff
                            nnz += 1
                            diag += coeff

            row_indices[nnz] = idx
            col_indices[nnz] = idx
            data[nnz] = diag
            nnz += 1

    row_indices = row_indices[:nnz]
    col_indices = col_indices[:nnz]
    data = data[:nnz]

    return row_indices, col_indices, data, b


def create_diffusion_matrix_polar(
    gaps: np.ndarray,
    r_inner: float,
    r_outer: float,
    theta_extent: float,
    theta_bc_code: int,
    p_inner: float,
    p_outer: float,
) -> Tuple[coo_matrix, np.ndarray, float, float]:
    """Create sparse matrix for polar diffusion problem."""
    n_r, n_theta = gaps.shape

    if n_r < 2:
        raise ValueError("Need at least two radial nodes.")
    if r_outer <= r_inner:
        raise ValueError("Outer radius must exceed inner radius.")

    dr = (r_outer - r_inner) / (n_r - 1)
    if n_theta <= 1:
        dtheta = 1.0
    elif theta_bc_code == THETA_BC_PERIODIC:
        dtheta = theta_extent / n_theta
    else:
        dtheta = theta_extent / (n_theta - 1)

    row, col, data, b = _build_matrix_elements_polar(
        n_r,
        n_theta,
        gaps,
        r_inner,
        dr,
        dtheta,
        p_inner,
        p_outer,
        theta_bc_code,
    )

    N = n_r * n_theta
    A = coo_matrix((data, (row, col)), shape=(N, N), dtype=np.float64)
    return A, b, dr, dtheta


@jit(nopython=True)
def _calculate_gradients_polar(
    n_r: int,
    n_theta: int,
    p: np.ndarray,
    dr: float,
    dtheta: float,
    r_inner: float,
    theta_bc_code: int,
    p_inner: float,
    p_outer: float,
    gaps: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    dpdr = np.zeros((n_r, n_theta))
    dpdtheta = np.zeros((n_r, n_theta))

    for i in range(n_r):
        r_i = r_inner + i * dr
        for j in range(n_theta):
            if gaps[i, j] <= 0.0:
                continue

            if i == 0:
                if n_r > 1:
                    dpdr[i, j] = (p[1, j] - p_inner) / dr
                else:
                    dpdr[i, j] = 0.0
            elif i == n_r - 1:
                dpdr[i, j] = (p_outer - p[i - 1, j]) / dr
            else:
                dpdr[i, j] = (p[i + 1, j] - p[i - 1, j]) / (2.0 * dr)

            if n_theta > 1 and dtheta > 0.0:
                if theta_bc_code == THETA_BC_PERIODIC:
                    j_plus = (j + 1) % n_theta
                    j_minus = (j - 1 + n_theta) % n_theta
                    dpdtheta[i, j] = (p[i, j_plus] - p[i, j_minus]) / (2.0 * dtheta)
                else:
                    if j == 0 or j == n_theta - 1:
                        dpdtheta[i, j] = 0.0
                    else:
                        dpdtheta[i, j] = (p[i, j + 1] - p[i, j - 1]) / (2.0 * dtheta)

    return dpdr, dpdtheta


@jit(nopython=True)
def _calculate_flux_polar(
    n_r: int,
    n_theta: int,
    r_inner: float,
    dr: float,
    gaps: np.ndarray,
    dpdr: np.ndarray,
    dpdtheta: np.ndarray,
) -> np.ndarray:
    flux = np.empty((n_r, n_theta, 2), dtype=np.float64)

    for i in range(n_r):
        r_i = r_inner + i * dr
        for j in range(n_theta):
            conductivity = gaps[i, j] ** 3
            if conductivity <= 0.0:
                flux[i, j, 0] = np.nan
                flux[i, j, 1] = np.nan
            else:
                flux[i, j, 0] = -conductivity * dpdr[i, j]
                if r_i > 0.0:
                    flux[i, j, 1] = -conductivity * dpdtheta[i, j] / r_i
                else:
                    flux[i, j, 1] = 0.0

    return flux


def solve_diffusion_polar(
    gaps: np.ndarray,
    r_inner: float,
    r_outer: float,
    theta_extent: float,
    theta_bc_code: int,
    solver: str = "pardiso",
    rtol: Optional[float] = None,
    p_inner: float = 1.0,
    p_outer: float = 0.0,
    save_matrix: bool = False,
    save_matrix_type: str = "coo",
) -> Tuple[np.ndarray, float, float]:
    """Solve the diffusion problem in polar coordinates."""
    mean_gap = np.mean(gaps)
    if mean_gap <= 0.0:
        raise ValueError("Invalid gap field.")

    A, b, dr, dtheta = create_diffusion_matrix_polar(
        gaps, r_inner, r_outer, theta_extent, theta_bc_code, p_inner, p_outer
    )

    if save_matrix:
        logger.info(f"Saving transport matrix and RHS to npz files in {save_matrix_type} format.")
        if save_matrix_type == "coo":
            save_npz("transport_matrix_polar.npz", A, compressed=True)
        elif save_matrix_type == "csr":
            save_npz("transport_matrix_polar.npz", A.tocsr(), compressed=True)
        elif save_matrix_type == "csc":
            save_npz("transport_matrix_polar.npz", A.tocsc(), compressed=True)
        else:
            logger.warning(f"Unknown save_matrix_type: {save_matrix_type}, defaulting to 'coo'.")
            save_npz("transport_matrix_polar.npz", A, compressed=True)
        np.savez_compressed("transport_rhs_polar.npz", b=b)

    solver_options = [
        "none",
        "auto",
        "cholesky",
        "pardiso",
        "scipy-spsolve",
        "scipy",
        "petsc-cg",
        "petsc-mumps",
    ]

    if '.' in solver:
        solver_name, preconditioner = solver.split('.')
    else:
        solver_name = solver
        preconditioner = None

    if solver_name not in solver_options:
        logger.warning(
            f"Unknown solver: {solver_name}, using 'petsc-cg' with 'hypre' preconditioner instead."
        )
        solver_name = "petsc-cg"
        preconditioner = "hypre"

    if solver_name in {"none", "auto"}:
        solver_name = "petsc-cg"
        if preconditioner is None:
            preconditioner = "hypre"
        logger.info("Auto-selecting 'petsc-cg' solver with 'hypre' preconditioner.")

    if solver_name in {"scipy", "petsc-cg"}:
        if rtol is not None:
            logger.info(f"Using user-defined relative tolerance for iterative solver: {rtol:.3e}")
        else:
            max_gap_cubed = np.max(gaps ** 3)
            if max_gap_cubed > 0:
                rtol = min(1e-10, max_gap_cubed * 1e-12)
                logger.info(f"Setting relative tolerance for iterative solver to {rtol:.3e} based on max gap.")
            else:
                raise ValueError("Invalid gap field for tolerance calculation.")

    if solver_name == "cholesky":
        logger.info("Using CHOLMOD solver from scikit-sparse.")
        from sksparse.cholmod import cholesky

        A_csc = A.tocsc()
        factor = cholesky(A_csc)
        solution = factor.solve_A(b)
    elif solver_name == "pardiso":
        logger.info("Using PARDISO solver from Intel oneAPI MKL.")
        import pypardiso

        A_csr = A.tocsr()
        pardiso_solver = pypardiso.PyPardisoSolver()
        pardiso_solver.set_iparm(1, 1)
        pardiso_solver.set_iparm(24, 1)
        pardiso_solver.set_matrix_type(1)
        solution = pardiso_solver.solve(A_csr, b)
    elif solver_name == "scipy":
        logger.info("Using SciPy Conjugate Gradient iterative solver.")
        from scipy.sparse.linalg import cg

        A_csr = A.tocsr()
        _preconditioner = preconditioner
        if preconditioner == "amg-sa":
            _preconditioner = "amg-smooth_aggregation"
        elif preconditioner == "amg-rs":
            _preconditioner = "amg-rs"
        elif preconditioner is None:
            _preconditioner = "amg-rs"

        try:
            M = get_preconditioner(A_csr, method=_preconditioner)
        except Exception as exc:
            logger.error("Failed to create AMG preconditioner. Error: %s", str(exc))
            raise RuntimeError("Failed to create AMG preconditioner.") from exc

        solution, info = cg(A_csr, b, M=M, rtol=rtol, maxiter=6000)

        if info > 0:
            logger.warning(f"Convergence to tolerance not achieved in {info} iterations.")
        elif info < 0:
            logger.error(f"Illegal input or breakdown: {info}.")
    elif solver_name == "petsc-cg":
        from petsc4py import PETSc

        A_csr = A.tocsr()
        petsc_mat = PETSc.Mat().createAIJ(size=A_csr.shape, csr=(A_csr.indptr, A_csr.indices, A_csr.data))
        ksp = PETSc.KSP().create()
        ksp.setOperators(petsc_mat)
        ksp.setType('gmres')
        try:
            ksp.setGMRESRestart(200)
        except AttributeError:
            pass

        pc = ksp.getPC()
        if preconditioner == "gamg":
            pc.setType('gamg')
        elif preconditioner == "hypre":
            pc.setType('hypre')
            try:
                pc.setHYPREType('boomeramg')
            except Exception:
                logger.debug("Could not set HYPRE type to boomeramg; continuing with default.")
        elif preconditioner == "ilu":
            pc.setType('ilu')
        else:
            logger.info(f"Unknown preconditioner {preconditioner}. Using default ILU preconditioner.")
            pc.setType('ilu')

        if rtol is not None:
            ksp.setTolerances(rtol=rtol)
        ksp.setFromOptions()

        b_vec = PETSc.Vec().createWithArray(b)
        x_vec = b_vec.duplicate()
        ksp.solve(b_vec, x_vec)

        reason = ksp.getConvergedReason()
        if reason <= 0:
            logger.warning(
                "PETSc solver failed to converge (reason=%d, residual=%.3e). Falling back to SciPy spsolve.",
                reason,
                ksp.getResidualNorm(),
            )
            from scipy.sparse.linalg import spsolve

            solution = spsolve(A_csr.tocsc(), b)
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "PETSc converged in %d iterations with residual %.3e",
                    ksp.getIterationNumber(),
                    ksp.getResidualNorm(),
                )
            solution = x_vec.getArray().copy()
    elif solver_name == "petsc-mumps":
        logger.info("Using PETSc MUMPS direct solver.")
        from petsc4py import PETSc

        A_csr = A.tocsr()
        petsc_mat = PETSc.Mat().createAIJ(size=A.shape, csr=(A_csr.indptr, A_csr.indices, A_csr.data))
        b_vec = PETSc.Vec().createWithArray(b)
        x_vec = b_vec.duplicate()

        ksp = PETSc.KSP().create()
        ksp.setOperators(petsc_mat)
        ksp.setType('preonly')
        pc = ksp.getPC()
        pc.setType('lu')
        pc.setFactorSolverType('mumps')
        ksp.setFromOptions()
        ksp.solve(b_vec, x_vec)
        solution = x_vec.getArray()
    else:
        raise ValueError(f"Solver '{solver_name}' is not supported in polar solver.")

    return solution.reshape(gaps.shape), dr, dtheta


def connectivity_analysis_polar(gaps: np.ndarray, theta_bc_code: int) -> Optional[np.ndarray]:
    """Detect percolation between inner and outer radius."""
    binary = gaps > 0
    n_r, n_theta = gaps.shape
    labels = label(binary, connectivity=1)

    if theta_bc_code == THETA_BC_PERIODIC and n_theta > 1:
        left_boundary = labels[:, 0]
        right_boundary = labels[:, -1]
        for i in range(n_r):
            left_label = left_boundary[i]
            right_label = right_boundary[i]
            if left_label > 0 and right_label > 0 and left_label != right_label:
                labels[labels == right_label] = left_label

    inner_labels = set(labels[0, :]) - {0}
    outer_labels = set(labels[-1, :]) - {0}

    percolating_labels = inner_labels & outer_labels

    if percolating_labels:
        selected_label = next(iter(percolating_labels))
        logger.info(f"Percolation detected with label {selected_label}.")
        return gaps * (labels == selected_label)

    logger.info("No percolation detected between inner and outer boundaries.")
    return None


@jit(nopython=True)
def _dilate_gaps_polar(gaps: np.ndarray, iterations: int, theta_bc_code: int) -> np.ndarray:
    n_r, n_theta = gaps.shape
    dilated = gaps.copy()

    for _ in range(iterations):
        temp = dilated.copy()
        for i in range(n_r):
            for j in range(n_theta):
                if dilated[i, j] == 0.0:
                    max_neighbor = 0.0
                    if i > 0 and dilated[i - 1, j] > max_neighbor:
                        max_neighbor = dilated[i - 1, j]
                    if i < n_r - 1 and dilated[i + 1, j] > max_neighbor:
                        max_neighbor = dilated[i + 1, j]

                    if n_theta > 1:
                        if theta_bc_code == THETA_BC_PERIODIC:
                            j_minus = (j - 1 + n_theta) % n_theta
                            j_plus = (j + 1) % n_theta
                            if dilated[i, j_minus] > max_neighbor:
                                max_neighbor = dilated[i, j_minus]
                            if dilated[i, j_plus] > max_neighbor:
                                max_neighbor = dilated[i, j_plus]
                        else:
                            if j > 0 and dilated[i, j - 1] > max_neighbor:
                                max_neighbor = dilated[i, j - 1]
                            if j < n_theta - 1 and dilated[i, j + 1] > max_neighbor:
                                max_neighbor = dilated[i, j + 1]

                    if max_neighbor > 0.0:
                        temp[i, j] = max_neighbor
        dilated = temp

    return dilated


def solve_fluid_problem_polar(
    gaps: np.ndarray,
    r_inner: float,
    r_outer: float,
    solver: str = "auto",
    rtol: Optional[float] = None,
    theta_extent: float = 2.0 * np.pi,
    theta_bc: str = "auto",
    p_inner: float = 1.0,
    p_outer: float = 0.0,
    dilation_iterations: int = 1,
    save_matrix: bool = False,
    save_matrix_type: str = "coo",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Solve the Reynolds equation on a polar grid defined between r_inner and r_outer.

    Parameters
    ----------
    gaps : np.ndarray
        Gap field array of shape (n_r, n_theta).
    r_inner : float
        Inner radius (> 0).
    r_outer : float
        Outer radius (> r_inner).
    solver : str, optional
        Solver specification (see `solve_diffusion_polar`).
    rtol : float, optional
        Relative tolerance for iterative solvers.
    theta_extent : float, optional
        Angular extent of the computational domain in radians. Default is full circle (2π).
    theta_bc : str, optional
        Angular boundary condition: 'auto', 'periodic', or 'symmetry'. 'auto' selects
        'periodic' when theta_extent ~ 2π, otherwise 'symmetry'.
    p_inner : float, optional
        Pressure at the inner radius.
    p_outer : float, optional
        Pressure at the outer radius.
    dilation_iterations : int, optional
        Number of dilation iterations applied before solving (default 1).
    save_matrix : bool, optional
        If True, store matrix and RHS to disk for debugging.
    save_matrix_type : str, optional
        Storage type for matrix saving ('coo', 'csr', 'csc').

    Returns
    -------
    (gaps_filtered, pressure, flux, dr, dtheta)
    """
    logger.info("Starting polar fluid solver.")

    if gaps.ndim != 2:
        raise ValueError("Gap field must be a 2D array (n_r, n_theta).")

    n_r, n_theta = gaps.shape
    if n_r < 2:
        raise ValueError("Polar solver requires at least two radial nodes.")
    if r_inner <= 0.0:
        raise ValueError("Inner radius must be positive for polar solver.")
    if r_outer <= r_inner:
        raise ValueError("Outer radius must exceed inner radius.")
    if theta_extent <= 0.0:
        raise ValueError("Angular extent must be positive.")

    if theta_bc == "auto":
        theta_bc = "periodic" if np.isclose(theta_extent, 2.0 * np.pi) else "symmetry"

    theta_bc_lower = theta_bc.lower()
    if theta_bc_lower not in {"periodic", "symmetry"}:
        raise ValueError("theta_bc must be 'auto', 'periodic', or 'symmetry'.")
    theta_bc_code = THETA_BC_PERIODIC if theta_bc_lower == "periodic" else THETA_BC_SYMMETRY

    logger.info("Checking connectivity between inner and outer radii.")
    start = time.time()
    gaps_filtered = connectivity_analysis_polar(gaps, theta_bc_code)
    logger.info("Connectivity analysis: CPU time = %.3f sec", time.time() - start)

    if gaps_filtered is None:
        logger.warning("No percolating channel detected. Returning None results.")
        return None, None, None, None, None

    if dilation_iterations > 0:
        logger.info("Applying dilation (%d iterations) to preserve boundary channels.", dilation_iterations)
        gaps_dilated = _dilate_gaps_polar(gaps_filtered, dilation_iterations, theta_bc_code)
    else:
        gaps_dilated = gaps_filtered

    logger.info("Solving diffusion problem in polar coordinates.")
    try:
        start_time = time.time()
        pressure, dr, dtheta = solve_diffusion_polar(
            gaps_dilated,
            r_inner,
            r_outer,
            theta_extent,
            theta_bc_code,
            solver=solver,
            rtol=rtol,
            p_inner=p_inner,
            p_outer=p_outer,
            save_matrix=save_matrix,
            save_matrix_type=save_matrix_type,
        )
        logger.info("Fluid solver: CPU time = %.3f sec", time.time() - start_time)
    except Exception as exc:
        logger.error("Error in polar fluid solver: %s", str(exc))
        return None, None, None, None, None

    # Explicitly enforce Dirichlet boundary values (useful after iterative solves)
    if n_theta > 0:
        inner_mask = gaps_dilated[0, :] > 0.0
        outer_mask = gaps_dilated[-1, :] > 0.0
        if np.any(inner_mask):
            pressure[0, inner_mask] = p_inner
        if np.any(outer_mask):
            pressure[-1, outer_mask] = p_outer

    logger.info("Calculating gradients and flux.")
    dpdr, dpdtheta = _calculate_gradients_polar(
        n_r,
        n_theta,
        pressure,
        dr,
        dtheta,
        r_inner,
        theta_bc_code,
        p_inner,
        p_outer,
        gaps_dilated,
    )

    flux = _calculate_flux_polar(n_r, n_theta, r_inner, dr, gaps_dilated, dpdr, dpdtheta)

    # Mask flux outside the original percolating channel
    channel_mask = gaps_filtered > 0.0
    flux[~channel_mask] = np.nan

    logger.info("Polar fluid solver finished.")

    return gaps_filtered, pressure, flux, dr, dtheta


def get_preconditioner(A, method="amg-rs"):
    if method == "amg-smooth_aggregation":
        import pyamg
        from scipy.sparse.linalg import LinearOperator

        try:
            logger.info("Using Smoothed Aggregation AMG preconditioner.")
            ml = pyamg.smoothed_aggregation_solver(A, max_coarse=A.shape[0] // 1000)
            M = ml.aspreconditioner(cycle='V')
            return M
        except Exception:
            logger.warning("AMG smooth aggregation failed, falling back to AMG.RS.")
            method = "amg-rs"

    if method == "amg-rs":
        import pyamg
        from scipy.sparse.linalg import LinearOperator

        logger.info("Using Ruge-Stuben AMG preconditioner.")
        try:
            ml = pyamg.ruge_stuben_solver(A, max_coarse=A.shape[0] // 1000, CF='RS')
            M = LinearOperator(A.shape, matvec=lambda v: ml.solve(v, tol=1e-2, maxiter=1))
            return M
        except Exception as exc:
            logger.warning("AMG.RS failed (%s), cannot construct preconditioner.", str(exc))
            raise RuntimeError("AMG.RS failed")

    raise ValueError(f"Unknown preconditioner method: {method}")


def compute_total_flux_polar(
    filtered_gaps: np.ndarray,
    flux: np.ndarray,
    r_inner: float,
    r_outer: float,
    dtheta: float,
) -> Tuple[float, float]:
    """
    Compute total flux and conservation error across inner and outer boundaries.
    Returns Q_total and flux_conservation_error.
    """
    if filtered_gaps is None or flux is None:
        raise ValueError("Flux computation requires valid gaps and flux arrays.")

    n_r, n_theta = filtered_gaps.shape
    if n_r < 2:
        raise ValueError("Flux computation requires at least two radial nodes.")

    flux_inner = 0.0
    active_inner = 0
    for j in range(n_theta):
        if not np.isnan(flux[0, j, 0]) and filtered_gaps[0, j] > 0:
            flux_inner += flux[0, j, 0] * r_inner * dtheta
            active_inner += 1

    flux_outer = 0.0
    active_outer = 0
    for j in range(n_theta):
        if not np.isnan(flux[-1, j, 0]) and filtered_gaps[-1, j] > 0:
            flux_outer += flux[-1, j, 0] * r_outer * dtheta
            active_outer += 1

    Q_total = 0.5 * (abs(flux_inner) + abs(flux_outer))
    flux_conservation_error = abs(flux_inner - flux_outer) / max(Q_total, 1e-15)

    logger.info("> Flux computation (polar) <")
    logger.info(
        "Inner flux (r = %.3e):     %.6e [Active cells: %d]", r_inner, flux_inner, active_inner
    )
    logger.info(
        "Outer flux (r = %.3e):     %.6e [Active cells: %d]", r_outer, flux_outer, active_outer
    )
    logger.info("Total average flux (Q_total): %.6e", Q_total)
    logger.info(
        "Conservation error:   %.2e (%.2f%%)",
        flux_conservation_error,
        flux_conservation_error * 100.0,
    )

    return Q_total, flux_conservation_error


def _warmup_numba_functions_polar():
    """Warm up Numba JIT compilation with minimal test cases."""
    logger.info("Warming up polar Numba JIT compilation...")
    n_r_test = 3
    n_theta_test = 4
    g_test = np.ones((n_r_test, n_theta_test), dtype=np.float64) * 0.1
    dr_test = 0.1
    dtheta_test = 2.0 * np.pi / n_theta_test
    try:
        _build_matrix_elements_polar(
            n_r_test,
            n_theta_test,
            g_test,
            1.0,
            dr_test,
            dtheta_test,
            1.0,
            0.0,
            THETA_BC_PERIODIC,
        )
    except Exception:
        pass

    try:
        _calculate_gradients_polar(
            n_r_test,
            n_theta_test,
            np.ones((n_r_test, n_theta_test), dtype=np.float64),
            dr_test,
            dtheta_test,
            1.0,
            THETA_BC_PERIODIC,
            1.0,
            0.0,
            g_test,
        )
    except Exception:
        pass

    try:
        dpdr_test = np.ones((n_r_test, n_theta_test), dtype=np.float64)
        dpdtheta_test = np.ones((n_r_test, n_theta_test), dtype=np.float64)
        _calculate_flux_polar(
            n_r_test,
            n_theta_test,
            1.0,
            dr_test,
            g_test,
            dpdr_test,
            dpdtheta_test,
        )
    except Exception:
        pass

    try:
        _dilate_gaps_polar(g_test, 1, THETA_BC_PERIODIC)
    except Exception:
        pass

    logger.info("Polar Numba JIT compilation warmed up.")


_warmup_numba_functions_polar()



"""
Finite-Difference Reynolds Fluid Flow Solver

Author: Vladislav A. Yastrebov (CNRS, Mines Paris - PSL)
AI: Cursor, Claude, ChatGPT
Date: Aug 2024-Sept 2025
License: BSD 3-Clause
"""
import numpy as np
from scipy.ndimage import label
from scipy.sparse import lil_matrix, coo_matrix
from numba import jit, prange
## For iterative solver if needed
from pyamg import smoothed_aggregation_solver
from scipy.sparse.linalg import spilu, LinearOperator, cg
# Eventually
#  from scipy.sparse.linalg import gmres, bicgstab, spsolve

from pypardiso import spsolve as pypardiso_spsolve

import time
import logging
import gc

logger = logging.getLogger(__name__)

def setup_logging(level=logging.INFO):
    handler = logging.StreamHandler()
    formatter = logging.Formatter('/FS: %(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)

def set_verbosity(level='info'):
    levels = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL,
    }
    logger.setLevel(levels.get(level.lower(), logging.INFO))

@jit(nopython=True)
def _build_matrix_elements(n, g, penalty):
    """Numba-accelerated matrix element calculation with external reservoir BCs"""
    dx = 1.0 / (n - 1)
    N = n * n
    
    # Pre-allocate arrays for matrix construction
    row_indices = []
    col_indices = []
    data = []
    b = np.zeros(N)
    
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            
            if g[i,j] == 0:  # Blocked cells
                row_indices.append(idx)
                col_indices.append(idx)
                data.append(1.0)
                b[idx] = 0
            else:  # Open cells (including boundaries)
                if i == 0:  # Left boundary - external reservoir at p=0
                    # Calculate conductivities
                    g_e = 0.5 * (g[i,j]**3 + g[i+1,j]**3) if i+1 < n and g[i+1,j] > 0 else 0
                    g_w = g[i,j]**3  # Connection to external reservoir at p=0
                    g_n = 0.5 * (g[i,j]**3 + g[i,(j+1)%n]**3) if g[i,(j+1)%n] > 0 else 0
                    g_s = 0.5 * (g[i,j]**3 + g[i,(j-1)%n]**3) if g[i,(j-1)%n] > 0 else 0
                    
                    # Diagonal element
                    row_indices.append(idx)
                    col_indices.append(idx)
                    data.append(-(g_e + g_w + g_n + g_s) / dx)
                    
                    # Off-diagonal elements
                    if g_e > 0:
                        row_indices.append(idx)
                        col_indices.append((i+1) * n + j)
                        data.append(g_e / dx)
                    if g_n > 0:
                        row_indices.append(idx)
                        col_indices.append(i * n + (j+1)%n)
                        data.append(g_n / dx)
                    if g_s > 0:
                        row_indices.append(idx)
                        col_indices.append(i * n + (j-1)%n)
                        data.append(g_s / dx)
                    
                    # RHS contribution from external reservoir (p=0)
                    b[idx] = -g_w / dx * 0.0  # = 0, but keeping for clarity
                    
                elif i == n-1:  # Right boundary - external reservoir at p=1
                    # Calculate conductivities
                    g_e = g[i,j]**3  # Connection to external reservoir at p=1
                    g_w = 0.5 * (g[i,j]**3 + g[i-1,j]**3) if i-1 >= 0 and g[i-1,j] > 0 else 0
                    g_n = 0.5 * (g[i,j]**3 + g[i,(j+1)%n]**3) if g[i,(j+1)%n] > 0 else 0
                    g_s = 0.5 * (g[i,j]**3 + g[i,(j-1)%n]**3) if g[i,(j-1)%n] > 0 else 0
                    
                    # Diagonal element
                    row_indices.append(idx)
                    col_indices.append(idx)
                    data.append(-(g_e + g_w + g_n + g_s) / dx)
                    
                    # Off-diagonal elements
                    if g_w > 0:
                        row_indices.append(idx)
                        col_indices.append((i-1) * n + j)
                        data.append(g_w / dx)
                    if g_n > 0:
                        row_indices.append(idx)
                        col_indices.append(i * n + (j+1)%n)
                        data.append(g_n / dx)
                    if g_s > 0:
                        row_indices.append(idx)
                        col_indices.append(i * n + (j-1)%n)
                        data.append(g_s / dx)
                    
                    # RHS contribution from external reservoir (p=1)
                    b[idx] = -g_e / dx * 1.0
                    
                else:  # Interior points
                    g_e = 0.5 * (g[i,j]**3 + g[i+1,j]**3)
                    g_w = 0.5 * (g[i,j]**3 + g[i-1,j]**3)
                    g_n = 0.5 * (g[i,j]**3 + g[i,(j+1)%n]**3)
                    g_s = 0.5 * (g[i,j]**3 + g[i,(j-1)%n]**3)
                    
                    if g[i+1,j] == 0:
                        g_e = 0
                    if g[i-1,j] == 0:
                        g_w = 0
                    if g[i,(j+1)%n] == 0:
                        g_n = 0
                    if g[i,(j-1)%n] == 0:
                        g_s = 0
                        
                    # Diagonal element
                    row_indices.append(idx)
                    col_indices.append(idx)
                    data.append(-(g_e + g_w + g_n + g_s) / dx)
                    
                    # Off-diagonal elements
                    if g[i+1,j] > 0:
                        row_indices.append(idx)
                        col_indices.append((i+1) * n + j)
                        data.append(g_e / dx)
                    if g[i-1,j] > 0:
                        row_indices.append(idx)
                        col_indices.append((i-1) * n + j)
                        data.append(g_w / dx)
                    if g[i,(j+1)%n] > 0:
                        row_indices.append(idx)
                        col_indices.append(i * n + (j+1)%n)
                        data.append(g_n / dx)
                    if g[i,(j-1)%n] > 0:
                        row_indices.append(idx)
                        col_indices.append(i * n + (j-1)%n)
                        data.append(g_s / dx)
    
    return np.array(row_indices), np.array(col_indices), np.array(data), b

"""
Create the sparse matrix for the diffusion problem with non-homogeneous gap field,
properly handling zero or near-zero gap regions.
"""
def create_diffusion_matrix(n, g, penalty=None):
    """Create matrix using Numba-accelerated element calculation"""
    
    # Use numba-accelerated function for heavy computation
    row_indices, col_indices, data, b = _build_matrix_elements(n, g, 0)  # penalty not used anymore
    
    # Create sparse matrix
    N = n * n
    A = coo_matrix((data, (row_indices, col_indices)), shape=(N, N), dtype=np.float64)
    A = A.tolil()  # Convert to lil for any remaining operations
    
    return A, b


@jit(nopython=True)
def _calculate_gradients_simple_bc(n, p, dx, gaps):
    """Simple gradient calculation with proper boundary conditions"""
    dpdx = np.zeros((n, n))
    dpdy = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if gaps[i, j] > 0:
                # X-direction gradient
                if i == 0:  # Left boundary: p = 0
                    # Use forward difference: dp/dx = (p[1] - p[0])/dx
                    # But we know p[0] should be 0, so use actual BC
                    dpdx[i, j] = (p[i, j] - 0.0) / dx
                elif i == n-1:  # Right boundary: p = 1
                    # Use backward difference: dp/dx = (p[n-1] - p[n-2])/dx
                    # But we know p[n-1] should be 1, so use actual BC
                    dpdx[i, j] = (1.0 - p[i, j]) / dx
                else:  # Interior: central difference
                    dpdx[i, j] = (p[i+1, j] - p[i-1, j]) / (2.0 * dx)
                
                # Y-direction gradient (periodic)
                j_plus = (j + 1) % n
                j_minus = (j - 1) % n
                dpdy[i, j] = (p[i, j_plus] - p[i, j_minus]) / (2.0 * dx)
            
    return dpdx, dpdy

@jit(nopython=True)
def _dilate_gaps_numba(gaps, iterations=1):
    """Numba-accelerated morphological dilation of gap field"""
    n = gaps.shape[0]
    dilated = gaps.copy()
    
    for _ in range(iterations):
        temp = dilated.copy()
        for i in range(n):
            for j in range(n):
                if dilated[i, j] == 0:  # Only dilate into zero regions
                    # Check 4-connected neighbors
                    max_neighbor = 0.0
                    
                    # Check left neighbor
                    if i > 0 and dilated[i-1, j] > max_neighbor:
                        max_neighbor = dilated[i-1, j]
                    
                    # Check right neighbor
                    if i < n-1 and dilated[i+1, j] > max_neighbor:
                        max_neighbor = dilated[i+1, j]
                    
                    # Check down neighbor (periodic)
                    j_down = (j - 1) % n
                    if dilated[i, j_down] > max_neighbor:
                        max_neighbor = dilated[i, j_down]
                    
                    # Check up neighbor (periodic)
                    j_up = (j + 1) % n
                    if dilated[i, j_up] > max_neighbor:
                        max_neighbor = dilated[i, j_up]
                    
                    if max_neighbor > 0:
                        temp[i, j] = max_neighbor
        
        dilated = temp
    
    return dilated

@jit(nopython=True)
def _threshold_numba(matrix, z0):
    """Numba-accelerated threshold function"""
    result = np.zeros_like(matrix, dtype=np.int32)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > z0:
                result[i, j] = 1
    return result

def threshold(matrix, z0):
    return _threshold_numba(matrix, z0)

@jit(nopython=True)
def _calculate_gradients_with_bc_numba(n, p, dx, gaps):
    """Gradient calculation respecting Dirichlet boundary conditions"""
    dpdx = np.zeros((n, n))
    dpdy = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # X-direction gradient (horizontal)
            if i == 0:  # Left boundary: p = 0
                # Use boundary condition: p[0,j] = 0, forward difference to interior
                if gaps[i, j] > 0:
                    dpdx[i, j] = (p[i+1, j] - 0.0) / dx  # Known BC: p=0 at left
                else:
                    dpdx[i, j] = 0.0
            elif i == n-1:  # Right boundary: p = 1 (where gap > 0)
                # Use boundary condition: p[n-1,j] = 1 for open channels
                if gaps[i, j] > 0:
                    dpdx[i, j] = (1.0 - p[i-1, j]) / dx  # Known BC: p=1 at right
                else:
                    dpdx[i, j] = 0.0
            else:  # Interior points - central difference
                if gaps[i, j] > 0:
                    # Check if neighbors exist for central difference
                    if gaps[i-1, j] > 0 and gaps[i+1, j] > 0:
                        dpdx[i, j] = (p[i+1, j] - p[i-1, j]) / (2.0 * dx)
                    elif gaps[i+1, j] > 0:  # Only right neighbor
                        dpdx[i, j] = (p[i+1, j] - p[i, j]) / dx
                    elif gaps[i-1, j] > 0:  # Only left neighbor
                        dpdx[i, j] = (p[i, j] - p[i-1, j]) / dx
                    else:
                        dpdx[i, j] = 0.0
                else:
                    dpdx[i, j] = 0.0
            
            # Y-direction gradient (vertical) - periodic boundaries
            if gaps[i, j] > 0:
                j_plus = (j + 1) % n
                j_minus = (j - 1) % n
                
                # Use central difference for periodic boundaries
                if gaps[i, j_plus] > 0 and gaps[i, j_minus] > 0:
                    dpdy[i, j] = (p[i, j_plus] - p[i, j_minus]) / (2.0 * dx)
                elif gaps[i, j_plus] > 0:  # Only up neighbor
                    dpdy[i, j] = (p[i, j_plus] - p[i, j]) / dx
                elif gaps[i, j_minus] > 0:  # Only down neighbor
                    dpdy[i, j] = (p[i, j] - p[i, j_minus]) / dx
                else:
                    dpdy[i, j] = 0.0
            else:
                dpdy[i, j] = 0.0
    
    return dpdx, dpdy

@jit(nopython=True)
def _calculate_flux_numba(n, gaps, dpdx, dpdy):
    """Numba-accelerated flux calculation"""
    flux = np.zeros((n, n, 2))
    
    for i in range(n):
        for j in range(n):
            conductivity = gaps[i, j]**3
            flux[i, j, 0] = -conductivity * dpdx[i, j]
            flux[i, j, 1] = -conductivity * dpdy[i, j]
    
    return flux

@jit(nopython=True)
def _filter_flux_numba(n, gaps, flux):
    """Numba-accelerated flux filtering - conservative approach"""
    filtered_flux = flux.copy()
    
    for i in range(n):
        for j in range(n):
            # Only filter if current cell has zero gap
            if gaps[i,j] == 0:
                filtered_flux[i,j,0] = np.nan
                filtered_flux[i,j,1] = np.nan
            else:
                # For x-direction flux: only filter if both neighboring x-cells are closed
                if gaps[(i+1)%n,j] == 0 and gaps[(i-1)%n,j] == 0:
                    filtered_flux[i,j,0] = np.nan
                # For y-direction flux: only filter if both neighboring y-cells are closed  
                if gaps[i,(j+1)%n] == 0 and gaps[i,(j-1)%n] == 0:
                    filtered_flux[i,j,1] = np.nan

            # Alternative: Even more conservative - only filter the current cell if it's closed
            # if gaps[i,j] == 0:
            #     filtered_flux[i,j,0] = np.nan
            #     filtered_flux[i,j,1] = np.nan
    
    return filtered_flux

def solve_diffusion(n, g, solver="auto"):
    """Solve diffusion with external reservoir boundary conditions"""
    mean_gap = np.mean(g)
    if mean_gap <= 0:
        raise ValueError("Invalid gap field")

    A, b = create_diffusion_matrix(n, g, None)  # No penalty needed
    A = A.tocsr()  

    # Auto-select solver based on problem size
    if solver == "auto":
        if A.shape[0] < 5000:  
            solver = "direct"
        else:  # Large problems
            solver = "iterative"
        logger.info(f"Auto-selected solver: {solver}")

    if solver == "direct":
        gc.collect()      
        p = pypardiso_spsolve(A, b)
    elif solver == "iterative":
        M = None
        try:
            M = get_preconditioner(A, method="amg")
        except:
            logger.warning("AMG preconditioner failed, using ILU instead.")
            try:
                M = get_preconditioner(A, method="ilu")
            except:
                logger.warning("ILU preconditioner failed, using no preconditioner.")
                M = None

        # Calculate relative tolerance based on gap field
        max_gap_cubed = np.max(g**3)
        if max_gap_cubed > 0:
            rtol = min(1e-12, max_gap_cubed * 1e-14)
        else:
            raise ValueError("Invalid gap field for tolerance calculation")

        p, info = cg(A, b, M=M, rtol=rtol, maxiter=6000)
                
        if info > 0:
            print(f"Convergence to tolerance not achieved in {info} iterations")
        elif info < 0:
            print(f"Illegal input or breakdown: {info}")
        
    return p.reshape((n, n))

def solve_fluid_problem(gaps, solver):
    logger.info("Starting fluid solver.")
    n = gaps.shape[0]
    if n == 0:
        logger.error("Empty gap field.")
        return None, None, None

    x = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, x)

    logger.info("Checking connectivity.")
    # construct connectivity matrix - use accelerated threshold
    binary = threshold(gaps, 0)

    # Select separate regions
    labels, numL = label(binary)    

    # Make labels periodic
    for i in range(n):
        if labels[i,0] > 0:
            color = labels[i,0]
            opposite = labels[i,-1]
            if opposite > 0:
                labels[labels == opposite] = color

    # Check if there is a percolation
    left_side = labels[0,:]
    right_side = labels[-1,:]
    unique_left_side = np.unique(left_side)
    unique_right_side = np.unique(right_side)
    selected_color = -1

    for i in unique_left_side:
        if i != 0:
            for j in unique_right_side:
                if j != 0 and i == j:
                    selected_color = i
                    break
    if selected_color == -1:
        logger.info("No percolation detected.")
        return None, None, None

    # To get rid of lakes surrounded by contact (trapped fluid)
    gaps_original = gaps * (labels == selected_color)
    
    # Apply dilation before solving to preserve boundary data
    logger.info("Applying dilation to preserve boundary channels.")
    gaps_dilated = _dilate_gaps_numba(gaps_original, iterations=1)

    logger.info("Solving diffusion problem.")
    try:
        # Solve for pressure using dilated gaps
        start_time = time.time()
        p = solve_diffusion(n, gaps_dilated, solver)
        logger.info("Fluid solver: CPU time for n = {0:d}: {1:.3f} sec".format(n, time.time() - start_time))
    except Exception as e:
        logger.error("Error in fluid solver: ", e)
        return None, None, None

    logger.info("Fluid solver finished.")
    logger.info("Calculating flux with simple boundary condition approach.")

    # Calculate gradients with proper boundary conditions
    dx = 1 / (n - 1)
    dpdx, dpdy = _calculate_gradients_simple_bc(n, p, dx, gaps_original)

    # Use numba-accelerated flux calculation with ORIGINAL gaps
    flux = _calculate_flux_numba(n, gaps_original, dpdx, dpdy)
    filtered_flux = _filter_flux_numba(n, gaps_original, flux)
    
    logger.info("finished.")

    return gaps_original, p, filtered_flux

def get_preconditioner(A, method="amg"):
    if method == "amg":
        try:
            ml = smoothed_aggregation_solver(A, max_coarse=A.shape[0] // 1000)
            return ml.aspreconditioner(cycle="V")
        except:
            logger.warning("AMG failed, falling back to ILU")
            method = "ilu"
    
    if method == "ilu":
        try:
            ilu = spilu(A.tocsc(), drop_tol=1e-5)
            return LinearOperator(A.shape, ilu.solve)
        except:
            logger.warning("ILU failed, using no preconditioner")
            return None
    
    return None

# Total flux calculation
def compute_total_flux(filtered_gaps, flux, N0):
        """
        Compute total flux and conservation error by integrating over inlet and outlet boundaries.
        Returns Q_total and flux_conservation_error.
        """
        dy = 1.0 / (N0 - 1)  # Grid spacing in y-direction

        # Inlet flux (x=0, i=0): integrate flux_x over y-direction
        flux_inlet = 0.0
        active_inlet_cells = 0
        for j in range(N0):
            if not np.isnan(flux[0, j, 0]) and filtered_gaps[0, j] > 0:
                flux_inlet += flux[0, j, 0] * dy
                active_inlet_cells += 1

        # Outlet flux (x=1, i=N0-1): integrate flux_x over y-direction  
        flux_outlet = 0.0
        active_outlet_cells = 0
        for j in range(N0):
            if not np.isnan(flux[N0-1, j, 0]) and filtered_gaps[N0-1, j] > 0:
                flux_outlet += flux[N0-1, j, 0] * dy
                active_outlet_cells += 1

        # Total flux through domain 
        Q_total = 0.5 * (abs(flux_inlet) + abs(flux_outlet))
        flux_conservation_error = abs(flux_inlet - flux_outlet) / max(Q_total, 1e-15)

        logger.info("> Flux computation <")
        logger.info(f"Inlet flux (x=0):     {flux_inlet:.6e} [Active cells: {active_inlet_cells}]")
        logger.info(f"Outlet flux (x=1):    {flux_outlet:.6e} [Active cells: {active_outlet_cells}]")
        logger.info(f"Total average flux (Q_total): {Q_total:.6e}")
        logger.info(f"Conservation error:   {flux_conservation_error:.2e} ({flux_conservation_error*100:.2f}%)")

        return Q_total, flux_conservation_error


def _warmup_numba_functions():
    """Warm up Numba JIT compilation with minimal test cases"""
    logger.info("Warming up Numba JIT compilation...")
    
    # Create minimal test data
    n_test = 3  # Very small grid
    g_test = np.ones((n_test, n_test), dtype=np.float64) * 0.1
    penalty_test = 1000.0
    
    # Warm up matrix building
    try:
        _build_matrix_elements(n_test, g_test, penalty_test)
    except:
        pass  # Ignore any errors during warmup
    
    # Warm up threshold function
    try:
        _threshold_numba(g_test, 0.05)
    except:
        pass
    
    # Warm up dilation function
    try:
        _dilate_gaps_numba(g_test, iterations=1)
    except:
        pass
    
    # Warm up simple gradient calculation
    try:
        p_test = np.ones((n_test, n_test), dtype=np.float64)
        _calculate_gradients_simple_bc(n_test, p_test, 0.5, g_test)
    except:
        pass
    
    # Warm up flux calculation
    try:
        dpdx_test = np.ones((n_test, n_test), dtype=np.float64)
        dpdy_test = np.ones((n_test, n_test), dtype=np.float64)
        _calculate_flux_numba(n_test, g_test, dpdx_test, dpdy_test)
    except:
        pass
    
    # Warm up flux filtering
    try:
        flux_test = np.ones((n_test, n_test, 2), dtype=np.float64)
        _filter_flux_numba(n_test, g_test, flux_test)
    except:
        pass
    
    logger.info("Numba JIT compilation warmed up.")

# Automatically warm up when module is imported
_warmup_numba_functions()

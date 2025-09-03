import numpy as np
from scipy.ndimage import label
from scipy.sparse import lil_matrix
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
    """Numba-accelerated matrix element calculation"""
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
            
            if i == n - 1:  # Right boundary (x = 1)
                if g[i,j] != 0:
                    row_indices.append(idx)
                    col_indices.append(idx)
                    data.append(penalty)
                    b[idx] = penalty
                else:
                    row_indices.append(idx)
                    col_indices.append(idx)
                    data.append(1.0)
                    b[idx] = 0
            elif i == 0:  # Left boundary (x = 0)
                row_indices.append(idx)
                col_indices.append(idx)
                data.append(penalty)
                b[idx] = 0
            else:
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
                    
                if g[i,j] > 0:
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
                else:
                    row_indices.append(idx)
                    col_indices.append(idx)
                    data.append(1.0)
    
    return np.array(row_indices), np.array(col_indices), np.array(data), b

"""
Create the sparse matrix for the diffusion problem with non-homogeneous gap field,
properly handling zero or near-zero gap regions.
"""
def create_diffusion_matrix(n, g, penalty):
    """Create matrix using Numba-accelerated element calculation"""
    from scipy.sparse import coo_matrix
    
    # Use numba-accelerated function for heavy computation
    row_indices, col_indices, data, b = _build_matrix_elements(n, g, penalty)
    
    # Create sparse matrix
    N = n * n
    A = coo_matrix((data, (row_indices, col_indices)), shape=(N, N), dtype=np.float64)
    A = A.tolil()  # Convert to lil for any remaining operations
    
    return A, b


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
    """Numba-accelerated flux filtering"""
    filtered_flux = flux.copy()
    
    for i in range(n):
        for j in range(n):
            if (gaps[i,j] == 0 or 
                gaps[(i+1)%n,j] == 0 or 
                gaps[i,(j+1)%n] == 0 or 
                gaps[(i-1)%n,j] == 0 or 
                gaps[i,(j-1)%n] == 0):
                filtered_flux[i,j,0] = np.nan
                filtered_flux[i,j,1] = np.nan
    
    return filtered_flux

def solve_diffusion(n, g, solver="auto"):
    mean_gap = np.mean(g)
    if mean_gap > 0:
        dx = 1.0 / (n - 1)
        penalty = mean_gap**3 / dx * 1e4
        logger.info(f"Using penalty: {penalty:.3e}")
    else:
        raise ValueError("Invalid gap field")

    A, b = create_diffusion_matrix(n, g, penalty)
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
            # # AMG preconditioner
            # ml = smoothed_aggregation_solver(A, max_coarse=A.shape[0] // 1000)
            # M = ml.aspreconditioner(cycle="V")

        except:
            logger.warning("AMG preconditioner failed, using ILU instead.")
            try:
                M = get_preconditioner(A, method="ilu")
            except:
                logger.warning("ILU preconditioner failed, using no preconditioner.")
                M = None
        # # Alternative preconditioner
        # ilu = spilu(A.tocsc(), drop_tol=1e-5)
        # M = LinearOperator(A.shape, ilu.solve)

        # Calculate relative tolerance based on gap field
        max_gap_cubed = np.max(g**3)
        if max_gap_cubed > 0:
            rtol = min(1e-12, max_gap_cubed * 1e-14)
        else:
            raise ValueError("Invalid gap field for tolerance calculation")

        p, info = cg(A, b, M=M, rtol=rtol, maxiter=6000)

        # Alternative solvers
        # p, info = gmres(A, b, M=M, rtol=rtol, maxiter=6000, restart=30)
        # p, info = bicgstab(A, b, M=M, rtol=1e-22, maxiter=10000) 
                
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
    gaps = gaps * (labels == selected_color)

    logger.info("Solving diffusion problem.")
    try:
        # Solve for pressure
        start_time = time.time()
        p = solve_diffusion(n, gaps, solver)
        logger.info("Fluid solver: CPU time for n = {0:d}: {1:.3f} sec".format(n, time.time() - start_time))
    except Exception as e:
        logger.error("Error in fluid solver: ", e)
        return None, None, None

    logger.info("Fluid solver finished.")
    logger.info("Calculating flux.")

    # Calculate flux using accelerated functions
    dx = 1 / (n - 1)
    dpdx = np.gradient(p, dx, axis=1)
    dpdy = np.gradient(p, dx, axis=0)

    # Use numba-accelerated flux calculation
    flux = _calculate_flux_numba(n, gaps, dpdx, dpdy)
    filtered_flux = _filter_flux_numba(n, gaps, flux)
    
    logger.info("finished.")

    return gaps, p, filtered_flux

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

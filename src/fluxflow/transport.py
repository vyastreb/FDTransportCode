"""
Finite-Difference Reynolds Fluid Flow Solver

Author: Vladislav A. Yastrebov (CNRS, Mines Paris - PSL)
AI: Cursor, Claude, ChatGPT
Date: Aug 2024-Sept 2025
License: BSD 3-Clause
"""

# FIXME: something wrong happens over connections of 1 grid cell thickness.
# TODO: adapt for compressible fluids (requires only postprocessing)

import numpy as np
from skimage.measure import label
from scipy.sparse import coo_matrix
from numba import jit, njit
import time
import logging
import gc

# If need control over number of threads
# os.environ['MKL_NUM_THREADS'] = '4'
# os.environ['OMP_NUM_THREADS'] = '4'

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

# new version of matrix builder
@njit
def face_k(a, b):
    # harmonic mean; if either side blocked -> 0
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return 2.0 * (a**3) * (b**3) / (a**3 + b**3)

@njit   
def _build_matrix_elements(n, g, penalty=0.0):
    """Numba-accelerated matrix element calculation with external reservoir BCs"""
    dx = 1.0 / (n - 1)
    N = n * n

    # Max 5 nonzeros per row (5-diagonal matrix)
    row_indices = np.empty(5 * N, dtype=np.int32)
    col_indices = np.empty(5 * N, dtype=np.int32)
    data = np.empty(5 * N, dtype=np.float64)
    b = np.zeros(N, dtype=np.float64)  

    nnz = 0
    for i in range(n):
        for j in range(n):
            idx = i * n + j

            if g[i, j] <= 0.0:  # blocked cell -> Dirichlet p=0
                row_indices[nnz] = idx
                col_indices[nnz] = idx
                data[nnz] = 1.0
                nnz += 1
                b[idx] = 0.0
                continue

            # neighbors (periodic in j)
            # east/west not periodic in i
            ge = face_k(g[i, j], g[i+1, j]) if i+1 < n else 0.0
            gw = face_k(g[i, j], g[i-1, j]) if i-1 >= 0 else 0.0
            gn = face_k(g[i, j], g[i, (j+1) % n])
            gs = face_k(g[i, j], g[i, (j-1) % n])

            diag = 0.0

            # West boundary: external reservoir p=0 at i==0
            if i == 0:
                # couple to external as Dirichlet term
                gw = g[i, j]**3  # face to reservoir
                diag += gw / dx
                # RHS += gw/dx * p_W (p_W=0) -> no change
            else:
                if gw > 0.0:
                    row_indices[nnz] = idx
                    col_indices[nnz] = (i - 1) * n + j
                    data[nnz] = -gw / dx
                    nnz += 1
                    diag += gw / dx

            # East boundary: external reservoir p=1 at i==n-1
            if i == n - 1:
                ge = g[i, j]**3
                diag += ge / dx
                b[idx] += ge / dx * 1.0
            else:
                if ge > 0.0:
                    row_indices[nnz] = idx
                    col_indices[nnz] = (i + 1) * n + j
                    data[nnz] = -ge / dx
                    nnz += 1
                    diag += ge / dx

            # periodic north
            if gn > 0.0:
                row_indices[nnz] = idx
                col_indices[nnz] = i * n + (j + 1) % n
                data[nnz] = -gn / dx
                nnz += 1
                diag += gn / dx

            # periodic south
            if gs > 0.0:
                row_indices[nnz] = idx
                col_indices[nnz] = i * n + (j - 1) % n
                data[nnz] = -gs / dx
                nnz += 1
                diag += gs / dx

            # diagonal
            row_indices[nnz] = idx
            col_indices[nnz] = idx
            data[nnz] = diag
            nnz += 1

    # trim
    row_indices = row_indices[:nnz]
    col_indices = col_indices[:nnz]
    data = data[:nnz]
    # b already length N

    return row_indices, col_indices, data, b




def create_diffusion_matrix(n, g, penalty=None):
    """
    Create the sparse matrix for the diffusion problem with non-homogeneous gap field, properly handling zero or near-zero gap regions.
    """    
    row_indices, col_indices, data, b = _build_matrix_elements(n, g, 0)
    
    N = n * n
    from scipy.sparse import csc_matrix
    
    A = coo_matrix((data, (row_indices, col_indices)), shape=(N, N), dtype=np.float64)
   
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
    
    A, b = create_diffusion_matrix(n, g, None) 

    # ************************ #
    # Solve the linear system  #
    # ************************ #
    # Known solvers
    SOLVERS = ["none", "auto", "cholesky", "pardiso", "scipy-spsolve", "scipy", "petsc", "petsc-cg", "petsc-mumps"]

    if '.' in solver:
        solver_name, preconditioner = solver.split('.')
    else:
        solver_name = solver
        preconditioner = None

    print("solver = ", solver)
    print("solver_name = ", solver_name)
    print("preconditioner = ", preconditioner)

    if solver_name not in SOLVERS:
        logger.warning(f"Unknown solver: {solver_name}, using 'cholesky' instead.")
        solver_name = "cholesky"
    if solver_name == "none" or solver_name == "auto":
        solver_name = "cholesky"
        logger.info("Auto-selecting 'cholesky' solver.")
    
    ####################################
    #     DIRECT CHOLESKY SOLVER       #
    ####################################        
    if solver_name == "cholesky":
        logger.info("Using CHOLMOD solver from scikit-sparse.")
        A = A.tocsc()
        from sksparse.cholmod import cholesky
        factor = cholesky(A)
        p = factor.solve_A(b)
    ####################################
    #    DIRECT MKL PARDISO SOLVER     #
    ####################################        
    elif solver_name == "pardiso": # Intel oneAPI Math Kernel Library PARDISO solver
        logger.info("Using PARDISO solver from Intel oneAPI MKL.")
        A = A.tocsr()
        import pypardiso
        pardiso_solver = pypardiso.PyPardisoSolver()
        # Fastest configuration, see https://www.smcm.iqfr.csic.es/docs/intel/mkl/mkl_manual/ssr/ssr_pardiso_parameters.htm
        pardiso_solver.set_iparm(1, 1)
        # To reduce memory usage 35% less, but 20% slower
        pardiso_solver.set_iparm(24, 1)  # Use less memory in parallel solve
        # To further reduce memory usage by 8%, but another 15% slower, is not worth it
        # pardiso_solver.set_iparm(2, 2)   # METIS nested dissection 
        pardiso_solver.set_matrix_type(1)  # Real and symmetric - 1, Real, symmetric positive definite matrix - 2 (but does not work, because apparently the matrix is not positive definite)
        # Extra parameters can be set if needed
        # # Configure for memory efficiency #1
        # # pardiso_solver.set_iparm(1, 0)   # Use default values
        # # pardiso_solver.set_iparm(8, 0)   # No iterative refinement (saves memory)
        # # pardiso_solver.set_iparm(10, 0)  # No scaling
        # # pardiso_solver.set_iparm(11, 0)  # No scaling
        # # pardiso_solver.set_iparm(13, 0)  # No weighted matching
        # # pardiso_solver.set_iparm(24, 1)  # Use less memory in parallel solve
        # # pardiso_solver.set_iparm(25, 1)   # Reduce parallel factorization memory
        # # pardiso_solver.set_iparm(31, 0)  # no partial solve
        # # pardiso_solver.set_iparm(36, 0)  # no Schur complement

        p = pardiso_solver.solve(A, b)
    ####################################
    #    DIRECT SOLVER FROM SCIPY      #
    ####################################
    # elif solver == "scipy.spsolve": # (too slow and memory consuming)
    #     from scipy.sparse.linalg import spsolve
    #     logger.info("Using SciPy spsolve (LU) solver.")
    #     A = A.tocsc()
    #     p = spsolve(A, b)
    ##########################################
    #   SCIPY ITERATIVE SOLVER WITH AMG PC   #
    ##########################################
    elif solver_name == "scipy":        
        logger.info("Using SciPy Conjugate Gradient iterative solver.")
        A = A.tocsr()
        from scipy.sparse.linalg import cg

        _preconditioner = preconditioner
        if preconditioner == "amg-sa":
            _preconditioner = "amg-smooth_aggregation"
        elif preconditioner == "amg-rs":
            _preconditioner = "amg-rs"
        elif preconditioner is None:
            _preconditioner = "amg-rs"  # default
        
        try:
            M = get_preconditioner(A, method=_preconditioner)
        except Exception as e:
            logger.error("Failed to create AMG preconditioner. Error: " + str(e))
            raise RuntimeError("Failed to create AMG preconditioner") from e
        
        # Calculate relative tolerance based on gap field
        max_gap_cubed = np.max(g**3)
        if max_gap_cubed > 0:
            rtol = min(1e-12, max_gap_cubed * 1e-14)
        else:
            raise ValueError("Invalid gap field for tolerance calculation")

        p, info = cg(A, b, M=M, rtol=rtol, maxiter=6000)
                
        if info > 0:
            logger.warning(f"Convergence to tolerance not achieved in {info} iterations")
        elif info < 0:
            logger.error(f"Illegal input or breakdown: {info}")
    ###########################################
    #   PETSC ITERATIVE SOLVER WITH GAMG PC   #
    ###########################################
    elif solver_name == "petsc-cg":
        from petsc4py import PETSc
        A = A.tocsr()
        A_p = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
        ksp = PETSc.KSP().create()
        ksp.setOperators(A_p)
        # TODO: can further try different PETSc solvers (the choice is huge: https://petsc.org/release/petsc4py/reference/petsc4py.PETSc.KSP.Type.html
        ksp.setType('cg')
        pc = ksp.getPC()
        if preconditioner == "gamg":
            pc.setType('gamg')
        elif preconditioner == "hypre":
            pc.setType('hypre')
            # pc.setHYPREType('boomeramg')
        elif preconditioner == "ilu":
            pc.setType('ilu') 
        else:
            logger.info(f"Unknown preconditioner {preconditioner}. Using default ILU preconditioner.")
            preconditioner = "ilu"
            pc.setType('ilu') # default

        logger.info(f"Using PETSc KSP iterative solver with {preconditioner} preconditioner.")

        # pc.setType('gamg')     
        # pc.setType('hypre')
        # pc.setType('ilu')
        # pc.setHYPREType('boomeramg')
        ksp.setTolerances(rtol=1e-8)
        ksp.setFromOptions()
        b_p = PETSc.Vec().createWithArray(b)
        x_p = b_p.duplicate()
        ksp.solve(b_p, x_p)
        p = x_p.getArray()
    elif solver_name == "petsc-mumps":
        logger.info("Using PETSc MUMPS direct solver.")
        from petsc4py import PETSc
        A = A.tocsr()
        A_p = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
        b_p = PETSc.Vec().createWithArray(b)
        x_p = b_p.duplicate()

        ksp = PETSc.KSP().create()
        ksp.setOperators(A_p)
        ksp.setType('preonly')
        pc = ksp.getPC()
        pc.setType('lu')
        pc.setFactorSolverType('mumps')

        # PETSc.Options()['mat_mumps_icntl_4'] = 2
        ksp.setFromOptions()
        ksp.solve(b_p, x_p)
        p = x_p.getArray()

    # gc.collect()
        
    return p.reshape((n, n))

def connectivity_analysis(gaps):
    binary = gaps > 0
    n = gaps.shape[0]
    labels = label(binary, connectivity=1)  # 4-connectivity is faster
    
    # Efficient periodic boundary conditions
    left_boundary = labels[:, 0]
    right_boundary = labels[:, -1]
    
    # Create label mapping for merging
    merge_map = {}
    for i in range(n):
        left_label = left_boundary[i]
        right_label = right_boundary[i] 
        if left_label > 0 and right_label > 0 and left_label != right_label:
            merge_map[right_label] = left_label
    
    # Apply merging (single pass)
    if merge_map:
        for old_label, new_label in merge_map.items():
            labels[labels == old_label] = new_label
    
    # Fast percolation check using sets
    top_labels = set(labels[0, :]) - {0}
    bottom_labels = set(labels[-1, :]) - {0}
    
    percolating_labels = top_labels & bottom_labels
    
    if percolating_labels:
        selected_color = next(iter(percolating_labels))
        logger.info(f"Percolation detected with label {selected_color}")
        gaps_original = gaps * (labels == selected_color)
        return gaps_original
    else:
        logger.info("No percolation detected.")
        return None

def solve_fluid_problem(gaps, solver):
    logger.info("Starting fluid solver.")

    n = gaps.shape[0]
    if n == 0:
        logger.error("Empty gap field.")
        return None, None, None

    x = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, x)

    logger.info("Checking connectivity.")

    start = time.time()
    gaps_original = connectivity_analysis(gaps)
    logger.info("Connectivity analysis: CPU time  = {1:.3f} sec".format(n, time.time() - start))

    
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
        logger.error(f"Error in fluid solver: {e}")
        return None, None, None

    logger.info("Fluid solver finished.")
    logger.info("Calculating flux with simple boundary condition approach.")

    # Calculate gradients with proper boundary conditions
    dx = 1 / (n - 1)
    dpdx, dpdy = _calculate_gradients_simple_bc(n, p, dx, gaps_original)

    # Use numba-accelerated flux calculation with ORIGINAL gaps
    flux = _calculate_flux_numba(n, gaps_original, dpdx, dpdy)
    # filtered_flux = _filter_flux_numba(n, gaps_original, flux)
    
    logger.info("finished.")

    return gaps_original, p, flux
    # return gaps_original, p, filtered_flux

def get_preconditioner(A, method = "amg.rs"):
    if method == "amg-smooth_aggregation":
        import pyamg
        from scipy.sparse.linalg import LinearOperator

        try:
            logger.info("Using Smoothed Aggregation AMG preconditioner.")
            ml = pyamg.smoothed_aggregation_solver(A, max_coarse=A.shape[0] //1000) # Maybe it could be further optimized

            M = ml.aspreconditioner(cycle='V')
            # M = LinearOperator(A.shape, matvec=lambda v: ml.solve(v, tol=1e-2, maxiter=1)) # alternative
            return M
        except:
            logger.warning("AMG.Smooth_Aggregation failed, try AMG.RS")
            method = "amg-rs"

    if method == "amg-rs":
        import pyamg
        from scipy.sparse.linalg import LinearOperator

        logger.info("Using Ruge-Stuben AMG preconditioner.")
        try:
            ml = pyamg.ruge_stuben_solver(A, max_coarse=A.shape[0] // 1000,CF='RS')
            M = LinearOperator(A.shape, matvec=lambda v: ml.solve(v, tol=1e-2, maxiter=1))
            # M = ml.aspreconditioner(cycle='V') # alternative
            return M
        except:
            logger.warning("AMG.RS failed, falling back to ILU")
            raise RuntimeError("AMG.RS failed")
    else:
        logger.warning(f"Unknown preconditioner method: {method}")
        raise ValueError(f"Unknown preconditioner method: {method}")

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

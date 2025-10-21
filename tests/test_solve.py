"""
Test Reynolds Fluid Flow Finite Difference Solver

Author: Vladislav A. Yastrebov (CNRS, Mines Paris - PSL)
AI: Cursor, Claude, ChatGPT
Date: Aug 2024-Sept 2025
License: BSD 3-Clause
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import ifft2, fftfreq, fftshift, ifftshift, fftn, ifftn
import time


import Transport_code_accelerated as FS
import RandomField as rf # https://github.com/vyastreb/SelfAffineSurfaceGenerator
from mpl_toolkits.axes_grid1 import make_axes_locatable

FS.setup_logging()  
FS.set_verbosity('info')
# FS.set_verbosity('warning')

def main():
    N0 = 512           # Size of the random field
    """
        Available solvers:
        + "scipy.amg-rs" - optimal for memory
        + "cholesky"     - optimal for speed (if enough memory), only 10% more than petsc
        + "petsc"        - very fast and memory efficient, 4x scipy.amg.rs and only 20% more memory
    """
    solver = "scipy.amg-smooth_aggregation"  # Choose solver here
    k_low =   8 / N0   # Lower cutoff of the power spectrum
    k_high = 20 / N0   # Upper cutoff of the power spectrum
    Hurst = 0.5         # Hurst exponent
    dim = 2             # Dimension of the random field
    seed = 23349        # Seed for the random number generator
    plateau = True      # Use plateau in the power spectrum
    np.random.seed(seed)

    # Generate a normalized random field
    random_field = rf.periodic_gaussian_random_field(dim = dim, N = N0, Hurst = Hurst, k_low = k_low, k_high = k_high, plateau = plateau)
    random_field /= np.std(random_field)

    x = np.linspace(0, 1, N0)
    X, Y = np.meshgrid(x, x)

    #################################################
    #           Compute and PLOT ALL                #
    #################################################

    delta = 0.3
    g = random_field + delta
    g[g < 0] = 0

    # Solve for the current orientation
    start = time.time()
    _, pressure, flux = FS.solve_fluid_problem(g, solver)
    print("Solver CPU time: ", time.time() - start, "s")

    # Save results in npz file
    output_name = f"fluid_flow_H_{Hurst}_kl_{int(k_low*N0):d}_ks_{int(k_high*N0):d}_N_{N0}_delta_{delta:.2f}_solver_{solver}.npz"
    print("Saving results to ", output_name)
    np.savez_compressed(output_name, pressure=pressure, flux=flux)

if __name__ == "__main__":
    main()

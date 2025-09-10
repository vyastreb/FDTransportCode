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

# Plotting function
def plot_fields(g, pressure, flux, suffix="", VectorField=False, plot_n_vectors = 50):
    # 
    X, Y = np.meshgrid(np.linspace(0, 1, g.shape[0]), np.linspace(0, 1, g.shape[1]))

    N0 = g.shape[0]
    gg = g.copy()
    gg[gg<=0] = np.nan

    # Plot gap function
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(gg, cmap='viridis', origin='lower', extent=[0, 1, 0, 1], interpolation="none")
    fig.colorbar(cax, label='Gap')
    ax.set_title('Gap Function')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()
    fig.savefig(f"FS_gap_{suffix}.png",dpi=300)

    # Pressure plot
    pressure[np.isnan(gg)] = np.nan
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(pressure, cmap='jet', origin='lower', extent=[0, 1, 0, 1], interpolation="none")
    ax.contour(X, Y, pressure, levels=50, colors='black', linewidths=0.5)
    # Make colorbar the same height as the plot
    divider = make_axes_locatable(ax)
    cax_cb = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(cax, cax=cax_cb, label='Pressure')
    ax.set_title('Pressure Distribution')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()
    fig.savefig(f"FS_pressure_n_{N0}_{suffix}.png",dpi=300)

    # Flux plot
    norm_flux = np.sqrt(flux[:,:,0]**2 + flux[:,:,1]**2)    

    fig, ax = plt.subplots(figsize=(10, 8))
    Vmin = np.nanmin(np.log10(norm_flux[norm_flux>0]))
    Vmax = np.nanmax(np.log10(norm_flux))
    Vmin += 0.8*abs(Vmax-Vmin)
    Vmax -= 0.*abs(Vmax-Vmin)
    # cax = ax.imshow(np.log10(norm_flux), cmap='jet', origin='lower', extent=[0, 1, 0, 1], vmin=Vmin, vmax=Vmax, interpolation="none")
    cax = ax.imshow(norm_flux, cmap='jet', origin='lower', extent=[0, 1, 0, 1], interpolation="none")
    if VectorField:
        every = N0 // plot_n_vectors
        ax.quiver(X[::every, ::every], Y[::every, ::every], 
                flux[::every, ::every, 0], flux[::every, ::every, 1],
                color='red', scale=None, width=0.002)    
    fig.colorbar(cax, label='Log10(Flux Magnitude)')
    ax.set_title('Flux Distribution')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()
    fig.savefig(f"FS_flux__n_{N0}_{suffix}.png", dpi=300)
    
############################################
#                     MAIN                 #
############################################

def main():
    N0 = 1000           # Size of the random field
    """
        Available solvers:
        + "scipy.amg.rs" - optimal for memory
        + "cholesky"     - optimal for speed (if enough memory), only 10% more than petsc
        + "petsc"        - very fast and memory efficient, 4x scipy.amg.rs and only 20% more memory
    """
    solver = "petsc"
    k_low =   4 / N0   # Lower cutoff of the power spectrum
    k_high = 32 / N0   # Upper cutoff of the power spectrum
    Hurst = 0.75         # Hurst exponent
    dim = 2             # Dimension of the random field
    seed = 12345        # Seed for the random number generator
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

    delta = 0.4
    g = random_field + delta
    g[g < 0] = 0

    # Solve for the current orientation
    start = time.time()
    filtered_gaps, pressure, flux = FS.solve_fluid_problem(g, solver)
    print("Solver CPU time: ", time.time() - start, "s")

    if flux is None:
        print("No flux solution found.")
    else:   
        Q_total, flux_conservation_error = FS.compute_total_flux(filtered_gaps, flux, N0)

        plot_fields(filtered_gaps, pressure, flux, suffix=f"n_{N0}", VectorField=False, plot_n_vectors = 50)

    # # Solve for rotated orientation
    # g = np.rot90(g)

    # start = time.time()
    # filtered_gaps, pressure, flux = FS.solve_fluid_problem(g, solver)
    # print("Time taken: ", time.time() - start, "s")

    # if flux is None:
    #     print("No flux solution found.")
    # else:
    #     Q_total, flux_conservation_error = FS.compute_total_flux(filtered_gaps, flux, N0)

    #     plot_fields(filtered_gaps, pressure, flux, suffix=f"n_{N0}", VectorField=False, plot_n_vectors = 50)

if __name__ == "__main__":
    main()

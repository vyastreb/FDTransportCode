"""
Test Reynolds Fluid Flow Finite Difference Solver

Author: Vladislav A. Yastrebov (CNRS, Mines Paris - PSL)
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

# Plotting function
def plot_fields(g, pressure, flux, suffix="", VectorField=False, plot_n_vectors = 50):
    # Plot gap function
    N0 = g.shape[0]
    gg = g.copy()
    gg[gg<=0] = np.nan
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
    fig.savefig(f"FS_pressure_{suffix}.png",dpi=300)

    # Flux plot
    norm_flux = np.sqrt(flux[:,:,0]**2 + flux[:,:,1]**2)
    every = N0 // plot_n_vectors

    fig, ax = plt.subplots(figsize=(10, 8))
    Vmin = np.nanmin(np.log10(norm_flux[norm_flux>0]))
    Vmax = np.nanmax(np.log10(norm_flux))
    Vmin += 0.8*abs(Vmax-Vmin)
    Vmax -= 0.*abs(Vmax-Vmin)
    print(f"Vmin: {Vmin}, Vmax: {Vmax}, Range: {Vmax - Vmin}")
    cax = ax.imshow(np.log10(norm_flux), cmap='jet', origin='lower', extent=[0, 1, 0, 1], vmin=Vmin, vmax=Vmax, interpolation="none")
    if VectorField:
        ax.quiver(X[::every, ::every], Y[::every, ::every], 
                flux[::every, ::every, 0], flux[::every, ::every, 1],
                color='red', scale=None, width=0.002)    
    fig.colorbar(cax, label='Log10(Flux Magnitude)')
    ax.set_title('Flux Distribution')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()
    fig.savefig("FS_flux_vector_n_{0:d}_solver_{1}.png".format(N0, solver), dpi=300)


N0 = 4096           # Size of the random field
solver = "direct"   # "direct" or "iterative" Solver to use for the diffusion problem
k_low =   4 / N0   # Lower cutoff of the power spectrum
k_high = 512 / N0   # Upper cutoff of the power spectrum
Hurst = 0.75         # Hurst exponent
dim = 2             # Dimension of the random field
seed = 12345        # Seed for the random number generator
plateau = True      # Use plateau in the power spectrum
np.random.seed(seed)

# Generate a random field
random_field = rf.periodic_gaussian_random_field(dim = dim, N = N0, Hurst = Hurst, k_low = k_low, k_high = k_high, plateau = plateau)

# Normalize
random_field /= np.std(random_field)

# Grid setup
x = np.linspace(0, 1, N0)
X, Y = np.meshgrid(x, x)

g = random_field + 1
g[g < 0] = 0  

#################################################
#
#   ####    ####   ##     ##     ##  #######
#  ##  ##  ##  ##  ##     ##     ##  ##
#  ##      ##  ##  ##      ##   ##   ##
#   ####   ##  ##  ##      ##   ##   #####
#      ##  ##  ##  ##       ## ##    ##
#  ##  ##  ##  ##  ##       ## ##    ##
#   ####    ####   #######   ###     #######
#
#################################################

if False:
    Num_steps = 20
    Delta = np.linspace(2, -1, num=Num_steps)

    G = np.zeros(Num_steps)
    Q = np.zeros(Num_steps)
    A = np.zeros(Num_steps)

    for step, delta in enumerate(Delta):
        g = random_field + delta
        g[g < 0] = 0
        filtered_gaps, pressure, flux = FS.solve_fluid_problem(g, solver)
        if flux is None:
            Q = Q[:step]
            G = G[:step]
            A = A[:step]
            break
        else:
            Q[step] = np.nansum(flux[:,:,0]) / N0**2
            G[step] = delta
            A[step] = np.sum(g == 0) / N0**2
            print("=========> Step:", step, "Delta:", delta, "Q =", Q[step], "G =", G[step], "A =", A[step])

    fig,ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].grid()
    ax[0].plot(G, Q, 'o-', color="skyblue")
    ax[0].set_xlabel('Gap (G)')
    ax[0].set_ylabel('Flux (Q)')
    ax[0].set_title('Gap vs Flux')

    ax[1].grid()
    ax[1].plot(A, Q, 'o-', color='firebrick')
    ax[1].set_xlabel('Real Contact Area (A)')
    ax[1].set_ylabel('Flux (Q)')
    ax[1].set_title('Real Contact Area vs Flux')
    plt.show()
    fig.savefig("FS_Q_vs_G_n_{0:d}_solver_{1}.png".format(N0, solver))

    exit(1)
#################################################
#           Compute and PLOT ALL                #
#################################################

delta = 0.3
g = random_field + delta
g[g < 0] = 0

start = time.time()
filtered_gaps, pressure, flux = FS.solve_fluid_problem(g, solver)
print("Time taken: ", time.time() - start, "s")

if flux is None:
    print("No flux solution found.")
else:
    Q1 = np.nansum(flux[:,:,0]) / N0**2
    Q2 = np.nansum(flux[:,:,1]) / N0**2
    flux_in = np.nansum(flux[1, :, 1]) / N0
    flux_out = np.nansum(flux[-2, :, 1]) / N0
    print(f"Entering flux at x=0: {flux_in}")
    print(f"Escaping flux at x=1: {flux_out}")
    G = delta
    A = np.sum(g == 0) / N0**2
    print("=========> Delta:", delta, "Q1 =", Q1, "Q2 =", Q2, "G =", G, "A =", A)

    plot_fields(filtered_gaps, pressure, flux, suffix=f"n_{N0}", VectorField=False, plot_n_vectors = 50)
exit(1)
g = np.rot90(g)

start = time.time()
filtered_gaps, pressure, flux = FS.solve_fluid_problem(g, solver)
print("Time taken: ", time.time() - start, "s")

if flux is None:
    print("No flux solution found.")
else:
    Q1 = np.nansum(flux[:,:,0]) / N0**2
    Q2 = np.nansum(flux[:,:,1]) / N0**2
    flux_in = np.nansum(flux[1, :, 1]) / N0
    flux_out = np.nansum(flux[-2, :, 1]) / N0
    print(f"Entering flux at x=0: {flux_in}")
    print(f"Escaping flux at x=1: {flux_out}")
    G = delta
    A = np.sum(g == 0) / N0**2
    print("=========> Delta:", delta, "Q1 =", Q1, "Q2 =", Q2, "G =", G, "A =", A)

    plot_fields(filtered_gaps, pressure, flux, suffix=f"n_{N0}_rotated", VectorField=False, plot_n_vectors = 50)





"""
Test of total flow evolution in Reynolds Fluid Flow over truncated fractal landscape

Author: Vladislav A. Yastrebov (CNRS, Mines Paris - PSL)
AI: Cursor, Claude, ChatGPT
Date: Sept 2025
License: BSD 3-Clause
"""

import numpy as np
import matplotlib.pyplot as plt

import Transport_code_accelerated as FS
import RandomField as rf # https://github.com/vyastreb/SelfAffineSurfaceGenerator
from mpl_toolkits.axes_grid1 import make_axes_locatable

FS.setup_logging()  
# FS.set_verbosity('info')
FS.set_verbosity('error')

def main():
    # Construct random self-affine surface
    N0 = 256           # Size of the random field
    solver = "petsc"   # or "cholesky" or "scipy.amg.rs"
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

###################################################
#                                                 #
#     ####    ####   ##     ##     ##  #######    #
#    ##  ##  ##  ##  ##     ##     ##  ##         #
#    ##      ##  ##  ##      ##   ##   ##         #
#     ####   ##  ##  ##      ##   ##   #####      #
#        ##  ##  ##  ##       ## ##    ##         #
#    ##  ##  ##  ##  ##       ## ##    ##         #
#     ####    ####   #######   ###     #######    #
#                                                 #
###################################################

    Num_steps = 20
    Delta1 = np.linspace(2, 0.5, num=Num_steps)
    Delta2 = np.linspace(0.45, 0, num=Num_steps)
    Delta = np.concatenate((Delta1, Delta2))    

    G = np.zeros(Num_steps*2)
    Q = np.zeros(Num_steps*2)
    A = np.zeros(Num_steps*2)

    for step, delta in enumerate(Delta):
        print(f"Step {step+1}/{len(Delta)}")
        # Surface truncation
        g = random_field + delta
        g[g < 0] = 0

        # Solve problem
        filtered_gaps, pressure, flux = FS.solve_fluid_problem(g, solver)

        # Post-processing
        if flux is None:
            Q = Q[:step]
            G = G[:step]
            A = A[:step]
            break
        else:
            flux_total, _ = FS.compute_total_flux(filtered_gaps, flux, N0)            
            Q[step] = flux_total
            G[step] = delta
            A[step] = np.sum(g == 0) / N0**2

    fig,ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].grid()
    ax[0].plot(G, Q, 'o-', color="skyblue")
    ax[0].set_xlabel('Gap (G)')
    ax[0].set_ylabel('Flux (Q)')
    ax[0].set_yscale('log')
    ax[0].set_title('Gap vs Flux')

    ax[1].grid()
    ax[1].plot(A, Q, 'o-', color='firebrick')
    ax[1].set_xlabel('Real Contact Area (A)')
    ax[1].set_ylabel('Flux (Q)')
    ax[1].set_yscale('log')
    ax[1].set_title('Real Contact Area vs Flux')
    plt.show()
    fig.savefig("FS_Q_vs_G_n_{0:d}_solver_{1}.png".format(N0, solver))

if __name__ == "__main__":
    main()
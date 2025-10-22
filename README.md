![Reynolds Fluid Solver](./docs/img/header.png)

# Finite-Difference Reynolds Fluid Solver

## Description

An efficient python code to solve diffusion equation on a Cartesian grid:
<!-- $$
\begin{aligned}
&\nabla\cdot\left(g^3 \nabla p\right) = 0,\\
&p(x=0) = 1, \quad p(x=1) = 0,\\ 
&p(y=0) = p(y=1), \quad 
\left.\frac{\partial p}{\partial y}\right|_{y=0} =
\left.\frac{\partial p}{\partial y}\right|_{y=1}
\end{aligned}
$$ -->
![equation to be solved](./docs/img/eq.png)

## What can it do?

+ It takes as input a gap field $g$.
+ It analyzes its connectivity and removes isolated islands and checks for percolation (whether a flow problem can be solved).
+ It dilates non-zero gap field to properly handle impenetrability of channels, it allows not to erode the domain for flux calculation.
+ It applies an inlet pressure $p_i=1$  on one side $x=0$ and an outlet pressure $p_0=0$ on the opposite side $x=1$ and uses periodic boundary conditions on the lateral sides $y=\{0,1\}$.
+ It constructs a sparse matrix with conductivity proportional to $g^3$.
+ Different solvers (direct and iterative with appropriate preconditioners) are selected and tuned to solve efficiently the resulting linear system of equations.
+ Total flux is properly computed.

## Usage

1. Clone the repository
```bash 
git clone git@github.com:vyastreb/FDTransportCode.git
cd FDTransportCode
```
2. Install the package and its dependencies
```bash
pip install -e .
pip install -r requirements.txt
```
or with conda
```bash
conda install --file requirements.txt
```
3. Run a minimal test (incompressible potential flow around a circular inclusion)
```python
import numpy as np
from fluxflow import transport as FS
import matplotlib.pyplot as plt

n = 100
X, Y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
gaps = (np.sqrt((X - 0.5)**2 + (Y - 0.5)**2) > 0.2).astype(float)

_, _, flux = FS.solve_fluid_problem(gaps, solver="auto")
if flux is not None:
    plt.imshow(np.sqrt(flux[:, :, 0]**2 + flux[:, :, 1]**2),
               origin='lower', cmap='jet')
    plt.show()
```
4. Run the test suit
```bash
python -m pytest -q
```
5. Or run these tests manually `/tests/test_evolution.py`, `/tests/test_solve.py` and `/tests/test_solvers.py`.

## Available Solvers and Preconditioners

The fluid flow solver supports several linear system solvers and preconditioners for efficient and robust solution of large sparse systems:

| Solver String | Solver Type | Preconditioner | Backend | Description |
|---------------|-------------|----------------|---------|-------------|
| `petsc-cg.hypre` | Iterative (CG) | HYPRE | PETSc | 🥇 CG with HYPRE BoomerAMG. The fastest for moderate problems. |
| `pardiso` | Direct | - | Intel MKL | PARDISO direct solver. Very fast but consumes a lot of memory. |
| `scipy.amg-rs` | Iterative (CG) | AMG (Ruge-Stuben) | SciPy/PyAMG | CG with Ruge-Stuben AMG. Only two times slower than the fastest.  |
| `scipy.amg-smooth_aggregation` | Iterative (CG) | AMG (Smoothed Aggregation) | SciPy/PyAMG | CG with Smoothed Aggregation AMG. Memory efficient, but relatively slow.|
| `cholesky` | Direct | - | scikit-sparse | CHOLMOD Cholesky decomposition. Slightly lower memory consumption for huge problems, but it is slow. |
| `petsc-cg.gamg` | Iterative (CG) | GAMG | PETSc | CG with Geometric Algebraic Multigrid. Not very reliable in performance, 2-3 times slower than the fastest solver. |
| `petsc-mumps` | Direct | - | PETSc/MUMPS | MUMPS direct solver via PETSc. For moderate problems, five times slower than the fastest solver. |
| `petsc-cg.ilu` | Iterative (CG) | ILU | PETSc | CG with Incomplete LU factorization. The slowest. |

Relevant CPU times for a relatively small problem with $N\times N = 2000\times 2000$ grid points.

| **Solver**                    | **CPU time (s)** |
|-------------------------------|-----------------:|
| petsc-cg.hypre                | 4.46 |
| pardiso                       | 8.53 |
| scipy.amg-rs                  | 8.96 |
| petsc-cg.gamg                 | 11.96 |
| scipy.amg-smooth_aggregation  | 15.48 |
| cholesky                      | 20.61 |
| petsc-mumps                   | 26.14 |
| petsc-cg.ilu                  | 134.98 |

**Rules of thumb:** 
- For fastest computation: use `pardiso` (consumes a lot of memory) or `petsc-cg.hypre` (the only difficulty is to install PETSc);
- For best memory efficiency: use `scipy.amg-rs`;
- For small-scale problems (for $N<2000$): use `pardiso`;
- For large-scale problems (for $N>2000$): use `petsc-cg.hypre`;
- Avoid `petsc-cg.ilu`.

The most reliable solvers for big problems are `petsc-cg.hypre` and `pardiso`. Here are the test data obtained on rough "contact" problems on Intel(R) Xeon(R) Platinum 8488C. Only solver's time is shown.

<table>
  <thead>
    <tr><th rowspan="2">N</th><th colspan="2">CPU time (s)</th></tr>
    <tr><th>PETSc-CG.Hypre</th><th>Intel MKL Pardiso</th></tr>
  </thead>
  <tbody>
    <tr><td>20 000</td><td>1059.22</td><td>∅</td></tr>
    <tr><td>10 000</td><td>278.18</td><td>112.38</td></tr>
    <tr><td>5 000</td><td>70.62</td><td>28.42</td></tr>
    <tr><td>2 500</td><td>17.72</td><td>6.34</td></tr>
    <tr><td>1 250</td><td>4.47</td><td>1.93</td></tr>
  </tbody>
</table>

∅ $-$ `pardiso` could not run as it required more than 256 GB or memory.

**CPU/RAM Performance**

Performance of the code on a truncated rough surface is shown below. The peak memory consumption and the CPU time required to perform connectivity analysis, constructing the matrix and solving the linear system are provided. The real number of DOFs is reported which corresponds to approximately 84% of the square grid $N\times N$ for $N\in\{500, 1\,000, 2\,000, 4\,000, 6\,000, 8\,000, 16\,000\}$.

![CPU and RAM performance of the solver](./docs/img/CPU_RAM_real_dof_performance.png)





## Illustration

An example of a fluid flow simulation, solved on the grid $N\times N = 8\,000 \times 8\,000$ which features a truncated self-affine rough surface with a rich spectrum. Solution time on my laptop with `petsc` is only 97 seconds and the peak memory consumption is 25.8 GB.

![Solution for 64 million grid points](./docs/img/illustration.jpg)

Another example for a grid $N\times N = 20\,000 \times 20\,000$. Simulation time (sequential) $\approx 17$ minutes on Intel(R) Xeon(R) Platinum 8488C with the peak memory below 230 GB with `petsc-cg.gamg` solver.

![Solution for 400 million grid points](./docs/img/illustration_2.jpg)

## Info

+ Author: Vladislav A. Yastrebov (CNRS, Mines Paris - PSL)
+ AI usage: Cursor & Copilot (different models), ChatGPT 4o, 5, Claude Sonnet 3.7, 4, 4.5
+ License: BSD 3-clause
+ Date: Sept-Oct 2025




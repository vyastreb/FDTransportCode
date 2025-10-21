# Solver Unit Test

## Overview

`test_solvers.py` is a comprehensive unit test that validates all available solvers and preconditioners in the Transport Code.

## Usage

### Basic Usage

```bash
python test_solvers.py
```

This will:
1. Generate a test problem with a random field (N0=128, k_low=1/N0, k_high=12/N0)
2. Test all available solvers and preconditioners
3. Print results to console
4. Save detailed results to `test_solvers.result`

### Tested Solvers

The test covers the following solver configurations:

#### Direct Solvers
- **cholesky**: CHOLMOD solver from scikit-sparse
- **pardiso**: Intel oneAPI MKL PARDISO solver
- **petsc-mumps**: PETSc with MUMPS direct solver

#### Iterative Solvers (SciPy)
- **scipy.amg.rs**: SciPy CG with Ruge-Stuben AMG preconditioner
- **scipy.amg.smooth_aggregation**: SciPy CG with Smoothed Aggregation AMG preconditioner

#### Iterative Solvers (PETSc)
- **petsc-cg.gamg**: PETSc CG with GAMG preconditioner
- **petsc-cg.hypre**: PETSc CG with Hypre preconditioner
- **petsc-cg.ilu**: PETSc CG with ILU preconditioner

## Output

### Console Output

The test provides real-time progress updates with:
- Test configuration details
- Gap field statistics
- Individual test results with timing and flux information
- Summary statistics
- Speed comparison of successful solvers

### Result File

The `test_solvers.result` file contains:
- Date and time of test execution
- Summary statistics (total/passed/failed tests)
- Detailed results for each solver configuration
- Error messages for failed tests

### Example Output Format

```
[PASS] cholesky                     | Time:   2.345s | Q_total: 1.234567e-03 | Conv.Error: 1.23e-08
[FAIL] pardiso                      | Error: Module not found: pypardiso
[PASS] scipy.amg.rs                 | Time:   5.678s | Q_total: 1.234567e-03 | Conv.Error: 2.34e-08
```

## Requirements

The test requires all dependencies for the Transport Code, including:
- numpy
- scipy
- numba
- scikit-sparse (for cholesky)
- pypardiso (for pardiso, optional)
- petsc4py (for PETSc solvers, optional)
- pyamg (for AMG preconditioners)
- RandomField module

## Exit Codes

- **0**: All tests passed
- **1**: One or more tests failed

## Customization

To modify test parameters, edit the `main()` function in `test_solvers.py`:

```python
# Test parameters
N0 = 128           # Grid size
k_low = 1.0 / N0   # Lower cutoff
k_high = 12.0 / N0 # Upper cutoff
```

## Notes

- Some solvers may fail if their dependencies are not installed (e.g., pardiso requires pypardiso)
- The test uses a fixed random seed (23349) for reproducibility
- Logging verbosity is set to 'warning' during tests to reduce output noise
- Each solver is tested on the same gap field for fair comparison

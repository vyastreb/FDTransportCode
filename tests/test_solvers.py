"""
Unit Test for Transport Solver
Tests all available solvers and preconditioners

Date: Oct 2025
License: CC0
Author: Claude Sonnet 4.5
Verified by: Vladislav A. Yastrebov (CNRS, Mines Paris - PSL)
"""

import numpy as np
import time
import sys
import traceback
from datetime import datetime

from fluxflow import transport as FS
from fluxflow import random_field as RF

# import pytest
# pytestmark = pytest.mark.skip(reason="WIP: Long test, run manually when needed.")

FS.setup_logging()
FS.set_verbosity('warning')

class SolverTestResult:
    """Store test results for each solver configuration"""
    def __init__(self, solver_name, preconditioner=None):
        self.solver_name = solver_name
        self.preconditioner = preconditioner
        self.success = False
        self.error_message = None
        self.cpu_time = None
        self.Q_total = None
        self.flux_conservation_error = None
        
    def full_name(self):
        """Return full solver name including preconditioner if applicable"""
        if self.preconditioner:
            return f"{self.solver_name}.{self.preconditioner}"
        return self.solver_name
    
    def __str__(self):
        status = "PASS" if self.success else "FAIL"
        name = self.full_name()
        
        if self.success:
            return (f"[{status}] {name:30s} | "
                   f"Time: {self.cpu_time:7.3f}s | ")
        else:
            error_short = self.error_message.split('\n')[0][:60] if self.error_message else "Unknown error"
            return f"[{status}] {name:30s} | Error: {error_short}"


def generate_test_problem(N0, k_low, k_high, Hurst=0.5, seed=23349, delta=0.3):
    """
    Generate a test problem with random field geometry.
    
    Parameters:
    -----------
    N0 : int
        Grid size
    k_low : float
        Lower cutoff of the power spectrum
    k_high : float
        Upper cutoff of the power spectrum
    Hurst : float
        Hurst exponent (default: 0.5)
    seed : int
        Random seed (default: 23349)
    delta : float
        Offset for gap field (default: 0.3)
        
    Returns:
    --------
    gaps : ndarray
        Gap field with shape (N0, N0)
    """
    np.random.seed(seed)
    
    # Generate normalized random field
    random_field = RF.periodic_gaussian_random_field(
        dim=2, 
        N=N0, 
        Hurst=Hurst, 
        k_low=k_low, 
        k_high=k_high, 
        plateau=True
    )
    random_field /= np.std(random_field)
    
    # Create gap field
    gaps = random_field + delta
    gaps[gaps < 0] = 0
    
    return gaps


def _test_solver(solver_name, preconditioner, gaps, N0):
    """
    Test a specific solver configuration.
    
    Parameters:
    -----------
    solver_name : str
        Name of the solver
    preconditioner : str or None
        Preconditioner name (if applicable)
    gaps : ndarray
        Gap field
    N0 : int
        Grid size
        
    Returns:
    --------
    result : SolverTestResult
        Test result object
    """
    result = SolverTestResult(solver_name, preconditioner)
    
    # Construct full solver string
    if preconditioner:
        solver_str = f"{solver_name}.{preconditioner}"
    else:
        solver_str = solver_name
    
    try:
        # Time the solver
        start_time = time.time()
        filtered_gaps, pressure, flux = FS.solve_fluid_problem(gaps.copy(), solver_str)
        elapsed_time = time.time() - start_time
        
        # Check if solution was found
        if flux is None or pressure is None or filtered_gaps is None:
            result.error_message = "No solution found (flux/pressure is None)"
            return result
        
        # Compute flux and conservation error
        Q_total, flux_conservation_error = FS.compute_total_flux(filtered_gaps, flux, N0)
        
        # Store results
        result.success = True
        result.cpu_time = elapsed_time
        
    except Exception as e:
        result.error_message = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
    
    return result


def run_all_tests(N0=128, k_low=None, k_high=None):
    """
    Run tests for all available solvers and preconditioners.
    
    Parameters:
    -----------
    N0 : int
        Grid size (default: 128)
    k_low : float
        Lower cutoff of power spectrum (default: 1/N0)
    k_high : float
        Upper cutoff of power spectrum (default: 12/N0)
        
    Returns:
    --------
    results : list of SolverTestResult
        List of test results
    """
    # Set default values
    if k_low is None:
        k_low = 1.0 / N0
    if k_high is None:
        k_high = 12.0 / N0
    
    print(f"{'='*80}")
    print(f"Transport Solver Unit Test")
    print(f"{'='*80}")
    print(f"Test Configuration:")
    print(f"  Grid size (N0):     {N0}")
    print(f"  k_low:              {k_low:.6f} ({k_low*N0:.2f}/N0)")
    print(f"  k_high:             {k_high:.6f} ({k_high*N0:.2f}/N0)")
    print(f"  Hurst exponent:     0.5")
    print(f"  Random seed:        23349")
    print(f"  Gap offset (delta): 0.3")
    print(f"{'='*80}\n")
    
    # Generate test problem
    print("Generating test problem...")
    gaps = generate_test_problem(N0, k_low, k_high)
    print(f"Gap field statistics:")
    print(f"  Mean:    {np.mean(gaps):.6f}")
    print(f"  Std:     {np.std(gaps):.6f}")
    print(f"  Min:     {np.min(gaps):.6f}")
    print(f"  Max:     {np.max(gaps):.6f}")
    print(f"  Zeros:   {np.sum(gaps == 0)} ({100*np.sum(gaps == 0)/gaps.size:.1f}%)")
    print()
    
    # Define all solver configurations to test
    # Based on Transport_code_accelerated.py
    solver_configs = [
        # Direct solvers
        ("cholesky", None),
        ("pardiso", None),
        
        # SciPy iterative solver with different preconditioners
        ("scipy", "amg-rs"),
        ("scipy", "amg-smooth_aggregation"),
        
        # PETSc iterative solver with different preconditioners
        ("petsc-cg", "gamg"),
        ("petsc-cg", "hypre"),
        ("petsc-cg", "ilu"),
        
        # PETSc direct solver
        ("petsc-mumps", None),
    ]
    
    results = []
    
    print(f"Testing {len(solver_configs)} solver configurations...\n")
    print(f"{'='*80}")
    
    for i, (solver_name, preconditioner) in enumerate(solver_configs, 1):
        full_name = f"{solver_name}.{preconditioner}" if preconditioner else solver_name
        print(f"[{i}/{len(solver_configs)}] Testing {full_name}...")
        
        result = _test_solver(solver_name, preconditioner, gaps, N0)
        results.append(result)
        
        print(f"    {result}")
        print()
    
    return results


def save_results(results, filename="test_solvers.result"):
    """
    Save test results to a file.
    
    Parameters:
    -----------
    results : list of SolverTestResult
        List of test results
    filename : str
        Output filename (default: "test_solvers.result")
    """
    with open(filename, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Transport Solver Unit Test Results\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # Summary statistics
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.success)
        failed_tests = total_tests - passed_tests
        
        f.write(f"Summary:\n")
        f.write(f"  Total tests:  {total_tests}\n")
        f.write(f"  Passed:       {passed_tests}\n")
        f.write(f"  Failed:       {failed_tests}\n")
        f.write(f"  Success rate: {100*passed_tests/total_tests:.1f}%\n\n")
        
        f.write("="*80 + "\n")
        f.write("Detailed Results:\n")
        f.write("="*80 + "\n\n")
        
        # Group by success/failure
        passed = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        if passed:
            f.write(f"PASSED TESTS ({len(passed)}):\n\n")
            f.write("-"*80 + "\n")
            for result in passed:
                f.write(f"{result}\n")
            f.write("\n")
        
        if failed:
            f.write(f"FAILED TESTS ({len(failed)}):\n\n")
            f.write("-"*80 + "\n")
            for result in failed:
                f.write(f"{result}\n")
                if result.error_message:
                    # Write first few lines of error
                    error_lines = result.error_message.split('\n')[:5]
                    for line in error_lines:
                        f.write(f"    {line}\n")
                f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("End of Report\n")
        f.write("="*80 + "\n")
    
    print(f"\nResults saved to: {filename}")


def print_summary(results):
    """
    Print a summary of test results.
    
    Parameters:
    -----------
    results : list of SolverTestResult
        List of test results
    """
    print(f"{'='*80}")
    print(f"Test Summary")
    print(f"{'='*80}")
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.success)
    failed_tests = total_tests - passed_tests
    
    print(f"Total tests:  {total_tests}")
    print(f"Passed:       {passed_tests} ({100*passed_tests/total_tests:.1f}%)")
    print(f"Failed:       {failed_tests} ({100*failed_tests/total_tests:.1f}%)")
    print()
    
    # Find fastest solver among successful ones
    passed = [r for r in results if r.success]
    if passed:
        fastest = min(passed, key=lambda r: r.cpu_time)
        print(f"Fastest solver: {fastest.full_name()} ({fastest.cpu_time:.3f}s)")
        
        # Show timing comparison
        print(f"\nTiming Comparison (successful solvers):")
        print(f"{'-'*80}")
        sorted_results = sorted(passed, key=lambda r: r.cpu_time)
        for result in sorted_results:
            speedup = result.cpu_time / fastest.cpu_time
            print(f"  {result.full_name():30s} | {result.cpu_time:7.3f}s | "
                  f"{speedup:5.2f}x slower than fastest")
    
    print(f"{'='*80}")


def test_solvers():
    """Main test function"""
    # Test parameters
    N0 = 2048
    k_low = 1.0 / N0
    k_high = 12.0 / N0
    
    # Run all tests
    results = run_all_tests(N0=N0, k_low=k_low, k_high=k_high)
    
    # Print summary
    print_summary(results)
    
    # Save results to file
    save_results(results, filename="test_solvers.result")
    
    assert all(r.success for r in results), "Some solver tests failed"


if __name__ == "__main__":
    test_solvers()

"""
Fluid Flow Field Plotting Utility

Plotting functions for gap, pressure, and flux fields from saved fluid simulation data.

AI-coded: Claude 4.5 in Cursor
Verified by Vladislav A. Yastrebov (CNRS, Mines Paris - PSL)
Date: Aug 2024-Sept 2025
License: BSD 3-Clause
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
import sys


def plot_gap(g, suffix="", output_filename=None):
    """
    Plot the gap field.
    
    Parameters:
    -----------
    g : ndarray
        2D array containing the gap field
    suffix : str, optional
        Suffix to add to the output filename
    output_filename : str, optional
        Custom output filename. If None, uses default naming.
    """
    N0 = g.shape[0]
    gg = g.copy()
    gg[gg <= 0] = np.nan

    # Plot gap function
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(gg, cmap='viridis', origin='lower', extent=[0, 1, 0, 1], interpolation="none")
    fig.colorbar(cax, label='Gap')
    ax.set_title('Gap Function')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()
    
    if output_filename is None:
        output_filename = f"FS_gap_{suffix}.png"
    fig.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=0)
    print(f"Gap plot saved to: {output_filename}")
    plt.close(fig)


def plot_pressure(pressure, g=None, suffix="", output_filename=None):
    """
    Plot the pressure field.
    
    Parameters:
    -----------
    pressure : ndarray
        2D array containing the pressure field
    g : ndarray, optional
        2D array containing the gap field (used for masking)
    suffix : str, optional
        Suffix to add to the output filename
    output_filename : str, optional
        Custom output filename. If None, uses default naming.
    """
    N0 = pressure.shape[0]
    X, Y = np.meshgrid(np.linspace(0, 1, pressure.shape[0]), np.linspace(0, 1, pressure.shape[1]))
    
    pressure_plot = pressure.copy()
    
    # Mask pressure where gap is zero or negative
    if g is not None:
        gg = g.copy()
        gg[gg <= 0] = np.nan
        pressure_plot[np.isnan(gg)] = np.nan
    
    # Pressure plot
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(pressure_plot, cmap='jet', origin='lower', extent=[0, 1, 0, 1], interpolation="none")
    ax.contour(X, Y, pressure_plot, levels=50, colors='black', linewidths=0.5)
    
    # Make colorbar the same height as the plot
    divider = make_axes_locatable(ax)
    cax_cb = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(cax, cax=cax_cb, label='Pressure')
    ax.set_title('Pressure Distribution')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()
    
    if output_filename is None:
        output_filename = f"FS_pressure_n_{N0}_{suffix}.png"
    fig.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=0)
    print(f"Pressure plot saved to: {output_filename}")
    plt.close(fig)


def plot_flux(flux, suffix="", output_filename=None):
    """
    Plot the flux field magnitude.
    
    Parameters:
    -----------
    flux : ndarray
        3D array containing the flux field (shape: [N, M, 2])
    suffix : str, optional
        Suffix to add to the output filename
    output_filename : str, optional
        Custom output filename. If None, uses default naming.
    """
    N0 = flux.shape[0]
    
    # Compute flux magnitude
    norm_flux = np.sqrt(flux[:, :, 0]**2 + flux[:, :, 1]**2)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate dynamic range for visualization
    valid_flux = norm_flux[norm_flux > 0]
    if len(valid_flux) > 0:
        Vmin = np.nanmin(np.log10(valid_flux))
        Vmax = np.nanmax(np.log10(norm_flux))
        Vmin += 0.8 * abs(Vmax - Vmin)
        Vmax -= 0. * abs(Vmax - Vmin)
    else:
        Vmin, Vmax = None, None
    
    cax = ax.imshow(np.log10(norm_flux), cmap='nipy_spectral', origin='lower', 
                    extent=[0, 1, 0, 1], vmin=Vmin, vmax=Vmax, interpolation="kaiser")
    plt.colorbar(cax, label='Log10(Flux Magnitude)')    
    plt.show()
    
    if output_filename is None:
        output_filename = f"FS_flux_n_{N0}_{suffix}.png"
    fig.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=0)
    print(f"Flux plot saved to: {output_filename}")
    plt.close(fig)


def load_data(filename):
    """
    Load simulation data from a numpy archive file.
    
    Parameters:
    -----------
    filename : str
        Path to the .npz file containing the simulation data
        
    Returns:
    --------
    dict : Dictionary containing the loaded arrays
    """
    try:
        data = np.load(filename)
        return data
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading file '{filename}': {e}")
        sys.exit(1)


def main():
    """
    Main function for command-line interface.
    """
    parser = argparse.ArgumentParser(
        description="Plot fluid flow fields from simulation data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -f data.npz -gap -pressure -flux
  %(prog)s -f data.npz -gap
  %(prog)s -f data.npz -pressure -flux --suffix test
        """
    )
    
    parser.add_argument('-f', '--file', type=str, required=True,
                        help='Input .npz file containing simulation data')
    parser.add_argument('-gap', '--gap', action='store_true',
                        help='Plot the gap field')
    parser.add_argument('-pressure', '--pressure', action='store_true',
                        help='Plot the pressure field')
    parser.add_argument('-flux', '--flux', action='store_true',
                        help='Plot the flux field')
    parser.add_argument('--suffix', type=str, default="",
                        help='Suffix to add to output filenames')
    
    args = parser.parse_args()
    
    # Check if at least one plot type is requested
    if not (args.gap or args.pressure or args.flux):
        print("Error: At least one plot type must be specified (-gap, -pressure, or -flux)")
        parser.print_help()
        sys.exit(1)
    
    # Load data
    print(f"Loading data from: {args.file}")
    data = load_data(args.file)
    
    # Check which fields are available
    available_fields = list(data.keys())
    print(f"Available fields in file: {available_fields}")
    
    # Plot requested fields
    if args.gap:
        if 'gap' in data:
            print("\nPlotting gap field...")
            plot_gap(data['gap'], suffix=args.suffix)
        else:
            print("Warning: 'gap' field not found in data file.")
    
    if args.pressure:
        if 'pressure' in data:
            print("\nPlotting pressure field...")
            g = data.get('gap', None)  # Use gap for masking if available
            plot_pressure(data['pressure'], g=g, suffix=args.suffix)
        else:
            print("Warning: 'pressure' field not found in data file.")
    
    if args.flux:
        if 'flux' in data:
            print("\nPlotting flux field...")
            plot_flux(data['flux'], suffix=args.suffix)
        else:
            print("Warning: 'flux' field not found in data file.")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

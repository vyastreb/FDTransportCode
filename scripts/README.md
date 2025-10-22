# Fluid Flow Field Plotting Utility

A simple plotting tool for visualizing fluid simulation results (gap, pressure, and flux fields).

## Usage

### Command Line

Plot one or more fields from a saved simulation file:

```bash
python fluid_plot.py -f input.npz -gap -pressure -flux
```


### Options

- `-f, --file` : Input .npz file containing simulation data (required)
- `-gap` : Plot the gap field
- `-pressure` : Plot the pressure field
- `-flux` : Plot the flux field magnitude
- `--suffix` : Add a suffix to output filenames (optional)

### Examples

```bash
# Plot all three fields
python fluid_plot.py -f simulation_data.npz -gap -pressure -flux

# Plot only the gap field
python fluid_plot.py -f simulation_data.npz -gap

# Plot pressure and flux with a custom suffix
python fluid_plot.py -f results.npz -pressure -flux --suffix test_run
```

### As Python Module

You can also import and use the plotting functions in your own scripts:

```python
import numpy as np
from fluid_plot import plot_gap, plot_pressure, plot_flux

# Load your data
data = np.load('simulation_data.npz')

# Plot individual fields
plot_gap(data['gap'], suffix='my_analysis')
plot_pressure(data['pressure'], g=data['gap'])
plot_flux(data['flux'])
```

## Data Format

The input .npz file should contain the following arrays:

- `gap`: 2D array of gap values
- `pressure`: 2D array of pressure values
- `flux`: 3D array of flux vectors (shape: [N, M, 2])

## Output

Plots are saved as PNG files with the following naming convention:

- Gap: `FS_gap_{suffix}.png`
- Pressure: `FS_pressure_n_{N}_{suffix}.png`
- Flux: `FS_flux_n_{N}_{suffix}.png`

where `{N}` is the grid size and `{suffix}` is the optional user-provided suffix.

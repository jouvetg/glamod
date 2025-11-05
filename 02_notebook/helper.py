import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import numpy as np
import matplotlib 
from IPython.display import HTML

def animate_glacier_evolution(nc_file='output.nc'):
    ds = xr.open_dataset(nc_file)

    var_t = ds['thk']
    times = var_t.time
    x, y = var_t.coords['x'], var_t.coords['y']
    extent = [x.min(), x.max(), y.min(), y.max()]
    
    # Select 16 evenly spaced time indices
    n_times = len(times)
    time_indices = np.linspace(0, n_times - 1, 16, dtype=int)
    
    # Create 4x4 subplot grid
    fig, axes = plt.subplots(4, 4, figsize=(16, 16), dpi=100)
    axes = axes.flatten()
    
    # Find global min and max for consistent colorbar
    thk_max = float(var_t.max())
    
    for idx, time_idx in enumerate(time_indices):
        ax = axes[idx]
        thk_data = var_t.isel(time=time_idx)
        
        # Plot ice thickness
        im = ax.imshow(thk_data, origin="lower", cmap='Blues', 
                      extent=extent, vmin=0, vmax=thk_max)
        
        # Format the time value
        time_val = times[time_idx].values
        ax.set_title(f'Time: {time_val}', fontsize=10)
        ax.axis("off")
    
    # Add a single colorbar for all subplots
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, label='Ice Thickness (m)')
    
    plt.suptitle('Glacier Ice Thickness Evolution', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 0.92, 0.96])
    
    ds.close()
    plt.show()


def analyze_time_series(ts_file):
    """
    Analyze and visualize glacier time series data from output_ts.nc
    
    Parameters:
    -----------
    ts_file : str
        Path to the time series file (output_ts.nc)
    
    Returns:
    --------
    None (displays plots and prints statistics)
    """
    print(f"Reading time series data from: {ts_file}\n")
    
    # Load the time series dataset
    ds_ts = xr.open_dataset(ts_file)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot ice volume over time
    ds_ts['vol'].plot(ax=ax1, linewidth=2, color='blue')
    ax1.set_ylabel('Ice Volume [km³]', fontsize=12)
    ax1.set_xlabel('Time [years]', fontsize=12)
    ax1.set_title('Glacier Ice Volume Evolution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot ice area over time
    ds_ts['area'].plot(ax=ax2, linewidth=2, color='green')
    ax2.set_ylabel('Ice Area [km²]', fontsize=12)
    ax2.set_xlabel('Time [years]', fontsize=12)
    ax2.set_title('Glacier Ice Area Evolution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    initial_volume = float(ds_ts['vol'].isel(time=0))
    final_volume = float(ds_ts['vol'].isel(time=-1))
    volume_change = final_volume - initial_volume
    
    print(f"\nIce Volume:")
    print(f"  Initial: {initial_volume:.2f} km³")
    print(f"  Final:   {final_volume:.2f} km³")
    print(f"  Change:  {volume_change:.2f} km³")
    
    initial_area = float(ds_ts['area'].isel(time=0))
    final_area = float(ds_ts['area'].isel(time=-1))
    area_change = final_area - initial_area
    
    print(f"\nIce Area:")
    print(f"  Initial: {initial_area:.2f} km²")
    print(f"  Final:   {final_area:.2f} km²")
    print(f"  Change:  {area_change:.2f} km²")
    
    print("="*60)
    
    # Close the dataset
    ds_ts.close()
 
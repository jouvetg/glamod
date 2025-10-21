import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import numpy as np
import matplotlib 

def animate_glacier_evolution(nc_file='output.nc'):
    # Load the dataset
    ds = xr.open_dataset(nc_file)
 
    # Extract data
    var_t = ds['thk']
    var_u = ds['velsurf_mag']
    times = var_t.time
    x = var_t.coords['x']
    y = var_t.coords['y']

    extent = [x.min(), x.max(), y.min(), y.max()]
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(6, 8),dpi=100) 
    ## im = ax.contourf(x, y, var.isel(time=0), levels=50, cmap='Blues')
    im0 = ax.imshow(var_t.isel(time=0), origin="lower", cmap='binary', extent=extent) 
    im  = ax.imshow( np.where(var_t.isel(time=0)>0,var_u.isel(time=0), np.nan),  origin="lower", cmap="turbo", extent=extent, 
                     norm=matplotlib.colors.LogNorm(vmin=1, vmax=200) ) 

    cbar = plt.colorbar(im, ax=ax, label='Speed (m)')
    ax.set_title(f'Glacier Evolution\nTime: {times[0].values}')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_aspect('equal')
    ax.axis("off")
 
    # Update function for animation
    def update(frame):
        for c in ax.collections:
            c.remove()  # Remove previous contours to update with new data
#        ax.contourf(x, y, var.isel(time=frame), levels=50, cmap='Blues')
        ax.imshow(var_t.isel(time=frame), origin="lower", cmap='binary', extent=extent) 
        ax.imshow( np.where(var_t.isel(time=frame)>0, var_u.isel(time=frame), np.nan),  origin="lower", cmap="turbo", extent=extent, 
                   norm=matplotlib.colors.LogNorm(vmin=1, vmax=200) )  

        ax.set_title(f'Glacier Evolution\nTime: {times[frame].values}')
        return ax
    
    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(times), interval=200, blit=False)
    
    # Display the animation in the notebook
    display_animation = HTML(ani.to_jshtml())
    plt.close()  # Prevents static display of the plot
    ds.close()
    
    return display_animation


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
 
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import numpy as np
import matplotlib
import pandas as pd

def plot_input_data(file = "input_saved.nc"):
    import xarray as xr
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm  # Import LogNorm for logarithmic scaling

    df = pd.read_csv('Samples_2024.csv')
    id = list(df.iloc[:, 0])
    sp = np.array(df.iloc[:, 3:])  # Get the rest of the columns (numeric values)

    # Open the NetCDF file 
    dataset = xr.open_dataset(file)

    # Read variables
    topg = (dataset['usurf'])
    thk =  (dataset['thk']) 
    x =    (dataset['x'])
    y =    (dataset['y'])

    extent = [x.min(), x.max(), y.min(), y.max()]

    # Set up the figure and subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))  # Change to 3 columns for 3 variables

    # Plot the first variable (usurf)
    c1 = axes[0].imshow(topg, cmap='viridis', origin = 'lower', aspect='auto',extent=extent)  # Display usurf with 'viridis' colormap
    axes[0].set_title('Surface Variable (usurf)')
    axes[0].set_xlabel('X-axis Label')  # Replace with the actual x-axis label
    axes[0].set_ylabel('Y-axis Label')  # Replace with the actual y-axis label
    axes[0].invert_yaxis()  # Invert y-axis to match the original orientation
    axes[0].set_aspect('equal')  # Set aspect ratio to be equal
    axes[0].scatter(sp[:, 0], sp[:, 1], c='r', marker='o', s=10)
    for ii in range(sp.shape[0]):
       axes[0].text(sp[ii, 0], sp[:, 1], id[ii], fontsize=7, c='r') 
    axes[0].invert_yaxis() 
    plt.colorbar(c1, ax=axes[0], orientation='vertical', label='usurf units')  # Add colorbar

    # Plot the second variable (thk) with 'jet' colormap
    c2 = axes[1].imshow(thk, cmap='jet', origin = 'lower', aspect='auto',extent=extent)  # Display thk with 'jet' colormap
    axes[1].set_title('Ice Thickness Variable (thk)')
    axes[1].set_xlabel('X-axis Label')  # Replace with the actual x-axis label
    axes[1].set_ylabel('Y-axis Label')  # Replace with the actual y-axis label
    axes[1].invert_yaxis()  # Invert y-axis to match the original orientation
    axes[1].set_aspect('equal')  # Set aspect ratio to be equal
    axes[1].scatter(sp[:, 0], sp[:, 1], c='r', marker='o', s=10)
    for ii in range(sp.shape[0]):
       axes[1].text(sp[ii, 0], sp[:, 1], id[ii], fontsize=7, c='r') 
    axes[1].invert_yaxis() 
    plt.colorbar(c2, ax=axes[1], orientation='vertical', label='Ice Thickness (units)')  # Add colorbar

    # Adjust layout
    plt.tight_layout()

    # Show the plots
    plt.show()

    # Optionally, close the dataset explicitly (not necessary in most cases)
    dataset.close()


def animate_glacier_evolution(nc_file='output.nc', variable='thk', log_scale=False, show_samples=True):
    """
    Animate glacier evolution in a 4x4 grid showing 16 time steps
    
    Parameters:
    -----------
    nc_file : str
        Path to the output NetCDF file (output.nc)
    variable : str
        Variable to plot ('thk' for ice thickness, 'velbar_mag' for velocity magnitude)
    log_scale : bool
        If True, use logarithmic color scale (useful for velocity)
    show_samples : bool
        If True, overlay sample locations from Samples_2024.csv
    
    Returns:
    --------
    None (displays plots)
    """
    ds = xr.open_dataset(nc_file)

    # Load sample data if available
    samples_x, samples_y, sample_ids = None, None, None
    if show_samples:
        try:
            df = pd.read_csv('Samples_2024.csv')
            sample_ids = list(df.iloc[:, 0])
            sp = np.array(df.iloc[:, 3:])
            samples_x, samples_y = sp[:, 0], sp[:, 1]
        except:
            show_samples = False

    var_t = ds[variable]
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
    var_max = float(var_t.max())
    var_min = float(var_t.min())
    
    # For velocity in log scale, set appropriate min value
    if log_scale and variable == 'velbar_mag':
        # Find minimum positive value
        var_data_all = var_t.values
        var_positive = var_data_all[var_data_all > 0]
        if len(var_positive) > 0:
            var_min = max(0.1, float(np.percentile(var_positive, 1)))  # Use 1st percentile to avoid outliers
        else:
            var_min = 0.1
        var_max = float(np.percentile(var_data_all, 99))  # Use 99th percentile
    
    # Choose colormap based on variable
    if variable == 'thk':
        cmap = 'Blues'
        cbar_label = 'Ice Thickness (m)'
        title = 'Glacier Ice Thickness Evolution'
    elif variable == 'velbar_mag':
        cmap = 'plasma'
        cbar_label = 'Velocity Magnitude (m/yr)' + (' (log scale)' if log_scale else '')
        title = 'Glacier Velocity Evolution' + (' (Log Scale)' if log_scale else '')
    else:
        cmap = 'viridis'
        cbar_label = variable
        title = f'{variable} Evolution'
    
    # Set normalization
    if log_scale:
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=var_min, vmax=var_max)
    else:
        norm = None
        
    for idx, time_idx in enumerate(time_indices):
        ax = axes[idx]
        var_data = var_t.isel(time=time_idx)
        
        # For log scale velocity, mask zeros
        if log_scale and variable == 'velbar_mag':
            var_data = var_data.where(var_data > 0)
        
        # Plot variable
        if log_scale:
            im = ax.imshow(var_data, origin="lower", cmap=cmap, 
                          extent=extent, norm=norm)
        else:
            im = ax.imshow(var_data, origin="lower", cmap=cmap, 
                          extent=extent, vmin=var_min if variable != 'thk' else 0, 
                          vmax=var_max)
        
        # Overlay samples if requested
        if show_samples and samples_x is not None:
            ax.scatter(samples_x, samples_y, c='red', marker='o', s=20, edgecolors='black', linewidths=0.5)
        
        # Format the time value
        time_val = times[time_idx].values
        ax.set_title(f'Time: {time_val}', fontsize=10)
        ax.axis("off")
    
    # Add a single colorbar for all subplots
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, label=cbar_label)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
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


def plot_thk_at_sample(file = 'output.nc'):

    from scipy.interpolate import RectBivariateSpline
    import netCDF4 as nc
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    df = pd.read_csv('Samples_2024.csv')
    ID = df.iloc[:, 0]
    sp = np.array(df.iloc[:, 3:])  # Get the rest of the columns (numeric values)
    
    # Step 1: Open the NetCDF file
    dataset = nc.Dataset(file, 'r')  # 'r' for read-only mode 
    thk = np.squeeze(dataset.variables['thk']).astype("float32")  # Access the variable 'thk' 
    times = dataset.variables['time'][:]  # read all time values
    x = dataset.variables['x'][:]  # read all x coordinates
    y = dataset.variables['y'][:]  # read all y coordinates 
    dataset.close()

    ##################

    thk_i = []
    for it in range(0, thk.shape[0]): 
        f = RectBivariateSpline(x, y, np.transpose(thk[it,:,:]))
        thk_i.append(f(sp[:, 0], sp[:, 1], grid=False))
    
    thk_i = np.stack(thk_i, axis=0) 

    ##################
    
    import matplotlib.pyplot as plt
    
    fig, axs = plt.subplots(nrows=sp.shape[0], ncols=1, figsize=(12, 12), dpi=100)
    
    for i in range(sp.shape[0]):
        ax = axs[i]
        ax.plot(times, thk_i[:, i], 'k', markersize=5)  # Plot the points
        ax.set_title(f'Sample {ID[i]}', fontsize=12)  # Title for each subplot
        ax.set_xlabel('Time (years)', fontsize=10)
        ax.set_ylabel('Ice Thickness (m)', fontsize=10)
        ax.set_ylim([0, 300])
        ax.set_xticks(np.arange(times[0], times[-1] + 1, 20))
    
    plt.subplots_adjust(hspace=0.5)  # Adjust the spacing between subplots
    
    plt.show()

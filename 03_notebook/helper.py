import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import numpy as np
import matplotlib 
from IPython.display import HTML

def animate_glacier_evolution(nc_file='output.nc', variable='thk', log_scale=False):
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
    
    Returns:
    --------
    None (displays plots)
    """
    ds = xr.open_dataset(nc_file)

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


def plot_flowline_evolution(nc_file='output.nc', flowline_file='data/RGI2000-v7.0-G-11-01706/downstream_line.pkl', 
                           glacier_grid_file='data/RGI2000-v7.0-G-11-01706/glacier_grid.json'):
    """
    Plot glacier thickness evolution along the flowline in a 4x4 grid
    
    Parameters:
    -----------
    nc_file : str
        Path to the output NetCDF file (output.nc)
    flowline_file : str
        Path to the flowline pickle file (downstream_line.pkl)
    glacier_grid_file : str
        Path to the glacier grid JSON file (glacier_grid.json)
    
    Returns:
    --------
    None (displays plots)
    """
    import pickle
    import gzip
    import json
    from scipy.interpolate import RegularGridInterpolator
    
    print(f"Reading glacier evolution data from: {nc_file}")
    print(f"Reading flowline data from: {flowline_file}\n")
    
    # Load the OGGM glacier grid info
    try:
        with open(glacier_grid_file, 'r') as f:
            grid_info = json.load(f)
        x0, y0 = grid_info['x0y0']
        dx_oggm, dy_oggm = grid_info['dxdy']
        print(f"OGGM grid: origin=({x0:.2f}, {y0:.2f}), resolution=({dx_oggm}, {dy_oggm})")
    except Exception as e:
        print(f"Error loading glacier grid info: {e}")
        return
    
    # Load the flowline coordinates (in OGGM grid indices)
    try:
        dl = pickle.load(gzip.open(flowline_file, 'rb'))
        flowline = dl['downstream_line']
        flowline_coords = np.array(flowline.coords)
        print(f"Flowline loaded: {len(flowline_coords)} points\n")
    except Exception as e:
        print(f"Error loading flowline: {e}")
        return
    
    # Load the glacier evolution dataset (IGM output)
    ds = xr.open_dataset(nc_file)
    
    # Get IGM grid coordinates
    x_coords = ds['x'].values
    y_coords = ds['y'].values
    times = ds['time'].values
    
    # Convert flowline from OGGM grid indices to real coordinates
    flowline_x = x0 + flowline_coords[:, 0] * dx_oggm
    flowline_y = y0 + flowline_coords[:, 1] * dy_oggm
    
    # Extend the flowline upward to capture more of the glacier
    # Start from the last (highest) point and trace upward following the glacier
    print("Extending flowline to capture upper glacier...")
    
    # Get initial ice thickness to find glacier extent
    thk_initial = ds['thk'].isel(time=0).values
    usurf_initial = ds['usurf'].isel(time=0).values
    
    # Find the direction to extend (roughly northward/upward)
    last_x, last_y = flowline_x[0], flowline_y[0]  # Start point (highest existing point)
    
    # Sample points upward from the last flowline point
    extension_points_x = []
    extension_points_y = []
    
    # Define search direction (continue in the direction of the flowline)
    if len(flowline_x) > 1:
        dx_dir = flowline_x[0] - flowline_x[1]
        dy_dir = flowline_y[0] - flowline_y[1]
        norm = np.sqrt(dx_dir**2 + dy_dir**2)
        if norm > 0:
            dx_dir /= norm
            dy_dir /= norm
        else:
            dx_dir, dy_dir = 0, 1  # Default to north
    else:
        dx_dir, dy_dir = 0, 1
    
    # Extend upward with small steps
    step_size = 100  # 100m steps
    current_x, current_y = last_x, last_y
    
    from scipy.interpolate import RegularGridInterpolator
    interp_thk = RegularGridInterpolator((y_coords, x_coords), thk_initial,
                                         bounds_error=False, fill_value=0)
    
    for i in range(100):  # Max 100 steps = 10km extension
        # Take a step in the direction
        test_x = current_x + dx_dir * step_size
        test_y = current_y + dy_dir * step_size
        
        # Check if there's ice at this location
        thk_val = interp_thk(np.array([[test_y, test_x]]))[0]
        
        if thk_val > 1.0:  # At least 1m of ice
            extension_points_x.append(test_x)
            extension_points_y.append(test_y)
            current_x, current_y = test_x, test_y
        else:
            # No more ice, stop extending
            break
    
    # Prepend extension points to flowline (reverse order since we went upward)
    if extension_points_x:
        flowline_x = np.concatenate([extension_points_x[::-1], flowline_x])
        flowline_y = np.concatenate([extension_points_y[::-1], flowline_y])
        print(f"  Extended flowline by {len(extension_points_x)} points ({len(extension_points_x)*step_size/1000:.2f} km)")
    else:
        print(f"  No extension needed")
    
    print(f"  Total flowline points: {len(flowline_x)}\n")
    
    # Calculate distance along flowline
    distances = np.zeros(len(flowline_x))
    for i in range(1, len(flowline_x)):
        distances[i] = distances[i-1] + np.sqrt(
            (flowline_x[i] - flowline_x[i-1])**2 + 
            (flowline_y[i] - flowline_y[i-1])**2
        )
    
    distances_km = distances / 1000.0  # Convert to km
    
    # Select 16 evenly spaced time indices
    n_times = len(times)
    time_indices = np.linspace(0, n_times - 1, 16, dtype=int)
    
    # Create 4x4 subplot grid
    fig, axes = plt.subplots(4, 4, figsize=(16, 16), dpi=100)
    axes = axes.flatten()
    
    # Find global min and max for consistent y-axis
    all_topg = []
    all_usurf = []
    all_vel = []
    
    for time_idx in time_indices:
        # Extract data at this time
        topg_data = ds['topg'].isel(time=time_idx).values
        usurf_data = ds['usurf'].isel(time=time_idx).values
        velbar_mag_data = ds['velbar_mag'].isel(time=time_idx).values
        
        # Create interpolators
        interp_topg = RegularGridInterpolator((y_coords, x_coords), topg_data, 
                                              bounds_error=False, fill_value=None)
        interp_usurf = RegularGridInterpolator((y_coords, x_coords), usurf_data,
                                               bounds_error=False, fill_value=None)
        interp_vel = RegularGridInterpolator((y_coords, x_coords), velbar_mag_data,
                                             bounds_error=False, fill_value=None)
        
        # Sample along flowline
        points = np.column_stack([flowline_y, flowline_x])
        topg_profile = interp_topg(points)
        usurf_profile = interp_usurf(points)
        vel_profile = interp_vel(points)
        
        all_topg.extend(topg_profile[~np.isnan(topg_profile)])
        all_usurf.extend(usurf_profile[~np.isnan(usurf_profile)])
        all_vel.extend(vel_profile[~np.isnan(vel_profile)])
    
    y_min_global = min(all_topg) - 50
    y_max_global = max(all_usurf) + 50
    vel_max_global = max(all_vel) if all_vel else 100
    
    # Plot each time step
    for idx, time_idx in enumerate(time_indices):
        ax = axes[idx]
        
        # Extract data at this time
        topg_data = ds['topg'].isel(time=time_idx).values
        usurf_data = ds['usurf'].isel(time=time_idx).values
        thk_data = ds['thk'].isel(time=time_idx).values
        velbar_mag_data = ds['velbar_mag'].isel(time=time_idx).values
        
        # Create interpolators
        interp_topg = RegularGridInterpolator((y_coords, x_coords), topg_data,
                                              bounds_error=False, fill_value=None)
        interp_usurf = RegularGridInterpolator((y_coords, x_coords), usurf_data,
                                               bounds_error=False, fill_value=None)
        interp_thk = RegularGridInterpolator((y_coords, x_coords), thk_data,
                                             bounds_error=False, fill_value=None)
        interp_vel = RegularGridInterpolator((y_coords, x_coords), velbar_mag_data,
                                             bounds_error=False, fill_value=None)
        
        # Sample along flowline
        points = np.column_stack([flowline_y, flowline_x])
        topg_profile = interp_topg(points)
        usurf_profile = interp_usurf(points)
        thk_profile = interp_thk(points)
        vel_profile = interp_vel(points)
        
        # Plot bedrock
        ax.plot(distances_km, topg_profile, 'k-', linewidth=2, label='Bedrock')
        
        # Plot ice thickness as filled area
        ax.fill_between(distances_km, topg_profile, usurf_profile, 
                       where=(thk_profile > 0.1), alpha=0.6, color='steelblue',
                       label='Ice')
        
        # Plot surface
        ax.plot(distances_km, usurf_profile, 'b-', linewidth=1.5, label='Surface')
        
        # Format plot
        time_val = times[time_idx]
        ax.set_title(f'Year: {time_val:.0f}', fontsize=10, fontweight='bold')
        ax.set_ylim(y_min_global, y_max_global)
        ax.set_xlabel('Distance along flowline (km)', fontsize=9)
        ax.set_ylabel('Elevation (m)', fontsize=9, color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax.grid(True, alpha=0.3)
        
        # Create secondary y-axis for velocity
        ax2 = ax.twinx()
        ax2.plot(distances_km, vel_profile, 'r-', linewidth=1.5, alpha=0.7, label='Velocity')
        ax2.set_ylabel('Velocity (m/yr)', fontsize=9, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(0, vel_max_global * 1.1)
        
        # Add legend only to first subplot
        if idx == 0:
            # Combine legends from both axes
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
    
    plt.suptitle('Glacier Cross-Section Evolution Along Flowline', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    ds.close()
    plt.show()
    
    print(f"\nFlowline length: {distances_km[-1]:.2f} km")
    print(f"Time period: {times[0]:.0f} - {times[-1]:.0f}")


def plot_flowline_velocity(nc_file='output.nc', flowline_file='data/RGI2000-v7.0-G-11-01706/downstream_line.pkl', 
                           glacier_grid_file='data/RGI2000-v7.0-G-11-01706/glacier_grid.json'):
    """
    Plot glacier velocity evolution along the flowline in a 4x4 grid (log scale)
    Shows velocity instead of ice thickness for better visualization of flow dynamics
    
    Parameters:
    -----------
    nc_file : str
        Path to the output NetCDF file (output.nc)
    flowline_file : str
        Path to the flowline pickle file (downstream_line.pkl)
    glacier_grid_file : str
        Path to the glacier grid JSON file (glacier_grid.json)
    
    Returns:
    --------
    None (displays plots)
    """
    import pickle
    import gzip
    import json
    from scipy.interpolate import RegularGridInterpolator
    
    print(f"Reading glacier evolution data from: {nc_file}")
    print(f"Reading flowline data from: {flowline_file}\n")
    
    # Load the OGGM glacier grid info
    try:
        with open(glacier_grid_file, 'r') as f:
            grid_info = json.load(f)
        x0, y0 = grid_info['x0y0']
        dx_oggm, dy_oggm = grid_info['dxdy']
        print(f"OGGM grid: origin=({x0:.2f}, {y0:.2f}), resolution=({dx_oggm}, {dy_oggm})")
    except Exception as e:
        print(f"Error loading glacier grid info: {e}")
        return
    
    # Load the flowline coordinates (in OGGM grid indices)
    try:
        dl = pickle.load(gzip.open(flowline_file, 'rb'))
        flowline = dl['downstream_line']
        flowline_coords = np.array(flowline.coords)
        print(f"Flowline loaded: {len(flowline_coords)} points\n")
    except Exception as e:
        print(f"Error loading flowline: {e}")
        return
    
    # Load the glacier evolution dataset (IGM output)
    ds = xr.open_dataset(nc_file)
    
    # Get IGM grid coordinates
    x_coords = ds['x'].values
    y_coords = ds['y'].values
    times = ds['time'].values
    
    # Convert flowline from OGGM grid indices to real coordinates
    flowline_x = x0 + flowline_coords[:, 0] * dx_oggm
    flowline_y = y0 + flowline_coords[:, 1] * dy_oggm
    
    # Extend the flowline upward to capture more of the glacier
    print("Extending flowline to capture upper glacier...")
    
    # Get initial ice thickness to find glacier extent
    thk_initial = ds['thk'].isel(time=0).values
    
    # Find the direction to extend
    last_x, last_y = flowline_x[0], flowline_y[0]
    
    # Define search direction
    if len(flowline_x) > 1:
        dx_dir = flowline_x[0] - flowline_x[1]
        dy_dir = flowline_y[0] - flowline_y[1]
        norm = np.sqrt(dx_dir**2 + dy_dir**2)
        if norm > 0:
            dx_dir /= norm
            dy_dir /= norm
        else:
            dx_dir, dy_dir = 0, 1
    else:
        dx_dir, dy_dir = 0, 1
    
    # Extend upward
    step_size = 100
    current_x, current_y = last_x, last_y
    extension_points_x = []
    extension_points_y = []
    
    from scipy.interpolate import RegularGridInterpolator
    interp_thk = RegularGridInterpolator((y_coords, x_coords), thk_initial,
                                         bounds_error=False, fill_value=0)
    
    for i in range(100):
        test_x = current_x + dx_dir * step_size
        test_y = current_y + dy_dir * step_size
        thk_val = interp_thk(np.array([[test_y, test_x]]))[0]
        
        if thk_val > 1.0:
            extension_points_x.append(test_x)
            extension_points_y.append(test_y)
            current_x, current_y = test_x, test_y
        else:
            break
    
    # Prepend extension points
    if extension_points_x:
        flowline_x = np.concatenate([extension_points_x[::-1], flowline_x])
        flowline_y = np.concatenate([extension_points_y[::-1], flowline_y])
        print(f"  Extended flowline by {len(extension_points_x)} points ({len(extension_points_x)*step_size/1000:.2f} km)")
    else:
        print(f"  No extension needed")
    
    print(f"  Total flowline points: {len(flowline_x)}\n")
    
    # Calculate distance along flowline
    distances = np.zeros(len(flowline_x))
    for i in range(1, len(flowline_x)):
        distances[i] = distances[i-1] + np.sqrt(
            (flowline_x[i] - flowline_x[i-1])**2 + 
            (flowline_y[i] - flowline_y[i-1])**2
        )
    
    distances_km = distances / 1000.0
    
    # Select 16 evenly spaced time indices
    n_times = len(times)
    time_indices = np.linspace(0, n_times - 1, 16, dtype=int)
    
    # Create 4x4 subplot grid
    fig, axes = plt.subplots(4, 4, figsize=(16, 16), dpi=100)
    axes = axes.flatten()
    
    # Find global min and max
    all_topg = []
    all_vel = []
    
    for time_idx in time_indices:
        topg_data = ds['topg'].isel(time=time_idx).values
        velbar_mag_data = ds['velbar_mag'].isel(time=time_idx).values
        
        interp_topg = RegularGridInterpolator((y_coords, x_coords), topg_data, 
                                              bounds_error=False, fill_value=None)
        interp_vel = RegularGridInterpolator((y_coords, x_coords), velbar_mag_data,
                                             bounds_error=False, fill_value=None)
        
        points = np.column_stack([flowline_y, flowline_x])
        topg_profile = interp_topg(points)
        vel_profile = interp_vel(points)
        
        all_topg.extend(topg_profile[~np.isnan(topg_profile)])
        all_vel.extend(vel_profile[(~np.isnan(vel_profile)) & (vel_profile > 0)])
    
    y_min_global = min(all_topg) - 100
    vel_min_global = max(0.1, min(all_vel)) if all_vel else 0.1  # Min for log scale
    vel_max_global = max(all_vel) * 2 if all_vel else 100  # Max for log scale
    
    # Plot each time step
    for idx, time_idx in enumerate(time_indices):
        ax = axes[idx]
        
        # Extract data
        topg_data = ds['topg'].isel(time=time_idx).values
        velbar_mag_data = ds['velbar_mag'].isel(time=time_idx).values
        
        # Create interpolators
        interp_topg = RegularGridInterpolator((y_coords, x_coords), topg_data,
                                              bounds_error=False, fill_value=None)
        interp_vel = RegularGridInterpolator((y_coords, x_coords), velbar_mag_data,
                                             bounds_error=False, fill_value=None)
        
        # Sample along flowline
        points = np.column_stack([flowline_y, flowline_x])
        topg_profile = interp_topg(points)
        vel_profile = interp_vel(points)
        
        # Replace zero velocities with small value for log scale
        vel_profile_log = np.where(vel_profile > 0.01, vel_profile, np.nan)
        
        # Plot bedrock
        ax.plot(distances_km, topg_profile, 'k-', linewidth=2, label='Bedrock')
        
        # Create secondary y-axis for velocity (log scale)
        ax2 = ax.twinx()
        ax2.semilogy(distances_km, vel_profile_log, 'r-', linewidth=2, label='Velocity (log)')
        ax2.set_ylabel('Velocity (m/yr, log scale)', fontsize=9, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(vel_min_global, vel_max_global)
        ax2.grid(False)
        
        # Format plot
        time_val = times[time_idx]
        ax.set_title(f'Year: {time_val:.0f}', fontsize=10, fontweight='bold')
        ax.set_ylim(y_min_global, y_min_global + (vel_max_global - vel_min_global) * 50)  # Scale based on velocity range
        ax.set_xlabel('Distance along flowline (km)', fontsize=9)
        ax.set_ylabel('Elevation (m)', fontsize=9, color='black')
        ax.tick_params(axis='y', labelcolor='black')
        ax.grid(True, alpha=0.3)
        
        # Add legend only to first subplot
        if idx == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
    
    plt.suptitle('Glacier Velocity Evolution Along Flowline (Log Scale)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    ds.close()
    plt.show()
    
    print(f"\nFlowline length: {distances_km[-1]:.2f} km")
    print(f"Velocity range: {vel_min_global:.2f} to {vel_max_global:.2f} m/yr")
    print(f"Time period: {times[0]:.0f} - {times[-1]:.0f}")



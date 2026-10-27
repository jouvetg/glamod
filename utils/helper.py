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


def plot_flowline_evolution(nc_file='output.nc', rgi_id=None, data_dir='data', flowline_file=None, glacier_grid_file=None):
    """
    Plot glacier thickness evolution along the flowline in a 4x4 grid
    
    Parameters:
    -----------
    nc_file : str
        Path to the output NetCDF file (output.nc)
    rgi_id : str
        RGI ID of the glacier; the OGGM flowline is read from data_dir/rgi_id/
    flowline_file : str
        Path to the flowline pickle file (downstream_line.pkl), overrides rgi_id
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
    flowline_file, glacier_grid_file = _flowline_files(rgi_id, data_dir, flowline_file, glacier_grid_file)
    
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


def plot_flowline_velocity(nc_file='output.nc', rgi_id=None, data_dir='data', flowline_file=None, glacier_grid_file=None):
    """
    Plot glacier velocity evolution along the flowline in a 4x4 grid (log scale)
    Shows velocity instead of ice thickness for better visualization of flow dynamics
    
    Parameters:
    -----------
    nc_file : str
        Path to the output NetCDF file (output.nc)
    rgi_id : str
        RGI ID of the glacier; the OGGM flowline is read from data_dir/rgi_id/
    flowline_file : str
        Path to the flowline pickle file (downstream_line.pkl), overrides rgi_id
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
    flowline_file, glacier_grid_file = _flowline_files(rgi_id, data_dir, flowline_file, glacier_grid_file)
    
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


def compare_modeled_observed_velocity(input_file='data/input.nc', output_file='output.nc', year=2020):
    """
    Compare modeled and observed surface velocity
    
    Parameters:
    -----------
    input_file : str
        Path to the input NetCDF file containing observed velocities (uvelsurfobs, vvelsurfobs)
    output_file : str
        Path to the output NetCDF file containing modeled velocities (uvelsurf, vvelsurf)
    year : int
        Year to extract from the modeled output
    
    Returns:
    --------
    None (displays plots)
    """
    print(f"Comparing modeled (year {year}) vs observed velocity\n")
    
    # Load input data with observed velocities
    ds_input = xr.open_dataset(input_file)
    
    # Load output data with modeled velocities
    ds_output = xr.open_dataset(output_file)
    # observations on the model grid (the run may use a coarser grid: inputs.local.coarsening)
    ds_input = ds_input.interp(x=ds_output['x'], y=ds_output['y'])
    
    # Calculate observed velocity magnitude from components
    uvelsurfobs = ds_input['uvelsurfobs'].values
    vvelsurfobs = ds_input['vvelsurfobs'].values
    vel_obs_mag = np.sqrt(uvelsurfobs**2 + vvelsurfobs**2)
    
    # Get modeled velocity at the specified year
    # Find the time index closest to the specified year
    times = ds_output['time'].values
    time_idx = np.argmin(np.abs(times - year))
    actual_year = times[time_idx]
    
    print(f"Using modeled data from year: {actual_year:.1f}")
    
    # Extract modeled velocity components
    uvelsurf = ds_output['uvelsurf'].isel(time=time_idx).values
    vvelsurf = ds_output['vvelsurf'].isel(time=time_idx).values
    vel_mod_mag = np.sqrt(uvelsurf**2 + vvelsurf**2)
    
    # Also get the ice thickness mask
    thk = ds_output['thk'].isel(time=time_idx).values
    ice_mask = thk > 1.0  # At least 1m of ice
    
    # Get coordinates
    x_coords = ds_output['x'].values
    y_coords = ds_output['y'].values
    extent = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()]
    
    # Mask velocities where there's no ice
    vel_obs_mag_masked = np.where(ice_mask, vel_obs_mag, np.nan)
    vel_mod_mag_masked = np.where(ice_mask, vel_mod_mag, np.nan)
    
    # Calculate difference
    vel_diff = vel_mod_mag - vel_obs_mag
    vel_diff_masked = np.where(ice_mask, vel_diff, np.nan)
    
    # Create figure with 3 subplots in a row
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Find global min/max for consistent color scales
    vmax = max(np.nanmax(vel_obs_mag_masked), np.nanmax(vel_mod_mag_masked))
    vmin = 0
    
    # Use log scale for better visualization
    from matplotlib.colors import LogNorm
    norm = LogNorm(vmin=max(0.1, np.nanpercentile(vel_obs_mag_masked, 1)), 
                   vmax=np.nanpercentile(vel_obs_mag_masked, 99))
    
    # Plot 1: Observed velocity
    ax1 = axes[0]
    im1 = ax1.imshow(vel_obs_mag_masked, origin='lower', cmap='plasma', 
                     extent=extent, norm=norm)
    ax1.set_title('Observed Surface Velocity', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    plt.colorbar(im1, ax=ax1, label='Velocity (m/yr)', fraction=0.046, pad=0.04)
    
    # Plot 2: Modeled velocity
    ax2 = axes[1]
    im2 = ax2.imshow(vel_mod_mag_masked, origin='lower', cmap='plasma', 
                     extent=extent, norm=norm)
    ax2.set_title(f'Modeled Surface Velocity (Year {actual_year:.0f})', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    plt.colorbar(im2, ax=ax2, label='Velocity (m/yr)', fraction=0.046, pad=0.04)
    
    # Plot 3: Difference (modeled - observed)
    ax3 = axes[2]
    diff_max = np.nanpercentile(np.abs(vel_diff_masked), 95)
    im3 = ax3.imshow(vel_diff_masked, origin='lower', cmap='RdBu_r', 
                     extent=extent, vmin=-diff_max, vmax=diff_max)
    ax3.set_title('Difference (Modeled - Observed)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    plt.colorbar(im3, ax=ax3, label='Velocity Difference (m/yr)', fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate and print statistics
    # Flatten and remove NaN values
    vel_obs_flat = vel_obs_mag_masked[ice_mask].flatten()
    vel_mod_flat = vel_mod_mag_masked[ice_mask].flatten()
    
    # Remove any remaining NaNs
    valid_mask = ~(np.isnan(vel_obs_flat) | np.isnan(vel_mod_flat))
    vel_obs_flat = vel_obs_flat[valid_mask]
    vel_mod_flat = vel_mod_flat[valid_mask]
    
    # Calculate statistics
    from scipy.stats import pearsonr
    correlation = pearsonr(vel_obs_flat, vel_mod_flat)[0]
    rmse = np.sqrt(np.mean((vel_mod_flat - vel_obs_flat)**2))
    bias = np.mean(vel_mod_flat - vel_obs_flat)
    
    # Print statistics
    print("\n" + "="*60)
    print("VELOCITY COMPARISON STATISTICS")
    print("="*60)
    print(f"Correlation coefficient (R): {correlation:.3f}")
    print(f"RMSE: {rmse:.2f} m/yr")
    print(f"Bias (modeled - observed): {bias:.2f} m/yr")
    print(f"Number of points: {len(vel_obs_flat)}")
    print(f"\nObserved velocity range: {vel_obs_flat.min():.2f} - {vel_obs_flat.max():.2f} m/yr")
    print(f"Modeled velocity range: {vel_mod_flat.min():.2f} - {vel_mod_flat.max():.2f} m/yr")
    print("="*60)
    
    # Close datasets
    ds_input.close()
    ds_output.close()





# ---------------------------------------------------------------------------
# Helpers added for the 2026 course (IGM 3.2)
# ---------------------------------------------------------------------------

def _flowline_files(rgi_id, data_dir, flowline_file, glacier_grid_file):
    """Resolve the OGGM flowline / grid files from an RGI ID (or explicit paths)."""
    import os
    if flowline_file is None or glacier_grid_file is None:
        if rgi_id is None:
            raise ValueError("Give rgi_id='RGI2000-v7.0-G-11-xxxxx' (or flowline_file and glacier_grid_file)")
        flowline_file = flowline_file or os.path.join(data_dir, rgi_id, 'downstream_line.pkl')
        glacier_grid_file = glacier_grid_file or os.path.join(data_dir, rgi_id, 'glacier_grid.json')
    return flowline_file, glacier_grid_file


def latest_run(outputs='outputs'):
    """Return the most recent IGM run folder (the one containing output.nc) under `outputs`."""
    import glob, os
    runs = [os.path.dirname(f) for f in glob.glob(os.path.join(outputs, '**', 'output.nc'), recursive=True)]
    if not runs:
        raise FileNotFoundError(f"No output.nc found under '{outputs}'. Run IGM first.")
    run = max(runs, key=lambda d: os.path.getmtime(os.path.join(d, 'output.nc')))
    print(f"Latest run: {run}")
    return run


def show_dashboard(run_dir, n=4):
    """Display `n` evenly spaced dashboard_XXXX.png frames written by the live_dashboard output."""
    import glob, os
    frames = sorted(glob.glob(os.path.join(run_dir, 'dashboard_*.png')))
    if not frames:
        print("No dashboard frames found (is live_dashboard with headless: true in the outputs?)")
        return
    idx = np.unique(np.linspace(0, len(frames) - 1, min(n, len(frames))).astype(int))
    fig, axes = plt.subplots(1, len(idx), figsize=(5 * len(idx), 5))
    for ax, i in zip(np.atleast_1d(axes), idx):
        ax.imshow(plt.imread(frames[i]))
        ax.set_title(os.path.basename(frames[i]))
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_input_data(file='data/input.nc'):
    """Plot surface elevation, ice thickness and (if available) observed surface speed of an IGM input file."""
    ds = xr.open_dataset(file)
    x, y = ds['x'], ds['y']
    extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]
    panels = [('usurf', 'Surface elevation (m)', 'terrain', None),
              ('thk', 'Ice thickness (m)', 'Blues', None)]
    if 'uvelsurfobs' in ds and 'vvelsurfobs' in ds:
        speed = np.sqrt(ds['uvelsurfobs']**2 + ds['vvelsurfobs']**2)
        panels.append((speed, 'Observed surface speed (m/yr)', 'plasma', None))
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5))
    for ax, (var, label, cmap, _) in zip(axes, panels):
        data = ds[var] if isinstance(var, str) else var
        if not isinstance(var, str) or var == 'thk':
            data = data.where(ds['thk'] > 1) if 'thk' in ds else data
        im = ax.imshow(data, origin='lower', extent=extent, cmap=cmap)
        ax.set_title(label)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        plt.colorbar(im, ax=ax, shrink=0.8)
    print(f"Grid: {ds.sizes['x']} x {ds.sizes['y']} cells, dx = {float(x[1] - x[0]):.0f} m")
    if 'thk' in ds:
        dx = float(x[1] - x[0])
        print(f"Initial ice volume: {float(ds['thk'].sum()) * dx**2 / 1e9:.2f} km3, "
              f"area: {float((ds['thk'] > 1).sum()) * dx**2 / 1e6:.1f} km2")
    plt.tight_layout()
    plt.show()
    ds.close()


def plot_smb_simple(ela=3000.0, gradabl=0.009, gradacc=0.005, accmax=2.0, zmin=1500, zmax=4500):
    """Plot the elevation-dependent SMB used by `smb: method: simple`."""
    z = np.linspace(zmin, zmax, 300)
    smb = np.where(z < ela, gradabl, gradacc) * (z - ela)
    smb = np.minimum(smb, accmax)
    plt.figure(figsize=(5, 5))
    plt.plot(smb, z)
    plt.axhline(ela, color='r', ls='--', label=f'ELA = {ela:.0f} m')
    plt.axvline(0, color='k', lw=0.5)
    plt.xlabel('SMB (m ice eq./yr)')
    plt.ylabel('Elevation (m)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def plot_climate(rgi_id, data_dir='data', window_years=10):
    """Plot the OGGM historical climate (annual mean temperature, annual precipitation) of a glacier."""
    import os
    ds = xr.open_dataset(os.path.join(data_dir, rgi_id, 'climate_historical.nc'))
    temp = ds['temp'].groupby('time.year').mean()
    prcp = ds['prcp'].groupby('time.year').sum()
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(temp.year, temp, color='tab:red', alpha=0.3)
    ax1.plot(temp.year, temp.rolling(year=window_years, center=True).mean(), color='tab:red', lw=2)
    ax1.set_ylabel(f"Mean annual temperature (°C)\nat {ds.attrs.get('ref_hgt', '?')} m", color='tab:red')
    ax2 = ax1.twinx()
    ax2.plot(prcp.year, prcp, color='tab:blue', alpha=0.3)
    ax2.plot(prcp.year, prcp.rolling(year=window_years, center=True).mean(), color='tab:blue', lw=2)
    ax2.set_ylabel('Annual precipitation (kg m$^{-2}$)', color='tab:blue')
    ax1.set_xlabel('Year')
    ax1.set_title(f"Historical climate ({ds.attrs.get('climate_source', 'OGGM')}), {window_years}-yr running mean")
    ax1.grid(alpha=0.3)
    plt.show()
    t_ref = float(temp.sel(year=slice(1960, 1990)).mean())
    t_now = float(temp.sel(year=slice(2000, 2019)).mean())
    print(f"Climate record: {int(temp.year[0])}-{int(temp.year[-1])}")
    t_8110 = float(temp.sel(year=slice(1981, 2010)).mean())
    print(f"Mean temperature 1960-1990: {t_ref:.2f} °C, 1981-2010: {t_8110:.2f} °C, 2000-2019: {t_now:.2f} °C")
    print(f"-> offsets vs. the reference period 1960-1990 (delta_temp): "
          f"1981-2010: {t_8110 - t_ref:+.1f} °C, 2000-2019: {t_now - t_ref:+.1f} °C")
    ds.close()


def compare_runs(run_dirs, labels=None, observations=None):
    """
    Compare ice volume and area through time of several IGM runs (e.g. scenarios).

    run_dirs : list of run folders (each containing output_ts.nc)
    labels : list of legend labels (default: folder names)
    observations : optional dict {'area': (years, values), 'vol': (years, values)} to overlay
    """
    import os
    labels = labels or [os.path.basename(os.path.normpath(r)) for r in run_dirs]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
    for run, lab in zip(run_dirs, labels):
        ds = xr.open_dataset(os.path.join(run, 'output_ts.nc'))
        ax1.plot(ds['time'], ds['vol'], lw=2, label=lab)
        ax2.plot(ds['time'], ds['area'], lw=2, label=lab)
        ds.close()
    if observations:
        for key, ax in (('vol', ax1), ('area', ax2)):
            if key in observations:
                ax.plot(*observations[key], 'ko', label='observations')
    ax1.set_ylabel('Ice volume (km$^3$)')
    ax2.set_ylabel('Glaciated area (km$^2$)')
    for ax in (ax1, ax2):
        ax.set_xlabel('Year')
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.show()


def plot_response_time(run_dir):
    """Response-time experiment (params_response_time): ELA forcing and volume response through time.

    Reads the oscillation parameters in run_dir/.hydra/config.yaml, plots the ELA and the ice
    volume, and estimates the lag between the ELA minimum and the volume maximum.
    """
    import os, yaml
    with open(os.path.join(run_dir, '.hydra', 'config.yaml')) as f:
        cfg = yaml.safe_load(f)
    p = cfg['processes']['smb_oscillating']
    t0 = cfg['processes']['time']['start']
    ds = xr.open_dataset(os.path.join(run_dir, 'output_ts.nc'))
    t, vol = ds['time'].values, ds['vol'].values
    ds.close()
    ela_anom = -p['amplitude'] * np.sin(2 * np.pi * (t - t0) / p['period'])
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(t, ela_anom, color='tab:red', lw=2)
    ax1.set_ylabel('ELA anomaly (m)', color='tab:red')
    ax1.invert_yaxis()   # low ELA (glacier-friendly) up
    ax2 = ax1.twinx()
    ax2.plot(t, vol, color='tab:blue', lw=2)
    ax2.set_ylabel('Ice volume (km$^3$)', color='tab:blue')
    ax1.set_xlabel('Model time (yr)')
    ax1.grid(alpha=0.3)
    # lag between an ELA minimum (at t0 + period/4 + n*period) and the volume maximum of the last period
    late = t >= t[-1] - p['period']
    if t[-1] - t0 >= 2 * p['period']:
        t_vol_max = t[late][np.argmax(vol[late])]
        lag = (t_vol_max - t0 - p['period'] / 4) % p['period']
        ax1.set_title(f"Period {p['period']:.0f} yr, amplitude {p['amplitude']:.0f} m: "
                      f"volume peaks ~{lag:.0f} yr after the ELA minimum")
        print(f"Lag between ELA minimum and volume maximum: ~{lag:.0f} yr "
              f"(volume amplitude: {vol[late].max() - vol[late].min():.3f} km3)")
    plt.show()


def plot_particles(run_dir, year=None, rh_min=0.0):
    """Moraine experiment (params_past_moraines): particles (debris) on top of the ice thickness map.

    run_dir : run folder containing output.nc and trajectories/traj-<year>.csv
    year : year to show (default: last one); the closest particle file is used
    rh_min : keep particles with relative height >= rh_min (0 = bed, 1 = surface), e.g. 0.95 for
             the debris at the surface only (where moraines form)
    Colour = relative height in the ice.
    """
    import os, glob
    files = sorted(glob.glob(os.path.join(run_dir, 'trajectories', 'traj-*.csv')))
    if not files:
        print("No particle files found (run params_past_moraines first)")
        return
    years = np.array([float(os.path.basename(f)[5:-4].replace('-', '.')) for f in files])
    i = len(files) - 1 if year is None else int(np.argmin(np.abs(years - year)))
    P = np.loadtxt(files[i], delimiter=',', skiprows=1, ndmin=2)   # columns: Id, x, y, z, rh, t
    P = P[P[:, 4] >= rh_min] if len(P) else P
    ds = xr.open_dataset(os.path.join(run_dir, 'output.nc'))
    k = int(np.argmin(np.abs(ds['time'].values - years[i])))
    thk = ds['thk'].isel(time=k)
    x, y = ds['x'].values, ds['y'].values
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(thk.where(thk > 1), origin='lower', extent=[x.min(), x.max(), y.min(), y.max()],
              cmap='Blues')
    if len(P):
        sc = ax.scatter(P[:, 1], P[:, 2], c=P[:, 4], s=2, cmap='copper_r', vmin=0, vmax=1)
        plt.colorbar(sc, ax=ax, shrink=0.8, label='relative height (0 = bed, 1 = surface)')
    ax.set_title(f"Particles in {years[i]:.0f} ({len(P)} particles)")
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    plt.show()
    ds.close()


def compare_velocity_misfit(input_file, run_dirs, year=2020):
    """Ice-flow parameter exploration (run_sensitivity.sh): modelled vs observed surface speed for several runs.

    For each run, reads arrhenius and tau_ref in .hydra/config.yaml and computes the RMSE and the
    mean bias (model - observation) of the surface speed on the glacier. Returns a table (DataFrame)
    and, if the runs form an arrhenius x tau_ref grid, shows the RMSE as a map.
    """
    import os, yaml
    import pandas as pd
    ds_in = xr.open_dataset(input_file)
    rows = []
    for run in run_dirs:
        with open(os.path.join(run, '.hydra', 'config.yaml')) as f:
            phys = yaml.safe_load(f)['processes']['iceflow']['physics']
        ds = xr.open_dataset(os.path.join(run, 'output.nc'))
        obs = ds_in.interp(x=ds['x'], y=ds['y'])    # observations on the model grid (coarsening)
        vobs = np.sqrt(obs['uvelsurfobs']**2 + obs['vvelsurfobs']**2).values
        k = int(np.argmin(np.abs(ds['time'].values - year)))
        vmod = np.sqrt(ds['uvelsurf'].isel(time=k)**2 + ds['vvelsurf'].isel(time=k)**2).values
        mask = (ds['thk'].isel(time=k).values > 1) & np.isfinite(vobs) & (vobs > 0)
        ds.close()
        d = vmod[mask] - vobs[mask]
        rows.append({'run': os.path.basename(os.path.normpath(run)),
                     'arrhenius': phys['viscosity']['arrhenius'],
                     'tau_ref': phys['sliding']['tau_ref'],
                     'rmse (m/yr)': np.sqrt(np.mean(d**2)), 'bias (m/yr)': np.mean(d)})
    ds_in.close()
    df = pd.DataFrame(rows)
    grid = df.pivot_table(index='arrhenius', columns='tau_ref', values='rmse (m/yr)')
    if grid.shape[0] > 1 and grid.shape[1] > 1:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        im = ax.imshow(grid.values, origin='lower', cmap='viridis_r')
        ax.set_xticks(range(grid.shape[1]), [f"{v:g}" for v in grid.columns])
        ax.set_yticks(range(grid.shape[0]), [f"{v:g}" for v in grid.index])
        for (r, c), v in np.ndenumerate(grid.values):
            ax.text(c, r, f"{v:.0f}", ha='center', va='center', color='w')
        ax.set_xlabel('tau_ref (MPa)')
        ax.set_ylabel('arrhenius (MPa$^{-3}$ a$^{-1}$)')
        plt.colorbar(im, ax=ax, label='speed RMSE (m/yr)')
        ax.set_title(f"Modelled vs observed surface speed ({year})")
        plt.show()
    return df.sort_values('rmse (m/yr)')


# ---------------------------------------------------------------------------
# Sample sites (session 4, Mont Miné): ice thickness and deglaciation dates
# ---------------------------------------------------------------------------

def _read_samples(samples_file):
    """Read a sample CSV: first column = ID, last two columns = UTM x, y (same projection as IGM)."""
    import pandas as pd
    df = pd.read_csv(samples_file)
    return list(df.iloc[:, 0]), df.iloc[:, -2].to_numpy(float), df.iloc[:, -1].to_numpy(float)


def plot_samples_map(nc_file, samples_file='Samples_2024.csv', year=None):
    """Map of ice thickness (at `year`, default: first time step) with the sample sites."""
    ids, sx, sy = _read_samples(samples_file)
    ds = xr.open_dataset(nc_file)
    thk = ds['thk'].sel(time=year, method='nearest') if (year is not None and 'time' in ds.dims) else \
          (ds['thk'].isel(time=0) if 'time' in ds.dims else ds['thk'])
    extent = [float(ds.x.min()), float(ds.x.max()), float(ds.y.min()), float(ds.y.max())]
    plt.figure(figsize=(7, 7))
    if 'usurf' in ds:
        us = ds['usurf'].isel(time=0) if 'time' in ds['usurf'].dims else ds['usurf']
        plt.imshow(us, origin='lower', extent=extent, cmap='gray')
    im = plt.imshow(thk.where(thk > 1), origin='lower', extent=extent, cmap='Blues')
    plt.colorbar(im, shrink=0.7, label='Ice thickness (m)')
    plt.scatter(sx, sy, c='r', s=25, edgecolors='k')
    for i, x, y in zip(ids, sx, sy):
        plt.annotate(str(i), (x, y), fontsize=8, color='r', xytext=(3, 3), textcoords='offset points')
    if not np.all((sx > extent[0]) & (sx < extent[1]) & (sy > extent[2]) & (sy < extent[3])):
        print("Warning: some samples are outside the model domain")
    plt.title(f"Ice thickness {'' if 'time' not in thk.coords else int(thk.time)} and sample sites")
    plt.xlabel('x (m)'); plt.ylabel('y (m)')
    plt.show()
    ds.close()


def plot_thk_at_samples(nc_file, samples_file='Samples_2024.csv', ages=None, thk_min=1.0):
    """
    Ice thickness through time at each sample site, and the modelled deglaciation year
    (last time the site is ice-covered, thickness > thk_min).

    ages : optional dict {sample_id: measured age in years before 2024 (or calendar year if > 1000)}
    Returns a pandas DataFrame with the modelled deglaciation year of each sample.
    """
    import pandas as pd
    ids, sx, sy = _read_samples(samples_file)
    ds = xr.open_dataset(nc_file)
    thk = ds['thk'].interp(x=xr.DataArray(sx, dims='s'), y=xr.DataArray(sy, dims='s'))
    times = ds['time'].values
    n = len(ids)
    fig, axs = plt.subplots(n, 1, figsize=(10, 1.8 * n), sharex=True, squeeze=False)
    rows = []
    for i in range(n):
        h = thk.isel(s=i).values
        covered = np.where(h > thk_min)[0]
        deglac = times[covered[-1] + 1] if len(covered) and covered[-1] + 1 < len(times) else (np.nan if len(covered) else times[0])
        ax = axs[i, 0]
        ax.plot(times, h, 'k')
        if np.isfinite(deglac):
            ax.axvline(deglac, color='b', ls='--', label=f'modelled ice-free: {deglac:.0f}')
        if ages and ids[i] in ages:
            a = ages[ids[i]]
            yr = a if a > 1000 else 2024 - a
            ax.axvline(yr, color='r', label=f'measured: {yr:.0f}')
        ax.set_ylabel('thk (m)')
        ax.set_title(f'Sample {ids[i]}', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        rows.append(dict(sample=ids[i], modelled_deglaciation=deglac,
                         max_thk_m=float(np.nanmax(h)) if len(h) else np.nan))
    axs[-1, 0].set_xlabel('Year')
    plt.tight_layout()
    plt.show()
    ds.close()
    return pd.DataFrame(rows)

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
       axes[0].text(sp[ii, 0], sp[ii, 1], id[ii], fontsize=7, c='r') 
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
       axes[1].text(sp[ii, 0], sp[ii, 1], id[ii], fontsize=7, c='r') 
    axes[1].invert_yaxis() 
    plt.colorbar(c2, ax=axes[1], orientation='vertical', label='Ice Thickness (units)')  # Add colorbar

    # Adjust layout
    plt.tight_layout()

    # Show the plots
    plt.show()

    # Optionally, close the dataset explicitly (not necessary in most cases)
    dataset.close()

def animate_glacier_evolution(nc_file='output.nc'):
    # Load the dataset
    ds = xr.open_dataset(nc_file)

    df = pd.read_csv('Samples_2024.csv')
    id = list(df.iloc[:, 0])
    sp = np.array(df.iloc[:, 3:])  # Get the rest of the columns (numeric values)
    
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
    ax.scatter(sp[:, 0], sp[:, 1], c='k', marker='o', s=10)

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
        ax.scatter(sp[:, 0], sp[:, 1], c='k', marker='o', s=10)
        for ii in range(sp.shape[0]):
            ax.text(sp[ii, 0], sp[ii, 1], id[ii], fontsize=7, c='r') 

        ax.set_title(f'Glacier Evolution\nTime: {times[frame].values}')
        return ax
    
    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(times), interval=200, blit=False)
    
    # Display the animation in the notebook
    display_animation = HTML(ani.to_jshtml())
    plt.close()  # Prevents static display of the plot
    ds.close()
    
    return display_animation

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

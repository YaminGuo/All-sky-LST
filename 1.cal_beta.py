import pandas as pd
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import glob
import os
import netCDF4 as nc
import matplotlib.pyplot as plt
from pyhdf.SD import SD, SDC
import h5py



path_LAI='/Volumes/GYM/DLR/LAI/20211127/'
out_path='/Volumes/GYM/DLR/allskyLST/1.beta/GLASS/'
LAI_files=glob.glob(path_LAI+'GLASS*.hdf')
file_latlon = '/Users/yaminguo/Desktop/GYM/Work/LST/code/DLR/ABIL2_Cloud_latlon.nc'
ds = nc.Dataset(file_latlon)
goes_lat = ds.variables['lat_CTT_02'][:]  # shape = (5424, 5424)
goes_lon = ds.variables['lon_CTT_02'][:]
points = np.stack([goes_lat.ravel(), goes_lon.ravel()], axis=-1)
for file in LAI_files:

    filename = os.path.basename(file)
    date_str = filename.split('.')[2][1:]

    hdf = SD(file, SDC.READ)
    LAI = hdf.select('LAI')[:]
    LAI = np.where(LAI == 255, np.nan, LAI * 0.1).astype(np.float32)

    # lat_LAI = hdf.select('Lat')[:]
    # lon_LAI = hdf.select('Lon')[:]
    lat_LAI= np.arange(90, -90, -0.05)
    lon_LAI=np.arange(-180, 180, 0.05)
    plt.figure(figsize=(10, 6))
    plt.imshow(LAI, cmap='inferno')
    plt.title("LAI")
    plt.colorbar(label='LAI')
    plt.tight_layout()
    plt.show()
    interp_func = RegularGridInterpolator(
            (lat_LAI, lon_LAI),
            LAI,
            bounds_error=False,
            fill_value=np.nan
        )
    LAI_on_goes = interp_func(points).reshape(goes_lat.shape)
    plt.figure(figsize=(10, 6))
    plt.imshow(LAI_on_goes, cmap='inferno')
    plt.title("LAI")
    plt.colorbar(label='LAI')
    plt.tight_layout()
    plt.show()
    # np.save(out_path+'GLASSLAI_on_GOES_grid_'+date_str,LAI_on_goes.astype(np.float32))


    beta = 0.5 * np.exp(-2.13 * (0.88 - 0.78 * np.exp(-0.6 * LAI_on_goes)))
    np.save(out_path+'beta_on_GOES_grid_'+date_str,beta.astype(np.float32))

    plt.figure(figsize=(10, 6))
    plt.imshow(beta, cmap='inferno')
    plt.title("beta")
    plt.colorbar(label='beta')
    plt.tight_layout()
    plt.show()

import netCDF4 as nc
from datetime import datetime, timezone, timedelta
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import os


out_path='/Volumes/GYM/DLR/allskyLST/4.delta_DSRDLR/'

CERES_clear='/Volumes/GYM/DLR/DSR_CERES/CERES_SYN1deg-1H_Terra-Aqua-NOAA20_Ed4.2_Subset_20211101-20211130_DSRDLR_clearsky.nc'
CERES_allsky='/Volumes/GYM/DLR/RN/CERES_SYN1deg-1H_Terra-Aqua-NOAA20_Ed4.2_Subset_20211101-20211231_Rn.nc'


CERES_DS=nc.Dataset(CERES_clear, mode='r')
# print(CERESS_DS.variables)
CERES_lat=CERES_DS.variables['lat'][:]
CERES_lon=CERES_DS.variables['lon'][:]
CERES_time=CERES_DS.variables['time'][:]
ceres_ref_time = datetime(2000, 3, 1, 0, 0, 0)
CERES_datetime = np.array([ceres_ref_time + timedelta(days=float(d)) for d in CERES_time])
# target time
date_str='20211127'
start_date = datetime(2021, 11, 27)
end_date   = datetime(2021, 11, 27, 23, 59)  # 包含整天
time_mask = (CERES_datetime >= start_date) & (CERES_datetime <= end_date)
selected_indices = np.where(time_mask)[0]
CERES_DSR = CERES_DS.variables['adj_atmos_sw_down_clr_surface_1h'][selected_indices, :, :].filled(np.nan)
CERES_DLR = CERES_DS.variables['adj_atmos_lw_down_clr_surface_1h'][selected_indices, :, :].filled(np.nan)


CERES_DS_all=nc.Dataset(CERES_allsky, mode='r')
CERES_lat_all=CERES_DS_all.variables['lat'][:]
CERES_lon_all=CERES_DS_all.variables['lon'][:]
CERES_time_all=CERES_DS_all.variables['time'][:]
CERES_datetime_all = np.array([ceres_ref_time + timedelta(days=float(d)) for d in CERES_time_all])
time_mask_all = (CERES_datetime_all >= start_date) & (CERES_datetime_all <= end_date)
selected_indices_all = np.where(time_mask_all)[0]
CERES_DSR_all = CERES_DS_all.variables['adj_atmos_sw_down_all_surface_1h'][selected_indices_all, :, :].filled(np.nan)
CERES_DLR_all = CERES_DS_all.variables['adj_atmos_lw_down_all_surface_1h'][selected_indices_all, :, :].filled(np.nan)

delta_DSR=CERES_DSR_all-CERES_DSR
delta_DLR=CERES_DLR_all-CERES_DLR
if CERES_lat[0] < CERES_lat[-1]:
    CERES_lat = CERES_lat[::-1]
    delta_DSR = delta_DSR[:, ::-1, :]
    delta_DLR = delta_DLR[:, ::-1, :]# 翻转纬度维度,从北向南
plt.figure(figsize=(10, 6))
plt.imshow(delta_DLR[18,:,:], cmap='inferno')
plt.title("RN")
plt.colorbar(label='RN')
plt.tight_layout()
plt.show()
file_latlon='/Users/yaminguo/Desktop/GYM/Work/LST/code/DLR/ABIL2_Cloud_latlon.nc'
ds = nc.Dataset(file_latlon)
goes_lat = ds.variables['lat_CTT_02'][:]  # shape = (5424, 5424)
goes_lon = ds.variables['lon_CTT_02'][:]
interp_points = np.stack([goes_lat.ravel(), goes_lon.ravel()], axis=-1)


T = delta_DSR.shape[0]
delta_DSR_interp = np.full((T, goes_lat.shape[0], goes_lat.shape[1]), np.nan, dtype=np.float32)
delta_DLR_interp = np.full((T, goes_lat.shape[0], goes_lat.shape[1]), np.nan, dtype=np.float32)
for t in range(T):
    delta_DSR_2D = delta_DSR[t, :, :]
    delta_DLR_2D = delta_DLR[t, :, :]

    interp_func_DSR = RegularGridInterpolator(
        (CERES_lat, CERES_lon),
        delta_DSR_2D,
        bounds_error=False,
        fill_value=np.nan
    )
    interp_func_DLR = RegularGridInterpolator(
        (CERES_lat, CERES_lon),
        delta_DLR_2D,
        bounds_error=False,
        fill_value=np.nan
    )


    delta_DSR_interp[t, :, :] = interp_func_DSR(interp_points).reshape(goes_lat.shape)
    delta_DLR_interp[t, :, :] = interp_func_DLR(interp_points).reshape(goes_lat.shape)

    print(f"Interpolated time step {t+1}/{T}")

plt.figure(figsize=(10, 6))
plt.imshow(delta_DLR_interp[18,:,:], cmap='inferno')
plt.title("RN")
plt.colorbar(label='RN')
plt.tight_layout()
plt.show()





mask = ~np.isnan(delta_DSR_interp)
compressed = delta_DSR_interp[mask]
np.savez_compressed(os.path.join(out_path, f'delta_DSR_on_GOES_grid_{date_str}.npz'), data=compressed, mask=mask)

mask_DLR = ~np.isnan(delta_DLR_interp)
compressed_DLR = delta_DLR_interp[mask_DLR]
np.savez_compressed(os.path.join(out_path, f'delta_DLR_on_GOES_grid_{date_str}.npz'), data=compressed_DLR, mask=mask_DLR)

a=1
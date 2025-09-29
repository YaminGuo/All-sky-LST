import pandas as pd
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import glob
import os
import netCDF4 as nc
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta
import re
def compute_sunrise_utc_grid(lat_grid, lon_grid, date):
    lat_rad = np.radians(lat_grid)
    day_of_year = date.timetuple().tm_yday

    decl = 23.44 * np.cos(np.radians(360 / 365 * (day_of_year + 10)))
    decl_rad = np.radians(decl)

    cos_omega = -np.tan(lat_rad) * np.tan(decl_rad)
    cos_omega = np.clip(cos_omega, -1.0, 1.0)
    omega = np.arccos(cos_omega)

    sunrise_local = 12 - omega * 180 / np.pi / 15
    sunrise_utc = (sunrise_local - lon_grid / 15.0) % 24
    return sunrise_utc

def extract_local_hour_data(var_24h, target_hour):
    H, W = target_hour.shape
    hour_index = np.floor(target_hour).astype(int) % 24
    result = np.full((H, W), np.nan, dtype=var_24h.dtype)
    for h in range(24):
        mask = (hour_index == h)
        if np.any(mask):
            result[mask] = var_24h[h][mask]
    return result

def extract_noon_and_sunrise_data(LST, G, lat, lon, date=datetime(2021, 11, 27)):
    sunrise_utc = compute_sunrise_utc_grid(lat, lon, date)
    noon_utc = (12 - lon / 15.0) % 24

    LST_sunrise = extract_local_hour_data(LST, sunrise_utc)
    G_sunrise   = extract_local_hour_data(G, sunrise_utc)

    LST_noon = extract_local_hour_data(LST, noon_utc)
    G_noon   = extract_local_hour_data(G, noon_utc)

    return LST_noon, G_noon, LST_sunrise, G_sunrise

delta_Z = 0.1

file='/Volumes/GYM/DLR/RN/CERES_SYN1deg-1H_Terra-Aqua-NOAA20_Ed4.2_Subset_20211101-20211231_Rn.nc'
beta_path='/Volumes/GYM/DLR/allskyLST/1.beta/GLASS/'
LST_path='/Volumes/GYM/DLR/allskyLST/0.clearskyLST/'
target_date=datetime(2021, 11, 27)

CERES_DS=nc.Dataset(file, mode='r')
CERES_lat=CERES_DS.variables['lat'][:]
CERES_lon=CERES_DS.variables['lon'][:]
CERES_time=CERES_DS.variables['time'][:]
ceres_ref_time = datetime(2000, 3, 1, 0, 0, 0)
CERES_datetime = np.array([ceres_ref_time + timedelta(days=float(d)) for d in CERES_time])
# target time
start_date = target_date - timedelta(days=15)
end_date = target_date +timedelta(days=15)
n_days = (end_date - start_date).days + 1

file_latlon='/Users/yaminguo/Desktop/GYM/Work/LST/code/DLR/ABIL2_Cloud_latlon.nc'
ds = nc.Dataset(file_latlon)
goes_lat = ds.variables['lat_CTT_02'][:]  # shape = (5424, 5424)
goes_lon = ds.variables['lon_CTT_02'][:]
interp_points = np.stack([goes_lat.ravel(), goes_lon.ravel()], axis=-1)

Kg_list=[]
for day in range(n_days):
    G_day24 = []
    daystart=start_date + timedelta(days=day)
    dayend=start_date + timedelta(days=day+1)
    time_mask = (CERES_datetime >= daystart) & (CERES_datetime < dayend)
    selected_indices = np.where(time_mask)[0]
    if len(selected_indices)!=24:
        continue
    time_str = daystart.strftime('%Y%m%d')
    doy = daystart.timetuple().tm_yday
    print(time_str)

    # beta
    beta_files = glob.glob(os.path.join(beta_path, "beta_on_GOES_grid_*.npy"))
    doy_beta = []
    pattern = re.compile(r'^beta_on_GOES_grid_(\d{4})(\d{3})\.npy$')
    for f in beta_files:
        base = os.path.basename(f)
        m = pattern.search(base)
        if m:
            doy_beta.append(int(m.group(2)))
    nearest_doy = min(doy_beta, key=lambda x: (abs(x - doy), x))

    if abs(nearest_doy - doy)>30:
        print('no matching beta')
        continue
    beta_doy_str=str(nearest_doy).zfill(3)
    beta=np.load(beta_path+f"beta_on_GOES_grid_2021{beta_doy_str}.npy")
    # plt.figure(figsize=(10, 6))
    # plt.imshow(beta, cmap='inferno')
    # plt.title("beta")
    # plt.colorbar(label='beta')
    # plt.tight_layout()
    # plt.show()
    CERES_DSR = CERES_DS.variables['adj_atmos_sw_down_all_surface_1h'][selected_indices, :, :].filled(np.nan)
    CERES_USR = CERES_DS.variables['adj_atmos_sw_up_all_surface_1h'][selected_indices, :, :].filled(np.nan)

    CERES_DLR = CERES_DS.variables['adj_atmos_lw_down_all_surface_1h'][selected_indices, :, :].filled(np.nan)

    CERES_ULR = CERES_DS.variables['adj_atmos_lw_up_all_surface_1h'][selected_indices, :, :].filled(np.nan)

    CERES_RN = CERES_DSR - CERES_USR + CERES_DLR - CERES_ULR
    if CERES_lat[0] < CERES_lat[-1]:
        CERES_lat = CERES_lat[::-1]
        CERES_RN = CERES_RN[:, ::-1, :]

    for idx in range(24):
        print(idx)
        RN_hour = CERES_RN[idx]
        interp_func = RegularGridInterpolator(
            (CERES_lat, CERES_lon),
            RN_hour,
            bounds_error=False,
            fill_value=np.nan
        )

        RN_interp = interp_func(interp_points).reshape(goes_lat.shape)
        G = RN_interp * beta
        G_day24.append(G)
    G_daily_stack = np.stack(G_day24, axis=0)
    del G_day24
    # read LST
    LST_file=LST_path+f'LST_Downscaling_Model_Output_{time_str}.nc'
    ds = nc.Dataset(LST_file)
    LST = ds.variables['LST_high'][:].filled(np.nan)
    # plt.figure(figsize=(10, 6))
    # plt.imshow(LST[18,:,:], cmap='inferno')
    # plt.title("LST")
    # plt.colorbar(label='LST')
    # plt.tight_layout()
    # plt.show()
    LST_noon, G_noon, LST_sunrise, G_sunrise = extract_noon_and_sunrise_data(LST, G_daily_stack, goes_lat, goes_lon)
    # plt.figure(figsize=(10, 6))
    # plt.imshow(LST_noon, cmap='inferno')
    # plt.title("LST")
    # plt.colorbar(label='LST')
    # plt.tight_layout()
    # plt.show()
    num = G_noon - G_sunrise
    den = LST_noon- LST_sunrise

    mask = (np.abs(den) > 1e-3) & (~np.isnan(num)) & (~np.isnan(den))
    kg = np.full_like(num, np.nan, dtype=np.float32)
    kg[mask] = delta_Z * num[mask] / den[mask]
    kg[(kg <= 0) | (kg > 20)] = np.nan
    # plt.figure(figsize=(10, 6))
    # plt.imshow(kg, cmap='inferno')
    # plt.title("Kg")
    # plt.colorbar(label='Kg')
    # plt.tight_layout()
    # plt.show()
    Kg_list.append(kg)
if len(Kg_list) > 0:
    kg_stack = np.stack(Kg_list, axis=0)  # (days, H, W)
    kg_median = np.nanmedian(kg_stack, axis=0)  # (H, W)

    plt.figure(figsize=(10, 6))
    plt.imshow(kg_median, cmap='inferno')
    plt.title("kg_median")
    plt.colorbar(label='kg_median')
    plt.tight_layout()
    plt.show()

    np.save("Kg_median_GOES_grid_20211112_20211127_20211212.npy", kg_median)
    print("Kg median image saved ")
else:
    print("No valid k_g data found. Nothing saved.")



import math

import netCDF4 as nc
import pandas as pd
import numpy as np

from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import xgboost as xgb
from datetime import datetime, timezone, timedelta

import glob



def LST_cal(ULR,DLR,BBE_Low):
    sigma = 5.670374419e-8

    mask = (~np.isnan(ULR)) & (~np.isnan(DLR)) & (~np.isnan(BBE_Low)) & (BBE_Low > 0)

    numerator = ULR[mask] - (1 - BBE_Low[mask]) * DLR[mask]
    denominator = sigma * BBE_Low[mask]

    valid = (numerator > 0) & (denominator > 0)
    temp = np.full_like(numerator, np.nan)
    temp[valid] = (numerator[valid] / denominator[valid]) ** 0.25

    LST_ERA5 = np.full_like(ULR, np.nan)
    LST_ERA5[mask] = temp

    return LST_ERA5

def plotarray(var):
    plt.figure(figsize=(10, 6))
    plt.imshow(var, origin='upper', cmap='plasma')
    plt.show()
    return

NDVI_path='/Volumes/GYM/DLR/NDVI/'
BBE_path='/Volumes/GYM/DLR/BBE/'
out_path='/Volumes/GYM/DLR/allskyLST/0.clearskyLST/'
target_date=datetime(2021, 11, 27)
# 0.read ERA5
ERA5_file='/Volumes/GYM/DLR/ERA5_clear_LW/ERA5_LongwaveRadiation_clearsky_202111_202112.nc'
ds = nc.Dataset(ERA5_file)
lat_ERA5 = ds.variables['latitude'][:]
lon = ds.variables['longitude'][:]
lon_shifted = np.where(lon > 180, lon - 360, lon)
sorted_idx = np.argsort(lon_shifted)
lon_ERA5 = lon_shifted[ sorted_idx]

lon_grid, lat_grid = np.meshgrid(lon_ERA5, lat_ERA5)
target_points = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=-1)

time = ds.variables['valid_time'][:]
start_time = pd.Timestamp('1970-01-01 00:00:00')
time_deltas = pd.to_timedelta(time.data, unit='s')
time_trans1=start_time+time_deltas

start_date = target_date - timedelta(days=15)
end_date = target_date +timedelta(days=15)
# start_date = datetime(2021, 11, 27)
# end_date   = datetime(2021, 11, 27, 23, 59)
n_days = (end_date - start_date).days + 1



# 2.DEM
DEM_high=np.load('DEM_on_GOES_grid.npy')
DEM_low=np.flipud(np.load('DEM_LowRes_Global2.npy'))



file_latlon = '/Users/yaminguo/Desktop/GYM/Work/LST/code/DLR/ABIL2_Cloud_latlon.nc'
ds_goesgrid = nc.Dataset(file_latlon)
goes_lat = ds_goesgrid.variables['lat_CTT_02'][:]  # shape = (5424, 5424)
goes_lon = ds_goesgrid.variables['lon_CTT_02'][:]
target_points_goes = np.stack([goes_lat.ravel(), goes_lon.ravel()], axis=-1)


del goes_lon



for day in range(n_days):
    LST_high_all = []
    LST_low_all = []
    daystart=start_date + timedelta(days=day)
    dayend=start_date + timedelta(days=day+1)
    time_mask = (time_trans1 >= daystart) & (time_trans1 < dayend)
    selected_indices = np.where(time_mask)[0]
    time_str = daystart.strftime('%Y%m%d')
    print(time_str)

    # 1.BBE reduce
    BBE_file = BBE_path+f'VIIRS.LSE.npp.{time_str}.v2r2.nc'

    ds_bbe = nc.Dataset(BBE_file)
    bbe = ds_bbe.variables['emis_bb'][:].filled(np.nan) * 0.001 + 0.9
    lat_raw = np.linspace(90, -90, bbe.shape[0])  # 北到南
    lon_raw = np.linspace(-180, 180, bbe.shape[1])

    bbe_interp_func = RegularGridInterpolator(
        (lat_raw, lon_raw),
        bbe,
        bounds_error=False,
        fill_value=np.nan
    )

    BBE_LowRes_Global = bbe_interp_func(target_points).reshape(721, 1440)

    del bbe, lat_raw, lon_raw
    # 3.NDVI

    NDVI_file =glob.glob(NDVI_path+f'VIIRS-Land_v001_NPP13C1_S-NPP_{time_str}_*.nc')
    if len(NDVI_file)!=1:
        print('No NDVI file')
        continue
    ds_ndvi = nc.Dataset(NDVI_file[0])
    ndvi = ds_ndvi.variables['NDVI'][:].filled(np.nan)[0, :, :]
    lat_ndvi_raw = ds_ndvi.variables['latitude'][:].filled(np.nan)
    lon_ndvi_raw = ds_ndvi.variables['longitude'][:].filled(np.nan)
    NDVI_interp_func = RegularGridInterpolator(
        (lat_ndvi_raw, lon_ndvi_raw),
        ndvi,
        bounds_error=False,
        fill_value=np.nan
    )

    NDVI_LowRes_Global = NDVI_interp_func(target_points).reshape(lon_grid.shape)
    NDVI_high = NDVI_interp_func(target_points_goes).reshape(goes_lat.shape)


    for idx in selected_indices:
        print(time_trans1[idx])

        DLR = ds.variables['strdc'][idx, :, :].data / 3600
        NLR = ds.variables['strc'][idx, :, :].data / 3600
        ULR = DLR - NLR
        del NLR
        DLR_shifted = DLR[:, sorted_idx]
        ULR_shifted = ULR[:, sorted_idx]




        LST_ERA5=LST_cal(ULR_shifted, DLR_shifted, BBE_LowRes_Global)

        # plotarray(LST_ERA5)
        # plotarray(DEM_low)
        # plotarray(NDVI_LowRes_Global)

        # downscaling
        mask = ~np.isnan(LST_ERA5) & ~np.isnan(DEM_low) & ~np.isnan(NDVI_LowRes_Global)
        if mask.sum() < 1000:
            print(mask.sum())
        X_train = np.stack([
            DEM_low[mask],
            NDVI_LowRes_Global[mask]
        ], axis=1)
        y_train = LST_ERA5[mask]
        # print('model training...')
        model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, verbosity=0)
        model.fit(X_train, y_train)

        mask_high = ~np.isnan(DEM_high) & ~np.isnan(NDVI_high)
        # print('model prediction...')
        X_pred = np.stack([
            DEM_high[mask_high],
            NDVI_high[mask_high]
        ], axis=1)
        y_pred_valid = model.predict(X_pred)
        y_pred = np.full(DEM_high.shape, np.nan, dtype=np.float32)

        y_pred[mask_high] = y_pred_valid
        LST_high_all.append(y_pred)
        LST_low_all.append(LST_ERA5)
    print('saving...')
    out_file = out_path+f"LST_Downscaling_Model_Output_{time_str}.nc"

    time_dim = 24
    y_high, x_high = DEM_high.shape

    y_low, x_low = LST_ERA5.shape

    with nc.Dataset(out_file, 'w', format='NETCDF4') as ds_out:
        ds_out.createDimension('time', time_dim)
        ds_out.createDimension('y_high', y_high)
        ds_out.createDimension('x_high', x_high)
        ds_out.createDimension('y_low', y_low)
        ds_out.createDimension('x_low', x_low)

        times = ds_out.createVariable('time', 'i4', ('time',))
        times.units = "hours since 2021-11-27 00:00:00"
        times.calendar = "standard"
        times[:] = np.arange(24)

        var_lst_high = ds_out.createVariable('LST_high', 'f4', ('time', 'y_high', 'x_high'),
                                         zlib=True, complevel=4, fill_value=np.nan)
        var_lst_low = ds_out.createVariable('LST_low', 'f4', ('time', 'y_low', 'x_low'),
                                        zlib=True, complevel=4, fill_value=np.nan)
        var_dem_high = ds_out.createVariable('DEM_high', 'f4', ('y_high', 'x_high'),
                                         zlib=True, complevel=4, fill_value=np.nan)
        var_ndvi_high = ds_out.createVariable('NDVI_high', 'f4', ('y_high', 'x_high'),
                                          zlib=True, complevel=4, fill_value=np.nan)

        var_lst_high[:, :, :] = LST_high_all
        var_lst_low[:, :, :] = LST_low_all
        var_dem_high[:, :] = DEM_high
        var_ndvi_high[:, :] = NDVI_high
    #
    # plt.figure(figsize=(10, 6))
    # plt.imshow(LST_ERA5, origin='upper', cmap='plasma')
    # plt.title(f"2021-11-27 {hour}:00:00")
    # plt.colorbar(label='LST_LowRes (K)')
    # plt.tight_layout()
    # plt.show()
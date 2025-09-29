import numpy as np

import netCDF4 as nc
import matplotlib.pyplot as plt

LST_file='/Volumes/GYM/DLR/allskyLST/0.clearskyLST/LST_Downscaling_Model_Output_20211127.nc'
delta_T_path='/Volumes/GYM/DLR/allskyLST/5.delta_T/'
hour=18
LST_clear='/Volumes/GYM/DLR/ABI/ABI-L2-LST2KMF/2021/331/18/OR_ABI-L2-LST2KMF-M6_G16_s20213311800208_e20213311809516_c20213311810366.nc'
ds_clear=nc.Dataset(LST_clear)
LST_clear=ds_clear.variables['LST'][:].filled(np.nan)

LST0=np.load('/Volumes/GYM/DLR/allskyLST/5.delta_T/LST0_correct_20211127_18.npy')

delta_T_fix=np.load(delta_T_path+f"delta_T_20211127_{str(hour)}_fixpointIteration50.npy")
delta_T_fix[np.isnan(LST0)]=np.nan

delta_T_newton=np.load(delta_T_path+f"delta_T_20211127_{str(hour)}_Newtoniteration50.npy" )
delta_T_newton[np.isnan(LST0)]=np.nan

delta_T_quartic=np.load(delta_T_path+f"delta_T_20211127_{str(hour)}_quartic.npy")
delta_T_quartic[np.isnan(LST0)]=np.nan


# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

im0 = axes[0].imshow(delta_T_fix, cmap="jet",vmin=0, vmax=-15)
axes[0].set_title("ΔT (Fixed-point)")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(delta_T_newton, cmap="jet",vmin=0, vmax=-15)
axes[1].set_title("ΔT (Newton)")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

im2 = axes[2].imshow(delta_T_quartic, cmap="jet",vmin=0, vmax=-15)
axes[2].set_title("ΔT (Quartic)")
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

plt.suptitle(f"Comparison of ΔT Methods (2021-11-27 Hour={hour})", fontsize=14)
plt.tight_layout()
plt.show()

a=1
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

im0 = axes[0].imshow(LST_clear, cmap="jet",vmin=240, vmax=320)
axes[0].set_title("GOES-R LST2KM Product")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(LST0, cmap="jet",vmin=240, vmax=320)
axes[1].set_title("Hypothetical Clear-Sky LST")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

im2 = axes[2].imshow(LST0+delta_T_quartic, cmap="jet",vmin=240, vmax=320)
axes[2].set_title("all sky LST(Quartic)")
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

plt.suptitle(f"LST for 2021-11-27 {hour}:00 UTC", fontsize=14)
plt.tight_layout()
plt.show()

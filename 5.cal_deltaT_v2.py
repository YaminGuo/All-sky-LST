import numpy as np
import netCDF4 as nc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt

def LST_corrected(LST0,LST_clear,BBE,albedo,NDVI,DEM):
    train_mask = (
            np.isfinite(LST0) &
            np.isfinite(LST_clear) &
            np.isfinite(albedo) & np.isfinite(BBE) &
            np.isfinite(NDVI) & np.isfinite(DEM)
    )
    apply_mask = (
            np.isfinite(LST0) &
            np.isfinite(albedo) & np.isfinite(BBE) &
            np.isfinite(NDVI) & np.isfinite(DEM)
    )
    X = np.stack([
        LST0[train_mask],
        albedo[train_mask],
        BBE[train_mask],
        NDVI[train_mask],
        DEM[train_mask]
    ], axis=1)

    y = (LST_clear - LST0)[train_mask]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=None, n_jobs=-1, random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"MAE={mae:.2f} K, RMSE={rmse:.2f} K")
    importances = rf.feature_importances_
    features = ["LST0_era", "albedo", "BBE", "NDVI", "DEM"]
    for f, imp in zip(features, importances):
        print(f"{f}: {imp:.3f}")

    plt.barh(features, importances)
    plt.xlabel("Feature importance")
    plt.title("Random Forest feature importance")
    plt.show()
    X_apply = np.stack([
        LST0[apply_mask], albedo[apply_mask], BBE[apply_mask],
        NDVI[apply_mask], DEM[apply_mask]
    ], axis=1)
    bias_hat = np.full_like(LST0, np.nan, dtype=float)
    bias_hat[apply_mask] = rf.predict(X_apply)

    LST0_corr = LST0 + bias_hat
    return LST0_corr

def solve_deltaT_fixed_point(
    albedo, delta_DSR, delta_DLR, beta,
    LST0, BBE, kg, delta_Z, sigma,
    valid_mask,
    max_iter=50,
    tolerance=1e-3,
    rad_threshold=np.inf,
    verbose=True
):
    """
    Solve ΔT using fixed-point iteration:
        ΔT_{n+1} = (ΔZ/kg) * β * ΔRn(ΔT_n)

    where
        ΔRn = (1 - albedo)*ΔDSR + BBE*ΔDLR
              - σ*BBE * [ (LST0 + ΔT)^4 - LST0^4 ]

    Parameters
    ----------
    albedo, delta_DSR, delta_DLR, beta, LST0, BBE, kg, delta_Z : ndarray
        2D arrays of the same shape. Units must be consistent.
        LST_correct must be in Kelvin.
    sigma : float
        Stefan–Boltzmann constant (≈ 5.670374419e-8 W m^-2 K^-4).
    valid_mask : ndarray of bool
        Boolean mask of valid pixels (True = include in iteration).
    max_iter : int
        Maximum number of iterations.
    tolerance : float
        Convergence criterion: |ΔT_new - ΔT| < tolerance.
    rad_threshold : float
        Additional constraint: |ΔRn| < rad_threshold.
        Default = infinity (no constraint).
    verbose : bool
        If True, print progress information at each iteration.

    Returns
    -------
    delta_T : ndarray
        Final ΔT solution (NaN for invalid pixels).
    converged : ndarray of bool
        Boolean mask of which valid pixels converged.
    n_iter : int
        Actual number of iterations performed.
    """
    # ==== Initialization ====
    delta_T = np.zeros_like(LST0, dtype=float)
    converged = np.zeros_like(LST0, dtype=bool)

    # Precompute constant factor (ΔZ/kg)*β
    K = (delta_Z / kg) * beta

    # ==== Iteration ====
    for i in range(max_iter):
        lst_plus = LST0 + delta_T

        delta_Rn = ((1.0 - albedo) * delta_DSR +
                    BBE * delta_DLR -
                    sigma * BBE * (lst_plus**4 - LST0**4))

        delta_T_new = K * delta_Rn

        diff = np.abs(delta_T_new - delta_T)
        rad_ok = np.abs(delta_Rn) < rad_threshold

        conv_now = (diff < tolerance) & rad_ok
        update_mask = valid_mask & (~converged)

        # Update ΔT and convergence status
        delta_T[update_mask] = delta_T_new[update_mask]
        converged |= (conv_now & valid_mask)

        if verbose:
            total_valid = int(np.sum(valid_mask))
            conv_count = int(np.sum(converged & valid_mask))
            print(f"Iteration {i+1}: converged pixels = {conv_count} / {total_valid}")

        if np.all(converged[valid_mask]):
            if verbose:
                print(f"All valid pixels converged at iteration {i+1}")
            n_iter = i + 1
            break
    else:
        if verbose:
            print("Warning: Not all valid pixels converged within max_iter")
        n_iter = max_iter

    # Assign NaN for invalid pixels
    delta_T = np.where(valid_mask, delta_T, np.nan)
    return delta_T, converged, n_iter

def solve_deltaT_newton(
    albedo, delta_DSR, delta_DLR, beta, LST0, BBE, kg, delta_Z, sigma,
    valid_mask=None, max_iter=20, tol_f=1e-3, tol_x=1e-3, step_clip=5.0, damping=1.0
):
    """
    Solve for ΔT using Newton's method (vectorized with NumPy).

    Equation:
        f(ΔT) = ΔT - K * [S - σ * BBE * ((T0+ΔT)^4 - T0^4)] = 0
        where K = (ΔZ / kg) * beta
              S = (1 - albedo) * ΔDSR + BBE * ΔDLR

    Parameters
    ----------
    albedo, delta_DSR, delta_DLR, beta, LST0, BBE, kg, delta_Z : ndarray
        Input fields (2D arrays).
    sigma : float
        Stefan–Boltzmann constant (≈ 5.67e-8 W m^-2 K^-4).
    valid_mask : ndarray of bool, optional
        Boolean mask for valid pixels. If None, will be inferred.
    max_iter : int
        Maximum number of Newton iterations.
    tol_f : float
        Residual tolerance for convergence (K).
    tol_x : float
        Step-size tolerance for convergence (K).
    step_clip : float
        Maximum absolute step per iteration (K), prevents divergence.
    damping : float
        Damping factor (0 < damping ≤ 1). Lower values improve stability.

    Returns
    -------
    delta_T : ndarray
        Estimated ΔT (same shape as inputs, NaN for invalid pixels).
    converged : ndarray of bool
        Convergence mask (True = converged).
    """

    # Precompute constants
    K = (delta_Z / kg) * beta
    S = (1.0 - albedo) * delta_DSR + BBE * delta_DLR

    shape = LST0.shape
    if valid_mask is None:
        valid_mask = (
            np.isfinite(K) & np.isfinite(S) & np.isfinite(LST0) &
            np.isfinite(BBE) & (kg != 0)
        )

    # --- Initial guess: first-order linearization ---
    denom = 1.0 + 4.0 * K * sigma * BBE * (LST0**3)
    delta_T = np.zeros_like(LST0, dtype=float)
    delta_T[valid_mask] = (K[valid_mask] * S[valid_mask]) / denom[valid_mask]

    converged = np.zeros(shape, dtype=bool)

    # --- Newton iterations ---
    for i in range(max_iter):
        lst_plus = LST0 + delta_T

        # ΔRn(T) = S - σ * BBE * [(T0+ΔT)^4 - T0^4]
        delta_Rn = S - sigma * BBE * (lst_plus**4 - LST0**4)

        # Residual f(ΔT) = ΔT - K * ΔRn
        f = delta_T - K * delta_Rn

        # Derivative f'(ΔT) = 1 + 4 * K * σ * BBE * (T0+ΔT)^3
        df = 1.0 + 4.0 * K * sigma * BBE * (lst_plus**3)

        # Newton step (with clipping and damping)
        step = -f / df
        if step_clip is not None:
            step = np.clip(step, -step_clip, step_clip)
        if damping is not None:
            step = damping * step

        # Update only valid and not-yet-converged pixels
        upd = valid_mask & (~converged)
        delta_T[upd] = delta_T[upd] + step[upd]

        # Convergence check: residual OR step size
        conv_now = (np.abs(f) < tol_f) | (np.abs(step) < tol_x)
        converged = converged | (conv_now & valid_mask)

        if np.all(converged[valid_mask]):
            break

    # Set invalid pixels to NaN
    delta_T = np.where(valid_mask, delta_T, np.nan)
    return delta_T, converged
import numpy as np

def solve_deltaT_quartic(
    albedo, delta_DSR, delta_DLR, beta, LST0, BBE, kg, delta_Z, sigma,
    valid_mask=None, dt_range=(-50.0, 50.0), imag_tol=1e-6
):
    """
    Solve ΔT by reducing to a quartic polynomial and finding real roots per pixel.

    Quartic:
        C*ΔT^4 + 4C*T0*ΔT^3 + 6C*T0^2*ΔT^2 + (1+4C*T0^3)*ΔT - A = 0
        where C = K*sigma*BBE, A = K*S, K=(ΔZ/kg)*beta,
              S=(1-albedo)*ΔDSR + BBE*ΔDLR

    Parameters
    ----------
    ... (all inputs are 2D arrays of the same shape) ...
    dt_range : tuple(float, float)
        Acceptable physical range for ΔT in Kelvin (min, max).
    imag_tol : float
        Imaginary tolerance to treat a root as real.

    Returns
    -------
    delta_T : 2D array
        Selected ΔT root (NaN for invalid).
    picked_root_ok : 2D bool
        Whether a physically-plausible real root was found.
    """
    # Shapes & masks
    shape = LST0.shape
    if valid_mask is None:
        valid_mask = (
            np.isfinite(albedo) & np.isfinite(delta_DSR) & np.isfinite(delta_DLR) &
            np.isfinite(beta)   & np.isfinite(LST0)     & np.isfinite(BBE) &
            np.isfinite(kg)     & (kg != 0)             & np.isfinite(delta_Z)
        )

    # Constants
    K = (delta_Z / kg) * beta
    S = (1.0 - albedo) * delta_DSR + BBE * delta_DLR
    A = K * S
    C = K * sigma * BBE

    # Linearized initial guess (for tie-breaking and fallbacks)
    denom = 1.0 + 4.0 * C * (LST0**3)
    deltaT_lin = np.where(valid_mask, (A / denom), np.nan)

    # Output
    delta_T = np.full(shape, np.nan, dtype=float)
    ok = np.zeros(shape, dtype=bool)

    # Helper: residual f(ΔT) for ranking roots
    def residual(dt, A_, C_, T0_):
        return dt + 4*C_*T0_**3*dt + 6*C_*T0_**2*dt**2 + 4*C_*T0_*dt**3 + C_*dt**4 - A_
        # Equivalent to the quartic polynomial value

    # Iterate over valid pixels (numpy.roots works 1D per call)
    rows, cols = np.where(valid_mask)
    dt_min, dt_max = dt_range

    for r, c in zip(rows, cols):
        T0 = LST0[r, c]
        A_ = A[r, c]
        C_ = C[r, c]

        # Degenerate case: C ~ 0 => linear fallback ΔT ≈ A
        if not np.isfinite(C_) or abs(C_) < 1e-20:
            delta_T[r, c] = A_
            ok[r, c] = True
            continue

        # Coefficients for numpy.roots (highest degree first)
        # C*ΔT^4 + 4CT0*ΔT^3 + 6CT0^2*ΔT^2 + (1+4CT0^3*C? no!) -> (1 + 4*C*T0^3)
        coeffs = np.array([
            C_,
            4.0 * C_ * T0,
            6.0 * C_ * T0**2,
            1.0 + 4.0 * C_ * T0**3,
            -A_
        ], dtype=float)

        roots = np.roots(coeffs)  # complex array length 4
        # Keep (approximately) real roots
        real_roots = roots[np.abs(roots.imag) < imag_tol].real

        if real_roots.size == 0:
            # Fallback: use linearized solution
            delta_T[r, c] = deltaT_lin[r, c]
            ok[r, c] = np.isfinite(delta_T[r, c])
            continue

        # Filter by physical range
        cand = real_roots[(real_roots >= dt_min) & (real_roots <= dt_max)]
        if cand.size == 0:
            # If none in range, take the one with smallest residual anyway
            cand = real_roots

        # Rank by |residual|, tie-break by closeness to linearized initial guess
        fvals = np.array([abs(residual(dt, A_, C_, T0)) for dt in cand])
        order = np.argsort(fvals)
        best = cand[order[0]]

        # Tie-break: if multiple similarly good, choose closest to deltaT_lin
        # (we already picked the smallest residual; optional secondary check)
        if cand.size > 1:
            top = cand[order[:min(3, cand.size)]]
            # choose by distance to deltaT_lin
            dists = np.abs(top - deltaT_lin[r, c])
            best = top[np.argmin(dists)]

        delta_T[r, c] = best
        ok[r, c] = True

    return delta_T, ok
sigma = 5.67e-8       # Stefan-Boltzmann  (W/m²·K⁴)
delta_Z = 0.1         #
max_iter = 50         #
tolerance = 1e-3      #
rad_threshold = 20


hour=18

outpath='/Volumes/GYM/DLR/allskyLST/5.delta_T/'

albedo_file='/Volumes/GYM/DLR/ABI/ABI-L2-LSAF/2021/331/18/OR_ABI-L2-LSAF-M6_G16_s20213311800208_e20213311809516_c20213311810398.nc'
delta_DSR_file='/Volumes/GYM/DLR/allskyLST/4.delta_DSRDLR/delta_DSR_on_GOES_grid_20211127.npz'
delta_DLR_file='/Volumes/GYM/DLR/allskyLST/4.delta_DSRDLR/delta_DLR_on_GOES_grid_20211127.npz'
LST_file='/Volumes/GYM/DLR/allskyLST/0.clearskyLST/LST_Downscaling_Model_Output_20211127.nc'
LST_clear='/Volumes/GYM/DLR/ABI/ABI-L2-LST2KMF/2021/331/18/OR_ABI-L2-LST2KMF-M6_G16_s20213311800208_e20213311809516_c20213311810366.nc'


ds_albedo = nc.Dataset(albedo_file)
albedo = ds_albedo.variables['LSA'][:].filled(np.nan)

delta_DSR_npz = np.load(delta_DSR_file)
delta_DSR= np.full(delta_DSR_npz['mask'].shape, np.nan, dtype=np.float32)
delta_DSR[delta_DSR_npz['mask']] = delta_DSR_npz['data']
delta_DSR=delta_DSR[hour,:,:]



delta_DLR_npz = np.load(delta_DLR_file)
delta_DLR= np.full(delta_DLR_npz['mask'].shape, np.nan, dtype=np.float32)
delta_DLR[delta_DLR_npz['mask']] = delta_DLR_npz['data']
delta_DLR=delta_DLR[hour,:,:]


beta = np.load("/Volumes/GYM/DLR/allskyLST/1.beta/GLASS/beta_on_GOES_grid_2021329.npy")



ds = nc.Dataset(LST_file)
LST = ds.variables['LST_high'][:].filled(np.nan)
NDVI=ds.variables['NDVI_high'][:].filled(np.nan)
DEM=ds.variables['DEM_high'][:].filled(np.nan)

LST0 = LST[hour,:,:]  # 晴空下初始 LST（单位 K）

kg = np.load("/Volumes/GYM/DLR/allskyLST/3.Kg/Kg_median_GOES_grid_20211112_20211127_20211212.npy")                  # Kg 系数（单位 W/m²·K）
BBE=np.load('/Volumes/GYM/DLR/BBE/BBE_on_GOES_grid_20211127.npy')


# LST0 adjust

ds_clear=nc.Dataset(LST_clear)
LST_clear=ds_clear.variables['LST'][:].filled(np.nan)
LST_correct=np.load(outpath+f"LST0_correct_20211127_{str(hour)}.npy")
# LST_correct=LST_corrected(LST0,LST_clear,BBE,albedo,NDVI,DEM)
# np.save(outpath+f"LST0_correct_20211127_{str(hour)}.npy", LST_correct)
# plt.figure(figsize=(10, 6))
# plt.imshow(LST_correct, cmap='jet', vmin=240, vmax=320)
# plt.title("Corrected Hypothetical Clear-Sky LST")
# plt.colorbar(label='LST0')
# plt.tight_layout()
# plt.show()

a=1

# ==== 3. mask ====
valid_mask = (~np.isnan(albedo) & ~np.isnan(delta_DSR) & ~np.isnan(delta_DLR) &
              ~np.isnan(beta) & ~np.isnan(LST_correct) & ~np.isnan(BBE) &
              ~np.isnan(kg) & (kg != 0))
print('ready to solve..')
delta_T_fix, conv, n_iter = solve_deltaT_fixed_point(
    albedo, delta_DSR, delta_DLR, beta,
    LST_correct, BBE, kg, delta_Z, sigma,
    valid_mask=valid_mask,
    max_iter=50, tolerance=1e-3, rad_threshold=np.inf,
    verbose=True
)

print('saving delta_T_fix...')
np.save(outpath+f"delta_T_20211127_{str(hour)}_fixpointIteration50.npy", delta_T_fix)

#

delta_T_Newtoniteration50, converged = solve_deltaT_newton(
    albedo, delta_DSR, delta_DLR, beta, LST_correct, BBE, kg, delta_Z, sigma,
    valid_mask=valid_mask, max_iter=50, tol_f=1e-3, tol_x=1e-3,
    step_clip=5.0, damping=0.8
)
print(f"converged pixels = {np.sum(converged)} / {converged.size}")
print('saving delta_T_Newton...')
np.save(outpath+f"delta_T_20211127_{str(hour)}_Newtoniteration50.npy", delta_T_Newtoniteration50)



delta_T_quartic, picked_ok = solve_deltaT_quartic(
    albedo, delta_DSR, delta_DLR, beta, LST_correct, BBE, kg, delta_Z, sigma,
    valid_mask=valid_mask, dt_range=(-20, 20)
)

print("Picked real roots:", np.sum(picked_ok & valid_mask), "/", np.sum(valid_mask))
print('saving delta_T_quartic...')
np.save(outpath+f"delta_T_20211127_{str(hour)}_quartic.npy", delta_T_quartic)


# np.save(outpath+f"delta_T_20211127_{str(hour)}_iteration50.npy", delta_T)
# plt.figure(figsize=(10, 6))
# plt.imshow(delta_T, cmap='inferno')
# plt.title("delta_T")
# plt.colorbar(label='delta_T')
# plt.tight_layout()
# plt.show()




#!/usr/bin/env python
import os
import time
import numpy as np
import dask.array as da
import psutil
from dask.distributed import Client, wait
from dask import config
from dask_jobqueue import SLURMCluster

def main():
    # Increase max message size to 5GB (adjust if needed)
    config.set({'distributed.comm.max_message_size': 5000000000})
    
    # Print memory info
    mem = psutil.virtual_memory()
    print("Total Memory: {:.2f} GB".format(mem.total / 1e9))
    print("Available Memory: {:.2f} GB".format(mem.available / 1e9))
    
    # Create SLURMCluster
    cluster = SLURMCluster(
    account='che190010',
    queue='wholenode',
    cores=1,
    processes=4,
    memory="200GB",
    walltime="03:00:00",
    job_directives_skip=['--mem'],
    # Propagate all environment variables
    
)

    cluster.scale(4)
    time.sleep(3)
    
    client = Client(cluster)
    print("Client connected:", client)
    print("Dashboard:", client.dashboard_link)

    # File paths for merged arrays and selectors
    aw_path = "input/aw99.npy"
    bw_path = "input/bw99.npy"
    energy_selector_file = "input/energy_selector99.npy"
    force_selector_file = "input/force_selector99.npy"

    print("Loading merged Zarr arrays with Dask...")
    aw = np.load(aw_path)
    bw = np.load(bw_path)

    print("Loading merged selectors from disk...")
    energy_selector = np.load(energy_selector_file)
    force_selector = np.load(force_selector_file)
    # Split into training and testing sets
    hlfpnt = aw.shape[0] // 8
    dA_train = aw[hlfpnt:, :]
    
    dA_train = da.from_array(aw[hlfpnt:, :], chunks=(aw[hlfpnt:, :].shape[0] // 20, -1))

    d_b_train = bw[hlfpnt:]
    dA_test = aw[:hlfpnt, :]
    d_b_test = bw[:hlfpnt]


    print(dA_train.shape)
    start_fit = time.time()

    print("Computing TSQR factorization...")
    #Q, R = da.linalg.tsqr(dA_train)
    
    
    U, s, V = da.linalg.svd(dA_train)
    
    print("Computing Qᵀ * y...")
    #Qt_b = da.dot(Q.T, d_b_train)
    
    Ut_y = da.dot(U.T, d_b_train)
    
    print("Solving for regression coefficients...")
    #coef = da.linalg.solve(R, Qt_b).compute()
    coef = da.dot(V.T, (Ut_y / s)).compute()
    
    print("Computing predictions...")
    fit_time = time.time() - start_fit
    print("Model fitting was solved in {:.3f} sec.".format(fit_time))


    y_pred = da.dot(dA_test, coef)

    global_rmse = da.sqrt(((y_pred - d_b_test) ** 2).mean())
    global_rmse = global_rmse.compute()
    print("Global Test RMSE:", global_rmse)

    # Convert selectors to Dask arrays (using a suitable chunk size, here we use hlfpnt)
    energy_selector_d = da.from_array(energy_selector, chunks=hlfpnt)
    force_selector_d  = da.from_array(force_selector, chunks=hlfpnt)

    # Split selectors into training and testing parts
    energy_selector_train = energy_selector_d[:hlfpnt]
    energy_selector_test  = energy_selector_d[hlfpnt:]
    force_selector_train  = force_selector_d[:hlfpnt]
    force_selector_test   = force_selector_d[hlfpnt:]

    # Compute predictions lazily using Dask dot (coef can be a NumPy array)
    train_pred = da.dot(aw[:hlfpnt, :], coef)
    test_pred  = da.dot(aw[hlfpnt:, :], coef)

    # Compute residuals for RMSE (squared differences) and MAE (absolute differences)
    train_residual_sq = da.square(train_pred - bw[:hlfpnt])
    test_residual_sq  = da.square(test_pred - bw[hlfpnt:])
    train_abs_residual = da.abs(train_pred - bw[:hlfpnt])
    test_abs_residual  = da.abs(test_pred - bw[hlfpnt:])

    # Compute RMSE separately. Note: energies are scaled by 22500.
    energy_train_rmse = da.sqrt(da.sum(train_residual_sq * energy_selector_train / 22500) /
                                da.sum(energy_selector_train))
    force_train_rmse  = da.sqrt(da.sum(train_residual_sq * force_selector_train) /
                                da.sum(force_selector_train))
    energy_test_rmse  = da.sqrt(da.sum(test_residual_sq * energy_selector_test / 22500) /
                                da.sum(energy_selector_test))
    force_test_rmse   = da.sqrt(da.sum(test_residual_sq * force_selector_test) /
                                da.sum(force_selector_test))

    # Compute MAE separately. (For energy, note the scaling factor; adjust as needed.)
    energy_train_mae = da.sum(train_abs_residual * energy_selector_train / 150) / da.sum(energy_selector_train)
    force_train_mae  = da.sum(train_abs_residual * force_selector_train) / da.sum(force_selector_train)
    energy_test_mae  = da.sum(test_abs_residual * energy_selector_test / 150) / da.sum(energy_selector_test)
    force_test_mae   = da.sum(test_abs_residual * force_selector_test) / da.sum(force_selector_test)

    # Compute all metrics at once; only final scalars are loaded into memory.
    results = da.compute(energy_train_rmse, force_train_rmse,
                        energy_test_rmse, force_test_rmse,
                        energy_train_mae, force_train_mae,
                        energy_test_mae, force_test_mae)

    (energy_train_rmse_val, force_train_rmse_val,
    energy_test_rmse_val, force_test_rmse_val,
    energy_train_mae_val, force_train_mae_val,
    energy_test_mae_val, force_test_mae_val) = results

    print("Energy Training RMSE:", energy_train_rmse_val)
    print("Force Training RMSE: ", force_train_rmse_val)
    print("Energy Testing RMSE: ", energy_test_rmse_val)
    print("Force Testing RMSE:  ", force_test_rmse_val)

    print("Energy Training MAE:", energy_train_mae_val)
    print("Force Training MAE: ", force_train_mae_val)
    print("Energy Testing MAE: ", energy_test_mae_val)
    print("Force Testing MAE:  ", force_test_mae_val)

    client.close()



if __name__ == "__main__":
    main()

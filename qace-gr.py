#!/usr/bin/env python
import os
import time
import numpy as np
import dask.array as da
import psutil
from dask.distributed import Client, wait
from dask import config, delayed
from dask_jobqueue import SLURMCluster

def load_numpy_chunk(file_path, chunks):
    """Load NumPy array as a Dask array directly on workers."""
    return da.from_array(np.load(file_path, mmap_mode='r'), chunks=chunks)

def main():
    # Increase max message size to 5GB
    config.set({
        'distributed.comm.max_message_size': 5000000000,
        'distributed.comm.timeouts.connect': '60s',  # Increase timeouts for stability
        'distributed.comm.timeouts.tcp': '60s'
    })
    
    # Print memory info
    mem = psutil.virtual_memory()
    print("Total Memory: {:.2f} GB".format(mem.total / 1e9))
    print("Available Memory: {:.2f} GB".format(mem.available / 1e9))
    
    # Create SLURMCluster with your original settings
    cluster = SLURMCluster(
        account='che190010',
        queue='wholenode',
        cores=4,              # 4 cores per worker
        processes=1,          # Single process, rely on threads
        memory="200GB",
        walltime="03:00:00",
        job_directives_skip=['--mem'],
        n_workers=2           # Scale to 2 workers for better parallelism
    )
    
    cluster.scale(2)  # 2 workers, 8 cores total
    time.sleep(5)     # Give cluster time to stabilize
    
    client = Client(cluster)
    print("Client connected:", client)
    print("Dashboard:", client.dashboard_link)

    print("Waiting for 2 workers to spawn...")
    client.wait_for_workers(n_workers=2, timeout=300)  # Wait up to 5 minutes
    print("2 workers are ready!")

    # File paths
    aw_path = "/anvil/projects/x-che190010/dursun/FitSNAP/QuadraticMLIP/examples/fitsnap_qACE/input/aw99.npy"
    bw_path = "/anvil/projects/x-che190010/dursun/FitSNAP/QuadraticMLIP/examples/fitsnap_qACE/input/bw99.npy"
    energy_selector_file = "/anvil/projects/x-che190010/dursun/FitSNAP/QuadraticMLIP/examples/fitsnap_qACE/input/energy_selector99.npy"
    force_selector_file = "/anvil/projects/x-che190010/dursun/FitSNAP/QuadraticMLIP/examples/fitsnap_qACE/input/force_selector99.npy"

    # Load data lazily on workers with optimized chunk size (~1-2 GB)
    print("Loading data with Dask...")
    chunk_size = 20
    aw = client.persist(load_numpy_chunk(aw_path, chunk_size))  
    bw = client.persist(load_numpy_chunk(bw_path, chunk_size))

    energy_selector = client.persist(load_numpy_chunk(energy_selector_file, chunk_size[0]))
    force_selector = client.persist(load_numpy_chunk(force_selector_file, chunk_size[0]))
    wait([aw, bw, energy_selector, force_selector])  # Ensure data is loaded

    # Split into training and testing sets
    hlfpnt = aw.shape[0] // 8
    dA_train = aw[hlfpnt:, :]
    d_b_train = bw[hlfpnt:]
    dA_test = aw[:hlfpnt, :]
    d_b_test = bw[:hlfpnt]

    print("Training shape:", dA_train.shape)
    start_fit = time.time()

    # TSQR factorization with persistence
    print("Computing TSQR factorization...")
    Q, R = da.linalg.tsqr(dA_train)
    Q, R = client.persist([Q, R])  # Persist to avoid recomputation
    wait([Q, R])

    print("Computing Qᵀ * y...")
    Qt_b = da.dot(Q.T, d_b_train).persist()
    wait(Qt_b)

    print("Solving for regression coefficients...")
    coef = da.linalg.solve(R, Qt_b).compute()

    fit_time = time.time() - start_fit
    print("Model fitting solved in {:.3f} sec.".format(fit_time))

    # Compute predictions
    y_pred = da.dot(dA_test, coef)
    global_rmse = da.sqrt(((y_pred - d_b_test) ** 2).mean()).compute()
    print("Global Test RMSE:", global_rmse)

    # Compute metrics with fused operations
    def compute_metrics(pred, true, energy_sel, force_sel):
        residual_sq = da.square(pred - true)
        abs_residual = da.abs(pred - true)
        energy_rmse = da.sqrt(da.sum(residual_sq * energy_sel / 22500) / da.sum(energy_sel))
        force_rmse = da.sqrt(da.sum(residual_sq * force_sel) / da.sum(force_sel))
        energy_mae = da.sum(abs_residual * energy_sel / 150) / da.sum(energy_sel)
        force_mae = da.sum(abs_residual * force_sel) / da.sum(force_sel)
        return energy_rmse, force_rmse, energy_mae, force_mae

    train_pred = da.dot(dA_train, coef)
    test_pred = y_pred  # Reuse from global RMSE
    train_metrics = da.compute(*compute_metrics(train_pred, d_b_train, energy_selector[hlfpnt:], force_selector[hlfpnt:]))
    test_metrics = da.compute(*compute_metrics(test_pred, d_b_test, energy_selector[:hlfpnt], force_selector[:hlfpnt]))

    # Unpack and print results
    energy_train_rmse, force_train_rmse, energy_train_mae, force_train_mae = train_metrics
    energy_test_rmse, force_test_rmse, energy_test_mae, force_test_mae = test_metrics

    print("Energy Training RMSE:", energy_train_rmse)
    print("Force Training RMSE: ", force_train_rmse)
    print("Energy Testing RMSE: ", energy_test_rmse)
    print("Force Testing RMSE:  ", force_test_rmse)
    print("Energy Training MAE:", energy_train_mae)
    print("Force Training MAE: ", force_train_mae)
    print("Energy Testing MAE: ", energy_test_mae)
    print("Force Testing MAE:  ", force_test_mae)

    client.close()

if __name__ == "__main__":
    main()
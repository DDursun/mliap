from mpi4py import MPI
import numpy as np
import time
from sklearn.linear_model import LinearRegression

def compute_local_coef(A_local, b_local):
    # Use scikit-learn LinearRegression without intercept
    model = LinearRegression(fit_intercept=False)
    model.fit(A_local, b_local)
    return model.coef_

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    aw_path = "/anvil/projects/x-che190010/dursun/FitSNAP/QuadraticMLIP/examples/fitsnap_qACE/input/aw99.npy"
    bw_path = "/anvil/projects/x-che190010/dursun/FitSNAP/QuadraticMLIP/examples/fitsnap_qACE/input/bw99.npy"
    energy_selector_path = "/anvil/projects/x-che190010/dursun/FitSNAP/QuadraticMLIP/examples/fitsnap_qACE/input/energy_selector99.npy"
    force_selector_path = "/anvil/projects/x-che190010/dursun/FitSNAP/QuadraticMLIP/examples/fitsnap_qACE/input/force_selector99.npy"

    A_all = np.load(aw_path, mmap_mode='r')
    b_all = np.load(bw_path, mmap_mode='r')
    n_samples = A_all.shape[0]

    test_count = n_samples // 8
    train_count = n_samples - test_count

    if rank == 0:
        A_test = A_all[:test_count, :]
        b_test = b_all[:test_count]
        A_train = A_all[test_count:, :]
        b_train = b_all[test_count:]
        energy_selector = np.load(energy_selector_path, mmap_mode='r')
        force_selector = np.load(force_selector_path, mmap_mode='r')
        energy_selector_train = energy_selector[test_count:]
        energy_selector_test = energy_selector[:test_count]
        force_selector_train = force_selector[test_count:]
        force_selector_test = force_selector[:test_count]
    else:
        A_test = b_test = A_train = b_train = None
        energy_selector_train = energy_selector_test = None
        force_selector_train = force_selector_test = None

    # Determine training data partition for worker ranks (ranks 1 to size-1)
    base = train_count // (size - 1)
    remainder = train_count % (size - 1)
    worker_id = rank - 1  # for worker processes

    # Synchronize all processes before training starts
    comm.Barrier()

    if rank != 0:
        # Determine local training slice indices
        if worker_id < remainder:
            local_count = base + 1
            local_start = worker_id * (base + 1)
        else:
            local_count = base
            local_start = remainder * (base + 1) + (worker_id - remainder) * base
        global_start = test_count + local_start
        global_end = global_start + local_count

        A_local = A_all[global_start:global_end, :]
        b_local = b_all[global_start:global_end]

        # Record local training start time immediately after barrier
        t_local_start = time.time()
        coef_local = compute_local_coef(A_local, b_local)
        t_local_elapsed = time.time() - t_local_start
        print("Rank {} computed local model in {:.3f} sec on {} samples".format(rank, t_local_elapsed, local_count))
        
        # Send both the computed coefficients and the start time to rank 0
        comm.send((coef_local, t_local_start), dest=0, tag=22)

    if rank == 0:
        local_coefs = []
        local_start_times = []
        for src in range(1, size):
            coef_local, t_local_start = comm.recv(source=src, tag=22)
            local_coefs.append(coef_local)
            local_start_times.append(t_local_start)
        ensemble_coef = np.mean(local_coefs, axis=0)

        # Global training time: from the earliest worker start to now (coefficients are ready)
        global_end_time = time.time()
        earliest_start = min(local_start_times)
        global_training_time = global_end_time - earliest_start
        print("Global training time: {:.3f} sec".format(global_training_time))

        test_pred = A_test @ ensemble_coef
        train_pred = A_train @ ensemble_coef

        global_rmse = np.sqrt(np.mean((test_pred - b_test) ** 2))
        print("Global Test RMSE:", global_rmse)

        train_residual_sq = (train_pred - b_train) ** 2
        test_residual_sq = (test_pred - b_test) ** 2
        train_abs_residual = np.abs(train_pred - b_train)
        test_abs_residual = np.abs(test_pred - b_test)

        energy_train_rmse = np.sqrt(np.sum(train_residual_sq * energy_selector_train / 22500) / np.sum(energy_selector_train))
        force_train_rmse = np.sqrt(np.sum(train_residual_sq * force_selector_train) / np.sum(force_selector_train))
        energy_test_rmse = np.sqrt(np.sum(test_residual_sq * energy_selector_test / 22500) / np.sum(energy_selector_test))
        force_test_rmse = np.sqrt(np.sum(test_residual_sq * force_selector_test) / np.sum(force_selector_test))

        energy_train_mae = np.sum(train_abs_residual * energy_selector_train / 150) / np.sum(energy_selector_train)
        force_train_mae = np.sum(train_abs_residual * force_selector_train) / np.sum(force_selector_train)
        energy_test_mae = np.sum(test_abs_residual * energy_selector_test / 150) / np.sum(energy_selector_test)
        force_test_mae = np.sum(test_abs_residual * force_selector_test) / np.sum(force_selector_test)

        print("Energy Training RMSE:", energy_train_rmse)
        print("Force Training RMSE: ", force_train_rmse)
        print("Energy Testing RMSE: ", energy_test_rmse)
        print("Force Testing RMSE:  ", force_test_rmse)
        print("Energy Training MAE:", energy_train_mae)
        print("Force Training MAE: ", force_train_mae)
        print("Energy Testing MAE: ", energy_test_mae)
        print("Force Testing MAE:  ", force_test_mae)

if __name__ == "__main__":
    main()

import os
import time
import numpy as np
import psutil

def main():
    mem = psutil.virtual_memory()
    print("Total Memory: {:.2f} GB".format(mem.total / 1e9))
    print("Available Memory: {:.2f} GB".format(mem.available / 1e9))

    aw_path = "/anvil/projects/x-che190010/dursun/FitSNAP/QuadraticMLIP/examples/fitsnap_qACE/input/aw99.npy"
    bw_path = "/anvil/projects/x-che190010/dursun/FitSNAP/QuadraticMLIP/examples/fitsnap_qACE/input/bw99.npy"
    energy_selector_file = "/anvil/projects/x-che190010/dursun/FitSNAP/QuadraticMLIP/examples/fitsnap_qACE/input/energy_selector99.npy"
    force_selector_file = "/anvil/projects/x-che190010/dursun/FitSNAP/QuadraticMLIP/examples/fitsnap_qACE/input/force_selector99.npy"

    print("Loading data from .npy files...")
    aw_np = np.load(aw_path)
    bw_np = np.load(bw_path)
    energy_selector = np.load(energy_selector_file)
    force_selector = np.load(force_selector_file)

    print("aw shape:", aw_np.shape)
    print("bw shape:", bw_np.shape)
    print("energy selector shape:", energy_selector.shape)
    print("force selector shape:", force_selector.shape)

    # Shape checks
    assert bw_np.shape[0] == aw_np.shape[0], "Mismatch: bw and aw row counts differ"
    assert energy_selector.shape[0] == bw_np.shape[0], "Mismatch: energy_selector and bw row counts differ"
    assert force_selector.shape[0] == bw_np.shape[0], "Mismatch: force_selector and bw row counts differ"

    reduced_aw_np = aw_np  # No dimensionality reduction applied here

    hlfpnt = reduced_aw_np.shape[0] // 8
    dA_test = reduced_aw_np[:hlfpnt, :]
    d_b_test = bw_np[:hlfpnt]
    dA_train = reduced_aw_np[hlfpnt:, :]
    d_b_train = bw_np[hlfpnt:]

    energy_selector_test = energy_selector[:hlfpnt]
    energy_selector_train = energy_selector[hlfpnt:]
    force_selector_test = force_selector[:hlfpnt]
    force_selector_train = force_selector[hlfpnt:]

    start = time.time()
    print("Computing SVD on training data...")
    U, s, Vt = np.linalg.svd(dA_train, full_matrices=False)
    Ut_y = np.dot(U.T, d_b_train)
    coef = np.dot(Vt.T, (Ut_y / s))
    coef_time = time.time() - start
    print("Regression was solved in {:.3f} sec.".format(coef_time))

    print("Computing predictions on test data...")
    test_pred = np.dot(dA_test, coef)
    train_pred = np.dot(dA_train, coef)

    global_rmse = np.sqrt(np.mean((test_pred - d_b_test) ** 2))
    print("Global Test RMSE:", global_rmse)

    train_residual_sq = (train_pred - d_b_train) ** 2
    test_residual_sq = (test_pred - d_b_test) ** 2
    train_abs_residual = np.abs(train_pred - d_b_train)
    test_abs_residual = np.abs(test_pred - d_b_test)

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

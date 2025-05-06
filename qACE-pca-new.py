import os
import time
import numpy as np
import psutil
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression

def main():
    # Memory Information
    mem = psutil.virtual_memory()
    print("Total Memory: {:.2f} GB".format(mem.total / 1e9))
    print("Available Memory: {:.2f} GB".format(mem.available / 1e9))

    # File Paths
    aw_path = "/anvil/projects/x-che190010/dursun/FitSNAP/QuadraticMLIP/examples/fitsnap_qACE/input/aw99.npy"
    bw_path = "/anvil/projects/x-che190010/dursun/FitSNAP/QuadraticMLIP/examples/fitsnap_qACE/input/bw99.npy"
    energy_selector_file = "/anvil/projects/x-che190010/dursun/FitSNAP/QuadraticMLIP/examples/fitsnap_qACE/input/energy_selector99.npy"
    force_selector_file = "/anvil/projects/x-che190010/dursun/FitSNAP/QuadraticMLIP/examples/fitsnap_qACE/input/force_selector99.npy"

    # Load Data
    aw_np = np.load(aw_path)
    bw_np = np.load(bw_path)
    energy_selector = np.load(energy_selector_file)
    force_selector = np.load(force_selector_file)

    print("aw shape:", aw_np.shape)
    print("bw shape:", bw_np.shape)

    # Split first (no leak)
    hlfpnt = aw_np.shape[0] // 8
    X_test_raw = aw_np[:hlfpnt, :]
    y_test = bw_np[:hlfpnt]
    X_train_raw = aw_np[hlfpnt:, :]
    y_train = bw_np[hlfpnt:]

    energy_selector_test = energy_selector[:hlfpnt]
    energy_selector_train = energy_selector[hlfpnt:]
    force_selector_test = force_selector[:hlfpnt]
    force_selector_train = force_selector[hlfpnt:]

    # Feature Selection + PCA on train set
    top_percent = 25
    pca_percent = 25
    num_features = aw_np.shape[1]
    k_best_features = int(num_features * top_percent / 100)
    pca_components = int(num_features * pca_percent / 100)

    start_fs = time.time()

    # 1. SelectKBest
    selector = SelectKBest(score_func=f_regression, k=k_best_features)
    selector.fit(X_train_raw, y_train)
    X_train_sel = selector.transform(X_train_raw)
    X_test_sel = selector.transform(X_test_raw)
    selected_indices = selector.get_support(indices=True)

    # 2. PCA on remaining features
    remaining_indices = [i for i in range(num_features) if i not in selected_indices]
    X_train_remain = X_train_raw[:, remaining_indices]
    X_test_remain = X_test_raw[:, remaining_indices]

    pca = PCA(n_components=pca_components)
    X_train_pca = pca.fit_transform(X_train_remain)
    X_test_pca = pca.transform(X_test_remain)

    # 3. Combine selected + PCA
    dA_train = np.hstack((X_train_sel, X_train_pca))
    dA_test = np.hstack((X_test_sel, X_test_pca))

    feature_selection_time = time.time() - start_fs
    print(f"Feature selection + PCA done in {feature_selection_time:.3f} sec.")

    # SVD fitting
    start = time.time()
    print("Computing SVD on training data...")
    U, s, Vt = np.linalg.svd(dA_train, full_matrices=False)
    Ut_y = np.dot(U.T, y_train)
    coef = np.dot(Vt.T, (Ut_y / s))
    coef_time = time.time() - start
    print("Regression was solved in {:.3f} sec.".format(coef_time))

    print("Computing predictions on test data...")
    test_pred = np.dot(dA_test, coef)
    train_pred = np.dot(dA_train, coef)

    # Print overall RMSE
    global_rmse = np.sqrt(np.mean((test_pred - y_test) ** 2))
    print("Global Test RMSE:", global_rmse)

    # Per-group metrics
    train_residual_sq = (train_pred - y_train) ** 2
    test_residual_sq = (test_pred - y_test) ** 2
    train_abs_residual = np.abs(train_pred - y_train)
    test_abs_residual = np.abs(test_pred - y_test)

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

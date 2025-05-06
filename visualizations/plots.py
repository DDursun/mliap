import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# === Data ===
experiments = [
    "Linear ACE",
    "Neural SNAP",
    "qACE",
    "qACE-Dask",
    "Batch",
    "Half-columns",
    "Half-rows",
    "Half-PCA",
    "Half-smart"
]

energy_rmse = np.array([10.33, 6.48, 5.97, 5.77, 6.34, 6.37, 5.99, 22.53, 6.33])
force_rmse = np.array([151, 161, 106, 106, 108, 108.6, 107, 324, 110.4])
feature_time = np.array([0, 0, 0, 0, 0, 9.4, 0, 1288, 115.9])
training_time = np.array([7.9, 30038, 848.9, 274.8, 137, 264.5, 418.6, 225, 225.5])
total_runtime = feature_time + training_time

# Filter for Pareto plots
pareto_experiments = ["qACE", "Batch", "Half-columns", "Half-rows", "Half-smart", "qACE-Dask"]
pareto_energy = energy_rmse[[experiments.index(e) for e in pareto_experiments]]
pareto_force = force_rmse[[experiments.index(e) for e in pareto_experiments]]
pareto_runtime = total_runtime[[experiments.index(e) for e in pareto_experiments]]

# === Define markers and colors by group ===
marker_dict = {
    "qACE": ('o', 'red'),
    "Batch": ('X', 'royalblue'),
    "qACE-Dask": ('P', 'royalblue'),
    "Half-columns": ('^', 'forestgreen'),
    "Half-rows": ('v', 'forestgreen'),
    "Half-smart": ('D', 'forestgreen')
}

# === Helper function for scatter plot with group and marker legends ===
def plot_pareto(runtime, rmse, ylabel, title, ylim_range, savefile, group_legend_location='lower right'):
    plt.figure(figsize=(10, 6))
    
    handles = []
    labels = []

    for name in ["qACE", "Batch", "qACE-Dask", "Half-columns", "Half-rows", "Half-smart"]:
        i = pareto_experiments.index(name)
        marker, color = marker_dict[name]
        sc = plt.scatter(runtime[i], rmse[i], color=color, marker=marker, s=100, label=name)
        plt.annotate(name, (runtime[i], rmse[i]),
                     textcoords="offset points", xytext=(0, 10), ha='center', va='bottom', fontsize=11)
        handles.append(sc)
        labels.append(name)

    plt.xlabel('Training time (s)', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.grid(color='lightgray', linestyle='--', linewidth=0.5)

    x_margin = (max(runtime) - min(runtime)) * 0.05
    plt.xlim(min(runtime) - x_margin, max(runtime) + x_margin)
    plt.ylim(ylim_range)

    # First legend: specific experiments (no fit line)
    legend1 = plt.legend(handles, labels, fontsize=9, loc='upper right', frameon=False)

    # Second group legend: rectangles for color meaning
    extra_patches = [
        Patch(facecolor='forestgreen', edgecolor='none', label='Data Reduction'),
        Patch(facecolor='royalblue', edgecolor='none', label='Parallelization')
    ]
    plt.gca().add_artist(legend1)
    plt.legend(handles=extra_patches, fontsize=9, loc=group_legend_location, frameon=False)

    plt.tight_layout()
    plt.savefig(savefile)
    plt.close()

# === PLOT 2: Energy (group legend on lower right) ===
plot_pareto(
    pareto_runtime, pareto_energy,
    'Energy RMSE (meV)', 'Energy RMSE vs Training time',
    (5.73, 6.45), "pareto_energy_runtime.png",
    group_legend_location='lower right'
)

# === PLOT 3: Force (group legend on lower left) ===
plot_pareto(
    pareto_runtime, pareto_force,
    'Force RMSE (meV/Å)', 'Force RMSE vs Training time',
    (105.5, 111.1), "pareto_force_runtime.png",
    group_legend_location='lower left'
)

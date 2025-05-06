import matplotlib.pyplot as plt
import numpy as np

# === Filtered Data ===
selected_experiments = ["Half-columns", "Half-PCA", "Half-smart"]
energy_rmse = np.array([6.37, 22.53, 6.33])
force_rmse = np.array([108.6, 324, 110.4])
preproc_time = np.array([9.4, 1288, 115.9])
training_time = np.array([264.5, 225, 225.5])
total_runtime = preproc_time + training_time

# === Bar Chart ===
x = np.arange(len(selected_experiments))
width = 0.25
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

# Energy RMSE + Runtime
ax1.bar(x - width / 2, energy_rmse, width, color='tab:red', label='Energy RMSE (meV)')
for i, val in enumerate(energy_rmse):
    ax1.text(x[i] - width / 2, val + 0.4, f'{val:.2f}', ha='center', fontsize=10)

ax1.set_ylim(0, max(energy_rmse) + 5)
ax1.set_ylabel('Energy RMSE (meV)')
ax1.set_title('Energy RMSE and Runtime')
ax1.legend(loc='upper left')

ax1b = ax1.twinx()
ax1b.bar(x + width / 2, training_time, width, color='steelblue', label='Training time')
ax1b.bar(x + width / 2, preproc_time, width, bottom=training_time, color='lightblue', label='Preprocessing time')
for i, total in enumerate(total_runtime):
    ax1b.text(x[i] + width / 2, training_time[i] + preproc_time[i] + 20, f'{total:.0f}s', ha='center', fontsize=10)
ax1b.set_ylim(0, max(total_runtime) + 100)
ax1b.set_ylabel('Runtime (s)')
ax1b.legend(loc='upper right')

# Force RMSE + Runtime
ax2.bar(x - width / 2, force_rmse, width, color='orange', label='Force RMSE (meV/Å)')
for i, val in enumerate(force_rmse):
    ax2.text(x[i] - width / 2, val + 5, f'{val:.1f}', ha='center', fontsize=10)

ax2.set_ylim(0, max(force_rmse) + 50)
ax2.set_ylabel('Force RMSE (meV/Å)')
ax2.set_title('Force RMSE and Runtime')
ax2.legend(loc='upper left')

ax2b = ax2.twinx()
ax2b.bar(x + width / 2, training_time, width, color='steelblue', label='Training time')
ax2b.bar(x + width / 2, preproc_time, width, bottom=training_time, color='lightblue', label='Preprocessing time')
for i, total in enumerate(total_runtime):
    ax2b.text(x[i] + width / 2, training_time[i] + preproc_time[i] + 20, f'{total:.0f}s', ha='center', fontsize=10)
ax2b.set_ylim(0, max(total_runtime) + 100)
ax2b.set_ylabel('Runtime (s)')
ax2b.legend(loc='upper right')

wrapped_labels = [e.replace(" ", "\n").replace("w/", "w/\n") for e in selected_experiments]
plt.xticks(x, wrapped_labels, rotation=0, ha='center')
plt.tight_layout()
plt.savefig("bar_half_selected_fixed_ylim.png")
plt.close()

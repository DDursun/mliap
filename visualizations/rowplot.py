import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# Data
df = pd.DataFrame({
    "Fraction of Training Data (%)": [100, 50, 25, 10],
    "Training Time (s)": [848.876, 418.643, 197.748, 71.251],
    "Energy RMSE (meV)": [5.9711, 5.9903, 6.1360, 7.0945],
    "Force RMSE (meV/Å)": [106.2462, 107.1032, 110.5010, 140.2544]
})

training_time = df["Training Time (s)"]
energy_rmse = df["Energy RMSE (meV)"]
force_rmse = df["Force RMSE (meV/Å)"]
fractions = df["Fraction of Training Data (%)"]

# Constants
marker_width = 20
energy_marker_height = 0.1
force_marker_height = 4.0
blue_color = '#1f77b4'
red_color = '#800020'

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Dotted line plots
ax1.plot(training_time, energy_rmse, color=blue_color, alpha=0.7, linewidth=2, linestyle='dotted')
ax2.plot(training_time, force_rmse, color=red_color, alpha=0.7, linewidth=2, linestyle='dotted')

# Plot Energy RMSE
for i in range(len(df)):
    x = training_time[i]
    y = energy_rmse[i]
    frac = fractions[i] / 100

    full_rect = mpatches.Rectangle((x - marker_width / 2, y - energy_marker_height / 2),
                                   marker_width, energy_marker_height,
                                   facecolor='none', edgecolor='black', zorder=3)
    filled_rect = mpatches.Rectangle((x - marker_width / 2, y - energy_marker_height / 2),
                                     marker_width, energy_marker_height * frac,
                                     facecolor=blue_color, edgecolor='black', zorder=3)
    ax1.add_patch(full_rect)
    ax1.add_patch(filled_rect)

    # Add label
    if fractions[i] in [25, 50, 100]:
        ax1.text(x, y + 0.07, f"{int(fractions[i])}%", ha='center', fontsize=12, color='black')
    else:
        ax1.text(x + marker_width / 2 + 10, y, f"{int(fractions[i])}%", va='center', fontsize=12, color='black')

ax1.set_ylabel("Energy RMSE (meV)", color='black', fontsize=14)
ax1.tick_params(axis='y', labelcolor='black', labelsize=12)
ax1.set_title("Energy RMSE vs Training Time", fontsize=16)
ax1.grid(True)

# Plot Force RMSE
for i in range(len(df)):
    x = training_time[i]
    y = force_rmse[i]
    frac = fractions[i] / 100

    full_rect = mpatches.Rectangle((x - marker_width / 2, y - force_marker_height / 2),
                                   marker_width, force_marker_height,
                                   facecolor='none', edgecolor='black', zorder=3)
    filled_rect = mpatches.Rectangle((x - marker_width / 2, y - force_marker_height / 2),
                                     marker_width, force_marker_height * frac,
                                     facecolor=red_color, edgecolor='black', zorder=3)
    ax2.add_patch(full_rect)
    ax2.add_patch(filled_rect)

    # Add label
    if fractions[i] in [25, 50, 100]:
        ax2.text(x, y + 2.5, f"{int(fractions[i])}%", ha='center', fontsize=12, color='black')
    else:
        ax2.text(x + marker_width / 2 + 10, y, f"{int(fractions[i])}%", va='center', fontsize=12, color='black')

ax2.set_xlabel("Training Time (s)", fontsize=14)
ax2.set_ylabel("Force RMSE (meV/Å)", color='black', fontsize=14)
ax2.tick_params(axis='y', labelcolor='black', labelsize=12)
ax2.tick_params(axis='x', labelsize=12)
ax2.set_title("Force RMSE vs Training Time", fontsize=16)
ax2.grid(True)

plt.tight_layout()
plt.savefig('rmse_vs_training_time_dotted.png', dpi=300, bbox_inches='tight')
plt.show()

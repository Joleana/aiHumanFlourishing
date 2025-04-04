import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt

# ------------------ Load Data ------------------
forced_mode = "replenish"
forced_mode_value = ""
forced_mode_value += f"_{forced_mode}"
filename = f"simulation_results_high_cost{forced_mode_value}.csv"
# Load your simulation data (replace with your actual file path)
simulation_data = pd.read_csv(f"{filename}")

# Define Ryff dimensions
dimensions = ['Purpose', 'Autonomy', 'Personal Growth', 
              'Environmental Mastery', 'Positive Relations', 'Self-Acceptance']

# ------------------ Descriptive Statistics ------------------
# Mean and standard deviation of each dimension
print(f"Descriptive Statistics (Means and STD) by Environment: {forced_mode}")
descriptive_stats = simulation_data.groupby('environment')[dimensions].agg(['mean', 'std'])
print(descriptive_stats)

# Superstimuli interaction in mixed environment
mixed_data = simulation_data[simulation_data['environment'] == 'mixed']
mean_superstimuli_time = mixed_data['superstimuli_saturations'].mean()
std_superstimuli_time = mixed_data['superstimuli_saturations'].std()
print(f"\nAverage time spent interacting with superstimuli: {mean_superstimuli_time:.2f} (SD={std_superstimuli_time:.2f})")

# ------------------ Inferential Statistics ------------------
print(f"\nDimension Level Comparison (Mixed vs Nourishing): {forced_mode}")
for dim in dimensions:
    mixed = simulation_data[simulation_data['environment'] == 'mixed'][dim]
    nourishing = simulation_data[simulation_data['environment'] == 'nourishing'][dim]
    
    stat, p = mannwhitneyu(mixed, nourishing, alternative='two-sided')
    print(f"{dim}: U={stat:.2f}, p-value={p:.4f}")

# Engagement time comparison
if 'ryff_saturations' in simulation_data.columns:
    mixed_replenish = mixed_data['ryff_saturations']
    nourishing_replenish = simulation_data[simulation_data['environment'] == 'nourishing']['ryff_saturations']
    
    stat, p = mannwhitneyu(mixed_replenish, nourishing_replenish, alternative='two-sided')
    print(f"\nReplenishing Time (Mixed vs Nourishing): U={stat:.2f}, p-value={p:.4f}")

env_value = simulation_data['environment'].unique()[0].capitalize()
if len(forced_mode_value) > 0:
    forced_mode_value = forced_mode_value.replace('_', ' ').title()
else:
    forced_mode_value = 'Dynamic'
# ------------------ Visualization ------------------
# Boxplots for Ryff dimensions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, dim in enumerate(dimensions):
    simulation_data.boxplot(column=dim, by='environment', ax=axes[i])
    axes[i].set_title(dim)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Saturation Level')
    if dim == 'Purpose':
        axes[i].text(1.5, 90, 'Purpose hijacked by\n"Outrage Media"\nin Mixed Environment', 
                     fontsize=10, color='red', ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.5))
    else:
        axes
print(forced_mode_value)
plt.suptitle(f"Ryff Dimension Levels by Mixed vs. Nourishing Environmnet: {forced_mode_value} Mode", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Pie chart for engagement time in mixed environment
if 'mixed' in simulation_data['environment'].unique():
    mixed_data = simulation_data[simulation_data['environment'] == 'mixed']
    if not mixed_data.empty:
        mean_superstimuli_time = mixed_data['time_superstimuli'].mean()
        mean_replenishing_time = mixed_data['time_replenishing'].mean()
        engagement_times = [mean_superstimuli_time, mean_replenishing_time]
        plt.figure(figsize=(8,6))
        plt.pie(engagement_times, labels=['Superstimuli', 'Replenishing'], autopct='%1.1f%%', colors=['#FF000090', 'xkcd:lavender pink'], startangle=90)
        plt.title(f"Average Engagement Time in Mixed Environment: {forced_mode_value} Mode")
        plt.axis('equal')
        plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Read the CSV file
df = pd.read_csv("models/gpt2-ag_news/all_checkpoints_evaluation_results_safety_copy.csv")  # Replace with your actual filename

# Remove duplicates
df = df.drop_duplicates()

# Function to create group names
def get_group_name(model_name):
    # Special case for train_full models
    if model_name.startswith('train_full_'):
        return 'train_full'
    
    # For other models, remove the number between penultimate and ultimate underscore
    parts = model_name.split('_')
    if len(parts) >= 2:
        # Remove the penultimate part (the varying number)
        group_parts = parts[:-1]  # Remove last part temporarily
        if group_parts:
            group_parts = group_parts[:-1] + [parts[-1]]  # Remove penultimate, keep last
            return '_'.join(group_parts)
    
    return model_name

# Create group column
df['group'] = df.iloc[:, 0].apply(get_group_name)  # Assuming first column is model name

# Group by group name and checkpoint, calculate mean and 95% CI
grouped = df.groupby(['group', df.columns[1]]).agg({  # Assuming second column is checkpoint
    df.columns[3]: ['mean', 'std', 'count']  # Assuming fourth column is accuracy
}).reset_index()

# Flatten column names
grouped.columns = ['group', 'checkpoint', 'mean_accuracy', 'std_accuracy', 'count']

# Calculate 95% confidence interval
grouped['ci_95'] = grouped.apply(
    lambda row: stats.t.ppf(0.975, row['count']-1) * row['std_accuracy'] / np.sqrt(row['count']) 
    if row['count'] > 1 else 0, 
    axis=1
)

# Create the plot
plt.figure(figsize=(12, 7))

# Get unique groups and assign colors
groups = grouped['group'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))

# Plot each group
for i, group in enumerate(groups):
    if ("top" in group and "C-1" in group) or "full" in group:
        group_data = grouped[grouped['group'] == group].sort_values('checkpoint')
        
        plt.errorbar(
            group_data['checkpoint'], 
            group_data['mean_accuracy'],
            yerr=group_data['ci_95'],
            label=group,
            color=colors[i],
            marker='o',
            capsize=5,
            capthick=2,
            linewidth=2,
            markersize=6
        )

plt.xscale('log')
plt.xlabel('Checkpoint Number (log scale)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Model Accuracy vs Checkpoint (Mean ± 95% CI)', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("accuracy_vs_step_grouped_err.png")

# Optional: Print summary statistics
print("\nSummary Statistics:")
print("="*80)
for group in groups:
    group_data = df[df['group'] == group]
    print(f"\n{group}:")
    print(f"  Number of models: {group_data.iloc[:, 0].nunique()}")
    print(f"  Total data points: {len(group_data)}")
    print(f"  Checkpoint range: {group_data.iloc[:, 1].min()} - {group_data.iloc[:, 1].max()}")
    print(f"  Accuracy range: {group_data.iloc[:, 3].min():.4f} - {group_data.iloc[:, 3].max():.4f}")
import pandas as pd
import matplotlib.pyplot as plt

# Load the results CSV
data = pd.read_csv("results_2.csv", sep='\t')

# Ensure numeric conversion
data['K'] = pd.to_numeric(data['K'], errors='coerce')
data['Precision'] = pd.to_numeric(data['Precision'], errors='coerce')
data['Recall'] = pd.to_numeric(data['Recall'], errors='coerce')
data['F1'] = pd.to_numeric(data['F1'], errors='coerce')

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(21, 12), sharex=False, sharey=False)


# Define tasks and metrics
tasks = ['overall', 'rare']
metrics = ['Precision', 'Recall', 'F1']

# Plotting function
def plot_kernel_comparison_by_task(data, metric, ax, title):
    ax.set_xlabel('K', fontsize=14)
    ax.set_ylabel(metric, fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0, 1)
    ax.grid(True)
    # for kernel in ['GCN', 'GIN']:
    for aggregator in ['avg', 'max']:
        subset = data[data['Aggregator'] == aggregator]
        ax.plot(subset['K'], subset[metric], label=aggregator, marker='o')

# Generate plots
for i, task in enumerate(tasks):
    task_data = data[
        (data['Prediction_Task'] == task) &
        (data['Kernel'] == 'GCN') #&
        #(data['Aggregator'] == 'max')
    ]
    for j, metric in enumerate(metrics):
        ax = axes[i, j]
        plot_kernel_comparison_by_task(task_data, metric, ax,
            f"{metric} vs K for GCN's Avg and Max Aggregators")

        ax.legend()

plt.tight_layout()
plt.savefig('gcn_avg_vs_max.png')
# plt.show()


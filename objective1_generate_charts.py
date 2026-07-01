import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

INPUT_CSV  = "./objective1_scaling_results.csv"
OUTPUT_DIR = "./graphs_comparisons"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_CSV)

# ── Consistent styling across all charts ─────────────────────────────────
STRATEGY_ORDER  = [
    'Our Framework (K-Means+++NSGA-II)',
    'PSO Scheduler',
    'GA Scheduler',
    'Weighted Sum Method'
]
STRATEGY_LABELS = [
    'Proposed Framework\n(K-Means++ + NSGA-II)',
    'PSO Scheduler',
    'GA Scheduler',
    'Weighted Sum Method\n(WSM)'
]
STRATEGY_COLORS = ['#1B3A6B', '#E67E22', '#2ECC71', '#95A5A6'] # Dark Blue, Orange, Green, Grey

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.0,
})

# ── Metric definitions: (column, y-axis label, filename, title) ─────────
METRICS = [
    ('makespan_s', 'Makespan Time for Scheduling (s)',
     'fig_makespan', 'Makespan scaling over task volumes'),
    ('total_energy_mJ', 'Total Energy Consumption (mJ)',
     'fig_energy', 'Energy consumption scaling'),
    ('network_load_KB', 'Network Load at Task Execution (KB)',
     'fig_network', 'Network load scaling'),
    ('transmission_delay_ms', 'Transmission Delay (ms)',
     'fig_transmission_delay', 'Transmission delay scaling'),
    ('response_time_ms', 'Response Time (ms)',
     'fig_response_time', 'Response time scaling'),
    ('cpu_time_ms', 'CPU Time (ms)',
     'fig_cpu_time', 'CPU execution time scaling'),
    ('resource_utilization_pct', 'Resource Utilization (%)',
     'fig_resource_util', 'Resource utilization scaling'),
    ('throughput_tasks_per_s', 'Throughput (tasks/s)',
     'fig_throughput', 'System throughput scaling'),
]


def plot_grouped_bar(metric_col, ylabel, filename, title):
    task_counts = sorted(df['no_of_tasks'].unique())
    n_groups = len(task_counts)
    n_bars   = len(STRATEGY_ORDER)
    bar_width = 0.85 / n_bars
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for i, (strategy, label, color) in enumerate(
            zip(STRATEGY_ORDER, STRATEGY_LABELS, STRATEGY_COLORS)):
        values = []
        for tc in task_counts:
            row = df[(df['no_of_tasks'] == tc) & (df['strategy'] == strategy)]
            values.append(row[metric_col].values[0] if len(row) else 0)
        offset = (i - (n_bars - 1) / 2) * bar_width
        ax.bar(x + offset, values, bar_width, label=label,
               color=color, edgecolor='#222222', linewidth=0.6)

    ax.set_xlabel('No. of Tasks', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([str(tc) for tc in task_counts])
    # Place legend outside or neatly inside
    ax.legend(loc='upper left', frameon=True, fontsize=9.5)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"{filename}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_degree_of_imbalance():
    task_counts = sorted(df['no_of_tasks'].unique())
    # Plot only for the metaheuristic multi-node schedulers (proposed, pso, ga, wsm)
    n_groups = len(task_counts)
    n_bars   = len(STRATEGY_ORDER)
    bar_width = 0.85 / n_bars
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for i, (strategy, label, color) in enumerate(
            zip(STRATEGY_ORDER, STRATEGY_LABELS, STRATEGY_COLORS)):
        values = []
        for tc in task_counts:
            row = df[(df['no_of_tasks'] == tc) & (df['strategy'] == strategy)]
            values.append(row['degree_of_imbalance'].values[0] if len(row) and not pd.isna(row['degree_of_imbalance'].values[0]) else 0)
        offset = (i - (n_bars - 1) / 2) * bar_width
        ax.bar(x + offset, values, bar_width, label=label,
               color=color, edgecolor='#222222', linewidth=0.6)

    ax.set_xlabel('No. of Tasks', fontsize=12, fontweight='bold')
    ax.set_ylabel('Degree of Imbalance (CV of node loads)', fontsize=12, fontweight='bold')
    ax.set_title('Load Imbalance Comparison', fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([str(tc) for tc in task_counts])
    ax.legend(loc='upper left', frameon=True, fontsize=9.5)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig_degree_of_imbalance.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    print("[PLOTTING] Generating Objective 1 scaling charts (without pure edge/cloud baselines)...")
    for col, ylabel, filename, title in METRICS:
        plot_grouped_bar(col, ylabel, filename, title)
    plot_degree_of_imbalance()
    print("[PLOTTING] Complete! All scaling charts saved to graphs_comparisons/")

if __name__ == '__main__':
    main()
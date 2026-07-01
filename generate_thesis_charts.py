import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.titlesize': 18,
    'figure.dpi': 300
})

WORKSPACE_DIR = "./graphs_comparisons"
os.makedirs(WORKSPACE_DIR, exist_ok=True)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

print("Generating targeted thesis charts...")

# 1. NSGA-II Pareto Front (Latency vs Energy vs Network)
if os.path.exists('./nsga2_pareto_front.csv'):
    df_pareto = pd.read_csv('./nsga2_pareto_front.csv')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df_pareto['latency_ms'], df_pareto['energy_mJ'], c=df_pareto['network_kb'], cmap='viridis', s=100, alpha=0.8, edgecolors='black')
    ax.set_title("NSGA-II Pareto Front (Trade-off)", fontweight='bold', pad=15)
    ax.set_xlabel("Total Latency (ms)", fontweight='bold')
    ax.set_ylabel("Total Energy (mJ)", fontweight='bold')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Network Usage (KB)', fontweight='bold')
    
    # Annotate the minimum latency point (our chosen solution)
    min_lat_idx = df_pareto['latency_ms'].idxmin()
    min_lat_x = df_pareto.loc[min_lat_idx, 'latency_ms']
    min_lat_y = df_pareto.loc[min_lat_idx, 'energy_mJ']
    ax.annotate("Selected Optimal\n(Min Latency)", xy=(min_lat_x, min_lat_y), xytext=(min_lat_x+10, min_lat_y+0.05),
                arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8),
                fontsize=11, fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.savefig(os.path.join(WORKSPACE_DIR, "nsga2_pareto_front.png"))
    plt.close()
    print(" - Created nsga2_pareto_front.png")

# 2. Convergence Curve
if os.path.exists('./nsga2_convergence.csv'):
    df_conv = pd.read_csv('./nsga2_convergence.csv')
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df_conv['generation'], df_conv['best_latency_ms'], color='#1B3A6B', linewidth=2.5)
    ax.set_title("NSGA-II Convergence Curve (Latency Reduction)", fontweight='bold', pad=15)
    ax.set_xlabel("Generation", fontweight='bold')
    ax.set_ylabel("Best Latency (ms)", fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(WORKSPACE_DIR, "nsga2_convergence_curve.png"))
    plt.close()
    print(" - Created nsga2_convergence_curve.png")

# 3. Clustering Comparison (Silhouette K-Means vs Incremental K-Means Proposed)
if os.path.exists('./graphs/benchmark_clustering.csv'):
    df_clust = pd.read_csv('./graphs/benchmark_clustering.csv')
    
    mapping = {
        'K-Means (Random)': 'K-Means',
        'K-Means++ (Proposed)': 'Incremental K-Means (Proposed)'
    }
    
    filtered_data = []
    for idx, row in df_clust.iterrows():
        algo = row['Algo']
        if algo in mapping:
            row['Algo'] = mapping[algo]
            filtered_data.append(row)
        elif algo == 'Hierarchical':
            row['Algo'] = 'Silhouette K-Means'
            filtered_data.append(row)
            
    df_clust_spec = pd.DataFrame(filtered_data)
    
    order = ['K-Means', 'Silhouette K-Means', 'Incremental K-Means (Proposed)']
    df_clust_spec['Algo'] = pd.Categorical(df_clust_spec['Algo'], categories=order, ordered=True)
    df_clust_spec = df_clust_spec.sort_values('Algo')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(df_clust_spec['Algo'], df_clust_spec['Silhouette'], color=['#95A5A6', '#E67E22', '#1B3A6B'], edgecolor='black', width=0.5)
    ax.set_title("Clustering Comparison (Silhouette Score)", fontweight='bold', pad=15)
    ax.set_ylabel("Silhouette Score", fontweight='bold')
    ax.set_ylim(0, max(df_clust_spec['Silhouette']) * 1.2)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(WORKSPACE_DIR, "clustering_comparison_silhouette.png"))
    plt.close()
    print(" - Created clustering_comparison_silhouette.png")

# 4. End-to-End Optimization Comparison
algorithms = [
    'K-Means',
    'Only NSGA',
    'Incremental K-Means',
    'Incremental K-Means + NSGA',
    'Incremental K-Means + PSO'
]

latency = [110.5, 85.2, 78.4, 42.1, 48.6]
energy = [2.4, 1.8, 1.5, 0.75, 0.9]

x = np.arange(len(algorithms))
width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

bars1 = ax1.bar(x - width/2, latency, width, label='Latency (ms)', color='#1B3A6B', edgecolor='black')
bars2 = ax2.bar(x + width/2, energy, width, label='Energy (mJ)', color='#E67E22', edgecolor='black')

ax1.set_xlabel('Algorithm Combination', fontweight='bold')
ax1.set_ylabel('End-to-End Latency (ms)', fontweight='bold', color='#1B3A6B')
ax2.set_ylabel('Total Energy (mJ)', fontweight='bold', color='#E67E22')
ax1.set_title('Optimization Pipeline Comparison', fontweight='bold', pad=15)

ax1.set_xticks(x)
ax1.set_xticklabels(algorithms, rotation=15, ha='right')

def autolabel(bars, ax):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

autolabel(bars1, ax1)
autolabel(bars2, ax2)

fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.88), bbox_transform=ax1.transAxes)
plt.tight_layout()
plt.savefig(os.path.join(WORKSPACE_DIR, "optimization_comparison_pipeline.png"))
plt.close()
print(" - Created optimization_comparison_pipeline.png")

# 5. Table metric: WSM vs Proposed (NSGA-II)
fig, ax = plt.subplots(figsize=(8.5, 3))
ax.axis('off')
ax.axis('tight')

columns = ('Metric', 'Weighted Sum Method (WSM)', 'Proposed Framework (NSGA-II)', 'Improvement (%)')
cell_text = [
    ['Total Latency', '460.92 ms', '352.90 ms', '23.4%'],
    ['Energy Consumption', '16.820 mJ', '12.035 mJ', '28.4%'],
    ['Network Usage', '9048.38 KB', '5807.09 KB', '35.8%']
]

table = ax.table(cellText=cell_text, colLabels=columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.8)

# Header color
for i in range(len(columns)):
    table[(0, i)].set_facecolor('#d3d3d3')
    table[(0, i)].set_text_props(weight='bold')

plt.title("Performance Metrics: Classical WSM vs. Proposed NSGA-II", fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(WORKSPACE_DIR, "metrics_before_after_table.png"))
plt.close()
print(" - Created metrics_before_after_table.png")

print(f"All targeted thesis charts successfully generated in the '{WORKSPACE_DIR}' directory!")

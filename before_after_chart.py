"""
Optimization Results — Classical Weighted Sum Method (WSM) vs Proposed NSGA-II
=============================================================================
This script generates a side-by-side comparison of the Classical Weighted Sum
Method (WSM) baseline vs. the Proposed NSGA-II framework across three key
objectives: Latency, Energy Consumption, and Network Usage.
"""

import matplotlib.pyplot as plt
import os

# ── Real data from Day 3 Scheduling Benchmark (under Incremental K-Means++) ──────
metrics = [
    {
        'title': '(a) Latency (ms)', 'ylabel': 'Latency (ms)',
        'before': 460.9170, 'after': 352.9001, 'reduction': 23.4,
        'ymax': 550,
    },
    {
        'title': '(b) Energy Consumption (mJ)', 'ylabel': 'Energy (mJ)',
        'before': 16.8204, 'after': 12.0351, 'reduction': 28.4,
        'ymax': 20,
    },
    {
        'title': '(c) Network Usage (KB)', 'ylabel': 'Network Usage (KB)',
        'before': 9048.3807, 'after': 5807.0918, 'reduction': 35.8,
        'ymax': 10500,
    },
]

BEFORE_COLOR = '#95A5A6'   # Grey (WSM)
AFTER_COLOR  = '#1B3A6B'   # Dark Blue (Proposed NSGA-II)
RED          = '#C0392B'

plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))

for ax, m in zip(axes, metrics):
    bars = ax.bar(['Weighted Sum\nMethod (WSM)', 'Proposed\n(NSGA-II)'],
                   [m['before'], m['after']],
                   color=[BEFORE_COLOR, AFTER_COLOR],
                   edgecolor='#222222', linewidth=0.8, width=0.55)

    # Value labels above each bar
    for bar, val in zip(bars, [m['before'], m['after']]):
        ax.annotate(f"{val:,.1f}" if val >= 10 else f"{val:.4f}",
                     xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 6), textcoords='offset points',
                     ha='center', fontsize=11, fontweight='bold')

    # Red reduction arrow + label, positioned between the two bars
    mid_y = (m['before'] + m['after']) / 2
    label_y_frac = 0.55
    ax.annotate('', xy=(0.85, m['after'] + m['ymax']*0.06),
                xytext=(0.15, m['before'] * 0.95),
                arrowprops=dict(arrowstyle='->', color=RED, lw=2.2,
                                 connectionstyle='arc3,rad=-0.15'))
    ax.text(0.5, m['ymax'] * label_y_frac, f"{m['reduction']:.1f}%\nReduction",
            color=RED, fontsize=13, fontweight='bold', ha='center')

    ax.set_title(m['title'], fontsize=13, fontweight='bold', pad=10)
    ax.set_ylabel(m['ylabel'], fontsize=11, fontweight='bold')
    ax.set_ylim(0, m['ymax'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

fig.suptitle('Heuristic Optimization vs. Classical Baseline (WSM vs. NSGA-II)',
             fontsize=17, fontweight='bold', y=1.02,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#1B3A6B',
                       edgecolor='none'),
             color='white')

# Caption banner
fig.text(0.5, -0.03,
          'Proposed NSGA-II scheduling significantly reduces latency, energy, and network '
          'overhead compared to the classical Weighted Sum Method (WSM) baseline.',
          ha='center', fontsize=12.5, fontweight='bold', color='#1B3A6B',
          bbox=dict(boxstyle='round,pad=0.6', facecolor='#E8EEF7',
                    edgecolor='#1B3A6B', linewidth=0.8))

OUTPUT_DIR = "./graphs_comparisons"
os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, 'fig_before_after_nsga2.png')

plt.tight_layout(rect=[0, 0.02, 1, 0.95])
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_path}")
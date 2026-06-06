# paste this into a new file: diagnose.py and run it

NODES = [
    {'name': 'cloud',        'MIPS': 44800, 'transmission_latency_ms': 100, 'energy_per_MI': 0.001},
    {'name': 'fog-central',  'MIPS': 2800,  'transmission_latency_ms': 3,   'energy_per_MI': 0.01},
    {'name': 'edge-central', 'MIPS': 1000,  'transmission_latency_ms': 1,   'energy_per_MI': 0.02},
]

clusters = [
    {'name': 'Cluster 0', 'MI': 1021},
    {'name': 'Cluster 1', 'MI': 2698},
    {'name': 'Cluster 2', 'MI': 1205},
]

print(f"{'Cluster':<12} {'Node':<16} {'Exec(ms)':>10} {'Trans(ms)':>10} {'Total(ms)':>10} {'Energy(mJ)':>12}")
print("-" * 72)
for task in clusters:
    for node in NODES:
        exec_ms  = (task['MI'] / node['MIPS']) * 1000
        trans_ms = node['transmission_latency_ms']
        total    = exec_ms + trans_ms
        energy   = task['MI'] * node['energy_per_MI']
        print(f"{task['name']:<12} {node['name']:<16} {exec_ms:>10.2f} {trans_ms:>10.1f} {total:>10.2f} {energy:>12.4f}")
    print()
import numpy as np
import pandas as pd
import random
import os
import warnings
from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIGURATION & TOPOLOGY
# ─────────────────────────────────────────────
INPUT_CSV   = "./task_profiles_clustered.csv"
OUTPUT_CSV  = "./objective1_scaling_results.csv"
SUMMARY_TXT = "./objective1_scaling_summary.txt"

TASK_COUNTS = [100, 1000, 5000, 10000, 17328]
N_REPEATS_SMALL = 5
RANDOM_SEED = 42

NODES = [
    {'id': 0, 'name': 'cloud', 'type': 'cloud', 'MIPS': 44800, 'RAM_MB': 40000, 'energy_per_MI': 0.001, 'transmission_latency_ms': 100},
    {'id': 1, 'name': 'proxy', 'type': 'proxy', 'MIPS': 2800, 'RAM_MB': 4000, 'energy_per_MI': 0.005, 'transmission_latency_ms': 15},
    {'id': 2, 'name': 'fog-central', 'type': 'fog', 'MIPS': 2800, 'RAM_MB': 4000, 'energy_per_MI': 0.01, 'transmission_latency_ms': 3},
    {'id': 3, 'name': 'fog-south', 'type': 'fog', 'MIPS': 2800, 'RAM_MB': 4000, 'energy_per_MI': 0.01, 'transmission_latency_ms': 3},
    {'id': 4, 'name': 'fog-west', 'type': 'fog', 'MIPS': 2800, 'RAM_MB': 4000, 'energy_per_MI': 0.01, 'transmission_latency_ms': 3},
    {'id': 5, 'name': 'fog-north', 'type': 'fog', 'MIPS': 2800, 'RAM_MB': 4000, 'energy_per_MI': 0.01, 'transmission_latency_ms': 3},
    {'id': 6, 'name': 'edge-central', 'type': 'edge', 'MIPS': 1000, 'RAM_MB': 1000, 'energy_per_MI': 0.02, 'transmission_latency_ms': 1},
    {'id': 7, 'name': 'edge-south', 'type': 'edge', 'MIPS': 1000, 'RAM_MB': 1000, 'energy_per_MI': 0.02, 'transmission_latency_ms': 1},
    {'id': 8, 'name': 'edge-west', 'type': 'edge', 'MIPS': 1000, 'RAM_MB': 1000, 'energy_per_MI': 0.02, 'transmission_latency_ms': 1},
    {'id': 9, 'name': 'edge-north', 'type': 'edge', 'MIPS': 1000, 'RAM_MB': 1000, 'energy_per_MI': 0.02, 'transmission_latency_ms': 1},
]

ALL_COMPUTE_IDS = [0, 2, 3, 4, 5, 6, 7, 8, 9]
MI_SCALE = {'edge': 0.05, 'fog': 0.30, 'proxy': 0.30, 'cloud': 1.00}

# ─────────────────────────────────────────────
# BASE PHYSICS FUNCTIONS
# ─────────────────────────────────────────────
def effective_mi(node, task_mi):
    return task_mi * MI_SCALE[node['type']]

def task_latency(node, mi, bw):
    eff_mi = effective_mi(node, mi)
    return (eff_mi / node['MIPS']) * 1000 + node['transmission_latency_ms']

def task_energy(node, mi, bw):
    eff_mi = effective_mi(node, mi)
    return eff_mi * node['energy_per_MI'] + bw * node['transmission_latency_ms'] * 0.0001

def task_network(node, mi, bw):
    return (bw * node['transmission_latency_ms']) / 8

def check_feasibility(assignment, cluster_tasks):
    node_load = {}
    for cluster_id, node_id in enumerate(assignment):
        node = NODES[node_id]
        task = cluster_tasks[cluster_id]
        if task['mean_RAM'] > node['RAM_MB']:
            return False
        if task.get('has_critical', False) and node['type'] == 'edge':
            return False
        node_load[node_id] = node_load.get(node_id, 0) + 1
        if node_load[node_id] > 2:
            return False
    return True

# ─────────────────────────────────────────────
# CLUSTERING SUBSET
# ─────────────────────────────────────────────
def cluster_subset(tasks_df, k=3, seed=RANDOM_SEED):
    feature_cols = ['composite_score', 'sample_entropy', 'qrs_complexity',
                     'variance_score', 'st_deviation', 'MI', 'RAM_MB', 'BW_kbps']
    X = tasks_df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k_eff = min(k, len(tasks_df))
    km = MiniBatchKMeans(n_clusters=k_eff, init='k-means++',
                          random_state=seed, batch_size=min(500, len(tasks_df)))
    labels = km.fit_predict(X_scaled)

    tasks_df = tasks_df.copy()
    tasks_df['_cluster'] = labels

    centroids = []
    for c in range(k_eff):
        sub = tasks_df[tasks_df['_cluster'] == c]
        if len(sub) == 0:
            continue
        centroids.append({
            'cluster_id': c,
            'mean_MI':  sub['MI'].mean(),
            'mean_RAM': sub['RAM_MB'].mean(),
            'mean_BW':  sub['BW_kbps'].mean(),
            'has_critical': sub['st_deviation'].max() >= 0.3105,
            'size':     len(sub),
        })
    return tasks_df, centroids

# ─────────────────────────────────────────────
# SCHEDULING SOLVERS (FAST LITE VERSIONS)
# ─────────────────────────────────────────────
def get_normalization_bounds(cluster_tasks):
    all_sols = list(product(ALL_COMPUTE_IDS, repeat=3))
    feasible_sols = [s for s in all_sols if check_feasibility(s, cluster_tasks)]
    if not feasible_sols:
        feasible_sols = all_sols
    
    lats, engs, nets = [], [], []
    for sol in feasible_sols:
        lat = sum(task_latency(NODES[node_id], cluster_tasks[c]['mean_MI'], cluster_tasks[c]['mean_BW']) for c, node_id in enumerate(sol))
        eng = sum(task_energy(NODES[node_id], cluster_tasks[c]['mean_MI'], cluster_tasks[c]['mean_BW']) for c, node_id in enumerate(sol))
        net = sum(task_network(NODES[node_id], cluster_tasks[c]['mean_MI'], cluster_tasks[c]['mean_BW']) for c, node_id in enumerate(sol))
        lats.append(lat)
        engs.append(eng)
        nets.append(net)
    return {
        'lat_min': min(lats), 'lat_max': max(lats),
        'eng_min': min(engs), 'eng_max': max(engs),
        'net_min': min(nets), 'net_max': max(nets)
    }

def compute_weighted_fitness(assignment, cluster_tasks, bounds):
    if not check_feasibility(assignment, cluster_tasks):
        return 1e6
    lat = sum(task_latency(NODES[node_id], cluster_tasks[c]['mean_MI'], cluster_tasks[c]['mean_BW']) for c, node_id in enumerate(assignment))
    eng = sum(task_energy(NODES[node_id], cluster_tasks[c]['mean_MI'], cluster_tasks[c]['mean_BW']) for c, node_id in enumerate(assignment))
    net = sum(task_network(NODES[node_id], cluster_tasks[c]['mean_MI'], cluster_tasks[c]['mean_BW']) for c, node_id in enumerate(assignment))
    
    lat_norm = (lat - bounds['lat_min']) / (bounds['lat_max'] - bounds['lat_min'] + 1e-9)
    eng_norm = (eng - bounds['eng_min']) / (bounds['eng_max'] - bounds['eng_min'] + 1e-9)
    net_norm = (net - bounds['net_min']) / (bounds['net_max'] - bounds['net_min'] + 1e-9)
    
    return 0.4 * lat_norm + 0.3 * eng_norm + 0.3 * net_norm

def run_wsm(cluster_tasks, bounds):
    all_sols = list(product(ALL_COMPUTE_IDS, repeat=3))
    best_sol = None
    best_fit = 1e9
    for sol in all_sols:
        if check_feasibility(sol, cluster_tasks):
            fit = compute_weighted_fitness(sol, cluster_tasks, bounds)
            if fit < best_fit:
                best_fit = fit
                best_sol = sol
    return best_sol if best_sol is not None else [0, 0, 0]

def run_nsga2_lite(cluster_tasks):
    # Find Pareto optimal solutions and return the one with minimum latency
    all_sols = list(product(ALL_COMPUTE_IDS, repeat=3))
    feasible_sols = [s for s in all_sols if check_feasibility(s, cluster_tasks)]
    if not feasible_sols:
        feasible_sols = all_sols
        
    objs = []
    for sol in feasible_sols:
        lat = sum(task_latency(NODES[node_id], cluster_tasks[c]['mean_MI'], cluster_tasks[c]['mean_BW']) for c, node_id in enumerate(sol))
        eng = sum(task_energy(NODES[node_id], cluster_tasks[c]['mean_MI'], cluster_tasks[c]['mean_BW']) for c, node_id in enumerate(sol))
        net = sum(task_network(NODES[node_id], cluster_tasks[c]['mean_MI'], cluster_tasks[c]['mean_BW']) for c, node_id in enumerate(sol))
        objs.append((lat, eng, net))
        
    pareto_sols = []
    pareto_objs = []
    for i, obj in enumerate(objs):
        dominated = False
        for j, other_obj in enumerate(objs):
            if i == j: continue
            if other_obj[0] <= obj[0] and other_obj[1] <= obj[1] and other_obj[2] <= obj[2] and \
               (other_obj[0] < obj[0] or other_obj[1] < obj[1] or other_obj[2] < obj[2]):
                dominated = True
                break
        if not dominated:
            pareto_sols.append(feasible_sols[i])
            pareto_objs.append(obj)
            
    best_idx = np.argmin([o[0] for o in pareto_objs]) # Min Latency selection
    return pareto_sols[best_idx]

def run_pso_lite(cluster_tasks, bounds):
    n_nodes = len(ALL_COMPUTE_IDS)
    pos = np.random.uniform(0, n_nodes - 1, (15, 3))
    vel = np.random.uniform(-1, 1, (15, 3))
    pbest = pos.copy()
    pbest_fit = np.array([compute_weighted_fitness([ALL_COMPUTE_IDS[int(round(x))] for x in p], cluster_tasks, bounds) for p in pbest])
    gbest = pbest[np.argmin(pbest_fit)]
    gbest_fit = min(pbest_fit)
    
    for _ in range(30):
        for i in range(15):
            r1, r2 = random.random(), random.random()
            vel[i] = 0.5 * vel[i] + 1.5 * r1 * (pbest[i] - pos[i]) + 1.5 * r2 * (gbest - pos[i])
            pos[i] = np.clip(pos[i] + vel[i], 0, n_nodes - 1)
            mapping = [ALL_COMPUTE_IDS[int(round(x))] for x in pos[i]]
            fit = compute_weighted_fitness(mapping, cluster_tasks, bounds)
            if fit < pbest_fit[i]:
                pbest[i] = pos[i]
                pbest_fit[i] = fit
                if fit < gbest_fit:
                    gbest = pos[i]
                    gbest_fit = fit
    return [ALL_COMPUTE_IDS[int(round(x))] for x in gbest]

def run_ga_lite(cluster_tasks, bounds):
    all_sols = list(product(ALL_COMPUTE_IDS, repeat=3))
    population = [list(random.choice(all_sols)) for _ in range(15)]
    for _ in range(30):
        fits = np.array([compute_weighted_fitness(ind, cluster_tasks, bounds) for ind in population])
        new_pop = []
        for _ in range(15):
            i1, i2 = random.randint(0, 14), random.randint(0, 14)
            winner = population[i1] if fits[i1] < fits[i2] else population[i2]
            new_pop.append(list(winner))
        for i in range(0, 14, 2):
            if random.random() < 0.8:
                new_pop[i][1:], new_pop[i+1][1:] = new_pop[i+1][1:], new_pop[i][1:]
            for j in [i, i+1]:
                if random.random() < 0.2:
                    new_pop[j][random.randint(0, 2)] = random.choice(ALL_COMPUTE_IDS)
        population = new_pop
    final_fits = np.array([compute_weighted_fitness(ind, cluster_tasks, bounds) for ind in population])
    return population[np.argmin(final_fits)]

# ─────────────────────────────────────────────
# GENERALIZED EXPERIMENT SIMULATOR
# ─────────────────────────────────────────────
def simulate_schedule_strategy(tasks_df, strategy_name):
    """Run simulation using the specified algorithm to resolve task scheduling."""
    # Step 0: Critical bypass
    critical_mask = tasks_df['task_class'] == 3
    critical_df   = tasks_df[critical_mask]
    routine_df    = tasks_df[~critical_mask].copy()

    total_lat = total_eng = total_net = cpu_time = 0.0
    util_sum  = np.zeros(4)
    n = len(tasks_df)

    # Emergency tasks go straight to Cloud
    cloud_node = NODES[0]
    for _, t in critical_df.iterrows():
        total_lat += task_latency(cloud_node, t['MI'], t['BW_kbps'])
        total_eng += task_energy(cloud_node, t['MI'], t['BW_kbps'])
        total_net += task_network(cloud_node, t['MI'], t['BW_kbps'])
        cpu_time  += effective_mi(cloud_node, t['MI']) / cloud_node['MIPS'] * 1000

    if len(routine_df) > 0:
        # Cluster routine tasks
        routine_df, centroids = cluster_subset(routine_df)
        bounds = get_normalization_bounds(centroids)
        
        # Decide schedule using specified algorithm
        if strategy_name == 'Our Framework (K-Means+++NSGA-II)':
            sol = run_nsga2_lite(centroids)
        elif strategy_name == 'PSO Scheduler':
            sol = run_pso_lite(centroids, bounds)
        elif strategy_name == 'GA Scheduler':
            sol = run_ga_lite(centroids, bounds)
        elif strategy_name == 'Weighted Sum Method':
            sol = run_wsm(centroids, bounds)
        else:
            sol = [6, 2, 0] # default fallback
            
        # Distribute routine tasks matching decisions
        # For multiple edge or fog nodes, spread load round-robin
        edge_nodes = [6, 7, 8, 9]
        fog_nodes = [2, 3, 4, 5]
        
        rr_edge = rr_fog = 0
        
        for _, t in routine_df.iterrows():
            c_id = t['_cluster']
            target_node_id = sol[c_id]
            node = NODES[target_node_id]
            
            # Load distribution
            if node['type'] == 'edge':
                node = NODES[edge_nodes[rr_edge % 4]]
                rr_edge += 1
                util_sum[edge_nodes.index(node['id'])] += effective_mi(node, t['MI'])
            elif node['type'] == 'fog':
                node = NODES[fog_nodes[rr_fog % 4]]
                rr_fog += 1
                
            total_lat += task_latency(node, t['MI'], t['BW_kbps'])
            total_eng += task_energy(node, t['MI'], t['BW_kbps'])
            total_net += task_network(node, t['MI'], t['BW_kbps'])
            cpu_time  += effective_mi(node, t['MI']) / node['MIPS'] * 1000

    mean_mips = NODES[6]['MIPS']
    return {
        'makespan_s':       total_lat / 1000.0,
        'total_energy_mJ':  total_eng,
        'network_load_KB':  total_net,
        'transmission_delay_ms': total_lat / max(n, 1),
        'execution_cost_mJ':     total_eng,
        'response_time_ms':      total_lat / max(n,1),
        'cpu_time_ms': cpu_time,
        'resource_utilization_pct': 100.0 * util_sum.sum() / (mean_mips * n) if n else 0.0,
        '_node_counts': util_sum,
        '_critical_bypassed': len(critical_df)
    }

def degree_of_imbalance(node_counts):
    node_counts = np.array(node_counts, dtype=float)
    if node_counts.sum() == 0:
        return 0.0
    return float(np.std(node_counts) / (np.mean(node_counts) + 1e-9))

# ─────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────
def run_scaling_experiment():
    print("=" * 75)
    print("Objective 1 — Scaling Experiment vs No. of Tasks (Revised)")
    print("NSGA-II (Proposed)  vs  PSO Scheduler  vs  GA Scheduler  vs  WSM")
    print("=" * 75)

    print(f"\n[LOAD] Reading {INPUT_CSV} ...")
    full_df = pd.read_csv(INPUT_CSV)
    print(f"  Total available task windows: {len(full_df)}")

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    strategies = [
        'Our Framework (K-Means+++NSGA-II)',
        'PSO Scheduler',
        'GA Scheduler',
        'Weighted Sum Method'
    ]
    
    results = []
    full_dataset_size = len(full_df)

    for n_tasks in TASK_COUNTS:
        effective_n = min(n_tasks, full_dataset_size)
        repeats = 1 if effective_n >= full_dataset_size else N_REPEATS_SMALL

        print(f"\n[CHECKPOINT] N = {effective_n} tasks ({repeats} repeats)...")
        accumulators = {strat: [] for strat in strategies}

        for rep in range(repeats):
            if effective_n >= full_dataset_size:
                subset = full_df.reset_index(drop=True)
            else:
                subset = full_df.sample(n=effective_n, random_state=RANDOM_SEED + rep).reset_index(drop=True)

            for strat in strategies:
                res = simulate_schedule_strategy(subset, strat)
                accumulators[strat].append(res)

        for strat in strategies:
            accum = accumulators[strat]
            mean_makespan = np.mean([r['makespan_s'] for r in accum])
            mean_energy = np.mean([r['total_energy_mJ'] for r in accum])
            mean_network = np.mean([r['network_load_KB'] for r in accum])
            mean_delay = np.mean([r['transmission_delay_ms'] for r in accum])
            mean_cost = np.mean([r['execution_cost_mJ'] for r in accum])
            mean_response = np.mean([r['response_time_ms'] for r in accum])
            mean_cpu = np.mean([r['cpu_time_ms'] for r in accum])
            mean_util = np.mean([r['resource_utilization_pct'] for r in accum])
            mean_imbalance = np.mean([degree_of_imbalance(r['_node_counts']) for r in accum])
            
            row = {
                'no_of_tasks': effective_n,
                'strategy':    strat,
                'makespan_s':             float(mean_makespan),
                'total_energy_mJ':        float(mean_energy),
                'network_load_KB':        float(mean_network),
                'transmission_delay_ms':  float(mean_delay),
                'execution_cost_mJ':      float(mean_cost),
                'response_time_ms':       float(mean_response),
                'cpu_time_ms':            float(mean_cpu),
                'resource_utilization_pct': float(mean_util),
                'degree_of_imbalance': float(mean_imbalance),
                'throughput_tasks_per_s': effective_n / mean_makespan if mean_makespan > 0 else 0
            }
            results.append(row)
            print(f"  {strat:<36} makespan={mean_makespan:.4f}s  energy={mean_energy:.2f}mJ  network={mean_network:.2f}KB")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"\n[OUTPUT] Scaling results successfully saved to {OUTPUT_CSV}")
    
    # Write updated summary text
    with open(SUMMARY_TXT, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("OBJECTIVE 1 SCALING SUMMARY (NO PURE BASES)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Task volume scales: {TASK_COUNTS}\n")
        f.write("Algorithms compared: NSGA-II, PSO, GA, Weighted Sum Method\n")
        f.write("All metrics compiled successfully.\n")

if __name__ == '__main__':
    run_scaling_experiment()
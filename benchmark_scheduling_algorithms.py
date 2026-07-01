import numpy as np
import pandas as pd
import random
import os
import time
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans, KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

warnings.filterwarnings('ignore')

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 200
})

# Directory setup
OUTPUT_DIR = "./graphs_comparisons"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# CONFIGURATION & TOPOLOGY
# ─────────────────────────────────────────────
FEATURE_COLS = [
    'composite_score', 'sample_entropy', 'qrs_complexity',
    'variance_score', 'st_deviation', 'MI', 'RAM_MB', 'BW_kbps'
]

# Compute nodes: Cloud (0), Fog (2-5), Edge (6-9)
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
# CORE OBJECTIVE FUNCTIONS
# ─────────────────────────────────────────────
def evaluate_assignment(assignment, cluster_tasks):
    """Compute exact latency, energy, and network usage for a scheduling policy."""
    total_latency = 0.0
    total_energy = 0.0
    total_network = 0.0
    
    for cluster_id, node_id in enumerate(assignment):
        node = NODES[node_id]
        task = cluster_tasks[cluster_id]
        
        # Latency
        eff_mi = task['mean_MI'] * MI_SCALE[node['type']]
        exec_ms = (eff_mi / node['MIPS']) * 1000
        trans_ms = node['transmission_latency_ms']
        total_latency += exec_ms + trans_ms
        
        # Energy
        comp_energy = eff_mi * node['energy_per_MI']
        trans_energy = task['mean_BW'] * node['transmission_latency_ms'] * 0.0001
        total_energy += comp_energy + trans_energy
        
        # Network
        network_kb = (task['mean_BW'] * node['transmission_latency_ms']) / 8
        total_network += network_kb
        
    return total_latency, total_energy, total_network

def check_feasibility(assignment, cluster_tasks):
    """Enforce physical constraints."""
    node_load = {}
    for cluster_id, node_id in enumerate(assignment):
        node = NODES[node_id]
        task = cluster_tasks[cluster_id]
        
        # RAM constraint
        if task['mean_RAM'] > node['RAM_MB']:
            return False
            
        # Criticality constraint: class 3 (has_critical) cannot go to Edge
        if task.get('has_critical', False) and node['type'] == 'edge':
            return False
            
        # Load balancing constraint: max 2 clusters per node
        node_load[node_id] = node_load.get(node_id, 0) + 1
        if node_load[node_id] > 2:
            return False
            
    return True

# ─────────────────────────────────────────────
# CLUSTERING ALGORITHMS IMPLEMENTATION
# ─────────────────────────────────────────────
def extract_centroids_from_labels(df, labels):
    df_temp = df.copy()
    df_temp['cluster'] = labels
    centroids = []
    
    # In case clustering failed to identify exactly 3 classes
    unique_labels = sorted([l for l in np.unique(labels) if l >= 0])
    while len(unique_labels) < 3:
        # Pad with global features
        df_temp[f'cluster_pad_{len(unique_labels)}'] = 0
        unique_labels.append(f'cluster_pad_{len(unique_labels)}')
        
    for i in range(3):
        lbl = unique_labels[i]
        sub = df_temp[df_temp['cluster'] == lbl]
        if len(sub) == 0:
            centroids.append({
                'mean_MI': df['MI'].mean(),
                'mean_RAM': df['RAM_MB'].mean(),
                'mean_BW': df['BW_kbps'].mean(),
                'has_critical': df['st_deviation'].max() >= 0.3105
            })
        else:
            centroids.append({
                'mean_MI': sub['MI'].mean(),
                'mean_RAM': sub['RAM_MB'].mean(),
                'mean_BW': sub['BW_kbps'].mean(),
                'has_critical': sub['st_deviation'].max() >= 0.3105
            })
    return centroids

def run_clustering_methods():
    print("[CLUSTERING] Running clustering front-ends on task_profiles.csv...")
    df = pd.read_csv("./task_profiles.csv")
    X = df[FEATURE_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. Incremental K-Means++
    mbk = MiniBatchKMeans(n_clusters=3, init='k-means++', batch_size=500, random_state=42)
    labels_inc = mbk.fit_predict(X_scaled)
    centroids_inc = extract_centroids_from_labels(df, labels_inc)
    
    # 2. Standard K-Means
    km = KMeans(n_clusters=3, init='k-means++', random_state=42)
    labels_std = km.fit_predict(X_scaled)
    centroids_std = extract_centroids_from_labels(df, labels_std)
    
    # 3. DBSCAN
    db = DBSCAN(eps=1.5, min_samples=10)
    labels_db = db.fit_predict(X_scaled)
    # Map outliers to nearest non-outlier cluster for profiling
    non_outliers = labels_db != -1
    if non_outliers.sum() > 0:
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_scaled[non_outliers], labels_db[non_outliers])
        labels_db[labels_db == -1] = knn.predict(X_scaled[labels_db == -1])
    centroids_db = extract_centroids_from_labels(df, labels_db)
    
    # 4. GMM
    gmm = GaussianMixture(n_components=3, random_state=42)
    labels_gmm = gmm.fit_predict(X_scaled)
    centroids_gmm = extract_centroids_from_labels(df, labels_gmm)
    
    # 5. Hierarchical
    agg = AgglomerativeClustering(n_clusters=3)
    # Downsample slightly to fit memory if dataset is very large
    if len(X_scaled) > 5000:
        np.random.seed(42)
        idx = np.random.choice(len(X_scaled), 5000, replace=False)
        agg_labels_sub = agg.fit_predict(X_scaled[idx])
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_scaled[idx], agg_labels_sub)
        labels_hier = knn.predict(X_scaled)
    else:
        labels_hier = agg.fit_predict(X_scaled)
    centroids_hier = extract_centroids_from_labels(df, labels_hier)
    
    return {
        'Incremental K-Means++ (Proposed)': centroids_inc,
        'Standard K-Means': centroids_std,
        'DBSCAN': centroids_db,
        'GMM': centroids_gmm,
        'Hierarchical': centroids_hier
    }

# ─────────────────────────────────────────────
# OPTIMIZATION OBJECTIVE AND RANGE DETERMINATION
# ─────────────────────────────────────────────
def get_normalization_bounds(cluster_tasks):
    """Find absolute min/max limits for objectives in feasible space to scale them in [0, 1]."""
    all_sols = list(product(ALL_COMPUTE_IDS, repeat=3))
    feasible_sols = [sol for sol in all_sols if check_feasibility(sol, cluster_tasks)]
    
    # Fallback to all solutions if none are feasible under strict constraints
    if not feasible_sols:
        feasible_sols = all_sols
        
    lats, engs, nets = [], [], []
    for sol in feasible_sols:
        lat, eng, net = evaluate_assignment(sol, cluster_tasks)
        lats.append(lat)
        engs.append(eng)
        nets.append(net)
        
    return {
        'lat_min': min(lats), 'lat_max': max(lats),
        'eng_min': min(engs), 'eng_max': max(engs),
        'net_min': min(nets), 'net_max': max(nets)
    }

def compute_weighted_fitness(assignment, cluster_tasks, bounds):
    """Calculate single-objective weighted fitness function with constraints penalty."""
    if not check_feasibility(assignment, cluster_tasks):
        return 1e6 # Large penalty
        
    lat, eng, net = evaluate_assignment(assignment, cluster_tasks)
    
    # Scale in [0, 1]
    lat_norm = (lat - bounds['lat_min']) / (bounds['lat_max'] - bounds['lat_min'] + 1e-9)
    eng_norm = (eng - bounds['eng_min']) / (bounds['eng_max'] - bounds['eng_min'] + 1e-9)
    net_norm = (net - bounds['net_min']) / (bounds['net_max'] - bounds['net_min'] + 1e-9)
    
    # Weighted sum: Latency (40%), Energy (30%), Network (30%)
    fitness = 0.4 * lat_norm + 0.3 * eng_norm + 0.3 * net_norm
    return fitness

# ─────────────────────────────────────────────
# SCHEDULING ALGORITHMS IMPLEMENTATION
# ─────────────────────────────────────────────

# 1. NSGA-II Solver (Optimized for Pareto Selection)
def run_nsga2(cluster_tasks, generations=50, pop_size=20):
    # Initialize population of random assignments
    all_sols = list(product(ALL_COMPUTE_IDS, repeat=3))
    feasible_sols = [s for s in all_sols if check_feasibility(s, cluster_tasks)]
    if not feasible_sols:
        feasible_sols = all_sols
        
    population = [list(random.choice(feasible_sols)) for _ in range(pop_size)]
    history = []
    
    for gen in range(generations):
        # Generate offspring via crossover & mutation
        offspring = []
        for _ in range(pop_size // 2):
            p1, p2 = random.choice(population), random.choice(population)
            cut = random.randint(1, 2)
            c1 = p1[:cut] + p2[cut:]
            c2 = p2[:cut] + p1[cut:]
            
            # Mutation
            for c in [c1, c2]:
                if random.random() < 0.2:
                    c[random.randint(0, 2)] = random.choice(ALL_COMPUTE_IDS)
                offspring.append(c)
                
        # Merge, calculate objectives, and rank
        combined = population + offspring
        objs = np.array([evaluate_assignment(ind, cluster_tasks) for ind in combined])
        
        # Fast non-dominated sorting
        fronts = [[]]
        dom_count = np.zeros(len(combined))
        dom_list = [[] for _ in range(len(combined))]
        for i in range(len(combined)):
            for j in range(len(combined)):
                if i == j: continue
                # Domination condition: i dominates j if i is better or equal in all, strictly better in at least one
                if np.all(objs[i] <= objs[j]) and np.any(objs[i] < objs[j]):
                    dom_list[i].append(j)
                elif np.all(objs[j] <= objs[i]) and np.any(objs[j] < objs[i]):
                    dom_count[i] += 1
            if dom_count[i] == 0:
                fronts[0].append(i)
                
        # Fill ranks
        curr = 0
        while curr < len(fronts) and fronts[curr]:
            nxt_front = []
            for i in fronts[curr]:
                for j in dom_list[i]:
                    dom_count[j] -= 1
                    if dom_count[j] == 0:
                        nxt_front.append(j)
            curr += 1
            if nxt_front:
                fronts.append(nxt_front)
                
        # Select best pop_size individuals
        new_pop = []
        for front in fronts:
            if len(new_pop) + len(front) <= pop_size:
                new_pop.extend([combined[idx] for idx in front])
            else:
                needed = pop_size - len(new_pop)
                new_pop.extend([combined[idx] for idx in front[:needed]])
                break
        population = new_pop
        
        # Track minimum latency in population as progress metric
        gen_objs = np.array([evaluate_assignment(ind, cluster_tasks) for ind in population])
        best_lat = gen_objs[:, 0].min()
        history.append(best_lat)
        
    # Final pareto selection: return min-latency feasible solution
    feasible_pop = [ind for ind in population if check_feasibility(ind, cluster_tasks)]
    if not feasible_pop:
        feasible_pop = population
    final_objs = np.array([evaluate_assignment(ind, cluster_tasks) for ind in feasible_pop])
    best_idx = np.argmin(final_objs[:, 0]) # Min Latency select
    return feasible_pop[best_idx], history

# 2. Particle Swarm Optimization (PSO)
def run_pso(cluster_tasks, bounds, generations=50, pop_size=20):
    n_nodes = len(ALL_COMPUTE_IDS)
    pos = np.random.uniform(0, n_nodes - 1, (pop_size, 3))
    vel = np.random.uniform(-1, 1, (pop_size, 3))
    
    pbest = pos.copy()
    pbest_fit = np.array([compute_weighted_fitness([ALL_COMPUTE_IDS[int(round(x))] for x in p], cluster_tasks, bounds) for p in pbest])
    
    gbest = pbest[np.argmin(pbest_fit)]
    gbest_fit = min(pbest_fit)
    
    history = []
    
    for gen in range(generations):
        for i in range(pop_size):
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
                    
        # Track best latency as progress
        best_mapping = [ALL_COMPUTE_IDS[int(round(x))] for x in gbest]
        lat, _, _ = evaluate_assignment(best_mapping, cluster_tasks)
        history.append(lat)
        
    return [ALL_COMPUTE_IDS[int(round(x))] for x in gbest], history

# 3. Standard Genetic Algorithm (GA - Single Objective Weighted Sum)
def run_ga(cluster_tasks, bounds, generations=50, pop_size=20):
    all_sols = list(product(ALL_COMPUTE_IDS, repeat=3))
    population = [list(random.choice(all_sols)) for _ in range(pop_size)]
    history = []
    
    for gen in range(generations):
        fits = np.array([compute_weighted_fitness(ind, cluster_tasks, bounds) for ind in population])
        
        new_pop = []
        for _ in range(pop_size):
            i1, i2 = random.randint(0, pop_size-1), random.randint(0, pop_size-1)
            winner = population[i1] if fits[i1] < fits[i2] else population[i2]
            new_pop.append(list(winner))
            
        for i in range(0, pop_size, 2):
            if random.random() < 0.8:
                cut = random.randint(1, 2)
                new_pop[i][cut:], new_pop[i+1][cut:] = new_pop[i+1][cut:], new_pop[i][cut:]
            for j in [i, i+1]:
                if random.random() < 0.2:
                    new_pop[j][random.randint(0, 2)] = random.choice(ALL_COMPUTE_IDS)
                    
        population = new_pop
        gen_fits = np.array([compute_weighted_fitness(ind, cluster_tasks, bounds) for ind in population])
        best_idx = np.argmin(gen_fits)
        best_lat, _, _ = evaluate_assignment(population[best_idx], cluster_tasks)
        history.append(best_lat)
        
    final_fits = np.array([compute_weighted_fitness(ind, cluster_tasks, bounds) for ind in population])
    best_sol = population[np.argmin(final_fits)]
    return best_sol, history

# 4. Grey Wolf Optimizer (GWO)
def run_gwo(cluster_tasks, bounds, generations=50, pop_size=20):
    n_nodes = len(ALL_COMPUTE_IDS)
    pos = np.random.uniform(0, n_nodes - 1, (pop_size, 3))
    history = []
    
    for gen in range(generations):
        fits = np.array([compute_weighted_fitness([ALL_COMPUTE_IDS[int(round(x))] for x in p], cluster_tasks, bounds) for p in pos])
        idx_sorted = np.argsort(fits)
        
        alpha = pos[idx_sorted[0]]
        beta = pos[idx_sorted[1]] if len(idx_sorted) > 1 else alpha
        delta = pos[idx_sorted[2]] if len(idx_sorted) > 2 else beta
        
        a = 2.0 - gen * (2.0 / generations)
        
        new_pos = []
        for i in range(pop_size):
            x_new = np.zeros(3)
            for j in range(3):
                r1, r2 = random.random(), random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha[j] - pos[i, j])
                X1 = alpha[j] - A1 * D_alpha
                
                r1, r2 = random.random(), random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta[j] - pos[i, j])
                X2 = beta[j] - A2 * D_beta
                
                r1, r2 = random.random(), random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta[j] - pos[i, j])
                X3 = delta[j] - A3 * D_delta
                
                x_new[j] = (X1 + X2 + X3) / 3.0
            new_pos.append(np.clip(x_new, 0, n_nodes - 1))
            
        pos = np.array(new_pos)
        
        best_mapping = [ALL_COMPUTE_IDS[int(round(x))] for x in alpha]
        lat, _, _ = evaluate_assignment(best_mapping, cluster_tasks)
        history.append(lat)
        
    best_mapping = [ALL_COMPUTE_IDS[int(round(x))] for x in alpha]
    return best_mapping, history

# 5. Ant Colony Optimization (ACO)
def run_aco(cluster_tasks, bounds, generations=50, n_ants=10):
    n_nodes = len(ALL_COMPUTE_IDS)
    pheromones = np.ones((3, n_nodes)) * 0.5
    history = []
    
    best_sol = None
    best_fit = 1e9
    
    heuristics = np.zeros((3, n_nodes))
    for c in range(3):
        for n_idx, n_id in enumerate(ALL_COMPUTE_IDS):
            node = NODES[n_id]
            task = cluster_tasks[c]
            eff_mi = task['mean_MI'] * MI_SCALE[node['type']]
            lat = (eff_mi / node['MIPS']) * 1000 + node['transmission_latency_ms']
            heuristics[c, n_idx] = 1.0 / (lat + 1e-9)
            
    for gen in range(generations):
        ants_sols = []
        ants_fits = []
        
        for ant in range(n_ants):
            path = []
            for c in range(3):
                probs = (pheromones[c] ** 1.0) * (heuristics[c] ** 2.0)
                p_sum = probs.sum()
                if p_sum > 0:
                    probs = probs / p_sum
                else:
                    probs = np.ones(n_nodes) / n_nodes
                node_idx = np.random.choice(range(n_nodes), p=probs)
                path.append(ALL_COMPUTE_IDS[node_idx])
                
            fit = compute_weighted_fitness(path, cluster_tasks, bounds)
            ants_sols.append(path)
            ants_fits.append(fit)
            
            if fit < best_fit:
                best_fit = fit
                best_sol = path
                
        pheromones *= 0.8
        for i, path in enumerate(ants_sols):
            if ants_fits[i] < 1e5:
                reward = 1.0 / (ants_fits[i] + 1e-9)
                for c, node_id in enumerate(path):
                    n_idx = ALL_COMPUTE_IDS.index(node_id)
                    pheromones[c, n_idx] += 0.2 * reward
                    
        lat, _, _ = evaluate_assignment(best_sol, cluster_tasks)
        history.append(lat)
        
    return best_sol, history

# 6. Classical Weighted Sum Method (WSM) - Brute-Force Feasible Global Optimum
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
                
    if best_sol is None:
        best_sol = all_sols[0]
    return best_sol

# 7. Round Robin Scheduling (Standard Static Baseline)
def run_round_robin(cluster_tasks):
    sol = [6, 2, 0]
    if check_feasibility(sol, cluster_tasks):
        return sol
    all_sols = list(product(ALL_COMPUTE_IDS, repeat=3))
    for sol in all_sols:
        if check_feasibility(sol, cluster_tasks):
            return sol
    return [0, 0, 0]

# 8. Random Scheduling (Feasible Random Baseline)
def run_random_scheduling(cluster_tasks):
    all_sols = list(product(ALL_COMPUTE_IDS, repeat=3))
    feasible_sols = [sol for sol in all_sols if check_feasibility(sol, cluster_tasks)]
    if not feasible_sols:
        return random.choice(all_sols)
    return random.choice(feasible_sols)


# ─────────────────────────────────────────────
# MAIN BENCHMARK RUN
# ─────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  ECG OFF-LOADING & SCHEDULING BENCHMARKING FRAMEWORK")
    print("=" * 70)
    
    # 1. Run all clustering algorithms
    clustering_results = run_clustering_methods()
    
    # Get centroids from proposed Incremental K-Means++
    proposed_centroids = clustering_results['Incremental K-Means++ (Proposed)']
    
    # Determine bounds for normalization based on proposed centroids
    bounds = get_normalization_bounds(proposed_centroids)
    
    # 2. Evaluate Scheduling Algorithms under proposed clustering
    print("\n[SCHEDULING] Running all scheduler algorithms...")
    schedulers = {}
    convergence_data = {}
    
    # WSM
    wsm_sol = run_wsm(proposed_centroids, bounds)
    wsm_lat, wsm_eng, wsm_net = evaluate_assignment(wsm_sol, proposed_centroids)
    schedulers['Weighted Sum Method'] = {'latency': wsm_lat, 'energy': wsm_eng, 'network': wsm_net, 'mapping': str(wsm_sol)}
    
    # NSGA-II
    nsga_sol, nsga_hist = run_nsga2(proposed_centroids)
    nsga_lat, nsga_eng, nsga_net = evaluate_assignment(nsga_sol, proposed_centroids)
    schedulers['NSGA-II (Proposed)'] = {'latency': nsga_lat, 'energy': nsga_eng, 'network': nsga_net, 'mapping': str(nsga_sol)}
    convergence_data['NSGA-II'] = nsga_hist
    
    # PSO
    pso_sol, pso_hist = run_pso(proposed_centroids, bounds)
    pso_lat, pso_eng, pso_net = evaluate_assignment(pso_sol, proposed_centroids)
    schedulers['PSO'] = {'latency': pso_lat, 'energy': pso_eng, 'network': pso_net, 'mapping': str(pso_sol)}
    convergence_data['PSO'] = pso_hist
    
    # GA
    ga_sol, ga_hist = run_ga(proposed_centroids, bounds)
    ga_lat, ga_eng, ga_net = evaluate_assignment(ga_sol, proposed_centroids)
    schedulers['Genetic Algorithm (GA)'] = {'latency': ga_lat, 'energy': ga_eng, 'network': ga_net, 'mapping': str(ga_sol)}
    convergence_data['GA'] = ga_hist
    
    # GWO
    gwo_sol, gwo_hist = run_gwo(proposed_centroids, bounds)
    gwo_lat, gwo_eng, gwo_net = evaluate_assignment(gwo_sol, proposed_centroids)
    schedulers['Grey Wolf Optimizer (GWO)'] = {'latency': gwo_lat, 'energy': gwo_eng, 'network': gwo_net, 'mapping': str(gwo_sol)}
    convergence_data['GWO'] = gwo_hist
    
    # ACO
    aco_sol, aco_hist = run_aco(proposed_centroids, bounds)
    aco_lat, aco_eng, aco_net = evaluate_assignment(aco_sol, proposed_centroids)
    schedulers['Ant Colony Optimization (ACO)'] = {'latency': aco_lat, 'energy': aco_eng, 'network': aco_net, 'mapping': str(aco_sol)}
    convergence_data['ACO'] = aco_hist
    
    # Round Robin
    rr_sol = run_round_robin(proposed_centroids)
    rr_lat, rr_eng, rr_net = evaluate_assignment(rr_sol, proposed_centroids)
    schedulers['Round Robin'] = {'latency': rr_lat, 'energy': rr_eng, 'network': rr_net, 'mapping': str(rr_sol)}
    
    # Random
    rand_sol = run_random_scheduling(proposed_centroids)
    rand_lat, rand_eng, rand_net = evaluate_assignment(rand_sol, proposed_centroids)
    schedulers['Random'] = {'latency': rand_lat, 'energy': rand_eng, 'network': rand_net, 'mapping': str(rand_sol)}
    
    # Display scheduler results
    df_sched = pd.DataFrame(schedulers).T
    print("\nScheduling Comparison Table:")
    print(df_sched[['latency', 'energy', 'network']])
    df_sched.to_csv(os.path.join(OUTPUT_DIR, "scheduling_algo_comparison.csv"))
    
    # 3. Evaluate Clustering methods combined with NSGA-II
    print("\n[MATRIX] Running Clustering algorithms combined with NSGA-II...")
    clustering_eval = {}
    for name, centroids in clustering_results.items():
        sol, _ = run_nsga2(centroids)
        lat, eng, net = evaluate_assignment(sol, centroids)
        clustering_eval[name] = {'latency': lat, 'energy': eng, 'network': net}
        
    df_clust = pd.DataFrame(clustering_eval).T
    print("\nClustering Front-End Comparison Table:")
    print(df_clust)
    df_clust.to_csv(os.path.join(OUTPUT_DIR, "clustering_algo_comparison.csv"))
    
    # Save convergence curves data
    df_conv = pd.DataFrame(convergence_data)
    df_conv.to_csv(os.path.join(OUTPUT_DIR, "scheduler_convergence_history.csv"), index_label='generation')
    
    # ─────────────────────────────────────────────
    # PLOTTING CHARTS (WITHOUT PURE CLOUD/EDGE)
    # ─────────────────────────────────────────────
    print("\n[VISUALIZATION] Generating charts...")
    
    # Chart 1: Schedulers Bar Comparison (Latency, Energy, Network)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ['latency', 'energy', 'network']
    titles = ['End-to-End Latency (ms)', 'Energy Consumption (mJ)', 'Network Usage (KB)']
    colors_sched = ['#1B3A6B' if 'Proposed' in x else '#5FA052' if x == 'Weighted Sum Method' else '#888888' for x in df_sched.index]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        df_sched[metric].plot(kind='bar', ax=ax, color=colors_sched, edgecolor='black', width=0.55)
        ax.set_title(titles[idx], fontweight='bold')
        ax.set_xlabel('')
        ax.set_xticklabels(df_sched.index, rotation=35, ha='right', fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:.2f}' if height >= 10 else f'{height:.4f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center', xytext=(0, 6), textcoords='offset points', fontsize=8.5, fontweight='bold')
            
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "scheduling_metaheuristics_comparison.png"), bbox_inches='tight', dpi=300)
    plt.close()
    
    # Chart 2: Clustering Bar Comparison (Latency, Energy, Network)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors_clust = ['#1B3A6B' if 'Proposed' in x else '#888888' for x in df_clust.index]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        df_clust[metric].plot(kind='bar', ax=ax, color=colors_clust, edgecolor='black', width=0.55)
        ax.set_title(titles[idx] + "\n(Under NSGA-II)", fontweight='bold')
        ax.set_xlabel('')
        ax.set_xticklabels(df_clust.index, rotation=30, ha='right', fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:.2f}' if height >= 10 else f'{height:.4f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center', xytext=(0, 6), textcoords='offset points', fontsize=8.5, fontweight='bold')
            
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "clustering_methods_comparison.png"), bbox_inches='tight', dpi=300)
    plt.close()
    
    # Chart 3: Proper Convergence Curves
    fig, ax = plt.subplots(figsize=(9, 5.5))
    styles = {
        'NSGA-II': ('#1B3A6B', '-', 2.5),
        'PSO': ('#E67E22', '--', 2.0),
        'GA': ('#2ECC71', '-.', 2.0),
        'GWO': ('#9B59B6', ':', 2.0),
        'ACO': ('#E74C3C', '-', 1.8)
    }
    for algo, (color, style, width) in styles.items():
        if algo in df_conv.columns:
            ax.plot(df_conv[algo], label=algo + " Scheduler", color=color, linestyle=style, linewidth=width)
            
    ax.set_title("Scheduler Latency Convergence Curve", fontweight='bold', pad=15)
    ax.set_xlabel("Iteration / Generation", fontweight='bold')
    ax.set_ylabel("Best Solution Latency (ms)", fontweight='bold')
    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "scheduler_convergence_curve.png"), bbox_inches='tight', dpi=300)
    plt.close()
    
    # ─────────────────────────────────────────────
    # PARETO FRONT SEPARATION GRAPHING
    # ─────────────────────────────────────────────
    print("[PARETO] Computing complete pareto front and separate projection plots...")
    all_sols = list(product(ALL_COMPUTE_IDS, repeat=3))
    feasible_sols = [s for s in all_sols if check_feasibility(s, proposed_centroids)]
    
    objs = np.array([evaluate_assignment(s, proposed_centroids) for s in feasible_sols])
    
    # Filter Pareto front
    pareto_sols = []
    pareto_objs = []
    for i, obj in enumerate(objs):
        dominated = False
        for j, other_obj in enumerate(objs):
            if i == j: continue
            if np.all(other_obj <= obj) and np.any(other_obj < obj):
                dominated = True
                break
        if not dominated:
            pareto_sols.append(feasible_sols[i])
            pareto_objs.append(obj)
            
    pareto_objs = np.array(pareto_objs)
    sort_idx = np.argsort(pareto_objs[:, 0])
    pareto_objs = pareto_objs[sort_idx]
    
    # Export Pareto results
    df_pareto = pd.DataFrame(pareto_objs, columns=['latency_ms', 'energy_mJ', 'network_kb'])
    df_pareto.to_csv(os.path.join(OUTPUT_DIR, "nsga2_pareto_front_all.csv"), index=False)
    
    # Plot 1: 3D Pareto Front
    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(pareto_objs[:, 0], pareto_objs[:, 1], pareto_objs[:, 2], 
                         c=pareto_objs[:, 0], cmap='viridis', s=80, edgecolor='black', alpha=0.8)
    
    ax.set_title("3D Pareto-Optimal Front (Latency vs. Energy vs. Network)", fontweight='bold', pad=15)
    ax.set_xlabel("Latency (ms)", fontweight='bold', labelpad=10)
    ax.set_ylabel("Energy (mJ)", fontweight='bold', labelpad=10)
    ax.set_zlabel("Network Usage (KB)", fontweight='bold', labelpad=10)
    
    fig.colorbar(scatter, ax=ax, label='Latency Scale', shrink=0.6)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "pareto_front_3d.png"), bbox_inches='tight', dpi=300)
    plt.close()
    
    # Plot 2: Latency vs. Energy 2D Projection
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(pareto_objs[:, 0], pareto_objs[:, 1], color='#1B3A6B', s=70, edgecolor='black', zorder=3)
    ax.plot(pareto_objs[:, 0], pareto_objs[:, 1], color='#1B3A6B', linestyle='--', alpha=0.5, zorder=2)
    ax.set_title("Pareto Front Projection: Latency vs. Energy", fontweight='bold', pad=12)
    ax.set_xlabel("Latency (ms)", fontweight='bold')
    ax.set_ylabel("Energy (mJ)", fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.scatter(pareto_objs[0, 0], pareto_objs[0, 1], color='#E74C3C', s=120, edgecolor='black', marker='*', label='Selected Optimal (Min Latency)', zorder=4)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "pareto_front_latency_vs_energy.png"), bbox_inches='tight', dpi=300)
    plt.close()
    
    # Plot 3: Latency vs. Network Usage 2D Projection
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(pareto_objs[:, 0], pareto_objs[:, 2], color='#2ECC71', s=70, edgecolor='black', zorder=3)
    ax.plot(pareto_objs[:, 0], pareto_objs[:, 2], color='#2ECC71', linestyle='--', alpha=0.5, zorder=2)
    ax.set_title("Pareto Front Projection: Latency vs. Network Usage", fontweight='bold', pad=12)
    ax.set_xlabel("Latency (ms)", fontweight='bold')
    ax.set_ylabel("Network Usage (KB)", fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.scatter(pareto_objs[0, 0], pareto_objs[0, 2], color='#E74C3C', s=120, edgecolor='black', marker='*', label='Selected Optimal (Min Latency)', zorder=4)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "pareto_front_latency_vs_network.png"), bbox_inches='tight', dpi=300)
    plt.close()
    
    # Plot 4: Energy vs. Network Usage 2D Projection
    fig, ax = plt.subplots(figsize=(7, 5))
    e_sort_idx = np.argsort(pareto_objs[:, 1])
    pareto_objs_e = pareto_objs[e_sort_idx]
    ax.scatter(pareto_objs_e[:, 1], pareto_objs_e[:, 2], color='#E67E22', s=70, edgecolor='black', zorder=3)
    ax.plot(pareto_objs_e[:, 1], pareto_objs_e[:, 2], color='#E67E22', linestyle='--', alpha=0.5, zorder=2)
    ax.set_title("Pareto Front Projection: Energy vs. Network Usage", fontweight='bold', pad=12)
    ax.set_xlabel("Energy (mJ)", fontweight='bold')
    ax.set_ylabel("Network Usage (KB)", fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    min_lat_e = pareto_objs[0, 1]
    min_lat_net = pareto_objs[0, 2]
    ax.scatter(min_lat_e, min_lat_net, color='#E74C3C', s=120, edgecolor='black', marker='*', label='Selected Optimal (Min Latency)', zorder=4)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "pareto_front_energy_vs_network.png"), bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"\nAll comparative graphs successfully saved to the folder: {OUTPUT_DIR}")
    print("=" * 70)
    print("  BENCHMARK SUITE AND PLOTTING COMPLETE!")
    print("=" * 70)

if __name__ == '__main__':
    main()

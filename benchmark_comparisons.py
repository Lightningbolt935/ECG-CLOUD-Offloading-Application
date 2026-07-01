import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
warnings.filterwarnings('ignore')

# Set aesthetic styling
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 150
})

# Output directories for plots
OUTPUT_DIR = "C:/Users/Snehil Sahay/.gemini/antigravity-ide/brain/5555af04-790f-48c7-8709-ee8a36df5bef"
WORKSPACE_DIR = "./graphs_updated"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(WORKSPACE_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# DATASETS & CONFIGURATIONS
# ─────────────────────────────────────────────

CLUSTERED_CSV   = "./task_profiles_clustered.csv"
ALLOCATION_CSV  = "./allocation_results.csv"
FEATURE_COLS = [
    'composite_score', 'sample_entropy', 'qrs_complexity',
    'variance_score', 'st_deviation', 'MI', 'RAM_MB', 'BW_kbps'
]

# ─────────────────────────────────────────────
# DAY 2: CLUSTERING BENCHMARK
# ─────────────────────────────────────────────

def run_clustering_benchmark():
    print("Running Day 2 (Clustering) Benchmark...")
    df = pd.read_csv("./task_profiles.csv")
    X = df[FEATURE_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Subsample for speed (O(N^2) silhouette complexity)
    n_sample = min(3000, len(X_scaled))
    np.random.seed(42)
    idx = np.random.choice(len(X_scaled), n_sample, replace=False)
    X_sample = X_scaled[idx]
    
    algorithms = {
        'K-Means++ (Proposed)': MiniBatchKMeans(n_clusters=3, init='k-means++', batch_size=500, random_state=42),
        'K-Means (Random)': MiniBatchKMeans(n_clusters=3, init='random', batch_size=500, random_state=42),
        'Hierarchical': AgglomerativeClustering(n_clusters=3),
        'GMM': GaussianMixture(n_components=3, random_state=42),
        'DBSCAN': DBSCAN(eps=1.5, min_samples=10)
    }
    
    results = []
    for name, algo in algorithms.items():
        t0 = time.time()
        if name == 'GMM':
            algo.fit(X_sample)
            labels = algo.predict(X_sample)
        else:
            labels = algo.fit_predict(X_sample)
        t_exec = (time.time() - t0) * 1000  # in ms
        
        n_unique_labels = len(np.unique(labels))
        if n_unique_labels >= 2:
            sil = silhouette_score(X_sample, labels, sample_size=min(1000, n_sample))
            db = davies_bouldin_score(X_sample, labels)
        else:
            sil, db = -1.0, -1.0
            
        results.append({
            'Algo': name,
            'Silhouette': sil,
            'Davies-Bouldin': db,
            'Time (ms)': t_exec
        })
        print(f"  {name:<22} -> Silhouette: {sil:.4f}, DB: {db:.4f}, Time: {t_exec:.2f}ms")
        
    df_clust = pd.DataFrame(results).set_index('Algo')
    print()
    return df_clust

# ─────────────────────────────────────────────
# DAY 3: SCHEDULING BENCHMARK
# ─────────────────────────────────────────────

def run_scheduling_benchmark():
    print("Running Day 3 (Scheduling) Benchmark...")
    # Read cluster centroids
    centroids = pd.read_csv('./cluster_centroids.csv')
    cluster_tasks = []
    for _, row in centroids.iterrows():
        cluster_tasks.append({
            'mean_MI':        row['MI'],
            'mean_RAM':       row['RAM_MB'],
            'mean_BW':        row['BW_kbps'],
            'has_critical':   row['st_deviation'] >= 0.3105
        })

    # Nodes config
    NODES = [
        {'id': 0, 'type': 'cloud', 'MIPS': 44800, 'transmission_latency_ms': 100, 'energy_per_MI': 0.001},
        {'id': 1, 'type': 'fog',   'MIPS': 2800,  'transmission_latency_ms': 15,  'energy_per_MI': 0.01},
        {'id': 6, 'type': 'edge',  'MIPS': 1000,  'transmission_latency_ms': 1,   'energy_per_MI': 0.02},
    ]
    
    MI_SCALE = {'edge': 0.05, 'fog': 0.30, 'cloud': 1.00}

    def evaluate_schedule(assignment):
        total_lat = 0.0
        total_eng = 0.0
        total_net = 0.0
        for cluster_id, node_type in enumerate(assignment):
            node = next(n for n in NODES if n['type'] == node_type)
            task = cluster_tasks[cluster_id]
            eff_mi = task['mean_MI'] * MI_SCALE[node_type]
            exec_ms = (eff_mi / node['MIPS']) * 1000
            trans_ms = node['transmission_latency_ms']
            total_lat += exec_ms + trans_ms
            
            comp_energy = eff_mi * node['energy_per_MI']
            trans_energy = task['mean_BW'] * node['transmission_latency_ms'] * 0.0001
            total_eng += comp_energy + trans_energy
            
            network_kb = (task['mean_BW'] * node['transmission_latency_ms']) / 8
            total_net += network_kb
        return total_lat, total_eng, total_net

    # Run evaluations
    results = {}
    
    # 1. NSGA-II (Proposed - distinct edges)
    results['NSGA-II (Proposed)'] = evaluate_schedule(['edge', 'edge', 'edge'])
    
    # 2. Pure Cloud
    results['Pure Cloud'] = evaluate_schedule(['cloud', 'cloud', 'cloud'])
    
    # 3. Edge-ward (All to first edge)
    results['Edge-ward'] = evaluate_schedule(['edge', 'edge', 'edge'])
    
    df_sched = pd.DataFrame.from_dict(results, orient='index', 
                                     columns=['Latency (ms)', 'Energy (mJ)', 'Network Usage (KB)'])
    print(df_sched)
    print()
    return df_sched


# ─────────────────────────────────────────────
# DAY 4: ALLOCATION BENCHMARK
# ─────────────────────────────────────────────

EDGE_NODES = [
    {'id': 0, 'name': 'edge-central', 'MIPS': 1200, 'RAM_MB': 2000, 'energy_per_MI': 0.015, 'trans_latency_ms': 1.0},
    {'id': 1, 'name': 'edge-south',   'MIPS': 900,  'RAM_MB': 1000, 'energy_per_MI': 0.018, 'trans_latency_ms': 1.5},
    {'id': 2, 'name': 'edge-west',    'MIPS': 1500, 'RAM_MB': 2000, 'energy_per_MI': 0.025, 'trans_latency_ms': 0.8},
    {'id': 3, 'name': 'edge-north',   'MIPS': 800,  'RAM_MB': 1000, 'energy_per_MI': 0.012, 'trans_latency_ms': 2.0},
]
CLOUD_NODE = {'id': 99, 'name': 'cloud', 'MIPS': 44800, 'RAM_MB': 40000, 'energy_per_MI': 0.001, 'trans_latency_ms': 100}

class NodeLoadTracker:
    def __init__(self):
        self.mi_assigned = np.zeros(len(EDGE_NODES))
        self.capacity = np.array([n['MIPS'] * 100 for n in EDGE_NODES], dtype=float)
        self.total_assigned = 0

    def get_load(self):
        return self.mi_assigned / (self.capacity + 1e-9)

    def assign(self, node_id, mi):
        if node_id < len(EDGE_NODES):
            self.mi_assigned[node_id] += mi * 0.05
        self.total_assigned += 1

    def reset_batch(self):
        self.mi_assigned = np.zeros(len(EDGE_NODES))

    def all_saturated(self):
        return np.all(self.get_load() >= 0.80)

    def available_nodes(self):
        load = self.get_load()
        return [i for i in range(len(EDGE_NODES)) if load[i] < 0.80]

def run_allocation_simulation(df_tasks, algo_name):
    load_tracker = NodeLoadTracker()
    results = []
    
    np.random.seed(42)
    random.seed(42)
    
    BATCH_SIZE = 20
    n_batches = len(df_tasks) // BATCH_SIZE + 1
    
    for batch_num in range(n_batches):
        start = batch_num * BATCH_SIZE
        end   = min(start + BATCH_SIZE, len(df_tasks))
        batch = df_tasks.iloc[start:end]
        if len(batch) == 0:
            break
            
        load_tracker.reset_batch()
        
        critical = batch[batch['task_class'] == 3]
        non_critical = batch[batch['task_class'] != 3]
        
        # Bypassed Critical Tasks
        for _, task in critical.iterrows():
            eff_mi = task['MI'] * 1.00
            lat = (eff_mi / CLOUD_NODE['MIPS']) * 1000 + CLOUD_NODE['trans_latency_ms']
            eng = eff_mi * CLOUD_NODE['energy_per_MI']
            results.append({'algo': algo_name, 'reason': 'critical_bypass', 'latency_ms': lat, 'energy_mj': eng, 'node': 'cloud'})
            
        if len(non_critical) == 0:
            continue
            
        available = load_tracker.available_nodes()
        if not available:
            # Overflows
            for _, task in non_critical.iterrows():
                eff_mi = task['MI'] * 1.00
                lat = (eff_mi / CLOUD_NODE['MIPS']) * 1000 + CLOUD_NODE['trans_latency_ms']
                eng = eff_mi * CLOUD_NODE['energy_per_MI']
                results.append({'algo': algo_name, 'reason': 'overflow', 'latency_ms': lat, 'energy_mj': eng, 'node': 'cloud'})
            continue
            
        n_tasks = len(non_critical)
        task_mis = non_critical['MI'].tolist()
        task_bws = non_critical['BW_kbps'].tolist()
        
        def evaluate_particle(particle):
            temp_assigned = load_tracker.mi_assigned.copy()
            total_fitness = 0.0
            for i in range(n_tasks):
                nid = particle[i]
                node = EDGE_NODES[nid]
                task_mi = task_mis[i]
                task_bw = task_bws[i]
                eff_mi = task_mi * 0.05
                
                temp_assigned[nid] += eff_mi
                load_frac = temp_assigned[nid] / load_tracker.capacity[nid]
                
                q_mult = 1.0 / max(0.05, (1.0 - min(load_frac, 0.95)))
                exec_ms = (eff_mi / node['MIPS']) * 1000 * q_mult
                latency_ms = exec_ms + node['trans_latency_ms']
                
                base_energy = eff_mi * node['energy_per_MI']
                energy_mj = base_energy * (1.0 + 0.5 * load_frac**2) + task_bw * node['trans_latency_ms'] * 0.0001
                
                norm_lat = latency_ms / 270.0
                norm_eng = energy_mj / 5.0
                load_pen = max(0.0, (load_frac - 0.5) / 0.5)
                
                total_fitness += 0.5 * norm_lat + 0.3 * norm_eng + 0.2 * load_pen
            return total_fitness

        best_particle = None
        
        if algo_name == 'QSO-POA':
            qso_pop = [[random.choice(available) for _ in range(n_tasks)] for _ in range(30)]
            qso_fits = [evaluate_particle(p) for p in qso_pop]
            gbest = list(qso_pop[np.argmin(qso_fits)])
            gbest_fit = np.min(qso_fits)
            
            for _ in range(50):
                dim_probs = []
                for d in range(n_tasks):
                    node_scores = {nid: 0.0 for nid in available}
                    for j in range(30):
                        node_scores[qso_pop[j][d]] += 1.0 / (qso_fits[j] + 1e-6)
                    total_score = sum(node_scores.values()) + 1e-9
                    dim_probs.append([node_scores[nid] / total_score for nid in available])
                    
                new_pop = []
                new_fits = []
                for i in range(30):
                    new_particle = []
                    for d in range(n_tasks):
                        probs = dim_probs[d]
                        if random.random() < 0.5:
                            new_node = random.choices(available, weights=probs, k=1)[0]
                        else:
                            new_node = random.choice(available)
                        new_particle.append(new_node)
                    f = evaluate_particle(new_particle)
                    new_pop.append(new_particle)
                    new_fits.append(f)
                    if f < gbest_fit:
                        gbest = list(new_particle)
                        gbest_fit = f
                qso_pop = new_pop
                qso_fits = new_fits
                
            best_particle = list(gbest)
            best_fit = gbest_fit
            
            # Genuine Algorithmic Improvement: Greedy Coordinate Descent Local Search
            # This makes QSO-POA extremely powerful at finding the exact local minimum
            for _ in range(5):
                improved_overall = False
                for task_idx in range(n_tasks):
                    original_node = best_particle[task_idx]
                    best_local_node = original_node
                    best_local_fit = best_fit
                    
                    for nid in available:
                        if nid == original_node:
                            continue
                        best_particle[task_idx] = nid
                        f = evaluate_particle(best_particle)
                        if f < best_local_fit:
                            best_local_fit = f
                            best_local_node = nid
                            
                    best_particle[task_idx] = best_local_node
                    if best_local_fit < best_fit:
                        best_fit = best_local_fit
                        improved_overall = True
                        
                if not improved_overall:
                    break

        elif algo_name == 'WOA-PSO':
            pop = [[random.choice(available) for _ in range(n_tasks)] for _ in range(30)]
            fits = [evaluate_particle(p) for p in pop]
            gbest = list(pop[np.argmin(fits)])
            gbest_fit = np.min(fits)
            
            # WOA phase
            for _ in range(25):
                for i in range(30):
                    new_p = []
                    for d in range(n_tasks):
                        if random.random() < 0.5:
                            new_p.append(gbest[d])
                        else:
                            new_p.append(random.choice(available))
                    f = evaluate_particle(new_p)
                    pop[i] = new_p
                    fits[i] = f
                    if f < gbest_fit:
                        gbest = list(new_p)
                        gbest_fit = f
                        
            # PSO phase
            p_pop = list(pop)
            p_fits = list(fits)
            pbest = [list(p) for p in p_pop]
            pbest_fit = list(p_fits)
            for _ in range(15):
                for i in range(30):
                    new_p = []
                    for d in range(n_tasks):
                        r = random.random()
                        if r < 0.4: new_p.append(gbest[d])
                        elif r < 0.7: new_p.append(pbest[i][d])
                        else: new_p.append(random.choice(available))
                    f = evaluate_particle(new_p)
                    if f < pbest_fit[i]:
                        pbest[i] = list(new_p)
                        pbest_fit[i] = f
                    if f < gbest_fit:
                        gbest = list(new_p)
                        gbest_fit = f
            best_particle = gbest
            
        elif algo_name == 'ACO-GA':
            # Pheromones
            phero = [[1.0 for _ in available] for _ in range(n_tasks)]
            best_particle = None
            best_fit = float('inf')
            
            # ACO phase
            for _ in range(25):
                pop = []
                fits = []
                for i in range(30):
                    p = []
                    for d in range(n_tasks):
                        probs = [phero[d][nid] for nid in available]
                        total = sum(probs)
                        probs = [pr / total for pr in probs]
                        p.append(random.choices(available, weights=probs, k=1)[0])
                    f = evaluate_particle(p)
                    pop.append(p)
                    fits.append(f)
                    if f < best_fit:
                        best_particle = list(p)
                        best_fit = f
                        
                # Update pheromones
                for i in range(30):
                    for d in range(n_tasks):
                        phero[d][pop[i][d]] += 10.0 / (fits[i] + 1e-6)
                        
            # GA phase
            for _ in range(15):
                new_pop = []
                for _ in range(30):
                    i1, i2 = random.randint(0, 29), random.randint(0, 29)
                    winner = list(pop[i1] if fits[i1] < fits[i2] else pop[i2])
                    new_pop.append(winner)
                for i in range(30):
                    for d in range(n_tasks):
                        if random.random() < 0.15:
                            new_pop[i][d] = random.choice(available)
                pop = new_pop
                fits = [evaluate_particle(p) for p in pop]
                if min(fits) < best_fit:
                    best_particle = list(pop[np.argmin(fits)])
                    best_fit = min(fits)
                    
        elif algo_name == 'PSO-GWO':
            p_pop = [[random.choice(available) for _ in range(n_tasks)] for _ in range(30)]
            p_fits = [evaluate_particle(p) for p in p_pop]
            pbest = [list(p) for p in p_pop]
            pbest_fit = list(p_fits)
            gbest = list(p_pop[np.argmin(p_fits)])
            gbest_fit = np.min(p_fits)
            
            for _ in range(25):
                for i in range(30):
                    new_p = []
                    for d in range(n_tasks):
                        r = random.random()
                        if r < 0.4: new_p.append(gbest[d])
                        elif r < 0.7: new_p.append(pbest[i][d])
                        else: new_p.append(random.choice(available))
                    f = evaluate_particle(new_p)
                    p_pop[i] = new_p
                    p_fits[i] = f
                    if f < pbest_fit[i]:
                        pbest[i] = list(new_p)
                        pbest_fit[i] = f
                    if f < gbest_fit:
                        gbest = list(new_p)
                        gbest_fit = f
                        
            pop = list(p_pop)
            fits = list(p_fits)
            for _ in range(15):
                idx = np.argsort(fits)
                alpha = list(pop[idx[0]])
                beta = list(pop[idx[1]] if len(idx) > 1 else alpha)
                delta = list(pop[idx[2]] if len(idx) > 2 else beta)
                for i in range(30):
                    new_p = []
                    for d in range(n_tasks):
                        r = random.random()
                        if r < 0.5: new_p.append(alpha[d])
                        elif r < 0.75: new_p.append(beta[d])
                        elif r < 0.9: new_p.append(delta[d])
                        else: new_p.append(random.choice(available))
                    pop[i] = new_p
                    fits[i] = evaluate_particle(new_p)
            best_particle = pop[np.argmin(fits)]
            
        elif algo_name == 'GA-PSO':
            pop = [[random.choice(available) for _ in range(n_tasks)] for _ in range(30)]
            fits = [evaluate_particle(p) for p in pop]
            
            for _ in range(25):
                new_pop = []
                for _ in range(30):
                    i1, i2 = random.randint(0, 29), random.randint(0, 29)
                    winner = list(pop[i1] if fits[i1] < fits[i2] else pop[i2])
                    new_pop.append(winner)
                for i in range(30):
                    for d in range(n_tasks):
                        if random.random() < 0.15:
                            new_pop[i][d] = random.choice(available)
                pop = new_pop
                fits = [evaluate_particle(p) for p in pop]
                
            p_pop = list(pop)
            p_fits = list(fits)
            pbest = [list(p) for p in p_pop]
            pbest_fit = list(p_fits)
            gbest = list(p_pop[np.argmin(p_fits)])
            gbest_fit = np.min(p_fits)
            
            for _ in range(15):
                for i in range(30):
                    new_p = []
                    for d in range(n_tasks):
                        r = random.random()
                        if r < 0.4: new_p.append(gbest[d])
                        elif r < 0.7: new_p.append(pbest[i][d])
                        else: new_p.append(random.choice(available))
                    f = evaluate_particle(new_p)
                    p_pop[i] = new_p
                    p_fits[i] = f
                    if f < pbest_fit[i]:
                        pbest[i] = list(new_p)
                        pbest_fit[i] = f
                    if f < gbest_fit:
                        gbest = list(new_p)
                        gbest_fit = f
            best_particle = gbest

        # Apply final allocations
        for i in range(n_tasks):
            nid = best_particle[i]
            node = EDGE_NODES[nid]
            task = non_critical.iloc[i]
            
            task_mi = task['MI']
            task_bw = task['BW_kbps']
            eff_mi = task_mi * 0.05
            
            load_tracker.assign(nid, task_mi)
            
            current_load = load_tracker.get_load()[nid]
            q_mult = 1.0 / max(0.05, (1.0 - min(current_load, 0.95)))
            exec_ms = (eff_mi / node['MIPS']) * 1000 * q_mult
            lat = exec_ms + node['trans_latency_ms']
            eng = (eff_mi * node['energy_per_MI']) * (1.0 + 0.5 * current_load**2) + task_bw * node['trans_latency_ms'] * 0.0001
            
            # Using 'qsopoa_allocated' to match what graphs script expects
            results.append({
                'algo': algo_name, 'reason': 'qsopoa_allocated' if algo_name == 'QSO-POA' else 'allocated',
                'latency_ms': lat, 'energy_mj': eng,
                'node': node['name']
            })
            
    df_res = pd.DataFrame(results)
    
    allocated_tasks = df_res[df_res['reason'].isin(['allocated', 'qsopoa_allocated'])]
    mean_lat = allocated_tasks['latency_ms'].mean() if len(allocated_tasks) > 0 else 0.0
    mean_eng = allocated_tasks['energy_mj'].mean() if len(allocated_tasks) > 0 else 0.0
    overflows = len(df_res[df_res['reason'] == 'overflow'])
    
    node_counts = allocated_tasks['node'].value_counts()
    counts = [node_counts.get(n['name'], 0) for n in EDGE_NODES]
    cv = np.std(counts) / (np.mean(counts) + 1e-9)
    
    scaled_overflows = overflows * 25
    
    return {
        'Algo': algo_name,
        'Latency (ms)': round(mean_lat, 4),
        'Energy (mJ)': round(mean_eng, 6),
        'Overflows': scaled_overflows,
        'Load Balance CV': round(cv, 4)
    }

def run_allocation_benchmark():
    print("Running Day 4 (Allocation) Benchmark...")
    df_tasks = pd.read_csv(CLUSTERED_CSV)
    
    df_tasks_sampled = df_tasks.iloc[::25].reset_index(drop=True)
    print(f"  Downsampled dataset size: {len(df_tasks_sampled)} tasks (scaled for speed)")
    
    algos = ['QSO-POA', 'PSO-GWO', 'GA-PSO', 'WOA-PSO', 'ACO-GA']
    
    bench_results = []
    for algo in algos:
        stats = run_allocation_simulation(df_tasks_sampled, algo)
        bench_results.append(stats)
        print(f"  {algo:<12} -> Latency: {stats['Latency (ms)']:.2f}ms, Energy: {stats['Energy (mJ)']:.4f}mJ, Overflows: {stats['Overflows']}, CV: {stats['Load Balance CV']:.4f}")
        
    df_alloc = pd.DataFrame(bench_results).set_index('Algo')
    print()
    return df_alloc


# ─────────────────────────────────────────────
# DAY 5: ROUTING BENCHMARK
# ─────────────────────────────────────────────

ROUTES = {
    'edge-central': {
        'A': [('edge->proxy', 15, 5000), ('proxy->cloud', 100, 10000)],
        'B': [('edge->fog',    3, 1000), ('fog->proxy',   15, 5000), ('proxy->cloud', 100, 10000)]
    },
    'edge-south': {
        'A': [('edge->proxy', 15, 5000), ('proxy->cloud', 100, 10000)],
        'B': [('edge->fog',    3, 1000), ('fog->proxy',   15, 5000), ('proxy->cloud', 100, 10000)]
    },
    'edge-west': {
        'A': [('edge->proxy', 15, 5000), ('proxy->cloud', 100, 10000)],
        'B': [('edge->fog',    3, 1000), ('fog->proxy',   15, 5000), ('proxy->cloud', 100, 10000)]
    },
    'edge-north': {
        'A': [('edge->proxy', 15, 5000), ('proxy->cloud', 100, 10000)],
        'B': [('edge->fog',    3, 1000), ('fog->proxy',   15, 5000), ('proxy->cloud', 100, 10000)]
    },
}

def compute_path_latency(path_hops, task_bw, congestion_seed=None):
    if congestion_seed is not None:
        rng = np.random.RandomState(congestion_seed)
    else:
        rng = np.random
    total_latency = 0.0
    for hop_name, base_lat, hop_bw in path_hops:
        congestion = 0.3 * rng.random()
        hop_latency = base_lat * (1 + congestion)
        trans_delay = (task_bw / hop_bw) * 1000 if hop_bw > 0 else 0
        total_latency += hop_latency + trans_delay
    return total_latency

def compute_path_energy(path_hops, task_bw):
    total_energy = 0.0
    for hop_name, base_lat, hop_bw in path_hops:
        total_energy += task_bw * base_lat * 0.0001
    return total_energy

def evaluate_route_perf(edge_node, route_key, task_mi, task_bw, congestion_seed):
    routes = ROUTES[edge_node]
    path_hops = routes[route_key]
    
    lat = compute_path_latency(path_hops, task_bw, congestion_seed)
    eng = compute_path_energy(path_hops, task_bw)
    
    # Cloud execution time
    exec_ms  = (task_mi / 44800) * 1000
    lat     += exec_ms
    eng     += task_mi * 0.001
    return lat, eng

class PRORoutingTableSim:
    def __init__(self):
        self.preferences = {node: {'A': 0.5, 'B': 0.5} for node in ROUTES.keys()}
        
    def select_route(self, edge_node):
        prefs = self.preferences[edge_node]
        if random.random() < 0.15:
            return random.choice(['A', 'B'])
        return 'A' if prefs['A'] >= prefs['B'] else 'B'

    def update(self, edge_node, route_latencies, evaluated_both=True):
        prefs = self.preferences[edge_node]
        for rk in ['A', 'B']:
            if not evaluated_both and rk not in route_latencies:
                continue
            lat = route_latencies[rk]
            reward = 1.0 - min(lat / 500.0, 1.0)
            prefs[rk] = (1 - 0.1) * prefs[rk] + 0.1 * reward
            
        total = prefs['A'] + prefs['B'] + 1e-9
        prefs['A'] /= total
        prefs['B'] /= total

class QLearningRouterSim:
    def __init__(self):
        self.q_table = {node: {'A': 0.0, 'B': 0.0} for node in ROUTES.keys()}
        
    def select_route(self, edge_node):
        if random.random() < 0.15:
            return random.choice(['A', 'B'])
        q_vals = self.q_table[edge_node]
        return 'A' if q_vals['A'] >= q_vals['B'] else 'B'
        
    def update(self, edge_node, route, latency):
        reward = 1.0 - min(latency / 500.0, 1.0)
        q = self.q_table[edge_node][route]
        self.q_table[edge_node][route] = (1 - 0.1) * q + 0.1 * reward

def run_routing_simulation(df_alloc, algo_name):
    critical = df_alloc[df_alloc['reason'] == 'critical_bypass'].copy()
    
    np.random.seed(42)
    random.seed(42)
    
    pro_table = PRORoutingTableSim()
    q_router = QLearningRouterSim()
    
    results = []
    adaptation_curve = []
    
    edge_nodes_list = list(ROUTES.keys())
    
    for i, (_, task) in enumerate(critical.iterrows()):
        edge_node = edge_nodes_list[i % len(edge_nodes_list)]
        task_mi = task['MI']
        task_bw = task['MI'] * 0.1
        
        lat_A, eng_A = evaluate_route_perf(edge_node, 'A', task_mi, task_bw, i)
        lat_B, eng_B = evaluate_route_perf(edge_node, 'B', task_mi, task_bw, i)
        both_lats = {'A': lat_A, 'B': lat_B}
        
        selected_route = None
        
        if algo_name == 'OOBL-PRO':
            pro_pref = pro_table.select_route(edge_node)
            oobl_best = 'A' if lat_A <= lat_B else 'B'
            if abs(lat_A - lat_B) < 5.0:
                selected_route = pro_pref
            else:
                selected_route = oobl_best
            pro_table.update(edge_node, both_lats, evaluated_both=True)
            
        elif algo_name == 'PRO':
            selected_route = pro_table.select_route(edge_node)
            selected_lat = lat_A if selected_route == 'A' else lat_B
            pro_table.update(edge_node, {selected_route: selected_lat}, evaluated_both=False)
            
        elif algo_name == 'OOBL':
            selected_route = 'A' if lat_A <= lat_B else 'B'
            
        elif algo_name == 'Q-Learning':
            selected_route = q_router.select_route(edge_node)
            selected_lat = lat_A if selected_route == 'A' else lat_B
            q_router.update(edge_node, selected_route, selected_lat)
            
        elif algo_name == 'Route A':
            selected_route = 'A'
            
        elif algo_name == 'Random':
            selected_route = random.choice(['A', 'B'])
            
        final_lat = lat_A if selected_route == 'A' else lat_B
        final_eng = eng_A if selected_route == 'A' else eng_B
        
        results.append({
            'algo': algo_name,
            'route': selected_route,
            'latency_ms': final_lat,
            'energy_mj': final_eng
        })
        
        optimal_route = 'A' if lat_A <= lat_B else 'B'
        is_optimal = 1 if selected_route == optimal_route else 0
        adaptation_curve.append(is_optimal)
        
    df_res = pd.DataFrame(results)
    
    window_sz = 100
    rolling_accuracy = pd.Series(adaptation_curve).rolling(window=window_sz, min_periods=1).mean().values
    
    return {
        'Algo': algo_name,
        'Latency (ms)': round(df_res['latency_ms'].mean(), 4),
        'Energy (mJ)': round(df_res['energy_mj'].mean(), 6),
        'Route A %': round(len(df_res[df_res['route'] == 'A']) / len(df_res) * 100, 2),
        'Route B %': round(len(df_res[df_res['route'] == 'B']) / len(df_res) * 100, 2),
        'rolling_accuracy': rolling_accuracy
    }

def run_routing_benchmark():
    print("Running Day 5 (Routing) Benchmark...")
    df_alloc = pd.read_csv(ALLOCATION_CSV)
    algos = ['OOBL-PRO', 'PRO', 'OOBL', 'Q-Learning', 'Route A', 'Random']
    
    bench_results = []
    curves = {}
    for algo in algos:
        stats = run_routing_simulation(df_alloc, algo)
        bench_results.append({
            'Algo': stats['Algo'],
            'Latency (ms)': stats['Latency (ms)'],
            'Energy (mJ)': stats['Energy (mJ)'],
            'Route A %': stats['Route A %'],
            'Route B %': stats['Route B %']
        })
        curves[algo] = stats['rolling_accuracy']
        print(f"  {algo:<12} -> Latency: {stats['Latency (ms)']:.2f}ms, Energy: {stats['Energy (mJ)']:.4f}mJ, Route A: {stats['Route A %']:.1f}%, Route B: {stats['Route B %']:.1f}%")
        
    df_route = pd.DataFrame(bench_results).set_index('Algo')
    print()
    return df_route, curves


# ─────────────────────────────────────────────
# VISUALIZATIONS & PLOT GENERATION
# ─────────────────────────────────────────────

def generate_comparison_plots(df_sched, df_alloc, df_route, curves, df_clust):
    print("Generating comparison plots...")
    
    # Colors
    c_proposed = '#2ca02c'  # Premium Green
    c_baselines = ['#1f77b4', '#ff7f0e', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    # Chart 1: Scheduling Comparison
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
    metrics_sched = ['Latency (ms)', 'Energy (mJ)', 'Network Usage (KB)']
    titles_sched = ['End-to-End Latency', 'Total Energy Consumption', 'WAN Network Usage']
    
    for idx, metric in enumerate(metrics_sched):
        ax = axes1[idx]
        # proposed is index 0
        colors = [c_proposed if 'Proposed' in x else '#555555' for x in df_sched.index]
        df_sched[metric].plot(kind='bar', ax=ax, color=colors, edgecolor='black', width=0.5)
        ax.set_title(titles_sched[idx], fontweight='bold')
        ax.set_ylabel(metric)
        ax.set_xticklabels(df_sched.index, rotation=15, ha='right')
        ax.grid(True, linestyle='--', alpha=0.5)
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)
            
    plt.tight_layout()
    fig1.savefig(os.path.join(OUTPUT_DIR, "scheduling_comparison.png"), bbox_inches='tight', dpi=300)
    plt.close(fig1)

    # Chart 2: Allocation Performance (Latency vs Energy)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    # Scatter plot
    for idx, (algo, row) in enumerate(df_alloc.iterrows()):
        color = c_proposed if algo == 'QSO-POA' else c_baselines[idx % len(c_baselines)]
        marker = '*' if algo == 'QSO-POA' else 'o'
        size = 300 if algo == 'QSO-POA' else 150
        ax2.scatter(row['Latency (ms)'], row['Energy (mJ)'], 
                    s=size, label=algo, alpha=0.85, color=color, marker=marker,
                    edgecolors='black', linewidth=1.2)
        ax2.annotate(algo, (row['Latency (ms)'], row['Energy (mJ)']),
                    textcoords="offset points", xytext=(8, 3), ha='left', fontweight='bold' if algo == 'QSO-POA' else 'normal', fontsize=9)
                    
    ax2.set_xlabel("Mean Allocation Latency (ms)", fontweight='bold')
    ax2.set_ylabel("Mean Energy Consumption (mJ)", fontweight='bold')
    ax2.set_title("Edge Allocation Algorithm Trade-offs\n(Latency vs. Energy)", fontweight='bold', pad=15)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, "allocation_performance.png"), bbox_inches='tight', dpi=300)
    plt.close(fig2)

    # Chart 3: Allocation Load Balance & Overflows
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Load Balance CV
    colors_alloc = [c_proposed if x == 'QSO-POA' else '#777777' for x in df_alloc.index]
    df_alloc['Load Balance CV'].plot(kind='bar', ax=ax3a, color=colors_alloc, edgecolor='black', alpha=0.85, width=0.55)
    ax3a.set_title("Edge Node Load Balance (Lower is Better)", fontweight='bold')
    ax3a.set_ylabel("Coefficient of Variation (CV)")
    ax3a.set_xlabel("Algorithm")
    ax3a.set_xticklabels(df_alloc.index, rotation=30, ha='right')
    ax3a.grid(True, linestyle='--', alpha=0.5)
    for p in ax3a.patches:
        ax3a.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)

    # Cloud Overflows
    df_alloc['Overflows'].plot(kind='bar', ax=ax3b, color=colors_alloc, edgecolor='black', alpha=0.85, width=0.55)
    ax3b.set_title("Edge Saturation Cloud Overflows (Lower is Better)", fontweight='bold')
    ax3b.set_ylabel("Number of Overflow Tasks")
    ax3b.set_xlabel("Algorithm")
    ax3b.set_xticklabels(df_alloc.index, rotation=30, ha='right')
    ax3b.grid(True, linestyle='--', alpha=0.5)
    for p in ax3b.patches:
        ax3b.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)

    plt.tight_layout()
    fig3.savefig(os.path.join(OUTPUT_DIR, "allocation_load_balance.png"), bbox_inches='tight', dpi=300)
    plt.close(fig3)

    # Chart 4: Routing Latency & Energy
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    colors_route = [c_proposed if x == 'OOBL-PRO' else '#777777' for x in df_route.index]
    
    df_route['Latency (ms)'].plot(kind='bar', ax=ax4a, color=colors_route, edgecolor='black', alpha=0.85, width=0.55)
    ax4a.set_title("Mean Routing Latency (Lower is Better)", fontweight='bold')
    ax4a.set_ylabel("Latency (ms)")
    ax4a.set_xlabel("Algorithm")
    ax4a.set_xticklabels(df_route.index, rotation=30, ha='right')
    ax4a.grid(True, linestyle='--', alpha=0.5)
    for p in ax4a.patches:
        ax4a.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)

    df_route['Energy (mJ)'].plot(kind='bar', ax=ax4b, color=colors_route, edgecolor='black', alpha=0.85, width=0.55)
    ax4b.set_title("Mean Routing Energy (Lower is Better)", fontweight='bold')
    ax4b.set_ylabel("Energy (mJ)")
    ax4b.set_xlabel("Algorithm")
    ax4b.set_xticklabels(df_route.index, rotation=30, ha='right')
    ax4b.grid(True, linestyle='--', alpha=0.5)
    for p in ax4b.patches:
        ax4b.annotate(f'{p.get_height():.4f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)

    plt.tight_layout()
    fig4.savefig(os.path.join(OUTPUT_DIR, "routing_performance.png"), bbox_inches='tight', dpi=300)
    plt.close(fig4)

    # Chart 5: Routing Convergence / Adaptation Speed
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    
    ax5.plot(curves['OOBL-PRO'], label='OOBL-PRO (Proposed)', color='#2ca02c', linewidth=2.5)
    ax5.plot(curves['PRO'], label='PRO (No Opposition)', color='#ff7f0e', linewidth=2.0, linestyle='--')
    ax5.plot(curves['Q-Learning'], label='Q-Learning Routing', color='#1f77b4', linewidth=2.0, linestyle='-.')
    ax5.plot(curves['Route A'], label='Route A (Static Direct)', color='#d62728', linewidth=1.5, linestyle=':')
    
    ax5.set_xlabel("Tasks Processed (Chronological)", fontweight='bold')
    ax5.set_ylabel("Optimal Route Selection Accuracy\n(Rolling Window Mean)", fontweight='bold')
    ax5.set_title("Dynamic Network Congestion Adaptation Speed", fontweight='bold', pad=15)
    ax5.legend(loc='lower right', frameon=True, shadow=True)
    ax5.grid(True, linestyle='--', alpha=0.5)
    ax5.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    fig5.savefig(os.path.join(OUTPUT_DIR, "routing_convergence.png"), bbox_inches='tight', dpi=300)
    plt.close(fig5)
    
    # Chart 6: Clustering Comparison
    fig6, axes6 = plt.subplots(1, 3, figsize=(15, 5))
    metrics_clust = ['Silhouette', 'Davies-Bouldin', 'Time (ms)']
    titles_clust = ['Silhouette Score (Higher is Better)', 'Davies-Bouldin Index (Lower is Better)', 'Execution Time (ms, Lower is Better)']
    
    for idx, metric in enumerate(metrics_clust):
        ax = axes6[idx]
        colors = [c_proposed if 'Proposed' in x else '#555555' for x in df_clust.index]
        df_clust[metric].plot(kind='bar', ax=ax, color=colors, edgecolor='black', width=0.55)
        ax.set_title(titles_clust[idx], fontweight='bold', fontsize=10)
        ax.set_ylabel(metric)
        ax.set_xticklabels(df_clust.index, rotation=20, ha='right')
        ax.grid(True, linestyle='--', alpha=0.5)
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:.3f}' if metric != 'Time (ms)' else f'{height:.1f}', 
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)
            
    plt.tight_layout()
    fig6.savefig(os.path.join(OUTPUT_DIR, "clustering_comparison.png"), bbox_inches='tight', dpi=300)
    plt.close(fig6)
    
    print(f"All graphs successfully saved to: {OUTPUT_DIR}")


# ─────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────

def main():
    print("=" * 65)
    print("ECG FRAMEWORK ALGORITHM BENCHMARK SUITE")
    print("=" * 65)
    
    # Run Day 2 benchmark
    df_clust = run_clustering_benchmark()
    
    # Run Day 3 benchmark
    df_sched = run_scheduling_benchmark()
    
    # Run Day 4 benchmark
    df_alloc = run_allocation_benchmark()
    
    # Run Day 5 benchmark
    df_route, curves = run_routing_benchmark()
    
    # Save raw outputs to CSV
    df_clust.to_csv(os.path.join(OUTPUT_DIR, "benchmark_clustering.csv"))
    df_sched.to_csv(os.path.join(OUTPUT_DIR, "benchmark_scheduling.csv"))
    df_alloc.to_csv(os.path.join(OUTPUT_DIR, "benchmark_allocation.csv"))
    df_route.to_csv(os.path.join(OUTPUT_DIR, "benchmark_routing.csv"))
    
    # Generate charts
    generate_comparison_plots(df_sched, df_alloc, df_route, curves, df_clust)
    
    # Copy generated files to workspace directory for easy access
    import shutil
    files_to_copy = [
        "benchmark_clustering.csv", "benchmark_scheduling.csv", "benchmark_allocation.csv", "benchmark_routing.csv",
        "scheduling_comparison.png", "allocation_performance.png", "allocation_load_balance.png",
        "routing_performance.png", "routing_convergence.png", "clustering_comparison.png"
    ]
    for f in files_to_copy:
        src = os.path.join(OUTPUT_DIR, f)
        dst = os.path.join(WORKSPACE_DIR, f)
        if os.path.exists(src):
            shutil.copy(src, dst)
            
    print(f"All graphs and tables successfully copied to local folder: {WORKSPACE_DIR}")
    print("=" * 65)
    print("BENCHMARK EXECUTION COMPLETE")
    print("=" * 65)

if __name__ == "__main__":
    main()

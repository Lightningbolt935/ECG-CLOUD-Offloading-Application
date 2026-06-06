"""
ECG Edge Node Allocation — Day 4
==================================
QSO-POA: Quokka Swarm Optimization + Puma Optimization Algorithm
for fine-grained real-time task allocation across edge nodes.

INPUT : scheduling_policy.csv    (from Day 3 — which cluster goes to edge)
        task_profiles_clustered.csv (from Day 2 — all task windows with cluster IDs)
OUTPUT: allocation_results.csv   — per-task node assignments
        qsopoa_summary.txt       — thesis-ready summary

WHAT THIS MODULE DOES:
    NSGA-II (Day 3) decided: "All 3 clusters go to edge tier."
    QSO-POA decides: "For each arriving task, which specific edge node
    handles it — edge-central, edge-south, edge-west, or edge-north?"

    This is a real-time, continuous allocation problem. Tasks arrive
    as a stream (simulating IoT sensor output). QSO-POA distributes
    them across 4 edge nodes to:
        1. Minimize total latency
        2. Balance load (prevent any node exceeding 80% capacity)
        3. Minimize energy consumption
        4. Overflow to cloud if ALL edge nodes are saturated

QSO — QUOKKA SWARM OPTIMIZATION:
    Inspired by foraging behavior of quokkas (small Australian marsupials).
    Each quokka = one candidate allocation solution.
    Quokkas explore globally, sharing best-found allocations.
    Handles exploration (finding good regions of solution space).

POA — PUMA OPTIMIZATION ALGORITHM:
    After QSO finds the globally best region, POA refines it locally.
    Inspired by puma hunting strategy: broad territorial exploration
    followed by precise targeting.
    Handles exploitation (fine-tuning within the best region).

TOGETHER:
    QSO (global exploration) → POA (local exploitation)
    This explore-then-exploit pattern is a standard metaheuristic
    design principle that avoids premature convergence to local optima.
"""

import numpy as np
import pandas as pd
import random
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

CLUSTERED_CSV   = "./task_profiles_clustered.csv"
POLICY_CSV      = "./scheduling_policy.csv"
OUTPUT_CSV      = "./allocation_results.csv"
SUMMARY_TXT     = "./qsopoa_summary.txt"

# Edge nodes available for allocation
EDGE_NODES = [
    {'id': 0, 'name': 'edge-central', 'MIPS': 1000, 'RAM_MB': 1000,
     'energy_per_MI': 0.02, 'trans_latency_ms': 1, 'zone': 'Central Chennai'},
    {'id': 1, 'name': 'edge-south',   'MIPS': 1000, 'RAM_MB': 1000,
     'energy_per_MI': 0.02, 'trans_latency_ms': 1, 'zone': 'South Chennai'},
    {'id': 2, 'name': 'edge-west',    'MIPS': 1000, 'RAM_MB': 1000,
     'energy_per_MI': 0.02, 'trans_latency_ms': 1, 'zone': 'West Chennai'},
    {'id': 3, 'name': 'edge-north',   'MIPS': 1000, 'RAM_MB': 1000,
     'energy_per_MI': 0.02, 'trans_latency_ms': 1, 'zone': 'North Chennai'},
]

# Cloud fallback (when ALL edge nodes saturated)
CLOUD_NODE = {
    'id': 99, 'name': 'cloud', 'MIPS': 44800, 'RAM_MB': 40000,
    'energy_per_MI': 0.001, 'trans_latency_ms': 100, 'zone': 'Cloud'
}

N_EDGE_NODES      = len(EDGE_NODES)
SATURATION_THRESH = 0.80   # 80% load = saturated
MI_SCALE_EDGE     = 0.05   # Edge processes 5% of full MI (pre-screening)
MI_SCALE_CLOUD    = 1.00   # Cloud processes 100% of full MI

# QSO parameters
QSO_POP_SIZE   = 30    # number of quokkas
QSO_ITERATIONS = 50    # QSO iterations per batch
QSO_ALPHA      = 0.5   # step size for position update
QSO_BETA       = 1.5   # social learning weight

# POA parameters
POA_ITERATIONS = 30    # POA refinement iterations
POA_STEP_INIT  = 0.3   # initial step size (decreases over iterations)

# Simulation: process tasks in batches (simulating streaming)
BATCH_SIZE = 100       # tasks per allocation batch


# ─────────────────────────────────────────────
# NODE LOAD TRACKER
# Tracks current utilization of each edge node
# ─────────────────────────────────────────────

class NodeLoadTracker:
    """
    Tracks real-time computational load on each edge node.

    Load = sum of MI currently assigned / total node capacity
    When load > SATURATION_THRESH, node is considered saturated.

    In a real deployment this would track actual CPU utilization.
    In simulation we track cumulative MI assignments per time window.
    """
    def __init__(self):
        # Running total of MI assigned per node
        self.mi_assigned  = np.zeros(N_EDGE_NODES)
        # Capacity per node per batch window
        # MIPS * batch_duration_sec = max MI per window
        # batch_duration = BATCH_SIZE tasks * 5sec/task = 500 sec
        self.capacity     = np.array([n['MIPS'] * 500 for n in EDGE_NODES],
                                     dtype=float)
        self.total_assigned = 0

    def get_load(self):
        """Returns load fraction [0,1] for each edge node."""
        return self.mi_assigned / (self.capacity + 1e-9)

    def assign(self, node_id, mi):
        """Record a task assignment to a node."""
        if node_id < N_EDGE_NODES:
            self.mi_assigned[node_id] += mi * MI_SCALE_EDGE
        self.total_assigned += 1

    def reset_batch(self):
        """Reset load counters at start of each batch."""
        self.mi_assigned = np.zeros(N_EDGE_NODES)

    def all_saturated(self):
        """True if ALL edge nodes exceed saturation threshold."""
        return np.all(self.get_load() >= SATURATION_THRESH)

    def available_nodes(self):
        """Returns IDs of non-saturated edge nodes."""
        load = self.get_load()
        return [i for i in range(N_EDGE_NODES)
                if load[i] < SATURATION_THRESH]


# ─────────────────────────────────────────────
# FITNESS FUNCTION
# Evaluates quality of a node allocation decision
# ─────────────────────────────────────────────

def allocation_fitness(node_id, task_mi, task_bw, load_tracker):
    """
    Fitness function for QSO-POA allocation decisions.

    Evaluates assigning a task (with given MI and BW) to a specific node.
    Lower fitness = better allocation.

    FORMULA:
        fitness = w1 * normalized_latency
                + w2 * normalized_energy
                + w3 * load_penalty

    Components:
        normalized_latency : latency of this assignment / max possible latency
        normalized_energy  : energy of this assignment / max possible energy
        load_penalty       : penalty for assigning to an already-loaded node
                             (0 if load < 50%, scales up to 1.0 at saturation)

    Weights (w1=0.5, w2=0.3, w3=0.2):
        Latency is primary objective in medical monitoring (w=0.5).
        Energy is secondary for battery-constrained edge devices (w=0.3).
        Load balancing prevents hotspot failures (w=0.2).

    THESIS DEFENSE:
        "The fitness function directly operationalizes the scheduling
        objectives identified by NSGA-II (latency, energy) with an
        additional load balancing term to ensure QoS across all edge
        nodes in a multi-patient monitoring scenario."
    """
    node  = EDGE_NODES[node_id]
    load  = load_tracker.get_load()

    # Effective MI at edge (pre-screening = 5%)
    eff_mi = task_mi * MI_SCALE_EDGE

    # Latency: execution + transmission
    latency_ms = (eff_mi / node['MIPS']) * 1000 + node['trans_latency_ms']

    # Energy: computation + transmission
    energy_mj  = eff_mi * node['energy_per_MI'] + \
                 task_bw * node['trans_latency_ms'] * 0.0001

    # Load penalty: 0 below 50% load, increases to 1.0 at saturation
    current_load = load[node_id]
    load_penalty = max(0.0, (current_load - 0.5) / 0.5)

    # Normalization constants (empirical from our MI range)
    max_latency = 270.0   # ms (MI=5000 at MIPS=1000: 5000*0.05/1000*1000 + 1)
    max_energy  = 5.0     # mJ

    norm_latency = latency_ms / max_latency
    norm_energy  = energy_mj  / max_energy

    # Weighted fitness (lower = better)
    fitness = 0.5 * norm_latency + 0.3 * norm_energy + 0.2 * load_penalty

    return fitness, latency_ms, energy_mj


# ─────────────────────────────────────────────
# QSO — QUOKKA SWARM OPTIMIZATION
# Global exploration across edge nodes
# ─────────────────────────────────────────────

def qso_allocate(task_mi, task_bw, load_tracker, available_nodes):
    """
    QSO global exploration to find best edge node for one task.

    ALGORITHM:
        1. Initialize population: each quokka = one candidate node ID
        2. For each iteration:
            a. Evaluate fitness of each quokka's position
            b. Update best-known position (global best = gbest)
            c. Update each quokka's position toward gbest with randomness
        3. Return gbest (best node found)

    POSITION UPDATE:
        In continuous optimization, positions are real-valued and updated as:
            pos_new = pos_old + alpha*(gbest - pos_old) + beta*rand*perturbation

        For our discrete problem (node IDs are integers), we use a
        probability-based selection: quokkas vote for nodes, the
        probability of each node being selected is proportional to
        how many quokkas currently occupy it (social learning).

    WHY QSO SPECIFICALLY:
        QSO was proposed for continuous optimization but adapted here
        for discrete node selection. Its key advantage over random
        search is the social learning component — quokkas that have
        found good nodes 'attract' other quokkas, focusing the search
        on promising regions while maintaining diversity through
        individual perturbation.

    Args:
        task_mi       : task's mean MI value
        task_bw       : task's mean bandwidth
        load_tracker  : current node load state
        available_nodes: list of non-saturated node IDs

    Returns:
        best_node_id  : node to assign this task to
        best_fitness  : fitness value of this assignment
    """
    if not available_nodes:
        return None, float('inf')

    if len(available_nodes) == 1:
        nid = available_nodes[0]
        fit, lat, eng = allocation_fitness(nid, task_mi, task_bw, load_tracker)
        return nid, fit

    # Initialize quokka positions (random node IDs from available nodes)
    pop = [random.choice(available_nodes) for _ in range(QSO_POP_SIZE)]

    # Evaluate initial fitness
    fitness_vals = []
    for nid in pop:
        f, _, _ = allocation_fitness(nid, task_mi, task_bw, load_tracker)
        fitness_vals.append(f)

    # Global best
    gbest_idx = np.argmin(fitness_vals)
    gbest     = pop[gbest_idx]
    gbest_fit = fitness_vals[gbest_idx]

    # QSO iterations
    for iteration in range(QSO_ITERATIONS):
        # Social learning: build probability distribution over nodes
        # Nodes with more quokkas and better fitness get higher probability
        node_scores = {nid: 0.0 for nid in available_nodes}
        for i, nid in enumerate(pop):
            # Invert fitness so lower fitness = higher score
            node_scores[nid] += 1.0 / (fitness_vals[i] + 1e-6)

        total_score = sum(node_scores.values()) + 1e-9
        probs = [node_scores[nid] / total_score for nid in available_nodes]

        # Update each quokka's position
        new_pop = []
        new_fitness = []
        for i in range(QSO_POP_SIZE):
            # With probability alpha: move toward gbest
            # With probability (1-alpha): explore randomly
            if random.random() < QSO_ALPHA:
                # Social learning: sample from probability distribution
                new_node = random.choices(available_nodes, weights=probs, k=1)[0]
            else:
                # Individual exploration: random node
                new_node = random.choice(available_nodes)

            f, _, _ = allocation_fitness(new_node, task_mi, task_bw, load_tracker)
            new_pop.append(new_node)
            new_fitness.append(f)

            # Update global best
            if f < gbest_fit:
                gbest     = new_node
                gbest_fit = f

        pop          = new_pop
        fitness_vals = new_fitness

    return gbest, gbest_fit


# ─────────────────────────────────────────────
# POA — PUMA OPTIMIZATION ALGORITHM
# Local refinement around QSO's best solution
# ─────────────────────────────────────────────

def poa_refine(qso_best_node, task_mi, task_bw, load_tracker, available_nodes):
    """
    POA local exploitation — refines QSO's result.

    ALGORITHM:
        1. Start from QSO's best node (gbest)
        2. In each iteration, evaluate all neighbors of current best
           (neighbors = all other available nodes)
        3. If any neighbor has better fitness, move there
        4. Decrease step size (simulated annealing-like cooling)
        5. Return final best node

    BIOLOGICAL ANALOGY:
        A puma first identifies its hunting territory broadly (QSO),
        then precisely stalks its prey within that territory (POA).
        The decreasing step size models the puma narrowing its focus
        as it approaches the target.

    WHY POA AFTER QSO:
        QSO may converge to the correct node ID but not be sure —
        there may be subtle load differences between similar nodes.
        POA's exhaustive local comparison ensures we pick the
        definitively best option among the remaining candidates.

    For our discrete problem with only 4 edge nodes, POA simplifies to:
        Re-evaluate all available nodes with current load state and
        pick the minimum fitness — this is correct because POA's
        'local neighborhood' for discrete node selection IS all
        available nodes.

    Args:
        qso_best_node : node ID returned by QSO
        task_mi, task_bw, load_tracker, available_nodes : same as QSO

    Returns:
        best_node_id, best_fitness, latency_ms, energy_mj
    """
    if not available_nodes:
        return qso_best_node, float('inf'), 0, 0

    best_node    = qso_best_node
    best_fitness, best_lat, best_eng = allocation_fitness(
        best_node, task_mi, task_bw, load_tracker
    )

    step = POA_STEP_INIT

    for iteration in range(POA_ITERATIONS):
        improved = False

        # Evaluate all available nodes (neighborhood search)
        for nid in available_nodes:
            if nid == best_node:
                continue

            f, lat, eng = allocation_fitness(nid, task_mi, task_bw, load_tracker)

            # Accept if better by more than step threshold
            # (step decreases each iteration = increasingly selective)
            if f < best_fitness - step * 0.01:
                best_node    = nid
                best_fitness = f
                best_lat     = lat
                best_eng     = eng
                improved     = True

        # Decrease step size (puma narrows focus)
        step *= 0.9

        if not improved:
            break  # converged

    return best_node, best_fitness, best_lat, best_eng


# ─────────────────────────────────────────────
# MAIN ALLOCATION FUNCTION
# Combines QSO + POA for each task
# ─────────────────────────────────────────────

def allocate_task(task_mi, task_bw, task_class, load_tracker):
    """
    Allocate one task to a node using QSO-POA pipeline.

    DECISION FLOW:
        1. Check if task is Class 3 (Critical) → cloud directly
        2. Check if all edge nodes saturated → cloud (overflow)
        3. Run QSO to find best edge node globally
        4. Run POA to refine QSO's result locally
        5. Assign to best node, update load tracker

    Returns dict with allocation decision and performance metrics.
    """
    # Step 1: Critical task → cloud immediately
    if task_class == 3:
        eff_mi  = task_mi * MI_SCALE_CLOUD
        lat_ms  = (eff_mi / CLOUD_NODE['MIPS']) * 1000 + \
                   CLOUD_NODE['trans_latency_ms']
        eng_mj  = eff_mi * CLOUD_NODE['energy_per_MI']
        return {
            'node_id':   CLOUD_NODE['id'],
            'node_name': CLOUD_NODE['name'],
            'reason':    'critical_bypass',
            'latency_ms': round(lat_ms, 4),
            'energy_mj':  round(eng_mj, 6),
            'load_before': list(load_tracker.get_load().round(3))
        }

    # Step 2: All edge saturated → cloud overflow
    if load_tracker.all_saturated():
        eff_mi  = task_mi * MI_SCALE_CLOUD
        lat_ms  = (eff_mi / CLOUD_NODE['MIPS']) * 1000 + \
                   CLOUD_NODE['trans_latency_ms']
        eng_mj  = eff_mi * CLOUD_NODE['energy_per_MI']
        return {
            'node_id':   CLOUD_NODE['id'],
            'node_name': CLOUD_NODE['name'],
            'reason':    'edge_saturated_overflow',
            'latency_ms': round(lat_ms, 4),
            'energy_mj':  round(eng_mj, 6),
            'load_before': list(load_tracker.get_load().round(3))
        }

    # Step 3: Get available (non-saturated) edge nodes
    available = load_tracker.available_nodes()

    # Step 4: QSO global exploration
    qso_best, qso_fitness = qso_allocate(
        task_mi, task_bw, load_tracker, available
    )

    # Step 5: POA local refinement
    poa_best, poa_fitness, lat_ms, eng_mj = poa_refine(
        qso_best, task_mi, task_bw, load_tracker, available
    )

    # Record load before assignment
    load_before = list(load_tracker.get_load().round(3))

    # Update load tracker
    load_tracker.assign(poa_best, task_mi)

    node = EDGE_NODES[poa_best]
    return {
        'node_id':    node['id'],
        'node_name':  node['name'],
        'reason':     'qsopoa_allocated',
        'latency_ms': round(lat_ms, 4),
        'energy_mj':  round(eng_mj, 6),
        'load_before': load_before
    }


# ─────────────────────────────────────────────
# BATCH PROCESSING SIMULATION
# Simulates streaming task arrival
# ─────────────────────────────────────────────

def run_allocation():
    print("=" * 65)
    print("ECG Edge Node Allocation — Day 4")
    print("QSO-POA: Quokka Swarm + Puma Optimization")
    print("=" * 65)

    # Load clustered task profiles
    print("\n[LOAD] Reading clustered task profiles...")
    df = pd.read_csv(CLUSTERED_CSV)
    print(f"  Loaded {len(df)} tasks across {df['cluster_id'].nunique()} clusters")

    # Only process tasks that NSGA-II assigned to edge
    # (Class 3 critical tasks bypass to cloud directly)
    print(f"  Tasks by class: " +
          ", ".join([f"Class{c}={len(df[df['task_class']==c])}"
                     for c in range(4)]))

    # Initialize load tracker
    load_tracker = NodeLoadTracker()

    # Process tasks in batches
    results = []
    n_batches = len(df) // BATCH_SIZE + 1

    print(f"\n[ALLOCATION] Processing {len(df)} tasks in "
          f"{n_batches} batches of {BATCH_SIZE}...")
    print(f"  Saturation threshold: {SATURATION_THRESH*100:.0f}%")
    print(f"  QSO: {QSO_POP_SIZE} quokkas, {QSO_ITERATIONS} iterations")
    print(f"  POA: {POA_ITERATIONS} refinement iterations")
    print()

    edge_count   = 0
    cloud_forced = 0
    cloud_overflow = 0

    for batch_num in range(n_batches):
        start = batch_num * BATCH_SIZE
        end   = min(start + BATCH_SIZE, len(df))
        batch = df.iloc[start:end]

        if len(batch) == 0:
            break

        # Reset load tracker at start of each batch
        # (represents a new time window)
        load_tracker.reset_batch()

        for _, task in batch.iterrows():
            alloc = allocate_task(
                task_mi    = task['MI'],
                task_bw    = task['BW_kbps'],
                task_class = task['task_class'],
                load_tracker = load_tracker
            )

            result = {
                'record_id':     task['record_id'],
                'window_id':     task['window_id'],
                'cluster_id':    task['cluster_id'],
                'task_class':    task['task_class'],
                'MI':            task['MI'],
                'composite':     task['composite_score'],
                'assigned_node': alloc['node_name'],
                'reason':        alloc['reason'],
                'latency_ms':    alloc['latency_ms'],
                'energy_mj':     alloc['energy_mj'],
            }
            results.append(result)

            if alloc['reason'] == 'qsopoa_allocated':
                edge_count += 1
            elif alloc['reason'] == 'critical_bypass':
                cloud_forced += 1
            else:
                cloud_overflow += 1

        if (batch_num + 1) % 20 == 0 or batch_num == 0:
            print(f"  Batch {batch_num+1:>4}/{n_batches} — "
                  f"edge={edge_count}, "
                  f"cloud_forced={cloud_forced}, "
                  f"cloud_overflow={cloud_overflow}")

    # Build results DataFrame
    results_df = pd.DataFrame(results)

    # ── Summary Statistics ────────────────────────────────────────
    print("\n" + "=" * 65)
    print("ALLOCATION COMPLETE — Summary")
    print("=" * 65)

    total = len(results_df)
    print(f"\nTotal tasks allocated : {total}")
    print(f"  Edge (QSO-POA)      : {edge_count} ({100*edge_count/total:.1f}%)")
    print(f"  Cloud (Critical)    : {cloud_forced} ({100*cloud_forced/total:.1f}%)")
    print(f"  Cloud (Overflow)    : {cloud_overflow} ({100*cloud_overflow/total:.1f}%)")

    print("\nEdge Node Load Distribution:")
    node_counts = results_df[results_df['reason']=='qsopoa_allocated']\
                  ['assigned_node'].value_counts()
    for node in EDGE_NODES:
        count = node_counts.get(node['name'], 0)
        pct   = 100 * count / max(edge_count, 1)
        bar   = '█' * int(pct / 5)
        print(f"  {node['name']:<16}: {count:>5} tasks ({pct:>5.1f}%) {bar}")

    print("\nPerformance Metrics (edge-allocated tasks):")
    edge_tasks = results_df[results_df['reason'] == 'qsopoa_allocated']
    if len(edge_tasks) > 0:
        print(f"  Mean latency    : {edge_tasks['latency_ms'].mean():.4f} ms")
        print(f"  Mean energy     : {edge_tasks['energy_mj'].mean():.6f} mJ")
        print(f"  Max latency     : {edge_tasks['latency_ms'].max():.4f} ms")
        print(f"  Min latency     : {edge_tasks['latency_ms'].min():.4f} ms")

    # Load balance metric (Coefficient of Variation of node assignments)
    if edge_count > 0:
        counts = [node_counts.get(n['name'], 0) for n in EDGE_NODES]
        cv = np.std(counts) / (np.mean(counts) + 1e-9)
        print(f"\nLoad Balance (CV of task counts): {cv:.4f}")
        print(f"  (0=perfect balance, 1=all on one node)")
        balance = "Excellent" if cv < 0.1 else \
                  "Good"      if cv < 0.2 else \
                  "Moderate"  if cv < 0.4 else "Poor"
        print(f"  Assessment: {balance}")
    else:
        cv = 0

    # Save outputs
    results_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')

    summary_lines = [
        "=" * 65,
        "QSO-POA ALLOCATION RESULTS — For Thesis",
        "=" * 65,
        "",
        f"Algorithm          : QSO (Quokka Swarm) + POA (Puma Optimization)",
        f"QSO Population     : {QSO_POP_SIZE} quokkas",
        f"QSO Iterations     : {QSO_ITERATIONS} per task",
        f"POA Iterations     : {POA_ITERATIONS} per task",
        f"Saturation thresh  : {SATURATION_THRESH*100:.0f}%",
        f"Overflow policy    : Direct to cloud",
        "",
        "Allocation Results:",
        f"  Total tasks      : {total}",
        f"  Edge (QSO-POA)   : {edge_count} ({100*edge_count/total:.1f}%)",
        f"  Cloud (Critical) : {cloud_forced} ({100*cloud_forced/total:.1f}%)",
        f"  Cloud (Overflow) : {cloud_overflow} ({100*cloud_overflow/total:.1f}%)",
        "",
        "Load Balance:",
        f"  CV of assignments: {cv:.4f}",
        f"  Assessment       : {balance if edge_count > 0 else 'N/A'}",
    ]
    if len(edge_tasks) > 0:
        summary_lines += [
            "",
            "Edge Performance:",
            f"  Mean latency : {edge_tasks['latency_ms'].mean():.4f} ms",
            f"  Mean energy  : {edge_tasks['energy_mj'].mean():.6f} mJ",
        ]

    with open(SUMMARY_TXT, 'w', encoding='utf-8') as f:
        f.write("\n".join(summary_lines))

    print(f"\n[OUTPUTS]")
    print(f"  {OUTPUT_CSV}   — per-task allocation decisions")
    print(f"  {SUMMARY_TXT}  — thesis-ready summary")
    print("\n[NEXT STEP] Run oobl_pro_day5.py for cloud offloading routing")
    print("=" * 65)

    return results_df


if __name__ == "__main__":
    run_allocation()
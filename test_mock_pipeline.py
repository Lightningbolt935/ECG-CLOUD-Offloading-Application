import numpy as np
import pandas as pd
import wfdb
import os
import sys
import warnings
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks, butter, filtfilt

warnings.filterwarnings('ignore')

# Configurations from pipeline v4
DATA_DIR        = "./mitdb"
FEATURE_COLS = [
    'composite_score', 'sample_entropy', 'qrs_complexity',
    'variance_score', 'st_deviation', 'MI', 'RAM_MB', 'BW_kbps'
]
COMPOSITE_T1    = 0.2747
COMPOSITE_T2    = 0.4172
ST_THRESHOLD_MV = 0.3105
SAMPLING_RATE   = 360
WINDOW_SIZE     = 5 * SAMPLING_RATE  # 1800 samples

# Day 3 Node configurations for scheduler lookup
NODES_SPECS = {
    6: 'edge-central (Edge)',
    7: 'edge-south (Edge)',
    8: 'edge-west (Edge)',
    9: 'edge-north (Edge)',
    1: 'fog-central (Fog)',
    0: 'cloud (Cloud)'
}

# ─────────────────────────────────────────────
# DAY 1 PREPROCESSING AND METRICS
# ─────────────────────────────────────────────

def bandpass_filter(signal_mv, fs=360, lowcut=0.5, highcut=40.0):
    nyq  = 0.5 * fs
    low  = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.999)
    if low >= high:
        return signal_mv
    try:
        b, a = butter(4, [low, high], btype='band')
        return filtfilt(b, a, signal_mv)
    except Exception:
        return signal_mv

def sample_entropy(signal_mv, m=2, r_factor=0.2):
    sig = signal_mv[:300].astype(float)
    s_min, s_max = np.min(sig), np.max(sig)
    if s_max == s_min:
        return 0.0
    sig = (sig - s_min) / (s_max - s_min)
    N = len(sig)
    r = r_factor * np.std(sig)
    if r == 0:
        return 0.0
    def count_matches(m_len):
        count = 0
        for i in range(N - m_len):
            tmpl = sig[i:i + m_len]
            for j in range(i + 1, N - m_len):
                if np.max(np.abs(sig[j:j + m_len] - tmpl)) < r:
                    count += 1
        return count
    A = count_matches(m + 1)
    B = count_matches(m)
    if B == 0 or A == 0:
        return 0.0
    return float(-np.log(A / B))

def qrs_complexity_score(signal_mv, fs=360):
    filtered = bandpass_filter(signal_mv, fs)
    peaks, _ = find_peaks(filtered, distance=int(0.15*fs), prominence=0.3)
    if len(peaks) < 3:
        return 0.3
    rr = np.diff(peaks).astype(float)
    rr = rr[(rr >= 72) & (rr <= 1800)]
    if len(rr) < 2:
        return 0.3
    cv = np.std(rr) / (np.mean(rr) + 1e-6)
    return float(min(cv / 0.15, 1.0))

def variance_score(signal_mv):
    s_min, s_max = np.min(signal_mv), np.max(signal_mv)
    if s_max == s_min:
        return 0.0
    norm = (signal_mv - s_min) / (s_max - s_min)
    return float(min(np.var(norm) / 0.08, 1.0))

def st_deviation_score(signal_mv, fs=360):
    filtered = bandpass_filter(signal_mv, fs)
    peaks, _ = find_peaks(filtered, distance=int(0.15*fs), prominence=0.3)
    if len(peaks) == 0:
        return 0.0, 0.0
    st_devs = []
    for peak in peaks:
        pr_start = peak - int(0.16 * fs)
        pr_end   = peak - int(0.12 * fs)
        st_point = peak + int(0.14 * fs)
        if pr_start < 0 or pr_end <= pr_start or st_point >= len(filtered):
            continue
        baseline = np.mean(filtered[pr_start:pr_end])
        st_devs.append(abs(filtered[st_point] - baseline))
    if not st_devs:
        return 0.0, 0.0
    mean_dev = float(np.mean(st_devs))
    score    = float(min(mean_dev / 0.5, 1.0))
    return score, mean_dev

def compute_complexity_score(window_mv, fs=360):
    se_score         = float(min(sample_entropy(window_mv) / 2.0, 1.0))
    qrs_score        = float(qrs_complexity_score(window_mv, fs))
    var_score        = float(variance_score(window_mv))
    st_score, st_raw = st_deviation_score(window_mv, fs)
    composite = (0.30 * se_score  +
                 0.25 * qrs_score +
                 0.25 * var_score +
                 0.20 * st_score)
    return {
        'sample_entropy':  round(se_score,         4),
        'qrs_complexity':  round(qrs_score,        4),
        'variance_score':  round(var_score,         4),
        'st_deviation':    round(st_score,          4),
        'st_deviation_mv': round(st_raw,            4),
        'composite_score': round(float(composite),  4)
    }

def assign_task_parameters(complexity_score, st_deviation_mv):
    if st_deviation_mv >= ST_THRESHOLD_MV:
        return {
            'MI': 5000, 'RAM_MB': 50, 'BW_kbps': 500,
            'task_class': 3, 'priority': 'CRITICAL',
            'offload_target': 'CLOUD_FORCED'
        }
    if complexity_score < COMPOSITE_T1:
        return {
            'MI': int(800 + complexity_score * 1466),
            'RAM_MB': 10, 'BW_kbps': 100,
            'task_class': 0, 'priority': 'LOW',
            'offload_target': 'EDGE'
        }
    elif complexity_score < COMPOSITE_T2:
        return {
            'MI': int(1500 + (complexity_score - COMPOSITE_T1) * 5000),
            'RAM_MB': 25, 'BW_kbps': 250,
            'task_class': 1, 'priority': 'MEDIUM',
            'offload_target': 'EDGE_OR_FOG'
        }
    else:
        return {
            'MI': int(3000 + (complexity_score - COMPOSITE_T2) * 5000),
            'RAM_MB': 40, 'BW_kbps': 400,
            'task_class': 2, 'priority': 'HIGH',
            'offload_target': 'CLOUD_PREFERRED'
        }

# ─────────────────────────────────────────────
# RUN TESTING PIPELINE FLOW
# ─────────────────────────────────────────────

def run_mock_test(record_id='100', window_idx=10):
    print("=" * 70)
    print(f"ECG CLOUD OFFLOADING MOCK PIPELINE TEST TRACE")
    print(f"Loading Record: {record_id} | Window Index: {window_idx}")
    print("=" * 70)
    
    # ── STEP 1: Load signal ────────────────────
    record_path = os.path.join(DATA_DIR, str(record_id))
    if not os.path.exists(record_path + ".dat"):
        print(f"[ERROR] Record file not found at: {record_path}")
        sys.exit(1)
        
    record = wfdb.rdrecord(record_path)
    signal_mv = record.p_signal[:, 0]
    fs = record.fs
    
    start_sample = window_idx * WINDOW_SIZE
    end_sample = start_sample + WINDOW_SIZE
    
    if end_sample >= len(signal_mv):
        print(f"[ERROR] Window index {window_idx} is out of bounds for signal length.")
        sys.exit(1)
        
    window = signal_mv[start_sample:end_sample]
    print(f"\n[STAGE 1] ECG Window Capture (Client Wearable)")
    print(f"  - Loaded samples range : {start_sample} to {end_sample} ({WINDOW_SIZE} samples)")
    print(f"  - Window duration      : 5.0 seconds")
    print(f"  - Raw signal limits    : {window.min():.3f} mV to {window.max():.3f} mV")
    print(f"  - Package size         : 3.6 KB packet (16-bit encoding)")
    
    # ── STEP 2: Feature Extraction & Class Assignment ──
    print(f"\n[STAGE 2] Complexity Profiling (Day 1 Pipeline)")
    scores = compute_complexity_score(window, fs)
    params = assign_task_parameters(scores['composite_score'], scores['st_deviation_mv'])
    
    print(f"  - Sample Entropy Score : {scores['sample_entropy']:.4f}")
    print(f"  - QRS Complexity Score : {scores['qrs_complexity']:.4f}")
    print(f"  - Variance Score       : {scores['variance_score']:.4f}")
    print(f"  - ST segment dev (mV)  : {scores['st_deviation_mv']:.4f} mV")
    print(f"  - Composite Score      : {scores['composite_score']:.4f}")
    print(f"  - Assigned Class       : Class {params['task_class']} ({params['priority']} priority)")
    print(f"  - Task Requirements    : MI={params['MI']}, RAM={params['RAM_MB']}MB, BW={params['BW_kbps']}kbps")
    print(f"  - Profiling Action     : {params['offload_target']}")
    
    # ── STEP 3: Task Clustering (Day 2 Pipeline) ──
    print(f"\n[STAGE 3] Incremental K-Means++ Clustering (Day 2)")
    # Load profile scaler
    if not os.path.exists("./task_profiles.csv") or not os.path.exists("./cluster_centroids.csv"):
        print("[ERROR] Day 1 or Day 2 outputs missing from workspace.")
        sys.exit(1)
        
    df_profiles = pd.read_csv("./task_profiles.csv")
    scaler = StandardScaler()
    scaler.fit(df_profiles[FEATURE_COLS].values)
    
    centroids_df = pd.read_csv("./cluster_centroids.csv")
    centroids_scaled = scaler.transform(centroids_df[FEATURE_COLS].values)
    
    # Construct task features
    task_features = [
        scores['composite_score'], scores['sample_entropy'], scores['qrs_complexity'],
        scores['variance_score'], scores['st_deviation'], params['MI'], params['RAM_MB'], params['BW_kbps']
    ]
    
    task_scaled = scaler.transform([task_features])[0]
    distances = [np.linalg.norm(task_scaled - c) for c in centroids_scaled]
    cluster_id = np.argmin(distances)
    
    print(f"  - Distances to centroids: " + ", ".join([f"Cluster {c}: {d:.3f}" for c, d in enumerate(distances)]))
    print(f"  - Assigned Cluster ID  : Cluster {cluster_id}")
    
    # ── STEP 4: Objective Scheduling (Day 3 Policy Lookup) ──
    print(f"\n[STAGE 4] Tier-Level Scheduling (Day 3 NSGA-II Policy)")
    if not os.path.exists("./scheduling_policy.csv"):
        print("[ERROR] Scheduling policy csv missing.")
        sys.exit(1)
        
    policy_df = pd.read_csv("./scheduling_policy.csv")
    sched_row = policy_df[policy_df['cluster_id'] == cluster_id]
    if sched_row.empty:
        # Default fallback
        sched_node_id = 7
        sched_node_name = "edge-south"
        sched_tier = "edge"
    else:
        sched_node_id = int(sched_row['node_id'].values[0])
        sched_node_name = sched_row['node_name'].values[0]
        sched_tier = sched_row['node_type'].values[0]
        
    print(f"  - Policy decision : Map Cluster {cluster_id} to tier [{sched_tier.upper()}]")
    print(f"  - Default Target  : {NODES_SPECS.get(sched_node_id, 'Unknown')}")
    
    # ── STEP 5: Real-time Swarm Allocation (Day 4 QSO-POA Broker) ──
    print(f"\n[STAGE 5] Edge Swarm Allocation (Day 4 QSO-POA)")
    # Set simulated load
    node_names = ['edge-central', 'edge-south', 'edge-west', 'edge-north']
    np.random.seed(42)
    loads = np.random.uniform(0.1, 0.75, size=4)
    print(f"  - Current edge node loads: " + ", ".join([f"{node_names[i]}: {loads[i]*100:.1f}%" for i in range(4)]))
    
    # Swarm Decision
    # If Class 3 -> cloud directly
    if params['task_class'] == 3:
        allocated_node = "cloud"
        alloc_reason = "critical_bypass"
        alloc_latency = (params['MI'] * 1.00 / 44800) * 1000 + 100.0
    else:
        # Available edge nodes
        available = [i for i in range(4) if loads[i] < 0.80]
        if not available:
            allocated_node = "cloud"
            alloc_reason = "edge_saturated_overflow"
            alloc_latency = (params['MI'] * 1.00 / 44800) * 1000 + 100.0
        else:
            # Swarm selects node with minimum load under equal parameters
            best_idx = available[np.argmin(loads[available])]
            allocated_node = node_names[best_idx]
            alloc_reason = "qsopoa_allocated"
            eff_mi = params['MI'] * 0.05
            alloc_latency = (eff_mi / 1000.0) * 1000.0 + 1.0
            
    print(f"  - Allocation Decision: {allocated_node}")
    print(f"  - Allocation Reason  : {alloc_reason}")
    print(f"  - Allocation Latency : {alloc_latency:.2f} ms")
    
    # ── STEP 6: Network Routing (Day 5 OOBL-PRO Router) ──
    if "cloud" in allocated_node:
        print(f"\n[STAGE 6] Dynamic Congestion Cloud Routing (Day 5 OOBL-PRO)")
        # Simulate Chennai route delays under congestion
        # Route A: edge -> proxy -> cloud
        # Route B: edge -> fog -> proxy -> cloud
        task_bw = params['MI'] * 0.1
        lat_A = 15.0 + 100.0 + (task_bw / 5000.0) * 1000.0 + (task_bw / 10000.0) * 1000.0 + (params['MI'] / 44800.0) * 1000.0
        lat_B = 3.0 + 15.0 + 100.0 + (task_bw / 1000.0) * 1000.0 + (task_bw / 5000.0) * 1000.0 + (task_bw / 10000.0) * 1000.0 + (params['MI'] / 44800.0) * 1000.0
        
        # OOBL selects lower-latency path
        selected_route = "Route A (Direct)" if lat_A <= lat_B else "Route B (via Fog)"
        routing_lat = min(lat_A, lat_B)
        
        print(f"  - Route A calculated latency: {lat_A:.2f} ms")
        print(f"  - Route B calculated latency: {lat_B:.2f} ms")
        print(f"  - Selected Network Route    : {selected_route}")
        print(f"  - Total Routing Latency     : {routing_lat:.2f} ms")
    else:
        print(f"\n[STAGE 6] Routing Bypassed")
        print(f"  - Task processed locally at the edge. No WAN cloud routing needed.")
        
    print("\n" + "=" * 70)
    print(f"MOCK TEST COMPLETE — Diagnostic offloaded successfully!")
    print("=" * 70)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Mock ECG offloading pipeline test')
    parser.path_args = parser.add_argument('--record', type=str, default='100', help='MIT-BIH Record ID (e.g. 100, 207)')
    parser.path_args = parser.add_argument('--window', type=int, default=10, help='Window index (default: 10)')
    args = parser.parse_args()
    
    run_mock_test(args.record, args.window)

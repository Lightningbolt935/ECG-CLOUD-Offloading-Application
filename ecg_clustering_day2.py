"""
ECG Task Clustering — Day 2
=============================
Incremental K-Means++ on 8-dimensional feature vector.

INPUT : task_profiles.csv  (from Day 1 v4)
OUTPUT: task_profiles_clustered.csv
        cluster_centroids.csv
        cluster_summary.txt   (paste into thesis)

FEATURE VECTOR (8 dimensions):
    Signal complexity:  composite_score, sample_entropy,
                        qrs_complexity, variance_score, st_deviation
    Task parameters:    MI, RAM_MB, BW_kbps

WHY INCREMENTAL (MiniBatchKMeans):
    ECG data arrives as a continuous stream from wearable devices.
    Standard K-Means requires all data upfront — infeasible for
    real-time edge deployment. MiniBatchKMeans processes data in
    batches, updating centroids incrementally via partial_fit().
    This simulates the streaming behavior of the real system.

WHY K-MEANS++ INITIALIZATION:
    Random centroid initialization can converge to poor local optima.
    K-Means++ selects initial centroids with probability proportional
    to squared distance from existing centroids, giving O(log k)
    approximation guarantee (Arthur & Vassilvitskii, 2007).

CHOOSING K:
    We evaluate k=2 to k=8 using:
    1. Silhouette Score   — measures cluster cohesion vs separation
    2. Davies-Bouldin Index — lower is better (compact, separated clusters)
    3. Elbow method (inertia) — find the "elbow" in distortion curve
    The optimal k is selected where silhouette is maximized.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

INPUT_CSV    = "./task_profiles.csv"
OUTPUT_CSV   = "./task_profiles_clustered.csv"
CENTROIDS_CSV = "./cluster_centroids.csv"
SUMMARY_TXT  = "./cluster_summary.txt"

# Features for clustering (8-dimensional)
FEATURE_COLS = [
    'composite_score',
    'sample_entropy',
    'qrs_complexity',
    'variance_score',
    'st_deviation',
    'MI',
    'RAM_MB',
    'BW_kbps'
]

BATCH_SIZE = 500   # simulate streaming: 500 windows per batch
                   # at 5sec/window this is ~41 minutes of ECG per batch
                   # realistic for an edge node processing multiple patients


# ─────────────────────────────────────────────
# STEP 1 — LOAD AND SCALE
# ─────────────────────────────────────────────

def load_and_scale(csv_path):
    """
    Load task profiles and standardize features.

    WHY STANDARDIZE?
        MI ranges 800-5000, composite_score ranges 0-1.
        Without scaling, MI would dominate the distance calculation
        purely due to its larger numerical range, not because it's
        more important. StandardScaler gives each feature zero mean
        and unit variance, so all 8 features contribute equally
        to cluster distance.

    THESIS NOTE:
        "Feature vectors were standardized using z-score normalization
        prior to clustering to prevent high-magnitude task parameters
        (MI: 800–5000) from dominating the Euclidean distance metric
        over normalized complexity scores (range: 0–1)."
    """
    df = pd.read_csv(csv_path)

    # Verify all feature columns exist
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[FEATURE_COLS].values.astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X_scaled, scaler


# ─────────────────────────────────────────────
# STEP 2 — FIND OPTIMAL K
# ─────────────────────────────────────────────

def find_optimal_k(X_scaled, k_range=range(2, 9)):
    """
    Evaluate k=2..8 using Silhouette Score and Davies-Bouldin Index.

    SILHOUETTE SCORE:
        For each point i:
            a(i) = mean distance to points in same cluster
            b(i) = mean distance to points in nearest other cluster
            s(i) = (b(i) - a(i)) / max(a(i), b(i))
        Score = mean s(i) over all points. Range: [-1, 1].
        Higher is better. >0.5 = good structure.

    DAVIES-BOULDIN INDEX:
        For each cluster i:
            R(i,j) = (scatter_i + scatter_j) / distance(centroid_i, centroid_j)
        DB = mean of max R(i,j) over all i.
        Lower is better. Measures compactness relative to separation.

    ELBOW (INERTIA):
        Sum of squared distances to cluster centroids.
        Look for the "elbow" — point where adding more clusters
        gives diminishing returns.

    We use Silhouette as primary criterion, DB as confirmation.
    """
    print("\n[STEP 2] Finding optimal k...")
    print(f"  {'k':>4} {'Silhouette':>12} {'Davies-Bouldin':>16} {'Inertia':>12}")
    print("  " + "-" * 46)

    results = []

    # Use a subsample for speed (silhouette is O(N²))
    # 3000 samples is sufficient for stable estimates
    n_sample = min(3000, len(X_scaled))
    idx      = np.random.choice(len(X_scaled), n_sample, replace=False)
    X_sample = X_scaled[idx]

    for k in k_range:
        model = MiniBatchKMeans(
            n_clusters=k,
            init='k-means++',
            n_init=10,
            batch_size=BATCH_SIZE,
            random_state=42
        )
        labels   = model.fit_predict(X_sample)
        inertia  = model.inertia_

        # Need at least 2 clusters with >1 sample for these metrics
        if len(np.unique(labels)) < 2:
            continue

        sil = silhouette_score(X_sample, labels, sample_size=min(1000, n_sample))
        db  = davies_bouldin_score(X_sample, labels)

        results.append({
            'k': k, 'silhouette': sil,
            'davies_bouldin': db, 'inertia': inertia
        })
        print(f"  {k:>4} {sil:>12.4f} {db:>16.4f} {inertia:>12.1f}")

    # Select optimal k — highest silhouette score
    best = max(results, key=lambda x: x['silhouette'])
    print(f"\n  → Optimal k = {best['k']} "
          f"(silhouette={best['silhouette']:.4f}, "
          f"DB={best['davies_bouldin']:.4f})")

    return best['k'], results


# ─────────────────────────────────────────────
# STEP 3 — INCREMENTAL K-MEANS++
# Simulates streaming ECG data
# ─────────────────────────────────────────────

def incremental_kmeans(X_scaled, k, batch_size=BATCH_SIZE):
    """
    Incremental K-Means++ via MiniBatchKMeans.partial_fit().

    SIMULATION OF STREAMING:
        Data is fed in batches of `batch_size` windows,
        simulating continuous ECG stream from wearable devices.
        Each batch updates cluster centroids incrementally.

        In real deployment:
            - Each 5-second ECG window arrives at the edge node
            - Features are extracted on-device
            - Cluster assignment computed in O(k) time
            - Centroid updated with exponential moving average

    WHY THIS IS NOT JUST BATCH K-MEANS:
        partial_fit() updates centroids using:
            centroid_new = centroid_old + lr × (x - centroid_old)
        where lr decays over time. This means:
        1. New data has less influence than early data
           (prevents catastrophic forgetting)
        2. No need to store all historical data
           (memory efficient for edge deployment)
        3. Adapts to concept drift — if patient's ECG pattern
           changes (e.g., onset of AF), centroids gradually shift

    THESIS:
        "The incremental variant processes data in batches of 500
        windows (approximately 41 minutes of ECG at 5s/window),
        updating cluster centroids via exponentially decaying
        learning rate. This approach accommodates concept drift
        inherent in long-term wearable ECG monitoring without
        requiring full model retraining."
    """
    print(f"\n[STEP 3] Incremental K-Means++ (k={k}, batch_size={batch_size})")

    model = MiniBatchKMeans(
        n_clusters=k,
        init='k-means++',
        n_init=10,
        batch_size=batch_size,
        random_state=42,
        max_iter=100
    )

    n_batches = len(X_scaled) // batch_size + 1
    print(f"  Total windows : {len(X_scaled)}")
    print(f"  Batch size    : {batch_size}")
    print(f"  Total batches : {n_batches}")
    print()

    inertia_history = []

    for i in range(0, len(X_scaled), batch_size):
        batch = X_scaled[i:i + batch_size]
        if len(batch) == 0:
            continue
        model.partial_fit(batch)

        # Track inertia every 5 batches
        batch_num = i // batch_size + 1
        if batch_num % 5 == 0 or batch_num == 1:
            inertia = model.inertia_
            inertia_history.append((batch_num, inertia))
            print(f"  Batch {batch_num:>3}/{n_batches} — inertia: {inertia:.2f}")

    print(f"\n  ✓ Clustering complete")
    print(f"  Final inertia : {model.inertia_:.2f}")

    return model, inertia_history


# ─────────────────────────────────────────────
# STEP 4 — ASSIGN LABELS AND INTERPRET CLUSTERS
# ─────────────────────────────────────────────

def assign_and_interpret(df, X_scaled, model, scaler):
    """
    Assign cluster labels to all windows and interpret each cluster.

    Cluster interpretation:
        Each cluster centroid in standardized space is inverse-transformed
        back to original units for human interpretation.
        We then characterize each cluster by its mean feature values
        and assign a scheduling label (Edge / Fog / Cloud).
    """
    print("\n[STEP 4] Assigning cluster labels and interpreting clusters...")

    # Assign labels
    labels = model.predict(X_scaled)
    df = df.copy()
    df['cluster_id'] = labels

    # Inverse transform centroids to original feature space
    centroids_scaled   = model.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)
    centroids_df       = pd.DataFrame(centroids_original, columns=FEATURE_COLS)
    centroids_df.index.name = 'cluster_id'

    # Interpret each cluster
    print()
    print("  Cluster Profiles (original feature space):")
    print(f"  {'Cluster':>8} {'n':>6} {'%':>6} "
          f"{'Composite':>10} {'QRS':>8} {'ST(mV)':>8} "
          f"{'MI':>7} {'RAM':>6} {'BW':>6}  Scheduling Label")
    print("  " + "-" * 85)

    cluster_profiles = []
    total = len(df)

    for cid in sorted(df['cluster_id'].unique()):
        subset   = df[df['cluster_id'] == cid]
        n        = len(subset)
        pct      = 100 * n / total
        comp     = subset['composite_score'].mean()
        qrs      = subset['qrs_complexity'].mean()
        st_mv    = subset['st_deviation_mv'].mean()
        mi       = subset['MI'].mean()
        ram      = subset['RAM_MB'].mean()
        bw       = subset['BW_kbps'].mean()

        # Assign scheduling label based on centroid characteristics
        label = scheduling_label(comp, qrs, st_mv, mi)

        print(f"  {cid:>8} {n:>6} {pct:>5.1f}% "
              f"{comp:>10.4f} {qrs:>8.4f} {st_mv:>8.4f} "
              f"{mi:>7.0f} {ram:>6.1f} {bw:>6.1f}  {label}")

        cluster_profiles.append({
            'cluster_id':      cid,
            'n_windows':       n,
            'pct_windows':     round(pct, 2),
            'mean_composite':  round(comp, 4),
            'mean_qrs':        round(qrs,  4),
            'mean_st_mv':      round(st_mv, 4),
            'mean_MI':         round(mi,  1),
            'mean_RAM_MB':     round(ram, 1),
            'mean_BW_kbps':    round(bw,  1),
            'scheduling_label': label
        })

    return df, pd.DataFrame(cluster_profiles), centroids_df


def scheduling_label(composite, qrs, st_mv, mi):
    """
    Assign human-readable scheduling recommendation to a cluster.
    Based on cluster centroid characteristics.
    This is used for interpretation only — actual scheduling
    is done by NSGA-III in Day 3.
    """
    if st_mv >= 0.3105:
        return "CLOUD_FORCED (Critical)"
    elif mi >= 3000 or composite >= 0.42:
        return "CLOUD_PREFERRED (Complex)"
    elif mi >= 1500 or composite >= 0.27:
        return "EDGE_OR_FOG (Moderate)"
    else:
        return "EDGE (Simple)"


# ─────────────────────────────────────────────
# STEP 5 — VALIDATE CLUSTERING QUALITY
# ─────────────────────────────────────────────

def validate_clustering(X_scaled, labels):
    """
    Final clustering quality metrics on full dataset.
    These go into your thesis results table.
    """
    print("\n[STEP 5] Clustering Quality Metrics (full dataset):")

    # Subsample for silhouette (O(N²) complexity)
    n_sample = min(5000, len(X_scaled))
    idx      = np.random.choice(len(X_scaled), n_sample, replace=False)

    sil = silhouette_score(X_scaled[idx], labels[idx])
    db  = davies_bouldin_score(X_scaled[idx], labels[idx])

    print(f"  Silhouette Score     : {sil:.4f}  (target > 0.3)")
    print(f"  Davies-Bouldin Index : {db:.4f}   (lower is better)")

    quality = ("Good" if sil > 0.5 else
               "Acceptable" if sil > 0.3 else
               "Weak — consider adjusting k")
    print(f"  Cluster quality      : {quality}")

    # Cluster purity against task_class
    return sil, db


# ─────────────────────────────────────────────
# STEP 6 — SAVE AND SUMMARIZE
# ─────────────────────────────────────────────

def save_outputs(df_clustered, cluster_profiles, centroids_df,
                 optimal_k, sil, db, k_results):

    # Save clustered profiles
    df_clustered.to_csv(OUTPUT_CSV, index=False)

    # Save centroids
    centroids_df.to_csv(CENTROIDS_CSV)

    # Generate thesis-ready summary
    summary_lines = []
    summary_lines.append("=" * 65)
    summary_lines.append("CLUSTERING RESULTS — For Thesis")
    summary_lines.append("=" * 65)
    summary_lines.append("")
    summary_lines.append(f"Algorithm    : Incremental K-Means++ (MiniBatchKMeans)")
    summary_lines.append(f"Optimal k    : {optimal_k} (selected by max silhouette score)")
    summary_lines.append(f"Batch size   : {BATCH_SIZE} windows (~41 min ECG per batch)")
    summary_lines.append(f"Feature dims : 8 (5 complexity + 3 task parameters)")
    summary_lines.append("")
    summary_lines.append("Quality Metrics:")
    summary_lines.append(f"  Silhouette Score     : {sil:.4f}")
    summary_lines.append(f"  Davies-Bouldin Index : {db:.4f}")
    summary_lines.append("")
    summary_lines.append("k Selection Results:")
    summary_lines.append(f"  {'k':>4} {'Silhouette':>12} {'Davies-Bouldin':>16}")
    for r in k_results:
        marker = " ← selected" if r['k'] == optimal_k else ""
        summary_lines.append(
            f"  {r['k']:>4} {r['silhouette']:>12.4f} "
            f"{r['davies_bouldin']:>16.4f}{marker}"
        )
    summary_lines.append("")
    summary_lines.append("Cluster Profiles:")
    for _, row in cluster_profiles.iterrows():
        summary_lines.append(
            f"  Cluster {row['cluster_id']}: "
            f"n={row['n_windows']} ({row['pct_windows']}%), "
            f"MI={row['mean_MI']:.0f}, "
            f"composite={row['mean_composite']:.3f}, "
            f"QRS={row['mean_qrs']:.3f}, "
            f"ST={row['mean_st_mv']:.3f}mV → {row['scheduling_label']}"
        )

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    with open(SUMMARY_TXT, 'w', encoding='utf-8') as f:
        f.write(summary_text)

    print(f"\n[OUTPUTS]")
    print(f"  {OUTPUT_CSV}    — clustered task profiles")
    print(f"  {CENTROIDS_CSV} — cluster centroids")
    print(f"  {SUMMARY_TXT}   — thesis-ready summary")
    print("\n[NEXT STEP] Run nsga3_scheduling_day3.py")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_clustering():
    print("=" * 60)
    print("ECG Task Clustering — Day 2")
    print("Incremental K-Means++ on 8-dimensional feature space")
    print("=" * 60)

    # Step 1 — Load and scale
    print("\n[STEP 1] Loading and scaling features...")
    df, X_scaled, scaler = load_and_scale(INPUT_CSV)
    print(f"  Loaded {len(df)} windows, {len(FEATURE_COLS)} features")
    print(f"  Features: {FEATURE_COLS}")

    # Step 2 — Find optimal k
    optimal_k, k_results = find_optimal_k(X_scaled)

    # Step 3 — Incremental K-Means++
    model, inertia_history = incremental_kmeans(X_scaled, optimal_k)

    # Step 4 — Assign and interpret
    df_clustered, cluster_profiles, centroids_df = assign_and_interpret(
        df, X_scaled, model, scaler
    )

    # Step 5 — Validate
    labels = df_clustered['cluster_id'].values
    sil, db = validate_clustering(X_scaled, labels)

    # Step 6 — Save
    save_outputs(df_clustered, cluster_profiles, centroids_df,
                 optimal_k, sil, db, k_results)

    return df_clustered, model, cluster_profiles


if __name__ == "__main__":
    run_clustering()
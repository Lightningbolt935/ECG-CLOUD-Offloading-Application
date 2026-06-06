"""
ECG Complexity Score Validation + Otsu Threshold Selection
===========================================================
PURPOSE:
    1. Validate that complexity scores correlate with MIT-BIH
       clinical annotations (unsupervised → supervised correlation)
    2. Use Otsu's method to find optimal class boundaries
       from the data itself — not hand-picked values
    3. Produce publishable validation table for thesis

This script uses task_profiles.csv (already generated).
It does NOT recompute any signals — pure analysis only.

RUN:
    python validate_scores.py

OUTPUT:
    - Printed validation table (copy into thesis)
    - Otsu thresholds to paste into pipeline v4
    - otsu_thresholds.txt (save these values)
    - validation_results.csv
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kruskal
import warnings
warnings.filterwarnings('ignore')

CSV_PATH = "./task_profiles.csv"

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

def load_and_prepare():
    df = pd.read_csv(CSV_PATH)

    # Classify windows by annotation composition
    # This uses the beat labels already collected in the CSV
    # These labels are NOT used in scoring — only for validation here
    def window_type(row):
        total = row['normal_beats'] + row['pvc_beats'] + \
                row['apb_beats'] + row['other_beats']
        if total == 0:
            return 'unlabeled'
        normal_ratio = row['normal_beats'] / total
        if normal_ratio >= 0.80:
            return 'predominantly_normal'
        elif normal_ratio >= 0.40:
            return 'mixed'
        else:
            return 'predominantly_abnormal'

    df['window_type'] = df.apply(window_type, axis=1)
    return df


# ─────────────────────────────────────────────
# OTSU'S METHOD
# Finds optimal threshold(s) to separate distributions
# ─────────────────────────────────────────────

def otsu_threshold(values, n_thresholds=1):
    """
    Otsu's multi-threshold method.

    WHAT IT DOES:
        Finds the threshold(s) that minimize intra-class variance
        (equivalently, maximize inter-class variance) in a
        1D distribution. Originally used in image segmentation
        (Otsu, 1979), here applied to score distributions.

    WHY THIS IS VALID FOR YOUR PAPER:
        The thresholds are derived mathematically from the empirical
        distribution of complexity scores across all 17,328 windows.
        They are not chosen by the researcher — they are the
        statistically optimal separation points given the data.
        This is a standard, peer-reviewed method with thousands
        of citations.

    Args:
        values: 1D array of scores
        n_thresholds: 1 for binary split, 2 for 3-class split

    Returns:
        list of threshold values
    """
    values = np.array(values)
    values = values[~np.isnan(values)]

    # Use 1000 candidate threshold points
    candidates = np.linspace(values.min(), values.max(), 1000)

    if n_thresholds == 1:
        best_thresh = candidates[0]
        best_var    = np.inf

        for t in candidates:
            w0 = values[values <= t]
            w1 = values[values >  t]
            if len(w0) == 0 or len(w1) == 0:
                continue
            # Weighted intra-class variance
            var = (len(w0) * np.var(w0) + len(w1) * np.var(w1)) / len(values)
            if var < best_var:
                best_var    = var
                best_thresh = t

        return [round(float(best_thresh), 4)]

    elif n_thresholds == 2:
        best_t1, best_t2 = candidates[0], candidates[-1]
        best_var = np.inf

        # Reduce search space for speed
        for t1 in candidates[::5]:
            for t2 in candidates[::5]:
                if t2 <= t1:
                    continue
                w0 = values[values <= t1]
                w1 = values[(values > t1) & (values <= t2)]
                w2 = values[values > t2]
                if len(w0) == 0 or len(w1) == 0 or len(w2) == 0:
                    continue
                var = (len(w0)*np.var(w0) +
                       len(w1)*np.var(w1) +
                       len(w2)*np.var(w2)) / len(values)
                if var < best_var:
                    best_var = var
                    best_t1, best_t2 = t1, t2

        return [round(float(best_t1), 4), round(float(best_t2), 4)]


# ─────────────────────────────────────────────
# VALIDATION ANALYSIS
# ─────────────────────────────────────────────

def run_validation(df):
    print("=" * 65)
    print("VALIDATION REPORT — ECG Complexity Score Analysis")
    print("=" * 65)

    metrics = ['composite_score', 'sample_entropy',
               'qrs_complexity', 'variance_score', 'st_deviation']

    # ── 1. Spearman Correlation with Abnormality Ratio ──────────────
    print("\n[1] SPEARMAN CORRELATION WITH ANNOTATION-BASED ABNORMALITY RATIO")
    print("    (measures how well each score tracks clinical ground truth)")
    print()
    print(f"    {'Metric':<22} {'ρ (rho)':>10} {'p-value':>12} {'Interpretation'}")
    print("    " + "-" * 60)

    correlations = {}
    for metric in metrics:
        rho, pval = spearmanr(df[metric], df['abnormal_ratio'])
        correlations[metric] = (rho, pval)
        sig = "✓ significant" if pval < 0.001 else "✗ not significant"
        strength = ("strong" if abs(rho) > 0.5 else
                    "moderate" if abs(rho) > 0.3 else "weak")
        print(f"    {metric:<22} {rho:>10.4f} {pval:>12.2e}  {sig} ({strength})")

    # ── 2. Mean Scores by Window Type ───────────────────────────────
    print("\n[2] MEAN COMPLEXITY SCORES BY CLINICAL WINDOW TYPE")
    print("    (validates directional correctness — abnormal > normal)")
    print()

    types_order = ['predominantly_normal', 'mixed', 'predominantly_abnormal']
    type_labels = {
        'predominantly_normal':    'Predominantly Normal (≥80% N beats)',
        'mixed':                   'Mixed (40-80% N beats)',
        'predominantly_abnormal':  'Predominantly Abnormal (<40% N beats)'
    }

    header = f"    {'Window Type':<42}"
    for m in ['composite', 'qrs', 'st_mv', 'samp_ent', 'variance']:
        header += f" {m:>9}"
    print(header)
    print("    " + "-" * 90)

    type_stats = {}
    for wtype in types_order:
        subset = df[df['window_type'] == wtype]
        if len(subset) == 0:
            continue
        n       = len(subset)
        comp    = subset['composite_score'].mean()
        qrs     = subset['qrs_complexity'].mean()
        st_mv   = subset['st_deviation_mv'].mean()
        se      = subset['sample_entropy'].mean()
        var     = subset['variance_score'].mean()
        type_stats[wtype] = {'n': n, 'composite': comp}
        label = type_labels[wtype]
        print(f"    {label:<42} {comp:>9.4f} {qrs:>9.4f} "
              f"{st_mv:>9.4f} {se:>9.4f} {var:>9.4f}  (n={n})")

    # Direction check
    print()
    if ('predominantly_normal' in type_stats and
            'predominantly_abnormal' in type_stats):
        norm_score  = type_stats['predominantly_normal']['composite']
        abn_score   = type_stats['predominantly_abnormal']['composite']
        direction_ok = abn_score > norm_score
        print(f"    Direction check (abnormal > normal): "
              f"{'✓ PASS' if direction_ok else '✗ FAIL'}")
        print(f"    Effect size: {abn_score - norm_score:.4f} "
              f"({((abn_score-norm_score)/norm_score*100):.1f}% increase)")

    # ── 3. Kruskal-Wallis Test ───────────────────────────────────────
    print("\n[3] KRUSKAL-WALLIS TEST (statistical significance of group differences)")
    print("    H0: complexity scores are equal across window types")
    print()

    valid_types = [t for t in types_order
                   if t in df['window_type'].values and
                   len(df[df['window_type'] == t]) > 0]

    if len(valid_types) >= 2:
        groups = [df[df['window_type'] == t]['composite_score'].values
                  for t in valid_types]
        h_stat, p_val = kruskal(*groups)
        print(f"    H-statistic : {h_stat:.4f}")
        print(f"    p-value     : {p_val:.2e}")
        sig = "✓ REJECT H0" if p_val < 0.001 else "✗ FAIL TO REJECT H0"
        print(f"    Decision    : {sig} (α=0.001)")
        if p_val < 0.001:
            print("    → Complexity scores differ significantly across")
            print("      window types. Scores are clinically meaningful.")

    # ── 4. Otsu Threshold Selection ──────────────────────────────────
    print("\n[4] OTSU THRESHOLD SELECTION")
    print("    Finding optimal class boundaries from data distribution")
    print()

    composite_scores = df['composite_score'].values
    st_mv_values     = df['st_deviation_mv'].values

    # Two thresholds for composite → 3 non-critical classes
    print("    Computing 2-threshold Otsu for composite score...")
    t1, t2 = otsu_threshold(composite_scores, n_thresholds=2)
    print(f"    Composite thresholds: T1={t1}, T2={t2}")
    print(f"    → Class 0 (Simple)  : composite < {t1}")
    print(f"    → Class 1 (Moderate): {t1} ≤ composite < {t2}")
    print(f"    → Class 2 (Complex) : composite ≥ {t2}")

    # One threshold for ST mV → critical boundary
    print()
    print("    Computing 1-threshold Otsu for ST deviation (mV)...")
    st_thresh = otsu_threshold(st_mv_values, n_thresholds=1)[0]
    print(f"    ST threshold: {st_thresh} mV")
    print(f"    → Normal/Borderline: ST < {st_thresh} mV")
    print(f"    → Critical override: ST ≥ {st_thresh} mV")

    # Clinical reasonableness check
    print()
    aha_threshold = 0.1  # AHA minimum significance
    if st_thresh >= aha_threshold:
        print(f"    ✓ Otsu ST threshold ({st_thresh}mV) ≥ AHA minimum (0.1mV)")
        print(f"      Clinically justified.")
    else:
        print(f"    ⚠ Otsu ST threshold ({st_thresh}mV) < AHA minimum (0.1mV)")
        print(f"      Using max(Otsu, 0.1mV) = {max(st_thresh, 0.1):.4f}mV")
        st_thresh = max(st_thresh, 0.1)

    # Show what distribution looks like with Otsu thresholds
    print()
    print("    Projected class distribution with Otsu thresholds:")
    total = len(df)
    critical = df[df['st_deviation_mv'] >= st_thresh]
    non_critical = df[df['st_deviation_mv'] < st_thresh]
    c0 = non_critical[non_critical['composite_score'] <  t1]
    c1 = non_critical[(non_critical['composite_score'] >= t1) &
                      (non_critical['composite_score'] <  t2)]
    c2 = non_critical[non_critical['composite_score'] >= t2]
    c3 = critical

    for cls, label, subset in [
        (0, 'Simple (Edge)',          c0),
        (1, 'Moderate (Edge/Fog)',    c1),
        (2, 'Complex (Cloud)',        c2),
        (3, 'Critical (Forced)',      c3)
    ]:
        n   = len(subset)
        pct = 100 * n / total
        print(f"    Class {cls} — {label:22s}: {n:5d} ({pct:.1f}%)")

    # ── 5. Ablation Study Preview ────────────────────────────────────
    print("\n[5] ABLATION STUDY — Individual Metric Contributions")
    print("    (removing each metric and measuring correlation drop)")
    print()
    print(f"    {'Configuration':<35} {'ρ with abnormal_ratio':>22} {'Δρ':>8}")
    print("    " + "-" * 68)

    weights = {'sample_entropy': 0.30, 'qrs_complexity': 0.25,
               'variance_score': 0.25, 'st_deviation':   0.20}

    base_rho = correlations['composite_score'][0]
    print(f"    {'Full model (all 4 metrics)':<35} {base_rho:>22.4f} {'—':>8}")

    for removed in weights.keys():
        remaining = {k: v for k, v in weights.items() if k != removed}
        total_w   = sum(remaining.values())
        ablated   = sum(df[m] * (w / total_w)
                        for m, w in remaining.items())
        rho_ab, _ = spearmanr(ablated, df['abnormal_ratio'])
        delta     = rho_ab - base_rho
        label     = f"Without {removed.replace('_',' ')}"
        print(f"    {label:<35} {rho_ab:>22.4f} {delta:>+8.4f}")

    # ── Save thresholds ──────────────────────────────────────────────
    thresholds = {
        'composite_t1':     t1,
        'composite_t2':     t2,
        'st_threshold_mv':  round(float(st_thresh), 4)
    }

    with open('./otsu_thresholds.txt', 'w') as f:
        f.write("# Otsu-derived thresholds for ECG Task Profiling Pipeline\n")
        f.write("# Paste these into ecg_pipeline_day1_v4.py\n\n")
        for k, v in thresholds.items():
            f.write(f"{k} = {v}\n")

    df.to_csv('./validation_results.csv', index=False)

    print("\n" + "=" * 65)
    print("OUTPUTS SAVED")
    print("=" * 65)
    print("  otsu_thresholds.txt    — paste into pipeline v4")
    print("  validation_results.csv — full annotated dataset")
    print()
    print("PASTE THESE INTO ecg_pipeline_day1_v4.py:")
    print(f"  COMPOSITE_T1     = {t1}")
    print(f"  COMPOSITE_T2     = {t2}")
    print(f"  ST_THRESHOLD_MV  = {thresholds['st_threshold_mv']}")
    print("=" * 65)

    return thresholds


if __name__ == "__main__":
    print("Loading task_profiles.csv...")
    df = load_and_prepare()
    print(f"Loaded {len(df)} windows from {df['record_id'].nunique()} records\n")
    thresholds = run_validation(df)
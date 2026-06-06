"""
ECG Task Profiling Pipeline — Day 1 v4 (FINAL)
================================================
All thresholds derived from data using Otsu's method.
No manually chosen boundaries.

Otsu thresholds (from validate_scores.py):
    COMPOSITE_T1    = 0.2747  (Simple / Moderate boundary)
    COMPOSITE_T2    = 0.4172  (Moderate / Complex boundary)
    ST_THRESHOLD_MV = 0.3105  (Critical emergency override)

Validation results (from validate_scores.py):
    Spearman ρ = 0.2754, p < 10⁻²⁹⁸  (statistically significant)
    Kruskal-Wallis H = 1949.67, p < 0.001
    QRS complexity dominant contributor (Δρ = -0.118 on removal)

Expected class distribution:
    Class 0 — Simple   : ~17%
    Class 1 — Moderate : ~55%
    Class 2 — Complex  : ~13%
    Class 3 — Critical : ~15%
"""

import wfdb
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

DATA_DIR        = "./mitdb"
OUTPUT_CSV      = "./task_profiles.csv"
WINDOW_SIZE_SEC = 5
SAMPLING_RATE   = 360
WINDOW_SIZE     = WINDOW_SIZE_SEC * SAMPLING_RATE  # 1800 samples

# Otsu-derived thresholds — DO NOT change manually
# These were computed from 17,328 windows via Otsu's method
COMPOSITE_T1    = 0.2747   # Simple / Moderate boundary
COMPOSITE_T2    = 0.4172   # Moderate / Complex boundary
ST_THRESHOLD_MV = 0.3105   # Critical override (≥ AHA 0.1mV minimum)

RECORD_IDS = [
    '100','101','102','103','104','105','106','107',
    '108','109','111','112','113','114','115','116',
    '117','118','119','121','122','123','124','200',
    '201','202','203','205','207','208','209','210',
    '212','213','214','215','217','219','220','221',
    '222','223','228','230','231','232','233','234'
]


# ─────────────────────────────────────────────
# PREPROCESSING
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


# ─────────────────────────────────────────────
# COMPLEXITY METRICS
# (unchanged from v3 — only thresholds change)
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# TASK PARAMETER ASSIGNMENT
# Uses Otsu-derived thresholds throughout
# ─────────────────────────────────────────────

def assign_task_parameters(complexity_score, st_deviation_mv):
    """
    Task class assignment using Otsu-derived thresholds.

    Thresholds:
        COMPOSITE_T1    = 0.2747  (Otsu, from 17,328-window distribution)
        COMPOSITE_T2    = 0.4172  (Otsu, from 17,328-window distribution)
        ST_THRESHOLD_MV = 0.3105  (Otsu, ≥ AHA 0.1mV clinical minimum)

    MI mapping (linear within each band):
        Class 0: MI = 800  + score × 1466   range [800,  1200]
        Class 1: MI = 1500 + score × 5000   range [1500, 3000]
        Class 2: MI = 3000 + score × 5000   range [3000, 5000]
        Class 3: MI = 5000 (fixed maximum)
    """
    # Emergency override using Otsu-derived ST threshold
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
# RECORD PROCESSING
# ─────────────────────────────────────────────

def process_record(record_id):
    record_path = os.path.join(DATA_DIR, record_id)
    try:
        record     = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, 'atr')
        signal_mv  = record.p_signal[:, 0]
        fs         = record.fs

        tasks = []
        for w in range(len(signal_mv) // WINDOW_SIZE):
            start  = w * WINDOW_SIZE
            end    = start + WINDOW_SIZE
            window = signal_mv[start:end]

            if np.sum(np.isnan(window)) > WINDOW_SIZE * 0.1:
                continue
            window = np.where(np.isnan(window), np.nanmean(window), window)

            scores = compute_complexity_score(window, fs)
            params = assign_task_parameters(
                scores['composite_score'],
                scores['st_deviation_mv']
            )

            ann_times  = annotation.sample
            ann_labels = annotation.symbol
            window_ann = [ann_labels[i] for i, t in enumerate(ann_times)
                          if start <= t < end]

            total_beats   = len(window_ann)
            normal_beats  = window_ann.count('N')
            abnormal_ratio = ((total_beats - normal_beats) / total_beats
                              if total_beats > 0 else 0.0)

            tasks.append({
                'record_id':        record_id,
                'window_id':        w,
                'window_start_sec': round(start / fs, 2),
                'window_end_sec':   round(end   / fs, 2),
                **scores,
                **params,
                'normal_beats':    normal_beats,
                'pvc_beats':       window_ann.count('V'),
                'apb_beats':       window_ann.count('A'),
                'other_beats':     total_beats - normal_beats
                                   - window_ann.count('V')
                                   - window_ann.count('A'),
                'abnormal_ratio':  round(abnormal_ratio, 4)
            })

        print(f"  ✓ Record {record_id}: {len(tasks)} windows")
        return tasks

    except FileNotFoundError:
        print(f"  ✗ Record {record_id}: not found")
        return []
    except Exception as e:
        print(f"  ✗ Record {record_id}: {e}")
        return []


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_pipeline():
    print("=" * 60)
    print("ECG Task Profiling Pipeline v4 (FINAL)")
    print(f"Thresholds: T1={COMPOSITE_T1}, T2={COMPOSITE_T2}, "
          f"ST={ST_THRESHOLD_MV}mV  [Otsu-derived]")
    print("=" * 60)

    if not os.path.exists(DATA_DIR):
        print(f"[ERROR] '{DATA_DIR}' not found.")
        return

    all_tasks = []
    print(f"\nProcessing {len(RECORD_IDS)} records...")
    for rid in RECORD_IDS:
        all_tasks.extend(process_record(rid))

    df = pd.DataFrame(all_tasks)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Total windows : {len(df)}")
    print(f"Records       : {df['record_id'].nunique()}")

    print("\nTask Class Distribution:")
    labels = {0:'Simple (Edge)', 1:'Moderate (Edge/Fog)',
              2:'Complex (Cloud)', 3:'Critical (Cloud Forced)'}
    for cls, label in labels.items():
        n   = len(df[df['task_class'] == cls])
        pct = 100 * n / len(df)
        print(f"  Class {cls} — {label:25s}: {n:5d} ({pct:.1f}%)")

    print(f"\nMI: min={df['MI'].min()}  max={df['MI'].max()}  "
          f"mean={df['MI'].mean():.0f}  std={df['MI'].std():.0f}")

    print("\nST Deviation (mV):")
    print(f"  < 0.1mV  (normal)    : "
          f"{len(df[df['st_deviation_mv']<0.1]):5d} "
          f"({100*len(df[df['st_deviation_mv']<0.1])/len(df):.1f}%)")
    print(f"  0.1-0.31mV (border)  : "
          f"{len(df[(df['st_deviation_mv']>=0.1)&(df['st_deviation_mv']<ST_THRESHOLD_MV)]):5d} "
          f"({100*len(df[(df['st_deviation_mv']>=0.1)&(df['st_deviation_mv']<ST_THRESHOLD_MV)])/len(df):.1f}%)")
    print(f"  ≥{ST_THRESHOLD_MV}mV (critical) : "
          f"{len(df[df['st_deviation_mv']>=ST_THRESHOLD_MV]):5d} "
          f"({100*len(df[df['st_deviation_mv']>=ST_THRESHOLD_MV])/len(df):.1f}%)")

    print(f"\n[NEXT STEP] Run ecg_clustering_day2.py")
    print("=" * 60)
    return df


if __name__ == "__main__":
    run_pipeline()
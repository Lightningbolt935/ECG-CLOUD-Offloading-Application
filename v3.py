"""
ECG Task Profiling Pipeline — Day 1 v3
=======================================
ROOT CAUSE FIX:
    ST deviation was being computed on normalized signal [0,1].
    This divided the clinical 0.1mV threshold by the full signal
    amplitude range (~2mV), making every window appear as critical.

    v3 rule: ST deviation computed in RAW MILLIVOLTS only.
    Clinical threshold: 0.1mV (AHA standard, hard reference).
    
    SampEn and Variance still use normalized signal internally
    (they measure shape/structure, not absolute amplitude).
    QRS uses raw signal for peak detection (prominence in mV).

Expected distribution after fix:
    Class 0 — Simple   : ~35-45%
    Class 1 — Moderate : ~35-40%
    Class 2 — Complex  : ~10-20%
    Class 3 — Critical :  ~3-7%
"""

import wfdb
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import spearmanr
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

DATA_DIR       = "./mitdb"
OUTPUT_CSV     = "./task_profiles.csv"
WINDOW_SIZE_SEC = 5
SAMPLING_RATE  = 360
WINDOW_SIZE    = WINDOW_SIZE_SEC * SAMPLING_RATE  # 1800 samples

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
    """
    Bandpass filter in mV domain.
    Removes baseline wander (<0.5Hz) and high-freq noise (>40Hz).
    Signal remains in millivolts after filtering.
    """
    nyq  = 0.5 * fs
    low  = max(lowcut / nyq,  0.001)
    high = min(highcut / nyq, 0.999)
    if low >= high:
        return signal_mv
    try:
        b, a = butter(4, [low, high], btype='band')
        return filtfilt(b, a, signal_mv)
    except Exception:
        return signal_mv


# ─────────────────────────────────────────────
# METRIC 1 — SAMPLE ENTROPY
# Works on normalized signal (measures shape, not amplitude)
# ─────────────────────────────────────────────

def sample_entropy(signal_mv, m=2, r_factor=0.2):
    """
    Sample Entropy — signal unpredictability.
    
    Operates on normalized signal internally because SampEn
    measures structural complexity (pattern recurrence), not
    absolute amplitude. Normalization ensures comparability
    across patients with different ECG amplitudes.

    Formula: SampEn(m,r,N) = -ln(A/B)
        m = 2 (template length, standard)
        r = 0.2 × std(normalized signal) (tolerance)
        B = template matches of length m
        A = template matches of length m+1

    Uses first 300 samples for O(N²) tractability.
    Returns float in [0, ~2.5], higher = more complex.
    """
    # Normalize internally for shape comparison
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
        num_windows = N - m_len
        if num_windows <= 0:
            return 0
        
        # Pre-extract all windows to vectorize the inner loop
        windows = np.array([sig[i:i + m_len] for i in range(num_windows)])
        
        count = 0
        for i in range(num_windows - 1):
            # Compute distance from current window to all subsequent windows simultaneously
            diffs = np.max(np.abs(windows[i+1:] - windows[i]), axis=1)
            count += np.sum(diffs < r)
        return count

    A = count_matches(m + 1)
    B = count_matches(m)
    if B == 0 or A == 0:
        return 0.0
    return float(-np.log(A / B))


# ─────────────────────────────────────────────
# METRIC 2 — QRS COMPLEXITY
# Uses filtered mV signal for peak detection
# ─────────────────────────────────────────────

def qrs_complexity_score(signal_mv, fs=360):
    """
    QRS Complexity — RR interval irregularity.
    
    Uses prominence-based peak detection on filtered mV signal.
    Prominence threshold: 0.3mV (typical R-peak is 0.5-2.0mV,
    noise/artifact peaks are typically <0.2mV).

    CV threshold = 0.15:
        Normal sinus rhythm:    CV ~ 0.02-0.08
        Mild arrhythmia:        CV ~ 0.08-0.15
        Significant arrhythmia: CV > 0.15

    Returns float in [0,1].
    """
    filtered = bandpass_filter(signal_mv, fs)

    # Peak detection in mV domain
    # 0.3mV prominence filters noise without missing true R-peaks
    peaks, _ = find_peaks(
        filtered,
        distance=int(0.15 * fs),
        prominence=0.3          # mV — physically meaningful threshold
    )

    if len(peaks) < 3:
        return 0.3  # insufficient peaks → moderate uncertainty

    rr = np.diff(peaks).astype(float)
    # Filter physiologically impossible RR intervals
    # HR range 20-300 BPM → RR 72-1800 samples at 360Hz
    rr = rr[(rr >= 72) & (rr <= 1800)]

    if len(rr) < 2:
        return 0.3

    cv = np.std(rr) / (np.mean(rr) + 1e-6)
    return float(min(cv / 0.15, 1.0))


# ─────────────────────────────────────────────
# METRIC 3 — VARIANCE SCORE
# Normalized signal (amplitude-independent shape measure)
# ─────────────────────────────────────────────

def variance_score(signal_mv):
    """
    Normalized variance — morphological amplitude diversity.
    
    Normalized internally because variance purpose here is to
    capture waveform shape diversity, not absolute amplitude.
    Patient A may have 0.5mV R-peaks and patient B 1.5mV R-peaks
    due to body habitus — both can have equally complex morphology.

    Empirical ceiling 0.08 from MIT-BIH distribution analysis:
        Normal records:       variance ~ 0.01-0.04
        Pathological records: variance ~ 0.05-0.10
    """
    s_min, s_max = np.min(signal_mv), np.max(signal_mv)
    if s_max == s_min:
        return 0.0
    norm = (signal_mv - s_min) / (s_max - s_min)
    return float(min(np.var(norm) / 0.08, 1.0))


# ─────────────────────────────────────────────
# METRIC 4 — ST DEVIATION  ← THE KEY FIX
# Raw millivolts — NO normalization
# ─────────────────────────────────────────────

def st_deviation_score(signal_mv, fs=360):
    """
    ST Segment Deviation — computed in raw millivolts.

    ROOT CAUSE FIX (v3):
        Previous versions normalized signal to [0,1] before
        computing ST deviation. This caused the 0.1mV clinical
        threshold to be compared against a dimensionless ratio,
        making every window appear critical (mean ST=0.87).

        Fix: use p_signal directly (already in mV per wfdb docs).
        Clinical threshold 0.1mV is applied in the mV domain.

    METHOD (beat-by-beat, per AHA measurement standard):
        For each detected R-peak:
            baseline  = mean of PR segment (160-120ms before R)
                        This is the isoelectric reference line.
            st_point  = signal value at J+60ms (140ms after R)
                        J point = end of QRS complex.
            deviation = |st_point - baseline|  [in mV]

        Mean deviation across all beats in window → score.

    NORMALIZATION for composite:
        score = min(mean_deviation_mV / 0.5, 1.0)
        
        Why 0.5mV ceiling?
            0.1mV = clinical significance threshold → score 0.2
            0.2mV = moderate deviation             → score 0.4
            0.5mV = severe ST change               → score 1.0
        This gives good dynamic range while anchoring 0.1mV at 0.2.

    EMERGENCY THRESHOLD (applied in assign_task_parameters):
        Raw deviation > 0.2mV → Critical flag
        (2× the clinical significance threshold = definitive abnormality)

    Returns: float in [0,1] AND raw_deviation_mv for threshold use
    """
    filtered = bandpass_filter(signal_mv, fs)

    # Peak detection in mV domain
    peaks, _ = find_peaks(
        filtered,
        distance=int(0.15 * fs),
        prominence=0.3
    )

    if len(peaks) == 0:
        return 0.0, 0.0  # score, raw_mv

    st_deviations_mv = []

    for peak in peaks:
        # PR segment: 160ms to 120ms before R-peak
        pr_start = peak - int(0.16 * fs)  # 58 samples before R
        pr_end   = peak - int(0.12 * fs)  # 43 samples before R

        # ST point: J+60ms = ~140ms after R-peak
        st_point = peak + int(0.14 * fs)  # 50 samples after R

        if pr_start < 0 or pr_end <= pr_start or st_point >= len(filtered):
            continue

        baseline_mv = np.mean(filtered[pr_start:pr_end])
        st_mv       = filtered[st_point]
        deviation   = abs(st_mv - baseline_mv)
        st_deviations_mv.append(deviation)

    if not st_deviations_mv:
        return 0.0, 0.0

    mean_dev_mv = float(np.mean(st_deviations_mv))

    # Normalize to [0,1] for composite score
    # Ceiling: 0.5mV (severe ST change)
    score = float(min(mean_dev_mv / 0.5, 1.0))

    return score, mean_dev_mv


# ─────────────────────────────────────────────
# COMPOSITE SCORE
# ─────────────────────────────────────────────

def compute_complexity_score(window_mv, fs=360):
    """
    Weighted composite complexity score.

    Weights (defended in thesis as follows):
        SampEn  0.30 — primary nonlinear complexity indicator,
                        most established in biomedical literature
        QRS     0.25 — direct arrhythmia severity indicator
        Var     0.25 — morphological diversity indicator
        ST      0.20 — emergency triage indicator; has separate
                        raw mV pathway for emergency override,
                        so lower composite weight is appropriate

    Formula:
        C = 0.30×SampEn_norm + 0.25×QRS + 0.25×Var + 0.20×ST_norm

    All components in [0,1]. Composite in [0,1].
    """
    se_score          = float(min(sample_entropy(window_mv) / 2.0, 1.0))
    qrs_score         = float(qrs_complexity_score(window_mv, fs))
    var_score         = float(variance_score(window_mv))
    st_score, st_raw  = st_deviation_score(window_mv, fs)

    composite = (0.30 * se_score  +
                 0.25 * qrs_score +
                 0.25 * var_score +
                 0.20 * st_score)

    return {
        'sample_entropy':   round(se_score,          4),
        'qrs_complexity':   round(qrs_score,         4),
        'variance_score':   round(var_score,          4),
        'st_deviation':     round(st_score,           4),
        'st_deviation_mv':  round(st_raw,             4),  # raw mV — for validation
        'composite_score':  round(float(composite),   4)
    }


# ─────────────────────────────────────────────
# TASK PARAMETER ASSIGNMENT
# Emergency threshold now uses raw mV, not normalized score
# ─────────────────────────────────────────────

def assign_task_parameters(complexity_score, st_deviation_mv):
    """
    Maps complexity → iFogSim2 task parameters.

    Emergency override: st_deviation_mv > 0.2mV
        Why 0.2mV?
        AHA defines 0.1mV as clinically significant ST change.
        We use 2× this threshold (0.2mV) as the emergency cutoff
        to ensure the override triggers only on definitive
        abnormality, not borderline cases. This is a principled,
        clinically referenced choice — not an arbitrary number.

    Class boundaries (to be confirmed by Otsu thresholding
    in validate_scores.py — these are initial estimates):
        Class 0: composite < 0.30
        Class 1: 0.30 ≤ composite < 0.55
        Class 2: composite ≥ 0.55
        Class 3: st_mv > 0.2 (emergency override)

    MI assignment (linear within band):
        Class 0: 800  + composite×1333  → [800,  1200]
        Class 1: 1500 + composite×5000  → [1500, 3000]
        Class 2: 3000 + composite×4000  → [3000, 5000]
        Class 3: 5000 fixed (maximum)
    """
    # Emergency override — raw mV threshold (AHA-referenced)
    if st_deviation_mv > 0.2:
        return {
            'MI': 5000, 'RAM_MB': 50, 'BW_kbps': 500,
            'task_class': 3, 'priority': 'CRITICAL',
            'offload_target': 'CLOUD_FORCED'
        }

    if complexity_score < 0.30:
        return {
            'MI': int(800 + complexity_score * 1333),
            'RAM_MB': 10, 'BW_kbps': 100,
            'task_class': 0, 'priority': 'LOW',
            'offload_target': 'EDGE'
        }
    elif complexity_score < 0.55:
        return {
            'MI': int(1500 + (complexity_score - 0.30) * 5000),
            'RAM_MB': 25, 'BW_kbps': 250,
            'task_class': 1, 'priority': 'MEDIUM',
            'offload_target': 'EDGE_OR_FOG'
        }
    else:
        return {
            'MI': int(3000 + (complexity_score - 0.55) * 4000),
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

        # p_signal is already in physical units (mV) — no conversion needed
        signal_mv  = record.p_signal[:, 0]
        fs         = record.fs

        tasks = []
        num_windows = len(signal_mv) // WINDOW_SIZE

        for w in range(num_windows):
            start  = w * WINDOW_SIZE
            end    = start + WINDOW_SIZE
            window = signal_mv[start:end]

            # Skip windows with excessive NaN
            if np.sum(np.isnan(window)) > WINDOW_SIZE * 0.1:
                continue
            window = np.where(np.isnan(window), np.nanmean(window), window)

            scores = compute_complexity_score(window, fs)
            params = assign_task_parameters(
                scores['composite_score'],
                scores['st_deviation_mv']
            )

            # Beat annotations in this window
            ann_times  = annotation.sample
            ann_labels = annotation.symbol
            window_ann = [ann_labels[i] for i, t in enumerate(ann_times)
                          if start <= t < end]

            beat_counts = {
                'normal_beats': window_ann.count('N'),
                'pvc_beats':    window_ann.count('V'),
                'apb_beats':    window_ann.count('A'),
                'other_beats':  len(window_ann)
                                - window_ann.count('N')
                                - window_ann.count('V')
                                - window_ann.count('A')
            }

            # Abnormality ratio — used for validation only, not for scoring
            total_beats = len(window_ann)
            abnormal_ratio = (
                (total_beats - window_ann.count('N')) / total_beats
                if total_beats > 0 else 0.0
            )

            task = {
                'record_id':        record_id,
                'window_id':        w,
                'window_start_sec': round(start / fs, 2),
                'window_end_sec':   round(end / fs, 2),
                **scores,
                **params,
                **beat_counts,
                'abnormal_ratio':   round(abnormal_ratio, 4)
            }
            tasks.append(task)

        print(f"  ✓ Record {record_id}: {len(tasks)} windows processed")
        return tasks

    except FileNotFoundError:
        print(f"  ✗ Record {record_id}: not found in {DATA_DIR}")
        return []
    except Exception as e:
        print(f"  ✗ Record {record_id}: error — {e}")
        return []


# ─────────────────────────────────────────────
# QUICK SANITY CHECK
# Run this first — verifies ST is working correctly
# ─────────────────────────────────────────────

def sanity_check():
    """
    Verifies that Record 100 (normal) has lower ST deviation
    than Record 207 (ventricular fibrillation/complex arrhythmia).
    
    This is a minimum correctness requirement — if this fails,
    the ST metric is not functioning as intended.
    """
    print("Running sanity check...")
    print("(Record 100 = normal sinus, Record 207 = complex arrhythmia)")
    print()

    results = {}
    for rec_id in ['100', '207']:
        rec  = wfdb.rdrecord(os.path.join(DATA_DIR, rec_id))
        sig  = rec.p_signal[:, 0]
        window = sig[:WINDOW_SIZE]
        window = np.where(np.isnan(window), np.nanmean(window), window)
        scores = compute_complexity_score(window, rec.fs)
        results[rec_id] = scores
        print(f"Record {rec_id}:")
        print(f"  ST deviation (normalized) : {scores['st_deviation']}")
        print(f"  ST deviation (mV)         : {scores['st_deviation_mv']} mV")
        print(f"  QRS complexity            : {scores['qrs_complexity']}")
        print(f"  Sample entropy            : {scores['sample_entropy']}")
        print(f"  Composite score           : {scores['composite_score']}")
        print()

    # The key check
    st_100 = results['100']['st_deviation_mv']
    st_207 = results['207']['st_deviation_mv']
    comp_100 = results['100']['composite_score']
    comp_207 = results['207']['composite_score']

    print("SANITY CHECK RESULTS:")
    st_ok   = st_207 >= st_100
    comp_ok = comp_207 >= comp_100
    print(f"  ST(207) >= ST(100)         : {'✓ PASS' if st_ok   else '✗ FAIL'}")
    print(f"  Composite(207) >= Comp(100): {'✓ PASS' if comp_ok else '✗ FAIL'}")

    if st_ok and comp_ok:
        print("\n✓ Sanity check passed. Running full pipeline is safe.")
    else:
        print("\n✗ Sanity check failed. Do NOT run full pipeline yet.")
        print("  Share these numbers and we will diagnose further.")

    return st_ok and comp_ok


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline():
    print("=" * 60)
    print("ECG Task Profiling Pipeline v3")
    print("MIT-BIH Arrhythmia Database → iFogSim2 Task Parameters")
    print("=" * 60)

    if not os.path.exists(DATA_DIR):
        print(f"[ERROR] '{DATA_DIR}' not found.")
        return

    all_tasks = []
    print(f"\nProcessing {len(RECORD_IDS)} records...")
    for record_id in RECORD_IDS:
        tasks = process_record(record_id)
        all_tasks.extend(tasks)

    if not all_tasks:
        print("[ERROR] No tasks generated.")
        return

    df = pd.DataFrame(all_tasks)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE — Summary")
    print("=" * 60)
    print(f"Total task windows  : {len(df)}")
    print(f"Records processed   : {df['record_id'].nunique()}")
    print(f"Output saved to     : {OUTPUT_CSV}")

    print("\nTask Class Distribution:")
    labels = {0:'Simple (Edge)', 1:'Moderate (Edge/Fog)',
              2:'Complex (Cloud)', 3:'Critical (Cloud Forced)'}
    for cls, label in labels.items():
        count = len(df[df['task_class'] == cls])
        pct   = 100 * count / len(df)
        print(f"  Class {cls} — {label:25s}: {count:5d} ({pct:.1f}%)")

    print("\nMI Statistics:")
    print(f"  Min : {df['MI'].min()}")
    print(f"  Max : {df['MI'].max()}")
    print(f"  Mean: {df['MI'].mean():.0f}")
    print(f"  Std : {df['MI'].std():.0f}")

    print("\nComplexity Score Statistics:")
    for col in ['composite_score','sample_entropy',
                'qrs_complexity','variance_score',
                'st_deviation','st_deviation_mv']:
        print(f"  Mean {col:20s}: {df[col].mean():.4f}")

    print("\nST Deviation (mV) Distribution:")
    print(f"  < 0.1mV (normal)     : "
          f"{len(df[df['st_deviation_mv'] < 0.1]):5d} "
          f"({100*len(df[df['st_deviation_mv'] < 0.1])/len(df):.1f}%)")
    print(f"  0.1-0.2mV (borderline): "
          f"{len(df[(df['st_deviation_mv'] >= 0.1) & (df['st_deviation_mv'] < 0.2)]):5d} "
          f"({100*len(df[(df['st_deviation_mv'] >= 0.1) & (df['st_deviation_mv'] < 0.2)])/len(df):.1f}%)")
    print(f"  > 0.2mV (critical)   : "
          f"{len(df[df['st_deviation_mv'] > 0.2]):5d} "
          f"({100*len(df[df['st_deviation_mv'] > 0.2])/len(df):.1f}%)")

    print("\n[NEXT STEP] Run validate_scores.py for Otsu thresholds")
    print("=" * 60)
    return df


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'check':
        sanity_check()
    else:
        # Always run sanity check first
        if sanity_check():
            print()
            run_pipeline()
        else:
            print("Fix the issue before running full pipeline.")
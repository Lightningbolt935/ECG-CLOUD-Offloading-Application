# ECG Cloud Offloading Application

A comprehensive framework for profiling, clustering, and scheduling electrocardiogram (ECG) processing tasks across Edge, Fog, and Cloud computing environments. This project simulates an intelligent healthcare IoT ecosystem where ECG signals are analyzed for complexity and offloaded to appropriate computing tiers using advanced optimization algorithms.

## Overview

Continuous ECG monitoring generates massive amounts of data that can overwhelm edge devices (e.g., wearable monitors). This application provides a multi-stage pipeline to:
1. **Profile** ECG signals to assess their computational complexity and clinical criticality.
2. **Cluster** similar tasks to optimize resource allocation.
3. **Schedule and Allocate** tasks using multi-objective optimization (NSGA-II) and hybrid swarm intelligence (QSO-POA) algorithms to minimize latency, energy consumption, and cost while maximizing throughput.

## System Architecture

The pipeline consists of four main modules:

### 1. ECG Task Profiling (Day 1)
- **File**: `ecg_pipeline_day1_v4.py`
- Analyzes MIT-BIH Arrhythmia Database (`mitdb`) signals using a sliding window approach.
- Computes complexity metrics including Sample Entropy, QRS Complexity, Signal Variance, and ST-segment Deviation.
- Assigns tasks to four classes (Simple, Moderate, Complex, Critical) using Otsu-derived thresholds.
- Maps classes to appropriate offloading targets: Edge, Fog, or Cloud.

### 2. Task Clustering (Day 2)
- **File**: `ecg_clustering_day2.py`
- Groups tasks with similar computational profiles (MI, RAM, Bandwidth) using machine learning clustering techniques.
- Reduces the scheduling search space and improves allocation efficiency.

### 3. Multi-Objective Scheduling (Day 3)
- **File**: `nsga2_scheduling_day3.py`
- Utilizes the Non-dominated Sorting Genetic Algorithm II (NSGA-II).
- Balances trade-offs between execution time, energy consumption, and network usage.

### 4. Advanced Allocation (Day 4)
- **Files**: `qso_poa_day4.py` / `QSOPOAAllocator.java` / `ECGOffloadingApplication.java`
- Implements a hybrid Quantum Swarm Optimization (QSO) and Pelican Optimization Algorithm (POA).
- Dynamically assigns grouped tasks to specific virtual machines (VMs) or edge nodes.

## Prerequisites

- **Python 3.8+**
  - `numpy`, `pandas`, `scipy`, `wfdb`
- **Java 11+** (for the Java-based simulation components)

## Dataset

This project utilizes the **MIT-BIH Arrhythmia Database**. The raw data files (`.dat`, `.hea`, `.atr`) should be placed in the `mitdb/` directory.

## Usage

Run the pipeline sequentially:

1. **Profile the data:**
   ```bash
   python ecg_pipeline_day1_v4.py
   ```
   *Generates `task_profiles.csv`.*

2. **Cluster the tasks:**
   ```bash
   python ecg_clustering_day2.py
   ```

3. **Run NSGA-II Scheduling:**
   ```bash
   python nsga2_scheduling_day3.py
   ```

4. **Execute the QoS-POA Allocator:**
   ```bash
   python qso_poa_day4.py
   ```
   *(Alternatively, run the Java simulation using `ECGOffloadingApplication.java`)*

## Validation

- **File**: `validate_scores.py`
- Validates the statistical significance of the computed complexity scores against clinical annotations.

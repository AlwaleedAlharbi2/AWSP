# AWSP++: ILP-Grounded Adaptive Weighted Sampling for VM Placement

**Paper**: *AWSP++: ILP-Grounded Adaptive Weighted Sampling for Fast, Energy-Aware, Fragmentation-Minimizing Virtual Machine Placement in Heterogeneous Cloud Data Centers*

**Venue**: IEEE Transactions (2026)

---

## Abstract

AWSP++ is an online VM placement framework whose decision rule is derived from a 0/1 ILP minimising a composite objective of host activation cost and weighted residual slack under per-resource headroom. For each arriving VM, it samples *k* eligible hosts using a bucketised power-of-*k* scheme, computes an ILP-aligned slack score per candidate, and assigns the VM to the lowest-scoring host. Two mechanisms drive the gain: a dominant-resource bucket index (M1) and a reactive rebalancing gate (M4). AWSP++ achieves O(k|R|) per-decision cost, independent of host count.

---

## Key Results (Table V)

| Method | Active Hosts | Energy (kWh) | SLA (%) | t_d (ms) |
|--------|-------------|-------------|---------|----------|
| PABFD [15] | 1842 | 472.6 | 2.41 | 1.84 |
| GRVMP [17] | 1731 | 451.2 | 1.92 | 0.92 |
| LBMBP [14] | 1689 | 442.4 | 2.18 | 1.42 |
| Entropy-VMP [24] | 1702 | 446.9 | 2.05 | 1.61 |
| HUNTER [33] | 1654 | 438.1 | 1.71 | 7.86 |
| K-RL-VMP [25] | 1671 | 440.7 | 1.84 | 3.24 |
| **AWSP++ (ours)** | **1502** | **402.8** | **1.18** | **0.39** |

Results on Google Cluster v3. AWSP++ reduces active hosts by 18.42%, energy by 21.07%, and SLA violations by 39.84% vs. best baselines.

---

## Architecture Overview

```
VM arrival stream
      |
      v
+------------------+      +------------------+
| M1: Bucket Index |----> | M2: Power-of-k   |
| (dominant        |      | Sampler          |
|  resource, band) |      | k candidates     |
+------------------+      +------------------+
                                    |
                                    v
                          +------------------+
                          | M3: ILP-Aligned  |
                          | Scoring Module   |
                          | score(j,v) eq.24 |
                          +------------------+
                                    |
                                    v
                          +------------------+
                          | M4: Reactive     |  <-- bucket feedback
                          | Rebalancer       |
                          | gate eq. 33      |
                          +------------------+
                                    |
                                    v
                           Placement Decision
                           O(k|R|) per VM
```

---

## Datasets

| Dataset | VMs | Resources | Window | Download |
|---------|-----|-----------|--------|----------|
| Google Cluster Trace v3 | 1,287,453 | CPU, RAM | 24h | [github.com/google/cluster-data](https://github.com/google/cluster-data) |
| Azure Packing 2020 | 692,144 | CPU, RAM | 14d | [github.com/Azure/AzurePublicDataset](https://github.com/Azure/AzurePublicDataset) |
| Bitbrains GWA-T-12 | 1,750 | CPU, RAM, BW | 30d | [atlarge-research.com/gwa-t-12](https://atlarge-research.com/gwa-t-12/) |

### Data Placement

```
data/
  google_cluster/
    task_events.csv         # from gsutil cp gs://clusterdata-2019-a
    machine_events.csv
  azure_packing/
    vmtable.csv.gz          # AzureTracesForPacking2020
  bitbrains/
    faststorage/            # GWA-T-12 FastStorage workload
    rnd/                    # GWA-T-12 Rnd workload
```

All scripts fall back to a **synthetic trace** if real data is absent.

---

## Installation

```bash
git clone https://github.com/your-org/AWSP_Implementation.git
cd AWSP_Implementation
pip install -r requirements.txt
```

Requires Python 3.10+.

---

## How to Run

### Download / generate data

```bash
# Generate synthetic traces (fast, no download needed)
python data/download_data.py --synthetic

# Optional: download Bitbrains (public)
python data/download_data.py --bitbrains
```

### Train (run simulation)

```bash
# Quick run with synthetic data (Google trace, 200 hosts, 10k VMs)
python train.py --use_synthetic --n_hosts 200 --n_vms 10000

# Full experiment matching paper (5 seeds, 3 datasets)
python train.py --dataset google  --n_hosts 1000 --n_vms 50000
python train.py --dataset azure   --n_hosts 1000 --n_vms 50000
python train.py --dataset bitbrains --n_hosts 100 --n_vms 5000

# k-sensitivity sweep (Fig. 8)
python train.py --k_sweep --use_synthetic
```

### Evaluate

```bash
# Evaluate on one dataset with result figures
python evaluate.py --dataset google --figures

# Evaluate on all three datasets (reproduces Table V)
python evaluate.py --all_datasets --figures
```

### Inference (single VM)

```bash
# Place a single VM: CPU=0.3, RAM=0.4, BW=0.05
python inference.py --cpu 0.3 --ram 0.4 --bw 0.05

# Interactive mode
python inference.py --interactive

# Batch from JSON
python inference.py --request_file my_vms.json
```

### Notebook

```bash
cd notebooks
jupyter notebook demo.ipynb
```

---

## Hyperparameters (Table IV)

| Parameter | Value |
|-----------|-------|
| Simulator | CloudSim 7 (Java 17) |
| Host CPU | Intel Xeon Gold 6242 (16C, 2.8GHz) |
| Host RAM | 192 GB DDR4 |
| Host BW | 25 Gb/s |
| P_idle / P_max | 86 W / 287 W |
| Sampling k | 4 (default) |
| Headroom theta_r | 0.80 (default) |
| (w_CPU, w_RAM, w_BW) | (0.4, 0.4, 0.2) |
| (alpha, beta) | (1.0, 0.5) |
| lambda (SLA weight) | 1.0 |
| Random seeds | {17, 42, 73, 128, 257} |

---

## Folder Structure

```
AWSP_Implementation/
|
+-- figures/                    All 14 paper figures + 6 table images (HD)
|     +-- fig01_motivation_scalar_vs_shape.png
|     +-- fig02_system_overview_awsp.png
|     +-- fig03_complete_architecture_awsp.png
|     +-- ...
|     +-- table01_comparative_analysis.png
|     +-- ...
|
+-- data/                       Dataset folder (place real traces here)
|     +-- download_data.py      Download / synthetic generation script
|
+-- checkpoints/                Saved model state (after train.py)
+-- results/                    Output CSVs, plots, inference results
|
+-- notebooks/
|     +-- demo.ipynb            Step-by-step Jupyter walkthrough
|
+-- config.py                   All hyperparameters (Table IV)
+-- dataset.py                  Dataset loaders + synthetic generator
+-- model.py                    AWSP++ (M1-M4) + all 6 baselines
+-- train.py                    Simulation runner + logging
+-- evaluate.py                 Test-set evaluation + figure generation
+-- inference.py                Single-VM placement + visualisation
+-- utils.py                    Metrics, stats, plotting helpers
+-- requirements.txt            Python dependencies
+-- README.md
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{awsppp2026,
  title     = {AWSP++: ILP-Grounded Adaptive Weighted Sampling for Fast,
               Energy-Aware, Fragmentation-Minimizing Virtual Machine Placement
               in Heterogeneous Cloud Data Centers},
  year      = {2026},
}
```

---

## Reproducibility

All results in the paper can be reproduced by:

```bash
# Reproduce Table V (requires real traces or use --use_synthetic)
python train.py --dataset google   --n_hosts 2000 --n_vms 100000
python train.py --dataset azure    --n_hosts 2000 --n_vms 100000
python train.py --dataset bitbrains --n_hosts 100 --n_vms 5000
python evaluate.py --all_datasets --figures

# Reproduce Fig. 8 (k sensitivity)
python train.py --k_sweep --dataset bitbrains

# Reproduce Fig. 10 (scalability)
python evaluate.py --figures
```

Five random seeds `{17, 42, 73, 128, 257}` are used for all experiments.

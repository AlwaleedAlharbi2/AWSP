"""
config.py - AWSP++ Configuration
Paper: "AWSP++: ILP-Grounded Adaptive Weighted Sampling for Fast, Energy-Aware,
        Fragmentation-Minimizing Virtual Machine Placement in Heterogeneous Cloud Data Centers"

All hyperparameters and settings taken directly from the paper.
Section references are noted inline.
"""

import os

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
LOGS_DIR = os.path.join(BASE_DIR, "results", "logs")

# ============================================================
# DATASET SETTINGS  (Section VI-A)
# ============================================================
DATASETS = {
    "google": {
        "name": "Google Cluster Trace v3",
        "url": "https://github.com/google/cluster-data",
        "local_path": os.path.join(DATA_DIR, "google_cluster"),
        "vms": 1_287_453,
        "resources": ["cpu", "ram"],
        "window_hours": 24,
        "arrival_pattern": "diurnal_bursty",
    },
    "azure": {
        "name": "Azure Packing 2020 (AzureTracesForPacking2020)",
        "url": "https://github.com/Azure/AzurePublicDataset/blob/master/AzureTracesForPacking2020.md",
        "local_path": os.path.join(DATA_DIR, "azure_packing"),
        "vms": 692_144,
        "resources": ["cpu", "ram"],
        "window_days": 14,
        "arrival_pattern": "diurnal",
    },
    "bitbrains": {
        "name": "Bitbrains GWA-T-12",
        "url": "https://atlarge-research.com/gwa-t-12/",
        "local_path": os.path.join(DATA_DIR, "bitbrains"),
        "vms": 1_750,
        "resources": ["cpu", "ram", "bw"],
        "window_days": 30,
        "arrival_pattern": "enterprise_quarterly",
    },
}

# Default dataset for experiments
DEFAULT_DATASET = "google"

# ============================================================
# RESOURCE DIMENSIONS  (Section III-A, Table II)
# ============================================================
RESOURCES = ["cpu", "ram", "bw"]            # R - set of resource dimensions

# Resource weights  w_r  (Section III-D, eq. 7)
# Paper uses: (w_CPU, w_RAM, w_BW) = (0.4, 0.4, 0.2)  (Table IV)
RESOURCE_WEIGHTS = {
    "cpu": 0.4,
    "ram": 0.4,
    "bw":  0.2,
}

# Headroom factor  theta_r  (Section III-C, eq. 3)
# Default 0.80; swept over {0.6, 0.7, 0.8, 0.9, 1.0} in sensitivity analysis
HEADROOM = {
    "cpu": 0.80,
    "ram": 0.80,
    "bw":  0.80,
}
HEADROOM_SWEEP = [0.6, 0.7, 0.8, 0.9, 1.0]

# ============================================================
# HOST / DATA-CENTRE SETTINGS  (Table IV, Section VI-C)
# ============================================================
HOST_CPU_CORES = 16                     # Intel Xeon Gold 6242 (2.8 GHz, 16C)
HOST_RAM_GB = 192                       # 192 GB DDR4
HOST_BW_GBPS = 25                       # 25 Gb/s

# Power model constants (eq. 1)  - linear model of Beloglazov & Buyya [15]
P_IDLE_WATTS = 86.0                     # Idle power
P_MAX_WATTS = 287.0                     # Peak power

# Simulated host count sweep for scalability tests (Section VI-J, Fig. 10)
HOST_COUNT_SWEEP = [100, 300, 1000, 3000, 10_000]
DEFAULT_HOST_COUNT = 1_000

# ============================================================
# AWSP++ ALGORITHM HYPERPARAMETERS  (Section IV)
# ============================================================

# ILP objective weights  alpha, beta  (eq. 7)
# alpha = weight on host activation cost
# beta  = weight on fragmentation (weighted slack)
ALPHA = 1.0         # activation cost weight
BETA  = 0.5         # slack weight

# ILP weighting coefficient  lambda  (eq. 32) - SLA weight
LAMBDA_SLA = 1.0

# Sampling parameter  k  (Section IV-B, eq. 18)
# Default k=4; swept over {1, 2, 4, 8, 16} in sensitivity analysis (Section VI-J)
K_SAMPLE = 4
K_SWEEP = [1, 2, 4, 8, 16]

# Bucket bands B for dominant-resource discretisation (Section IV-A, eq. 13)
# B=4: low, medium-low, medium-high, high
NUM_BANDS = 4

# ============================================================
# REACTIVE REBALANCING SETTINGS  (Section V, eq. 27-28)
# ============================================================

# Overload threshold: rho_jr > theta_r - epsilon_up
EPSILON_UP = 0.05

# Underload threshold: sum_r w_r * rho_jr < tau_down
TAU_DOWN = 0.20

# VM migration cost parameters (Section V-A, eq. 30)
GAMMA_PER_MB = 0.001        # per-MB transfer energy (kWh/MB, illustrative)
DELTA_PER_SEC = 0.0001      # per-second network overhead (kWh/s, illustrative)
ETA_VM = 0.05               # per-migration SLA degradation factor (eq. 29)
L_MIN_FRACTION = 0.10       # minimum remaining lifetime fraction before a VM is migration-eligible

# ============================================================
# SIMULATION SETTINGS  (Table IV)
# ============================================================
RANDOM_SEEDS = [17, 42, 73, 128, 257]          # five seeds for statistical validity
DEFAULT_SEED = 42
SIMULATION_TICK_MS = 1.0                        # utilisation-update tick interval

# ============================================================
# BASELINES  (Section VI-D)
# ============================================================
BASELINES = ["PABFD", "GRVMP", "LBMBP", "Entropy-VMP", "HUNTER", "K-RL-VMP"]

# ============================================================
# EVALUATION METRICS  (Section VI-B)
# ============================================================
METRICS = [
    "active_hosts",         # |H_act|
    "energy_kwh",           # E (kWh)
    "weighted_slack",       # sum_{j,r} w_r * s_jr
    "sla_violation_rate",   # fraction (%) of VMs that exceeded headroom
    "migration_count",      # number of migrations by Algorithm 3
    "decision_time_ms",     # per-VM decision latency (ms)
]

# ============================================================
# OUTPUT PATHS
# ============================================================
CHECKPOINT_PATH = os.path.join(CHECKPOINTS_DIR, "awsp_state.pkl")
RESULTS_CSV     = os.path.join(RESULTS_DIR, "results_table.csv")
LOG_PATH        = os.path.join(LOGS_DIR, "training.log")
CONFUSION_MATRIX_PATH = os.path.join(RESULTS_DIR, "confusion_matrix.png")
ROC_CURVE_PATH        = os.path.join(RESULTS_DIR, "roc_curve.png")
LATENCY_PLOT_PATH     = os.path.join(RESULTS_DIR, "fig_scalability.png")
ENERGY_PLOT_PATH      = os.path.join(RESULTS_DIR, "fig_energy_diurnal.png")
SLACK_HIST_PATH       = os.path.join(RESULTS_DIR, "fig_slack_histogram.png")
SLA_BURST_PATH        = os.path.join(RESULTS_DIR, "fig_sla_burst.png")
HEATMAP_PATH          = os.path.join(RESULTS_DIR, "fig_improvement_heatmap.png")

# ============================================================
# ENSURE DIRECTORIES EXIST
# ============================================================
for _d in [DATA_DIR, CHECKPOINTS_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR]:
    os.makedirs(_d, exist_ok=True)

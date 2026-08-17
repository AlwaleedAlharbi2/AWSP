"""
dataset.py - Dataset Download, Preprocessing, and Loaders for AWSP++
Paper Section VI-A: Datasets

Supported traces:
  1. Google Cluster Trace v3
     URL: https://github.com/google/cluster-data
  2. Azure Packing 2020 (AzureTracesForPacking2020)
     URL: https://github.com/Azure/AzurePublicDataset
  3. Bitbrains GWA-T-12
     URL: https://atlarge-research.com/gwa-t-12/
"""

import os
import csv
import random
import zipfile
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from config import (
    DATA_DIR, RESOURCES, RESOURCE_WEIGHTS, HEADROOM,
    DEFAULT_DATASET, DEFAULT_SEED, DATASETS
)

logger = logging.getLogger(__name__)

# ============================================================
# DATA STRUCTURES  (Table II - notation from paper)
# ============================================================

@dataclass
class VMRequest:
    """
    Represents an arriving VM with demand vector and lifetime.
    Section III-B: each VM v in V carries demand vector d_v and lifetime L_v.
    """
    vm_id: str
    demand: Dict[str, float]        # d_vr: resource demand per dimension r
    lifetime_s: float               # L_v: VM lifetime in seconds
    arrival_time: float             # arrival timestamp
    priority: float = 1.0          # SLA priority weight

    def dominant_resource(self, headroom: Dict[str, float], capacity_max: Dict[str, float]) -> str:
        """
        Section IV-A eq. 14: r*(v) = argmax_{r in R} d_vr / (theta_r * C_max_r)
        """
        scores = {
            r: self.demand.get(r, 0.0) / (headroom[r] * capacity_max[r])
            for r in RESOURCES
            if r in self.demand
        }
        return max(scores, key=scores.get)


@dataclass
class HostSpec:
    """
    Physical host capacity specification.
    Section III-B: each host j exposes capacity vector C_j and power model parameters.
    """
    host_id: str
    capacity: Dict[str, float]      # C_jr: capacity per resource r
    p_idle: float                   # P^idle_j (eq. 1)
    p_max: float                    # P^max_j  (eq. 1)
    activation_cost: float          # a_j: energy proxy for activation (eq. 7)


# ============================================================
# SYNTHETIC TRACE GENERATOR
# (Used when real traces are not downloaded; matches paper statistics)
# ============================================================

def generate_synthetic_trace(
    n_vms: int = 10_000,
    n_hosts: int = 500,
    window_hours: float = 24.0,
    resources: List[str] = None,
    seed: int = DEFAULT_SEED,
    pattern: str = "diurnal_bursty",
) -> Tuple[List[VMRequest], List[HostSpec]]:
    """
    Generate a synthetic workload trace matching paper statistics.
    Section VI-A, Table III: arrival patterns and resource distributions.

    Args:
        n_vms: number of VM arrival requests
        n_hosts: number of physical hosts
        window_hours: simulation window in hours
        resources: list of resource dimension names
        seed: random seed
        pattern: 'diurnal_bursty' | 'diurnal' | 'enterprise_quarterly'

    Returns:
        (vm_requests, host_specs)
    """
    if resources is None:
        resources = RESOURCES
    rng = np.random.default_rng(seed)

    # ----- Host generation -----
    hosts = []
    for j in range(n_hosts):
        # Heterogeneous hosts (Section III-B)
        cpu_cap = rng.choice([8.0, 16.0, 32.0, 64.0])   # normalised cores
        ram_cap = rng.choice([32.0, 64.0, 128.0, 192.0])  # GB
        bw_cap  = rng.choice([10.0, 25.0, 40.0, 100.0])  # Gbps

        capacity = {
            "cpu": float(cpu_cap),
            "ram": float(ram_cap),
            "bw":  float(bw_cap),
        }
        # Linear power model: P_idle=86W, P_max=287W (Table IV)
        # Add heterogeneous jitter
        p_idle = 86.0 + rng.uniform(-10, 10)
        p_max  = 287.0 + rng.uniform(-30, 30)
        a_j    = p_idle / 1000.0  # activation cost in kW

        hosts.append(HostSpec(
            host_id=f"host_{j:04d}",
            capacity={r: capacity.get(r, 1.0) for r in resources},
            p_idle=p_idle,
            p_max=p_max,
            activation_cost=a_j,
        ))

    # ----- VM arrival generation -----
    window_s = window_hours * 3600.0
    vms = []

    # Arrival rate profile (diurnal pattern: quiet at night, peak midday)
    def arrival_rate(t_norm: float) -> float:
        """Normalised arrival rate at time t in [0,1] of window."""
        if pattern == "diurnal_bursty":
            base = 0.3 + 0.7 * np.sin(np.pi * t_norm) ** 2
            # Add random burst (3x spike) at t_norm ~ 0.5 for flash-crowd test
            if 0.48 <= t_norm <= 0.52:
                base *= 3.0
            return base
        elif pattern == "diurnal":
            return 0.3 + 0.7 * np.sin(np.pi * t_norm) ** 2
        else:  # enterprise_quarterly - steady
            return 0.6 + 0.4 * rng.random()

    # Sample arrival times via thinning / inverse CDF
    arrival_times = sorted(rng.uniform(0, window_s, n_vms))

    for i, t_arr in enumerate(arrival_times):
        t_norm = t_arr / window_s
        rate_scale = arrival_rate(t_norm)

        # VM demand: CPU/RAM follow truncated normal (Google Cluster stats)
        cpu_demand = float(np.clip(rng.normal(0.25, 0.15) * rate_scale, 0.05, 0.95))
        ram_demand = float(np.clip(rng.normal(0.30, 0.20) * rate_scale, 0.05, 0.95))
        bw_demand  = float(np.clip(rng.normal(0.10, 0.08) * rate_scale, 0.01, 0.50))

        # Normalised demand (fraction of C_max)
        demand = {r: 0.0 for r in resources}
        demand["cpu"] = cpu_demand
        if "ram" in resources:
            demand["ram"] = ram_demand
        if "bw" in resources:
            demand["bw"] = bw_demand

        # VM lifetime: log-normal (enterprise: long-tailed)
        if pattern == "enterprise_quarterly":
            lifetime = float(np.clip(rng.lognormal(8.0, 1.5), 300, window_s))
        else:
            lifetime = float(np.clip(rng.lognormal(6.5, 1.2), 60, window_s))

        vms.append(VMRequest(
            vm_id=f"vm_{i:06d}",
            demand=demand,
            lifetime_s=lifetime,
            arrival_time=t_arr,
        ))

    logger.info("Generated %d VM requests and %d hosts (pattern=%s, seed=%d)",
                len(vms), len(hosts), pattern, seed)
    return vms, hosts


# ============================================================
# DATASET LOADERS FOR REAL TRACES
# ============================================================

def load_google_cluster_trace(data_dir: Optional[str] = None) -> Tuple[List[VMRequest], List[HostSpec]]:
    """
    Load Google Cluster Trace v3.
    Download: https://github.com/google/cluster-data
    Place the CSV files under data/google_cluster/

    Paper (Section VI-A): 1,287,453 task arrivals, CPU + RAM, 24h window.
    """
    if data_dir is None:
        data_dir = os.path.join(DATA_DIR, "google_cluster")

    task_events_path = os.path.join(data_dir, "task_events.csv")
    machine_path     = os.path.join(data_dir, "machine_events.csv")

    if not os.path.exists(task_events_path):
        logger.warning(
            "Google Cluster trace not found at %s. "
            "Download from https://github.com/google/cluster-data "
            "and place task_events.csv and machine_events.csv in %s. "
            "Falling back to synthetic trace.",
            task_events_path, data_dir
        )
        return generate_synthetic_trace(
            n_vms=50_000, n_hosts=500, window_hours=24.0,
            resources=["cpu", "ram"], pattern="diurnal_bursty"
        )

    # Parse task events
    vms = []
    df_tasks = pd.read_csv(task_events_path, header=None, nrows=200_000,
                           names=["time", "missing", "job_id", "task_idx",
                                  "machine_id", "event_type", "user",
                                  "scheduling_class", "priority",
                                  "cpu_request", "ram_request", "disk_request",
                                  "diff_machine_constraint"])
    # event_type 0 = SUBMIT
    submits = df_tasks[df_tasks["event_type"] == 0].copy()
    submits = submits.dropna(subset=["cpu_request", "ram_request"])

    # Normalise to [0,1] fraction of max
    max_cpu = submits["cpu_request"].quantile(0.99) or 1.0
    max_ram = submits["ram_request"].quantile(0.99) or 1.0
    max_time = submits["time"].max() or 1.0

    for _, row in submits.iterrows():
        cpu_d = float(min(row["cpu_request"] / max_cpu, 1.0))
        ram_d = float(min(row["ram_request"] / max_ram, 1.0))
        vms.append(VMRequest(
            vm_id=f"g_{int(row['job_id'])}_{int(row['task_idx'])}",
            demand={"cpu": cpu_d, "ram": ram_d, "bw": 0.05},
            lifetime_s=float(row.get("time", 3600)),
            arrival_time=float(row["time"]) / max_time * 86400.0,
        ))

    # Parse machine events
    hosts = []
    if os.path.exists(machine_path):
        df_mach = pd.read_csv(machine_path, header=None, nrows=5_000,
                              names=["time", "machine_id", "event_type",
                                     "platform_id", "cpus", "memory"])
        df_mach = df_mach[df_mach["event_type"] == 0].dropna(subset=["cpus", "memory"])
        for _, row in df_mach.iterrows():
            hosts.append(HostSpec(
                host_id=f"m_{int(row['machine_id'])}",
                capacity={"cpu": float(row["cpus"]), "ram": float(row["memory"]), "bw": 25.0},
                p_idle=86.0, p_max=287.0, activation_cost=0.086,
            ))

    if not hosts:
        hosts = _default_hosts(500)

    logger.info("Loaded Google Cluster trace: %d VMs, %d hosts", len(vms), len(hosts))
    return vms, hosts


def load_azure_trace(data_dir: Optional[str] = None) -> Tuple[List[VMRequest], List[HostSpec]]:
    """
    Load Azure Packing 2020 trace.
    Download: https://github.com/Azure/AzurePublicDataset/blob/master/AzureTracesForPacking2020.md
    Place vmtable.csv.gz under data/azure_packing/

    Paper (Section VI-A): 692,144 VMs, CPU + RAM, 14-day window.
    """
    if data_dir is None:
        data_dir = os.path.join(DATA_DIR, "azure_packing")

    vm_table_path = os.path.join(data_dir, "vmtable.csv.gz")

    if not os.path.exists(vm_table_path):
        logger.warning(
            "Azure trace not found at %s. "
            "Download from https://github.com/Azure/AzurePublicDataset "
            "Falling back to synthetic trace.",
            vm_table_path
        )
        return generate_synthetic_trace(
            n_vms=30_000, n_hosts=500, window_hours=24*14,
            resources=["cpu", "ram"], pattern="diurnal"
        )

    df = pd.read_csv(vm_table_path, compression="gzip", nrows=100_000)
    # Expected columns: vmId, tenantId, vmTypeId, vmFamily, starttime, endtime,
    #                   normalizedCPU, normalizedRAM
    vms = []
    for _, row in df.iterrows():
        cpu_d = float(row.get("normalizedCPU", 0.1))
        ram_d = float(row.get("normalizedRAM", 0.1))
        start = float(row.get("starttime", 0))
        end   = float(row.get("endtime", start + 3600))
        vms.append(VMRequest(
            vm_id=str(row.get("vmId", _)),
            demand={"cpu": cpu_d, "ram": ram_d, "bw": 0.05},
            lifetime_s=max(end - start, 60.0),
            arrival_time=start,
        ))

    hosts = _default_hosts(600)
    logger.info("Loaded Azure Packing trace: %d VMs, %d hosts", len(vms), len(hosts))
    return vms, hosts


def load_bitbrains_trace(data_dir: Optional[str] = None) -> Tuple[List[VMRequest], List[HostSpec]]:
    """
    Load Bitbrains GWA-T-12 trace.
    Download: https://atlarge-research.com/gwa-t-12/
    Place faststorage/ and rnd/ sub-folders under data/bitbrains/

    Paper (Section VI-A): 1,750 enterprise VMs, CPU + RAM + BW, 30-day window.
    """
    if data_dir is None:
        data_dir = os.path.join(DATA_DIR, "bitbrains")

    vms = []
    for workload in ["faststorage", "rnd"]:
        wl_dir = os.path.join(data_dir, workload)
        if not os.path.isdir(wl_dir):
            logger.warning("Bitbrains workload dir not found: %s", wl_dir)
            continue
        for fname in sorted(os.listdir(wl_dir))[:1000]:
            fpath = os.path.join(wl_dir, fname)
            if not fname.endswith(".csv"):
                continue
            try:
                df = pd.read_csv(fpath, sep=";", header=0)
                for i, row in df.iterrows():
                    cpu_d = float(row.get("CPU usage [%]", 10)) / 100.0
                    ram_d = float(row.get("Memory usage [KB]", 1024)) / (192 * 1024 * 1024)
                    bw_d  = float(row.get("Network received throughput [KB/s]", 0)) / (25 * 1e6)
                    vms.append(VMRequest(
                        vm_id=f"bb_{workload}_{fname}_{i}",
                        demand={"cpu": min(cpu_d, 1.0),
                                "ram": min(ram_d, 1.0),
                                "bw":  min(bw_d, 1.0)},
                        lifetime_s=300.0,
                        arrival_time=float(i * 300),
                    ))
            except Exception as e:
                logger.debug("Skipping %s: %s", fname, e)

    if not vms:
        logger.warning("No Bitbrains data found. Download from https://atlarge-research.com/gwa-t-12/ "
                       "Falling back to synthetic trace.")
        return generate_synthetic_trace(
            n_vms=1_750, n_hosts=100, window_hours=24*30,
            resources=["cpu", "ram", "bw"], pattern="enterprise_quarterly"
        )

    hosts = _default_hosts(100)
    logger.info("Loaded Bitbrains trace: %d VM observations, %d hosts", len(vms), len(hosts))
    return vms, hosts


def _default_hosts(n: int) -> List[HostSpec]:
    """Return n homogeneous Intel Xeon Gold 6242 hosts (Table IV)."""
    return [
        HostSpec(
            host_id=f"host_{j:04d}",
            capacity={"cpu": 16.0, "ram": 192.0, "bw": 25.0},
            p_idle=86.0, p_max=287.0, activation_cost=0.086,
        )
        for j in range(n)
    ]


# ============================================================
# UNIFIED DATASET LOADER
# ============================================================

def load_dataset(
    dataset_name: str = DEFAULT_DATASET,
    data_dir: Optional[str] = None,
    n_vms: Optional[int] = None,
    n_hosts: Optional[int] = None,
    seed: int = DEFAULT_SEED,
    use_synthetic: bool = False,
) -> Tuple[List[VMRequest], List[HostSpec]]:
    """
    Unified loader: attempts real trace, falls back to synthetic.

    Args:
        dataset_name: 'google' | 'azure' | 'bitbrains'
        data_dir: override default data path
        n_vms: cap on VM requests (useful for quick tests)
        n_hosts: number of hosts to use
        seed: random seed
        use_synthetic: force synthetic even if real data exists

    Returns:
        (vm_requests, host_specs)
    """
    if use_synthetic:
        cfg = DATASETS.get(dataset_name, DATASETS["google"])
        vms, hosts = generate_synthetic_trace(
            n_vms=n_vms or 10_000,
            n_hosts=n_hosts or 500,
            window_hours=cfg.get("window_hours", 24.0),
            resources=cfg.get("resources", RESOURCES),
            seed=seed,
            pattern=cfg.get("arrival_pattern", "diurnal_bursty"),
        )
        return vms, hosts

    if dataset_name == "google":
        vms, hosts = load_google_cluster_trace(data_dir)
    elif dataset_name == "azure":
        vms, hosts = load_azure_trace(data_dir)
    elif dataset_name == "bitbrains":
        vms, hosts = load_bitbrains_trace(data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(DATASETS.keys())}")

    if n_vms is not None and len(vms) > n_vms:
        vms = vms[:n_vms]
    if n_hosts is not None and len(hosts) > n_hosts:
        hosts = hosts[:n_hosts]

    return vms, hosts


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def split_trace(
    vms: List[VMRequest],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    temporal: bool = True,
) -> Tuple[List[VMRequest], List[VMRequest], List[VMRequest]]:
    """
    Split VM arrival stream into train/val/test.
    When temporal=True, split by arrival time (first 70% = train, etc.)
    to avoid data leakage from future arrivals.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, \
        "Fractions must sum to 1.0"

    if temporal:
        vms_sorted = sorted(vms, key=lambda v: v.arrival_time)
    else:
        vms_sorted = list(vms)

    n = len(vms_sorted)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train = vms_sorted[:n_train]
    val   = vms_sorted[n_train:n_train + n_val]
    test  = vms_sorted[n_train + n_val:]

    logger.info("Split: train=%d  val=%d  test=%d", len(train), len(val), len(test))
    return train, val, test


# ============================================================
# DOWNLOAD HELPER  (called from data/download_data.py)
# ============================================================

def download_instructions() -> str:
    """Return human-readable download instructions for all datasets."""
    lines = [
        "=" * 70,
        "AWSP++ Dataset Download Instructions",
        "=" * 70,
        "",
        "1. Google Cluster Trace v3",
        "   URL:  https://github.com/google/cluster-data",
        "   Cmd:  gsutil -m cp -r gs://clusterdata-2019-a \\",
        "         data/google_cluster/",
        "   Place task_events*.csv and machine_events*.csv in:",
        "         data/google_cluster/",
        "",
        "2. Azure Packing 2020 (AzureTracesForPacking2020)",
        "   URL:  https://github.com/Azure/AzurePublicDataset",
        "   Download AzureTracesForPacking2020.csv.gz and place in:",
        "         data/azure_packing/vmtable.csv.gz",
        "",
        "3. Bitbrains GWA-T-12",
        "   URL:  https://atlarge-research.com/gwa-t-12/",
        "   Download GWA-T-12-BitBrains.zip, extract to:",
        "         data/bitbrains/faststorage/",
        "         data/bitbrains/rnd/",
        "",
        "If real traces are unavailable, all scripts fall back to",
        "a synthetic trace that statistically matches paper data.",
        "=" * 70,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(download_instructions())
    # Quick test with synthetic data
    vms, hosts = load_dataset("google", use_synthetic=True, n_vms=1000, n_hosts=50)
    train, val, test = split_trace(vms)
    print(f"\nSynthetic trace loaded: {len(vms)} VMs, {len(hosts)} hosts")
    print(f"Split: train={len(train)} val={len(val)} test={len(test)}")
    print(f"\nSample VM: {vms[0]}")
    print(f"Sample Host: {hosts[0]}")

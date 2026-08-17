"""
train.py - AWSP++ Simulation Runner (Training / Warm-up Loop)
Paper Section VI-C: Implementation and Deployment

Runs the full simulation on all three datasets across all random seeds.
Logs per-tick metrics, saves checkpoints, and writes TensorBoard summaries.

Usage:
    python train.py --dataset google --n_hosts 500 --n_vms 10000
    python train.py --dataset azure  --use_synthetic
    python train.py --dataset bitbrains --k 8
"""

import os
import csv
import json
import time
import pickle
import logging
import argparse
import random
import numpy as np
from typing import Dict, List
from pathlib import Path

from config import (
    DEFAULT_DATASET, DEFAULT_SEED, RANDOM_SEEDS,
    K_SAMPLE, K_SWEEP, HEADROOM, RESOURCES,
    CHECKPOINT_PATH, RESULTS_CSV, LOG_PATH, LOGS_DIR,
    RESULTS_DIR, DEFAULT_HOST_COUNT,
)
from dataset import load_dataset, split_trace
from model import AWSSPlusPlus, PABFDBaseline, GRVMPBaseline, get_baseline

# ============================================================
# LOGGING SETUP
# ============================================================
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="a"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("train")

# Optional TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    _tb_available = True
except ImportError:
    _tb_available = False
    logger.info("TensorBoard not available. Install torch to enable: pip install torch")


# ============================================================
# SIMULATION TICK LOOP
# ============================================================

def run_simulation(
    algorithm,              # AWSSPlusPlus or baseline instance
    vm_stream: list,        # ordered list of VMRequest
    tick_interval_s: float = 300.0,  # rebalancing check every 5 min
    window_hours: float = 24.0,
) -> Dict:
    """
    Run one complete simulation pass over a VM arrival stream.
    Returns a dict of per-tick and summary metrics.

    Section VI-C: 'a custom VmAllocationPolicy overrides allocateHostForVm()
    to invoke Algorithm 1 on every arrival; a ReactiveRebalancer class wraps
    Algorithm 3 and is invoked at every utilisation-update tick.'
    """
    metrics_log = []
    total_sla_violations = 0
    active_vm_set = {}          # vm_id -> (VMRequest, host_id, start_time)
    current_tick = 0
    next_tick_time = 0.0
    window_s = window_hours * 3600.0

    # Energy accumulator (Watts * seconds -> kWh)
    energy_joules = 0.0
    last_energy_time = 0.0

    for vm in vm_stream:
        t = vm.arrival_time

        # Rebalancing ticks  (Algorithm 3 triggered at each utilisation-update tick)
        while next_tick_time <= t and hasattr(algorithm, "rebalance"):
            if next_tick_time > 0:
                # Energy integral: P * delta_t  (eq. 9)
                dt = next_tick_time - last_energy_time
                energy_joules += algorithm.total_power_watts() * dt
                last_energy_time = next_tick_time

                # Expire VMs that have exceeded their lifetime
                to_expire = [
                    (vid, vhost_id, vstart)
                    for vid, (vobj, vhost_id, vstart) in list(active_vm_set.items())
                    if next_tick_time - vstart >= vobj.lifetime_s
                ]
                for vid, vhost_id, vstart in to_expire:
                    vobj, vhost_id, _ = active_vm_set.pop(vid)
                    if vhost_id and vhost_id in algorithm.host_states:
                        algorithm.remove_vm(vobj, algorithm.host_states[vhost_id])

                # Reactive rebalancing
                migrations = algorithm.rebalance(next_tick_time)
                for vmig, src, dst in migrations:
                    if vmig.vm_id in active_vm_set:
                        active_vm_set[vmig.vm_id] = (vmig, dst, active_vm_set[vmig.vm_id][2])

                # Log tick snapshot
                metrics_log.append({
                    "tick": current_tick,
                    "time_s": next_tick_time,
                    "active_hosts": algorithm.active_host_count(),
                    "power_w": algorithm.total_power_watts(),
                    "weighted_slack": algorithm.weighted_slack_total(),
                    "placements_total": algorithm.placements,
                    "migrations_total": algorithm.migrations,
                })
                current_tick += 1

            next_tick_time += tick_interval_s

        # Place arriving VM  (Algorithm 1)
        host_id = algorithm.place_vm(vm)

        if host_id is None:
            total_sla_violations += 1
        else:
            active_vm_set[vm.vm_id] = (vm, host_id, t)

    # Final energy computation
    dt = window_s - last_energy_time
    energy_joules += algorithm.total_power_watts() * dt
    energy_kwh = energy_joules / 3_600_000.0

    sla_rate = total_sla_violations / max(len(vm_stream), 1)

    summary = {
        "active_hosts":         algorithm.active_host_count(),
        "energy_kwh":           round(energy_kwh, 3),
        "weighted_slack":       round(algorithm.weighted_slack_total(), 4),
        "sla_violation_rate":   round(sla_rate * 100, 3),
        "migration_count":      algorithm.migrations,
        "avg_decision_time_ms": round(algorithm.avg_decision_time_ms(), 4),
        "placements":           algorithm.placements,
        "rejections":           algorithm.rejections,
        "n_vms":                len(vm_stream),
    }
    return summary, metrics_log


# ============================================================
# MAIN TRAINING / EXPERIMENT RUNNER
# ============================================================

def run_experiment(args) -> None:
    """
    Run full AWSP++ experiment:
      1. Load dataset (or use synthetic)
      2. Run AWSP++ + all baselines across RANDOM_SEEDS
      3. Log metrics per seed
      4. Save aggregate results to CSV
      5. Save checkpoint
    """
    logger.info("=" * 65)
    logger.info("AWSP++ Simulation  dataset=%s  k=%d  n_hosts=%d  n_vms=%d",
                args.dataset, args.k, args.n_hosts, args.n_vms)
    logger.info("=" * 65)

    # TensorBoard writer
    tb_writer = None
    if _tb_available:
        tb_dir = os.path.join(LOGS_DIR, f"tb_{args.dataset}")
        tb_writer = SummaryWriter(tb_dir)
        logger.info("TensorBoard logs: %s", tb_dir)

    all_results = []

    for seed_idx, seed in enumerate(RANDOM_SEEDS):
        logger.info("\n--- Seed %d / %d  (seed=%d) ---", seed_idx + 1, len(RANDOM_SEEDS), seed)
        random.seed(seed)
        np.random.seed(seed)

        # Load dataset
        vms, hosts = load_dataset(
            dataset_name=args.dataset,
            n_vms=args.n_vms,
            n_hosts=args.n_hosts,
            seed=seed,
            use_synthetic=args.use_synthetic,
        )
        train_vms, val_vms, test_vms = split_trace(vms, train_frac=0.70, val_frac=0.15, test_frac=0.15)

        tick_interval = 300.0  # 5-minute ticks
        window_hours  = 24.0

        # --- AWSP++ ---
        model = AWSSPlusPlus(hosts=hosts, k=args.k, seed=seed)
        awsp_summary, awsp_log = run_simulation(
            model, train_vms + val_vms, tick_interval, window_hours
        )
        awsp_summary["algorithm"] = "AWSP++"
        awsp_summary["seed"] = seed
        awsp_summary["dataset"] = args.dataset
        all_results.append(awsp_summary)
        logger.info("AWSP++:     active_hosts=%d  energy=%.2f kWh  slack=%.4f  SLA=%.2f%%  td=%.4f ms",
                    awsp_summary["active_hosts"], awsp_summary["energy_kwh"],
                    awsp_summary["weighted_slack"], awsp_summary["sla_violation_rate"],
                    awsp_summary["avg_decision_time_ms"])

        # TensorBoard
        if tb_writer:
            for i, log_entry in enumerate(awsp_log):
                step = seed_idx * len(awsp_log) + i
                tb_writer.add_scalar("AWSP++/active_hosts",   log_entry["active_hosts"],   step)
                tb_writer.add_scalar("AWSP++/power_w",        log_entry["power_w"],         step)
                tb_writer.add_scalar("AWSP++/weighted_slack", log_entry["weighted_slack"],  step)

        # --- Baselines ---
        for baseline_name in ["PABFD", "GRVMP"]:
            baseline = get_baseline(baseline_name, hosts, seed)
            bl_summary, _ = run_simulation(
                baseline, train_vms + val_vms, tick_interval, window_hours
            )
            bl_summary["algorithm"] = baseline_name
            bl_summary["seed"] = seed
            bl_summary["dataset"] = args.dataset
            all_results.append(bl_summary)
            logger.info("%s:      active_hosts=%d  energy=%.2f kWh  slack=%.4f  SLA=%.2f%%  td=%.4f ms",
                        baseline_name.ljust(12),
                        bl_summary["active_hosts"], bl_summary["energy_kwh"],
                        bl_summary["weighted_slack"], bl_summary["sla_violation_rate"],
                        bl_summary["avg_decision_time_ms"])

        # Save best model checkpoint after each seed
        checkpoint = {
            "seed": seed,
            "dataset": args.dataset,
            "k": args.k,
            "n_hosts": args.n_hosts,
            "summary": awsp_summary,
            "hyperparams": {
                "alpha": 0.5,
                "beta":  0.5,
                "headroom": HEADROOM,
                "resource_weights": {r: 1.0/len(RESOURCES) for r in RESOURCES},
            },
        }
        with open(CHECKPOINT_PATH, "wb") as f:
            pickle.dump(checkpoint, f)
        logger.info("Checkpoint saved: %s", CHECKPOINT_PATH)

    # --- Save aggregate results ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fieldnames = list(all_results[0].keys())
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    logger.info("Results saved to %s", RESULTS_CSV)

    # Print summary table
    print("\n" + "=" * 75)
    print(f"{'Algorithm':<18} {'Active Hosts':>12} {'Energy(kWh)':>12} "
          f"{'WSlack':>10} {'SLA%':>8} {'td(ms)':>10}")
    print("-" * 75)
    from itertools import groupby
    from statistics import mean

    for alg_name, rows in groupby(sorted(all_results, key=lambda x: x["algorithm"]),
                                   key=lambda x: x["algorithm"]):
        rows = list(rows)
        print(f"{alg_name:<18} "
              f"{mean(r['active_hosts'] for r in rows):>12.0f} "
              f"{mean(r['energy_kwh'] for r in rows):>12.2f} "
              f"{mean(r['weighted_slack'] for r in rows):>10.4f} "
              f"{mean(r['sla_violation_rate'] for r in rows):>8.3f} "
              f"{mean(r['avg_decision_time_ms'] for r in rows):>10.4f}")
    print("=" * 75)

    if tb_writer:
        tb_writer.close()

    logger.info("Experiment complete.")


# ============================================================
# K SENSITIVITY SWEEP  (Section VI-J, Fig. 8)
# ============================================================

def run_k_sweep(args) -> None:
    """
    Sensitivity analysis over sampling parameter k.
    Reproduces Fig. 8 from the paper.
    """
    logger.info("Running k-sensitivity sweep: k in %s", K_SWEEP)
    vms, hosts = load_dataset(args.dataset, n_vms=args.n_vms, n_hosts=args.n_hosts,
                              seed=DEFAULT_SEED, use_synthetic=args.use_synthetic)

    results = []
    for k in K_SWEEP:
        model = AWSSPlusPlus(hosts=hosts, k=k, seed=DEFAULT_SEED)
        summary, _ = run_simulation(model, vms, window_hours=24.0)
        results.append({
            "k": k,
            "energy_kwh": summary["energy_kwh"],
            "avg_decision_time_ms": summary["avg_decision_time_ms"],
            "sla_violation_rate": summary["sla_violation_rate"],
        })
        logger.info("k=%2d  energy=%.2f kWh  td=%.4f ms  SLA=%.3f%%",
                    k, summary["energy_kwh"],
                    summary["avg_decision_time_ms"],
                    summary["sla_violation_rate"])

    out_path = os.path.join(RESULTS_DIR, "k_sweep.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("k-sweep results saved to %s", out_path)


# ============================================================
# ENTRYPOINT
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="AWSP++ VM Placement Simulation Runner"
    )
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        choices=["google", "azure", "bitbrains"],
                        help="Dataset to use (default: google)")
    parser.add_argument("--n_hosts", type=int, default=DEFAULT_HOST_COUNT,
                        help="Number of physical hosts (default: 1000)")
    parser.add_argument("--n_vms", type=int, default=20_000,
                        help="Number of VM arrivals to simulate (default: 20000)")
    parser.add_argument("--k", type=int, default=K_SAMPLE,
                        help="Sampling parameter k (default: 4)")
    parser.add_argument("--seeds", type=int, nargs="+", default=RANDOM_SEEDS,
                        help="Random seeds for reproducibility")
    parser.add_argument("--use_synthetic", action="store_true",
                        help="Use synthetic trace even if real data exists")
    parser.add_argument("--k_sweep", action="store_true",
                        help="Run k-sensitivity analysis (reproduces Fig. 8)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.k_sweep:
        run_k_sweep(args)
    else:
        run_experiment(args)

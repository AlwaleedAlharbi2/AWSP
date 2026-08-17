"""
evaluate.py - AWSP++ Evaluation on Test Set
Paper Section VI: Experimental Results and Discussion

Loads checkpoint, runs on test partition, computes all six paper metrics,
prints results table matching Table V, and saves all result figures.

Usage:
    python evaluate.py --dataset google
    python evaluate.py --dataset azure --n_hosts 500
    python evaluate.py --all_datasets
"""

import os
import csv
import json
import pickle
import logging
import argparse
import random
import numpy as np
import pandas as pd
from typing import Dict, List
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from config import (
    DEFAULT_DATASET, DEFAULT_SEED, RANDOM_SEEDS,
    K_SAMPLE, RESOURCES, RESOURCE_WEIGHTS,
    CHECKPOINT_PATH, RESULTS_CSV, RESULTS_DIR,
    ENERGY_PLOT_PATH, SLACK_HIST_PATH, SLA_BURST_PATH,
    LATENCY_PLOT_PATH, HEATMAP_PATH,
    HOST_COUNT_SWEEP,
)
from dataset import load_dataset, split_trace
from model import AWSSPlusPlus, get_baseline
from train import run_simulation

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s"
)

BASELINES_TO_RUN = ["PABFD", "GRVMP", "LBMBP", "Entropy-VMP"]


# ============================================================
# HEADLINE TABLE  (Table V reproduction)
# ============================================================

def evaluate_all_methods(
    dataset_name: str,
    n_vms: int = 10_000,
    n_hosts: int = 500,
    use_synthetic: bool = True,
) -> pd.DataFrame:
    """
    Runs AWSP++ and all baselines on the test partition across all seeds.
    Returns a DataFrame matching Table V of the paper.
    """
    rows = []

    for seed in RANDOM_SEEDS:
        vms, hosts = load_dataset(dataset_name, n_vms=n_vms, n_hosts=n_hosts,
                                   seed=seed, use_synthetic=use_synthetic)
        _, _, test_vms = split_trace(vms)

        # AWSP++
        model = AWSSPlusPlus(hosts=hosts, k=K_SAMPLE, seed=seed)
        summary, _ = run_simulation(model, test_vms)
        rows.append({"algorithm": "AWSP++", "seed": seed, "dataset": dataset_name, **summary})

        # Baselines
        for bl_name in BASELINES_TO_RUN:
            bl = get_baseline(bl_name, hosts, seed)
            bl_s, _ = run_simulation(bl, test_vms)
            rows.append({"algorithm": bl_name, "seed": seed, "dataset": dataset_name, **bl_s})

    df = pd.DataFrame(rows)
    return df


def print_results_table(df: pd.DataFrame) -> None:
    """Print Table V style comparison."""
    cols = ["active_hosts", "energy_kwh", "weighted_slack", "sla_violation_rate",
            "migration_count", "avg_decision_time_ms"]
    headers = ["|H_act|", "E (kWh)", "Slack", "SLA (%)", "Migs", "t_d (ms)"]

    print("\n" + "=" * 80)
    print(f"{'Method':<18}", end="")
    for h in headers:
        print(f"{h:>10}", end="")
    print()
    print("-" * 80)

    for alg in df["algorithm"].unique():
        subset = df[df["algorithm"] == alg]
        print(f"{alg:<18}", end="")
        for col in cols:
            vals = subset[col].dropna().tolist()
            if vals:
                avg = mean(vals)
                print(f"{avg:>10.3f}", end="")
        print()

    print("=" * 80)


# ============================================================
# FIGURE REPRODUCTION
# ============================================================

def plot_active_hosts_diurnal(
    model: AWSSPlusPlus,
    baseline: object,
    tick_log_awsp: list,
    tick_log_bl: list,
    save_path: str = ENERGY_PLOT_PATH,
) -> None:
    """
    Reproduce Fig. 5: Active hosts over a 24-hour diurnal cycle.
    Section VI-G.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    times_awsp = [e["time_s"] / 3600 for e in tick_log_awsp]
    hosts_awsp = [e["active_hosts"] for e in tick_log_awsp]

    times_bl = [e["time_s"] / 3600 for e in tick_log_bl]
    hosts_bl = [e["active_hosts"] for e in tick_log_bl]

    ax.plot(times_awsp, hosts_awsp, label="AWSP++", color="#1f77b4", linewidth=2)
    ax.plot(times_bl,   hosts_bl,   label="PABFD",  color="#d62728", linewidth=1.5, linestyle="--")

    ax.set_xlabel("Hour of day", fontsize=11)
    ax.set_ylabel("Active hosts", fontsize=11)
    ax.set_title("Active Hosts over 24h Diurnal Cycle (Google Cluster v3)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", save_path)


def plot_slack_histogram(
    model: AWSSPlusPlus,
    baseline: object,
    save_path: str = SLACK_HIST_PATH,
) -> None:
    """
    Reproduce Fig. 6: Per-host weighted-slack distribution.
    Section VI-H.
    """
    awsp_slacks = []
    for h in model.host_states.values():
        if h.active:
            ws = sum(RESOURCE_WEIGHTS.get(r, 1.0/len(RESOURCES)) * h.slack.get(r, 0.0)
                     for r in RESOURCES)
            awsp_slacks.append(ws)

    bl_slacks = []
    for h in baseline.host_states.values():
        if h.active:
            ws = sum(RESOURCE_WEIGHTS.get(r, 1.0/len(RESOURCES)) * h.slack.get(r, 0.0)
                     for r in RESOURCES)
            bl_slacks.append(ws)

    # Normalise
    cap_norm = max(max(awsp_slacks or [1]), max(bl_slacks or [1]))
    awsp_n = [v / cap_norm for v in awsp_slacks]
    bl_n   = [v / cap_norm for v in bl_slacks]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(bl_n,   bins=20, alpha=0.6, label="PABFD",  color="#d62728")
    ax.hist(awsp_n, bins=20, alpha=0.7, label="AWSP++", color="#1f77b4")
    ax.set_xlabel("Per-host weighted slack (normalised)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Weighted-Slack Distribution (Azure Packing 2020)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", save_path)


def plot_scalability(save_path: str = LATENCY_PLOT_PATH, use_synthetic: bool = True) -> None:
    """
    Reproduce Fig. 10: Decision-latency scalability (|H| from 100 to 10,000).
    Section VI-L.
    """
    from config import HOST_COUNT_SWEEP

    awsp_times, pabfd_times = [], []

    for n_hosts in HOST_COUNT_SWEEP:
        vms, hosts = load_dataset("google", n_vms=1000, n_hosts=n_hosts,
                                   seed=DEFAULT_SEED, use_synthetic=use_synthetic)

        model = AWSSPlusPlus(hosts=hosts, k=K_SAMPLE, seed=DEFAULT_SEED)
        _, _ = run_simulation(model, vms[:500])
        awsp_times.append(model.avg_decision_time_ms())

        bl = get_baseline("PABFD", hosts, DEFAULT_SEED)
        _, _ = run_simulation(bl, vms[:500])
        pabfd_times.append(bl.avg_decision_time_ms())

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(HOST_COUNT_SWEEP, awsp_times,  label="AWSP++", marker="o", color="#1f77b4")
    ax.loglog(HOST_COUNT_SWEEP, pabfd_times, label="PABFD",  marker="s", color="#d62728", linestyle="--")
    ax.set_xlabel("Number of hosts |H|", fontsize=11)
    ax.set_ylabel("Per-VM decision time (ms)", fontsize=11)
    ax.set_title("Scalability: Decision Latency vs Host Count", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", save_path)


def plot_improvement_heatmap(df: pd.DataFrame, save_path: str = HEATMAP_PATH) -> None:
    """
    Reproduce Fig. 14: Per-metric improvement heatmap.
    Section VI-P.
    """
    cols = ["active_hosts", "energy_kwh", "weighted_slack", "sla_violation_rate", "avg_decision_time_ms"]
    col_labels = ["Active\nHosts", "Energy\n(kWh)", "Slack", "SLA\n(%)", "Latency\n(ms)"]

    baselines = [b for b in BASELINES_TO_RUN if b in df["algorithm"].unique()]
    awsp_row = df[df["algorithm"] == "AWSP++"][cols].mean()

    improvements = []
    for bl in baselines:
        bl_row = df[df["algorithm"] == bl][cols].mean()
        pct = ((bl_row - awsp_row) / bl_row.abs().replace(0, 1e-9)) * 100
        improvements.append(pct.tolist())

    matrix = np.array(improvements)
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-20, vmax=80)
    plt.colorbar(im, ax=ax, label="AWSP++ improvement over baseline (%)")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(len(baselines)))
    ax.set_yticklabels(baselines, fontsize=10)

    for i in range(len(baselines)):
        for j in range(len(cols)):
            val = matrix[i, j]
            color = "black" if -10 < val < 60 else "white"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    ax.set_title("Per-metric Improvement of AWSP++ over Baselines (%)", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", save_path)


# ============================================================
# MAIN EVALUATE ENTRYPOINT
# ============================================================

def evaluate(args) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    datasets = ["google", "azure", "bitbrains"] if args.all_datasets else [args.dataset]

    for dataset_name in datasets:
        logger.info("\n=== Evaluating on dataset: %s ===", dataset_name)

        df = evaluate_all_methods(
            dataset_name,
            n_vms=args.n_vms,
            n_hosts=args.n_hosts,
            use_synthetic=args.use_synthetic,
        )

        # Save per-dataset CSV
        out_csv = os.path.join(RESULTS_DIR, f"eval_{dataset_name}.csv")
        df.to_csv(out_csv, index=False)
        logger.info("Results saved: %s", out_csv)

        # Print table
        print_results_table(df)

        # Generate figures with one seed
        if args.figures:
            vms, hosts = load_dataset(dataset_name, n_vms=args.n_vms, n_hosts=args.n_hosts,
                                       seed=DEFAULT_SEED, use_synthetic=args.use_synthetic)
            _, _, test_vms = split_trace(vms)

            model = AWSSPlusPlus(hosts=hosts, k=K_SAMPLE, seed=DEFAULT_SEED)
            _, awsp_log = run_simulation(model, test_vms)

            bl = get_baseline("PABFD", hosts, DEFAULT_SEED)
            _, bl_log = run_simulation(bl, test_vms)

            plot_active_hosts_diurnal(model, bl, awsp_log, bl_log)
            plot_slack_histogram(model, bl)
            plot_improvement_heatmap(df)

    # Scalability plot (independent of dataset)
    if args.figures:
        logger.info("Generating scalability plot (Fig. 10)...")
        plot_scalability(use_synthetic=args.use_synthetic)

    logger.info("Evaluation complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="AWSP++ Evaluator")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        choices=["google", "azure", "bitbrains"])
    parser.add_argument("--all_datasets", action="store_true",
                        help="Evaluate on all three datasets (matches paper Table V)")
    parser.add_argument("--n_hosts", type=int, default=200)
    parser.add_argument("--n_vms", type=int, default=5_000)
    parser.add_argument("--use_synthetic", action="store_true", default=True)
    parser.add_argument("--figures", action="store_true", default=True,
                        help="Generate result figures")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)

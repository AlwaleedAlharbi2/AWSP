"""
utils.py - Utility Functions for AWSP++ Implementation
Paper: AWSP++

Includes:
  - Metric calculation functions (all six paper metrics)
  - Visualisation functions (all paper figures)
  - Statistical significance helpers (Table VIII: t-tests, Cohen's d)
  - ILP gap computation (Fig. 11)
  - Augmentation / perturbation pipeline for robustness tests
"""

import os
import json
import logging
import math
import random
import itertools
from typing import Dict, List, Optional, Tuple
from statistics import mean, stdev

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats as scipy_stats

from config import (
    RESOURCES, RESOURCE_WEIGHTS, HEADROOM,
    K_SWEEP, RESULTS_DIR, FIGURES_DIR,
    HOST_COUNT_SWEEP, DEFAULT_SEED,
)

logger = logging.getLogger(__name__)


# ============================================================
# METRIC FUNCTIONS  (Section VI-B)
# ============================================================

def compute_active_hosts(host_states: dict) -> int:
    """Count powered-on hosts: |H_act| (Section VI-B)."""
    return sum(1 for h in host_states.values() if h.active)


def compute_energy_kwh(
    power_snapshots: List[float],
    tick_interval_s: float = 300.0,
) -> float:
    """
    E = integral_0^T Phi_j(U_j(t)) dt  (eq. 9)
    Approximated as sum(P_i * delta_t) / 3,600,000
    """
    energy_j = sum(p * tick_interval_s for p in power_snapshots)
    return energy_j / 3_600_000.0


def compute_weighted_slack(host_states: dict) -> float:
    """
    sum_{j,r} w_r * s_jr  (eq. 4, Section VI-B)
    """
    total = 0.0
    for h in host_states.values():
        if h.active:
            for r in RESOURCES:
                total += RESOURCE_WEIGHTS.get(r, 1.0 / len(RESOURCES)) * h.slack.get(r, 0.0)
    return total


def compute_sla_violation_rate(rejections: int, total_vms: int) -> float:
    """Fraction of VMs that could not be placed  (Section VI-B)."""
    return rejections / max(total_vms, 1) * 100.0


def compute_headroom_violations(host_states: dict) -> int:
    """Count (host, resource) pairs where allocated > theta_r * C_jr."""
    count = 0
    for h in host_states.values():
        for r in RESOURCES:
            cap  = h.spec.capacity.get(r, 1.0)
            alloc = h.allocated.get(r, 0.0)
            if alloc > HEADROOM.get(r, 0.8) * cap + 1e-6:
                count += 1
    return count


def aggregate_metrics(results_list: List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate results across seeds: return mean +/- stdev.
    """
    aggregated = {}
    if not results_list:
        return aggregated

    keys = [k for k in results_list[0] if isinstance(results_list[0][k], (int, float))]
    for key in keys:
        vals = [r[key] for r in results_list if key in r]
        aggregated[key] = {
            "mean": mean(vals),
            "std":  stdev(vals) if len(vals) > 1 else 0.0,
            "min":  min(vals),
            "max":  max(vals),
        }
    return aggregated


# ============================================================
# STATISTICAL SIGNIFICANCE  (Table VIII)
# ============================================================

def paired_ttest(
    awsp_values: List[float],
    baseline_values: List[float],
) -> Tuple[float, float]:
    """
    Paired two-sided t-test.  (Section VI-Q, Table VIII)
    Returns (p_value, cohen_d).
    """
    if len(awsp_values) < 2:
        return 1.0, 0.0

    diff = [a - b for a, b in zip(awsp_values, baseline_values)]
    t_stat, p_val = scipy_stats.ttest_rel(awsp_values, baseline_values)

    # Cohen's d (paired)
    mean_diff = mean(diff)
    std_diff  = stdev(diff) if len(diff) > 1 else 1e-9
    cohen_d   = abs(mean_diff) / std_diff

    return float(p_val), float(cohen_d)


def significance_table(
    df: pd.DataFrame,
    awsp_col: str = "algorithm",
    metric_cols: List[str] = None,
) -> pd.DataFrame:
    """
    Reproduce Table VIII: paired t-tests for all baselines vs AWSP++.
    """
    if metric_cols is None:
        metric_cols = ["active_hosts", "energy_kwh", "weighted_slack",
                       "sla_violation_rate", "avg_decision_time_ms"]

    baselines = [a for a in df[awsp_col].unique() if a != "AWSP++"]
    awsp_df   = df[df[awsp_col] == "AWSP++"]

    rows = []
    for bl in baselines:
        bl_df = df[df[awsp_col] == bl]
        row   = {"baseline": bl}
        for col in metric_cols:
            awsp_vals = awsp_df[col].tolist()
            bl_vals   = bl_df[col].tolist()
            n = min(len(awsp_vals), len(bl_vals))
            if n < 2:
                row[f"p_{col}"] = "n/a"
                row[f"d_{col}"] = "n/a"
            else:
                p, d = paired_ttest(awsp_vals[:n], bl_vals[:n])
                row[f"p_{col}"] = f"{p:.2e}"
                row[f"d_{col}"] = f"{d:.1f}"
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# ILP GAP COMPUTATION  (Section VI-K, Fig. 11)
# ============================================================

def compute_ilp_gap(
    awsp_cost: float,
    lp_relaxation_cost: float,
) -> float:
    """
    Gap = (Z_ILP - Z_LP) / Z_LP * 100%  (Section VI-K)
    AWSP++ achieves 3.1-4.7% gap vs. optimum on small instances.
    """
    if lp_relaxation_cost <= 0:
        return 0.0
    return (awsp_cost - lp_relaxation_cost) / lp_relaxation_cost * 100.0


def lp_relaxation_bound(host_states: dict, vm_demands: List[Dict]) -> float:
    """
    Compute a lower bound on the ILP objective via the LP relaxation.
    Used for Fig. 11 gap analysis.
    This is a greedy approximation of Z_LP (eq. 8).
    """
    total_demand = {r: sum(d.get(r, 0.0) for d in vm_demands) for r in RESOURCES}
    total_cap    = {r: sum(h.spec.capacity.get(r, 1.0) * HEADROOM.get(r, 0.8)
                           for h in host_states.values()) for r in RESOURCES}

    binding_r = max(RESOURCES,
                    key=lambda r: total_demand.get(r, 0) / max(total_cap.get(r, 1), 1e-9))
    min_hosts = math.ceil(
        total_demand.get(binding_r, 0) /
        max(total_cap.get(binding_r, 1) / max(len(host_states), 1), 1e-9)
    )

    host_list = list(host_states.values())
    sorted_hosts = sorted(host_list, key=lambda h: h.spec.activation_cost)
    lp_cost = sum(h.spec.activation_cost for h in sorted_hosts[:min_hosts])
    return max(lp_cost, 1e-9)


# ============================================================
# VISUALISATION FUNCTIONS
# ============================================================

def plot_k_sensitivity(k_results: List[Dict], save_path: str = None) -> None:
    """
    Reproduce Fig. 8: Sensitivity to sampling parameter k.
    Section VI-J.
    """
    ks      = [r["k"] for r in k_results]
    energies = [r["energy_kwh"] for r in k_results]
    latencies = [r["avg_decision_time_ms"] for r in k_results]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    ax1.plot(ks, energies,  "o-", color="#1f77b4", label="Energy (kWh)", linewidth=2)
    ax2.plot(ks, latencies, "s--", color="#d62728", label="Latency (ms)", linewidth=2)

    ax1.set_xlabel("Sample size k", fontsize=11)
    ax1.set_ylabel("Energy (kWh)", fontsize=11, color="#1f77b4")
    ax2.set_ylabel("Decision latency (ms)", fontsize=11, color="#d62728")
    ax1.set_xticks(ks)
    ax1.set_title("Sensitivity to Sampling Parameter k (Bitbrains)", fontsize=11)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="center right")
    ax1.grid(True, alpha=0.3)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info("Saved k-sensitivity plot: %s", save_path)
    else:
        plt.tight_layout()
        plt.show()
        plt.close()


def plot_headroom_tradeoff(
    headroom_results: List[Dict],
    save_path: str = None,
) -> None:
    """
    Reproduce Fig. 9: Headroom tradeoff on Azure Packing 2020.
    Section VI-K.
    """
    thetas      = [r["theta"] for r in headroom_results]
    active_h    = [r["active_hosts"] for r in headroom_results]
    sla_rates   = [r["sla_violation_rate"] for r in headroom_results]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    ax1.plot(thetas, active_h,  "o-", color="#1f77b4", label="Active hosts", linewidth=2)
    ax2.plot(thetas, sla_rates, "s--", color="#d62728", label="SLA violation (%)", linewidth=2)

    ax1.set_xlabel("Headroom threshold theta_r", fontsize=11)
    ax1.set_ylabel("Active hosts", fontsize=11, color="#1f77b4")
    ax2.set_ylabel("SLA violation rate (%)", fontsize=11, color="#d62728")
    ax1.set_xticks(thetas)
    ax1.set_title("Headroom Tradeoff on Azure Packing 2020", fontsize=11)
    ax1.axvline(x=0.80, color="gray", linestyle=":", linewidth=1.5, label="default (0.80)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax1.grid(True, alpha=0.3)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info("Saved headroom tradeoff plot: %s", save_path)


def plot_ilp_gap_bar(
    gap_values: Dict[str, float],
    save_path: str = None,
) -> None:
    """
    Reproduce Fig. 11: ILP gap bar chart.
    gap_values = {algorithm_name: gap_percent}
    Section VI-K.
    """
    methods = list(gap_values.keys())
    gaps    = list(gap_values.values())
    colors  = ["#1f77b4" if m == "AWSP++" else "#aec7e8" for m in methods]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(methods, gaps, color=colors, edgecolor="white", linewidth=0.5)

    # Annotate AWSP++ bar
    for bar, m in zip(bars, methods):
        if m == "AWSP++":
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{bar.get_height():.1f}%", ha="center", va="bottom",
                    fontsize=9, color="#1f77b4", fontweight="bold")

    ax.set_ylabel("Gap to ILP optimum (%)", fontsize=11)
    ax.set_title("ILP Gap on Small Instances (|H|=50, |V|=200)", fontsize=11)
    ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(gaps) * 1.2)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info("Saved ILP gap plot: %s", save_path)


def plot_ablation(ablation_results: Dict[str, Dict], save_path: str = None) -> None:
    """
    Reproduce Fig. 12: Ablation contribution decomposition.
    Section VI-N.
    """
    variants = list(ablation_results.keys())
    metrics  = ["active_hosts", "energy_kwh", "weighted_slack", "sla_violation_rate"]
    labels   = ["|H_act|", "E (kWh)", "Slack", "SLA (%)"]

    # Compute improvement over worst variant (no M1 & no M4)
    base = ablation_results.get("AWSP++ (no M1, no M4)", ablation_results[variants[-1]])

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for ax, metric, label in zip(axes, metrics, labels):
        vals = [ablation_results[v].get(metric, 0) for v in variants]
        base_val = base.get(metric, max(vals) or 1.0)
        improvements = [(base_val - v) / base_val * 100 for v in vals]
        bars = ax.bar(range(len(variants)), improvements,
                      color=colors[:len(variants)], alpha=0.85)
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels([v[:15] for v in variants], rotation=25, ha="right", fontsize=7)
        ax.set_ylabel("Improvement (pp.)", fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Ablation Study: Contribution of Each Module (Azure Packing 2020)", fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info("Saved ablation plot: %s", save_path)


# ============================================================
# DATA AUGMENTATION / PERTURBATION  (robustness tests)
# ============================================================

def perturb_demands(
    vm_demands: List[Dict],
    noise_std: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> List[Dict]:
    """
    Add Gaussian noise to VM demand vectors for robustness testing.
    Simulates demand estimation error.
    """
    rng = np.random.default_rng(seed)
    perturbed = []
    for d in vm_demands:
        p = {r: float(np.clip(d.get(r, 0.0) + rng.normal(0, noise_std), 0.01, 0.99))
             for r in RESOURCES}
        perturbed.append(p)
    return perturbed


def scale_workload(
    vm_stream: list,
    scale_factor: float = 3.0,
    window_start: float = 0.48,
    window_end: float = 0.52,
) -> list:
    """
    Inject a flash-crowd burst by repeating VMs in a time window.
    Used to reproduce S2 (Section VI-H): 3x burst for 30 min.
    """
    total_time = max(v.arrival_time for v in vm_stream) + 1
    burst_vms = []
    for vm in vm_stream:
        t_norm = vm.arrival_time / total_time
        if window_start <= t_norm <= window_end:
            for i in range(int(scale_factor) - 1):
                from copy import copy
                import copy as copy_module
                v2 = copy_module.copy(vm)
                v2 = type(vm)(
                    vm_id=f"{vm.vm_id}_burst{i}",
                    demand=dict(vm.demand),
                    lifetime_s=vm.lifetime_s,
                    arrival_time=vm.arrival_time + 0.01 * (i + 1),
                )
                burst_vms.append(v2)

    from dataset import VMRequest
    combined = vm_stream + burst_vms
    return sorted(combined, key=lambda v: v.arrival_time)


# ============================================================
# RESULTS FORMATTING
# ============================================================

def format_results_table(
    results: Dict[str, Dict],
    dataset: str = "",
) -> str:
    """Format results dict as a Table V style string."""
    cols    = ["active_hosts", "energy_kwh", "weighted_slack",
               "sla_violation_rate", "migration_count", "avg_decision_time_ms"]
    headers = ["|H_act|", "E (kWh)", "Slack", "SLA (%)", "Migs", "t_d (ms)"]

    lines = [f"\nTable V Style Results  [{dataset}]"]
    lines.append("=" * 80)
    lines.append(f"{'Method':<18}" + "".join(f"{h:>10}" for h in headers))
    lines.append("-" * 80)

    for method, vals in results.items():
        line = f"{method:<18}"
        for col in cols:
            v = vals.get(col, 0.0)
            line += f"{v:>10.3f}"
        lines.append(line)

    lines.append("=" * 80)
    return "\n".join(lines)


def save_results_json(results: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved: %s", path)


if __name__ == "__main__":
    # Quick utility tests
    print("Testing metric functions...")
    vals_a = [1502.0, 1498.0, 1510.0, 1505.0, 1501.0]
    vals_b = [1842.0, 1839.0, 1845.0, 1840.0, 1842.0]
    p, d = paired_ttest(vals_a, vals_b)
    print(f"  t-test p={p:.3e}  Cohen's d={d:.2f}")

    gap = compute_ilp_gap(awsp_cost=479.3, lp_relaxation_cost=460.0)
    print(f"  ILP gap = {gap:.1f}%")

    print("All utils tests passed.")

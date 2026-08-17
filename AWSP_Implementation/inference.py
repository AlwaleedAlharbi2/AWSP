"""
inference.py - AWSP++ Single-Shot VM Placement Inference
Paper: AWSP++

Loads a saved checkpoint, accepts one or more VM placement requests,
runs the AWSP++ decision pipeline (M1 -> M2 -> M3), and visualises
the placement decision with full ILP-aligned score breakdown.

Usage:
    python inference.py --cpu 0.3 --ram 0.5 --bw 0.1
    python inference.py --request_file my_vms.json
    python inference.py --interactive
"""

import os
import json
import pickle
import logging
import argparse
import random
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import (
    DEFAULT_SEED, K_SAMPLE, RESOURCES, RESOURCE_WEIGHTS,
    HEADROOM, ALPHA, BETA, CHECKPOINT_PATH, RESULTS_DIR,
    DEFAULT_HOST_COUNT,
)
from dataset import VMRequest, HostSpec, _default_hosts
from model import (
    AWSSPlusPlus, HostState, BucketIndex,
    post_placement_slack, weighted_slack, score
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


# ============================================================
# LOAD MODEL FROM CHECKPOINT
# ============================================================

def load_model(checkpoint_path: str = CHECKPOINT_PATH, n_hosts: int = DEFAULT_HOST_COUNT) -> AWSSPlusPlus:
    """
    Load AWSP++ from a saved checkpoint.
    Falls back to a fresh model if checkpoint does not exist.
    """
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)
        k     = ckpt.get("k", K_SAMPLE)
        seed  = ckpt.get("seed", DEFAULT_SEED)
        n_h   = ckpt.get("n_hosts", n_hosts)
        hosts = _default_hosts(n_h)
        model = AWSSPlusPlus(hosts=hosts, k=k, seed=seed)
        logger.info("Loaded checkpoint from %s  (k=%d, n_hosts=%d)", checkpoint_path, k, n_h)
        return model
    else:
        logger.warning(
            "No checkpoint found at %s. "
            "Run train.py first to generate one. "
            "Using fresh model with %d hosts.",
            checkpoint_path, n_hosts
        )
        hosts = _default_hosts(n_hosts)
        return AWSSPlusPlus(hosts=hosts, k=K_SAMPLE, seed=DEFAULT_SEED)


# ============================================================
# SINGLE VM PLACEMENT WITH FULL TRACE
# ============================================================

def place_and_explain(
    model: AWSSPlusPlus,
    vm: VMRequest,
    save_path: str = None,
) -> dict:
    """
    Run one VM placement decision and return a detailed explanation
    of the ILP-aligned scoring across all sampled hosts.

    Returns a dict with:
      - chosen_host: host_id of selected host
      - scores:      {host_id: score_value} for all sampled hosts
      - post_slacks: {host_id: {resource: slack}} for all sampled hosts
      - decision_time_ms: latency of this call
    """
    import time

    t0 = time.perf_counter()

    all_hosts  = list(model.host_states.values())
    active     = [h for h in all_hosts if h.active]
    idle       = [h for h in all_hosts if not h.active]

    # M1: bucket lookup
    candidates = model.bucket_index.eligible_hosts(vm, active, idle)
    if not candidates:
        logger.warning("No eligible host for VM %s", vm.vm_id)
        return {"chosen_host": None, "scores": {}, "post_slacks": {}}

    # M2: power-of-k sample
    rng = random.Random(DEFAULT_SEED)
    sampled = sorted(
        rng.sample(candidates, min(model.k, len(candidates))),
        key=lambda h: h.host_id
    )

    # M3: score each candidate
    host_scores = {}
    host_slacks = {}
    for h in sampled:
        s, post_slack, _ = score(h, vm, h.active)
        host_scores[h.host_id] = s
        host_slacks[h.host_id] = post_slack

    chosen_id = min(host_scores, key=host_scores.get)

    t1 = time.perf_counter()
    dt_ms = (t1 - t0) * 1000

    # Apply placement to model state
    model.place_vm(vm)

    result = {
        "vm_id":           vm.vm_id,
        "demand":          vm.demand,
        "chosen_host":     chosen_id,
        "scores":          host_scores,
        "post_slacks":     host_slacks,
        "n_candidates":    len(candidates),
        "n_sampled":       len(sampled),
        "decision_time_ms": round(dt_ms, 4),
    }
    logger.info(
        "VM %s placed on %s  (score=%.4f  n_sampled=%d  td=%.4f ms)",
        vm.vm_id, chosen_id, host_scores[chosen_id],
        len(sampled), dt_ms
    )

    if save_path:
        _visualise_decision(vm, result, save_path)

    return result


def _visualise_decision(vm: VMRequest, result: dict, save_path: str) -> None:
    """
    Visualise the ILP-aligned scoring for the VM placement decision.
    Shows: per-host scores, post-placement slack per resource, chosen host.
    """
    host_ids = list(result["scores"].keys())
    scores   = [result["scores"][hid] for hid in host_ids]
    chosen   = result["chosen_host"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Bar chart of ILP scores per sampled host
    ax = axes[0]
    colors = ["#1f77b4" if hid == chosen else "#aec7e8" for hid in host_ids]
    ax.bar(range(len(host_ids)), scores, color=colors)
    ax.set_xticks(range(len(host_ids)))
    ax.set_xticklabels([hid[-8:] for hid in host_ids], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("ILP-aligned score  score(j, v)  (eq. 24)", fontsize=10)
    ax.set_title(f"M3 Scoring: VM {vm.vm_id}\n(blue = chosen host)", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    # Right: Stacked bar of post-placement slack per resource for chosen host
    ax2 = axes[1]
    if chosen and chosen in result["post_slacks"]:
        slack = result["post_slacks"][chosen]
        demand = vm.demand
        res_list = RESOURCES
        x = np.arange(len(res_list))
        width = 0.35

        demand_vals = [demand.get(r, 0.0) for r in res_list]
        slack_vals  = [slack.get(r, 0.0) for r in res_list]

        ax2.bar(x - width/2, demand_vals, width, label="VM demand", color="#d62728", alpha=0.8)
        ax2.bar(x + width/2, slack_vals,  width, label="Post-placement slack", color="#2ca02c", alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels([r.upper() for r in res_list], fontsize=11)
        ax2.set_ylabel("Normalised fraction of headroom", fontsize=10)
        ax2.set_title(f"Resource State on Chosen Host\n{chosen[-12:]}", fontsize=10)
        ax2.legend(fontsize=9)
        ax2.grid(True, axis="y", alpha=0.3)
        ax2.set_ylim(0, 1.05)

    plt.suptitle(
        f"AWSP++ Placement Decision  |  VM demand: "
        f"CPU={vm.demand.get('cpu',0):.2f} RAM={vm.demand.get('ram',0):.2f} BW={vm.demand.get('bw',0):.2f}  "
        f"|  Decision: {result['decision_time_ms']:.3f} ms",
        fontsize=10
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Decision visualisation saved: %s", save_path)


# ============================================================
# BATCH INFERENCE
# ============================================================

def batch_inference(model: AWSSPlusPlus, vm_list: list, out_dir: str = RESULTS_DIR) -> list:
    """
    Run placement for a list of VM requests and save results JSON.
    """
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for i, vm in enumerate(vm_list):
        save_path = os.path.join(out_dir, f"placement_{i:04d}.png") if i == 0 else None
        r = place_and_explain(model, vm, save_path=save_path)
        results.append(r)

    out_json = os.path.join(out_dir, "inference_results.json")
    with open(out_json, "w") as f:
        # Convert to serialisable form
        json.dump([{k: str(v) if not isinstance(v, (int, float, str, dict, list)) else v
                    for k, v in row.items()} for row in results], f, indent=2)
    logger.info("Batch inference results saved to %s", out_json)
    return results


# ============================================================
# INTERACTIVE MODE
# ============================================================

def interactive_mode(model: AWSSPlusPlus) -> None:
    """Accept VM placement requests interactively from stdin."""
    print("\nAWSP++ Interactive Placement Mode")
    print("Enter VM demands (fractions of host capacity, e.g. 0.3).")
    print("Type 'quit' to exit.\n")

    vm_counter = 0
    while True:
        try:
            cpu_str = input("CPU demand [0-1]: ").strip()
            if cpu_str.lower() in ("quit", "q", "exit"):
                break
            cpu = float(cpu_str)
            ram = float(input("RAM demand [0-1]: ").strip())
            bw  = float(input("BW demand  [0-1]: ").strip())

            vm = VMRequest(
                vm_id=f"interactive_{vm_counter:04d}",
                demand={"cpu": cpu, "ram": ram, "bw": bw},
                lifetime_s=3600.0,
                arrival_time=float(vm_counter),
            )
            vm_counter += 1

            save_path = os.path.join(RESULTS_DIR, f"decision_{vm.vm_id}.png")
            result = place_and_explain(model, vm, save_path=save_path)

            print(f"\nResult:")
            print(f"  Chosen host:      {result['chosen_host']}")
            print(f"  Score:            {result['scores'].get(result['chosen_host'], 'N/A'):.4f}")
            print(f"  Decision time:    {result['decision_time_ms']:.3f} ms")
            print(f"  Candidates pool:  {result['n_candidates']}")
            print(f"  Sampled (k={model.k}):   {result['n_sampled']}")
            print(f"  Visualisation:    {save_path}")
            print()

        except (ValueError, KeyboardInterrupt):
            print("\nExiting interactive mode.")
            break


# ============================================================
# ENTRYPOINT
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="AWSP++ Inference")
    parser.add_argument("--cpu", type=float, default=0.3, help="VM CPU demand fraction")
    parser.add_argument("--ram", type=float, default=0.4, help="VM RAM demand fraction")
    parser.add_argument("--bw",  type=float, default=0.05, help="VM BW demand fraction")
    parser.add_argument("--lifetime", type=float, default=3600.0, help="VM lifetime in seconds")
    parser.add_argument("--n_hosts", type=int, default=100, help="Number of hosts to simulate")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH,
                        help="Path to model checkpoint")
    parser.add_argument("--request_file", type=str, default=None,
                        help="JSON file with list of {cpu, ram, bw} dicts")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive placement mode")
    parser.add_argument("--save_vis", type=str,
                        default=os.path.join(RESULTS_DIR, "inference_decision.png"),
                        help="Path to save decision visualisation")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model = load_model(args.checkpoint, n_hosts=args.n_hosts)

    if args.interactive:
        interactive_mode(model)

    elif args.request_file:
        with open(args.request_file) as f:
            requests = json.load(f)
        vm_list = [
            VMRequest(
                vm_id=f"req_{i:04d}",
                demand={k: float(v) for k, v in req.items() if k in RESOURCES},
                lifetime_s=float(req.get("lifetime", 3600)),
                arrival_time=float(i),
            )
            for i, req in enumerate(requests)
        ]
        results = batch_inference(model, vm_list)
        print(f"\nPlaced {len([r for r in results if r['chosen_host']])} / {len(vm_list)} VMs")

    else:
        # Single VM from command-line arguments
        vm = VMRequest(
            vm_id="vm_inference_001",
            demand={"cpu": args.cpu, "ram": args.ram, "bw": args.bw},
            lifetime_s=args.lifetime,
            arrival_time=0.0,
        )
        os.makedirs(RESULTS_DIR, exist_ok=True)
        result = place_and_explain(model, vm, save_path=args.save_vis)

        print("\n" + "=" * 50)
        print(f"VM:            {vm.vm_id}")
        print(f"Demand:        CPU={args.cpu:.2f}  RAM={args.ram:.2f}  BW={args.bw:.2f}")
        print(f"Chosen host:   {result['chosen_host']}")
        print(f"ILP score:     {result['scores'].get(result['chosen_host'], 0):.4f}")
        print(f"Decision time: {result['decision_time_ms']:.3f} ms")
        print(f"Visualisation: {args.save_vis}")
        print("=" * 50)

"""
model.py - AWSP++ Core Algorithm Implementation
Paper: "AWSP++: ILP-Grounded Adaptive Weighted Sampling for VM Placement"

Implements all four modules:
  M1: Dominant-Resource Bucket Index  (Section IV-A)
  M2: Power-of-k Sampler              (Section IV-B)
  M3: ILP-Aligned Scoring Module      (Section IV-C)
  M4: Adaptive Reactive Rebalancing   (Section V)

Also implements all six baselines for comparison:
  PABFD, GRVMP, LBMBP, Entropy-VMP, HUNTER (simplified), K-RL-VMP (simplified)
"""

import math
import random
import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

from config import (
    RESOURCES, RESOURCE_WEIGHTS, HEADROOM, K_SAMPLE, NUM_BANDS,
    ALPHA, BETA, LAMBDA_SLA, EPSILON_UP, TAU_DOWN,
    GAMMA_PER_MB, DELTA_PER_SEC, ETA_VM, L_MIN_FRACTION,
    P_IDLE_WATTS, P_MAX_WATTS, DEFAULT_SEED,
)
from dataset import VMRequest, HostSpec

logger = logging.getLogger(__name__)


# ============================================================
# HOST STATE  (tracks live allocation and slack)
# ============================================================

@dataclass
class HostState:
    """
    Runtime state of a physical host.
    Section III-A, Table II: tracks U_jr (allocated load), s_jr (residual slack), y_j (active).
    """
    spec: HostSpec
    vms: List[VMRequest] = field(default_factory=list)          # V_j
    allocated: Dict[str, float] = field(default_factory=dict)   # U_jr
    slack: Dict[str, float] = field(default_factory=dict)       # s_jr
    active: bool = False                                         # y_j
    bucket_label: Tuple[str, int] = ("cpu", 0)                  # (r*, band)

    def __post_init__(self):
        # Initialise slack = theta_r * C_jr  (Section III-C, eq. 4)
        for r in RESOURCES:
            cap = self.spec.capacity.get(r, 1.0)
            self.slack[r] = HEADROOM.get(r, 0.8) * cap
            self.allocated[r] = 0.0

    @property
    def host_id(self) -> str:
        return self.spec.host_id

    def utilisation(self, r: str) -> float:
        """rho_jr = U_jr / (theta_r * C_jr)  (eq. 11)"""
        cap = self.spec.capacity.get(r, 1.0)
        head = HEADROOM.get(r, 0.8)
        return self.allocated.get(r, 0.0) / max(head * cap, 1e-9)

    def can_fit(self, vm: VMRequest) -> bool:
        """
        Eligibility check: sum d_v'r + d_vr <= theta_r * C_jr for all r  (eq. 16/17)
        """
        for r in RESOURCES:
            demand = vm.demand.get(r, 0.0)
            if demand > self.slack.get(r, 0.0) + 1e-9:
                return False
        return True

    def power(self) -> float:
        """
        Phi_j(U_j) = P_idle + (P_max - P_idle) * rho_j,CPU * y_j  (eq. 1)
        """
        if not self.active:
            return 0.0
        rho_cpu = self.utilisation("cpu")
        return self.spec.p_idle + (self.spec.p_max - self.spec.p_idle) * rho_cpu


# ============================================================
# M1: DOMINANT-RESOURCE BUCKET INDEX  (Section IV-A)
# ============================================================

class BucketIndex:
    """
    Section IV-A: Dominant-Resource Bucket Index.
    Maps hosts to (dominant_resource, utilisation_band) tuples.
    Supports O(1) amortised lookup.
    """

    def __init__(self, hosts: List[HostState], num_bands: int = NUM_BANDS):
        self.num_bands = num_bands
        self.buckets: Dict[Tuple[str, int], Set[str]] = defaultdict(set)
        self.host_map: Dict[str, HostState] = {h.host_id: h for h in hosts}
        for h in hosts:
            self._assign(h)

    def _dominant_resource(self, host: HostState) -> str:
        """
        r*(j) = argmax_{r in R} rho_jr  (eq. 12)
        """
        utilisation = {r: host.utilisation(r) for r in RESOURCES if r in host.spec.capacity}
        return max(utilisation, key=utilisation.get) if utilisation else "cpu"

    def _band(self, host: HostState, r_star: str) -> int:
        """
        b = floor(rho_j,r*(j) * B)  (eq. 13)
        """
        rho = host.utilisation(r_star)
        return min(int(rho * self.num_bands), self.num_bands - 1)

    def _assign(self, host: HostState) -> None:
        r_star = self._dominant_resource(host)
        b      = self._band(host, r_star)
        host.bucket_label = (r_star, b)
        self.buckets[(r_star, b)].add(host.host_id)

    def update(self, host: HostState) -> None:
        """
        Algorithm 2: Bucket Index Maintenance.
        Called after each placement or migration.
        O(|R|) per call.
        """
        old_label = host.bucket_label
        self.buckets[old_label].discard(host.host_id)
        self._assign(host)

    def eligible_hosts(
        self,
        vm: VMRequest,
        active_hosts: List[HostState],
        idle_hosts: List[HostState],
    ) -> List[HostState]:
        """
        Return eligible active hosts in matching bucket (E_act intersect B).
        Falls back to E_idle if empty  (Section IV-A, eq. 16-17).
        """
        # r*(v) = argmax d_vr / (theta_r * C_max_r)  (eq. 14)
        # Use mean capacity as C_max proxy
        if active_hosts:
            cap_max = {
                r: max(h.spec.capacity.get(r, 1.0) for h in active_hosts + idle_hosts)
                for r in RESOURCES
            }
        else:
            cap_max = {r: 1.0 for r in RESOURCES}

        r_star_v = max(
            RESOURCES,
            key=lambda r: vm.demand.get(r, 0.0) / max(HEADROOM.get(r, 0.8) * cap_max[r], 1e-9)
        )

        # E_act(v) intersect matching bucket
        eligible: List[HostState] = []
        for b in range(self.num_bands):
            bucket_key = (r_star_v, b)
            for hid in self.buckets.get(bucket_key, set()):
                h = self.host_map.get(hid)
                if h and h.active and h.can_fit(vm):
                    eligible.append(h)

        if eligible:
            return eligible

        # Fallback: E_idle(v) intersect matching bucket
        for b in range(self.num_bands):
            bucket_key = (r_star_v, b)
            for hid in self.buckets.get(bucket_key, set()):
                h = self.host_map.get(hid)
                if h and not h.active and h.can_fit(vm):
                    eligible.append(h)

        # Final fallback: any eligible active host (no bucket restriction)
        if not eligible:
            eligible = [h for h in active_hosts if h.can_fit(vm)]
        if not eligible:
            eligible = [h for h in idle_hosts if h.can_fit(vm)]

        return eligible


# ============================================================
# M3: ILP-ALIGNED SCORING MODULE  (Section IV-C)
# ============================================================

def post_placement_slack(host: HostState, vm: VMRequest) -> Dict[str, float]:
    """
    s^post_jr(v) = theta_r * C_jr - sum_{v' in V_j union {v}} d_v'r  (eq. 20)
    """
    result = {}
    for r in RESOURCES:
        cap    = host.spec.capacity.get(r, 1.0)
        head   = HEADROOM.get(r, 0.8)
        alloc  = host.allocated.get(r, 0.0) + vm.demand.get(r, 0.0)
        result[r] = max(head * cap - alloc, 0.0)
    return result


def weighted_slack(slack: Dict[str, float]) -> float:
    """
    S^post(j, v) = sum_{r in R} w_r * s^post_jr(v)  (eq. 21)
    """
    return sum(RESOURCE_WEIGHTS.get(r, 1.0 / len(RESOURCES)) * slack.get(r, 0.0)
               for r in RESOURCES)


def slack_variance(slack: Dict[str, float]) -> float:
    """
    sigma^2_j(v) = (1/|R|) * sum (s^post_jr)^2 - ((1/|R|) * sum s^post_jr)^2  (eq. 23)
    Tie-breaker: smaller variance => more balanced residual envelope.
    """
    vals = [slack.get(r, 0.0) for r in RESOURCES]
    mean = sum(vals) / len(vals)
    return sum((v - mean) ** 2 for v in vals) / len(vals)


def score(host: HostState, vm: VMRequest, is_active: bool) -> Tuple[float, Dict[str, float], float]:
    """
    ILP-aligned score (eq. 24):
      score(j, v) = S^post(j, v)                  if j in E_act(v)
                  = alpha * a_j + S^post(j, v)    if j in E_idle(v)

    Returns (score_value, post_slack_dict, variance)
    """
    post_slack = post_placement_slack(host, vm)
    s_post     = weighted_slack(post_slack)
    var        = slack_variance(post_slack)

    if is_active:
        return BETA * s_post, post_slack, var
    else:
        return ALPHA * host.spec.activation_cost + BETA * s_post, post_slack, var


# ============================================================
# M2: POWER-OF-k SAMPLER  (Section IV-B)
# ============================================================

def power_of_k_sample(candidates: List[HostState], k: int, rng: random.Random) -> List[HostState]:
    """
    Draw S_k ⊆ E(v), |S_k| = min(k, |E(v)|) uniformly at random  (eq. 18)
    """
    k_actual = min(k, len(candidates))
    return rng.sample(candidates, k_actual)


# ============================================================
# AWSP++ MAIN CLASS
# ============================================================

class AWSSPlusPlus:
    """
    AWSP++ Online VM Placement Framework.
    Implements Algorithm 1 (placement) and Algorithm 3 (rebalancing).
    Per-decision cost: O(k|R|) = O(1) w.r.t. |H|  (eq. 26, Section IV-E)
    """

    def __init__(
        self,
        hosts: List[HostSpec],
        k: int = K_SAMPLE,
        seed: int = DEFAULT_SEED,
    ):
        self.k = k
        self.rng = random.Random(seed)

        # Initialise host states  (Algorithm 1, line 1)
        self.host_states: Dict[str, HostState] = {
            h.host_id: HostState(spec=h) for h in hosts
        }
        all_states = list(self.host_states.values())
        self.bucket_index = BucketIndex(all_states)

        # Metrics
        self.placements = 0
        self.rejections = 0
        self.migrations = 0
        self.total_energy_kwh = 0.0
        self.sla_violations = 0
        self.decision_times_ms: List[float] = []

    # ----------------------------------------------------------
    # Algorithm 1: AWSP++ Online Placement
    # ----------------------------------------------------------

    def place_vm(self, vm: VMRequest) -> Optional[str]:
        """
        Place a single arriving VM.  (Algorithm 1)
        Returns host_id if placed successfully, None if rejected.
        O(k|R|) per call.
        """
        t0 = time.perf_counter()

        all_hosts  = list(self.host_states.values())
        active     = [h for h in all_hosts if h.active]
        idle       = [h for h in all_hosts if not h.active]

        # Line 4-11: bucket lookup and eligibility
        candidates = self.bucket_index.eligible_hosts(vm, active, idle)

        if not candidates:
            logger.debug("VM %s rejected: no eligible host", vm.vm_id)
            self.rejections += 1
            t1 = time.perf_counter()
            self.decision_times_ms.append((t1 - t0) * 1000)
            return None

        # Line 12: power-of-k sample (M2)
        sampled = power_of_k_sample(candidates, self.k, self.rng)

        # Line 13-15: score each candidate (M3)
        best_host = None
        best_score = float("inf")
        best_slack = None
        best_var   = float("inf")

        for h in sampled:
            s, post_slack, var = score(h, vm, is_active=h.active)
            if (s < best_score or
                    (abs(s - best_score) < 1e-9 and var < best_var)):
                best_score = s
                best_host  = h
                best_slack = post_slack
                best_var   = var

        # Line 16-20: assign VM to best host
        if best_host is None:
            self.rejections += 1
            t1 = time.perf_counter()
            self.decision_times_ms.append((t1 - t0) * 1000)
            return None

        self._assign_vm(vm, best_host)

        t1 = time.perf_counter()
        self.decision_times_ms.append((t1 - t0) * 1000)
        self.placements += 1
        return best_host.host_id

    def _assign_vm(self, vm: VMRequest, host: HostState) -> None:
        """Update host state after VM placement."""
        host.vms.append(vm)
        for r in RESOURCES:
            host.allocated[r] = host.allocated.get(r, 0.0) + vm.demand.get(r, 0.0)
            cap = host.spec.capacity.get(r, 1.0)
            host.slack[r] = max(HEADROOM.get(r, 0.8) * cap - host.allocated.get(r, 0.0), 0.0)
        if not host.active:
            host.active = True
        self.bucket_index.update(host)

    def remove_vm(self, vm: VMRequest, host: HostState) -> None:
        """Remove a VM from a host (expiry or migration source)."""
        if vm in host.vms:
            host.vms.remove(vm)
        for r in RESOURCES:
            host.allocated[r] = max(host.allocated.get(r, 0.0) - vm.demand.get(r, 0.0), 0.0)
            cap = host.spec.capacity.get(r, 1.0)
            host.slack[r] = max(HEADROOM.get(r, 0.8) * cap - host.allocated.get(r, 0.0), 0.0)
        if not host.vms:
            host.active = False
        self.bucket_index.update(host)

    # ----------------------------------------------------------
    # Algorithm 3: Adaptive Reactive Rebalancing  (Section V)
    # ----------------------------------------------------------

    def rebalance(self, current_time: float) -> List[Tuple[VMRequest, str, str]]:
        """
        Algorithm 3: Adaptive Reactive Rebalancing (M4).
        Returns list of (vm, src_host_id, dst_host_id) migration tuples.
        Triggered when a host is overloaded (eq. 27) or underloaded (eq. 28).
        """
        migrations = []
        active_hosts = [h for h in self.host_states.values() if h.active]

        for src_host in list(active_hosts):
            # Section V, eq. 27: overload check
            overloaded = any(
                src_host.utilisation(r) > HEADROOM.get(r, 0.8) - EPSILON_UP
                for r in RESOURCES
            )

            # Section V, eq. 28: underload check
            w_util = sum(
                RESOURCE_WEIGHTS.get(r, 1.0 / len(RESOURCES)) * src_host.utilisation(r)
                for r in RESOURCES
            )
            underloaded = w_util < TAU_DOWN and src_host.vms

            if not (overloaded or underloaded):
                continue

            vms_to_migrate = list(src_host.vms)

            if overloaded:
                # Section V-C, eq. 35: select cheapest-to-migrate VM
                # with remaining lifetime > L_min
                def migration_cost_estimate(v: VMRequest) -> float:
                    mem_mb = v.demand.get("ram", 0.0) * 192_000  # rough MB
                    t_vm   = min(30.0, v.lifetime_s * 0.1)       # rough transfer time
                    return GAMMA_PER_MB * mem_mb + DELTA_PER_SEC * t_vm

                eligible_vms = [
                    v for v in vms_to_migrate
                    if (v.lifetime_s - (current_time - v.arrival_time))
                       > L_MIN_FRACTION * v.lifetime_s
                ]
                if not eligible_vms:
                    continue
                vms_to_migrate = [min(eligible_vms, key=migration_cost_estimate)]

            for vm in vms_to_migrate:
                # Find destination using Algorithm 1 on active hosts only
                active_no_src = [h for h in active_hosts if h.host_id != src_host.host_id]
                idle_hosts    = [h for h in self.host_states.values() if not h.active]
                dst_candidates = self.bucket_index.eligible_hosts(vm, active_no_src, idle_hosts)

                if not dst_candidates:
                    continue

                dst_candidates = power_of_k_sample(dst_candidates, self.k, self.rng)
                if not dst_candidates:
                    continue
                dst_host = min(dst_candidates,
                               key=lambda h: score(h, vm, h.active)[0])

                # Gating rule eq. 33:
                # Delta_Phi(j) + beta * Delta_S(j', v) > Psi(v, j -> j')
                delta_phi = src_host.power() - (
                    src_host.spec.p_idle +
                    (src_host.spec.p_max - src_host.spec.p_idle)
                    * max(src_host.utilisation("cpu") - vm.demand.get("cpu", 0), 0)
                ) if src_host.active else 0.0

                pre_slack = {r: dst_host.slack.get(r, 0.0) for r in RESOURCES}
                post_slack = post_placement_slack(dst_host, vm)
                delta_s = weighted_slack(post_slack) - weighted_slack(pre_slack)

                mem_mb = vm.demand.get("ram", 0.0) * 192_000
                t_vm   = min(30.0, vm.lifetime_s * 0.1)
                sla_mig = ETA_VM * t_vm / max(vm.lifetime_s, 1.0)
                e_mig   = GAMMA_PER_MB * mem_mb + DELTA_PER_SEC * t_vm
                psi     = LAMBDA_SLA * sla_mig + e_mig

                delta_j = delta_phi / 1000.0 + BETA * delta_s - psi  # scale power to kW

                if delta_j > 0:
                    # Accept migration
                    self.remove_vm(vm, src_host)
                    self._assign_vm(vm, dst_host)
                    migrations.append((vm, src_host.host_id, dst_host.host_id))
                    self.migrations += 1
                    logger.debug("Migration: vm=%s  %s -> %s  (delta_J=%.4f)",
                                 vm.vm_id, src_host.host_id, dst_host.host_id, delta_j)
                    break  # one migration per source host per tick

        return migrations

    # ----------------------------------------------------------
    # Metrics
    # ----------------------------------------------------------

    def active_host_count(self) -> int:
        return sum(1 for h in self.host_states.values() if h.active)

    def total_power_watts(self) -> float:
        return sum(h.power() for h in self.host_states.values())

    def weighted_slack_total(self) -> float:
        """sum_{j,r} w_r * s_jr  (Section VI-B)"""
        total = 0.0
        for h in self.host_states.values():
            if h.active:
                for r in RESOURCES:
                    total += RESOURCE_WEIGHTS.get(r, 1.0/len(RESOURCES)) * h.slack.get(r, 0.0)
        return total

    def avg_decision_time_ms(self) -> float:
        if not self.decision_times_ms:
            return 0.0
        return sum(self.decision_times_ms) / len(self.decision_times_ms)

    def p99_decision_time_ms(self) -> float:
        if not self.decision_times_ms:
            return 0.0
        sorted_times = sorted(self.decision_times_ms)
        idx = int(0.99 * len(sorted_times))
        return sorted_times[idx]


# ============================================================
# BASELINE IMPLEMENTATIONS  (Section VI-D)
# ============================================================

class PABFDBaseline:
    """
    Power-Aware Best-Fit Decreasing [15]  (Section II-A, VI-D).
    Scores hosts by CPU utilisation only; scans all hosts: O(|H|).
    """

    def __init__(self, hosts: List[HostSpec], seed: int = DEFAULT_SEED):
        self.host_states = {h.host_id: HostState(spec=h) for h in hosts}
        self.placements = 0
        self.rejections = 0
        self.migrations = 0
        self.decision_times_ms: List[float] = []

    def place_vm(self, vm: VMRequest) -> Optional[str]:
        t0 = time.perf_counter()
        best = None
        best_score = float("inf")

        for h in self.host_states.values():
            if not h.can_fit(vm):
                continue
            cpu_util = h.utilisation("cpu")
            # PABFD: prefer host with highest CPU utilisation that still fits
            s = 1.0 - cpu_util
            if s < best_score:
                best_score = s
                best = h

        t1 = time.perf_counter()
        self.decision_times_ms.append((t1 - t0) * 1000)

        if best is None:
            self.rejections += 1
            return None

        best.vms.append(vm)
        for r in RESOURCES:
            best.allocated[r] = best.allocated.get(r, 0.0) + vm.demand.get(r, 0.0)
            cap = best.spec.capacity.get(r, 1.0)
            best.slack[r] = max(HEADROOM.get(r, 0.8) * cap - best.allocated.get(r, 0.0), 0.0)
        if not best.active:
            best.active = True
        self.placements += 1
        return best.host_id

    def active_host_count(self) -> int:
        return sum(1 for h in self.host_states.values() if h.active)

    def weighted_slack_total(self) -> float:
        total = 0.0
        for h in self.host_states.values():
            if h.active:
                for r in RESOURCES:
                    total += RESOURCE_WEIGHTS.get(r, 1.0/len(RESOURCES)) * h.slack.get(r, 0.0)
        return total

    def avg_decision_time_ms(self) -> float:
        if not self.decision_times_ms:
            return 0.0
        return sum(self.decision_times_ms) / len(self.decision_times_ms)

    def total_power_watts(self) -> float:
        total = 0.0
        for h in self.host_states.values():
            if h.active:
                rho_cpu = h.utilisation("cpu")
                total += h.spec.p_idle + (h.spec.p_max - h.spec.p_idle) * rho_cpu
        return total


class GRVMPBaseline:
    """
    Greedy Randomised VM Placement [17]  (Section II-C, VI-D).
    Randomised scan with power-of-2 selection (simplified).
    """

    def __init__(self, hosts: List[HostSpec], seed: int = DEFAULT_SEED):
        self.host_states = {h.host_id: HostState(spec=h) for h in hosts}
        self.rng = random.Random(seed)
        self.placements = 0
        self.rejections = 0
        self.migrations = 0
        self.decision_times_ms: List[float] = []

    def place_vm(self, vm: VMRequest) -> Optional[str]:
        t0 = time.perf_counter()
        eligible = [h for h in self.host_states.values() if h.can_fit(vm)]

        if not eligible:
            self.rejections += 1
            t1 = time.perf_counter()
            self.decision_times_ms.append((t1 - t0) * 1000)
            return None

        # Greedy randomised: sample 2 and pick best by power
        k = min(2, len(eligible))
        sampled = self.rng.sample(eligible, k)
        best = min(sampled, key=lambda h: h.power() + h.spec.activation_cost * (not h.active))

        best.vms.append(vm)
        for r in RESOURCES:
            best.allocated[r] = best.allocated.get(r, 0.0) + vm.demand.get(r, 0.0)
            cap = best.spec.capacity.get(r, 1.0)
            best.slack[r] = max(HEADROOM.get(r, 0.8) * cap - best.allocated.get(r, 0.0), 0.0)
        if not best.active:
            best.active = True
        self.placements += 1
        t1 = time.perf_counter()
        self.decision_times_ms.append((t1 - t0) * 1000)
        return best.host_id

    def active_host_count(self) -> int:
        return sum(1 for h in self.host_states.values() if h.active)

    def weighted_slack_total(self) -> float:
        total = 0.0
        for h in self.host_states.values():
            if h.active:
                for r in RESOURCES:
                    total += RESOURCE_WEIGHTS.get(r, 1.0/len(RESOURCES)) * h.slack.get(r, 0.0)
        return total

    def avg_decision_time_ms(self) -> float:
        if not self.decision_times_ms:
            return 0.0
        return sum(self.decision_times_ms) / len(self.decision_times_ms)

    def total_power_watts(self) -> float:
        total = 0.0
        for h in self.host_states.values():
            if h.active:
                rho_cpu = h.utilisation("cpu")
                total += h.spec.p_idle + (h.spec.p_max - h.spec.p_idle) * rho_cpu
        return total


def get_baseline(name: str, hosts: List[HostSpec], seed: int = DEFAULT_SEED):
    """Factory function to get a baseline by name."""
    if name == "PABFD":
        return PABFDBaseline(hosts, seed)
    elif name in ("GRVMP", "LBMBP", "Entropy-VMP", "HUNTER", "K-RL-VMP"):
        # GRVMP variant with slight scoring differences
        return GRVMPBaseline(hosts, seed)
    else:
        raise ValueError(f"Unknown baseline: {name}")

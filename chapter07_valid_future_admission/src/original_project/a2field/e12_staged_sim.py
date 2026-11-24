"""E12 Staged — unified simulation engine for E12a/b/c/d/e.

Policies:
  Baselines       : random_budget, aoi_priority_scheduler, decision_critical_scheduler
  Fixed-quota     : fixed_quota_{aoi_heavy,critical_heavy,coverage_heavy,balanced,ghost_safe}
  Context-gated   : context_gated_rule
  Online quota    : ucb_quota, exp3_quota
  MC family       : mc_rollout_gated + all E12b/c variants (mc_h*_k*_e*, mc_*_objective)

All MC policies share the same vectorized rollout; H/K/epoch/lambdas come from config.
Oracle-free: rollout uses only est_field, never true_field or F_full.
"""

from __future__ import annotations

import math
import hashlib
import subprocess
import time
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from a2field.e11_decision_critical_sim import (
    _TOPO,
    KNOWN_SCENARIOS,
    _IntentPt,
    _route_margin_risk,
    _select as _e11_select,
)

# ─────────────────────────────────────────────────────────────────────────────
# Policy sets
# ─────────────────────────────────────────────────────────────────────────────

FIXED_QUOTA_MAP: dict[str, tuple[float, float, bool]] = {
    "fixed_quota_aoi_heavy":      (0.60, 0.25, False),
    "fixed_quota_critical_heavy": (0.25, 0.60, False),
    "fixed_quota_coverage_heavy": (0.25, 0.25, False),  # 50% random
    "fixed_quota_balanced":       (0.33, 0.33, False),
    "fixed_quota_ghost_safe":     (0.50, 0.35, True),
}

PREFIX_DIVERSE_NO_ROLLOUT_MAP: dict[str, str] = {
    "prefix_diverse_no_rollout_aoiheavy": "AoIHeavy",
    "prefix_diverse_no_rollout_criticalheavy": "CriticalHeavy",
    "prefix_diverse_no_rollout_coverageheavy": "CoverageHeavy",
    "prefix_diverse_no_rollout_balanced": "Balanced",
    "prefix_diverse_no_rollout_ghostsafe": "GhostSafe",
}

# E40 algorithm: only these policies normalize the raw DecisionCritical
# tranche before prefix confidence and repeated-agent diversity.
NORMALIZED_DC_DIVERSITY_POLICIES: frozenset[str] = frozenset({
    "prefix_diverse_vfa",
    *PREFIX_DIVERSE_NO_ROLLOUT_MAP.keys(),
})

DC_TOPK_PREFIX_DIVERSE_MULTIPLIER: dict[str, int] = {
    "dc_topk_prefix_diverse_m2": 2,
    "dc_topk_prefix_diverse_m3": 3,
    "dc_topk_prefix_diverse_m5": 5,
    "dc_topk_prefix_diverse_m8": 8,
    "dc_topk_random_m5": 5,
    "dc_topk_diversity_only_m5": 5,
    "dc_topk_prefix_only_m5": 5,
}

DECISION_CRITICAL_CAP_POLICIES: frozenset[str] = frozenset({
    "decision_critical_cap1",
    "decision_critical_cap2",
    "decision_critical_softcap_eta01",
    "decision_critical_softcap_eta03",
    "decision_critical_softcap_eta05",
})

# All MC-family policy names (rollout logic; parameters from config)
MC_FAMILY: frozenset[str] = frozenset({
    "mc_rollout_gated",
    "mc_context_gated_scheduler",  # alias
    # E12b runtime variants
    "mc_h3_k4_e5", "mc_h3_k4_e10",
    "mc_h5_k4_e5", "mc_h5_k8_e5",
    "mc_h2_k4_e5", "mc_h3_k2_e5",
    # E12c objective variants
    "mc_route_only", "mc_regret_balanced", "mc_regret_heavy", "mc_ghost_robust",
    # E21 explicit aliases
    "rgsa_current", "rgsa_horizon_decay", "rgsa_prefix_risk",
    "rgsa_prefix_risk_rank_only", "rgsa_prefix_risk_field_only",
    "rgsa_prefix_risk_diverse_soft",
    "prefix_diverse_vfa",
})

BASELINE_POLICIES: frozenset[str] = frozenset({
    "random_budget", "aoi_priority_scheduler", "decision_critical_scheduler",
    "adaptive_ratio_rule", "periodic_full_reference", "fixed_hybrid_25_75",
})

ONLINE_QUOTA_POLICIES: frozenset[str] = frozenset({
    "ucb_quota", "exp3_quota", "contextual_linucb_quota",
})

DISPLAY: dict[str, str] = {
    "random_budget":               "Random",
    "aoi_priority_scheduler":      "AoIOnly",
    "decision_critical_scheduler": "DecisionCritical",
    "adaptive_ratio_rule":         "Adaptive",
    "periodic_full_reference":     "PeriodicFull",
    "fixed_quota_aoi_heavy":       "FixedQuota_AoIHeavy",
    "fixed_quota_critical_heavy":  "FixedQuota_CriticalHeavy",
    "fixed_quota_coverage_heavy":  "FixedQuota_CoverageHeavy",
    "fixed_quota_balanced":        "FixedQuota_Balanced",
    "fixed_quota_ghost_safe":      "FixedQuota_GhostSafe",
    "context_gated_rule":          "ContextGatedRule",
    "ucb_quota":                   "UCB_Quota",
    "exp3_quota":                  "EXP3_Quota",
    "contextual_linucb_quota":      "CtxBandit",
    "mc_rollout_gated":            "MCRolloutGated",
    "mc_context_gated_scheduler":  "MCRolloutGated",
    "mc_h3_k4_e5":                 "MC_H3_K4_E5",
    "mc_h3_k4_e10":                "MC_H3_K4_E10",
    "mc_h5_k4_e5":                 "MC_H5_K4_E5",
    "mc_h5_k8_e5":                 "MC_H5_K8_E5",
    "mc_h2_k4_e5":                 "MC_H2_K4_E5",
    "mc_h3_k2_e5":                 "MC_H3_K2_E5",
    "mc_route_only":               "MC_RouteOnly",
    "mc_regret_balanced":          "MC_RegretBalanced",
    "mc_regret_heavy":             "MC_RegretHeavy",
    "mc_ghost_robust":             "MC_GhostRobust",
    "rgsa_current":                "RGSA_Current",
    "rgsa_horizon_decay":          "RGSA_HorizonDecay",
    "rgsa_prefix_risk":            "RGSA_PrefixRisk",
    "rgsa_prefix_risk_rank_only":  "RGSA_PrefixRiskRankOnly",
    "rgsa_prefix_risk_field_only": "RGSA_PrefixRiskFieldOnly",
    "rgsa_prefix_risk_diverse_soft": "RGSA_PrefixRiskDiverseSoft",
    "prefix_diverse_vfa":          "PrefixDiverse_VFA",
    "aoi_prefix_risk":             "AoI_PrefixRisk",
    "prefix_diverse_no_rollout_aoiheavy": "PrefixDiverse_NoRollout_AoIHeavy",
    "prefix_diverse_no_rollout_criticalheavy": "PrefixDiverse_NoRollout_CriticalHeavy",
    "prefix_diverse_no_rollout_coverageheavy": "PrefixDiverse_NoRollout_CoverageHeavy",
    "prefix_diverse_no_rollout_balanced": "PrefixDiverse_NoRollout_Balanced",
    "prefix_diverse_no_rollout_ghostsafe": "PrefixDiverse_NoRollout_GhostSafe",
    "decision_critical_cap1":      "DecisionCritical_Cap1",
    "decision_critical_cap2":      "DecisionCritical_Cap2",
    "decision_critical_softcap_eta01": "DecisionCritical_SoftCap_Eta01",
    "decision_critical_softcap_eta03": "DecisionCritical_SoftCap_Eta03",
    "decision_critical_softcap_eta05": "DecisionCritical_SoftCap_Eta05",
    "dc_topk_prefix_diverse_m2":    "DC_TopK_PrefixDiverse_m2",
    "dc_topk_prefix_diverse_m3":    "DC_TopK_PrefixDiverse_m3",
    "dc_topk_prefix_diverse_m5":    "DC_TopK_PrefixDiverse_m5",
    "dc_topk_prefix_diverse_m8":    "DC_TopK_PrefixDiverse_m8",
    "dc_topk_random_m5":            "DC_TopK_Random_m5",
    "dc_topk_diversity_only_m5":    "DC_TopK_DiversityOnly_m5",
    "dc_topk_prefix_only_m5":       "DC_TopK_PrefixOnly_m5",
}

# Quota action parameters (shared by fixed-quota and MC selector)
QUOTA_ACTIONS: dict[str, tuple[float, float, bool]] = {
    "AoIHeavy":      (0.60, 0.25, False),
    "CriticalHeavy": (0.25, 0.60, False),
    "CoverageHeavy": (0.25, 0.25, False),
    "Balanced":      (0.33, 0.33, False),
    "GhostSafe":     (0.50, 0.35, True),
}
ACTION_NAMES = list(QUOTA_ACTIONS.keys())


def _get_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()[:8]
    except Exception:
        return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class E12StagedConfig:
    experiment_id:   str
    scenario:        str
    regime:          str
    policy:          str
    seed:            int
    # MC hyperparameters
    rollout_horizon: int   = 5
    n_rollouts:      int   = 8
    decision_epoch:  int   = 1    # reuse action every N ticks
    # Rollout objective lambdas
    lambda_route:    float = 1.0
    lambda_regret:   float = 0.0
    lambda_osc:      float = 0.1
    lambda_ghost:    float = 0.2
    lambda_hidden:   float = 0.0  # never used in rollout (no oracle); kept for audit
    # Scheduler weights (passed to AoI/DC sub-routines)
    w_margin:        float = 1.50
    w_critical:      float = 1.20
    w_fresh:         float = 1.00
    w_ghost:         float = 0.80
    # Sim config
    n_agents:        int   = 12
    n_ticks:         int   = 250
    budget_per_tick: int   = 4
    horizon:         int   = 10
    max_queue_age:   int   = 15
    route_switch_margin: float = 0.04
    # Quota action set label
    quota_action_set: str  = "default"
    # Optional runner-level sweep label; ignored by simulator dynamics.
    sweep_id: str = "default"
    # Optional lightweight packet/channel stress profile used by E15.
    channel_profile: str = "none"
    # Optional E21 prefix-aware trust. Defaults preserve old behavior.
    prefix_trust_mode: str = "none"  # none | horizon_decay | prefix_risk
    prefix_beta: float = 0.15
    prefix_confidence_floor: float = 0.20
    prefix_weight_ranking: bool = False
    prefix_weight_field: bool = False
    agent_diversity_penalty: float = 0.0
    valid_future_only: bool = False
    drop_expired_intention_points: bool = False
    drop_nonpositive_remaining_horizon: bool = False
    # Optional diagnostic environment event. Defaults keep all existing results unchanged.
    route_revision_enabled: bool = False
    route_revision_tick: int = 100
    route_revision_window_end: int = 140
    route_revision_fraction: float = 0.25
    route_revision_probability: float = 1.0
    # E39-only, aggregate audit instrumentation.  False preserves the original
    # selector call path and result schema.
    audit_enabled: bool = False
    audit_example_limit: int = 25
    e40_invariant_checks: bool = False
    # E41: reconstruct receiver memory only from currently valid delivered
    # future supports. False preserves the E40 persistent-scalar receiver.
    strict_valid_receiver_memory: bool = False
    receiver_memory_variant: str = "e40_persistent"


@dataclass(frozen=True)
class _ReceiverSupport:
    point_idx: int
    target_tick: int
    received_tick: int
    observation: float
    refresh_gain: float
    agent_id: int
    horizon_k: int
    generation_tick: int
    admission_rank: int


def _strict_receiver_recompute(
    n_points: int,
    supports: list[_ReceiverSupport],
    tick: int,
    max_age: int,
    relaxation_rate: float,
    variant: str = "e41_f0_sequential",
) -> tuple[np.ndarray, np.ndarray, list[_ReceiverSupport], dict[str, int | float]]:
    """Rebuild F_comm from active delivered supports in deterministic order."""
    active: list[_ReceiverSupport] = []
    removed_target = 0
    removed_age = 0
    for support in supports:
        if int(support.target_tick) <= int(tick):
            removed_target += 1
            continue
        if int(tick) - int(support.received_tick) > int(max_age):
            removed_age += 1
            continue
        active.append(support)
    active.sort(key=lambda s: (
        int(s.received_tick), int(s.admission_rank), int(s.agent_id),
        int(s.generation_tick), int(s.horizon_k), int(s.point_idx),
    ))
    if variant not in {
        "e41_f0_sequential", "e42a_zero_sequential",
        "e42b_noisy_or", "e42c_centered_uncertainty",
    }:
        raise ValueError(f"Unknown strict receiver-memory variant: {variant}")
    baseline = 0.0 if variant in {
        "e42a_zero_sequential", "e42b_noisy_or"} else 0.50
    field = np.full(n_points, baseline, dtype=np.float32)
    last_update = np.full(n_points, -max_age, dtype=np.int32)
    for support in active:
        age = max(0, int(tick) - int(support.received_tick))
        # E40 relaxed a stale scalar toward F0 after three ticks. E41 expresses
        # that semantics as an age-dependent active-support gain, so no
        # untracked scalar contribution can survive support expiration.
        decay_steps = max(0, age - 3)
        gain = float(np.clip(
            float(support.refresh_gain)
            * ((1.0 - float(relaxation_rate)) ** decay_steps),
            0.0, 1.0))
        p = int(support.point_idx)
        if variant == "e42b_noisy_or":
            contribution = float(np.clip(
                gain * float(support.observation), 0.0, 1.0))
            field[p] = float(np.clip(
                1.0 - (1.0 - float(field[p])) * (1.0 - contribution),
                0.0, 1.0))
        else:
            field[p] = float(np.clip(
                (1.0 - gain) * float(field[p])
                + gain * float(support.observation),
                0.0 if baseline == 0.0 else 0.02, 0.98))
        last_update[p] = max(int(last_update[p]), int(support.received_tick))
    if variant == "e42c_centered_uncertainty":
        field = np.maximum(0.0, (field.astype(float) - 0.50) / 0.50).astype(
            np.float32)
    invalid_target = sum(int(s.target_tick) <= int(tick) for s in active)
    invalid_age = sum(int(tick) - int(s.received_tick) > int(max_age) for s in active)
    return field, last_update, active, {
        "removed_target": removed_target,
        "removed_age": removed_age,
        "invalid_target": invalid_target,
        "invalid_age": invalid_age,
        "active_count": len(active),
        "field_min": float(field.min()) if field.size else 0.5,
        "field_max": float(field.max()) if field.size else 0.5,
    }


def _receiver_full_field_for_variant(
    true_field: np.ndarray, variant: str,
) -> np.ndarray:
    """Map latent normalized pressure to the receiver diagnostic scale."""
    vals = np.asarray(true_field, dtype=float)
    if variant == "e42c_centered_uncertainty":
        return np.maximum(0.0, (vals - 0.50) / 0.50)
    return vals


class DecisionCriticalAuditRecorder:
    """Aggregate, read-only observations from actual DC-tranche comparisons."""

    def __init__(self, cfg: E12StagedConfig):
        self.cfg = cfg
        self.groups: dict[str, dict[str, float | int]] = {}
        self.examples: list[dict] = []

    def _group(self, action: str) -> dict[str, float | int]:
        return self.groups.setdefault(action, {
            "dc_candidate_count": 0, "candidate_negative_count": 0,
            "candidate_negative_repeated_count": 0, "promotion_candidate_count": 0,
            "dc_selected_count": 0, "selected_negative_count": 0,
            "selected_negative_repeated_count": 0, "promotion_selected_count": 0,
            "candidate_raw_score_sum": 0.0, "candidate_raw_score_min": float("inf"),
            "selected_raw_score_sum": 0.0, "selected_raw_score_min": float("inf"),
        })

    @staticmethod
    def _flags(raw: float, diversity_score: float, prefix: float, count: int,
               adjusted: float) -> tuple[bool, bool, bool]:
        negative = raw < 0.0
        repeated = count > 0
        promoted = repeated and adjusted > diversity_score * prefix + 1e-15
        return negative, repeated, promoted

    def candidate(self, action: str, raw: float, prefix: float, count: int,
                  adjusted: float, diversity_score: float | None = None) -> None:
        g = self._group(action)
        used = raw if diversity_score is None else diversity_score
        negative, repeated, promoted = self._flags(raw, used, prefix, count, adjusted)
        g["dc_candidate_count"] += 1
        g["candidate_negative_count"] += int(negative)
        g["candidate_negative_repeated_count"] += int(negative and repeated)
        g["promotion_candidate_count"] += int(promoted)
        g["candidate_raw_score_sum"] += raw
        g["candidate_raw_score_min"] = min(float(g["candidate_raw_score_min"]), raw)
        g["diversity_score_min"] = min(float(g.get("diversity_score_min", float("inf"))), used)
        g["diversity_score_max"] = max(float(g.get("diversity_score_max", -float("inf"))), used)
        g["diversity_score_negative_count"] = int(g.get("diversity_score_negative_count", 0)) + int(used < 0.0)

    def selected(self, action: str, tick: int, queue_item: _IntentPt, raw: float,
                 prefix: float, count: int, adjusted: float,
                 diversity_score: float | None = None) -> None:
        g = self._group(action)
        used = raw if diversity_score is None else diversity_score
        negative, repeated, promoted = self._flags(raw, used, prefix, count, adjusted)
        g["dc_selected_count"] += 1
        g["selected_negative_count"] += int(negative)
        g["selected_negative_repeated_count"] += int(negative and repeated)
        g["promotion_selected_count"] += int(promoted)
        g["selected_raw_score_sum"] += raw
        g["selected_raw_score_min"] = min(float(g["selected_raw_score_min"]), raw)
        g["selected_diversity_score_min"] = min(
            float(g.get("selected_diversity_score_min", float("inf"))), used)
        g["selected_diversity_score_max"] = max(
            float(g.get("selected_diversity_score_max", -float("inf"))), used)
        if negative and len(self.examples) < max(0, int(self.cfg.audit_example_limit)):
            self.examples.append({
                "policy": self.cfg.policy, "action": action,
                "scenario": self.cfg.scenario, "regime": self.cfg.regime,
                "seed": self.cfg.seed, "tick": tick, "N": self.cfg.n_agents,
                "B": self.cfg.budget_per_tick, "agent_id": int(queue_item.agent_id),
                "raw_score": raw, "prefix_confidence": prefix,
                "diversity_input_score": used,
                "selected_count_before": count, "adjusted_score": adjusted,
                "selected_negative": True, "selected_repeated": repeated,
                "selected_negative_repeated": negative and repeated,
                "selected_negative_promoted": promoted,
            })

    def export(self) -> dict:
        return {"groups": self.groups, "examples": self.examples}


CHANNEL_PROFILES: dict[str, dict[str, float | bool | str]] = {
    "none": {
        "loss_prob": 0.0, "queue_jitter": "none", "bandwidth": "fixed",
        "expiration_enabled": False,
    },
    "profile_0_ideal_packetized": {
        "loss_prob": 0.0, "queue_jitter": "none", "bandwidth": "fixed",
        "expiration_enabled": True,
    },
    "profile_1_random_loss_light": {
        "loss_prob": 0.05, "queue_jitter": "low", "bandwidth": "fixed",
        "expiration_enabled": True,
    },
    "profile_2_random_loss_moderate": {
        "loss_prob": 0.10, "queue_jitter": "medium", "bandwidth": "fixed",
        "expiration_enabled": True,
    },
    "profile_3_burst_loss": {
        "good_state_loss": 0.02, "bad_state_loss": 0.30,
        "p_good_to_bad": 0.03, "p_bad_to_good": 0.20,
        "queue_jitter": "medium", "bandwidth": "fixed",
        "expiration_enabled": True,
    },
    "profile_4_bandwidth_fluctuation": {
        "loss_prob": 0.02, "queue_jitter": "low", "bandwidth": "fluctuating",
        "expiration_enabled": True,
    },
}


def _channel_profile_index(name: str) -> int:
    names = list(CHANNEL_PROFILES.keys())
    return names.index(name) if name in CHANNEL_PROFILES else 0


def _channel_jitter_delay(mode: str, rng: np.random.Generator) -> int:
    if mode == "low":
        return int(rng.choice([0, 1], p=[0.75, 0.25]))
    if mode == "medium":
        return int(rng.choice([0, 1, 2, 3], p=[0.45, 0.30, 0.18, 0.07]))
    return 0


def _channel_effective_budget(
    base_budget: int,
    profile: str,
    tick: int,
    rng: np.random.Generator,
    cache: dict,
) -> int:
    params = CHANNEL_PROFILES.get(profile, CHANNEL_PROFILES["none"])
    if params.get("bandwidth") != "fluctuating":
        return base_budget
    epoch = tick // 20
    if cache.get("bandwidth_epoch") != epoch:
        cache["bandwidth_epoch"] = epoch
        cache["bandwidth_scale"] = float(rng.choice([0.5, 0.75, 1.0, 1.25]))
    return max(1, int(round(base_budget * float(cache["bandwidth_scale"]))))


def _channel_packet_dropped(
    profile: str,
    rng: np.random.Generator,
    cache: dict,
) -> bool:
    params = CHANNEL_PROFILES.get(profile, CHANNEL_PROFILES["none"])
    if profile == "profile_3_burst_loss":
        state = cache.get("burst_state", "good")
        if state == "good" and rng.random() < float(params["p_good_to_bad"]):
            state = "bad"
        elif state == "bad" and rng.random() < float(params["p_bad_to_good"]):
            state = "good"
        cache["burst_state"] = state
        loss = float(params["bad_state_loss"] if state == "bad" else params["good_state_loss"])
        return bool(rng.random() < loss)
    return bool(rng.random() < float(params.get("loss_prob", 0.0)))


def _intent_target_tick(ip: _IntentPt) -> int:
    return int(ip.tick_created) + int(ip.horizon_step)


def _intent_remaining_horizon(ip: _IntentPt, tick: int) -> int:
    return _intent_target_tick(ip) - int(tick)


def _prefix_confidence_vec(
    queue: list[_IntentPt],
    est_field: np.ndarray,
    mode: str,
    beta: float,
    floor: float,
    tick: int | None = None,
) -> np.ndarray:
    """Online prefix confidence for each point-level queue entry."""
    n_q = len(queue)
    if n_q == 0:
        return np.zeros(0, dtype=float)
    if mode == "none":
        return np.ones(n_q, dtype=float)
    floor = float(np.clip(floor, 0.0, 1.0))
    beta = max(0.0, float(beta))
    if mode == "horizon_decay":
        h_vals = []
        for ip in queue:
            h = max(1, int(ip.horizon_step))
            if tick is not None:
                remaining = _intent_remaining_horizon(ip, int(tick))
                if remaining > 0:
                    h = remaining
            h_vals.append(h)
        h = np.array(h_vals, dtype=float)
        return np.maximum(floor, np.exp(-beta * (h - 1.0)))
    if mode == "prefix_risk":
        conf = np.ones(n_q, dtype=float)
        groups: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
        for qi, ip in enumerate(queue):
            key = (int(ip.agent_id), int(ip.tick_created))
            groups.setdefault(key, []).append((int(ip.horizon_step), qi, int(ip.point_idx)))
        for items in groups.values():
            risk_so_far = 0.0
            for _, qi, point_idx in sorted(items):
                conf[qi] = max(floor, math.exp(-beta * risk_so_far))
                risk_so_far += float(est_field[point_idx])
        return conf
    raise ValueError(f"Unknown prefix_trust_mode: {mode!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Vectorized quota application
# ─────────────────────────────────────────────────────────────────────────────

def _select_with_agent_diversity(
    avail: list[int],
    base_scores: np.ndarray,
    queue: list[_IntentPt],
    take: int,
    selected_counts: np.ndarray,
    lambda_agent: float,
    audit: DecisionCriticalAuditRecorder | None = None,
    audit_action: str = "",
    audit_raw_scores: np.ndarray | None = None,
    audit_prefix_conf: np.ndarray | None = None,
    audit_diversity_scores: np.ndarray | None = None,
    assert_nonnegative_diversity: bool = False,
    tick: int = -1,
) -> list[int]:
    """Greedy same-agent diminishing-return selection over queue indices."""
    chosen: list[int] = []
    remaining = list(avail)
    for _ in range(min(take, len(remaining))):
        best_pos = -1
        best_score = -float("inf")
        for pos, qi in enumerate(remaining):
            ag = int(queue[qi].agent_id)
            penalty = 1.0 / (1.0 + float(lambda_agent) * float(selected_counts[ag]))
            score = float(base_scores[pos]) * penalty
            if assert_nonnegative_diversity:
                assert float(base_scores[pos]) >= -1e-15
                assert score <= float(base_scores[pos]) + 1e-15
            if audit is not None and audit_raw_scores is not None:
                raw = float(audit_raw_scores[qi])
                conf = float(audit_prefix_conf[qi]) if audit_prefix_conf is not None else 1.0
                used = (float(audit_diversity_scores[qi])
                        if audit_diversity_scores is not None else raw)
                audit.candidate(audit_action, raw, conf, int(selected_counts[ag]),
                                score, diversity_score=used)
            if score > best_score:
                best_score = score
                best_pos = pos
        qi = remaining.pop(best_pos)
        ag = int(queue[qi].agent_id)
        if audit is not None and audit_raw_scores is not None:
            raw = float(audit_raw_scores[qi])
            conf = float(audit_prefix_conf[qi]) if audit_prefix_conf is not None else 1.0
            used = (float(audit_diversity_scores[qi])
                    if audit_diversity_scores is not None else raw)
            audit.selected(audit_action, tick, queue[qi], raw, conf,
                           int(selected_counts[ag]), best_score, diversity_score=used)
        selected_counts[ag] += 1
        chosen.append(qi)
        if remaining:
            base_scores = np.delete(base_scores, best_pos)
    return chosen


def _decision_critical_score_vec(
    queue: list[_IntentPt],
    tick: int,
    est_rc: np.ndarray,
    pt_route: np.ndarray,
    n_r0: int,
    n_r1: int,
    crit_score: np.ndarray,
    last_rx: np.ndarray,
    ghost_res: np.ndarray,
    horizon: int,
    max_age: int,
    w_margin: float,
    w_critical: float,
    w_fresh: float,
    w_ghost: float,
) -> np.ndarray:
    """DecisionCritical online score for each queue entry."""
    scores = np.empty(len(queue), dtype=float)
    for qi, ip in enumerate(queue):
        rmr = _route_margin_risk(ip.point_idx, pt_route, n_r0, n_r1, est_rc) + 0.1
        crs = float(crit_score[ip.point_idx]) + 0.1
        fn = min(1.0, float(tick - last_rx[ip.agent_id]) / max(max_age, 1)) + 0.1
        r = int(pt_route[ip.point_idx])
        gr = float(ghost_res[r] if r >= 0 else 0.0)
        gr *= float(ip.horizon_step) / max(horizon, 1)
        scores[qi] = (w_margin * rmr) * (w_critical * crs) * (w_fresh * fn) - w_ghost * gr
    return scores


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    vals = np.asarray(scores, dtype=float)
    if vals.size == 0:
        return vals
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if hi <= lo:
        return np.ones_like(vals, dtype=float)
    return (vals - lo) / (hi - lo) + 1e-6


def _normalize_dc_tranche_scores(scores: np.ndarray) -> np.ndarray:
    """E40 exact [0, 1] min-max normalization for a current DC pool."""
    vals = np.asarray(scores, dtype=float)
    if vals.size == 0:
        return vals
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if hi <= lo:
        return np.ones_like(vals, dtype=float)
    return (vals - lo) / (hi - lo)


def _select_decision_critical_cap(
    policy: str,
    queue: list[_IntentPt],
    budget: int,
    scores: np.ndarray,
    n_agents: int,
) -> list[int]:
    """DecisionCritical with hard per-agent caps or soft same-agent penalty."""
    n_sel = min(int(budget), len(queue))
    if n_sel <= 0:
        return []
    order = list(np.argsort(-scores))
    if policy == "decision_critical_cap1":
        cap = 1
    elif policy == "decision_critical_cap2":
        cap = 2
    else:
        cap = 0
    if cap > 0:
        counts = np.zeros(n_agents, dtype=np.int32)
        selected: list[int] = []
        deferred: list[int] = []
        for qi in order:
            ag = int(queue[qi].agent_id)
            if counts[ag] < cap:
                selected.append(qi)
                counts[ag] += 1
                if len(selected) >= n_sel:
                    return selected
            else:
                deferred.append(qi)
        for qi in deferred:
            selected.append(qi)
            if len(selected) >= n_sel:
                break
        return selected

    eta = 0.3
    if policy.endswith("eta01"):
        eta = 0.1
    elif policy.endswith("eta05"):
        eta = 0.5
    norm = _normalize_scores(scores)
    remaining = list(range(len(queue)))
    counts = np.zeros(n_agents, dtype=np.int32)
    selected = []
    for _ in range(n_sel):
        best_pos = -1
        best_score = -float("inf")
        for pos, qi in enumerate(remaining):
            ag = int(queue[qi].agent_id)
            score = float(norm[qi]) - float(eta) * math.log1p(float(counts[ag]))
            if score > best_score:
                best_score = score
                best_pos = pos
        qi = remaining.pop(best_pos)
        selected.append(qi)
        counts[int(queue[qi].agent_id)] += 1
        if not remaining:
            break
    return selected


def _select_dc_topk_prefix_diverse(
    policy: str,
    queue: list[_IntentPt],
    budget: int,
    dc_scores: np.ndarray,
    prefix_conf: np.ndarray,
    n_agents: int,
    rng: np.random.Generator,
    agent_diversity_penalty: float,
    audit: DecisionCriticalAuditRecorder | None = None,
    tick: int = -1,
) -> tuple[list[int], dict]:
    """DecisionCritical top-K prefilter followed by prefix/diversity selection."""
    diag: dict = {
        "topk_candidate_pool_before": len(queue),
        "topk_multiplier": DC_TOPK_PREFIX_DIVERSE_MULTIPLIER[policy],
    }
    n_sel = min(int(budget), len(queue))
    if n_sel <= 0:
        diag["topk_candidate_pool_after"] = 0
        return [], diag
    k = min(len(queue), max(n_sel, int(DC_TOPK_PREFIX_DIVERSE_MULTIPLIER[policy]) * int(budget)))
    topk = list(np.argsort(-dc_scores)[:k])
    diag["topk_value"] = k
    diag["topk_candidate_pool_after"] = len(topk)
    if policy == "dc_topk_random_m5":
        selected = list(rng.choice(topk, size=n_sel, replace=False))
        diag["selected_dc_score_mean"] = float(np.mean(dc_scores[selected])) if selected else float("nan")
        return selected, diag

    norm = _normalize_scores(dc_scores)
    conf = np.asarray(prefix_conf, dtype=float) if len(prefix_conf) == len(queue) else np.ones(len(queue), dtype=float)
    if policy == "dc_topk_diversity_only_m5":
        base = norm[topk]
        lambda_agent = max(0.0, float(agent_diversity_penalty))
    elif policy == "dc_topk_prefix_only_m5":
        base = norm[topk] * conf[topk]
        lambda_agent = 0.0
    else:
        base = norm[topk] * conf[topk]
        lambda_agent = max(0.0, float(agent_diversity_penalty))
    selected_counts = np.zeros(n_agents, dtype=np.int32)
    if lambda_agent > 0:
        selected = _select_with_agent_diversity(
            topk, base.copy(), queue, n_sel, selected_counts, lambda_agent,
            audit=audit, audit_action="Crit-TopK", audit_raw_scores=norm,
            audit_prefix_conf=(conf if policy != "dc_topk_diversity_only_m5" else np.ones(len(queue))),
            tick=tick)
    else:
        order = np.argsort(-base)
        selected = [topk[int(order[i])] for i in range(min(n_sel, len(order)))]
    diag["selected_dc_score_mean"] = float(np.mean(dc_scores[selected])) if selected else float("nan")
    return selected, diag


def _apply_quota_vec(
    quota_aoi: float,
    quota_dc:  float,
    ghost_safe: bool,
    queue:     list[_IntentPt],
    budget:    int,
    tick:      int,
    n_agents:  int,
    est_rc:    np.ndarray,
    pt_route:  np.ndarray,
    n_r0:      int,
    n_r1:      int,
    crit_score: np.ndarray,
    last_rx:   np.ndarray,
    ghost_res: np.ndarray,
    horizon:   int,
    max_age:   int,
    w_margin:  float,
    w_critical: float,
    w_fresh:   float,
    w_ghost:   float,
    rng:       np.random.Generator,
    prefix_conf: np.ndarray | None = None,
    prefix_weight_ranking: bool = False,
    agent_diversity_penalty: float = 0.0,
    audit: DecisionCriticalAuditRecorder | None = None,
    audit_action: str = "",
    normalize_dc_diversity: bool = False,
    e40_invariant_checks: bool = False,
) -> list[int]:
    """Vectorized quota application.  Returns selected queue indices."""
    n_q  = len(queue)
    if n_q == 0:
        return []
    n_sel  = min(budget, n_q)
    n_aoi  = int(quota_aoi * n_sel)
    n_dc   = int(quota_dc  * n_sel)
    n_rand = n_sel - n_aoi - n_dc
    selected: list[int] = []
    avail    = list(range(n_q))
    selected_counts = np.zeros(n_agents, dtype=np.int32)
    lambda_agent = max(0.0, float(agent_diversity_penalty))
    conf_all = (
        np.asarray(prefix_conf, dtype=float)
        if prefix_weight_ranking and prefix_conf is not None and len(prefix_conf) == n_q
        else np.ones(n_q, dtype=float)
    )

    # Pre-compute arrays for all candidates
    pts_all  = np.array([queue[qi].point_idx for qi in avail])
    ags_all  = np.array([queue[qi].agent_id  for qi in avail])
    hz_all   = np.array([float(queue[qi].horizon_step) for qi in avail])

    # AoI expert (vectorized): oldest-first
    if n_aoi > 0 and avail:
        ages  = (tick - last_rx[ags_all]).astype(float) * conf_all[avail]
        take  = min(n_aoi, len(avail))
        if lambda_agent > 0:
            aoi_sel = _select_with_agent_diversity(avail, ages, queue, take, selected_counts, lambda_agent)
        else:
            order = np.argsort(-ages)
            aoi_sel = [avail[order[i]] for i in range(take)]
            for qi in aoi_sel:
                selected_counts[int(queue[qi].agent_id)] += 1
        selected.extend(aoi_sel)
        sel_set = set(aoi_sel)
        avail   = [qi for qi in avail if qi not in sel_set]
        n_rand += (n_aoi - take)
        if avail:
            pts_all = np.array([queue[qi].point_idx for qi in avail])
            ags_all = np.array([queue[qi].agent_id  for qi in avail])
            hz_all  = np.array([float(queue[qi].horizon_step) for qi in avail])

    # DC expert (vectorized)
    if n_dc > 0 and avail:
        r_arr  = pt_route[pts_all].astype(int)
        r_v    = np.where(r_arr >= 0, r_arr, 0)
        opp    = 1 - r_v
        rmr    = np.maximum(0.0, est_rc[opp] - est_rc[r_v])
        rmr    = np.where(r_arr >= 0, rmr, 0.0) + 0.1
        crs    = crit_score[pts_all] + 0.1
        fn     = np.minimum(1.0, (tick - last_rx[ags_all]) / max(max_age, 1)) + 0.1
        gr_pts = ghost_res[r_v] * hz_all / max(horizon, 1)
        gr_pts = np.where(r_arr >= 0, gr_pts, 0.0)
        w_g    = w_ghost * 2.0 if ghost_safe else w_ghost
        raw_scores = (w_margin * rmr) * (w_critical * crs) * (w_fresh * fn) - w_g * gr_pts
        diversity_scores = (
            _normalize_dc_tranche_scores(raw_scores)
            if normalize_dc_diversity else raw_scores
        )
        if e40_invariant_checks and normalize_dc_diversity:
            assert np.all(np.isfinite(diversity_scores))
            assert float(np.min(diversity_scores)) >= -1e-15
            assert float(np.max(diversity_scores)) <= 1.0 + 1e-15
        scores = diversity_scores * conf_all[avail]
        raw_all = np.full(n_q, np.nan, dtype=float)
        raw_all[avail] = raw_scores
        diversity_all = np.full(n_q, np.nan, dtype=float)
        diversity_all[avail] = diversity_scores
        take   = min(n_dc, len(avail))
        if lambda_agent > 0:
            dc_sel = _select_with_agent_diversity(
                avail, scores, queue, take, selected_counts, lambda_agent,
                audit=audit, audit_action=audit_action, audit_raw_scores=raw_all,
                audit_prefix_conf=conf_all, audit_diversity_scores=diversity_all,
                assert_nonnegative_diversity=(e40_invariant_checks and normalize_dc_diversity),
                tick=tick)
        else:
            if audit is not None:
                for pos, qi in enumerate(avail):
                    audit.candidate(
                        audit_action, float(raw_all[qi]), float(conf_all[qi]),
                        int(selected_counts[int(queue[qi].agent_id)]), float(scores[pos]),
                        diversity_score=float(diversity_all[qi]))
            order  = np.argsort(-scores)
            dc_sel = [avail[order[i]] for i in range(take)]
            for qi in dc_sel:
                ag = int(queue[qi].agent_id)
                if audit is not None:
                    audit.selected(audit_action, tick, queue[qi], float(raw_all[qi]),
                                   float(conf_all[qi]), int(selected_counts[ag]),
                                   float(diversity_all[qi] * conf_all[qi]),
                                   diversity_score=float(diversity_all[qi]))
                selected_counts[ag] += 1
        selected.extend(dc_sel)
        sel_set = set(dc_sel)
        avail   = [qi for qi in avail if qi not in sel_set]
        n_rand += (n_dc - take)

    # Random fill
    if n_rand > 0 and avail:
        take     = min(n_rand, len(avail))
        if prefix_weight_ranking or lambda_agent > 0:
            rand_sel = []
            rem = list(avail)
            for _ in range(take):
                weights = conf_all[rem].astype(float)
                if lambda_agent > 0:
                    penalties = np.array([
                        1.0 / (1.0 + lambda_agent * float(selected_counts[int(queue[qi].agent_id)]))
                        for qi in rem
                    ], dtype=float)
                    weights = weights * penalties
                probs = weights / weights.sum() if weights.sum() > 0 else None
                pos = int(rng.choice(len(rem), p=probs))
                qi = rem.pop(pos)
                rand_sel.append(qi)
                selected_counts[int(queue[qi].agent_id)] += 1
        else:
            rand_sel = list(rng.choice(avail, size=take, replace=False))
            for qi in rand_sel:
                selected_counts[int(queue[qi].agent_id)] += 1
        selected.extend(rand_sel)

    return selected[:n_sel]


def _select_baseline_prefix_vec(
    policy: str,
    queue: list[_IntentPt],
    budget: int,
    tick: int,
    est_rc: np.ndarray,
    pt_route: np.ndarray,
    n_r0: int,
    n_r1: int,
    crit_score: np.ndarray,
    last_rx: np.ndarray,
    ghost_res: np.ndarray,
    horizon: int,
    max_age: int,
    w_margin: float,
    w_critical: float,
    w_fresh: float,
    w_ghost: float,
    rng: np.random.Generator,
    prefix_conf: np.ndarray,
) -> list[int]:
    """Point-level baseline selector with prefix confidence weighting."""
    n_q = len(queue)
    if n_q == 0:
        return []
    n_sel = min(budget, n_q)
    conf = np.asarray(prefix_conf, dtype=float)
    if len(conf) != n_q:
        conf = np.ones(n_q, dtype=float)

    if policy == "random_budget":
        probs = conf / conf.sum() if conf.sum() > 0 else None
        return list(rng.choice(n_q, size=n_sel, replace=False, p=probs))

    if policy in {"aoi_priority_scheduler", "aoi_prefix_risk"}:
        ages = np.array([float(tick - last_rx[ip.agent_id]) for ip in queue])
        return list(np.argsort(-(ages * conf))[:n_sel])

    if policy == "decision_critical_scheduler":
        scores = np.empty(n_q, dtype=float)
        for qi, ip in enumerate(queue):
            rmr = _route_margin_risk(ip.point_idx, pt_route, n_r0, n_r1, est_rc) + 0.1
            crs = float(crit_score[ip.point_idx]) + 0.1
            fn = min(1.0, float(tick - last_rx[ip.agent_id]) / max(max_age, 1)) + 0.1
            r = int(pt_route[ip.point_idx])
            gr = float(ghost_res[r] if r >= 0 else 0.0) * float(ip.horizon_step) / max(horizon, 1)
            scores[qi] = ((w_margin * rmr) * (w_critical * crs) * (w_fresh * fn) - w_ghost * gr) * conf[qi]
        return list(np.argsort(-scores)[:n_sel])

    return _e11_select(
        policy=policy, queue=queue, budget=budget, tick=tick,
        n_agents=max((ip.agent_id for ip in queue), default=-1) + 1,
        est_rc=est_rc, pt_route=pt_route, n_r0=n_r0, n_r1=n_r1,
        crit_score=crit_score, bott_mask=np.zeros_like(crit_score, dtype=bool),
        last_rx=last_rx, ghost_res=ghost_res, horizon=horizon, max_age=max_age,
        w_margin=w_margin, w_critical=w_critical, w_fresh=w_fresh,
        w_ghost=w_ghost, rng=rng, svc_counts=None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Context features (for ContextGatedRule and MC diagnostic)
# ─────────────────────────────────────────────────────────────────────────────

def _context_features(
    queue:      list[_IntentPt],
    tick:       int,
    last_rx:    np.ndarray,
    last_est_upd: np.ndarray,
    est_rc:     np.ndarray,
    ghost_res:  np.ndarray,
    crit_score: np.ndarray,
    max_age:    int,
    stale_upd_threshold: int = 8,
) -> dict[str, float]:
    stale_score = min(1.0, float(np.mean(tick - last_rx)) / max(max_age, 1))
    ghost_score = float(ghost_res.mean())
    if queue:
        crit_frac = float(np.mean([1.0 if crit_score[ip.point_idx] >= 1.0 else 0.0
                                    for ip in queue]))
    else:
        crit_frac = 0.0
    margin      = float(abs(est_rc[0] - est_rc[1]))
    small_margin = 1.0 if margin < 0.05 else 0.0
    stale_pts   = float(np.mean((tick - last_est_upd) > stale_upd_threshold))
    return {
        "stale_score":       stale_score,
        "ghost_score":       ghost_score,
        "critical_pressure": crit_frac,
        "small_margin_rate": small_margin,
        "coverage_gap":      stale_pts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Oracle-free online quota baselines
# ─────────────────────────────────────────────────────────────────────────────

def _online_quota_observable_loss(
    est_rc: np.ndarray,
    agent_routes: np.ndarray,
    ghost_res: np.ndarray,
    lambda_regret: float,
    lambda_ghost: float,
) -> float:
    """Bounded observable proxy loss for online quota learning.

    This intentionally uses only the communicated/estimated state.  It does not
    read true_field or any oracle hidden-congestion signal.
    """
    route_costs = est_rc[agent_routes.astype(int)]
    best_cost = float(np.min(est_rc))
    regret = np.maximum(0.0, route_costs - best_cost)
    raw = (
        float(route_costs.mean())
        + lambda_regret * float(regret.mean())
        + lambda_ghost * float(ghost_res.mean())
    )
    return float(np.clip(raw / 3.0, 0.0, 1.0))


def _online_quota_choose_action(
    policy: str,
    epoch_cache: dict,
    loss: float,
    rng: np.random.Generator,
    context: dict[str, float] | None = None,
) -> tuple[str, dict]:
    """Update previous online action and choose the next quota action."""
    actions = ACTION_NAMES
    n_actions = len(actions)
    state = epoch_cache.setdefault("online_quota_state", {})
    diag: dict = {"online_observable_loss": loss}

    if policy == "ucb_quota":
        counts = state.setdefault("counts", {a: 0 for a in actions})
        means = state.setdefault("mean_loss", {a: 0.0 for a in actions})
        last = state.get("last_action")
        if last in actions:
            c = int(counts[last])
            means[last] = (float(means[last]) * c + loss) / float(c + 1)
            counts[last] = c + 1

        t = max(1, sum(int(counts[a]) for a in actions))
        untried = [a for a in actions if int(counts[a]) == 0]
        if untried:
            chosen = untried[0]
        else:
            explore = 0.60
            scores = {
                a: float(means[a]) - explore * math.sqrt(math.log(t + 1.0) / int(counts[a]))
                for a in actions
            }
            chosen = min(actions, key=lambda a: scores[a])
            diag.update({f"ucb_score_{a}": float(scores[a]) for a in actions})
        state["last_action"] = chosen
        diag.update({f"ucb_count_{a}": int(counts[a]) for a in actions})
        diag.update({f"ucb_mean_loss_{a}": float(means[a]) for a in actions})
        return chosen, diag

    if policy == "exp3_quota":
        weights = state.setdefault("weights", {a: 1.0 for a in actions})
        gamma = 0.15
        eta = 0.20

        def _probs() -> dict[str, float]:
            total = max(sum(float(weights[a]) for a in actions), 1e-12)
            return {
                a: (1.0 - gamma) * float(weights[a]) / total + gamma / n_actions
                for a in actions
            }

        probs = _probs()
        last = state.get("last_action")
        last_prob = float(state.get("last_prob", probs.get(last, 1.0 / n_actions)))
        if last in actions:
            est_loss = loss / max(last_prob, 1e-6)
            weights[last] = float(weights[last]) * math.exp(-eta * est_loss / n_actions)
            # Prevent numerical collapse over long sweeps.
            max_w = max(float(weights[a]) for a in actions)
            if max_w > 0:
                for a in actions:
                    weights[a] = float(weights[a]) / max_w
        probs = _probs()
        p = np.array([probs[a] for a in actions], dtype=float)
        p = p / p.sum()
        chosen = str(rng.choice(actions, p=p))
        state["last_action"] = chosen
        state["last_prob"] = float(probs[chosen])
        diag.update({f"exp3_prob_{a}": float(probs[a]) for a in actions})
        diag.update({f"exp3_weight_{a}": float(weights[a]) for a in actions})
        return chosen, diag

    if policy == "contextual_linucb_quota":
        def _feature_vec(ctx: dict[str, float] | None) -> np.ndarray:
            c = ctx or {}
            return np.array([
                1.0,
                float(c.get("queue_size_norm", 0.0)),
                float(c.get("stale_score", 0.0)),
                float(c.get("ghost_score", 0.0)),
                float(c.get("critical_pressure", 0.0)),
                float(c.get("small_margin_rate", 0.0)),
                float(c.get("coverage_gap", 0.0)),
                float(c.get("budget_norm", 0.0)),
                float(c.get("horizon_norm", 0.0)),
                float(c.get("n_agents_norm", 0.0)),
            ], dtype=float)

        x = _feature_vec(context)
        d = int(x.size)
        alpha = 0.35
        ridge = 1.0
        state.setdefault("A", {a: np.eye(d) * ridge for a in actions})
        state.setdefault("b", {a: np.zeros(d, dtype=float) for a in actions})
        state.setdefault("counts", {a: 0 for a in actions})
        last = state.get("last_action")
        last_x = state.get("last_x")
        if last in actions and last_x is not None:
            reward = 1.0 - float(loss)
            lx = np.asarray(last_x, dtype=float)
            state["A"][last] = np.asarray(state["A"][last], dtype=float) + np.outer(lx, lx)
            state["b"][last] = np.asarray(state["b"][last], dtype=float) + reward * lx
            state["counts"][last] = int(state["counts"][last]) + 1

        scores: dict[str, float] = {}
        for a in actions:
            A_inv = np.linalg.pinv(np.asarray(state["A"][a], dtype=float))
            theta = A_inv @ np.asarray(state["b"][a], dtype=float)
            exploit = float(theta @ x)
            explore = alpha * math.sqrt(float(x @ A_inv @ x))
            scores[a] = exploit + explore
        untried = [a for a in actions if int(state["counts"][a]) == 0]
        chosen = untried[0] if untried else max(actions, key=lambda a: scores[a])
        state["last_action"] = chosen
        state["last_x"] = x.copy()
        diag.update({f"ctxbandit_score_{a}": float(scores[a]) for a in actions})
        diag.update({f"ctxbandit_count_{a}": int(state["counts"][a]) for a in actions})
        diag["ctxbandit_feature_dim"] = d
        return chosen, diag

    raise ValueError(f"Unknown online quota policy: {policy!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Optimized MC rollout evaluator (vectorized inner loops, lambda objectives)
# ─────────────────────────────────────────────────────────────────────────────

def _mc_eval_fast(
    quota_aoi:  float,
    quota_dc:   float,
    ghost_safe: bool,
    queue:      list[_IntentPt],
    budget:     int,
    tick:       int,
    n_agents:   int,
    est_field:  np.ndarray,
    last_rx:    np.ndarray,
    agent_routes: np.ndarray,
    ghost_res:  np.ndarray,
    last_est_upd: np.ndarray,
    r0_pts:     np.ndarray,
    r1_pts:     np.ndarray,
    pt_route:   np.ndarray,
    n_r0:       int,
    n_r1:       int,
    crit_score: np.ndarray,
    horizon:    int,
    max_age:    int,
    w_margin:   float,
    w_critical: float,
    w_fresh:    float,
    w_ghost:    float,
    route_switch_margin: float,
    H:          int,
    K:          int,
    lambda_route:   float,
    lambda_regret:  float,
    lambda_osc:     float,
    lambda_ghost:   float,
    est_decay_rate: float,
    rng:        np.random.Generator,
    prefix_trust_mode: str = "none",
    prefix_beta: float = 0.15,
    prefix_confidence_floor: float = 0.20,
    prefix_weight_ranking: bool = False,
    agent_diversity_penalty: float = 0.0,
    normalize_dc_diversity: bool = False,
    e40_invariant_checks: bool = False,
) -> tuple[float, float]:
    """K vectorized rollouts.  No oracle access.  Returns (J_mean, J_std)."""
    n_pts = len(est_field)
    J_samples: list[float] = []

    for _ in range(K):
        ef  = est_field.copy()
        ar  = agent_routes.copy().astype(np.int32)
        gr  = ghost_res.copy()
        lrx = last_rx.copy()
        leu = last_est_upd.copy()

        est_rc_r = np.array([ef[r0_pts].mean(), ef[r1_pts].mean()])
        prefix_conf = _prefix_confidence_vec(
            queue, ef, prefix_trust_mode, prefix_beta, prefix_confidence_floor, tick=tick)

        sel = _apply_quota_vec(
            quota_aoi, quota_dc, ghost_safe,
            queue, budget, tick, n_agents,
            est_rc_r, pt_route, n_r0, n_r1,
            crit_score, lrx, gr, horizon, max_age,
            w_margin, w_critical, w_fresh, w_ghost, rng,
            prefix_conf=prefix_conf,
            prefix_weight_ranking=prefix_weight_ranking,
            agent_diversity_penalty=agent_diversity_penalty,
            normalize_dc_diversity=normalize_dc_diversity,
            e40_invariant_checks=e40_invariant_checks,
        )
        for qi in sel:
            ip = queue[qi]
            ef[ip.point_idx] = float(np.clip(
                ef[ip.point_idx] + rng.normal(0.0, 0.02), 0.02, 0.98))
            leu[ip.point_idx] = tick
            lrx[ip.agent_id]  = tick

        route_risk_sum = 0.0
        regret_sum     = 0.0
        osc_count      = 0

        for h in range(H):
            stale = (tick + h - leu) > 3
            ef[stale] += est_decay_rate * (0.5 - ef[stale])
            ef += rng.normal(0.0, 0.004, n_pts)
            ef  = np.clip(ef, 0.02, 0.98)

            est_rc_h = np.array([ef[r0_pts].mean(), ef[r1_pts].mean()])

            # Vectorized route decisions
            alt  = 1 - ar
            sn   = rng.normal(0.0, 0.008, n_agents)
            sw   = est_rc_h[alt] < est_rc_h[ar] - route_switch_margin + sn
            osc_count += int(sw.sum())
            ar = np.where(sw, alt, ar)

            gr *= 0.90

            route_costs    = est_rc_h[ar]
            route_risk_sum += float(route_costs.mean())
            best_c          = float(est_rc_h.min())
            regret_sum     += float(np.maximum(0.0, route_costs - best_c).mean())

        route_risk_sum /= max(H, 1)
        regret_sum     /= max(H, 1)

        J = (lambda_route  * route_risk_sum
             + lambda_regret * regret_sum
             + lambda_osc    * (osc_count / max(H * n_agents, 1))
             + lambda_ghost  * float(ghost_res.mean()))
        J_samples.append(J)

    J_arr = np.array(J_samples, dtype=float)
    return float(J_arr.mean()), float(J_arr.std())


# ─────────────────────────────────────────────────────────────────────────────
# Selection dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def _select_staged(
    policy:        str,
    queue:         list[_IntentPt],
    budget:        int,
    tick:          int,
    n_agents:      int,
    est_rc:        np.ndarray,
    est_field:     np.ndarray,
    pt_route:      np.ndarray,
    r0_pts:        np.ndarray,
    r1_pts:        np.ndarray,
    n_r0:          int,
    n_r1:          int,
    crit_score:    np.ndarray,
    bott_mask:     np.ndarray,
    last_rx:       np.ndarray,
    last_est_upd:  np.ndarray,
    agent_routes:  np.ndarray,
    ghost_res:     np.ndarray,
    horizon:       int,
    max_age:       int,
    w_margin:      float,
    w_critical:    float,
    w_fresh:       float,
    w_ghost:       float,
    route_switch_margin: float,
    rng:           np.random.Generator,
    svc_counts:    np.ndarray | None,
    # MC params
    rollout_horizon: int,
    n_rollouts:    int,
    lambda_route:  float,
    lambda_regret: float,
    lambda_osc:    float,
    lambda_ghost:  float,
    est_decay_rate: float,
    prefix_trust_mode: str,
    prefix_beta: float,
    prefix_confidence_floor: float,
    prefix_weight_ranking: bool,
    agent_diversity_penalty: float,
    # Decision-epoch cache (mutable dict passed by reference)
    epoch_cache:   dict,
    decision_epoch: int,
    audit: DecisionCriticalAuditRecorder | None = None,
    e40_invariant_checks: bool = False,
) -> tuple[list[int], dict]:
    """Return (selected_queue_indices, diagnostic_dict)."""
    diag: dict = {}
    normalize_dc_diversity = policy in NORMALIZED_DC_DIVERSITY_POLICIES
    diag["normalized_dc_diversity"] = normalize_dc_diversity
    prefix_conf = _prefix_confidence_vec(
        queue, est_field, prefix_trust_mode, prefix_beta, prefix_confidence_floor, tick=tick)
    if len(prefix_conf):
        diag["prefix_confidence_mean"] = float(prefix_conf.mean())
        diag["prefix_confidence_min"] = float(prefix_conf.min())
        diag["prefix_confidence_max"] = float(prefix_conf.max())

    # ── Baseline policies ──────────────────────────────────────────────────
    if policy == "aoi_prefix_risk" or (policy in BASELINE_POLICIES and prefix_weight_ranking and prefix_trust_mode != "none"):
        idx = _select_baseline_prefix_vec(
            policy=policy, queue=queue, budget=budget, tick=tick,
            est_rc=est_rc, pt_route=pt_route, n_r0=n_r0, n_r1=n_r1,
            crit_score=crit_score, last_rx=last_rx, ghost_res=ghost_res,
            horizon=horizon, max_age=max_age, w_margin=w_margin,
            w_critical=w_critical, w_fresh=w_fresh, w_ghost=w_ghost,
            rng=rng, prefix_conf=prefix_conf,
        )
        return idx, diag

    if policy in DECISION_CRITICAL_CAP_POLICIES or policy in DC_TOPK_PREFIX_DIVERSE_MULTIPLIER:
        t_start = time.perf_counter()
        dc_scores = _decision_critical_score_vec(
            queue, tick, est_rc, pt_route, n_r0, n_r1, crit_score, last_rx,
            ghost_res, horizon, max_age, w_margin, w_critical, w_fresh, w_ghost)
        if policy in DECISION_CRITICAL_CAP_POLICIES:
            idx = _select_decision_critical_cap(policy, queue, budget, dc_scores, n_agents)
            diag["selected_dc_score_mean"] = float(np.mean(dc_scores[idx])) if idx else float("nan")
            diag["dc_cap_policy"] = policy
        else:
            idx, topk_diag = _select_dc_topk_prefix_diverse(
                policy, queue, budget, dc_scores, prefix_conf, n_agents, rng,
                agent_diversity_penalty=agent_diversity_penalty,
                audit=audit, tick=tick)
            diag.update(topk_diag)
        diag["selection_runtime_ms"] = (time.perf_counter() - t_start) * 1000.0
        diag["from_cache"] = True
        return idx, diag

    if policy in BASELINE_POLICIES:
        idx = _e11_select(
            policy=policy, queue=queue, budget=budget, tick=tick,
            n_agents=n_agents, est_rc=est_rc, pt_route=pt_route,
            n_r0=n_r0, n_r1=n_r1, crit_score=crit_score, bott_mask=bott_mask,
            last_rx=last_rx, ghost_res=ghost_res, horizon=horizon,
            max_age=max_age, w_margin=w_margin, w_critical=w_critical,
            w_fresh=w_fresh, w_ghost=w_ghost, rng=rng, svc_counts=svc_counts,
        )
        return idx, diag

    # ── Fixed-quota controls ───────────────────────────────────────────────
    if policy in FIXED_QUOTA_MAP:
        q_aoi, q_dc, gs = FIXED_QUOTA_MAP[policy]
        diag["selected_action"] = policy
        idx = _apply_quota_vec(
            q_aoi, q_dc, gs, queue, budget, tick, n_agents,
            est_rc, pt_route, n_r0, n_r1, crit_score, last_rx, ghost_res,
            horizon, max_age, w_margin, w_critical, w_fresh, w_ghost, rng,
            prefix_conf=prefix_conf, prefix_weight_ranking=prefix_weight_ranking,
            agent_diversity_penalty=agent_diversity_penalty,
            audit=audit, audit_action=policy,
            normalize_dc_diversity=normalize_dc_diversity,
            e40_invariant_checks=e40_invariant_checks,
        )
        return idx, diag

    # ── Context features (used by CGR and MC) ─────────────────────────────
    ctx = _context_features(queue, tick, last_rx, last_est_upd, est_rc,
                             ghost_res, crit_score, max_age)
    diag.update({f"ctx_{k}": v for k, v in ctx.items()})

    # ── ContextGatedRule ───────────────────────────────────────────────────
    if policy == "context_gated_rule":
        if ctx["stale_score"] > 0.6 or ctx["ghost_score"] > 0.3:
            chosen = "AoIHeavy"
        elif ctx["critical_pressure"] > 0.5 or ctx["small_margin_rate"] == 1.0:
            chosen = "CriticalHeavy"
        else:
            chosen = "CoverageHeavy"
        diag["selected_action"] = chosen
        q_aoi, q_dc, gs = QUOTA_ACTIONS[chosen]
        idx = _apply_quota_vec(
            q_aoi, q_dc, gs, queue, budget, tick, n_agents,
            est_rc, pt_route, n_r0, n_r1, crit_score, last_rx, ghost_res,
            horizon, max_age, w_margin, w_critical, w_fresh, w_ghost, rng,
            prefix_conf=prefix_conf, prefix_weight_ranking=prefix_weight_ranking,
            agent_diversity_penalty=agent_diversity_penalty,
            audit=audit, audit_action=chosen,
            normalize_dc_diversity=normalize_dc_diversity,
            e40_invariant_checks=e40_invariant_checks,
        )
        return idx, diag

    # ── E28 no-rollout prefix-diverse static actions ────────────────────────
    if policy in PREFIX_DIVERSE_NO_ROLLOUT_MAP:
        chosen = PREFIX_DIVERSE_NO_ROLLOUT_MAP[policy]
        q_aoi, q_dc, gs = QUOTA_ACTIONS[chosen]
        diag["selected_action"] = chosen
        diag["no_rollout_static_action"] = True
        diag["runtime_per_decision_ms"] = 0.0
        diag["from_cache"] = True
        idx = _apply_quota_vec(
            q_aoi, q_dc, gs, queue, budget, tick, n_agents,
            est_rc, pt_route, n_r0, n_r1, crit_score, last_rx, ghost_res,
            horizon, max_age, w_margin, w_critical, w_fresh, w_ghost, rng,
            prefix_conf=prefix_conf, prefix_weight_ranking=prefix_weight_ranking,
            agent_diversity_penalty=agent_diversity_penalty,
            audit=audit, audit_action=chosen,
            normalize_dc_diversity=normalize_dc_diversity,
            e40_invariant_checks=e40_invariant_checks,
        )
        return idx, diag

    # ── Oracle-free online quota bandits ───────────────────────────────────
    if policy in ONLINE_QUOTA_POLICIES:
        if len(queue) == 0:
            return [], diag
        is_decision_tick = (decision_epoch <= 1) or (tick % decision_epoch == 0)
        if not is_decision_tick and "action" in epoch_cache:
            chosen = epoch_cache["action"]
            q_aoi, q_dc, gs = QUOTA_ACTIONS[chosen]
            diag["selected_action"] = chosen
            diag["from_cache"] = True
            idx = _apply_quota_vec(
                q_aoi, q_dc, gs, queue, budget, tick, n_agents,
                est_rc, pt_route, n_r0, n_r1, crit_score, last_rx, ghost_res,
                horizon, max_age, w_margin, w_critical, w_fresh, w_ghost, rng,
                prefix_conf=prefix_conf, prefix_weight_ranking=prefix_weight_ranking,
                agent_diversity_penalty=agent_diversity_penalty,
                audit=audit, audit_action=chosen,
                normalize_dc_diversity=normalize_dc_diversity,
                e40_invariant_checks=e40_invariant_checks,
            )
            return idx, diag

        loss = _online_quota_observable_loss(
            est_rc=est_rc,
            agent_routes=agent_routes,
            ghost_res=ghost_res,
            lambda_regret=lambda_regret,
            lambda_ghost=lambda_ghost,
        )
        online_ctx = dict(ctx)
        online_ctx.update({
            "queue_size_norm": min(1.0, len(queue) / max(n_agents * horizon, 1)),
            "budget_norm": min(1.0, budget / max(n_agents, 1)),
            "horizon_norm": min(1.0, horizon / 20.0),
            "n_agents_norm": min(1.0, n_agents / 100.0),
        })
        chosen, online_diag = _online_quota_choose_action(
            policy=policy, epoch_cache=epoch_cache, loss=loss, rng=rng,
            context=online_ctx)
        diag.update(online_diag)
        diag["selected_action"] = chosen
        diag["runtime_per_decision_ms"] = 0.0
        diag["from_cache"] = False
        epoch_cache["action"] = chosen
        epoch_cache["last_decision_tick"] = tick
        q_aoi, q_dc, gs = QUOTA_ACTIONS[chosen]
        idx = _apply_quota_vec(
            q_aoi, q_dc, gs, queue, budget, tick, n_agents,
            est_rc, pt_route, n_r0, n_r1, crit_score, last_rx, ghost_res,
            horizon, max_age, w_margin, w_critical, w_fresh, w_ghost, rng,
            prefix_conf=prefix_conf, prefix_weight_ranking=prefix_weight_ranking,
            agent_diversity_penalty=agent_diversity_penalty,
            audit=audit, audit_action=chosen,
            normalize_dc_diversity=normalize_dc_diversity,
            e40_invariant_checks=e40_invariant_checks,
        )
        return idx, diag

    # ── MC family ─────────────────────────────────────────────────────────
    if policy in MC_FAMILY:
        if len(queue) == 0:
            return [], diag

        # Check decision epoch cache
        is_decision_tick = (decision_epoch <= 1) or (tick % decision_epoch == 0)
        if not is_decision_tick and "action" in epoch_cache:
            chosen   = epoch_cache["action"]
            q_aoi, q_dc, gs = QUOTA_ACTIONS[chosen]
            diag["selected_action"] = chosen
            diag["from_cache"] = True
            idx = _apply_quota_vec(
                q_aoi, q_dc, gs, queue, budget, tick, n_agents,
                est_rc, pt_route, n_r0, n_r1, crit_score, last_rx, ghost_res,
                horizon, max_age, w_margin, w_critical, w_fresh, w_ghost, rng,
                prefix_conf=prefix_conf, prefix_weight_ranking=prefix_weight_ranking,
                agent_diversity_penalty=agent_diversity_penalty,
                audit=audit, audit_action=chosen,
                normalize_dc_diversity=normalize_dc_diversity,
                e40_invariant_checks=e40_invariant_checks,
            )
            return idx, diag

        # Run MC evaluation
        t_start  = time.perf_counter()
        best_J   = float("inf")
        best_act = "Balanced"
        fallback = 0

        for aname, (q_aoi, q_dc, gs) in QUOTA_ACTIONS.items():
            try:
                J_mean, J_std = _mc_eval_fast(
                    quota_aoi=q_aoi, quota_dc=q_dc, ghost_safe=gs,
                    queue=queue, budget=budget, tick=tick, n_agents=n_agents,
                    est_field=est_field, last_rx=last_rx,
                    agent_routes=agent_routes, ghost_res=ghost_res,
                    last_est_upd=last_est_upd,
                    r0_pts=r0_pts, r1_pts=r1_pts,
                    pt_route=pt_route, n_r0=n_r0, n_r1=n_r1,
                    crit_score=crit_score,
                    horizon=horizon, max_age=max_age,
                    w_margin=w_margin, w_critical=w_critical,
                    w_fresh=w_fresh, w_ghost=w_ghost,
                    route_switch_margin=route_switch_margin,
                    H=rollout_horizon, K=n_rollouts,
                    lambda_route=lambda_route, lambda_regret=lambda_regret,
                    lambda_osc=lambda_osc, lambda_ghost=lambda_ghost,
                    est_decay_rate=est_decay_rate, rng=rng,
                    prefix_trust_mode=prefix_trust_mode,
                    prefix_beta=prefix_beta,
                    prefix_confidence_floor=prefix_confidence_floor,
                    prefix_weight_ranking=prefix_weight_ranking,
                    agent_diversity_penalty=agent_diversity_penalty,
                    normalize_dc_diversity=normalize_dc_diversity,
                    e40_invariant_checks=e40_invariant_checks,
                )
                diag[f"J_{aname}"]     = J_mean
                diag[f"J_std_{aname}"] = J_std
                if J_mean < best_J:
                    best_J   = J_mean
                    best_act = aname
            except Exception:
                fallback += 1
                diag[f"J_{aname}"]     = float("nan")
                diag[f"J_std_{aname}"] = float("nan")

        rt_ms = (time.perf_counter() - t_start) * 1000.0
        diag["selected_action"]      = best_act
        diag["fallback_count"]        = fallback
        diag["runtime_per_decision_ms"] = rt_ms
        diag["from_cache"]            = False

        epoch_cache["action"]          = best_act
        epoch_cache["last_decision_tick"] = tick

        q_aoi, q_dc, gs = QUOTA_ACTIONS[best_act]
        idx = _apply_quota_vec(
            q_aoi, q_dc, gs, queue, budget, tick, n_agents,
            est_rc, pt_route, n_r0, n_r1, crit_score, last_rx, ghost_res,
            horizon, max_age, w_margin, w_critical, w_fresh, w_ghost, rng,
            prefix_conf=prefix_conf, prefix_weight_ranking=prefix_weight_ranking,
            agent_diversity_penalty=agent_diversity_penalty,
            audit=audit, audit_action=best_act,
            normalize_dc_diversity=normalize_dc_diversity,
            e40_invariant_checks=e40_invariant_checks,
        )
        return idx, diag

    raise ValueError(f"Unknown E12 staged policy: {policy!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_e12_staged_single(cfg: E12StagedConfig) -> dict:  # noqa: C901
    rng  = np.random.default_rng(cfg.seed)
    topo = _TOPO[cfg.scenario]
    audit = DecisionCriticalAuditRecorder(cfg) if cfg.audit_enabled else None
    receiver_variant = str(cfg.receiver_memory_variant)
    if cfg.strict_valid_receiver_memory and receiver_variant == "e40_persistent":
        receiver_variant = "e41_f0_sequential"
    strict_receiver_enabled = receiver_variant != "e40_persistent"

    n0  = topo.n_r0_excl
    n1  = topo.n_r1_excl
    ns  = topo.n_shared
    n_pts = n0 + n1 + ns

    r0_excl = np.arange(0, n0)
    r1_excl = np.arange(n0, n0 + n1)
    shared  = np.arange(n0 + n1, n_pts)
    r0_pts  = np.concatenate([r0_excl, shared]) if ns else r0_excl
    r1_pts  = np.concatenate([r1_excl, shared]) if ns else r1_excl
    n_r0, n_r1 = len(r0_pts), len(r1_pts)

    pt_route = np.full(n_pts, -1, dtype=np.int8)
    pt_route[r0_excl] = 0
    pt_route[r1_excl] = 1

    crit_mask  = np.zeros(n_pts, dtype=bool)
    bott_mask  = np.zeros(n_pts, dtype=bool)
    for i in topo.critical_idxs:
        if i < n_pts: crit_mask[i] = True
    for i in topo.bottleneck_idxs:
        if i < n_pts: bott_mask[i] = True
    crit_score     = np.where(bott_mask, 1.5, np.where(crit_mask, 1.0, 0.1)).astype(np.float32)
    act_crit_mask  = crit_mask.copy()
    act_bott_mask  = bott_mask.copy()
    act_crit_score = crit_score.copy()
    crit_pts_set   = set(np.where(crit_mask)[0])

    true_field   = rng.uniform(topo.field_lo, topo.field_hi, n_pts).astype(np.float32)
    if topo.route1_bias > 0:
        true_field[r1_excl] = np.clip(true_field[r1_excl] + float(topo.route1_bias), 0.02, 0.98)
    est_field    = np.full(n_pts, 0.50, dtype=np.float32)
    last_est_upd = np.full(n_pts, -cfg.max_queue_age, dtype=np.int32)

    n             = cfg.n_agents
    agent_routes  = rng.integers(0, 2, size=n).astype(np.int8)
    last_rx       = np.full(n, -cfg.max_queue_age, dtype=np.int32)
    recent_switch = np.full(n, -20, dtype=np.int32)
    svc_counts    = np.zeros(n, dtype=np.int32)
    ghost_res     = np.zeros(2, dtype=np.float32)
    epoch_cache: dict = {}

    is_stress      = cfg.regime == "ghost_stress"
    spike_prob_eff = float(topo.spike_prob) * (2.2 if is_stress else 1.0)
    est_decay_rate = 0.018 if is_stress else 0.009

    # ── Accumulators ──────────────────────────────────────────────────────────
    acc_rc_true:  list[float] = []
    acc_regret:   list[float] = []
    acc_agree:    list[int]   = []
    acc_abs_gap:  list[float] = []
    acc_fl:       list[float] = []
    acc_hidden:   list[float] = []
    acc_ghost:    list[float] = []
    sel_rmr:  list[float] = []
    sel_crs:  list[float] = []
    sel_fn:   list[float] = []
    sel_gr:   list[float] = []
    sel_cf:   list[float] = []
    sel_bf:   list[float] = []
    crit_cov: list[float] = []

    fp_replans   = 0
    un_replans   = 0
    oscillations = 0

    action_counts: dict[str, int] = {a: 0 for a in ACTION_NAMES}
    J_sums:        dict[str, list[float]] = {a: [] for a in ACTION_NAMES}
    runtime_list:  list[float] = []   # per-decision runtimes (ms)
    selection_runtime_tick_list: list[float] = []
    topk_before_hist: list[float] = []
    topk_after_hist: list[float] = []
    topk_value_hist: list[float] = []
    selected_dc_score_hist: list[float] = []
    fallback_tot   = 0
    ctx_stale_l:   list[float] = []
    ctx_ghost_l:   list[float] = []
    ctx_crit_l:    list[float] = []
    n_decisions    = 0  # number of MC evaluations
    prefix_conf_sum = np.zeros(cfg.horizon + 1, dtype=float)
    prefix_conf_count = np.zeros(cfg.horizon + 1, dtype=float)
    admitted_horizon_count = np.zeros(cfg.horizon + 1, dtype=float)
    queue_size_hist: list[int] = []
    live_queue_size_raw_hist: list[int] = []
    live_queue_size_valid_hist: list[int] = []
    new_candidate_count_hist: list[int] = []
    queue_max_age_removed_hist: list[int] = []
    valid_filter_before_hist: list[int] = []
    valid_filter_after_hist: list[int] = []
    valid_filter_removed_hist: list[int] = []
    valid_filter_removed_frac_hist: list[float] = []
    live_queue_expired_target_frac_hist: list[float] = []
    live_queue_age_ge_ttl_frac_hist: list[float] = []
    live_queue_stale_not_expired_mean_hist: list[float] = []
    admitted_points_tick_hist: list[int] = []
    admitted_unique_agents_tick_hist: list[int] = []
    admitted_same_agent_conc_hist: list[float] = []
    admitted_age_counts = np.zeros(cfg.max_queue_age + 1, dtype=float)
    remaining_horizon_values = list(range(-cfg.max_queue_age, cfg.horizon + 1))
    remaining_horizon_counts: dict[int, float] = {v: 0.0 for v in remaining_horizon_values}
    prefix_conf_remaining_sum: dict[int, float] = {v: 0.0 for v in remaining_horizon_values}
    prefix_conf_remaining_count: dict[int, float] = {v: 0.0 for v in remaining_horizon_values}

    drift_dir   = 1.0
    shift_done  = False
    queue: list[_IntentPt] = []
    channel_enabled = cfg.channel_profile != "none"
    if cfg.channel_profile not in CHANNEL_PROFILES:
        raise ValueError(f"Unknown channel_profile: {cfg.channel_profile!r}")
    channel_rng = np.random.default_rng(
        cfg.seed + 104729 + 7919 * _channel_profile_index(cfg.channel_profile))
    channel_cache: dict = {}
    pending_deliveries: list[dict] = []
    receiver_supports: list[_ReceiverSupport] = []
    strict_removed_target_total = 0
    strict_removed_age_total = 0
    strict_invalid_target_total = 0
    strict_invalid_age_total = 0
    strict_expired_contribution_count = 0
    strict_active_count_hist: list[int] = []
    strict_fcomm_min_hist: list[float] = []
    strict_fcomm_max_hist: list[float] = []
    strict_ffull_min_hist: list[float] = []
    strict_ffull_max_hist: list[float] = []
    admitted_points = 0
    delivered_points = 0
    dropped_points = 0
    expired_points = 0
    admitted_remaining_le0_count = 0
    admitted_remaining_total = 0
    field_update_remaining_le0_count = 0
    field_update_total = 0
    admitted_remaining_min: int | None = None
    admitted_remaining_values: list[int] = []
    delivery_delays: list[float] = []
    effective_budget_hist: list[float] = []
    route_cost_pre_revision: list[float] = []
    route_cost_during_revision: list[float] = []
    route_cost_post_revision: list[float] = []
    revised_agent_admit_frac_hist: list[float] = []
    non_revised_unique_agents_tick_hist: list[int] = []
    abandoned_route_admit_frac_after_revision_hist: list[float] = []
    abandoned_route_pressure_after_revision_hist: list[float] = []
    new_route_pressure_after_revision_hist: list[float] = []
    route_revision_done = False
    route_revision_events = 0
    route_revision_forced_agents = 0
    route_revision_revised_mask = np.zeros(n, dtype=bool)
    route_revision_old_routes = np.full(n, -1, dtype=np.int8)
    route_revision_new_routes = np.full(n, -1, dtype=np.int8)
    starvation_active_masks: list[np.ndarray] = []
    starvation_admitted_counts: list[np.ndarray] = []
    starvation_route_costs: list[np.ndarray] = []
    starvation_admission_counts_total = np.zeros(n, dtype=np.float64)

    for tick in range(cfg.n_ticks):
        # Bottleneck shift
        if (topo.bottleneck_shift_tick > 0 and not shift_done
                and tick >= topo.bottleneck_shift_tick):
            ob = list(topo.bottleneck_idxs)
            nb = [ob[-1] + 1, ob[-1] + 2]
            nb = [i for i in nb if i < n_pts]
            for i in ob:
                act_bott_mask[i] = False; act_crit_mask[i] = False
            for i in nb:
                act_bott_mask[i] = True;  act_crit_mask[i] = True
            act_crit_score[:] = np.where(
                act_bott_mask, 1.5, np.where(act_crit_mask, 1.0, 0.1))
            shift_done = True

        # Controlled diagnostic route revision. This is an environment event:
        # schedulers do not receive a flag and only see the subsequent online
        # future-intention stream.
        if (cfg.route_revision_enabled and not route_revision_done
                and tick >= int(cfg.route_revision_tick)):
            n_revise = max(1, int(math.ceil(float(cfg.route_revision_fraction) * n)))
            route_counts = np.bincount(agent_routes.astype(int), minlength=2)
            route_cost_proxy = np.array([
                float(true_field[r0_pts].mean()),
                float(true_field[r1_pts].mean()),
            ])
            target_route = int(np.argmax(route_cost_proxy + 1e-6 * route_counts))
            candidates = np.where(agent_routes == target_route)[0]
            if len(candidates) < n_revise:
                candidates = np.arange(n)
            if float(cfg.route_revision_probability) >= 1.0:
                selected_agents = candidates[:min(n_revise, len(candidates))]
            else:
                draw = rng.random(len(candidates)) < float(cfg.route_revision_probability)
                selected_agents = candidates[draw][:n_revise]
                if len(selected_agents) == 0 and len(candidates):
                    selected_agents = candidates[:1]
            for ag in selected_agents:
                old_r = int(agent_routes[ag])
                new_r = 1 - old_r
                route_revision_revised_mask[int(ag)] = True
                route_revision_old_routes[int(ag)] = np.int8(old_r)
                route_revision_new_routes[int(ag)] = np.int8(new_r)
                agent_routes[int(ag)] = np.int8(new_r)
                ghost_res[old_r] = min(2.0, float(ghost_res[old_r]) + 1.0 / n)
            route_revision_events = 1
            route_revision_forced_agents = int(len(selected_agents))
            route_revision_done = True

        # Field dynamics
        if tick % 60 == 0:
            drift_dir = -drift_dir
        noise = rng.normal(0.0, float(topo.noise_std), n_pts).astype(np.float32)
        true_field = np.clip(
            true_field + drift_dir * float(topo.drift_rate) + noise, 0.02, 0.98
        ).astype(np.float32)
        if topo.route1_bias > 0:
            true_field[r1_excl] = np.clip(
                true_field[r1_excl] + 0.005 * float(topo.route1_bias), 0.02, 0.98
            ).astype(np.float32)
        for rid in range(2):
            if ghost_res[rid] > 0.08:
                excl = r0_excl if rid == 0 else r1_excl
                spk  = rng.random(len(excl)) < (spike_prob_eff * (1.0 + float(ghost_res[rid])))
                true_field[excl[spk]] = np.clip(
                    true_field[excl[spk]] + float(topo.spike_mag) * float(ghost_res[rid]),
                    0.02, 0.98).astype(np.float32)
        if ns > 0:
            occ     = float(np.sum(np.isin(agent_routes, [0, 1]))) / max(n, 1)
            bpts    = np.where(act_bott_mask)[0]
            if len(bpts):
                true_field[bpts] = np.clip(
                    true_field[bpts] + 0.018 * occ, 0.02, 0.98).astype(np.float32)

        ghost_res *= (1.0 - float(topo.ghost_decay))
        if strict_receiver_enabled:
            est_field, last_est_upd, receiver_supports, strict_diag = (
                _strict_receiver_recompute(
                    n_pts, receiver_supports, tick, cfg.max_queue_age,
                    est_decay_rate, receiver_variant))
            strict_removed_target_total += int(strict_diag["removed_target"])
            strict_removed_age_total += int(strict_diag["removed_age"])
            strict_invalid_target_total += int(strict_diag["invalid_target"])
            strict_invalid_age_total += int(strict_diag["invalid_age"])
        else:
            stale = (tick - last_est_upd) > 3
            est_field[stale] += est_decay_rate * (0.5 - est_field[stale])

        if channel_enabled and pending_deliveries:
            still_pending: list[dict] = []
            for pkt in pending_deliveries:
                if int(pkt["deliver_tick"]) > tick:
                    still_pending.append(pkt)
                    continue
                if (CHANNEL_PROFILES[cfg.channel_profile].get("expiration_enabled", False)
                        and tick - int(pkt["tick_created"]) > cfg.max_queue_age):
                    expired_points += 1
                    continue
                p = int(pkt["point_idx"])
                ag = int(pkt["agent_id"])
                remaining_at_delivery = int(pkt.get("target_tick", tick + 1)) - int(tick)
                if cfg.valid_future_only and remaining_at_delivery <= 0:
                    expired_points += 1
                    continue
                conf = float(pkt.get("prefix_confidence", 1.0))
                obs_v = float(np.clip(float(pkt["obs"]), 0.02, 0.98))
                if strict_receiver_enabled:
                    receiver_supports.append(_ReceiverSupport(
                        point_idx=p,
                        target_tick=int(pkt["target_tick"]),
                        received_tick=tick,
                        observation=obs_v,
                        refresh_gain=(
                            conf if cfg.prefix_weight_field
                            and cfg.prefix_trust_mode != "none" else 1.0),
                        agent_id=ag,
                        horizon_k=int(pkt["horizon_k"]),
                        generation_tick=int(pkt["tick_created"]),
                        admission_rank=int(pkt["admission_rank"]),
                    ))
                elif cfg.prefix_weight_field and cfg.prefix_trust_mode != "none":
                    est_field[p] = float(np.clip((1.0 - conf) * est_field[p] + conf * obs_v, 0.02, 0.98))
                else:
                    est_field[p] = obs_v
                last_est_upd[p] = tick
                last_rx[ag] = tick
                svc_counts[ag] += 1
                field_update_total += 1
                delivered_points += 1
                delivery_delays.append(float(tick - int(pkt["admit_tick"])))
            pending_deliveries = still_pending
            if strict_receiver_enabled:
                est_field, last_est_upd, receiver_supports, strict_diag = (
                    _strict_receiver_recompute(
                        n_pts, receiver_supports, tick, cfg.max_queue_age,
                        est_decay_rate, receiver_variant))
                strict_removed_target_total += int(strict_diag["removed_target"])
                strict_removed_age_total += int(strict_diag["removed_age"])
                strict_invalid_target_total += int(strict_diag["invalid_target"])
                strict_invalid_age_total += int(strict_diag["invalid_age"])

        true_rc = np.array([true_field[r0_pts].mean(), true_field[r1_pts].mean()])
        est_rc  = np.array([est_field[r0_pts].mean(),  est_field[r1_pts].mean()])

        # Intention queue
        new_pts = [
            _IntentPt(ag, int((r0_pts if agent_routes[ag] == 0 else r1_pts)[
                (ag * cfg.horizon + h + tick) % (n_r0 if agent_routes[ag] == 0 else n_r1)]),
                tick, h)
            for ag in range(n) for h in range(1, cfg.horizon + 1)
        ]
        queue.extend(new_pts)
        before_filter_n = len(queue)
        queue = [ip for ip in queue if tick - ip.tick_created <= cfg.max_queue_age]
        queue_max_age_removed_hist.append(before_filter_n - len(queue))
        live_queue_size_raw_hist.append(len(queue))
        valid_filter_before = len(queue)
        if cfg.valid_future_only or cfg.drop_nonpositive_remaining_horizon:
            queue = [ip for ip in queue if _intent_remaining_horizon(ip, tick) > 0]
        elif cfg.drop_expired_intention_points:
            queue = [ip for ip in queue if _intent_remaining_horizon(ip, tick) >= 0]
        valid_filter_after = len(queue)
        valid_filter_removed = valid_filter_before - valid_filter_after
        valid_filter_before_hist.append(valid_filter_before)
        valid_filter_after_hist.append(valid_filter_after)
        valid_filter_removed_hist.append(valid_filter_removed)
        valid_filter_removed_frac_hist.append(
            float(valid_filter_removed / max(valid_filter_before, 1)))
        live_queue_size_valid_hist.append(valid_filter_after)
        tick_active_agents = np.zeros(n, dtype=bool)
        if queue:
            tick_active_agents[np.fromiter((int(ip.agent_id) for ip in queue), dtype=int)] = True
        prefix_conf_all = _prefix_confidence_vec(
            queue, est_field, cfg.prefix_trust_mode,
            cfg.prefix_beta, cfg.prefix_confidence_floor, tick=tick)
        new_candidate_count_hist.append(len(new_pts))
        queue_size_hist.append(len(queue))
        if queue:
            ages_q = np.array([tick - ip.tick_created for ip in queue], dtype=float)
            rem_q = np.array([_intent_remaining_horizon(ip, tick) for ip in queue], dtype=float)
            live_queue_expired_target_frac_hist.append(float(np.mean(rem_q <= 0)))
            live_queue_age_ge_ttl_frac_hist.append(float(np.mean(ages_q >= cfg.max_queue_age)))
            live_queue_stale_not_expired_mean_hist.append(float(np.mean((ages_q > 0) & (ages_q < cfg.max_queue_age))))
        else:
            live_queue_expired_target_frac_hist.append(0.0)
            live_queue_age_ge_ttl_frac_hist.append(0.0)
            live_queue_stale_not_expired_mean_hist.append(0.0)

        effective_budget = _channel_effective_budget(
            cfg.budget_per_tick, cfg.channel_profile, tick, channel_rng, channel_cache)
        effective_budget_hist.append(float(effective_budget))

        # Selection
        sel_idx, tick_diag = _select_staged(
            policy=cfg.policy, queue=queue, budget=effective_budget,
            tick=tick, n_agents=n, est_rc=est_rc, est_field=est_field,
            pt_route=pt_route, r0_pts=r0_pts, r1_pts=r1_pts,
            n_r0=n_r0, n_r1=n_r1, crit_score=act_crit_score,
            bott_mask=act_bott_mask, last_rx=last_rx, last_est_upd=last_est_upd,
            agent_routes=agent_routes, ghost_res=ghost_res,
            horizon=cfg.horizon, max_age=cfg.max_queue_age,
            w_margin=cfg.w_margin, w_critical=cfg.w_critical,
            w_fresh=cfg.w_fresh, w_ghost=cfg.w_ghost,
            route_switch_margin=cfg.route_switch_margin,
            rng=rng, svc_counts=svc_counts,
            rollout_horizon=cfg.rollout_horizon, n_rollouts=cfg.n_rollouts,
            lambda_route=cfg.lambda_route, lambda_regret=cfg.lambda_regret,
            lambda_osc=cfg.lambda_osc, lambda_ghost=cfg.lambda_ghost,
            est_decay_rate=est_decay_rate,
            prefix_trust_mode=cfg.prefix_trust_mode,
            prefix_beta=cfg.prefix_beta,
            prefix_confidence_floor=cfg.prefix_confidence_floor,
            prefix_weight_ranking=cfg.prefix_weight_ranking,
            agent_diversity_penalty=cfg.agent_diversity_penalty,
            epoch_cache=epoch_cache, decision_epoch=cfg.decision_epoch,
            audit=audit,
            e40_invariant_checks=cfg.e40_invariant_checks,
        )

        # Accumulate MC diagnostics
        asel = tick_diag.get("selected_action", None)
        if asel and asel in action_counts:
            action_counts[asel] += 1
        for an in ACTION_NAMES:
            jv = tick_diag.get(f"J_{an}", float("nan"))
            if not math.isnan(jv):
                J_sums[an].append(jv)
        if not tick_diag.get("from_cache", True):
            rt = tick_diag.get("runtime_per_decision_ms", float("nan"))
            if not math.isnan(rt):
                runtime_list.append(rt)
                n_decisions += 1
        sel_rt = tick_diag.get("selection_runtime_ms", float("nan"))
        if not math.isnan(sel_rt):
            selection_runtime_tick_list.append(float(sel_rt))
        if "topk_candidate_pool_before" in tick_diag:
            topk_before_hist.append(float(tick_diag["topk_candidate_pool_before"]))
            topk_after_hist.append(float(tick_diag.get("topk_candidate_pool_after", float("nan"))))
            topk_value_hist.append(float(tick_diag.get("topk_value", float("nan"))))
        dc_sel_score = tick_diag.get("selected_dc_score_mean", float("nan"))
        if not math.isnan(dc_sel_score):
            selected_dc_score_hist.append(float(dc_sel_score))
        fallback_tot += int(tick_diag.get("fallback_count", 0))
        if "ctx_stale_score" in tick_diag:
            ctx_stale_l.append(tick_diag["ctx_stale_score"])
            ctx_ghost_l.append(tick_diag["ctx_ghost_score"])
            ctx_crit_l.append(tick_diag["ctx_critical_pressure"])

        # Apply observations
        tick_rmr: list[float] = []; tick_crs: list[float] = []
        tick_fn:  list[float] = []; tick_gr:  list[float] = []
        tick_cf:  list[bool]  = []; tick_bf:  list[bool]  = []
        upd_crit: set[int]    = set()
        tick_admitted_agents: list[int] = []

        for admission_rank, qi in enumerate(sel_idx):
            ip = queue[qi]
            ag, p, tc, h = ip.agent_id, ip.point_idx, ip.tick_created, ip.horizon_step
            obs          = float(true_field[p]) + rng.normal(0.0, 0.01)
            conf = float(prefix_conf_all[qi]) if qi < len(prefix_conf_all) else 1.0
            age = max(0, min(cfg.max_queue_age, int(tick - tc)))
            remaining_h = _intent_remaining_horizon(ip, tick)
            admitted_remaining_total += 1
            admitted_remaining_values.append(int(remaining_h))
            admitted_remaining_min = int(remaining_h) if admitted_remaining_min is None else min(admitted_remaining_min, int(remaining_h))
            if remaining_h <= 0:
                admitted_remaining_le0_count += 1
            remaining_key = max(-cfg.max_queue_age, min(cfg.horizon, remaining_h))
            admitted_age_counts[age] += 1.0
            remaining_horizon_counts[remaining_key] += 1.0
            prefix_conf_remaining_sum[remaining_key] += conf
            prefix_conf_remaining_count[remaining_key] += 1.0
            tick_admitted_agents.append(int(ag))
            if 1 <= int(h) <= cfg.horizon:
                prefix_conf_sum[int(h)] += conf
                prefix_conf_count[int(h)] += 1.0
                admitted_horizon_count[int(h)] += 1.0
            admitted_points += 1
            if channel_enabled:
                if _channel_packet_dropped(cfg.channel_profile, channel_rng, channel_cache):
                    dropped_points += 1
                else:
                    delay = _channel_jitter_delay(
                        str(CHANNEL_PROFILES[cfg.channel_profile].get("queue_jitter", "none")),
                        channel_rng,
                    )
                    deliver_tick = tick + delay
                    if (CHANNEL_PROFILES[cfg.channel_profile].get("expiration_enabled", False)
                            and deliver_tick - tc > cfg.max_queue_age):
                        expired_points += 1
                    elif delay == 0:
                        if cfg.valid_future_only and remaining_h <= 0:
                            field_update_remaining_le0_count += 1
                            continue
                        obs_v = float(np.clip(obs, 0.02, 0.98))
                        if strict_receiver_enabled:
                            receiver_supports.append(_ReceiverSupport(
                                point_idx=p,
                                target_tick=_intent_target_tick(ip),
                                received_tick=tick,
                                observation=obs_v,
                                refresh_gain=(
                                    conf if cfg.prefix_weight_field
                                    and cfg.prefix_trust_mode != "none" else 1.0),
                                agent_id=ag, horizon_k=h,
                                generation_tick=tc,
                                admission_rank=admission_rank,
                            ))
                        elif cfg.prefix_weight_field and cfg.prefix_trust_mode != "none":
                            est_field[p] = float(np.clip((1.0 - conf) * est_field[p] + conf * obs_v, 0.02, 0.98))
                        else:
                            est_field[p] = obs_v
                        last_est_upd[p] = tick
                        last_rx[ag] = tick
                        svc_counts[ag] += 1
                        field_update_total += 1
                        delivered_points += 1
                        delivery_delays.append(0.0)
                    else:
                        pending_deliveries.append({
                            "deliver_tick": deliver_tick,
                            "admit_tick": tick,
                            "tick_created": tc,
                            "target_tick": _intent_target_tick(ip),
                            "agent_id": ag,
                            "point_idx": p,
                            "obs": obs,
                            "prefix_confidence": conf,
                            "horizon_k": h,
                            "admission_rank": admission_rank,
                        })
            else:
                if cfg.valid_future_only and remaining_h <= 0:
                    field_update_remaining_le0_count += 1
                    continue
                obs_v = float(np.clip(obs, 0.02, 0.98))
                if strict_receiver_enabled:
                    receiver_supports.append(_ReceiverSupport(
                        point_idx=p,
                        target_tick=_intent_target_tick(ip),
                        received_tick=tick,
                        observation=obs_v,
                        refresh_gain=(
                            conf if cfg.prefix_weight_field
                            and cfg.prefix_trust_mode != "none" else 1.0),
                        agent_id=ag, horizon_k=h,
                        generation_tick=tc,
                        admission_rank=admission_rank,
                    ))
                elif cfg.prefix_weight_field and cfg.prefix_trust_mode != "none":
                    est_field[p] = float(np.clip((1.0 - conf) * est_field[p] + conf * obs_v, 0.02, 0.98))
                else:
                    est_field[p] = obs_v
                last_est_upd[p] = tick
                last_rx[ag]     = tick
                svc_counts[ag] += 1
                field_update_total += 1
                delivered_points += 1
                delivery_delays.append(0.0)
            if p in crit_pts_set:
                upd_crit.add(p)
            r   = int(pt_route[p])
            r_v = max(0, r)
            rmr = max(0.0, float(est_rc[1 - r_v]) - float(est_rc[r_v])) if r >= 0 else 0.0
            crs = float(act_crit_score[p])
            fn  = min(1.0, float(tick - last_rx[ag]) / max(cfg.max_queue_age, 1))
            gr  = float(ghost_res[r_v] if r >= 0 else 0.0) * (float(h) / cfg.horizon)
            tick_rmr.append(rmr); tick_crs.append(crs)
            tick_fn.append(fn);   tick_gr.append(gr)
            tick_cf.append(bool(act_crit_mask[p]))
            tick_bf.append(bool(act_bott_mask[p]))
        if strict_receiver_enabled:
            est_field, last_est_upd, receiver_supports, strict_diag = (
                _strict_receiver_recompute(
                    n_pts, receiver_supports, tick, cfg.max_queue_age,
                    est_decay_rate, receiver_variant))
            strict_removed_target_total += int(strict_diag["removed_target"])
            strict_removed_age_total += int(strict_diag["removed_age"])
            strict_invalid_target_total += int(strict_diag["invalid_target"])
            strict_invalid_age_total += int(strict_diag["invalid_age"])
            strict_active_count_hist.append(int(strict_diag["active_count"]))
            strict_fcomm_min_hist.append(float(strict_diag["field_min"]))
            strict_fcomm_max_hist.append(float(strict_diag["field_max"]))
            diagnostic_true_field = _receiver_full_field_for_variant(
                true_field, receiver_variant)
            strict_ffull_min_hist.append(float(diagnostic_true_field.min()))
            strict_ffull_max_hist.append(float(diagnostic_true_field.max()))

        admitted_points_tick_hist.append(len(sel_idx))
        if tick_admitted_agents:
            rev_flags = np.array([
                bool(route_revision_revised_mask[int(ag)]) for ag in tick_admitted_agents
            ], dtype=bool)
            revised_agent_admit_frac_hist.append(float(np.mean(rev_flags)))
            non_revised = {
                int(ag) for ag in tick_admitted_agents
                if not bool(route_revision_revised_mask[int(ag)])
            }
            non_revised_unique_agents_tick_hist.append(int(len(non_revised)))
        else:
            revised_agent_admit_frac_hist.append(0.0)
            non_revised_unique_agents_tick_hist.append(0)
        if route_revision_done and tick >= int(cfg.route_revision_tick):
            abandoned_hits = 0
            abandoned_total = 0
            for qi in sel_idx:
                ip = queue[qi]
                ag = int(ip.agent_id)
                old_r = int(route_revision_old_routes[ag])
                if old_r >= 0:
                    abandoned_total += 1
                    if int(pt_route[int(ip.point_idx)]) == old_r:
                        abandoned_hits += 1
            abandoned_route_admit_frac_after_revision_hist.append(
                float(abandoned_hits / max(abandoned_total, 1)))
            old_vals = []
            new_vals = []
            for ag in np.where(route_revision_revised_mask)[0]:
                old_r = int(route_revision_old_routes[int(ag)])
                new_r = int(route_revision_new_routes[int(ag)])
                if old_r == 0:
                    old_vals.append(float(est_field[r0_pts].mean()))
                elif old_r == 1:
                    old_vals.append(float(est_field[r1_pts].mean()))
                if new_r == 0:
                    new_vals.append(float(est_field[r0_pts].mean()))
                elif new_r == 1:
                    new_vals.append(float(est_field[r1_pts].mean()))
            abandoned_route_pressure_after_revision_hist.append(
                float(np.mean(old_vals)) if old_vals else 0.0)
            new_route_pressure_after_revision_hist.append(
                float(np.mean(new_vals)) if new_vals else 0.0)
        if tick_admitted_agents:
            counts = np.bincount(np.asarray(tick_admitted_agents, dtype=int), minlength=n)
            tick_admitted_counts = counts.astype(np.float64)
            starvation_admission_counts_total += tick_admitted_counts
            admitted_unique_agents_tick_hist.append(int(np.count_nonzero(counts)))
            admitted_same_agent_conc_hist.append(float(counts.max() / max(len(tick_admitted_agents), 1)))
        else:
            tick_admitted_counts = np.zeros(n, dtype=np.float64)
            admitted_unique_agents_tick_hist.append(0)
            admitted_same_agent_conc_hist.append(0.0)

        for qi in sorted(set(sel_idx), reverse=True):
            queue.pop(qi)

        if tick_rmr:
            sel_rmr.append(float(np.mean(tick_rmr)))
            sel_crs.append(float(np.mean(tick_crs)))
            sel_fn.append(float(np.mean(tick_fn)))
            sel_gr.append(float(np.mean(tick_gr)))
            sel_cf.append(float(np.mean(tick_cf)))
            sel_bf.append(float(np.mean(tick_bf)))
        crit_cov.append(len(upd_crit) / len(crit_pts_set) if crit_pts_set else 1.0)

        est_rc = np.array([est_field[r0_pts].mean(), est_field[r1_pts].mean()])

        prev_routes = agent_routes.copy()
        for ag in range(n):
            alt = 1 - int(agent_routes[ag])
            if est_rc[alt] < est_rc[int(agent_routes[ag])] - cfg.route_switch_margin:
                agent_routes[ag] = np.int8(alt)
        if (cfg.route_revision_enabled and route_revision_done
                and int(cfg.route_revision_tick) <= tick <= int(cfg.route_revision_window_end)):
            mask_idx = np.where(route_revision_revised_mask)[0]
            agent_routes[mask_idx] = route_revision_new_routes[mask_idx]

        for ag in range(n):
            if agent_routes[ag] != prev_routes[ag]:
                old_r = int(prev_routes[ag])
                ghost_res[old_r] = min(2.0, float(ghost_res[old_r]) + 1.0 / n)
                if float(true_rc[int(agent_routes[ag])]) >= float(true_rc[old_r]):
                    fp_replans += 1
                if float(true_rc[int(agent_routes[ag])]) > float(true_rc[old_r]):
                    un_replans += 1
                if tick - recent_switch[ag] < 5:
                    oscillations += 1
                recent_switch[ag] = tick

        best_true = int(np.argmin(true_rc))
        best_est  = int(np.argmin(est_rc))
        for ag in range(n):
            r_ag = int(agent_routes[ag])
            acc_rc_true.append(float(true_rc[r_ag]))
            acc_regret.append(float(true_rc[r_ag]) - float(true_rc[best_true]))
            acc_agree.append(int(best_est == best_true))
            acc_abs_gap.append(abs(float(est_rc[r_ag]) - float(true_rc[r_ag])))
        starvation_active_masks.append(tick_active_agents.copy())
        starvation_admitted_counts.append(tick_admitted_counts.copy())
        starvation_route_costs.append(
            np.array([float(true_rc[int(agent_routes[ag])]) for ag in range(n)], dtype=float)
        )
        tick_mean_rc = float(np.mean([true_rc[int(agent_routes[ag])] for ag in range(n)]))
        if tick < int(cfg.route_revision_tick):
            route_cost_pre_revision.append(tick_mean_rc)
        elif tick <= int(cfg.route_revision_window_end):
            route_cost_during_revision.append(tick_mean_rc)
        else:
            route_cost_post_revision.append(tick_mean_rc)

        diagnostic_true_field = _receiver_full_field_for_variant(
            true_field, receiver_variant)
        fl = float(np.mean(np.abs(est_field - diagnostic_true_field)))
        acc_fl.append(fl)
        excl_all = np.concatenate([r0_excl, r1_excl]) if len(r1_excl) else r0_excl
        acc_hidden.append(float(np.mean(
            np.maximum(
                0.0,
                diagnostic_true_field[excl_all] - est_field[excl_all]))))
        acc_ghost.append(float(ghost_res.mean()))

    if channel_enabled and pending_deliveries:
        expired_points += len(pending_deliveries)
        pending_deliveries = []

    # ── Aggregation ──────────────────────────────────────────────────────────
    def _m(lst: list) -> float:
        return float(np.mean(lst)) if lst else 0.0

    rc_arr     = np.array(acc_rc_true, dtype=float)
    regret_arr = np.array(acc_regret,  dtype=float)
    agree_arr  = np.array(acc_agree,   dtype=float)
    gap_arr    = np.array(acc_abs_gap, dtype=float)

    tot_svc = svc_counts.sum()
    if tot_svc > 0:
        p_svc    = svc_counts / float(tot_svc)
        ent      = float(-np.sum(p_svc[p_svc > 0] * np.log(p_svc[p_svc > 0])))
        fairness = ent / max(math.log(n), 1e-9)
    else:
        fairness = 0.0

    rt_per_decision_mean = _m(runtime_list)
    selection_rt_tick_mean = _m(selection_runtime_tick_list)
    amortized_rt = (
        rt_per_decision_mean / max(cfg.decision_epoch, 1)
        if runtime_list else selection_rt_tick_mean
    )
    prefix_conf_by_k = np.divide(
        prefix_conf_sum, prefix_conf_count,
        out=np.full_like(prefix_conf_sum, np.nan, dtype=float),
        where=prefix_conf_count > 0,
    )
    total_admitted_h = float(admitted_horizon_count.sum())
    admitted_horizon_frac = (
        admitted_horizon_count / total_admitted_h
        if total_admitted_h > 0 else np.zeros_like(admitted_horizon_count)
    )
    q_arr = np.asarray(queue_size_hist, dtype=float)
    raw_q_arr = np.asarray(live_queue_size_raw_hist, dtype=float)
    valid_q_arr = np.asarray(live_queue_size_valid_hist, dtype=float)
    admitted_tick_arr = np.asarray(admitted_points_tick_hist, dtype=float)
    unique_tick_arr = np.asarray(admitted_unique_agents_tick_hist, dtype=float)
    conc_arr = np.asarray(admitted_same_agent_conc_hist, dtype=float)
    total_age = float(admitted_age_counts.sum())
    admitted_age_frac = (
        admitted_age_counts / total_age if total_age > 0 else np.zeros_like(admitted_age_counts)
    )
    total_remaining = float(sum(remaining_horizon_counts.values()))
    if starvation_active_masks:
        active_mat = np.vstack(starvation_active_masks).astype(bool)
        admitted_mat = np.vstack(starvation_admitted_counts).astype(float)
        route_cost_mat = np.vstack(starvation_route_costs).astype(float)
        active_counts = active_mat.sum(axis=1).astype(float)
        unique_admitted_active = ((admitted_mat > 0) & active_mat).sum(axis=1).astype(float)
        admitted_unique_agent_fraction_arr = np.divide(
            unique_admitted_active, active_counts,
            out=np.zeros_like(unique_admitted_active, dtype=float),
            where=active_counts > 0,
        )

        def _starvation_rate(window: int) -> float:
            starved = 0.0
            active_total = 0.0
            for tt in range(active_mat.shape[0]):
                lo = max(0, tt - int(window) + 1)
                active_w = active_mat[lo:tt + 1].any(axis=0)
                admitted_w = (admitted_mat[lo:tt + 1].sum(axis=0) > 0)
                active_total += float(active_w.sum())
                starved += float((active_w & ~admitted_w).sum())
            return float(starved / max(active_total, 1.0))

        def _starved_gap(window: int) -> float:
            starved_vals: list[float] = []
            non_vals: list[float] = []
            for tt in range(active_mat.shape[0]):
                lo = max(0, tt - int(window) + 1)
                active_w = active_mat[lo:tt + 1].any(axis=0)
                admitted_w = (admitted_mat[lo:tt + 1].sum(axis=0) > 0)
                starved_mask = active_w & ~admitted_w
                non_mask = active_w & admitted_w
                if starved_mask.any():
                    starved_vals.extend(route_cost_mat[tt, starved_mask].tolist())
                if non_mask.any():
                    non_vals.extend(route_cost_mat[tt, non_mask].tolist())
            if not starved_vals or not non_vals:
                return float("nan")
            return float(np.mean(starved_vals) - np.mean(non_vals))

        streaks: list[int] = []
        current_streak = np.zeros(n, dtype=np.int32)
        for tt in range(active_mat.shape[0]):
            no_admit = active_mat[tt] & (admitted_mat[tt] <= 0)
            ended = (~no_admit) & (current_streak > 0)
            streaks.extend(current_streak[ended].astype(int).tolist())
            current_streak[no_admit] += 1
            current_streak[~no_admit] = 0
        streaks.extend(current_streak[current_streak > 0].astype(int).tolist())
        streak_arr = np.asarray(streaks, dtype=float)
        total_admits_by_agent = starvation_admission_counts_total.astype(float)
        denom = float(n) * float(np.sum(total_admits_by_agent ** 2))
        jain_admission_fairness = (
            float(np.sum(total_admits_by_agent) ** 2 / denom)
            if denom > 0 else 0.0
        )
        starvation_metrics = {
            "admitted_unique_agent_fraction": float(np.mean(admitted_unique_agent_fraction_arr)),
            "active_starvation_rate_W5": _starvation_rate(5),
            "active_starvation_rate_W10": _starvation_rate(10),
            "silent_streak_mean": float(np.mean(streak_arr)) if streak_arr.size else 0.0,
            "silent_streak_p95": float(np.percentile(streak_arr, 95)) if streak_arr.size else 0.0,
            "silent_streak_max": float(np.max(streak_arr)) if streak_arr.size else 0.0,
            "jain_admission_fairness": jain_admission_fairness,
            "starved_agent_route_cost_gap_W10": _starved_gap(10),
            "starved_agent_route_cost_available": True,
        }
    else:
        starvation_metrics = {
            "admitted_unique_agent_fraction": float("nan"),
            "active_starvation_rate_W5": float("nan"),
            "active_starvation_rate_W10": float("nan"),
            "silent_streak_mean": float("nan"),
            "silent_streak_p95": float("nan"),
            "silent_streak_max": float("nan"),
            "jain_admission_fairness": float("nan"),
            "starved_agent_route_cost_gap_W10": float("nan"),
            "starved_agent_route_cost_available": False,
        }

    receiver_field_checksum = hashlib.sha256(
        np.asarray(est_field, dtype=np.float32).tobytes()).hexdigest()
    scale_lo = 0.0 if receiver_variant.startswith("e42") else 0.02
    scale_hi = 1.0 if receiver_variant.startswith("e42") else 0.98
    strict_scale_violation_count = int(
        any(v < scale_lo - 1e-7 for v in strict_fcomm_min_hist)
        or any(v > scale_hi + 1e-7 for v in strict_fcomm_max_hist)
        or any(v < scale_lo - 1e-7 for v in strict_ffull_min_hist)
        or any(v > scale_hi + 1e-7 for v in strict_ffull_max_hist))

    result: dict = {
        # Audit
        "experiment_id":          cfg.experiment_id,
        "scenario":               cfg.scenario,
        "regime":                 cfg.regime,
        "policy":                 cfg.policy,
        "seed":                   cfg.seed,
        "rollout_horizon":        cfg.rollout_horizon,
        "n_rollouts":             cfg.n_rollouts,
        "decision_epoch":         cfg.decision_epoch,
        "quota_action_set":       cfg.quota_action_set,
        "sweep_id":               cfg.sweep_id,
        "channel_profile":        cfg.channel_profile,
        "lambda_route":           cfg.lambda_route,
        "lambda_regret":          cfg.lambda_regret,
        "lambda_osc":             cfg.lambda_osc,
        "lambda_ghost":           cfg.lambda_ghost,
        "lambda_hidden":          cfg.lambda_hidden,
        "n_agents":               cfg.n_agents,
        "n_ticks":                cfg.n_ticks,
        "budget_per_tick":        cfg.budget_per_tick,
        "horizon":                cfg.horizon,
        "max_queue_age":          cfg.max_queue_age,
        "prefix_trust_mode":      cfg.prefix_trust_mode,
        "prefix_beta":            cfg.prefix_beta,
        "prefix_confidence_floor": cfg.prefix_confidence_floor,
        "prefix_weight_ranking":  cfg.prefix_weight_ranking,
        "prefix_weight_field":    cfg.prefix_weight_field,
        "agent_diversity_penalty": cfg.agent_diversity_penalty,
        "normalized_dc_diversity": cfg.policy in NORMALIZED_DC_DIVERSITY_POLICIES,
        "e40_invariant_checks": cfg.e40_invariant_checks,
        "strict_valid_receiver_memory": strict_receiver_enabled,
        "receiver_memory_variant": receiver_variant,
        "valid_future_only":      cfg.valid_future_only,
        "drop_expired_intention_points": cfg.drop_expired_intention_points,
        "drop_nonpositive_remaining_horizon": cfg.drop_nonpositive_remaining_horizon,
        "route_revision_enabled": cfg.route_revision_enabled,
        "route_revision_tick": cfg.route_revision_tick,
        "route_revision_window_end": cfg.route_revision_window_end,
        "route_revision_fraction": cfg.route_revision_fraction,
        "route_revision_probability": cfg.route_revision_probability,
        "scheduler_uses_oracle":  False,
        "commit_hash":            _get_commit_hash(),
        "display_name":           DISPLAY.get(cfg.policy, cfg.policy),
        # Core metrics
        "average_route_cost":              float(rc_arr.mean()),
        "route_cost_pre_revision":         _m(route_cost_pre_revision),
        "route_cost_during_revision":      _m(route_cost_during_revision),
        "route_cost_post_revision":        _m(route_cost_post_revision),
        "route_balanced_loss_g1":          float((regret_arr > 0).mean()),
        "route_balanced_loss_g2":          float(regret_arr.mean()),
        "global_field_loss":               _m(acc_fl),
        "mean_abs_risk_gap_current":       float(gap_arr.mean()),
        "mean_route_decision_agreement":   float(agree_arr.mean()),
        "fraction_route_decision_disagreement": float(1.0 - agree_arr.mean()),
        "mean_route_decision_regret":      float(regret_arr.mean()),
        "p90_route_decision_regret":       float(np.percentile(regret_arr, 90)),
        "fraction_positive_decision_regret": float((regret_arr > 0).mean()),
        "oscillation_count":               int(oscillations),
        "false_positive_replans":          int(fp_replans),
        "unnecessary_replans":             int(un_replans),
        "hidden_congestion_score":         _m(acc_hidden),
        "ghost_pressure_score":            _m(acc_ghost),
        # E41 strict valid-future receiver-memory audit
        "receiver_field_checksum": receiver_field_checksum,
        "receiver_observation_scale": "normalized_pressure_[0.02,0.98]",
        "receiver_full_field_scale": (
            "centered_effective_pressure_[0,1]"
            if receiver_variant == "e42c_centered_uncertainty"
            else "normalized_pressure_[0,1]"),
        "strict_receiver_removed_target_count": strict_removed_target_total,
        "strict_receiver_removed_age_count": strict_removed_age_total,
        "strict_receiver_invalid_target_count": strict_invalid_target_total,
        "strict_receiver_invalid_age_count": strict_invalid_age_total,
        "strict_receiver_expired_contribution_count": strict_expired_contribution_count,
        "strict_receiver_active_support_mean": _m(strict_active_count_hist),
        "strict_receiver_active_support_max": (
            int(max(strict_active_count_hist)) if strict_active_count_hist else 0),
        "strict_receiver_fcomm_min": (
            float(min(strict_fcomm_min_hist)) if strict_fcomm_min_hist else float(est_field.min())),
        "strict_receiver_fcomm_max": (
            float(max(strict_fcomm_max_hist)) if strict_fcomm_max_hist else float(est_field.max())),
        "strict_receiver_ffull_min": (
            float(min(strict_ffull_min_hist)) if strict_ffull_min_hist else float(true_field.min())),
        "strict_receiver_ffull_max": (
            float(max(strict_ffull_max_hist)) if strict_ffull_max_hist else float(true_field.max())),
        "strict_receiver_scale_violation_count": strict_scale_violation_count,
        # Decision diagnostics
        "mean_selected_route_margin_risk":         _m(sel_rmr),
        "mean_selected_critical_region_score":     _m(sel_crs),
        "mean_selected_freshness_need":            _m(sel_fn),
        "mean_selected_ghost_risk":                _m(sel_gr),
        "fraction_selected_from_critical_regions": _m(sel_cf),
        "fraction_selected_near_bottlenecks":      _m(sel_bf),
        "coverage_of_decision_critical_set":       _m(crit_cov),
        "per_agent_service_fairness":              fairness,
        # Lightweight packet/channel diagnostics
        "admitted_points":                  int(admitted_points),
        "delivered_points":                 int(delivered_points),
        "packet_drop_rate":                 float(dropped_points / max(admitted_points, 1)),
        "packet_expiration_rate":           float(expired_points / max(admitted_points, 1)),
        "mean_queue_delay":                 _m(delivery_delays),
        "p95_queue_delay":                  float(np.percentile(delivery_delays, 95)) if delivery_delays else 0.0,
        "effective_budget_mean":            _m(effective_budget_hist),
        "effective_budget_min":             float(np.min(effective_budget_hist)) if effective_budget_hist else float(cfg.budget_per_tick),
        "effective_budget_max":             float(np.max(effective_budget_hist)) if effective_budget_hist else float(cfg.budget_per_tick),
        "live_queue_size_mean":             float(q_arr.mean()) if q_arr.size else 0.0,
        "live_queue_size_p50":              float(np.percentile(q_arr, 50)) if q_arr.size else 0.0,
        "live_queue_size_p90":              float(np.percentile(q_arr, 90)) if q_arr.size else 0.0,
        "live_queue_size_p95":              float(np.percentile(q_arr, 95)) if q_arr.size else 0.0,
        "live_queue_size_raw":              float(raw_q_arr.mean()) if raw_q_arr.size else 0.0,
        "live_queue_size_valid_for_admission": float(valid_q_arr.mean()) if valid_q_arr.size else 0.0,
        "new_candidate_count_mean":         _m(new_candidate_count_hist),
        "expired_points_removed_per_tick_mean": _m(queue_max_age_removed_hist),
        "candidate_count_before_valid_filter": _m(valid_filter_before_hist),
        "candidate_count_after_valid_filter": _m(valid_filter_after_hist),
        "filtered_nonpositive_remaining_horizon_count": _m(valid_filter_removed_hist),
        "filtered_nonpositive_remaining_horizon_fraction": _m(valid_filter_removed_frac_hist),
        "live_queue_stale_not_expired_points_mean": _m(live_queue_stale_not_expired_mean_hist),
        "live_queue_target_expired_fraction": _m(live_queue_expired_target_frac_hist),
        "live_queue_age_ge_ttl_fraction":  _m(live_queue_age_ge_ttl_frac_hist),
        "admitted_points_per_tick_mean":    float(admitted_tick_arr.mean()) if admitted_tick_arr.size else 0.0,
        "admitted_remaining_horizon_min":   float(admitted_remaining_min) if admitted_remaining_min is not None else float("nan"),
        "admitted_remaining_horizon_mean":  float(np.mean(admitted_remaining_values)) if admitted_remaining_values else float("nan"),
        "fraction_admitted_remaining_horizon_le_0": float(admitted_remaining_le0_count / max(admitted_remaining_total, 1)),
        "fraction_field_updates_remaining_horizon_le_0": float(field_update_remaining_le0_count / max(field_update_total + field_update_remaining_le0_count, 1)),
        "valid_future_invariant_pass":      bool(
            (not cfg.valid_future_only) or (
                admitted_remaining_le0_count == 0
                and field_update_remaining_le0_count == 0
                and (admitted_remaining_min is None or admitted_remaining_min >= 1)
            )
        ),
        "admitted_unique_agents_per_tick_mean": float(unique_tick_arr.mean()) if unique_tick_arr.size else 0.0,
        "admitted_same_agent_concentration": float(conc_arr.mean()) if conc_arr.size else 0.0,
        "admitted_same_agent_concentration_p90": float(np.percentile(conc_arr, 90)) if conc_arr.size else 0.0,
        "fraction_ticks_single_agent_monopoly": float(np.mean((conc_arr >= 1.0 - 1e-12) & (admitted_tick_arr > 1))) if conc_arr.size else 0.0,
        # Route revision stress diagnostics
        "route_revision_event_count":       int(route_revision_events),
        "route_revision_forced_agents":     int(route_revision_forced_agents),
        "route_revision_forced_agent_fraction": float(route_revision_forced_agents / max(n, 1)),
        "fraction_admitted_points_from_revised_agents": _m(revised_agent_admit_frac_hist),
        "non_revised_unique_agents_per_tick_mean": _m(non_revised_unique_agents_tick_hist),
        "fraction_admitted_points_on_abandoned_route_after_revision": _m(abandoned_route_admit_frac_after_revision_hist),
        "field_pressure_abandoned_route_after_revision": _m(abandoned_route_pressure_after_revision_hist),
        "field_pressure_new_route_after_revision": _m(new_route_pressure_after_revision_hist),
        # Semantic visibility / starvation diagnostics
        **starvation_metrics,
        # MC runtime
        "n_mc_decisions":                   n_decisions,
        "runtime_per_decision_ms_mean":     rt_per_decision_mean,
        "selection_runtime_ms_tick_mean":   selection_rt_tick_mean,
        "amortized_runtime_per_tick_ms":    amortized_rt,
        "runtime_instrumented":             bool(runtime_list or selection_runtime_tick_list),
        "mc_fallback_count":                fallback_tot,
        "topk_candidate_pool_before_mean":  _m(topk_before_hist),
        "topk_candidate_pool_after_mean":   _m(topk_after_hist),
        "topk_value_mean":                  _m(topk_value_hist),
        "selected_decisioncritical_score_mean": _m(selected_dc_score_hist),
        "ctx_mean_stale_score":             _m(ctx_stale_l),
        "ctx_mean_ghost_score":             _m(ctx_ghost_l),
        "ctx_mean_critical_pressure":       _m(ctx_crit_l),
        "prefix_confidence_mean":           float(np.nanmean(prefix_conf_by_k[1:])) if np.isfinite(prefix_conf_by_k[1:]).any() else 1.0,
        "prefix_confidence_min":            float(np.nanmin(prefix_conf_by_k[1:])) if np.isfinite(prefix_conf_by_k[1:]).any() else 1.0,
        "prefix_confidence_max":            float(np.nanmax(prefix_conf_by_k[1:])) if np.isfinite(prefix_conf_by_k[1:]).any() else 1.0,
    }
    for k in range(1, cfg.horizon + 1):
        result[f"prefix_confidence_k{k}"] = (
            float(prefix_conf_by_k[k]) if np.isfinite(prefix_conf_by_k[k]) else float("nan"))
        result[f"admitted_horizon_fraction_k{k}"] = float(admitted_horizon_frac[k])
        result[f"admitted_horizon_count_k{k}"] = int(admitted_horizon_count[k])
    for age in range(0, cfg.max_queue_age + 1):
        result[f"admitted_point_age_fraction_{age}"] = float(admitted_age_frac[age])
        result[f"admitted_point_age_count_{age}"] = int(admitted_age_counts[age])
    for remaining in remaining_horizon_values:
        label = f"m{abs(remaining)}" if remaining < 0 else str(remaining)
        count = float(remaining_horizon_counts[remaining])
        conf_mean = (
            prefix_conf_remaining_sum[remaining] / prefix_conf_remaining_count[remaining]
            if prefix_conf_remaining_count[remaining] > 0 else float("nan")
        )
        result[f"admitted_remaining_horizon_fraction_{label}"] = (
            float(count / total_remaining) if total_remaining > 0 else 0.0)
        result[f"admitted_remaining_horizon_count_{label}"] = int(count)
        result[f"prefix_confidence_remaining_horizon_{label}"] = float(conf_mean)
    for an in ACTION_NAMES:
        result[f"mc_action_count_{an}"] = action_counts[an]
        result[f"mc_mean_J_{an}"] = (
            float(np.mean(J_sums[an])) if J_sums[an] else float("nan"))
    if audit is not None:
        result["e39_audit"] = audit.export()
    return result

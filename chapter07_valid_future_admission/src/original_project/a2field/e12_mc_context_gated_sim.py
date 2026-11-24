"""E12 — Monte Carlo Context-Gated Future Intention Scheduling.

Two new policies:
  context_gated_rule       — deterministic rule-based gate, no rollouts
  mc_context_gated_scheduler — Monte Carlo rollout over quota-mix actions

Quota actions mix three scheduling experts (AoI / DecisionCritical / Random)
at different ratios.  MCRolloutGated evaluates each quota action by K short-
horizon stochastic rollouts and picks the action with the lowest estimated
route regret.

ORACLE-FREE GUARANTEE:
  The rollout uses only est_field (current communicated estimates), not
  true_field.  Observations in the rollout are modelled as small stochastic
  perturbations around the current estimate.  F_full / true route costs are
  never accessed inside the scheduler.

Topology and baseline policies are imported from e11_decision_critical_sim.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
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
# Quota action definitions
# ─────────────────────────────────────────────────────────────────────────────

# (q_aoi, q_dc, ghost_safe)
# q_rand = 1.0 - q_aoi - q_dc
_QUOTA_ACTIONS: dict[str, tuple[float, float, bool]] = {
    "AoIHeavy":     (0.60, 0.25, False),
    "CriticalHeavy":(0.25, 0.60, False),
    "CoverageHeavy":(0.25, 0.25, False),  # 50% random
    "Balanced":     (0.33, 0.33, False),
    "GhostSafe":    (0.50, 0.35, True),   # DC with ghost-prone downweighted
}
ACTION_NAMES = list(_QUOTA_ACTIONS.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class E12Config:
    scenario:   str
    regime:     str
    policy:     str
    seed:       int
    n_agents:   int   = 12
    n_ticks:    int   = 250
    budget_per_tick: int = 4
    horizon:    int   = 10
    max_queue_age: int = 15
    w_margin:   float = 1.50
    w_critical: float = 1.20
    w_fresh:    float = 1.00
    w_ghost:    float = 0.80
    route_switch_margin: float = 0.04
    rollout_horizon: int = 5
    n_rollouts:  int  = 8
    lambda_osc:  float = 0.1
    lambda_ghost: float = 0.2
    lambda_hidden: float = 0.2   # not used in rollout (no true_field); tracked for reference


# ─────────────────────────────────────────────────────────────────────────────
# Quota helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dc_score_single(
    ip: _IntentPt,
    pt_route: np.ndarray,
    n_r0: int,
    n_r1: int,
    est_rc: np.ndarray,
    crit_score: np.ndarray,
    last_rx: np.ndarray,
    ghost_res: np.ndarray,
    horizon: int,
    max_age: int,
    tick: int,
    w_margin: float,
    w_critical: float,
    w_fresh: float,
    w_ghost: float,
) -> float:
    """DC score for a single candidate, identical to decision_critical_scheduler."""
    rmr = _route_margin_risk(ip.point_idx, pt_route, n_r0, n_r1, est_rc) + 0.1
    crs = float(crit_score[ip.point_idx]) + 0.1
    fn  = min(1.0, float(tick - last_rx[ip.agent_id]) / max(max_age, 1)) + 0.1
    r   = int(pt_route[ip.point_idx])
    gr  = float(ghost_res[r] if r >= 0 else 0.0) * (float(ip.horizon_step) / max(horizon, 1))
    return (w_margin * rmr) * (w_critical * crs) * (w_fresh * fn) - w_ghost * gr


def _apply_quota(
    quota_aoi: float,
    quota_dc: float,
    ghost_safe: bool,
    queue: list[_IntentPt],
    budget: int,
    tick: int,
    n_agents: int,
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
) -> list[int]:
    """Apply a quota action to the queue.  Returns selected queue indices.

    Allocates floor(q_aoi*B) to AoI, floor(q_dc*B) to DC, remainder to Random.
    If an expert is short on candidates, unused quota transfers to Random.
    """
    n_q = len(queue)
    if n_q == 0:
        return []
    n_sel  = min(budget, n_q)
    n_aoi  = int(quota_aoi * n_sel)
    n_dc   = int(quota_dc  * n_sel)
    # remainder (may be boosted if experts under-fill)
    n_rand = n_sel - n_aoi - n_dc

    selected: list[int] = []
    available: list[int] = list(range(n_q))

    # ── AoI expert: oldest-agent-first ──────────────────────────────────────
    if n_aoi > 0 and available:
        ages  = np.array([float(tick - last_rx[queue[qi].agent_id]) for qi in available])
        order = list(np.argsort(-ages))
        take  = min(n_aoi, len(available))
        aoi_sel = [available[order[i]] for i in range(take)]
        selected.extend(aoi_sel)
        sel_set = set(aoi_sel)
        available = [qi for qi in available if qi not in sel_set]
        n_rand   += (n_aoi - take)   # transfer shortfall to random

    # ── DC expert ────────────────────────────────────────────────────────────
    if n_dc > 0 and available:
        w_g_eff = w_ghost * 2.0 if ghost_safe else w_ghost
        scores  = np.array([
            _dc_score_single(queue[qi], pt_route, n_r0, n_r1, est_rc,
                             crit_score, last_rx, ghost_res,
                             horizon, max_age, tick,
                             w_margin, w_critical, w_fresh, w_g_eff)
            for qi in available
        ])
        order   = list(np.argsort(-scores))
        take    = min(n_dc, len(available))
        dc_sel  = [available[order[i]] for i in range(take)]
        selected.extend(dc_sel)
        sel_set  = set(dc_sel)
        available = [qi for qi in available if qi not in sel_set]
        n_rand   += (n_dc - take)

    # ── Random: fill remainder ───────────────────────────────────────────────
    if n_rand > 0 and available:
        take     = min(n_rand, len(available))
        rand_sel = list(rng.choice(available, size=take, replace=False))
        selected.extend(rand_sel)

    return selected[:n_sel]


# ─────────────────────────────────────────────────────────────────────────────
# Context features
# ─────────────────────────────────────────────────────────────────────────────

def _context_features(
    queue: list[_IntentPt],
    tick: int,
    last_rx: np.ndarray,
    last_est_upd: np.ndarray,
    est_rc: np.ndarray,
    ghost_res: np.ndarray,
    crit_score: np.ndarray,
    n_agents: int,
    max_age: int,
    crit_threshold: float = 1.0,
    small_margin_threshold: float = 0.05,
    stale_upd_threshold: int = 8,
) -> dict[str, float]:
    """Compute 5 context features for ContextGatedRule."""
    # stale_score: normalised mean agent wait
    stale_score = float(np.mean(tick - last_rx)) / max(max_age, 1)
    stale_score = min(1.0, stale_score)

    # ghost_score: mean ghost residual
    ghost_score = float(ghost_res.mean())

    # critical_pressure: fraction of queue candidates in critical regions
    if queue:
        crit_frac = float(np.mean([
            1.0 if crit_score[ip.point_idx] >= crit_threshold else 0.0
            for ip in queue
        ]))
    else:
        crit_frac = 0.0

    # small_margin_rate: binary — route costs are close
    margin = float(abs(est_rc[0] - est_rc[1]))
    small_margin = 1.0 if margin < small_margin_threshold else 0.0

    # coverage_gap: fraction of points with stale estimates
    if len(last_est_upd) > 0:
        stale_pts = float(np.mean((tick - last_est_upd) > stale_upd_threshold))
    else:
        stale_pts = 0.0

    return {
        "stale_score":        stale_score,
        "ghost_score":        ghost_score,
        "critical_pressure":  crit_frac,
        "small_margin_rate":  small_margin,
        "coverage_gap":       stale_pts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo rollout evaluator
# ─────────────────────────────────────────────────────────────────────────────

def _mc_eval_action(
    quota_aoi: float,
    quota_dc: float,
    ghost_safe: bool,
    queue: list[_IntentPt],
    budget: int,
    tick: int,
    n_agents: int,
    # State to copy
    est_field: np.ndarray,
    last_rx: np.ndarray,
    agent_routes: np.ndarray,
    ghost_res: np.ndarray,
    last_est_upd: np.ndarray,
    # Route topology
    r0_pts: np.ndarray,
    r1_pts: np.ndarray,
    # Other
    pt_route: np.ndarray,
    n_r0: int,
    n_r1: int,
    crit_score: np.ndarray,
    horizon: int,
    max_age: int,
    w_margin: float,
    w_critical: float,
    w_fresh: float,
    w_ghost: float,
    route_switch_margin: float,
    H: int,
    K: int,
    lambda_osc: float,
    lambda_ghost: float,
    est_decay_rate: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Run K stochastic rollouts for a quota action.  Return (J_mean, J_std).

    ORACLE-FREE: Only est_field is used.  Observations in rollouts perturb
    the current estimate with small noise — no true_field access.
    lambda_hidden term is omitted (requires true_field); documented as 0.
    """
    n_pts = len(est_field)
    J_samples: list[float] = []

    for _ in range(K):
        # Copy mutable state
        ef = est_field.copy()
        ar = agent_routes.copy()
        gr = ghost_res.copy()
        lrx = last_rx.copy()
        leu = last_est_upd.copy()

        # Compute est_rc for this rollout's initial selection
        est_rc_roll = np.array([ef[r0_pts].mean(), ef[r1_pts].mean()])

        # Apply quota action: select points from current queue
        sel_idx = _apply_quota(
            quota_aoi, quota_dc, ghost_safe,
            queue, budget, tick, n_agents,
            est_rc_roll, pt_route, n_r0, n_r1,
            crit_score, lrx, gr,
            horizon, max_age,
            w_margin, w_critical, w_fresh, w_ghost,
            rng,
        )

        # Stochastic observation: perturb est_field at selected points
        for qi in sel_idx:
            ip = queue[qi]
            p  = ip.point_idx
            obs_noise = rng.normal(0.0, 0.02)
            ef[p]  = float(np.clip(ef[p] + obs_noise, 0.02, 0.98))
            leu[p] = tick
            lrx[ip.agent_id] = tick

        # Rollout: H light ticks of state evolution
        J_h = 0.0
        osc_count = 0
        prev_ar = ar.copy()

        for h in range(H):
            # Stale-field decay toward prior 0.5
            stale_mask = (tick + h - leu) > 3
            ef[stale_mask] = ef[stale_mask] + est_decay_rate * (0.5 - ef[stale_mask])

            # Small field noise (freshness uncertainty)
            ef += rng.normal(0.0, 0.004, n_pts)
            ef  = np.clip(ef, 0.02, 0.98)

            # Route cost estimates
            est_rc_h = np.array([ef[r0_pts].mean(), ef[r1_pts].mean()])

            # Stochastic route decisions
            prev_ar = ar.copy()
            for ag in range(n_agents):
                alt = 1 - int(ar[ag])
                # Small route-risk noise on switch threshold
                switch_noise = rng.normal(0.0, 0.008)
                if (est_rc_h[alt] < est_rc_h[int(ar[ag])]
                        - route_switch_margin + switch_noise):
                    ar[ag] = np.int8(alt)

            # Oscillation count
            osc_count += int(np.sum(ar != prev_ar))

            # Ghost residual decay
            gr *= 0.90

            # Estimated regret (vs best est route; NOT true_field)
            best_est = int(np.argmin(est_rc_h))
            for ag in range(n_agents):
                r_ag = int(ar[ag])
                J_h += max(0.0, float(est_rc_h[r_ag]) - float(est_rc_h[best_est]))

        # Normalise regret over H ticks * n_agents
        J_h /= max(H * n_agents, 1)
        # Add oscillation penalty
        J_h += lambda_osc * (osc_count / max(H * n_agents, 1))
        # Add ghost penalty
        J_h += lambda_ghost * float(ghost_res.mean())

        J_samples.append(J_h)

    J_arr = np.array(J_samples, dtype=float)
    return float(J_arr.mean()), float(J_arr.std())


# ─────────────────────────────────────────────────────────────────────────────
# Main selection dispatcher for E12
# ─────────────────────────────────────────────────────────────────────────────

def _select_e12(
    policy: str,
    queue: list[_IntentPt],
    budget: int,
    tick: int,
    n_agents: int,
    est_rc: np.ndarray,
    est_field: np.ndarray,
    pt_route: np.ndarray,
    r0_pts: np.ndarray,
    r1_pts: np.ndarray,
    n_r0: int,
    n_r1: int,
    crit_score: np.ndarray,
    bott_mask: np.ndarray,
    last_rx: np.ndarray,
    last_est_upd: np.ndarray,
    agent_routes: np.ndarray,
    ghost_res: np.ndarray,
    horizon: int,
    max_age: int,
    w_margin: float,
    w_critical: float,
    w_fresh: float,
    w_ghost: float,
    route_switch_margin: float,
    rng: np.random.Generator,
    svc_counts: np.ndarray | None,
    rollout_horizon: int,
    n_rollouts: int,
    lambda_osc: float,
    lambda_ghost: float,
    est_decay_rate: float,
) -> tuple[list[int], dict]:
    """Return (selected_queue_indices, diagnostic_dict)."""
    n_q = len(queue)
    diag: dict = {}

    # ── Baselines: delegate to E11 _select ────────────────────────────────
    if policy in ("random_budget", "aoi_priority_scheduler",
                  "decision_critical_scheduler", "adaptive_ratio_rule",
                  "periodic_full_reference", "fixed_hybrid_25_75"):
        idx = _e11_select(
            policy=policy,
            queue=queue,
            budget=budget,
            tick=tick,
            n_agents=n_agents,
            est_rc=est_rc,
            pt_route=pt_route,
            n_r0=n_r0,
            n_r1=n_r1,
            crit_score=crit_score,
            bott_mask=bott_mask,
            last_rx=last_rx,
            ghost_res=ghost_res,
            horizon=horizon,
            max_age=max_age,
            w_margin=w_margin,
            w_critical=w_critical,
            w_fresh=w_fresh,
            w_ghost=w_ghost,
            rng=rng,
            svc_counts=svc_counts,
        )
        return idx, diag

    # ── Context features (shared by both new policies) ────────────────────
    ctx = _context_features(
        queue=queue,
        tick=tick,
        last_rx=last_rx,
        last_est_upd=last_est_upd,
        est_rc=est_rc,
        ghost_res=ghost_res,
        crit_score=crit_score,
        n_agents=n_agents,
        max_age=max_age,
    )
    diag["ctx_stale_score"]       = ctx["stale_score"]
    diag["ctx_ghost_score"]       = ctx["ghost_score"]
    diag["ctx_critical_pressure"] = ctx["critical_pressure"]
    diag["ctx_small_margin_rate"] = ctx["small_margin_rate"]
    diag["ctx_coverage_gap"]      = ctx["coverage_gap"]

    # ── ContextGatedRule: deterministic rule, no rollouts ─────────────────
    if policy == "context_gated_rule":
        if ctx["stale_score"] > 0.6 or ctx["ghost_score"] > 0.3:
            chosen = "AoIHeavy"
        elif ctx["critical_pressure"] > 0.5 or ctx["small_margin_rate"] == 1.0:
            chosen = "CriticalHeavy"
        else:
            chosen = "CoverageHeavy"

        diag["selected_action"] = chosen
        q_aoi, q_dc, gs = _QUOTA_ACTIONS[chosen]
        idx = _apply_quota(
            q_aoi, q_dc, gs,
            queue, budget, tick, n_agents,
            est_rc, pt_route, n_r0, n_r1,
            crit_score, last_rx, ghost_res,
            horizon, max_age,
            w_margin, w_critical, w_fresh, w_ghost,
            rng,
        )
        return idx, diag

    # ── MCRolloutGated: evaluate all actions via rollout ──────────────────
    if policy == "mc_context_gated_scheduler":
        if n_q == 0:
            return [], diag

        t_start = time.perf_counter()
        best_action = "Balanced"
        best_J      = float("inf")
        fallback    = 0

        for action_name, (q_aoi, q_dc, gs) in _QUOTA_ACTIONS.items():
            try:
                J_mean, J_std = _mc_eval_action(
                    quota_aoi=q_aoi,
                    quota_dc=q_dc,
                    ghost_safe=gs,
                    queue=queue,
                    budget=budget,
                    tick=tick,
                    n_agents=n_agents,
                    est_field=est_field,
                    last_rx=last_rx,
                    agent_routes=agent_routes,
                    ghost_res=ghost_res,
                    last_est_upd=last_est_upd,
                    r0_pts=r0_pts,
                    r1_pts=r1_pts,
                    pt_route=pt_route,
                    n_r0=n_r0,
                    n_r1=n_r1,
                    crit_score=crit_score,
                    horizon=horizon,
                    max_age=max_age,
                    w_margin=w_margin,
                    w_critical=w_critical,
                    w_fresh=w_fresh,
                    w_ghost=w_ghost,
                    route_switch_margin=route_switch_margin,
                    H=rollout_horizon,
                    K=n_rollouts,
                    lambda_osc=lambda_osc,
                    lambda_ghost=lambda_ghost,
                    est_decay_rate=est_decay_rate,
                    rng=rng,
                )
                diag[f"J_{action_name}"]     = J_mean
                diag[f"J_std_{action_name}"] = J_std
                if J_mean < best_J:
                    best_J      = J_mean
                    best_action = action_name
            except Exception:
                fallback += 1
                diag[f"J_{action_name}"]     = float("nan")
                diag[f"J_std_{action_name}"] = float("nan")

        diag["selected_action"]  = best_action
        diag["fallback_count"]   = fallback
        diag["runtime_ms"]       = (time.perf_counter() - t_start) * 1000.0

        q_aoi, q_dc, gs = _QUOTA_ACTIONS[best_action]
        idx = _apply_quota(
            q_aoi, q_dc, gs,
            queue, budget, tick, n_agents,
            est_rc, pt_route, n_r0, n_r1,
            crit_score, last_rx, ghost_res,
            horizon, max_age,
            w_margin, w_critical, w_fresh, w_ghost,
            rng,
        )
        return idx, diag

    raise ValueError(f"Unknown E12 policy: {policy!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_e12_single(cfg: E12Config) -> dict:  # noqa: C901
    rng  = np.random.default_rng(cfg.seed)
    topo = _TOPO[cfg.scenario]

    n0 = topo.n_r0_excl
    n1 = topo.n_r1_excl
    ns = topo.n_shared
    n_pts = n0 + n1 + ns

    r0_excl = np.arange(0, n0)
    r1_excl = np.arange(n0, n0 + n1)
    shared  = np.arange(n0 + n1, n_pts)

    r0_pts  = np.concatenate([r0_excl, shared]) if ns else r0_excl
    r1_pts  = np.concatenate([r1_excl, shared]) if ns else r1_excl
    n_r0    = len(r0_pts)
    n_r1    = len(r1_pts)

    pt_route = np.full(n_pts, -1, dtype=np.int8)
    pt_route[r0_excl] = 0
    pt_route[r1_excl] = 1

    crit_mask = np.zeros(n_pts, dtype=bool)
    bott_mask = np.zeros(n_pts, dtype=bool)
    for i in topo.critical_idxs:
        if i < n_pts:
            crit_mask[i] = True
    for i in topo.bottleneck_idxs:
        if i < n_pts:
            bott_mask[i] = True
    crit_score = np.where(bott_mask, 1.5, np.where(crit_mask, 1.0, 0.1)).astype(np.float32)
    act_crit_mask  = crit_mask.copy()
    act_bott_mask  = bott_mask.copy()
    act_crit_score = crit_score.copy()
    crit_pts_set   = set(np.where(crit_mask)[0])

    true_field = rng.uniform(topo.field_lo, topo.field_hi, n_pts).astype(np.float32)
    if topo.route1_bias > 0:
        true_field[r1_excl] = np.clip(
            true_field[r1_excl] + float(topo.route1_bias), 0.02, 0.98
        ).astype(np.float32)
    est_field    = np.full(n_pts, 0.50, dtype=np.float32)
    last_est_upd = np.full(n_pts, -cfg.max_queue_age, dtype=np.int32)

    n = cfg.n_agents
    agent_routes  = rng.integers(0, 2, size=n).astype(np.int8)
    last_rx       = np.full(n, -cfg.max_queue_age, dtype=np.int32)
    recent_switch = np.full(n, -20, dtype=np.int32)
    svc_counts    = np.zeros(n, dtype=np.int32)

    ghost_res = np.zeros(2, dtype=np.float32)

    is_stress      = cfg.regime == "ghost_stress"
    spike_prob_eff = float(topo.spike_prob) * (2.2 if is_stress else 1.0)
    est_decay_rate = 0.018 if is_stress else 0.009

    # ── Metric accumulators ──────────────────────────────────────────────────
    acc_rc_true:  list[float] = []
    acc_regret:   list[float] = []
    acc_agree:    list[int]   = []
    acc_abs_gap:  list[float] = []
    acc_fl:       list[float] = []
    acc_hidden:   list[float] = []
    acc_ghost:    list[float] = []

    fp_replans = 0
    un_replans = 0
    oscillations = 0

    sel_rmr:  list[float] = []
    sel_crs:  list[float] = []
    sel_fn:   list[float] = []
    sel_gr:   list[float] = []
    sel_cf:   list[float] = []
    sel_bf:   list[float] = []
    sel_hz:   list[float] = []
    sel_age:  list[float] = []
    crit_cov: list[float] = []

    # ── MC/CGR diagnostics ────────────────────────────────────────────────────
    action_counts: dict[str, int] = {a: 0 for a in ACTION_NAMES}
    J_sums: dict[str, list[float]] = {a: [] for a in ACTION_NAMES}
    runtime_ms_list: list[float] = []
    fallback_total  = 0
    ctx_stale_list:   list[float] = []
    ctx_ghost_list:   list[float] = []
    ctx_crit_list:    list[float] = []

    drift_dir  = 1.0
    shift_done = False
    queue: list[_IntentPt] = []

    for tick in range(cfg.n_ticks):

        if (topo.bottleneck_shift_tick > 0
                and not shift_done
                and tick >= topo.bottleneck_shift_tick):
            ob = list(topo.bottleneck_idxs)
            nb = [ob[-1] + 1, ob[-1] + 2]
            nb = [i for i in nb if i < n_pts]
            for i in ob:
                act_bott_mask[i] = False
                act_crit_mask[i] = False
            for i in nb:
                act_bott_mask[i] = True
                act_crit_mask[i] = True
            act_crit_score[:] = np.where(
                act_bott_mask, 1.5, np.where(act_crit_mask, 1.0, 0.1)
            )
            shift_done = True

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
                    0.02, 0.98
                ).astype(np.float32)

        if ns > 0:
            occupancy = float(np.sum(np.isin(agent_routes, [0, 1]))) / max(n, 1)
            bott_pts_idx = np.where(act_bott_mask)[0]
            if len(bott_pts_idx):
                true_field[bott_pts_idx] = np.clip(
                    true_field[bott_pts_idx] + 0.018 * occupancy, 0.02, 0.98
                ).astype(np.float32)

        ghost_res *= (1.0 - float(topo.ghost_decay))

        stale = (tick - last_est_upd) > 3
        est_field[stale] = (
            est_field[stale] + est_decay_rate * (0.5 - est_field[stale])
        )

        true_rc = np.array([true_field[r0_pts].mean(), true_field[r1_pts].mean()])
        est_rc  = np.array([est_field[r0_pts].mean(),  est_field[r1_pts].mean()])

        # Generate intention points
        new_pts: list[_IntentPt] = []
        for ag in range(n):
            rp = r0_pts if agent_routes[ag] == 0 else r1_pts
            for h in range(1, cfg.horizon + 1):
                p = int(rp[(ag * cfg.horizon + h + tick) % len(rp)])
                new_pts.append(_IntentPt(ag, p, tick, h))
        queue.extend(new_pts)
        queue = [ip for ip in queue if tick - ip.tick_created <= cfg.max_queue_age]

        # Policy selection
        sel_idx, tick_diag = _select_e12(
            policy=cfg.policy,
            queue=queue,
            budget=cfg.budget_per_tick,
            tick=tick,
            n_agents=n,
            est_rc=est_rc,
            est_field=est_field,
            pt_route=pt_route,
            r0_pts=r0_pts,
            r1_pts=r1_pts,
            n_r0=n_r0,
            n_r1=n_r1,
            crit_score=act_crit_score,
            bott_mask=act_bott_mask,
            last_rx=last_rx,
            last_est_upd=last_est_upd,
            agent_routes=agent_routes,
            ghost_res=ghost_res,
            horizon=cfg.horizon,
            max_age=cfg.max_queue_age,
            w_margin=cfg.w_margin,
            w_critical=cfg.w_critical,
            w_fresh=cfg.w_fresh,
            w_ghost=cfg.w_ghost,
            route_switch_margin=cfg.route_switch_margin,
            rng=rng,
            svc_counts=svc_counts,
            rollout_horizon=cfg.rollout_horizon,
            n_rollouts=cfg.n_rollouts,
            lambda_osc=cfg.lambda_osc,
            lambda_ghost=cfg.lambda_ghost,
            est_decay_rate=est_decay_rate,
        )

        # Accumulate MC diagnostics
        action_sel = tick_diag.get("selected_action", None)
        if action_sel and action_sel in action_counts:
            action_counts[action_sel] += 1
        for aname in ACTION_NAMES:
            J_val = tick_diag.get(f"J_{aname}", float("nan"))
            if not math.isnan(J_val):
                J_sums[aname].append(J_val)
        if "runtime_ms" in tick_diag:
            runtime_ms_list.append(tick_diag["runtime_ms"])
        fallback_total += int(tick_diag.get("fallback_count", 0))
        if "ctx_stale_score" in tick_diag:
            ctx_stale_list.append(tick_diag["ctx_stale_score"])
            ctx_ghost_list.append(tick_diag["ctx_ghost_score"])
            ctx_crit_list.append(tick_diag["ctx_critical_pressure"])

        # Process selected points
        tick_rmr: list[float] = []
        tick_crs: list[float] = []
        tick_fn:  list[float] = []
        tick_gr:  list[float] = []
        tick_cf:  list[bool]  = []
        tick_bf:  list[bool]  = []
        tick_hz:  list[int]   = []
        tick_age: list[float] = []
        upd_crit: set[int]    = set()

        for qi in sel_idx:
            ip = queue[qi]
            ag, p, tc, h = ip.agent_id, ip.point_idx, ip.tick_created, ip.horizon_step
            obs = float(true_field[p]) + rng.normal(0.0, 0.01)
            est_field[p]     = float(np.clip(obs, 0.02, 0.98))
            last_est_upd[p]  = tick
            last_rx[ag]      = tick
            svc_counts[ag]  += 1
            if p in crit_pts_set:
                upd_crit.add(p)
            rmr = _route_margin_risk(p, pt_route, n_r0, n_r1, est_rc) + 0.1
            crs = float(act_crit_score[p]) + 0.1
            fn  = min(1.0, float(tick - last_rx[ag]) / max(cfg.max_queue_age, 1)) + 0.1
            r   = int(pt_route[p])
            gr  = float(ghost_res[r] if r >= 0 else 0.0) * (float(h) / cfg.horizon)
            tick_rmr.append(rmr - 0.1)
            tick_crs.append(crs - 0.1)
            tick_fn.append(fn  - 0.1)
            tick_gr.append(gr)
            tick_cf.append(bool(act_crit_mask[p]))
            tick_bf.append(bool(act_bott_mask[p]))
            tick_hz.append(h)
            tick_age.append(float(tick - last_rx[ag]))

        for qi in sorted(set(sel_idx), reverse=True):
            queue.pop(qi)

        if tick_rmr:
            sel_rmr.append(float(np.mean(tick_rmr)))
            sel_crs.append(float(np.mean(tick_crs)))
            sel_fn.append(float(np.mean(tick_fn)))
            sel_gr.append(float(np.mean(tick_gr)))
            sel_cf.append(float(np.mean(tick_cf)))
            sel_bf.append(float(np.mean(tick_bf)))
            sel_hz.append(float(np.mean(tick_hz)))
            sel_age.append(float(np.mean(tick_age)))

        if crit_pts_set:
            crit_cov.append(len(upd_crit) / len(crit_pts_set))
        else:
            crit_cov.append(1.0)

        est_rc = np.array([est_field[r0_pts].mean(), est_field[r1_pts].mean()])

        prev_routes = agent_routes.copy()
        for ag in range(n):
            alt = 1 - int(agent_routes[ag])
            if est_rc[alt] < est_rc[int(agent_routes[ag])] - cfg.route_switch_margin:
                agent_routes[ag] = np.int8(alt)

        for ag in range(n):
            if agent_routes[ag] != prev_routes[ag]:
                old_r = int(prev_routes[ag])
                ghost_res[old_r] = min(2.0, float(ghost_res[old_r]) + 1.0 / n)
                old_c = float(true_rc[old_r])
                new_c = float(true_rc[int(agent_routes[ag])])
                if new_c >= old_c:
                    fp_replans += 1
                if new_c > old_c:
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

        fl = float(np.mean(np.abs(est_field - true_field)))
        acc_fl.append(fl)
        excl_all = np.concatenate([r0_excl, r1_excl]) if len(r1_excl) else r0_excl
        acc_hidden.append(float(np.mean(np.maximum(0.0, true_field[excl_all] - est_field[excl_all]))))
        acc_ghost.append(float(ghost_res.mean()))

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

    result = {
        # Core
        "average_route_cost":              float(rc_arr.mean()),
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
        # Decision diagnostics
        "mean_selected_route_margin_risk":         _m(sel_rmr),
        "mean_selected_critical_region_score":     _m(sel_crs),
        "mean_selected_freshness_need":            _m(sel_fn),
        "mean_selected_ghost_risk":                _m(sel_gr),
        "fraction_selected_from_critical_regions": _m(sel_cf),
        "fraction_selected_near_bottlenecks":      _m(sel_bf),
        "mean_selected_horizon_step":              _m(sel_hz),
        "mean_selected_agent_age":                 _m(sel_age),
        "coverage_of_decision_critical_set":       _m(crit_cov),
        "per_agent_service_fairness":              fairness,
        # MC / CGR diagnostics
        "mc_fallback_count": fallback_total,
        "mc_runtime_ms_mean": float(np.mean(runtime_ms_list)) if runtime_ms_list else 0.0,
        "mc_runtime_ms_max":  float(np.max(runtime_ms_list))  if runtime_ms_list else 0.0,
        "ctx_mean_stale_score":       _m(ctx_stale_list),
        "ctx_mean_ghost_score":       _m(ctx_ghost_list),
        "ctx_mean_critical_pressure": _m(ctx_crit_list),
        "scheduler_uses_oracle":      False,   # NO-ORACLE CHECK
    }
    # Action selection counts and mean J per action
    for aname in ACTION_NAMES:
        result[f"mc_action_count_{aname}"] = action_counts[aname]
        result[f"mc_mean_J_{aname}"] = (
            float(np.mean(J_sums[aname])) if J_sums[aname] else float("nan")
        )
    return result

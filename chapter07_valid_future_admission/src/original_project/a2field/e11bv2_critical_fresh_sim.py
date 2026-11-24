"""E11b-v2 — CriticalFresh Additive Scheduler (corrective experiment).

Fixes vs E11b-v1:
1. Score formula is ADDITIVE:
       score = wc*n_crs + wm*n_rmr + wf*n_fn - wg*n_gr + wfair*n_fair
   (v1 was multiplicative: (wc*crs)*(wm*rmr)*(wf*fn) - wg*gr + wfair*fair)
2. freshness_need = tick - last_est_upd[point_idx]   (per-POINT field-estimate age)
   v1 used tick - last_rx[agent_id] which is nearly constant in corridor topologies
   because all agents are served at similar rates.
3. Constant-component events are counted and returned as diagnostics.

Topologies and base policies are imported from e11_decision_critical_sim to avoid
code duplication.  The v2 sim adds last_est_upd to the _select signature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field
from typing import NamedTuple

import numpy as np

from a2field.e11_decision_critical_sim import (
    _TOPO,
    KNOWN_SCENARIOS,
    _IntentPt,
    _route_margin_risk,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class E11bV2Config:
    scenario:   str
    regime:     str   # "normal" | "ghost_stress"
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


# ─────────────────────────────────────────────────────────────────────────────
# Internal result from _select_v2
# ─────────────────────────────────────────────────────────────────────────────

class _SelectResult(NamedTuple):
    indices: list[int]
    fn_was_constant: bool  # freshness component was constant across candidates
    raw_fn_values: list[float]  # raw freshness of every candidate (for diagnostics)


# ─────────────────────────────────────────────────────────────────────────────
# Selection function
# ─────────────────────────────────────────────────────────────────────────────

def _norm01(arr: np.ndarray) -> tuple[np.ndarray, bool]:
    """Min-max normalize to [0,1].  Returns (normalized, was_constant)."""
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-9:
        return np.zeros_like(arr), True
    return (arr - lo) / (hi - lo), False


def _select_v2(
    policy: str,
    queue: list[_IntentPt],
    budget: int,
    tick: int,
    n_agents: int,
    est_rc: np.ndarray,
    pt_route: np.ndarray,
    n_r0: int,
    n_r1: int,
    crit_score: np.ndarray,
    bott_mask: np.ndarray,
    last_rx: np.ndarray,        # per-agent last received tick
    last_est_upd: np.ndarray,   # per-POINT last field-estimate update tick
    ghost_res: np.ndarray,
    horizon: int,
    max_age: int,
    w_margin: float,
    w_critical: float,
    w_fresh: float,
    w_ghost: float,
    rng: np.random.Generator,
    svc_counts: np.ndarray | None = None,
) -> _SelectResult:
    """Return selected indices plus freshness diagnostics."""
    n_q = len(queue)
    if n_q == 0:
        return _SelectResult([], False, [])
    n_sel = min(budget, n_q)

    _no_fn_diag = _SelectResult([], False, [])

    # ── Non-CriticalFresh policies (unchanged from E11) ────────────────────

    if policy == "random_budget":
        idx = list(rng.choice(n_q, size=n_sel, replace=False))
        return _SelectResult(idx, False, [])

    if policy == "periodic_full_reference":
        if tick % 20 == 0:
            by_pt: dict[int, int] = {}
            for qi, ip in enumerate(queue):
                if ip.point_idx not in by_pt:
                    by_pt[ip.point_idx] = qi
            selected = list(by_pt.values())
            if len(selected) < n_sel:
                pool = [qi for qi in range(n_q) if qi not in set(selected)]
                need = n_sel - len(selected)
                if pool:
                    selected += list(rng.choice(pool, size=min(need, len(pool)), replace=False))
            return _SelectResult(selected[:n_sel], False, [])
        idx = list(rng.choice(n_q, size=n_sel, replace=False))
        return _SelectResult(idx, False, [])

    if policy == "adaptive_ratio_rule":
        margin = float(abs(est_rc[0] - est_rc[1]))
        scores = np.ones(n_q, dtype=float)
        for qi, ip in enumerate(queue):
            if pt_route[ip.point_idx] >= 0:
                scores[qi] += 2.0
            if crit_score[ip.point_idx] > 1.0:
                scores[qi] += 1.0
            if margin < 0.06:
                scores[qi] *= 1.5 if crit_score[ip.point_idx] > 1.0 else 1.0
        probs = scores / scores.sum()
        idx = list(rng.choice(n_q, size=n_sel, replace=False, p=probs))
        return _SelectResult(idx, False, [])

    if policy == "fixed_hybrid_25_75":
        n_rand  = max(1, int(0.25 * n_sel))
        n_route = n_sel - n_rand
        route_pool = [qi for qi, ip in enumerate(queue)
                      if pt_route[ip.point_idx] >= 0 or crit_score[ip.point_idx] > 1.0]
        rand_pool  = list(range(n_q))
        if route_pool:
            route_sel = list(rng.choice(route_pool, size=min(n_route, len(route_pool)), replace=False))
        else:
            route_sel = []
        used = set(route_sel)
        remain = [qi for qi in rand_pool if qi not in used]
        rand_sel = []
        if remain:
            rand_sel = list(rng.choice(remain, size=min(n_rand, len(remain)), replace=False))
        idx = (route_sel + rand_sel)[:n_sel]
        return _SelectResult(idx, False, [])

    if policy == "aoi_priority_scheduler":
        ages = np.array([float(tick - last_rx[ip.agent_id]) for ip in queue])
        order = np.argsort(-ages)
        idx = list(order[:n_sel])
        return _SelectResult(idx, False, [])

    if policy == "decision_critical_scheduler":
        scores = np.empty(n_q, dtype=float)
        for qi, ip in enumerate(queue):
            rmr = _route_margin_risk(ip.point_idx, pt_route, n_r0, n_r1, est_rc) + 0.1
            crs = float(crit_score[ip.point_idx]) + 0.1
            fn  = min(1.0, float(tick - last_rx[ip.agent_id]) / max(max_age, 1)) + 0.1
            r   = int(pt_route[ip.point_idx])
            gr  = float(ghost_res[r] if r >= 0 else 0.0)
            gr *= float(ip.horizon_step) / max(horizon, 1)
            scores[qi] = (w_margin * rmr) * (w_critical * crs) * (w_fresh * fn) - w_ghost * gr
        order = np.argsort(-scores)
        idx = list(order[:n_sel])
        return _SelectResult(idx, False, [])

    # ── CriticalFresh family (ADDITIVE, per-point freshness) ───────────────
    #
    # score(p) = wc * n_crs(p)
    #          + wm * n_rmr(p)
    #          + wf * n_fn(p)          ← freshness_need = tick - last_est_upd[p]
    #          - wg * n_gr(p)
    #          + wfair * n_fair(p)
    #
    # All components are min-max normalised to [0,1] within the current tick's
    # candidate set.  If a component is constant (hi=lo), its normalised value
    # is 0.0 and the event is recorded in diagnostics.
    #
    # Variants:
    #   equal      wc=wm=wf=wg=1.0, wfair=0.2
    #   fresh      wf=2.0  (fresh-heavy)
    #   critical   wc=wm=2.0  (criticality+margin-heavy)
    #   ghost_safe wg=2.0  (ghost-penalised)
    if policy in ("critical_fresh_equal", "critical_fresh_fresh",
                  "critical_fresh_critical", "critical_fresh_ghost_safe"):
        if policy == "critical_fresh_fresh":
            wc, wm, wf, wg, wfair = 1.0, 1.0, 2.0, 1.0, 0.2
        elif policy == "critical_fresh_critical":
            wc, wm, wf, wg, wfair = 2.0, 2.0, 1.0, 1.0, 0.2
        elif policy == "critical_fresh_ghost_safe":
            wc, wm, wf, wg, wfair = 1.0, 1.0, 1.0, 2.0, 0.2
        else:  # critical_fresh_equal
            wc, wm, wf, wg, wfair = 1.0, 1.0, 1.0, 1.0, 0.2

        _svc = svc_counts if svc_counts is not None else np.zeros(n_agents, dtype=np.int32)
        avg_svc = float(_svc.mean()) + 1e-6

        raw_crs  = np.empty(n_q, dtype=float)
        raw_rmr  = np.empty(n_q, dtype=float)
        raw_fn   = np.empty(n_q, dtype=float)
        raw_gr   = np.empty(n_q, dtype=float)
        raw_fair = np.empty(n_q, dtype=float)
        for qi, ip in enumerate(queue):
            raw_rmr[qi]  = _route_margin_risk(ip.point_idx, pt_route, n_r0, n_r1, est_rc)
            raw_crs[qi]  = float(crit_score[ip.point_idx])
            # freshness_need: age of FIELD ESTIMATE at this point (per-point, not per-agent)
            raw_fn[qi]   = float(tick - last_est_upd[ip.point_idx])
            r            = int(pt_route[ip.point_idx])
            raw_gr[qi]   = float(ghost_res[r] if r >= 0 else 0.0) * (float(ip.horizon_step) / max(horizon, 1))
            raw_fair[qi] = 1.0 / (1.0 + float(_svc[ip.agent_id]) / avg_svc)

        n_crs,  _      = _norm01(raw_crs)
        n_rmr,  _      = _norm01(raw_rmr)
        n_fn,   fn_cst = _norm01(raw_fn)
        n_gr,   _      = _norm01(raw_gr)
        n_fair, _      = _norm01(raw_fair)

        # ADDITIVE combination (fixes multiplicative bug in E11b-v1)
        scores = wc * n_crs + wm * n_rmr + wf * n_fn - wg * n_gr + wfair * n_fair
        order  = np.argsort(-scores)
        idx    = list(order[:n_sel])
        return _SelectResult(idx, fn_cst, list(raw_fn))

    raise ValueError(f"Unknown policy: {policy!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_e11bv2_single(cfg: E11bV2Config) -> dict:  # noqa: C901
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

    is_stress = cfg.regime == "ghost_stress"
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
    sel_fn:   list[float] = []   # raw freshness_need of SELECTED points (tick - last_est_upd)
    sel_gr:   list[float] = []
    sel_cf:   list[float] = []
    sel_bf:   list[float] = []
    sel_hz:   list[float] = []
    sel_age:  list[float] = []
    sel_gp:   list[float] = []
    crit_cov: list[float] = []

    # ── Freshness diagnostics ─────────────────────────────────────────────────
    fn_constant_ticks = 0     # ticks where freshness was constant across all candidates
    fn_zero_selected  = 0     # total selected points where raw freshness = 0
    fn_total_selected = 0     # total selected points

    drift_dir  = 1.0
    shift_done = False
    queue: list[_IntentPt] = []

    for tick in range(cfg.n_ticks):

        # Bottleneck shift
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
            n_through = int(np.sum(np.isin(agent_routes, [0, 1])))
            occupancy = n_through / max(n, 1)
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

        new_pts: list[_IntentPt] = []
        for ag in range(n):
            rp = r0_pts if agent_routes[ag] == 0 else r1_pts
            for h in range(1, cfg.horizon + 1):
                p = int(rp[(ag * cfg.horizon + h + tick) % len(rp)])
                new_pts.append(_IntentPt(ag, p, tick, h))
        queue.extend(new_pts)
        queue = [ip for ip in queue if tick - ip.tick_created <= cfg.max_queue_age]

        result = _select_v2(
            policy=cfg.policy,
            queue=queue,
            budget=cfg.budget_per_tick,
            tick=tick,
            n_agents=n,
            est_rc=est_rc,
            pt_route=pt_route,
            n_r0=n_r0,
            n_r1=n_r1,
            crit_score=act_crit_score,
            bott_mask=act_bott_mask,
            last_rx=last_rx,
            last_est_upd=last_est_upd,
            ghost_res=ghost_res,
            horizon=cfg.horizon,
            max_age=cfg.max_queue_age,
            w_margin=cfg.w_margin,
            w_critical=cfg.w_critical,
            w_fresh=cfg.w_fresh,
            w_ghost=cfg.w_ghost,
            rng=rng,
            svc_counts=svc_counts,
        )
        sel_idx = result.indices
        if result.fn_was_constant:
            fn_constant_ticks += 1

        tick_rmr: list[float] = []
        tick_crs: list[float] = []
        tick_fn:  list[float] = []
        tick_gr:  list[float] = []
        tick_cf:  list[bool]  = []
        tick_bf:  list[bool]  = []
        tick_hz:  list[int]   = []
        tick_age: list[float] = []
        tick_gp:  list[bool]  = []
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

            rmr = _route_margin_risk(p, pt_route, n_r0, n_r1, est_rc)
            crs = float(act_crit_score[p])
            # Raw freshness for diagnostics: age of the estimate BEFORE this update
            fn_raw = float(tick - last_est_upd[p]) + 1.0  # +1 because we just set last_est_upd=tick
            # Note: last_est_upd[p] was just written to tick, so we compute retroactively
            # Use the pre-update value via result.raw_fn_values (indexed by queue position)
            r   = int(pt_route[p])
            gr  = float(ghost_res[r] if r >= 0 else 0.0) * (float(h) / cfg.horizon)

            tick_rmr.append(rmr)
            tick_crs.append(crs)
            tick_fn.append(fn_raw)
            tick_gr.append(gr)
            tick_cf.append(bool(act_crit_mask[p]))
            tick_bf.append(bool(act_bott_mask[p]))
            tick_hz.append(h)
            tick_age.append(float(tick - last_rx[ag]))
            tick_gp.append(h > 5 and (tick - recent_switch[ag]) < 10)

        # Use pre-update raw freshness from _select_v2 if available
        if result.raw_fn_values and sel_idx:
            pre_fn = [result.raw_fn_values[qi] for qi in sel_idx]
            for v in pre_fn:
                fn_total_selected += 1
                if v < 1e-6:
                    fn_zero_selected += 1
            if tick_fn:
                # Replace with pre-update values
                tick_fn = list(pre_fn)

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
            sel_gp.append(float(np.mean(tick_gp)))

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
            regret = float(true_rc[r_ag]) - float(true_rc[best_true])
            acc_regret.append(regret)
            acc_agree.append(int(best_est == best_true))
            acc_abs_gap.append(abs(float(est_rc[r_ag]) - float(true_rc[r_ag])))

        fl = float(np.mean(np.abs(est_field - true_field)))
        acc_fl.append(fl)
        excl_all = np.concatenate([r0_excl, r1_excl]) if len(r1_excl) else r0_excl
        acc_hidden.append(float(np.mean(np.maximum(0.0, true_field[excl_all] - est_field[excl_all]))))
        acc_ghost.append(float(ghost_res.mean()))

    def _m(lst: list) -> float:
        return float(np.mean(lst)) if lst else 0.0

    rc_arr     = np.array(acc_rc_true, dtype=float)
    regret_arr = np.array(acc_regret,  dtype=float)
    agree_arr  = np.array(acc_agree,   dtype=float)
    gap_arr    = np.array(acc_abs_gap, dtype=float)

    tot_svc = svc_counts.sum()
    if tot_svc > 0:
        p_svc = svc_counts / float(tot_svc)
        ent   = float(-np.sum(p_svc[p_svc > 0] * np.log(p_svc[p_svc > 0])))
        fairness = ent / max(math.log(n), 1e-9)
    else:
        fairness = 0.0

    fn_arr = np.array(sel_fn, dtype=float)
    n_ticks = max(cfg.n_ticks, 1)

    return {
        # Core metrics
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
        # Decision-critical diagnostics
        "mean_selected_route_margin_risk":         _m(sel_rmr),
        "mean_selected_critical_region_score":     _m(sel_crs),
        "mean_selected_freshness_need":            _m(sel_fn),   # raw point-estimate age
        "min_freshness_need":                      float(fn_arr.min()) if len(fn_arr) else 0.0,
        "max_freshness_need":                      float(fn_arr.max()) if len(fn_arr) else 0.0,
        "mean_selected_ghost_risk":                _m(sel_gr),
        "fraction_selected_from_critical_regions": _m(sel_cf),
        "fraction_selected_near_bottlenecks":      _m(sel_bf),
        "mean_selected_horizon_step":              _m(sel_hz),
        "mean_selected_agent_age":                 _m(sel_age),
        "coverage_of_decision_critical_set":       _m(crit_cov),
        "ghost_prone_selected_fraction":           _m(sel_gp),
        "per_agent_service_fairness":              fairness,
        # Freshness diagnostics (new in v2)
        "freshness_proxy":                 "point_estimate_age",
        "n_ticks_constant_freshness":      int(fn_constant_ticks),
        "fraction_ticks_constant_freshness": float(fn_constant_ticks) / n_ticks,
        "fraction_zero_freshness_need":    (
            float(fn_zero_selected) / fn_total_selected if fn_total_selected > 0 else 0.0
        ),
    }

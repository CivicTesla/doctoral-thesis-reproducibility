"""E10: Packetized uplink scheduling simulation — v2.

Topology-differentiated scalar field model.

Each topology has independent dynamics:
  dual_corridor  : balanced risk priors, moderate ghost, symmetric routes
  hourglass      : elevated risk priors, high bottleneck amplification,
                   slow ghost decay, high route-switch penalty
  warehouse_grid : low risk priors, multi-path damping, fast ghost recovery,
                   low route-switch penalty, better connectivity per packet

Schedulers:
  round_robin_scheduler    : cyclic positional rotation
  aoi_aware_scheduler      : AoI since last successful delivery (service-gap-oldest-first)
  task_aware_scheduler     : route_risk × route_congestion × pkt_info_value × staleness
  proportional_fair_scheduler : throughput-weighted fair share

Framing (paper language):
  - "lightweight packet-level uplink scheduling abstraction"
  - "standards-inspired link abstraction"
  - "decision-oriented semantic scheduling"
  NOT LTE / 5G NR / Wi-Fi compliant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field
from typing import Optional

import numpy as np

# ── Packet constants ───────────────────────────────────────────────────────────
POINT_BITS  = 64     # bits per future-intention point
HEADER_BITS = 128    # packet header overhead

# ── Scenario geometry: (grid_h, grid_w, upper_route_y, lower_route_y) ─────────
_SCENARIO_GEO: dict[str, tuple[int, int, float, float]] = {
    "dual_corridor":  (50, 80, 12.0, 37.0),
    "hourglass":      (50, 80, 10.0, 39.0),
    "warehouse_grid": (55, 90, 10.0, 44.0),
}

# ── Topology-specific route/field parameters ───────────────────────────────────
#
# These override E10Config field/route settings for each topology.
# Rationale per parameter:
#   field_init_lo/hi     : prior risk level on each route
#   field_noise_std      : per-tick Brownian noise on true field
#   ghost_spike_prob     : base ghost event probability (×2 in ghost_stress regime)
#   ghost_spike_mag      : magnitude of ghost spike
#   ghost_decay_per_tick : fraction of ghost_residual that decays per tick
#                          LOW  → ghost persists for many ticks (hourglass bottleneck)
#                          HIGH → ghost fades quickly (warehouse multi-path escape)
#   bottleneck_amplify   : multiplier on ghost spike magnitude
#                          hourglass bottleneck amplifies congestion surges
#   crit_weight          : per-route bias in agent route-decision (NOT in cost metric)
#                          reflects topology asymmetry seen by agents
#   route_switch_threshold : agents switch only when |est_diff| > threshold
#                            HIGH → agents reluctant to switch (hourglass bottleneck cost)
#                            LOW  → agents reroute easily (warehouse multi-path)
#   field_info_blend     : how much one delivered packet corrects the field estimate
#                          LOW  → bottleneck degrades information flow
#                          HIGH → dense connectivity improves estimation

_TOPOLOGY_PARAMS: dict[str, dict] = {
    "dual_corridor": {
        "field_init_lo":          0.30,
        "field_init_hi":          0.70,
        "field_noise_std":        0.060,
        "ghost_spike_prob":       0.080,
        "ghost_spike_mag":        0.38,
        "ghost_decay_per_tick":   0.14,   # moderate recovery
        "bottleneck_amplify":     1.00,   # no bottleneck
        "crit_weight":            (1.00, 1.00),  # symmetric routes
        "route_switch_threshold": 0.12,
        "field_info_blend":       0.55,
    },
    "hourglass": {
        "field_init_lo":          0.42,
        "field_init_hi":          0.80,   # elevated risk at bottleneck
        "field_noise_std":        0.080,  # high volatility through bottleneck
        "ghost_spike_prob":       0.120,
        "ghost_spike_mag":        0.48,
        "ghost_decay_per_tick":   0.045,  # SLOW — ghost persists at bottleneck
        "bottleneck_amplify":     1.60,   # surges amplified through narrow passage
        "crit_weight":            (1.40, 1.00),  # upper route more critical (main corridor)
        "route_switch_threshold": 0.22,   # HIGH — rerouting through bottleneck is costly
        "field_info_blend":       0.42,   # WEAK — bottleneck degrades packet info quality
    },
    "warehouse_grid": {
        "field_init_lo":          0.22,
        "field_init_hi":          0.56,   # lower priors (multi-path damping)
        "field_noise_std":        0.048,
        "ghost_spike_prob":       0.060,
        "ghost_spike_mag":        0.32,
        "ghost_decay_per_tick":   0.24,   # FAST — many escape paths dissipate pressure
        "bottleneck_amplify":     0.70,   # grid dampens congestion surges
        "crit_weight":            (1.00, 1.20),  # lower route slightly more critical
        "route_switch_threshold": 0.07,   # LOW — easy to reroute in grid
        "field_info_blend":       0.63,   # STRONG — dense connectivity, good estimates
    },
}


# ── Config ─────────────────────────────────────────────────────────────────────
@dataclass
class E10Config:
    # Scenario / simulation
    scenario_name:    str = "dual_corridor"
    regime:           str = "normal"
    capacity_mode:    str = "constant_capacity"
    scheduler:        str = "round_robin_scheduler"
    intention_policy: str = "random_budget"
    n_agents:  int = 60
    max_ticks: int = 100
    seed:      int = 0

    # Packet model (not topology-specific)
    base_n_points:  int   = 8
    max_packet_age: int   = 10
    max_queue_size: int   = 20

    # Capacity: constant mode
    total_capacity_bpt: float = 15_000.0

    # Capacity: distance-based
    bandwidth_bpt:       float = 200.0
    path_loss_alpha:     float = 3.0
    sinr_ref:            float = 20.0
    d_ref:               float = 10.0
    d_min:               float = 5.0
    max_rate_per_agent:  float = 1_000.0

    # Capacity: Markov fading
    good_rate_bpt:   float = 800.0
    bad_rate_bpt:    float = 100.0
    p_good_to_bad:   float = 0.05
    p_bad_to_good:   float = 0.20
    good_per:        float = 0.05
    bad_per:         float = 0.30

    # Legacy fields kept for backward compat — overridden by _TOPOLOGY_PARAMS
    route_switch_threshold: float = 0.12
    field_noise_std:        float = 0.06
    ghost_spike_prob:       float = 0.08
    ghost_spike_mag:        float = 0.40
    field_info_blend:       float = 0.55


# ── Packet ─────────────────────────────────────────────────────────────────────
@dataclass
class _Pkt:
    agent_id:     int
    tick_created: int
    target_route: int    # 0=upper, 1=lower, -1=both
    n_points:     int
    info_value:   float  # 0-1, quality of field estimate carried

    @property
    def size_bits(self) -> int:
        return HEADER_BITS + self.n_points * POINT_BITS


# ── Packet generation ──────────────────────────────────────────────────────────
def _gen_packet(agent_id: int, tick: int, route: int,
                cfg: E10Config, rng: np.random.Generator) -> Optional[_Pkt]:
    pol = cfg.intention_policy

    if pol == "random_budget":
        n_pts = int(rng.integers(3, 13))
        iv    = float(rng.uniform(0.30, 0.70))
        tgt   = int(rng.integers(0, 2))

    elif pol == "adaptive_ratio_rule":
        n_pts = cfg.base_n_points
        iv    = 0.78
        tgt   = route

    elif pol == "fixed_hybrid_25_75":
        n_pts = cfg.base_n_points
        iv    = 0.60
        tgt   = route if rng.random() < 0.75 else (1 - route)

    elif pol == "periodic_full_reference":
        if tick % 5 == 0:
            n_pts = cfg.base_n_points * 2
            iv    = 1.00
            tgt   = -1
        else:
            return None

    else:
        raise ValueError(f"Unknown intention_policy: {pol!r}")

    return _Pkt(agent_id=agent_id, tick_created=tick,
                target_route=tgt, n_points=n_pts, info_value=iv)


# ── Uplink capacity ────────────────────────────────────────────────────────────
def _compute_caps(cfg: E10Config, n: int,
                  pos_x: np.ndarray, pos_y: np.ndarray,
                  bs_x: float, bs_y: float,
                  channel_good: np.ndarray) -> tuple[np.ndarray, float]:
    if cfg.capacity_mode == "constant_capacity":
        agent_cap = np.full(n, np.inf)
        total_cap = cfg.total_capacity_bpt

    elif cfg.capacity_mode == "distance_based_capacity":
        dist      = np.maximum(np.hypot(pos_x - bs_x, pos_y - bs_y), cfg.d_min)
        sinr      = cfg.sinr_ref / (dist / cfg.d_ref) ** cfg.path_loss_alpha
        rate      = cfg.bandwidth_bpt * np.log2(1.0 + sinr)
        rate      = np.minimum(rate, cfg.max_rate_per_agent)
        agent_cap = rate
        total_cap = float(rate.sum())

    elif cfg.capacity_mode == "markov_fading_capacity":
        rate      = np.where(channel_good, cfg.good_rate_bpt, cfg.bad_rate_bpt)
        agent_cap = rate
        total_cap = float(rate.sum())

    else:
        raise ValueError(f"Unknown capacity_mode: {cfg.capacity_mode!r}")

    return agent_cap, total_cap


# ── Task-score helper (used by TaskAware scheduler and diagnostics) ────────────
def _task_score(i: int, queues: list[list[_Pkt]], tick: int,
                max_packet_age: int, est_field: np.ndarray,
                agent_routes: np.ndarray,
                route_congestion: np.ndarray) -> float:
    """Compute task-priority score for agent i's head-of-queue packet.

    score = route_risk × (1 + route_congestion) × info_value × staleness_fraction

    - route_risk        : estimated field intensity on agent's current route
    - route_congestion  : fraction of all agents currently on the same route
    - info_value        : quality of the packet's field estimate
    - staleness_fraction: how much of max_packet_age has elapsed (0→1)
    """
    if not queues[i]:
        return 0.0
    pkt       = queues[i][0]
    risk      = float(est_field[agent_routes[i]])
    cong      = float(route_congestion[agent_routes[i]])
    staleness = (tick - pkt.tick_created + 1) / max(max_packet_age, 1)
    return risk * (1.0 + cong) * pkt.info_value * staleness


# ── Scheduler priority ordering ────────────────────────────────────────────────
def _priority_order(cfg: E10Config, n: int,
                    queues: list[list[_Pkt]],
                    agent_cap: np.ndarray,
                    agent_routes: np.ndarray,
                    est_field: np.ndarray,
                    tick: int,
                    rr_ptr: int,
                    served_ema: np.ndarray,
                    last_served_tick: np.ndarray,
                    route_congestion: np.ndarray) -> list[int]:
    sch = cfg.scheduler

    if sch == "round_robin_scheduler":
        # Pure cyclic rotation — no info about queue state or channel
        return [(rr_ptr + i) % n for i in range(n)]

    if sch == "proportional_fair_scheduler":
        scores = np.where(served_ema > 0, agent_cap / served_ema, agent_cap)
        return np.argsort(-scores).tolist()

    if sch == "aoi_aware_scheduler":
        # AoI = ticks since last SUCCESSFUL delivery (service gap), not packet age.
        # Agents never served get the largest AoI.
        # Never-served agents get aoi = tick + 1 (guaranteed highest priority initially).
        aoi = np.where(last_served_tick >= 0,
                       tick - last_served_tick,
                       tick + 1).astype(float)
        # Only score agents that have packets to send
        has_pkt = np.array([len(q) > 0 for q in queues], dtype=float)
        scores  = aoi * has_pkt
        return np.argsort(-scores).tolist()

    if sch == "task_aware_scheduler":
        # task_score = route_risk × route_congestion × packet_info_value × staleness
        # Prioritises agents on risky congested routes with old, high-value packets.
        scores = np.array([
            _task_score(i, queues, tick, cfg.max_packet_age,
                        est_field, agent_routes, route_congestion)
            for i in range(n)
        ], dtype=float)
        return np.argsort(-scores).tolist()

    raise ValueError(f"Unknown scheduler: {cfg.scheduler!r}")


# ── Main simulation ────────────────────────────────────────────────────────────
def run_e10_single(cfg: E10Config) -> dict[str, float]:  # noqa: C901
    """Run one E10 simulation; return a flat dict of metrics."""
    rng = np.random.default_rng(cfg.seed)
    n   = cfg.n_agents

    # Load topology-specific parameters (override E10Config field/route settings)
    topo  = _TOPOLOGY_PARAMS.get(cfg.scenario_name, _TOPOLOGY_PARAMS["dual_corridor"])
    geo   = _SCENARIO_GEO.get(cfg.scenario_name, _SCENARIO_GEO["dual_corridor"])
    grid_h, grid_w, upper_y, lower_y = geo
    bs_x, bs_y = grid_w / 2.0, grid_h / 2.0

    field_init_lo    = topo["field_init_lo"]
    field_init_hi    = topo["field_init_hi"]
    noise_std        = topo["field_noise_std"]
    ghost_prob_base  = topo["ghost_spike_prob"]
    ghost_mag        = topo["ghost_spike_mag"]
    ghost_decay      = topo["ghost_decay_per_tick"]
    bott_amplify     = topo["bottleneck_amplify"]
    crit_weight      = np.array(topo["crit_weight"], dtype=float)
    switch_thresh    = topo["route_switch_threshold"]
    info_blend_base  = topo["field_info_blend"]

    # ghost_stress regime doubles spike frequency
    ghost_prob = ghost_prob_base * (2.0 if cfg.regime == "ghost_stress" else 1.0)

    # ── Agent state ───────────────────────────────────────────────────────────
    agent_routes  = rng.integers(0, 2, size=n)
    prev_routes   = agent_routes.copy()

    pos_x = rng.uniform(5, grid_w - 5, size=n)
    pos_y = np.where(agent_routes == 0, upper_y, lower_y).astype(float)

    channel_good = rng.random(n) > cfg.p_good_to_bad

    # Per-agent queues (FIFO, oldest packet at index 0)
    queues: list[list[_Pkt]] = [[] for _ in range(n)]

    # ── Field state ───────────────────────────────────────────────────────────
    true_field    = rng.uniform(field_init_lo, field_init_hi, size=2)
    est_field     = true_field.copy()
    ghost_residual = np.zeros(2)   # persistent ghost pressure per route

    # ── Scheduler state ───────────────────────────────────────────────────────
    avg_pkt_size = HEADER_BITS + cfg.base_n_points * POINT_BITS
    served_ema   = np.full(n, max(cfg.total_capacity_bpt / n, avg_pkt_size))
    served_cnt   = np.zeros(n)
    last_served_tick = np.full(n, -1, dtype=np.int64)  # -1 = never delivered
    rr_ptr = 0

    # ── Accumulators ──────────────────────────────────────────────────────────
    total_generated    = 0
    total_delivered    = 0
    total_expired_pkts = 0
    overflow_drops     = 0
    delays:       list[float] = []
    ages_deliv:   list[float] = []
    q_lens:       list[float] = []
    bits_tx_hist: list[float] = []
    bits_av_hist: list[float] = []

    route_costs:  list[float] = []
    perf_costs:   list[float] = []
    upper_errs:   list[float] = []
    lower_errs:   list[float] = []
    agreements:   list[float] = []
    regrets:      list[float] = []
    oscillations  = 0
    total_switches = 0

    ghost_residual_hist: list[float] = []

    # Scheduler diagnostic accumulators
    sched_ages_delivered:  list[float] = []
    sched_ivs_delivered:   list[float] = []
    sched_task_scores_del: list[float] = []
    sched_route0_del = 0
    sched_route1_del = 0
    sched_both_del   = 0

    # ── Tick loop ─────────────────────────────────────────────────────────────
    for tick in range(cfg.max_ticks):

        # 1. Route congestion (used by TaskAware)
        route_counts    = np.bincount(agent_routes, minlength=2)
        route_congestion = route_counts.astype(float) / max(n, 1)

        # 2. Evolve true field
        true_field += rng.normal(0, noise_std, size=2)
        ghost_residual *= (1.0 - ghost_decay)          # decay residual
        true_field     += ghost_residual * 0.35        # residual adds sustained pressure
        true_field      = np.clip(true_field, 0.05, 0.95)

        if rng.random() < ghost_prob:
            r_spike = int(rng.integers(0, 2))
            effective_mag = ghost_mag * bott_amplify   # amplified at bottleneck
            spike = min(effective_mag, 0.90 - true_field[r_spike])
            true_field[r_spike] = np.clip(true_field[r_spike] + spike, 0.05, 0.95)
            ghost_residual[r_spike] = min(
                ghost_residual[r_spike] + spike * 0.55, ghost_mag)

        ghost_residual_hist.append(float(ghost_residual.mean()))

        # Field estimate drifts slowly toward uncertain prior when no updates
        est_field += (0.5 - est_field) * 0.008

        # 3. Expire old packets
        for i in range(n):
            before = len(queues[i])
            queues[i] = [p for p in queues[i]
                         if tick - p.tick_created < cfg.max_packet_age]
            total_expired_pkts += before - len(queues[i])

        # 4. Generate packets
        for i in range(n):
            pkt = _gen_packet(i, tick, int(agent_routes[i]), cfg, rng)
            if pkt is None:
                continue
            total_generated += 1
            if len(queues[i]) < cfg.max_queue_size:
                queues[i].append(pkt)
            else:
                overflow_drops += 1

        q_lens.append(float(sum(len(q) for q in queues)) / n)

        # 5. Compute per-agent capacities
        agent_cap, total_cap = _compute_caps(cfg, n, pos_x, pos_y,
                                             bs_x, bs_y, channel_good)
        bits_av_hist.append(float(total_cap))

        # 6. Update Markov channel states
        if cfg.capacity_mode == "markov_fading_capacity":
            u   = rng.random(n)
            nxt = channel_good.copy()
            nxt[ channel_good] = u[ channel_good] > cfg.p_good_to_bad
            nxt[~channel_good] = u[~channel_good] < cfg.p_bad_to_good
            channel_good = nxt

        # 7. Schedule and deliver
        order  = _priority_order(cfg, n, queues, agent_cap, agent_routes,
                                 est_field, tick, rr_ptr, served_ema,
                                 last_served_tick, route_congestion)
        rr_ptr = (rr_ptr + 1) % n
        budget  = float(total_cap)
        ag_left = agent_cap.astype(float).copy()
        bits_tx = 0.0

        for i in order:
            if not queues[i]:
                continue
            pkt = queues[i][0]
            if pkt.size_bits > budget:
                continue
            if pkt.size_bits > ag_left[i]:
                continue
            # Markov packet error
            if cfg.capacity_mode == "markov_fading_capacity":
                per = cfg.good_per if channel_good[i] else cfg.bad_per
                if rng.random() < per:
                    queues[i].pop(0)   # errored — retransmit next tick
                    continue
            # Deliver
            queues[i].pop(0)
            delay = tick - pkt.tick_created
            delays.append(float(delay))
            ages_deliv.append(float(delay))
            total_delivered       += 1
            served_cnt[i]         += 1
            bits_tx               += pkt.size_bits
            budget                -= pkt.size_bits
            ag_left[i]            -= pkt.size_bits
            last_served_tick[i]    = tick

            # Scheduler diagnostics (inline to avoid index aliasing with _task_score)
            sched_ages_delivered.append(float(delay))
            sched_ivs_delivered.append(float(pkt.info_value))
            _risk  = float(est_field[agent_routes[i]])
            _cong  = float(route_congestion[agent_routes[i]])
            _stale = (tick - pkt.tick_created + 1) / max(cfg.max_packet_age, 1)
            sched_task_scores_del.append(_risk * (1.0 + _cong) * pkt.info_value * _stale)
            if pkt.target_route == 0:
                sched_route0_del += 1
            elif pkt.target_route == 1:
                sched_route1_del += 1
            else:
                sched_both_del += 1

            # Update field estimate (topology-specific blend)
            tgt   = pkt.target_route
            blend = info_blend_base * pkt.info_value
            if tgt == -1:
                est_field += blend * (true_field - est_field)
            else:
                est_field[tgt] += blend * (true_field[tgt] - est_field[tgt])

            # EMA served bits
            served_ema[i] = 0.9 * served_ema[i] + 0.1 * pkt.size_bits

            if budget <= 0:
                break

        bits_tx_hist.append(bits_tx)

        # 8. Route decisions
        # Oracle: uses topology-biased true field to pick best route
        biased_true = true_field * crit_weight
        perf_r      = int(biased_true[1] < biased_true[0])  # lower index = higher field

        # Agents use BIASED estimated field (critical-weight aware)
        biased_est  = est_field * crit_weight
        new_routes  = agent_routes.copy()
        for i in range(n):
            cur = int(agent_routes[i])
            alt = 1 - cur
            if biased_est[alt] < biased_est[cur] - switch_thresh:
                new_routes[i] = alt

        # Oscillation: switched this tick AND switched last tick
        if tick > 0:
            osc = int(((new_routes != agent_routes)
                       & (agent_routes != prev_routes)).sum())
            oscillations += osc

        switches = int((new_routes != agent_routes).sum())
        total_switches += switches

        prev_routes  = agent_routes.copy()
        agent_routes = new_routes

        # Update positions
        pos_y = np.where(agent_routes == 0, upper_y, lower_y).astype(float)
        pos_x = np.clip(pos_x + rng.normal(0, 2, size=n), 5, grid_w - 5)

        # Route cost = mean true field on agents' chosen routes (no crit bias in metric)
        cost_tick = float(np.mean(true_field[agent_routes]))
        route_costs.append(cost_tick)
        perf_costs.append(float(true_field[perf_r]))

        perf_routes_vec = np.full(n, perf_r, dtype=int)
        agreements.append(float(np.mean(agent_routes == perf_routes_vec)))
        regret_vec = true_field[agent_routes] - true_field[perf_routes_vec]
        regrets.append(float(np.mean(regret_vec)))

        upper_errs.append(float(abs(true_field[0] - est_field[0])))
        lower_errs.append(float(abs(true_field[1] - est_field[1])))

    # ── Final metrics ─────────────────────────────────────────────────────────
    eps         = 1e-9
    pdr         = total_delivered / max(total_generated, 1)
    efr         = total_expired_pkts / max(total_generated, 1)
    odr         = overflow_drops / max(total_generated, 1)

    sc2      = served_cnt ** 2
    fairness = (served_cnt.sum() ** 2) / (n * sc2.sum() + eps)

    total_avail = sum(bits_av_hist) + eps
    utilization = min(sum(bits_tx_hist) / total_avail, 1.0)

    avg_cost    = float(np.mean(route_costs))
    g1          = float(np.mean(upper_errs))
    g2          = float(np.mean(lower_errs))

    # Scheduler diagnostics
    del_n     = max(total_delivered, 1)
    del_total = sched_route0_del + sched_route1_del + sched_both_del

    svc_frac  = served_cnt / (served_cnt.sum() + eps)
    svc_entr  = float(-np.sum(svc_frac * np.log2(svc_frac + eps)))
    max_entr  = math.log2(n) if n > 1 else 1.0
    svc_entr_norm = svc_entr / max_entr  # 0=one agent dominates, 1=perfectly even

    return {
        # ── Field / route metrics ────────────────────────────────────────────
        "normalized_route_cost":          avg_cost,
        "route_balanced_loss_g1":         g1,
        "route_balanced_loss_g2":         g2,
        "global_field_loss":              (g1 + g2) / 2.0,
        "mean_abs_risk_gap_current":      float(abs(np.mean(regrets))),
        "mean_route_decision_agreement":  float(np.mean(agreements)),
        "mean_route_decision_regret":     float(np.mean(regrets)),
        "oscillation_count":              oscillations / max(cfg.max_ticks, 1),
        "route_switch_rate":              total_switches / max(cfg.max_ticks * n, 1),
        "ghost_residual_mean":            float(np.mean(ghost_residual_hist)),
        # ── Packet / network metrics ─────────────────────────────────────────
        "packet_delivery_ratio":          pdr,
        "mean_packet_delay":              float(np.mean(delays))       if delays else float(cfg.max_packet_age),
        "p95_packet_delay":               float(np.percentile(delays, 95)) if len(delays) > 1 else float(cfg.max_packet_age),
        "mean_packet_age_at_delivery":    float(np.mean(ages_deliv))   if ages_deliv else float(cfg.max_packet_age),
        "expired_packet_fraction":        efr,
        "queue_overflow_drop_fraction":   odr,
        "average_queue_length":           float(np.mean(q_lens))       if q_lens else 0.0,
        "uplink_utilization":             float(utilization),
        "scheduler_fairness_index":       float(fairness),
        # ── Scheduler diagnostics ────────────────────────────────────────────
        "sched_mean_selected_age":        float(np.mean(sched_ages_delivered))  if sched_ages_delivered  else 0.0,
        "sched_mean_selected_iv":         float(np.mean(sched_ivs_delivered))   if sched_ivs_delivered   else 0.0,
        "sched_route0_delivery_frac":     sched_route0_del / del_total          if del_total > 0 else 0.0,
        "sched_route1_delivery_frac":     sched_route1_del / del_total          if del_total > 0 else 0.0,
        "sched_service_entropy_norm":     float(svc_entr_norm),
        "sched_mean_task_score":          float(np.mean(sched_task_scores_del)) if sched_task_scores_del else 0.0,
    }

"""Official event-coupled runner shared by E5, E6, and E7."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import hashlib
import math
import os
import platform
import random
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from event_coupled_field import EventCoupledFieldPipeline
from a1_method_configs import METHOD_CONFIGS, get_method_config
from a1_pilot_tasks import (
    AgentOrderProvider,
    TASK_STREAM_OFFSET,
    build_task_trace,
    task_trace_hash,
)
from e6_network_model import COMM_NETWORK_REGIMES
from planning_admission import PLANNING_SCHEDULER_IDENTITY, configuration_hash
from a1_instrumentation import (
    deterministic_hash as _deterministic_hash,
    outcome_state_hash as _outcome_state_hash,
    planning_counts as _planning_counts,
    runtime_per_order as _runtime_per_order,
    summarize_risk,
)

REGIMES = ["Ideal", "MultiBS", "CenterBS", "Poor"]
POLICIES = ["Periodic-U2", "Periodic-U5", "AoITriggered", "CongestionTriggered",
            "TrajectoryChangeTriggered", "ValueAware", "ValueAwareV2"]
SEMANTIC_BASE_COMMIT = "46157b8ceadd8cf8ea8cda2ab5749fad5fe4e510"
INSTRUMENTATION_VERSION = "a1_event_audit_v2_progress"
OUTPUT_SCHEMA_VERSION = "a1_method_regime_instrumented_v2"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / filename)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def audit_mechanism_flags(audit_mechanism):
    """Return the two field-mechanism switches without changing method semantics."""
    if audit_mechanism not in {"M00", "M10", "M01", "M11"}:
        raise ValueError("audit_mechanism must be M00, M10, M01, or M11")
    return {
        "exclude_self_contribution": audit_mechanism in {"M10", "M11"},
        "versioned_replacement": audit_mechanism in {"M01", "M11"},
    }


class PipelineAgeAdapter:
    def __init__(self, pipeline): self.pipeline = pipeline
    @property
    def last_delivery_tick(self):
        # Policies call tick-last_delivery_tick. This adapter yields generation
        # timestamps, so AoI policies use source age rather than completion age.
        return [g if g is not None else -1_000_000
                for g in self.pipeline.last_usable_generation_tick]


class EventCommunicationManager:
    def __init__(self, pipeline, update_interval=1, policy=None, budget=None, horizon=24):
        self.pipeline, self.update_interval, self.policy = pipeline, update_interval, policy
        self.budget, self.horizon = budget, horizon
        self.last_uploaded_path = {}
        self.last_upload_tick = {}
        self.candidate_uploads = self.selected_uploads = 0
        self.selected_per_tick = []
        self.pending_generated = {}
        self.uploaded_this_tick = set()

    def _enqueue(self, agent, path, weight, update_type):
        tick = self.pipeline.digital_twin.current_tick
        if self.pipeline.selected is not None and agent.agent_id not in self.pipeline.selected:
            self.pending_generated[agent.agent_id] = (list(path[:self.horizon]), weight, update_type, tick)
            return False
        accepted = self.pipeline.enqueue(agent.agent_id, agent.position,
            tick, path[:self.horizon], weight, update_type)
        if accepted:
            self.uploaded_this_tick.add(agent.agent_id)
            self.last_uploaded_path[agent.agent_id] = list(path[:self.horizon])
            self.last_upload_tick[agent.agent_id] = self.pipeline.digital_twin.current_tick
        return accepted

    def prepare_tick(self, agents):
        tick = self.pipeline.digital_twin.current_tick
        self.uploaded_this_tick = set()
        self.pipeline.advance(tick, {a.agent_id: a.position for a in agents})
        active = [a for a in agents if not a.finished]
        candidates = [a for a in active if a.path]
        self.candidate_uploads += len(candidates)
        if self.policy is None:
            selected = candidates if tick % self.update_interval == 0 else []
        else:
            selected = self.policy.select(candidates, self.budget, tick,
                self.pipeline.digital_twin, self.last_uploaded_path,
                PipelineAgeAdapter(self.pipeline))
        if self.budget is not None and len(selected) > self.budget:
            raise AssertionError("selected uploads exceed scheduling budget")
        ids = [a.agent_id for a in selected]
        self.pipeline.select(ids)
        self.selected_uploads += len(ids); self.selected_per_tick.append(len(ids))
        for agent in agents:
            agent.can_upload = agent.agent_id in set(ids)
            agent.communication_status = "event-coupled"
            agent.communicated_reservation_handler = self._enqueue
        for agent in selected:
            pending = self.pending_generated.pop(agent.agent_id, None)
            if pending is not None:
                path, weight, update_type, generation_tick = pending
                accepted = self.pipeline.enqueue(agent.agent_id, agent.position, tick,
                    path, weight, update_type, generation_tick=generation_tick)
                if accepted:
                    self.uploaded_this_tick.add(agent.agent_id)
                    self.last_uploaded_path[agent.agent_id] = list(path)
                    self.last_upload_tick[agent.agent_id] = tick
        self.pipeline.record_ages(tick, [a.agent_id for a in agents],
            [a.agent_id for a in active if a.path])

    def reserve_rolling_predictions(self, agents, horizon, weight):
        for agent in agents:
            if not agent.finished and agent.path and agent.agent_id not in self.uploaded_this_tick:
                self._enqueue(agent, agent.path[:horizon], weight, "rolling_refresh")


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_evidence():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT, text=True)
        return commit, bool(dirty.strip()), hashlib.sha256(dirty.encode()).hexdigest()
    except (OSError, subprocess.SubprocessError):
        return "unavailable", True, "unavailable"


def _create_pilot_agents(sim, n_agents, map_manager, dt, planner, trace):
    starts = sim.generate_start_positions(n_agents, map_manager)
    agents = []
    for agent_id in range(n_agents):
        provider = AgentOrderProvider(trace[agent_id])
        agents.append(sim.WarehouseAgent(
            agent_id=agent_id,
            start=starts[agent_id],
            order=trace[agent_id][0],
            map_manager=map_manager,
            digital_twin=dt,
            planner=planner,
            order_provider=provider,
            reservation_horizon=25,
        ))
    return agents, starts


def run_case(experiment, seed, agents_n, update_interval, regime, policy_name,
             max_ticks, budget, queue_policy, method="proposed",
             deterministic_task_trace=False, audit_mechanism="M00",
             progress_path=None, progress_every_ticks=0):
    mechanism_flags = audit_mechanism_flags(audit_mechanism)
    exclude_self = mechanism_flags["exclude_self_contribution"]
    versioned_replacement = mechanism_flags["versioned_replacement"]
    random.seed(seed)
    method_config = get_method_config(method)
    sim = _load("cx_sim_event", "CX-simulation.py")
    tc_mod = _load("cx_tc_event", "CX-traffic_controller.py")
    policy = None
    if experiment == "E7":
        pol_mod = _load("e7_policy_event", "e7_update_policies.py")
        policy = pol_mod.make_policy(policy_name, rng=random.Random(seed + 33333))
    map_manager = sim.MapManager()
    dt = sim.DigitalTwin(p1_alpha=1.0, p1_beta=0.02, p1_decay=0.18,
        p1_threshold=method_config.p1_threshold,
        use_p1=method_config.use_field, communication_field=None,
        exclude_self_contribution=exclude_self,
        versioned_replacement=versioned_replacement)
    planner = sim.ReservationAStarPlanner(
        map_manager, dt, p1_weight=method_config.p1_weight)
    if deterministic_task_trace:
        task_trace = build_task_trace(seed, agents_n)
        agents, starts = _create_pilot_agents(
            sim, agents_n, map_manager, dt, planner, task_trace)
        trace_hash = task_trace_hash(task_trace)
    else:
        agents = sim.create_agents(
            agents_n, map_manager, dt, planner, reservation_horizon=25)
        starts = [agent.position for agent in agents]
        task_trace = [[list(agent.order)] for agent in agents]
        trace_hash = task_trace_hash(task_trace)
    pipeline = EventCoupledFieldPipeline(regime, dt, agents_n,
        random.Random(seed + 55555), queue_policy=queue_policy,
        censor_penalty=max_ticks+1, versioned_replacement=versioned_replacement)
    effective_budget = budget if experiment == "E7" else None
    manager = EventCommunicationManager(pipeline, update_interval=update_interval,
        policy=policy, budget=effective_budget)
    traffic = tc_mod.TrafficController(
        map_manager, dt, enable_risk_replan=method_config.enable_risk_replan,
        risk_scan_horizon=25, risk_threshold=15.0, risk_hard_threshold=2.0,
        risk_replan_cooldown=12, planning_budget_per_tick=10,
        rolling_reservation_horizon=24, rolling_reservation_weight=0.2,
        reservation_update_interval=1 if experiment == "E7" else update_interval)
    started = time.monotonic()
    time_to_10_orders = max_ticks + 1
    first_successful_order_tick = max_ticks + 1
    field_mass_samples, stale_mass_samples = [], []

    def emit_progress(tick_completed, status):
        """Persist a non-semantic checkpoint after a completed simulation tick.

        This deliberately reads only already-computed state and wall-clock/
        process accounting; it neither changes the simulation objects nor
        samples any random stream.  A scheduler signal can therefore leave a
        useful last checkpoint without changing the run's random sequence.
        """
        if progress_path is None:
            return
        route_outcomes = traffic.route_outcome_history
        external_events = [event for event in route_outcomes
                           if event["request_type"] == "risk_replan"
                           and event.get("external_threshold_trigger")]
        effective_events = [event for event in external_events
                            if event.get("external_potential_conflict_resolved")]
        elapsed = time.monotonic() - started
        atomic_json(Path(progress_path), {
            "status": status,
            "tick_completed": tick_completed,
            "ticks_total": max_ticks,
            "elapsed_seconds": elapsed,
            "ticks_per_second": tick_completed / elapsed if elapsed else 0.0,
            "delivered_orders": sum(agent.delivered_orders for agent in agents),
            "total_waits": sum(agent.wait_steps for agent in agents),
            "risk_replans": sum(agent.risk_replan_count for agent in agents),
            "external_risk_triggered_replans": len(external_events),
            "effective_external_risk_replans": len(effective_events),
            # Linux ru_maxrss is KiB; R1H records that unit explicitly.
            "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "seed": seed,
            "method": method,
            "audit_mechanism": audit_mechanism,
        })

    emit_progress(0, "initialized")
    for tick in range(max_ticks):
        traffic.step(
            agents, manager if method_config.upload_intentions else None)
        field_mass_samples.append(sum(dt.p1_time.values()))
        stale_mass_samples.append(dt.superseded_contribution_mass())
        if time_to_10_orders == max_ticks + 1:
            if sum(a.delivered_orders for a in agents) >= 10:
                time_to_10_orders = tick
        if first_successful_order_tick == max_ticks + 1:
            if sum(a.delivered_orders for a in agents) >= 1:
                first_successful_order_tick = tick
        if (progress_every_ticks and (tick + 1) % progress_every_ticks == 0):
            emit_progress(tick + 1, "running")
    runtime = time.monotonic() - started
    stats = pipeline.statistics(); invariants = pipeline.invariant_report()
    planning_metrics = traffic.planning_admission.metrics(
        expected_initial_agents=agents_n)
    planning_history = traffic.planning_admission.admission_history
    planning_request_history = traffic.planning_admission.request_audit_history
    route_outcomes = traffic.route_outcome_history
    planning_sequence_hash = _deterministic_hash(planning_history)
    planning_request_trace_hash = _deterministic_hash(planning_request_history)
    route_replan_outcome_hash = _deterministic_hash(route_outcomes)
    communication_event_hash = _deterministic_hash(pipeline.network_event_history)
    admission_margin = math.ceil(agents_n / 10) + 2
    invariants["planning_budget_respected"] = (
        planning_metrics["planning_budget_violations"] == 0)
    invariants["planning_queue_conserved"] = (
        planning_metrics["planning_conservation_violations"] == 0)
    invariants["initial_planning_admission_live"] = (
        planning_metrics["planning_agents_never_admitted"] == 0
        and planning_metrics["planning_max_initial_admission_tick"] <= admission_margin)
    if effective_budget is not None:
        invariants["selected_uploads_within_budget"] = all(x <= budget for x in manager.selected_per_tick)
    invariants["planner_and_delivery_same_field"] = planner.dt is pipeline.digital_twin
    field_values = list(dt.p1_time.values())
    invariants["no_nan_or_inf"] = all(math.isfinite(x) for x in field_values)
    invariants["baseline_field_empty"] = (
        method != "baseline" or not field_values)
    invariants["baseline_no_payloads"] = (
        method != "baseline" or pipeline.attempts == 0)
    invariants["intention_only_no_risk_replans"] = (
        method != "intention_only"
        or sum(a.risk_replan_count for a in agents) == 0)
    invariants["communication_required_for_field_updates"] = (
        not method_config.upload_intentions
        or pipeline.field_updates == dt.upload_events)
    config = {"experiment":experiment,"agents":agents_n,"ticks":max_ticks,
        "method":method,
        "audit_mechanism":audit_mechanism,
        "exclude_self_contribution":exclude_self,
        "versioned_replacement":versioned_replacement,
        "update_interval":update_interval,"network_regime":regime,"update_policy":policy_name or "Periodic",
        "budget":effective_budget,"queue_policy":queue_policy,"reservation_horizon":25,
        "rolling_horizon":24,"rolling_weight":0.2,"p1_alpha":1.0,"p1_beta":0.02,
        "p1_decay":0.18,"p1_threshold":method_config.p1_threshold,
        "p1_weight":method_config.p1_weight,
        "use_field":method_config.use_field,
        "upload_intentions":method_config.upload_intentions,
        "enable_risk_replan":method_config.enable_risk_replan,
        "payload_bytes_per_item":12,"header_bytes":32,"query_offset_enabled":False,
        "deterministic_task_trace":deterministic_task_trace,
        "semantics_version":"event-coupled-method-comparison-v1",
        **PLANNING_SCHEDULER_IDENTITY}
    semantic_config_hash = configuration_hash(config)
    full_run_config = {
        **config,
        "semantic_base_commit": SEMANTIC_BASE_COMMIT,
        "instrumentation_version": INSTRUMENTATION_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
    }
    full_run_config_hash = configuration_hash(full_run_config)
    base_config = {k:v for k,v in config.items() if k not in {"network_regime","update_policy","update_interval"}}
    base_config_hash = hashlib.sha256(json.dumps(base_config,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    code_commit, git_dirty, git_dirty_hash = _git_evidence()
    map_hashes = {
        name: _sha256_file(ROOT / "CodePheromon" / name)
        for name in ("MapMatrix.npy", "map.png", "labeled_red_points_by_warehouse.npz")
    }
    rng_seeds = {
        "task_order": seed + TASK_STREAM_OFFSET if deterministic_task_trace else seed,
        "agent_initialization": "deterministic-no-rng",
        "network_delivery_jitter": seed + 55555,
        "scheduling_tiebreak": seed + 33333 if experiment == "E7" else "deterministic",
    }
    legacy_outcome_state = [{
        "agent_id": a.agent_id, "position": a.position,
        "orders": a.delivered_orders, "items": a.total_completed_items,
        "waits": a.wait_steps, "steps": a.total_steps,
        "replans": a.replan_count, "risk_replans": a.risk_replan_count,
        "planned_path_lengths": a.planned_path_lengths,
    } for a in agents]
    legacy_outcome_hash = _deterministic_hash(legacy_outcome_state)
    field_state = sorted(
        (list(key) if isinstance(key, tuple) else key, value)
        for key, value in dt.p1_time.items())
    field_state_hash = _deterministic_hash(field_state)
    planning_counts = _planning_counts(planning_history)
    risk_metrics = summarize_risk(
        planning_history, planning_request_history, route_outcomes)
    external_risk_events = [event for event in route_outcomes
                            if event["request_type"] == "risk_replan"
                            and event.get("external_threshold_trigger")]
    effective_external_risk_events = [event for event in external_risk_events
                                      if event.get("external_potential_conflict_resolved")]
    invariants["planning_type_counts_sum_to_total"] = (
        sum(planning_counts.values()) == len(planning_history))
    invariants["risk_admissions_are_budgeted"] = (
        planning_counts["risk_replan"] == risk_metrics["risk_requests_admitted"])
    invariants["risk_execution_outcomes_conserved"] = (
        risk_metrics["risk_requests_completed"]
        == risk_metrics["risk_requests_admitted"])
    invariants["network_rng_call_accounting"] = (
        pipeline.rng_random_calls == pipeline.transmitted
        and pipeline.rng_randint_calls == pipeline.transmitted - pipeline.losses)
    outcome_state = {
        "agents": [{
            **legacy_outcome_state[index],
            "current_order": a.order,
            "completed_items": a.completed_items,
            "current_path_hash": _deterministic_hash(a.path),
            "status": a.status,
        } for index, a in enumerate(agents)],
        "planning_counts": planning_counts,
        "planning_request_trace_hash": planning_request_trace_hash,
        "planning_admission_sequence_hash": planning_sequence_hash,
        "route_replan_outcome_hash": route_replan_outcome_hash,
        "communication_event_hash": communication_event_hash,
        "communication_counters": stats,
        "field_state_hash": field_state_hash,
    }
    outcome_hash = _outcome_state_hash(outcome_state)
    row = {"experiment_id": f"{experiment}_event_coupled", "coupling": "event_coupled",
        "configuration_hash":full_run_config_hash,
        "semantic_configuration_hash":semantic_config_hash,
        "full_run_configuration_hash":full_run_config_hash,
        "base_configuration_hash":base_config_hash,
        "semantic_base_commit":SEMANTIC_BASE_COMMIT,
        "instrumentation_version":INSTRUMENTATION_VERSION,
        "output_schema_version":OUTPUT_SCHEMA_VERSION,
        "semantics_version":config["semantics_version"],"method":method,
        "method_configuration":json.dumps(method_config.as_dict(),sort_keys=True),
        "full_configuration":json.dumps(full_run_config,sort_keys=True),
        "network_configuration":json.dumps(COMM_NETWORK_REGIMES[regime],sort_keys=True),
        "code_commit":code_commit,"git_dirty":git_dirty,
        "git_dirty_status_hash":git_dirty_hash,
        "map_hashes":json.dumps(map_hashes,sort_keys=True),
        "task_order_trace_hash":trace_hash,
        "initial_state_hash":hashlib.sha256(json.dumps(
            {"starts":starts,"initial_orders":[x[0] for x in task_trace]},
            sort_keys=True,separators=(",",":")).encode()).hexdigest(),
        "rng_stream_seeds":json.dumps(rng_seeds,sort_keys=True),
        "legacy_outcome_state_hash":legacy_outcome_hash,
        "outcome_state_hash":outcome_hash,
        "field_state_hash":field_state_hash,
        "planning_request_trace_hash":planning_request_trace_hash,
        "planning_admission_sequence_hash": planning_sequence_hash,
        "route_replan_outcome_hash":route_replan_outcome_hash,
        "communication_event_hash":communication_event_hash,
        "seed": seed, "agents": agents_n, "update_interval": update_interval,
        "network_regime": regime, "update_policy": policy_name or "Periodic",
        "max_uploads_per_tick": effective_budget or "", "ticks": max_ticks,
        "queue_policy": queue_policy, "query_offset_enabled": False,
        "delivered_orders": sum(a.delivered_orders for a in agents),
        "completed_pickups": sum(a.total_completed_items for a in agents),
        "completed_deliveries": sum(a.delivered_orders for a in agents),
        "first_successful_order_tick": first_successful_order_tick,
        "total_waits": sum(a.wait_steps for a in agents),
        "waiting_per_order": 0.0, "time_to_10_orders":time_to_10_orders,
        "risk_replans": sum(a.risk_replan_count for a in agents),
        "external_risk_triggered_replans": len(external_risk_events),
        "effective_external_risk_replans": len(effective_external_risk_events),
        "external_risk_route_changed_replans": sum(
            event["outcome"] == "route_changed" for event in external_risk_events),
        "risk_self_only_trigger_evaluations": traffic.risk_audit["self_only_trigger_evaluations"],
        "risk_evaluations": traffic.risk_audit["evaluations"],
        "risk_self_contribution_ratio": (
            traffic.risk_audit["self_risk_sum"] / traffic.risk_audit["total_risk_sum"]
            if traffic.risk_audit["total_risk_sum"] else 0.0),
        "mean_field_mass": sum(field_mass_samples) / len(field_mass_samples) if field_mass_samples else 0.0,
        "mean_superseded_contribution_mass": (
            sum(stale_mass_samples) / len(stale_mass_samples) if stale_mass_samples else 0.0),
        "planning_initial_plan_admissions": planning_counts["initial_plan"],
        "planning_new_task_plan_admissions": planning_counts["new_task_plan"],
        "planning_blocked_replan_admissions": planning_counts["blocked_replan"],
        "planning_risk_replan_admissions": planning_counts["risk_replan"],
        **risk_metrics,
        "network_rng_random_calls": pipeline.rng_random_calls,
        "network_rng_randint_calls": pipeline.rng_randint_calls,
        **planning_metrics,
        "planning_admitted_but_never_moved": sum(
            a.agent_id in traffic.planning_admission.admitted_agents
            and a.total_steps == 0 for a in agents),
        "planning_admitted_but_never_completed": sum(
            a.agent_id in traffic.planning_admission.admitted_agents
            and a.delivered_orders == 0 for a in agents),
        "runtime_seconds": runtime, "candidate_uploads": manager.candidate_uploads,
        "selected_uploads": manager.selected_uploads, "field_mass": sum(field_values),
        **stats, "invariants_passed": all(invariants.values()), "completed": True}
    row["waiting_per_order"] = row["total_waits"]/row["delivered_orders"] if row["delivered_orders"] else math.nan
    row["risk_replans_per_order"] = row["risk_replans"]/row["delivered_orders"] if row["delivered_orders"] else math.nan
    row["runtime_per_order"] = _runtime_per_order(
        runtime, row["delivered_orders"])
    row["selected_candidate_ratio"] = manager.selected_uploads/manager.candidate_uploads if manager.candidate_uploads else 0.0
    row["utility_per_transmitted_kbyte"] = row["delivered_orders"]/(stats["transmitted_bytes"]/1024) if stats["transmitted_bytes"] else math.nan
    emit_progress(max_ticks, "completed_simulation")
    return row, invariants, {
        "admissions": planning_history,
        "requests": planning_request_history,
        "route_outcomes": route_outcomes,
        "network_events": pipeline.network_event_history,
    }


def atomic_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".event_", suffix=".tmp")
    with os.fdopen(fd, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
    os.replace(temp, path)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".event_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(value, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp, path)
    except BaseException:
        try:
            os.unlink(temp)
        except FileNotFoundError:
            pass
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", choices=["E5","E6","E7"], required=True)
    p.add_argument("--seed", type=int, required=True); p.add_argument("--agents", type=int, default=100)
    p.add_argument("--update-interval", type=int, default=2)
    p.add_argument("--network-regime", choices=REGIMES, required=True)
    p.add_argument("--update-policy", choices=POLICIES)
    p.add_argument("--max-ticks", type=int, default=3000)
    p.add_argument("--max-uploads-per-tick", type=int, default=25)
    p.add_argument("--queue-policy", choices=["fifo_keep","latest_wins"], default="latest_wins")
    p.add_argument("--method", choices=list(METHOD_CONFIGS), default="proposed")
    p.add_argument("--deterministic-task-trace", action="store_true")
    p.add_argument("--audit-mechanism", choices=["M00", "M10", "M01", "M11"], default="M00")
    p.add_argument("--output-tag", default="", help="optional suffix for isolated diagnostic outputs")
    p.add_argument("--progress-json", type=Path,
                   help="atomic non-semantic progress checkpoint path")
    p.add_argument("--progress-every-ticks", type=int, default=0,
                   help="write progress after every positive multiple of this many ticks")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    if args.experiment == "E7" and not args.update_policy: p.error("E7 requires --update-policy")
    if args.progress_every_ticks < 0: p.error("--progress-every-ticks must be non-negative")
    if args.progress_every_ticks and args.progress_json is None:
        p.error("--progress-every-ticks requires --progress-json")
    method_part = f"_{args.method}" if args.deterministic_task_trace or args.method != "proposed" else ""
    audit_part = "" if args.audit_mechanism == "M00" else f"_{args.audit_mechanism}"
    tag_part = f"_{args.output_tag}" if args.output_tag else ""
    name = f"{args.experiment.lower()}{method_part}{audit_part}{tag_part}_seed{args.seed:03d}_N{args.agents}_U{args.update_interval}_{args.network_regime}_{args.update_policy or 'Periodic'}.csv"
    output = args.out_dir / "runs" / name
    row, invariants, audit_logs = run_case(args.experiment,args.seed,args.agents,args.update_interval,
        args.network_regime,args.update_policy,args.max_ticks,args.max_uploads_per_tick,
        args.queue_policy,args.method,args.deterministic_task_trace,args.audit_mechanism,
        args.progress_json,args.progress_every_ticks)
    atomic_csv(output,row)
    atomic_json(args.out_dir/"invariants"/(output.stem+".json"), invariants)
    atomic_json(args.out_dir/"planning"/(output.stem+".json"), audit_logs["admissions"])
    atomic_json(args.out_dir/"risk"/(output.stem+".json"), {
        "requests": [x for x in audit_logs["requests"] if x["request_type"] == "risk_replan"],
        "outcomes": [x for x in audit_logs["route_outcomes"] if x["request_type"] == "risk_replan"],
    })
    atomic_json(args.out_dir/"events"/(output.stem+".json"), {
        "planning_requests": audit_logs["requests"],
        "route_outcomes": audit_logs["route_outcomes"],
        "network_events": audit_logs["network_events"],
    })
    print(output)
    if not all(invariants.values()): raise SystemExit("invariant failure")


if __name__ == "__main__": main()

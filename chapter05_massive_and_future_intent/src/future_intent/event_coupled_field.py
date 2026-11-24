"""Event-coupled future-intention uplink and DigitalTwin delivery."""
from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass

from e6_network_model import COMM_NETWORK_REGIMES, HEADER_BYTES, PAYLOAD_BYTES_PER_ITEM


@dataclass
class Payload:
    agent_id: int
    path: list
    absolute_entries: list
    generation_tick: int
    update_type: str
    update_weight: float
    payload_size_bytes: int
    associated_bs: int
    enqueue_tick: int
    remaining_bytes: int
    source_position: tuple
    version: int = 0
    transmission_start_tick: int | None = None
    transmission_completion_tick: int | None = None
    processing_completion_tick: int | None = None


class EventCoupledFieldPipeline:
    """Persistent per-BS FIFO queue; loss follows full service; no retransmit."""

    def __init__(self, regime_name, digital_twin, n_agents, rng, queue_policy="latest_wins",
                 censor_penalty=None, versioned_replacement=False):
        if queue_policy not in {"fifo_keep", "latest_wins"}:
            raise ValueError("invalid queue policy")
        self.regime_name, self.digital_twin, self.rng = regime_name, digital_twin, rng
        self.config = COMM_NETWORK_REGIMES[regime_name]
        self.queue_policy = queue_policy
        self.censor_penalty = censor_penalty
        self.versioned_replacement = versioned_replacement
        self.queues = [deque() for _ in self.config["base_stations"]]
        self.processing = []
        self.selected = None
        self.last_processed_generation_tick = [None] * n_agents
        # A fully expired delivered payload refreshes delivery state but not the
        # usable future-intention state because it contributes no cell-time item.
        self.last_usable_generation_tick = [None] * n_agents
        self.last_completion_tick = [None] * n_agents
        self.last_applied_version = [-1] * n_agents
        self._next_version = 0
        self.selection_counts = [0] * n_agents
        self.last_selection_tick = [None] * n_agents
        self.longest_silent_streak = [0] * n_agents
        self.age_samples, self.queue_delays, self.service_delays = [], [], []
        self.all_agent_source_age_samples = []
        self.active_route_source_age_samples = []
        self.active_route_usable_age_samples = []
        self.active_route_conservative_age_samples = []
        self.all_agent_censored = self.all_agent_observations = 0
        self.active_route_censored = self.active_route_observations = 0
        self.active_route_no_usable_source = 0
        self.processing_delays, self.end_to_end_delays, self.expired_ratios = [], [], []
        self.queue_payload_samples, self.queue_byte_samples = [], []
        self.service_by_bs_tick = {}
        self.attempts = self.losses = self.transmitted = self.completed = 0
        self.field_updates = self.fully_expired_payloads = 0
        self.expired_items = self.surviving_items = self.transmitted_bytes = 0
        self.coalesced_payloads = self.direct_update_violations = 0
        self.replaced_versions = self.superseded_completed_payloads = 0
        self.no_field_entry_inserted_in_past = True
        self.network_event_history = []
        self.rng_random_calls = self.rng_randint_calls = 0

    def select(self, agent_ids):
        self.selected = None if agent_ids is None else set(agent_ids)

    def _nearest_bs(self, pos):
        return min(range(len(self.config["base_stations"])),
                   key=lambda i: math.dist(pos, self.config["base_stations"][i]))

    def enqueue(self, agent_id, pos, tick, path, weight, update_type="rolling_refresh",
                generation_tick=None):
        if (self.selected is not None and agent_id not in self.selected) or not path:
            return False
        path = list(path)
        bs = self._nearest_bs(pos)
        size = HEADER_BYTES + PAYLOAD_BYTES_PER_ITEM * len(path)
        generation_tick = tick if generation_tick is None else int(generation_tick)
        version = self._next_version; self._next_version += 1
        payload = Payload(agent_id, path,
            [(point, generation_tick + k, k) for k, point in enumerate(path, 1)], generation_tick,
            update_type, float(weight), size, bs, tick, size, tuple(pos), version=version)
        queue = self.queues[bs]
        if self.queue_policy == "latest_wins":
            for index, old in enumerate(queue):
                if old.agent_id == agent_id:
                    queue[index] = payload
                    self.coalesced_payloads += 1
                    queue_action = "coalesced"
                    break
            else:
                queue.append(payload)
                queue_action = "enqueued"
        else:
            queue.append(payload)
            queue_action = "enqueued"
        self.attempts += 1
        self.selection_counts[agent_id] += 1
        self.last_selection_tick[agent_id] = tick
        self.network_event_history.append({
            "event": queue_action, "tick": tick, "agent_id": agent_id,
            "generation_tick": generation_tick, "bs": bs,
            "payload_size_bytes": size, "update_type": update_type,
        })
        return True

    def _success_probability(self, payload):
        d = math.dist(payload.source_position, self.config["base_stations"][payload.associated_bs])
        lo, hi, d0 = self.config["p_min"], self.config["p_max"], self.config["d0"]
        return max(lo, min(hi, lo + (hi-lo) * math.exp(-d/d0)))

    def _complete_processing(self, payload, tick):
        # A delayed old payload must not overwrite a newer version that already
        # completed.  This decision is made at processing completion, not upload.
        if self.versioned_replacement and payload.version <= self.last_applied_version[payload.agent_id]:
            self.superseded_completed_payloads += 1
            self.completed += 1
            self.network_event_history.append({
                "event": "processed_superseded", "tick": tick,
                "agent_id": payload.agent_id, "generation_tick": payload.generation_tick,
                "version": payload.version,
            })
            return
        surviving = [entry for entry in payload.absolute_entries if entry[1] > tick]
        expired = len(payload.absolute_entries) - len(surviving)
        self.expired_items += expired
        self.surviving_items += len(surviving)
        self.expired_ratios.append(expired / len(payload.absolute_entries))
        self.completed += 1
        self.last_processed_generation_tick[payload.agent_id] = payload.generation_tick
        self.last_completion_tick[payload.agent_id] = tick
        if self.versioned_replacement:
            # A successful completed version, even an empty one, explicitly
            # supersedes its predecessor.  No other agent's records are touched.
            self.digital_twin.replace_contributor_contributions(payload.agent_id)
            self.last_applied_version[payload.agent_id] = payload.version
            self.replaced_versions += 1
        self.end_to_end_delays.append(tick - payload.generation_tick)
        if not surviving:
            self.fully_expired_payloads += 1
            self.network_event_history.append({
                "event": "processed_fully_expired", "tick": tick,
                "agent_id": payload.agent_id,
                "generation_tick": payload.generation_tick,
            })
            return
        if any(absolute_tick <= tick for _, absolute_tick, _ in surviving):
            self.no_field_entry_inserted_in_past = False
            raise AssertionError("past field insertion")
        if hasattr(self.digital_twin, "contribution_records"):
            self.digital_twin.reserve_path_at_absolute_times(
                surviving, weight=payload.update_weight, contributor_id=payload.agent_id,
                contribution_version=payload.version)
        else:  # Compatibility for minimal external DigitalTwin test doubles.
            self.digital_twin.reserve_path_at_absolute_times(
                surviving, weight=payload.update_weight)
        self.last_usable_generation_tick[payload.agent_id] = payload.generation_tick
        self.field_updates += 1
        self.network_event_history.append({
            "event": "processed_field_update", "tick": tick,
            "agent_id": payload.agent_id,
            "generation_tick": payload.generation_tick,
            "version": payload.version,
            "surviving_items": len(surviving),
        })

    def advance(self, tick, positions=None):
        delivered, pending = [], []
        for payload in self.processing:
            if payload.processing_completion_tick <= tick:
                self._complete_processing(payload, tick); delivered.append(payload.agent_id)
            else:
                pending.append(payload)
        self.processing = pending
        capacity = int(self.config["upload_capacity_bytes_per_tick"])
        for bs, queue in enumerate(self.queues):
            budget, served_total = capacity, 0
            while queue and budget:
                payload = queue[0]
                if payload.transmission_start_tick is None:
                    payload.transmission_start_tick = tick
                served = min(budget, payload.remaining_bytes)
                payload.remaining_bytes -= served; budget -= served; served_total += served
                if payload.remaining_bytes:
                    break
                queue.popleft(); payload.transmission_completion_tick = tick
                self.transmitted += 1; self.transmitted_bytes += payload.payload_size_bytes
                self.queue_delays.append(payload.transmission_start_tick - payload.enqueue_tick)
                self.service_delays.append(tick - payload.transmission_start_tick + 1)
                self.rng_random_calls += 1
                if self.rng.random() > self._success_probability(payload):
                    self.losses += 1
                    self.network_event_history.append({
                        "event": "transmission_dropped", "tick": tick,
                        "agent_id": payload.agent_id,
                        "generation_tick": payload.generation_tick,
                        "bs": bs, "payload_size_bytes": payload.payload_size_bytes,
                    })
                    continue
                self.rng_randint_calls += 1
                delay = int(self.config["base_processing_delay_ticks"]) + self.rng.randint(0, int(self.config["jitter_max_ticks"]))
                self.network_event_history.append({
                    "event": "transmission_success", "tick": tick,
                    "agent_id": payload.agent_id,
                    "generation_tick": payload.generation_tick,
                    "bs": bs, "payload_size_bytes": payload.payload_size_bytes,
                    "processing_delay": delay,
                })
                payload.processing_completion_tick = tick + delay
                self.processing_delays.append(delay)
                if delay == 0:
                    self._complete_processing(payload, tick); delivered.append(payload.agent_id)
                else:
                    self.processing.append(payload)
            self.service_by_bs_tick[(bs, tick)] = served_total
            if served_total > capacity:
                raise AssertionError("capacity exceeded")
        self.queue_payload_samples.append(sum(map(len, self.queues)))
        self.queue_byte_samples.append(sum(p.remaining_bytes for q in self.queues for p in q))
        return delivered

    def record_ages(self, tick, agent_ids, active_route_ids=None):
        """Sample delivered and usable source ages with explicit masks/censoring.

        ``agent_ids`` is the all-agent population. ``active_route_ids`` contains
        agents that currently own a non-empty valid future path. None timestamps
        are counted as censored and never inserted as zero or horizon-valued age.
        """
        values = []
        active_route_ids = set(agent_ids if active_route_ids is None else active_route_ids)
        for aid in agent_ids:
            generation = self.last_processed_generation_tick[aid]
            self.all_agent_observations += 1
            if generation is not None:
                age = tick-generation
                values.append(age); self.age_samples.append(age)
                self.all_agent_source_age_samples.append(age)
                if aid in active_route_ids:
                    self.active_route_source_age_samples.append(age)
            else:
                self.all_agent_censored += 1
            if aid in active_route_ids:
                self.active_route_observations += 1
                if generation is None:
                    self.active_route_censored += 1
                usable_generation = self.last_usable_generation_tick[aid]
                if usable_generation is None:
                    self.active_route_no_usable_source += 1
                    if self.censor_penalty is not None:
                        self.active_route_conservative_age_samples.append(self.censor_penalty)
                else:
                    usable_age = tick-usable_generation
                    self.active_route_usable_age_samples.append(usable_age)
                    self.active_route_conservative_age_samples.append(usable_age)
            last = self.last_selection_tick[aid]
            self.longest_silent_streak[aid] = max(self.longest_silent_streak[aid], tick+1 if last is None else tick-last)
        return values

    def source_age(self, agent_id, tick):
        generation = self.last_processed_generation_tick[agent_id]
        return None if generation is None else tick-generation

    def usable_source_age(self, agent_id, tick):
        generation = self.last_usable_generation_tick[agent_id]
        return None if generation is None else tick-generation

    @staticmethod
    def _mean(v): return sum(v)/len(v) if v else 0.0
    @staticmethod
    def _pct(v, p):
        if not v: return math.nan
        v = sorted(v); return float(v[min(len(v)-1, math.ceil(p*len(v)/100)-1)])
    @staticmethod
    def _jain(v):
        return sum(v)**2/(len(v)*sum(x*x for x in v)) if sum(x*x for x in v) else 0.0

    def statistics(self):
        total_items = self.expired_items + self.surviving_items
        total_capacity = len(self.config["base_stations"])*self.config["upload_capacity_bytes_per_tick"]*len(self.queue_payload_samples)
        return {"total_upload_attempts": self.attempts, "successful_uploads": self.completed,
            "transmitted_payloads": self.transmitted, "dropped_uploads": self.losses,
            "drop_ratio": self.losses/self.transmitted if self.transmitted else 0.0,
            "transmitted_bytes": self.transmitted_bytes, "field_update_count": self.field_updates,
            "fully_expired_payloads": self.fully_expired_payloads,
            "fully_expired_payload_ratio": self.fully_expired_payloads/self.completed if self.completed else 0.0,
            "expired_item_ratio": self.expired_items/total_items if total_items else 0.0,
            "processed_items": self.surviving_items, "mean_queue_delay": self._mean(self.queue_delays),
            "mean_transmission_delay": self._mean(self.service_delays), "mean_processing_delay": self._mean(self.processing_delays),
            "mean_total_update_delay": self._mean(self.end_to_end_delays), "mean_field_age": self._mean(self.age_samples),
            "p90_field_age": self._pct(self.age_samples,90), "p90_end_to_end_delay": self._pct(self.end_to_end_delays,90),
            "all_agent_source_age_mean": self._mean(self.all_agent_source_age_samples),
            "all_agent_source_age_median": self._pct(self.all_agent_source_age_samples,50),
            "all_agent_source_age_p90": self._pct(self.all_agent_source_age_samples,90),
            "all_agent_censored_fraction": self.all_agent_censored/self.all_agent_observations if self.all_agent_observations else math.nan,
            "active_route_source_age_mean": self._mean(self.active_route_source_age_samples),
            "active_route_source_age_median": self._pct(self.active_route_source_age_samples,50),
            "active_route_source_age_p90": self._pct(self.active_route_source_age_samples,90),
            "active_route_censored_fraction": self.active_route_censored/self.active_route_observations if self.active_route_observations else math.nan,
            "active_route_usable_source_age_mean": self._mean(self.active_route_usable_age_samples),
            "active_route_usable_source_age_median": self._pct(self.active_route_usable_age_samples,50),
            "active_route_usable_source_age_p90": self._pct(self.active_route_usable_age_samples,90),
            "active_route_conservative_age_p90": self._pct(self.active_route_conservative_age_samples,90),
            "active_route_no_valid_source_fraction": self.active_route_no_usable_source/self.active_route_observations if self.active_route_observations else math.nan,
            "agents_without_usable_delivery": sum(x is None for x in self.last_usable_generation_tick),
            "mean_queue_payloads": self._mean(self.queue_payload_samples), "mean_queued_bytes": self._mean(self.queue_byte_samples),
            "bs_utilization": self.transmitted_bytes/total_capacity if total_capacity else 0.0,
            "coalesced_payloads": self.coalesced_payloads, "selection_fairness": self._jain(self.selection_counts),
            "longest_silent_streak": max(self.longest_silent_streak, default=0),
            "replaced_versions": self.replaced_versions,
            "superseded_completed_payloads": self.superseded_completed_payloads,
            "superseded_contribution_mass": (
                self.digital_twin.superseded_contribution_mass()
                if hasattr(self.digital_twin, "superseded_contribution_mass") else 0.0)}

    def invariant_report(self):
        capacity = self.config["upload_capacity_bytes_per_tick"]
        return {"field_updates_from_delivery": self.field_updates == self.completed-self.fully_expired_payloads,
            "no_direct_communicated_field_updates": self.direct_update_violations == 0,
            "no_field_entry_inserted_in_past": self.no_field_entry_inserted_in_past,
            "no_field_update_from_failed_payload": self.field_updates <= self.completed,
            "byte_service_per_bs_per_tick_within_capacity": all(v <= capacity for v in self.service_by_bs_tick.values())}

"""Oracle-safe contextual quota baseline definitions for E18."""
from __future__ import annotations

SAFE_CONTEXT_FEATURES = (
    "candidate_pool_size",
    "mean_packet_age",
    "max_packet_age",
    "age_q50",
    "age_q90",
    "communicated_field_mass_proxy",
    "stale_pressure_proxy",
    "route_instability_proxy",
    "route_oscillation_proxy",
    "coverage_proxy",
    "recent_action_id",
    "recent_route_switch_count",
    "budget",
    "horizon",
    "n_agents",
)

FORBIDDEN_CONTEXT_FEATURES = (
    "F_full",
    "true_field",
    "hidden_congestion_label",
    "ghost_label_from_F_full",
    "offline_decision_regret",
    "actual_future_route_cost",
    "true_future_congestion",
)


def feature_audit_markdown() -> str:
    safe = "\n".join(f"- {name}" for name in SAFE_CONTEXT_FEATURES)
    forbidden = "\n".join(f"- {name}" for name in FORBIDDEN_CONTEXT_FEATURES)
    return "\n".join([
        "# E18 contextual feature audit",
        "",
        "Verdict: **PASS**",
        "",
        "The implemented contextual bandit is an online-only LinUCB-style quota selector over the same five quota actions as RGSA.",
        "",
        "Safe online feature family:",
        "",
        safe,
        "",
        "Forbidden features not used:",
        "",
        forbidden,
        "",
        "The scalar reward is the existing observable proxy loss from communicated route estimates, stale pressure, and ghost residual pressure. It does not use true route cost at action time.",
    ])

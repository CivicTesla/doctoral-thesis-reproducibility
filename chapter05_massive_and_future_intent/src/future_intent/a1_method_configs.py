"""Single source of truth for A1 method semantics."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MethodConfig:
    name: str
    use_field: bool
    upload_intentions: bool
    enable_risk_replan: bool
    p1_weight: float
    p1_threshold: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


METHOD_CONFIGS = {
    "baseline": MethodConfig(
        name="baseline",
        use_field=False,
        upload_intentions=False,
        enable_risk_replan=False,
        p1_weight=0.0,
        p1_threshold=1.6,
    ),
    "intention_only": MethodConfig(
        name="intention_only",
        use_field=True,
        upload_intentions=True,
        enable_risk_replan=False,
        p1_weight=3.0,
        p1_threshold=1.6,
    ),
    "proposed": MethodConfig(
        name="proposed",
        use_field=True,
        upload_intentions=True,
        enable_risk_replan=True,
        p1_weight=3.0,
        p1_threshold=1.6,
    ),
    "hard_reservation": MethodConfig(
        name="hard_reservation",
        use_field=True,
        upload_intentions=True,
        enable_risk_replan=False,
        p1_weight=0.0,
        p1_threshold=1e-9,
    ),
}


def get_method_config(name: str) -> MethodConfig:
    try:
        return METHOD_CONFIGS[name]
    except KeyError as exc:
        raise ValueError(f"unknown A1 method {name!r}") from exc

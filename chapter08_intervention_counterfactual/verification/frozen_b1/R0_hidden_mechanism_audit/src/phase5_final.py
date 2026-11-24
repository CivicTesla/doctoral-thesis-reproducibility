"""Frozen-model and channel helpers for the Phase 5 final audit."""

from dataclasses import fields as dataclass_fields
import copy
import math

import numpy as np

from .phase5_communication import CommunicationScheme
from .phase5_event import EventScheme
from .simple_models import (
    DecisionTreeModel, LogisticModel, RandomForestModel, TreeNode)


def frozen_scheme(item):
    meta = item["scheme_definition"]
    definition = meta["definition"]
    if meta["category"] == "event":
        allowed = {field.name for field in dataclass_fields(EventScheme)}
        return EventScheme(**{
            key: tuple(value) if key == "fields" else value
            for key, value in definition.items() if key in allowed})
    allowed = {field.name for field in dataclass_fields(CommunicationScheme)}
    return CommunicationScheme(**{
        key: tuple(value) if key == "fields" else value
        for key, value in definition.items() if key in allowed})


def _tree(payload):
    node = TreeNode(float(payload["probability"]))
    if "feature" in payload:
        node.feature = int(payload["feature"])
        node.threshold = float(payload["threshold"])
        node.left = _tree(payload["left"])
        node.right = _tree(payload["right"])
    return node


def predict_frozen(artifact, feature_map):
    """Apply saved scaler/model parameters without fitting or calibration."""
    names = artifact["feature_order"]
    if set(feature_map) != set(names):
        missing = sorted(set(names) - set(feature_map))
        extra = sorted(set(feature_map) - set(names))
        raise ValueError(f"frozen feature schema mismatch: {missing=} {extra=}")
    x = np.asarray([[feature_map[name] for name in names]], float)
    mean = np.asarray(artifact["scaler_mean"], float)
    scale = np.asarray(artifact["scaler_scale"], float)
    z = (x - mean) / scale
    if artifact["model_type"] == "logistic":
        weights = np.asarray(
            artifact["coefficients_intercept_first"], float)
        score = np.clip(
            weights[0] + z @ weights[1:], -30, 30)
        return float(1 / (1 + np.exp(-score[0])))
    probabilities = []
    for tree_payload in artifact["trees"]:
        tree = DecisionTreeModel()
        tree.root = _tree(tree_payload)
        probabilities.append(float(tree.predict_proba(z)[0]))
    return float(np.mean(probabilities))


def fit_frozen_family(artifact, x, y):
    """Refit only the already-frozen family/hyperparameters for LOCO."""
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale[scale == 0] = 1.0
    z = (x - mean) / scale
    parameters = artifact["hyperparameters"]
    if artifact["model_type"] == "logistic":
        model = LogisticModel(l2=parameters["l2"])
    else:
        model = RandomForestModel(
            n_trees=31, max_depth=parameters["max_depth"],
            min_leaf=3, seed=23)
    model.fit(z, y)
    return model, mean, scale


def predict_refit(model, mean, scale, x):
    return model.predict_proba((x - mean) / scale)


def received_age_statistics(transmission):
    age = transmission.get("age")
    if age is not None:
        finite = np.asarray(age)[np.asarray(age) >= 0]
    else:
        mask = transmission.get("arrival_mask")
        finite = np.zeros(int(np.sum(mask))) if mask is not None else np.zeros(0)
    if not len(finite):
        return math.nan, math.nan
    return float(np.mean(finite)), float(np.quantile(finite, .9))


def manifest_hash_payload(manifest):
    """Return immutable fields used to detect post-freeze mutation."""
    return {
        "status": manifest["status"],
        "selection_data": manifest["selection_data"],
        "failure_retry_alias_excluded":
            manifest["failure_retry_alias_excluded"],
        "frozen_schemes": copy.deepcopy(manifest["frozen_schemes"]),
    }

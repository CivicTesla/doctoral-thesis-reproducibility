from __future__ import annotations

import numpy as np


def entropy(field: np.ndarray, eps: float = 1e-12) -> float:
    total = float(np.sum(field))
    if total <= eps:
        return 0.0
    p = field.ravel() / total
    p = p[p > eps]
    return float(-np.sum(p * np.log(p)))


def _argmax3(field: np.ndarray) -> tuple[int, int, int]:
    if field.size == 0:
        return (0, 0, 0)
    return tuple(int(v) for v in np.unravel_index(int(np.argmax(field)), field.shape))


def _topk_indices(field: np.ndarray, k: int) -> set[int]:
    flat = field.ravel()
    if flat.size == 0:
        return set()
    k = min(k, flat.size)
    idx = np.argpartition(flat, -k)[-k:]
    return set(int(i) for i in idx if flat[int(i)] > 0)


def _topk_coords(field: np.ndarray, k: int) -> list[tuple[int, int, int]]:
    flat = field.ravel()
    if flat.size == 0:
        return []
    k = min(k, flat.size)
    idx = np.argpartition(flat, -k)[-k:]
    idx = [int(i) for i in idx if flat[int(i)] > 0]
    idx.sort(key=lambda i: float(flat[i]), reverse=True)
    return [tuple(int(v) for v in np.unravel_index(i, field.shape)) for i in idx]


def _tolerant_peak_overlap(
    R_full: np.ndarray,
    R_comm: np.ndarray,
    top_k: int,
    peak_tolerance_radius: float,
) -> float:
    full_peaks = _topk_coords(R_full, top_k)
    comm_peaks = _topk_coords(R_comm, top_k)
    if not full_peaks:
        return 0.0
    matched = 0
    for ft, fy, fx in full_peaks:
        for ct, cy, cx in comm_peaks:
            dist = float(np.sqrt((ct - ft) ** 2 + (cy - fy) ** 2 + (cx - fx) ** 2))
            if dist <= peak_tolerance_radius:
                matched += 1
                break
    return matched / max(1, min(top_k, len(full_peaks)))


def _center_of_mass_2d(field: np.ndarray, eps: float = 1e-12) -> tuple[float, float]:
    spatial = np.sum(field, axis=0)
    total = float(np.sum(spatial))
    if total <= eps:
        return (0.0, 0.0)
    ys, xs = np.indices(spatial.shape)
    cy = float(np.sum(ys * spatial) / total)
    cx = float(np.sum(xs * spatial) / total)
    return (cy, cx)


def compute_metrics(
    R_full: np.ndarray,
    R_comm: np.ndarray,
    policy: str,
    top_k: int = 20,
    peak_tolerance_radius: float = 3.0,
    active_threshold_ratio: float = 0.05,
) -> dict:
    eps = 1e-12
    diff = R_comm - R_full
    mse = float(np.mean(diff * diff)) if diff.size else 0.0
    mae = float(np.mean(np.abs(diff))) if diff.size else 0.0
    full_sum = float(np.sum(R_full))
    comm_sum = float(np.sum(R_comm))
    normalized_mae = mae / (float(np.mean(R_full)) + eps)
    total_mass_ratio = comm_sum / (full_sum + eps)
    R_full_shape = R_full / (full_sum + eps)
    R_comm_shape = R_comm / (comm_sum + eps)
    shape_diff = R_comm_shape - R_full_shape
    shape_mae = float(np.mean(np.abs(shape_diff))) if shape_diff.size else 0.0
    shape_mse = float(np.mean(shape_diff * shape_diff)) if shape_diff.size else 0.0
    shape_fidelity_score = -shape_mae
    active_threshold = active_threshold_ratio * max(float(np.max(R_full)) if R_full.size else 0.0, eps)
    active_mask = (R_full > active_threshold) | (R_comm > active_threshold)
    if np.any(active_mask):
        active_shape_diff = shape_diff[active_mask]
        active_shape_mae = float(np.mean(np.abs(active_shape_diff)))
        active_shape_mse = float(np.mean(active_shape_diff * active_shape_diff))
    else:
        active_shape_mae = shape_mae
        active_shape_mse = shape_mse
    active_shape_fidelity_score = -active_shape_mae

    peak_full = _argmax3(R_full)
    peak_comm = _argmax3(R_comm)
    peak_full_val = float(R_full[peak_full]) if R_full.size else 0.0
    peak_comm_val = float(R_comm[peak_comm]) if R_comm.size else 0.0
    peak_time_error = abs(peak_comm[0] - peak_full[0])
    peak_location_error = float(np.hypot(peak_comm[1] - peak_full[1], peak_comm[2] - peak_full[2]))
    peak_intensity_error = abs(peak_comm_val - peak_full_val) / (peak_full_val + eps)

    spatial_entropy_full = entropy(np.sum(R_full, axis=0))
    spatial_entropy_comm = entropy(np.sum(R_comm, axis=0))
    entropy_diff = abs(spatial_entropy_comm - spatial_entropy_full)

    max_ref = max(float(np.max(R_full)) if R_full.size else 0.0, eps)
    high = 0.25 * max_ref
    low = 0.10 * max_ref
    hidden_mask = (R_full >= high) & (R_comm <= low)
    ghost_mask = (R_comm >= high) & (R_full <= low)
    high_ref_sum = float(np.sum(R_full[R_full >= high]))
    hidden_congestion = float(np.sum(np.maximum(R_full - R_comm, 0.0)[hidden_mask]) / (high_ref_sum + eps))
    ghost_pressure = float(np.sum(np.maximum(R_comm - R_full, 0.0)[ghost_mask]) / (high_ref_sum + eps))
    hidden_area_ratio = float(np.mean(hidden_mask)) if hidden_mask.size else 0.0
    ghost_area_ratio = float(np.mean(ghost_mask)) if ghost_mask.size else 0.0

    field_fidelity_score = -normalized_mae
    normalized_entropy_diff = entropy_diff / (spatial_entropy_full + eps)
    normalized_peak_location_error = peak_location_error / max(1.0, float(np.hypot(R_full.shape[1], R_full.shape[2])))
    normalized_peak_intensity_error = peak_intensity_error
    field_fidelity_composite = -(
        0.30 * normalized_mae
        + 0.20 * hidden_congestion
        + 0.20 * ghost_pressure
        + 0.10 * normalized_entropy_diff
        + 0.10 * normalized_peak_location_error
        + 0.10 * normalized_peak_intensity_error
    )

    top_full = _topk_indices(R_full, top_k)
    top_comm = _topk_indices(R_comm, top_k)
    top_k_peak_overlap_exact = len(top_full & top_comm) / max(1, min(len(top_full), top_k))
    top_k_peak_overlap_tolerant = _tolerant_peak_overlap(R_full, R_comm, top_k, peak_tolerance_radius)
    ghost_norm = ghost_pressure / (1.0 + ghost_pressure)
    peak_loss = 1.0 - top_k_peak_overlap_tolerant
    entropy_norm = normalized_entropy_diff
    mass_excess = max(total_mass_ratio - 1.0, 0.0)
    balanced_loss_g1 = (
        hidden_congestion
        + ghost_norm
        + 0.25 * entropy_norm
        + 0.25 * peak_loss
        + 0.10 * mass_excess
    )
    balanced_loss_g2 = (
        hidden_congestion
        + 2.0 * ghost_norm
        + 0.25 * entropy_norm
        + 0.25 * peak_loss
        + 0.10 * mass_excess
    )
    balanced_fidelity_score_g1 = -balanced_loss_g1
    balanced_fidelity_score_g2 = -balanced_loss_g2
    com_full_y, com_full_x = _center_of_mass_2d(R_full)
    com_comm_y, com_comm_x = _center_of_mass_2d(R_comm)
    field_displacement = float(np.hypot(com_comm_y - com_full_y, com_comm_x - com_full_x))

    return {
        "policy": policy,
        "mse": mse,
        "mae": mae,
        "normalized_mae": normalized_mae,
        "shape_mae": shape_mae,
        "shape_mse": shape_mse,
        "shape_fidelity_score": shape_fidelity_score,
        "active_shape_mae": active_shape_mae,
        "active_shape_mse": active_shape_mse,
        "active_shape_fidelity_score": active_shape_fidelity_score,
        "active_threshold": active_threshold,
        "field_mass_full": full_sum,
        "field_mass_comm": comm_sum,
        "field_mass_ratio": total_mass_ratio,
        "total_mass_ratio": total_mass_ratio,
        "peak_full_t": peak_full[0],
        "peak_full_y": peak_full[1],
        "peak_full_x": peak_full[2],
        "peak_comm_t": peak_comm[0],
        "peak_comm_y": peak_comm[1],
        "peak_comm_x": peak_comm[2],
        "peak_x_full": peak_full[2],
        "peak_y_full": peak_full[1],
        "peak_x_comm": peak_comm[2],
        "peak_y_comm": peak_comm[1],
        "peak_time_error": peak_time_error,
        "peak_location_error": peak_location_error,
        "peak_space_error": peak_location_error,
        "peak_intensity_error": peak_intensity_error,
        "spatial_entropy_full": spatial_entropy_full,
        "spatial_entropy_comm": spatial_entropy_comm,
        "entropy_full": spatial_entropy_full,
        "entropy_comm": spatial_entropy_comm,
        "entropy_diff": entropy_diff,
        "ghost_pressure": ghost_pressure,
        "hidden_congestion": hidden_congestion,
        "hidden_area_ratio": hidden_area_ratio,
        "ghost_area_ratio": ghost_area_ratio,
        "field_fidelity_score": field_fidelity_score,
        "field_fidelity_composite": field_fidelity_composite,
        "ghost_norm": ghost_norm,
        "peak_loss": peak_loss,
        "entropy_norm": entropy_norm,
        "mass_excess": mass_excess,
        "balanced_loss_g1": balanced_loss_g1,
        "balanced_loss_g2": balanced_loss_g2,
        "balanced_fidelity_score_g1": balanced_fidelity_score_g1,
        "balanced_fidelity_score_g2": balanced_fidelity_score_g2,
        "normalized_entropy_diff": normalized_entropy_diff,
        "normalized_peak_location_error": normalized_peak_location_error,
        "normalized_peak_intensity_error": normalized_peak_intensity_error,
        "top_k_peak_overlap": top_k_peak_overlap_tolerant,
        "top_k_peak_overlap_exact": top_k_peak_overlap_exact,
        "top_k_peak_overlap_tolerant": top_k_peak_overlap_tolerant,
        "center_of_mass_full_y": com_full_y,
        "center_of_mass_full_x": com_full_x,
        "center_of_mass_comm_y": com_comm_y,
        "center_of_mass_comm_x": com_comm_x,
        "field_displacement": field_displacement,
        "threshold_high": high,
        "threshold_low": low,
    }

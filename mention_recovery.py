from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
from alignment_emergence import fixation_arrays, duration_weighted_covariance
from object_recovery import FOVEAL_DISK, FOVEAL_RADIUS, EIGENVALUE_FLOOR, ourmodel_cloud, convex_hull_prediction, mask_metrics
from reflacxloader import ReflacxLoader
from stats_utils import paired_difference_stats
SEED, MIN_FIX, MIN_AXIS_FIX, AXIS_INFO_THRESHOLD = 42, 5, 7, 3.84
WINDOW_START, WINDOW_END = -3.5, -0.5
OUT_DIR = Path("results/03_mention_recovery")
METHODS, METRICS = ["our_model", "foveal_kde", "convex_hull"], ["iou", "dice", "recall", "precision"]
DIAGNOSTICS = ["centroid_inside_gt", "normalized_centroid_distance", "n_inside_gt", "inside_gt_fraction", "n_fix", "T_eff", "scatter_ratio", "orientation_info"]
EXCLUDED_LABELS = {"enlarged cardiac silhouette"}
def cloud_axis_features(fixations: list[Any]) -> dict[str, Any]:
    positions, durations = fixation_arrays(fixations)
    centroid = np.average(positions, axis=0, weights=durations)
    covariance = duration_weighted_covariance(fixations)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), EIGENVALUE_FLOOR)
    scatter_ratio = float(eigenvalues[1] / eigenvalues[0])
    t_eff = float(durations.sum() ** 2 / max(float((durations ** 2).sum()), 1e-9))
    return {"centroid": centroid, "scatter_ratio": scatter_ratio, "T_eff": t_eff, "orientation_info": float(t_eff * ((scatter_ratio - 1.0) ** 2) / max(scatter_ratio, 1e-9))}
def centroid_diagnostics(fixations: list[Any], ellipse: Any, centroid: np.ndarray) -> dict[str, Any]:
    cx, cy, rx, ry = ellipse.radial_coords
    distance = float(np.sqrt(((centroid[0] - cx) / max(rx, 1e-9)) ** 2 + ((centroid[1] - cy) / max(ry, 1e-9)) ** 2))
    n_inside = sum(1 for fix in fixations if ellipse.contains_point(fix.x, fix.y))
    return {"centroid_x": float(centroid[0]), "centroid_y": float(centroid[1]), "centroid_inside_gt": bool(distance <= 1.0), "normalized_centroid_distance": distance, "n_inside_gt": int(n_inside), "inside_gt_fraction": float(n_inside / max(len(fixations), 1))}
def axis_structured(row: dict[str, Any]) -> bool:
    return bool(row["n_fix"] >= MIN_AXIS_FIX and 2.0 <= row["scatter_ratio"] <= 20.0 and row["orientation_info"] >= AXIS_INFO_THRESHOLD)
def evaluate_mention_episode(ep: Any, loader: ReflacxLoader) -> dict[str, Any]:
    img_w, img_h = loader.get_image_dims(ep.patient_id, ep.study_id)
    crop = ep.ellipse.crop(int(img_w), int(img_h))
    target = ep.ellipse.mask(crop)
    features = cloud_axis_features(ep.fixations)
    diagnostics = centroid_diagnostics(ep.fixations, ep.ellipse, features["centroid"])
    method_metrics = {}
    for method, predictor in [("our_model", ourmodel_cloud), ("convex_hull", convex_hull_prediction)]:
        prediction, _info = predictor(ep.fixations, crop)
        method_metrics[method] = mask_metrics(prediction, target)
    x0, y0, x1, y1 = crop
    foveal = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    for fixation in ep.fixations:
        cx, cy = int(round(fixation.x - x0)), int(round(fixation.y - y0))
        px0, px1 = max(0, cx - FOVEAL_RADIUS), min(foveal.shape[1], cx + FOVEAL_RADIUS + 1)
        py0, py1 = max(0, cy - FOVEAL_RADIUS), min(foveal.shape[0], cy + FOVEAL_RADIUS + 1)
        if px0 >= px1 or py0 >= py1: continue
        kx0, ky0 = px0 - (cx - FOVEAL_RADIUS), py0 - (cy - FOVEAL_RADIUS)
        foveal[py0:py1, px0:px1] |= FOVEAL_DISK[ky0:ky0 + py1 - py0, kx0:kx0 + px1 - px0]
    method_metrics["foveal_kde"] = mask_metrics(foveal, target)
    return {"episode_key": f"{ep.patient_id}/{ep.study_id}/{ep.label}/{ep.word_start}-{ep.word_end}", "patient_id": ep.patient_id, "study_id": ep.study_id, "label": ep.label, "phrase": ep.phrase, "word_start": int(ep.word_start), "word_end": int(ep.word_end), "n_fix": len(ep.fixations), "T_eff": features["T_eff"], "scatter_ratio": features["scatter_ratio"], "orientation_info": features["orientation_info"], **diagnostics, "method_metrics": method_metrics}
def mean_value(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return float(np.mean(values)) if values else float("nan")
def summarize_stratum(rows: list[dict[str, Any]], name: str, predicate: Any, rng: np.random.Generator) -> dict[str, Any]:
    selected = [row for row in rows if predicate(row)]
    method_summary = {method: {metric: float(np.mean([row["method_metrics"][method][metric] for row in selected])) if selected else float("nan") for metric in METRICS} for method in METHODS}
    deltas = {}
    for baseline in ["foveal_kde", "convex_hull"]:
        contrast = {}
        for metric in METRICS:
            by_patient = defaultdict(list)
            for row in selected:
                delta = row["method_metrics"]["our_model"][metric] - row["method_metrics"][baseline][metric]
                by_patient[str(row["patient_id"])].append(float(delta))
            contrast[metric] = paired_difference_stats([float(np.mean(v)) for v in by_patient.values()], rng)
        deltas[f"our_model_minus_{baseline}"] = contrast
    diagnostic_summary = {key: mean_value(selected, key) for key in DIAGNOSTICS}
    diagnostic_summary["axis_structured_rate"] = float(np.mean([axis_structured(row) for row in selected])) if selected else float("nan")
    return {"criterion": name, "episodes": len(selected), "patients": len({row["patient_id"] for row in selected}), "diagnostics": diagnostic_summary, "method_summary": method_summary, "deltas": deltas}
def select_examples(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ["episode_key", "patient_id", "study_id", "label", "phrase", "word_start", "word_end", "n_fix", "centroid_x", "centroid_y", "centroid_inside_gt", "normalized_centroid_distance", "n_inside_gt", "inside_gt_fraction", "scatter_ratio", "orientation_info", "method_metrics"]
    def lift(row: dict[str, Any]) -> float: return float(row["method_metrics"]["our_model"]["iou"] - row["method_metrics"]["foveal_kde"]["iou"])
    def pack(row: dict[str, Any], role: str) -> dict[str, Any]: return {k: row[k] for k in fields} | {"role": role, "axis_structured": axis_structured(row)}
    pools = {
        "free_form_not_centered_partial_recovery": [r for r in rows if r.get("label") not in EXCLUDED_LABELS and not r["centroid_inside_gt"] and 1.0 < r["normalized_centroid_distance"] <= 2.0 and r["inside_gt_fraction"] >= 0.25 and lift(r) > 0.0],
        "target_centered_axis_weak": [r for r in rows if r.get("label") not in EXCLUDED_LABELS and r["centroid_inside_gt"] and not axis_structured(r) and r["inside_gt_fraction"] >= 0.50 and lift(r) > 0.0],
        "target_centered_axis_structured": [r for r in rows if r.get("label") not in EXCLUDED_LABELS and r["centroid_inside_gt"] and axis_structured(r) and r["inside_gt_fraction"] >= 0.50 and lift(r) > 0.0],
    }
    for name, pool in pools.items():
        pool.sort(key=lambda r: (lift(r), r["method_metrics"]["our_model"]["iou"], r["inside_gt_fraction"]), reverse=True)
        if not pool: raise AssertionError(f"No Exp3 figure candidates for {name}")
    used, groups = set(), []
    for gi in range(5):
        picked, labels = [], set()
        for role, pool in pools.items():
            choices = [r for r in pool if r["episode_key"] not in used]
            chosen = next((r for r in choices if r.get("label") not in labels), choices[0] if choices else pool[0])
            picked.append(pack(chosen, role)); used.add(chosen["episode_key"]); labels.add(chosen.get("label"))
        groups.append({"figure_id": f"{gi + 1:02d}", "examples": picked})
    return {name: [pack(r, name) for r in pool[:12]] for name, pool in pools.items()} | {"figure_candidates": groups}
def main() -> None:
    rng = np.random.default_rng(SEED)
    loader = ReflacxLoader(); loader.load_jsons()
    episodes = list(loader.iter_mention_window_episodes(min_fix=MIN_FIX, unique_label_only=True, selection_mode="all", start_rel_to_mention_start=WINDOW_START, end_rel_to_mention_start=WINDOW_END))
    rows = [evaluate_mention_episode(ep, loader) for ep in episodes]
    strata = {"free_form": lambda r: True, "target_centered": lambda r: r["centroid_inside_gt"], "target_not_centered": lambda r: not r["centroid_inside_gt"], "axis_structured": axis_structured, "axis_weak": lambda r: not axis_structured(r), "target_centered_axis_structured": lambda r: r["centroid_inside_gt"] and axis_structured(r), "target_centered_axis_weak": lambda r: r["centroid_inside_gt"] and not axis_structured(r), "target_not_centered_axis_structured": lambda r: (not r["centroid_inside_gt"]) and axis_structured(r), "n10_free_form": lambda r: r["n_fix"] >= 10, "n10_target_centered": lambda r: r["n_fix"] >= 10 and r["centroid_inside_gt"], "n10_target_centered_axis_structured": lambda r: r["n_fix"] >= 10 and r["centroid_inside_gt"] and axis_structured(r)}
    summaries = {name: summarize_stratum(rows, name, predicate, rng) for name, predicate in strata.items()}
    table = [{"criterion": s["criterion"], "episodes": s["episodes"], "patients": s["patients"], **s["diagnostics"], "method": method, **s["method_summary"][method]} for s in summaries.values() for method in METHODS]
    result = {"config": {"experiment": "mention_recovery", "window": [WINDOW_START, WINDOW_END], "min_fix_loaded": MIN_FIX, "strata": list(strata), "methods": METHODS, "metrics": METRICS, "axis_structured_rule": {"source": "stable_fixation_geometry", "n_fix_min": MIN_AXIS_FIX, "scatter_ratio_range": [2.0, 20.0], "orientation_info_min": AXIS_INFO_THRESHOLD, "threshold_interpretation": "chi_square_1df_95", "eigenvalue_floor": EIGENVALUE_FLOOR}}, "counts": {"episodes": len(rows), **{name: summary["episodes"] for name, summary in summaries.items()}}, "summaries": summaries, "rows": rows}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "mention_recovery.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUT_DIR / "mention_recovery_table.json").write_text(json.dumps({"rows": table, "summaries": summaries}, indent=2), encoding="utf-8")
    (OUT_DIR / "mention_recovery_examples.json").write_text(json.dumps(select_examples(rows), indent=2), encoding="utf-8")
    print(json.dumps(result["counts"], indent=2))
if __name__ == "__main__":
    main()

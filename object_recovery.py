from __future__ import annotations
import json
from collections import Counter
from math import log
from pathlib import Path
from typing import Any
import zlib
import numpy as np
from reflacxloader import ReflacxLoader
from stats_utils import paired_difference_stats
from scipy.spatial import ConvexHull
from matplotlib.path import Path as MplPath
from alignment_emergence import fixation_arrays, duration_weighted_covariance
RANDOM_SEED = 42
N_FOLDS = 5
MIN_FIXATIONS = 10
OUT_DIR = Path("results/02_object_recovery")
FOVEAL_PPD = 97.0
FOVEAL_RADIUS_PX = 2.0 * FOVEAL_PPD
N_SIGMA = 2.0
EIGENVALUE_FLOOR = 841.0
OUR_MODEL = {"g_parallel": 1.0, "g_perp": 0.4, "rho_gate": 2.0}
G_PARALLEL = [0.70, 0.80, 0.90, 1.00]
G_PERP = [0.30, 0.40, 0.50, 0.60, 0.70]
TUNING_SAMPLE_PIXELS = 1000
METHODS = ["our_model", "foveal_kde", "convex_hull"]
METRICS = ["iou", "dice", "recall", "precision"]
FOVEAL_RADIUS = int(np.ceil(FOVEAL_RADIUS_PX))
_yy, _xx = np.mgrid[-FOVEAL_RADIUS:FOVEAL_RADIUS + 1, -FOVEAL_RADIUS:FOVEAL_RADIUS + 1]
FOVEAL_DISK = (_xx**2 + _yy**2) <= FOVEAL_RADIUS_PX**2
def ourmodel_cloud(fixations: list[Any], crop: tuple[int, int, int, int], model: dict[str, float] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    cfg = model or OUR_MODEL
    positions, durations = fixation_arrays(fixations)
    centroid = np.average(positions, axis=0, weights=durations)
    fixation_covariance = duration_weighted_covariance(fixations)
    eigenvals, eigenvecs = np.linalg.eigh(fixation_covariance)
    eigenvals = np.maximum(eigenvals, EIGENVALUE_FLOOR)
    minor_variance, major_variance = float(eigenvals[0]), float(eigenvals[1])
    scatter_ratio = major_variance / minor_variance
    support_minor = max(minor_variance / cfg["g_perp"] ** 2, EIGENVALUE_FLOOR)
    support_major = max(major_variance / cfg["g_parallel"] ** 2, EIGENVALUE_FLOOR)
    use_orientation = bool(scatter_ratio >= cfg["rho_gate"] and support_major > support_minor)
    if use_orientation:
        support_covariance = eigenvecs @ np.diag([support_minor, support_major]) @ eigenvecs.T
    else:
        support_iso = float(np.sqrt(support_minor * support_major))
        support_minor = support_iso
        support_major = support_iso
        support_covariance = np.eye(2, dtype=np.float64) * support_iso
    support_covariance = 0.5 * (support_covariance + support_covariance.T)
    x0, y0, x1, y1 = crop
    yy, xx = np.mgrid[y0:y1, x0:x1]
    delta = np.stack([xx + 0.5 - centroid[0], yy + 0.5 - centroid[1]], axis=-1)
    mahalanobis2 = np.einsum("...i,ij,...j->...", delta, np.linalg.pinv(support_covariance), delta)
    mask = mahalanobis2 <= N_SIGMA**2
    info = {"centroid_x": float(centroid[0]), "centroid_y": float(centroid[1]), "scatter_ratio": float(scatter_ratio), "used_orientation": use_orientation, "g_parallel": float(cfg["g_parallel"]), "g_perp": float(cfg["g_perp"]), "rho_gate": float(cfg["rho_gate"]), "support_major_variance": float(support_major), "support_minor_variance": float(support_minor), "equivalent_area_multiplier": float(1.0 / (cfg["g_parallel"] * cfg["g_perp"])), "equivalent_aspect_onset": float(2.0 * log(cfg["g_parallel"] / cfg["g_perp"])), "support_area_px": float(mask.sum())}
    return mask, info
def fixation_heatmap_cloud(fixations: list[Any], crop: tuple[int, int, int, int]) -> tuple[np.ndarray, dict[str, Any]]:
    x0, y0, x1, y1 = crop
    mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    for fixation in fixations:
        cx, cy = int(round(fixation.x - x0)), int(round(fixation.y - y0))
        px0, px1 = max(0, cx - FOVEAL_RADIUS), min(mask.shape[1], cx + FOVEAL_RADIUS + 1)
        py0, py1 = max(0, cy - FOVEAL_RADIUS), min(mask.shape[0], cy + FOVEAL_RADIUS + 1)
        kx0, ky0 = px0 - (cx - FOVEAL_RADIUS), py0 - (cy - FOVEAL_RADIUS)
        mask[py0:py1, px0:px1] |= FOVEAL_DISK[ky0:ky0 + py1 - py0, kx0:kx0 + px1 - px0]
    return mask, {"foveal_radius_px": float(FOVEAL_RADIUS_PX), "foveal_area_px": float(mask.sum())}
def convex_hull_prediction(fixations: list[Any], crop: tuple[int, int, int, int]) -> tuple[np.ndarray, dict[str, Any]]:
    x0, y0, x1, y1 = crop
    height, width = y1 - y0, x1 - x0
    points = np.unique(np.array([[f.x - x0, f.y - y0] for f in fixations], dtype=np.float64), axis=0)
    hull = ConvexHull(points)
    polygon = points[hull.vertices]
    yy, xx = np.mgrid[0:height, 0:width]
    pixel_centers = np.column_stack([xx.ravel() + 0.5, yy.ravel() + 0.5])
    mask = MplPath(polygon).contains_points(pixel_centers).reshape(height, width)
    return mask, {"n_hull_points": int(len(points)), "hull_area_px": float(mask.sum())}
def mask_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    intersection = float(np.logical_and(prediction, target).sum())
    union = float(np.logical_or(prediction, target).sum())
    pred_area = float(prediction.sum())
    target_area = float(target.sum())
    dice_denominator = pred_area + target_area
    return {"iou": intersection / union if union else 0.0, "dice": 2.0 * intersection / dice_denominator if dice_denominator else 0.0, "recall": intersection / target_area if target_area else 0.0, "precision": intersection / pred_area if pred_area else 0.0}
def candidate_grid() -> list[dict[str, float]]:
    return [{"g_parallel": gp, "g_perp": gt, "rho_gate": OUR_MODEL["rho_gate"]} for gp in G_PARALLEL for gt in G_PERP if gt <= gp]
def build_episodes(loader: ReflacxLoader) -> list[dict[str, Any]]:
    episodes = []
    for patient_id, study_id in loader.study_keys():
        img_w, img_h = loader.get_image_dims(patient_id, study_id)
        if img_w is None or img_h is None: continue
        fixations = loader.get_fixations(patient_id, study_id, period="all")
        for annotation_index, ellipse in enumerate(loader.get_ellipses(patient_id, study_id)):
            inside = [fix for fix in fixations if ellipse.contains_point(fix.x, fix.y)]
            if len(inside) < MIN_FIXATIONS: continue
            crop = ellipse.crop(img_w, img_h)
            positions, durations = fixation_arrays(inside)
            centroid = np.average(positions, axis=0, weights=durations)
            eigenvals, eigenvecs = np.linalg.eigh(duration_weighted_covariance(inside))
            eigenvals = np.maximum(eigenvals, EIGENVALUE_FLOOR)
            episodes.append({"patient_id": patient_id, "study_id": study_id, "episode_key": f"{patient_id}/{study_id}/{annotation_index}", "annotation_index": int(annotation_index), "label": ellipse.labels[0] if ellipse.labels else None, "n_fix": int(len(inside)), "img_w": int(img_w), "img_h": int(img_h), "crop": crop, "ellipse_coords": [float(v) for v in ellipse.coords], "target": ellipse.mask(crop), "fixations": inside, "centroid": centroid, "eigenvals": eigenvals, "eigenvecs": eigenvecs, "scatter_ratio": float(eigenvals[1] / eigenvals[0])})
    return episodes
def evaluate_episode(episode: dict[str, Any], model: dict[str, float] | None = None, methods: list[str] | None = None, fold: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    methods = methods or METHODS
    base = {key: episode[key] for key in ["patient_id", "study_id", "episode_key", "annotation_index", "label", "n_fix"]}
    if fold is not None: base["fold"] = int(fold)
    rows = []
    for method in methods:
        if method == "our_model":
            prediction, info = ourmodel_cloud(episode["fixations"], episode["crop"], model)
        elif method == "foveal_kde":
            prediction, info = fixation_heatmap_cloud(episode["fixations"], episode["crop"])
        elif method == "convex_hull":
            prediction, info = convex_hull_prediction(episode["fixations"], episode["crop"])
        else:
            raise ValueError(f"Unknown method: {method}")
        rows.append({**base, "method": method, **mask_metrics(prediction, episode["target"]), **info})
    diagnostic = {**base, "img_w": episode["img_w"], "img_h": episode["img_h"], "crop": [int(v) for v in episode["crop"]], "ellipse_coords": episode["ellipse_coords"], "target_area_px": float(episode["target"].sum()), "method_metrics": {row["method"]: {k: row[k] for k in METRICS} for row in rows}}
    return rows, diagnostic
def make_patient_folds(episodes: list[dict[str, Any]], rng: np.random.Generator) -> list[dict[str, Any]]:
    patients = np.array(sorted({str(ep["patient_id"]) for ep in episodes}), dtype=object)
    rng.shuffle(patients)
    folds, seen, all_patients = [], set(), set(map(str, patients))
    for fold_id, test_array in enumerate(np.array_split(patients, N_FOLDS)):
        test_patients = set(map(str, test_array.tolist()))
        train_patients = all_patients - test_patients
        if train_patients & test_patients: raise AssertionError("Patient leakage between train and test")
        seen |= test_patients
        train_episodes = [ep for ep in episodes if str(ep["patient_id"]) in train_patients]
        test_episodes = [ep for ep in episodes if str(ep["patient_id"]) in test_patients]
        folds.append({"fold": int(fold_id), "train_patients": train_patients, "test_patients": test_patients, "train_episode_count": len(train_episodes), "test_episode_count": len(test_episodes)})
    if seen != all_patients: raise AssertionError("Every patient must appear in exactly one test fold")
    return folds
def patient_mean_metric(rows: list[dict[str, Any]], method: str, metric: str) -> float:
    by_patient: dict[str, list[float]] = {}
    for row in rows:
        if row["method"] == method:
            by_patient.setdefault(str(row["patient_id"]), []).append(float(row[metric]))
    return float(np.mean([np.mean(values) for values in by_patient.values()])) if by_patient else 0.0
def sample_episode_pixels(episode: dict[str, Any]) -> dict[str, np.ndarray]:
    x0, y0, x1, y1 = episode["crop"]
    width, height = x1 - x0, y1 - y0
    n_pixels = int(width * height)
    n_sample = int(min(TUNING_SAMPLE_PIXELS, n_pixels))
    seed = (zlib.crc32(str(episode["episode_key"]).encode("utf-8")) + RANDOM_SEED) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    flat = rng.integers(0, n_pixels, size=n_sample, endpoint=False)
    xs = (flat % width + x0).astype(np.float64) + 0.5
    ys = (flat // width + y0).astype(np.float64) + 0.5
    return {"xs": xs, "ys": ys, "target": episode["target"].ravel()[flat]}
def sampled_ourmodel_metrics(episode: dict[str, Any], model: dict[str, float], sample: dict[str, np.ndarray]) -> dict[str, float]:
    minor_variance, major_variance = float(episode["eigenvals"][0]), float(episode["eigenvals"][1])
    support_minor = max(minor_variance / model["g_perp"] ** 2, EIGENVALUE_FLOOR)
    support_major = max(major_variance / model["g_parallel"] ** 2, EIGENVALUE_FLOOR)
    if float(episode["scatter_ratio"]) >= model["rho_gate"] and support_major > support_minor:
        support_covariance = episode["eigenvecs"] @ np.diag([support_minor, support_major]) @ episode["eigenvecs"].T
    else:
        support_covariance = np.eye(2, dtype=np.float64) * float(np.sqrt(support_minor * support_major))
    delta = np.stack([sample["xs"] - episode["centroid"][0], sample["ys"] - episode["centroid"][1]], axis=-1)
    pred = np.einsum("...i,ij,...j->...", delta, np.linalg.pinv(support_covariance), delta) <= N_SIGMA**2
    target = sample["target"]
    intersection = float(np.logical_and(pred, target).mean())
    pred_area = float(pred.mean())
    target_area = float(target.mean())
    union = pred_area + target_area - intersection
    dice_denominator = pred_area + target_area
    return {"iou": intersection / union if union else 0.0, "dice": 2.0 * intersection / dice_denominator if dice_denominator else 0.0, "recall": intersection / target_area if target_area else 0.0, "precision": intersection / pred_area if pred_area else 0.0}
def select_candidate(train_episodes: list[dict[str, Any]], candidates: list[dict[str, float]]) -> tuple[dict[str, float], list[dict[str, float]]]:
    samples = {episode["episode_key"]: sample_episode_pixels(episode) for episode in train_episodes}
    records = []
    for candidate in candidates:
        rows = []
        for episode in train_episodes:
            rows.append({"patient_id": episode["patient_id"], "method": "our_model", **sampled_ourmodel_metrics(episode, candidate, samples[episode["episode_key"]])})
        record = {**candidate, **{f"train_patient_mean_{metric}": patient_mean_metric(rows, "our_model", metric) for metric in METRICS}}
        records.append(record)
    records.sort(key=lambda r: (-r["train_patient_mean_iou"], -r["train_patient_mean_dice"], -r["g_perp"], -r["g_parallel"], abs(r["g_parallel"] - r["g_perp"])))
    return {key: records[0][key] for key in ["g_parallel", "g_perp", "rho_gate"]}, records
def summarize_results(rows: list[dict[str, Any]], rng: np.random.Generator) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        summary.append({"method": method, "episodes": len(selected), "patients": len({row["patient_id"] for row in selected}), **{metric: float(np.mean([row[metric] for row in selected])) for metric in METRICS}})
    by_key = {(row["episode_key"], row["method"]): row for row in rows}
    contrasts = {}
    for baseline in ["foveal_kde", "convex_hull"]:
        contrasts[f"our_model_minus_{baseline}"] = {}
        for metric in METRICS:
            by_patient: dict[str, list[float]] = {}
            for episode_key, method in list(by_key):
                if method != "our_model" or (episode_key, baseline) not in by_key: continue
                ours, base = by_key[(episode_key, "our_model")], by_key[(episode_key, baseline)]
                by_patient.setdefault(str(ours["patient_id"]), []).append(float(ours[metric]) - float(base[metric]))
            contrasts[f"our_model_minus_{baseline}"][metric] = paired_difference_stats([float(np.mean(values)) for values in by_patient.values()], rng)
    return summary, contrasts
def fold_summary_row(fold: dict[str, Any], selected: dict[str, float], train_records: list[dict[str, float]], heldout_rows: list[dict[str, Any]]) -> dict[str, Any]:
    chosen = next(record for record in train_records if record["g_parallel"] == selected["g_parallel"] and record["g_perp"] == selected["g_perp"])
    row = {"fold": fold["fold"], "train_patients": len(fold["train_patients"]), "test_patients": len(fold["test_patients"]), "train_episodes": fold["train_episode_count"], "test_episodes": fold["test_episode_count"], "g_parallel": float(selected["g_parallel"]), "g_perp": float(selected["g_perp"]), "rho_gate": float(selected["rho_gate"]), "train_patient_mean_iou": float(chosen["train_patient_mean_iou"]), "train_patient_mean_dice": float(chosen["train_patient_mean_dice"])}
    for method in METHODS:
        selected_rows = [r for r in heldout_rows if r["method"] == method]
        for metric in ["iou", "dice"]:
            row[f"heldout_{method}_{metric}"] = float(np.mean([r[metric] for r in selected_rows]))
    return row
def modal_selected_model(fold_table: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter((float(row["g_parallel"]), float(row["g_perp"])) for row in fold_table)
    (g_parallel, g_perp), frequency = counts.most_common(1)[0]
    return {"g_parallel": g_parallel, "g_perp": g_perp, "rho_gate": OUR_MODEL["rho_gate"], "frequency": int(frequency), "folds": len(fold_table), "matches_our_model": bool(g_parallel == OUR_MODEL["g_parallel"] and g_perp == OUR_MODEL["g_perp"])}
def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    loader = ReflacxLoader(); loader.load_jsons()
    episodes = build_episodes(loader)
    folds, candidates = make_patient_folds(episodes, rng), candidate_grid()
    rows, diagnostics, fold_table, fold_summaries = [], [], [], []
    for fold in folds:
        train_episodes = [ep for ep in episodes if str(ep["patient_id"]) in fold["train_patients"]]
        test_episodes = [ep for ep in episodes if str(ep["patient_id"]) in fold["test_patients"]]
        selected, train_records = select_candidate(train_episodes, candidates)
        heldout_rows = []
        for episode in test_episodes:
            episode_rows, diagnostic = evaluate_episode(episode, selected, METHODS, fold["fold"])
            heldout_rows.extend(episode_rows); diagnostics.append(diagnostic)
        rows.extend(heldout_rows)
        fold_table.append(fold_summary_row(fold, selected, train_records, heldout_rows))
        fold_summaries.append({"fold": fold["fold"], "train_patients": len(fold["train_patients"]), "test_patients": len(fold["test_patients"]), "train_episodes": fold["train_episode_count"], "test_episodes": fold["test_episode_count"]})
    expected = {ep["episode_key"] for ep in episodes}
    for method in METHODS:
        keys = {row["episode_key"] for row in rows if row["method"] == method}
        if keys != expected: raise AssertionError(f"Held-out rows for {method} do not cover every episode exactly once")
    summary, contrasts = summarize_results(rows, rng)
    selected_model = modal_selected_model(fold_table)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {"config": {"seed": RANDOM_SEED, "n_folds": N_FOLDS, "min_fixations": MIN_FIXATIONS, "n_sigma": N_SIGMA, "our_model": OUR_MODEL, "candidate_grid": candidates, "selection_rule": "max sampled train patient-mean IoU, then Dice, larger g_perp, larger g_parallel, smaller axis gap", "training_selection_uses_sampled_pixels": True, "tuning_sample_pixels_per_episode": TUNING_SAMPLE_PIXELS, "heldout_metrics_are_exact": True, "foveal_radius_px": FOVEAL_RADIUS_PX}, "counts": {"eligible_episodes": len(episodes), "patients": len({ep["patient_id"] for ep in episodes}), "heldout_rows": len(rows)}, "folds": fold_summaries, "fold_table": fold_table, "selected_model": selected_model, "summary": summary, "contrasts": contrasts, "rows": rows}
    (OUT_DIR / "object_recovery.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUT_DIR / "object_recovery_table.json").write_text(json.dumps({"summary": summary, "contrasts": contrasts, "fold_table": fold_table, "selected_model": selected_model}, indent=2), encoding="utf-8")
    (OUT_DIR / "object_recovery_examples.json").write_text(json.dumps({"examples": sorted(diagnostics, key=lambda d: d["method_metrics"]["our_model"]["iou"], reverse=True)[:12]}, indent=2), encoding="utf-8")
if __name__ == "__main__":
    main()

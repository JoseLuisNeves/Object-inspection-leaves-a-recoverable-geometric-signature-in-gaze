from __future__ import annotations
import json, math
from typing import Any
from pathlib import Path
import numpy as np
from local_annotations import EllipseAnnotation
from reflacxloader import ReflacxLoader
from stats_utils import paired_difference_stats
PHASES, MIN_FIX, MIN_ACCEPTED_PLACEBOS, N_EXAMPLE_FIGURES, EXAMPLES_PER_FIGURE, MIN_EXAMPLE_ELLIPSE_ASPECT, SEED, EPS = ("acquisition", "extent"), 10, 5, 5, 3, 2.0, 42, 1e-9
METRICS = ["alignment_score_cos2", "alignment_error_deg", "main_axis_coverage", "minor_axis_coverage", "axis_preference"]
OUT_DIR = Path("results/01_alignment_emergence")
def fixation_arrays(fixations: list[Any]) -> tuple[np.ndarray, np.ndarray]: # returns position and duration arrays
    return (np.array([[fix.x, fix.y] for fix in fixations], dtype=np.float64), np.array([fix.duration for fix in fixations], dtype=np.float64))
def duration_weighted_covariance(fixations: list[Any]) -> np.ndarray:
    positions, durations = fixation_arrays(fixations)
    centroid = np.average(positions, axis=0, weights=durations)
    delta = positions - centroid
    cov = (durations[:, None, None] * delta[:, :, None] * delta[:, None, :]).sum(axis=0) / durations.sum()
    return 0.5 * (cov + cov.T)
def object_frame(ellipse: Any) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in ellipse.coords]
    w, h = x2 - x1, y2 - y1
    major = w >= h
    u_major = np.array([1.0, 0.0], dtype=np.float64) if major else np.array([0.0, 1.0], dtype=np.float64)
    u_minor = np.array([0.0, 1.0], dtype=np.float64) if major else np.array([1.0, 0.0], dtype=np.float64)
    return u_major, u_minor, max(w, h) / 2.0, min(w, h) / 2.0, max(w, h) / max(min(w, h), EPS)
def fixations_inside(ellipse: Any, fixations: list[Any], positions: np.ndarray) -> list[Any]:
    cx, cy, rx, ry = ellipse.radial_coords
    mask = ((positions[:, 0] - cx) / rx) ** 2 + ((positions[:, 1] - cy) / ry) ** 2 <= 1.0
    return [fix for fix, keep in zip(fixations, mask) if bool(keep)]
def phase_fixation_set(fixations: list[Any], phase: str, SPLIT_RANK: int = 3) -> list[Any]: #Split the inside-object fixations into the two Exp1 phases
    ordered = sorted(fixations, key=lambda fix: fix.start_time)
    if phase == "acquisition": return ordered[:SPLIT_RANK]
    if phase == "extent": return ordered[SPLIT_RANK:]
def fixation_dicts(fixations: list[Any]) -> list[dict[str, float]]:
    return [{"x": float(f.x), "y": float(f.y), "start_time": float(f.start_time), "end_time": float(f.end_time), "duration": float(f.duration)} for f in fixations]
def alignment_row(*, patient_id: str, study_id: str, episode_key: str, annotation_index: int, label: str | None, source: str, phase: str, ellipse: Any, fixations: list[Any]) -> dict[str, Any]:
    selected = phase_fixation_set(fixations, phase)
    cov = duration_weighted_covariance(selected)
    ev, vecs = np.linalg.eigh(cov)
    major_axis, minor_axis, r_major, r_minor, aspect = object_frame(ellipse)
    cloud_major_axis = vecs[:, -1]
    alignment_cos = float(np.clip(abs(float(cloud_major_axis @ major_axis)), 0.0, 1.0))
    alignment_rad = float(math.acos(alignment_cos))
    main_cov = float(major_axis @ cov @ major_axis / max(r_major ** 2, EPS))
    minor_cov = float(minor_axis @ cov @ minor_axis / max(r_minor ** 2, EPS))
    return {"patient_id": patient_id, "study_id": study_id, "episode_key": episode_key, "annotation_index": int(annotation_index), "label": label, "source": source, "phase": phase, "n_fix": int(len(selected)), "alignment_score_cos2": float(math.cos(2.0 * alignment_rad)), "alignment_error_deg": float(math.degrees(alignment_rad)), "main_axis_coverage": main_cov, "minor_axis_coverage": minor_cov, "axis_preference": float(main_cov - minor_cov), "axis_preference_log": float(math.log((main_cov + EPS) / (minor_cov + EPS))), "ellipse_aspect": float(aspect)}
def ellipse_bbox_iou(first: Any, second: Any) -> float:
    ax1, ay1, ax2, ay2 = first.coords
    bx1, by1, bx2, by2 = second.coords
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / max(union, 1e-9))
def sample_placebo_ellipses(*, true_ellipse: Any, fixations: list[Any], positions: np.ndarray, img_w: int, img_h: int, rng: np.random.Generator,) -> tuple[list[tuple[Any, list[Any]]], int]:
    MIN_FIX, N_PLACEBOS, MAX_ATTEMPTS, PLACEBO_BBOX_IOU_MAX = 7, 25, 5000, 0.05
    _cx, _cy, rx, ry = true_ellipse.radial_coords
    accepted, attempts = [], 0
    while len(accepted) < N_PLACEBOS and attempts < MAX_ATTEMPTS:
        attempts += 1
        cx, cy = float(rng.uniform(rx, img_w - rx)), float(rng.uniform(ry, img_h - ry))
        candidate = EllipseAnnotation(coords=(cx - rx, cy - ry, cx + rx, cy + ry), labels=["placebo"])
        if ellipse_bbox_iou(candidate, true_ellipse) > PLACEBO_BBOX_IOU_MAX: continue
        inside = fixations_inside(candidate, fixations, positions)
        if len(inside) >= MIN_FIX: accepted.append((candidate, inside))
    return accepted, attempts
def average_placebo_row(*, patient_id: str, study_id: str, episode_key: str, annotation_index: int, label: str | None, phase: str, placebos: list[tuple[Any, list[Any]]]) -> dict[str, Any]:
    rows = [alignment_row(patient_id=patient_id, study_id=study_id, episode_key=episode_key, annotation_index=annotation_index, label=label, source="random_placebo", phase=phase, ellipse=ellipse, fixations=fixations) for ellipse, fixations in placebos]
    out = dict(rows[0])
    for key in METRICS + ["axis_preference_log", "ellipse_aspect"]:
        out[key] = float(np.mean([r[key] for r in rows]))
    out["n_fix"] = float(np.mean([r["n_fix"] for r in rows]))
    out["n_valid_placebos"] = int(len(rows))
    out["n_placebos"] = int(len(placebos))
    return out
def rows_by_episode_phase(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row["episode_key"]), str(row["phase"])): row for row in rows}
def paired_delta_rows(contrast: str, left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]], metric: str, *, key_phase: bool = True) -> list[dict[str, Any]]:
    key = (lambda r: (str(r["episode_key"]), str(r["phase"]))) if key_phase else (lambda r: str(r["episode_key"]))
    left_by_key, right_by_key = {key(r): r for r in left_rows}, {key(r): r for r in right_rows}
    out = []
    for k in sorted(set(left_by_key) & set(right_by_key)):
        left, right = left_by_key[k], right_by_key[k]
        phase = k[1] if key_phase else "extent_minus_acquisition"
        left_value, right_value = float(left[metric]), float(right[metric])
        out.append({"contrast": contrast, "metric": metric, "patient_id": right["patient_id"], "study_id": right["study_id"], "episode_key": right["episode_key"], "phase": phase, "left_source": left["source"], "right_source": right["source"], "left_value": left_value, "right_value": right_value, "delta": right_value - left_value})
    return out
def interaction_delta_rows(true_rows: list[dict[str, Any]], placebo_rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    true_by_key, placebo_by_key = rows_by_episode_phase(true_rows), rows_by_episode_phase(placebo_rows)
    episode_keys = {episode_key for episode_key, _phase in true_by_key} & {episode_key for episode_key, _phase in placebo_by_key}
    out = []
    for episode_key in sorted(episode_keys):
        required = [(episode_key, "acquisition"), (episode_key, "extent")]
        if any(key not in true_by_key or key not in placebo_by_key for key in required): continue
        true_acq, true_extent, placebo_acq, placebo_extent = true_by_key[(episode_key, "acquisition")], true_by_key[(episode_key, "extent")], placebo_by_key[(episode_key, "acquisition")], placebo_by_key[(episode_key, "extent")]
        values = [float(true_acq[metric]), float(true_extent[metric]), float(placebo_acq[metric]), float(placebo_extent[metric])]
        acquisition_specificity = values[0] - values[2]
        extent_specificity = values[1] - values[3]
        out.append({"contrast": "object_specificity_gain", "metric": metric, "patient_id": true_extent["patient_id"], "study_id": true_extent["study_id"], "episode_key": episode_key, "phase": "interaction", "acquisition_specificity": acquisition_specificity, "extent_specificity": extent_specificity, "delta": extent_specificity - acquisition_specificity})
    return out
def build_contrast_rows(true_rows: list[dict[str, Any]], placebo_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    true_acq = [r for r in true_rows if r["phase"] == "acquisition"]
    true_extent = [r for r in true_rows if r["phase"] == "extent"]
    placebo_acq = [r for r in placebo_rows if r["phase"] == "acquisition"]
    placebo_extent = [r for r in placebo_rows if r["phase"] == "extent"]
    out = []
    for metric in METRICS:
        out.extend(paired_delta_rows("temporal_emergence", true_acq, true_extent, metric, key_phase=False))
        out.extend(paired_delta_rows("extent_object_specificity", placebo_extent, true_extent, metric))
        out.extend(paired_delta_rows("acquisition_object_specificity", placebo_acq, true_acq, metric))
        out.extend(interaction_delta_rows(true_rows, placebo_rows, metric))
    return out
def patient_aggregated_stats(delta_rows: list[dict[str, Any]], contrast: str, metric: str, rng: np.random.Generator) -> dict[str, Any]:
    selected = [row for row in delta_rows if row["contrast"] == contrast and row["metric"] == metric]
    by_patient: dict[str, list[float]] = {}
    for row in selected:
        value = float(row["delta"])
        by_patient.setdefault(str(row["patient_id"]), []).append(value)
    patient_values = [float(np.mean(values)) for values in by_patient.values()]
    stats = paired_difference_stats(patient_values, rng, higher_is_better=(metric != "alignment_error_deg"))
    stats.update({"contrast": contrast,"metric": metric, "episodes": int(len(selected)), "patients": int(len(patient_values))})
    return stats
def summarize_contrasts(delta_rows: list[dict[str, Any]], rng: np.random.Generator) -> list[dict[str, Any]]:
    contrasts = ["temporal_emergence", "extent_object_specificity", "acquisition_object_specificity", "object_specificity_gain"]
    out = []
    for contrast in contrasts:
        for metric in METRICS:
            out.append(patient_aggregated_stats(delta_rows, contrast, metric, rng))
    return out
def phase_summary_rows(true_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for phase in PHASES:
        rows = [row for row in true_rows if row["phase"] == phase]
        out.append({"phase": phase, "episodes": len(rows), "patients": len({r["patient_id"] for r in rows}), **{metric: float(np.mean([r[metric] for r in rows])) for metric in METRICS}, "axis_preference_log": float(np.mean([r["axis_preference_log"] for r in rows])), "ellipse_aspect": float(np.mean([r["ellipse_aspect"] for r in rows]))})
    return out
def select_example_cases(diagnostics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ["episode_key", "patient_id", "study_id", "image_id", "dicom_id", "annotation_index", "label", "img_w", "img_h", "ellipse_coords", "phase_fixations", "true_phase_rows"]
    scored = []
    for d in diagnostics:
        if not d.get("placebo_retained") or "placebo_phase_rows" not in d: continue
        x1, y1, x2, y2 = d["ellipse_coords"]; aspect = max(x2 - x1, y2 - y1) / max(min(x2 - x1, y2 - y1), 1e-9)
        if aspect < MIN_EXAMPLE_ELLIPSE_ASPECT: continue
        tr, pr = d["true_phase_rows"], d["placebo_phase_rows"]
        ta, te = tr["acquisition"]["alignment_score_cos2"], tr["extent"]["alignment_score_cos2"]
        pa, pe = pr["acquisition"]["alignment_score_cos2"], pr["extent"]["alignment_score_cos2"]
        case = {k: d[k] for k in keys}
        case["ellipse_aspect"] = float(aspect)
        case["example_score"] = float(max(0.0, te - ta) + max(0.0, te - pe) + 0.5 * max(0.0, (te - pe) - (ta - pa)))
        scored.append(case)
    out = sorted(scored, key=lambda d: d["example_score"], reverse=True)[:N_EXAMPLE_FIGURES * EXAMPLES_PER_FIGURE]
    assert len(out) == N_EXAMPLE_FIGURES * EXAMPLES_PER_FIGURE, f"Expected {N_EXAMPLE_FIGURES * EXAMPLES_PER_FIGURE} example cases, found {len(out)}"
    return [{"figure_id": f"{i + 1:02d}", "examples": out[i * EXAMPLES_PER_FIGURE:(i + 1) * EXAMPLES_PER_FIGURE]} for i in range(N_EXAMPLE_FIGURES)]
def episode_alignment_rows(*, patient_id: str, study_id: str, episode_key: str, annotation_index: int, label: str | None, true_ellipse: Any, fixations: list[Any], img_w: int, img_h: int, rng: np.random.Generator) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    positions, _durations = fixation_arrays(fixations)
    inside_true = fixations_inside(true_ellipse, fixations, positions)
    diagnostic = {"episode_key": episode_key, "patient_id": patient_id, "study_id": study_id, "image_id": study_id, "dicom_id": None, "annotation_index": int(annotation_index), "label": label, "n_true_inside": int(len(inside_true)), "eligible_true": bool(len(inside_true) >= MIN_FIX), "n_placebos": 0, "placebo_attempts": 0, "placebo_retained": False}
    if len(inside_true) < MIN_FIX:
        return [], [], diagnostic
    true_rows = [alignment_row(patient_id=patient_id, study_id=study_id, episode_key=episode_key, annotation_index=annotation_index, label=label, source="true", phase=phase, ellipse=true_ellipse, fixations=inside_true) for phase in PHASES]
    diagnostic.update({"img_w": int(img_w), "img_h": int(img_h), "ellipse_coords": [float(v) for v in true_ellipse.coords], "phase_fixations": {phase: fixation_dicts(phase_fixation_set(inside_true, phase)) for phase in PHASES}, "true_phase_rows": {row["phase"]: row for row in true_rows}})
    placebos, attempts = sample_placebo_ellipses(true_ellipse=true_ellipse, fixations=fixations, positions=positions, img_w=img_w, img_h=img_h, rng=rng)
    diagnostic["n_placebos"] = int(len(placebos))
    diagnostic["placebo_attempts"] = int(attempts)
    diagnostic["placebo_retained"] = bool(len(placebos) >= MIN_ACCEPTED_PLACEBOS)
    placebo_rows = []
    if len(placebos) >= MIN_ACCEPTED_PLACEBOS:
        placebo_rows = [average_placebo_row(patient_id=patient_id, study_id=study_id, episode_key=episode_key, annotation_index=annotation_index, label=label, phase=phase, placebos=placebos) for phase in PHASES]
        diagnostic["placebo_phase_rows"] = {row["phase"]: row for row in placebo_rows}
    return true_rows, placebo_rows, diagnostic
def main() -> None:
    rng = np.random.default_rng(SEED)
    loader = ReflacxLoader(); loader.load_jsons()
    true_rows, placebo_rows, diagnostics = [], [], []
    for patient_id, study_id in loader.study_keys():
        img_w, img_h = loader.get_image_dims(patient_id, study_id)
        if img_w is None or img_h is None: continue
        fixations = loader.get_fixations(patient_id, study_id, period="all")
        if not fixations: continue
        for annotation_index, ellipse in enumerate(loader.get_ellipses(patient_id, study_id)):
            label = ellipse.labels[0] if ellipse.labels else None
            key = f"{patient_id}/{study_id}/{annotation_index}"
            tr, pr, diag = episode_alignment_rows(patient_id=patient_id, study_id=study_id, episode_key=key, annotation_index=annotation_index, label=label, true_ellipse=ellipse, fixations=fixations, img_w=img_w, img_h=img_h, rng=rng)
            true_rows.extend(tr); placebo_rows.extend(pr); diagnostics.append(diag)
    delta_rows = build_contrast_rows(true_rows, placebo_rows); summaries = summarize_contrasts(delta_rows, rng)
    result = {"config": {"seed": SEED, "min_fix": MIN_FIX, "min_accepted_placebos": MIN_ACCEPTED_PLACEBOS}, "counts": {"true_rows": len(true_rows), "placebo_rows": len(placebo_rows), "delta_rows": len(delta_rows), "eligible_true_episodes": sum(d["eligible_true"] for d in diagnostics), "placebo_episodes_with_minimum": sum(d["placebo_retained"] for d in diagnostics)}, "summaries": summaries, "true_rows": true_rows, "placebo_rows": placebo_rows, "delta_rows": delta_rows, "diagnostics": diagnostics}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "alignment_emergence.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    by_key = {(s["contrast"], s["metric"]): s for s in summaries}
    table_rows, fields = [], ["mean_delta", "median_delta", "ci95", "wilcoxon_p", "cohen_dz", "improvement_rate"]
    for contrast in ["temporal_emergence", "extent_object_specificity", "acquisition_object_specificity", "object_specificity_gain"]:
        base = by_key[(contrast, "alignment_score_cos2")]
        table_rows.append({"contrast": contrast, "episodes": base["episodes"], "patients": base["patients"], **{m: {k: by_key[(contrast, m)][k] for k in fields} for m in METRICS}})
    (OUT_DIR / "alignment_emergence_table.json").write_text(json.dumps({"phase_rows": phase_summary_rows(true_rows), "contrast_rows": table_rows}, indent=2), encoding="utf-8")
    (OUT_DIR / "alignment_emergence_examples.json").write_text(json.dumps({"figures": select_example_cases(diagnostics)}, indent=2), encoding="utf-8")
if __name__ == "__main__":
    main()

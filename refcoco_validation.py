from __future__ import annotations
import json, math
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
from alignment_emergence import duration_weighted_covariance
from object_recovery import convex_hull_prediction, fixation_heatmap_cloud, mask_metrics, ourmodel_cloud
from refcocoloader import RefCocoLoader
from stats_utils import paired_difference_stats

SEED, MIN_FIX = 42, 10
PERIODS, METHODS, METRICS = ["all_inside", "posttarget_inside"], ["our_model", "foveal_kde", "convex_hull"], ["iou", "recall", "precision"]
OUT_DIR = Path("results/refcoco_validation")


def ellipse_axis_angle(ellipse: Any) -> float:
    x1, y1, x2, y2 = ellipse.coords
    return 0.0 if abs(x2 - x1) >= abs(y2 - y1) else math.pi / 2.0


def alignment_row(ep: Any) -> dict[str, Any]:
    target_angle = ellipse_axis_angle(ep.ellipse)
    out = {"subject_id": ep.patient_id, "episode_key": ep.study_id, "ref_id": ep.ref_id, "period": ep.period, "split": ep.split, "label": ep.label, "target": ep.target, "n_fix": ep.n_fix}
    for phase, fixes in {"acquisition": ep.fixations[:3], "extent": ep.fixations[3:]}.items():
        cov = duration_weighted_covariance(fixes)
        ev, vec = np.linalg.eigh(cov)
        axis = vec[:, -1]
        angle = math.atan2(float(axis[1]), float(axis[0]))
        err = abs(((angle - target_angle + math.pi / 2.0) % math.pi) - math.pi / 2.0)
        out[f"{phase}_cos2"] = float(math.cos(2.0 * err))
        out[f"{phase}_error_deg"] = float(math.degrees(err))
    cov = duration_weighted_covariance(ep.fixations)
    ev = np.maximum(np.linalg.eigvalsh(cov), 1e-9)
    durations = np.asarray([f.duration for f in ep.fixations], dtype=np.float64)
    t_eff = float(durations.sum() ** 2 / max(float((durations ** 2).sum()), 1e-9))
    out.update({"delta_cos2": out["extent_cos2"] - out["acquisition_cos2"], "delta_error_deg": out["extent_error_deg"] - out["acquisition_error_deg"], "scatter_ratio": float(ev[1] / ev[0]), "orientation_info": float(t_eff * ((ev[1] / ev[0] - 1.0) ** 2) / max(float(ev[1] / ev[0]), 1e-9))})
    return out


def evaluate_recovery_row(ep: Any) -> dict[str, Any]:
    x1, y1, x2, y2 = ep.ellipse.coords
    crop = (max(0, int(math.floor(x1 - 60))), max(0, int(math.floor(y1 - 60))), int(math.ceil(x2 + 60)), int(math.ceil(y2 + 60)))
    target = ep.ellipse.mask(crop)
    row = {"subject_id": ep.patient_id, "episode_key": ep.study_id, "ref_id": ep.ref_id, "period": ep.period, "split": ep.split, "label": ep.label, "target": ep.target, "n_fix": ep.n_fix, "method_metrics": {}}
    for method, predictor in [("our_model", ourmodel_cloud), ("foveal_kde", fixation_heatmap_cloud), ("convex_hull", convex_hull_prediction)]:
        pred, _info = predictor(ep.fixations, crop)
        row["method_metrics"][method] = mask_metrics(pred, target)
    return row


def summarize_alignment(rows: list[dict[str, Any]], rng: np.random.Generator) -> dict[str, Any]:
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows: by_subject[row["subject_id"]].append(row)
    subj_cos = [float(np.mean([r["delta_cos2"] for r in v])) for v in by_subject.values()]
    subj_err = [float(np.mean([r["delta_error_deg"] for r in v])) for v in by_subject.values()]
    return {"episodes": len(rows), "subjects": len(by_subject), "acquisition_cos2": float(np.mean([r["acquisition_cos2"] for r in rows])), "extent_cos2": float(np.mean([r["extent_cos2"] for r in rows])), "delta_cos2": paired_difference_stats(subj_cos, rng), "delta_error_deg": paired_difference_stats(subj_err, rng, higher_is_better=False)}


def summarize_recovery(rows: list[dict[str, Any]], rng: np.random.Generator) -> dict[str, Any]:
    summary = {m: {metric: float(np.mean([r["method_metrics"][m][metric] for r in rows])) for metric in METRICS} for m in METHODS}
    deltas = {}
    for baseline in ["foveal_kde", "convex_hull"]:
        deltas[f"our_model_minus_{baseline}"] = {}
        for metric in METRICS:
            by_subject: dict[str, list[float]] = defaultdict(list)
            for r in rows:
                by_subject[r["subject_id"]].append(float(r["method_metrics"]["our_model"][metric]) - float(r["method_metrics"][baseline][metric]))
            deltas[f"our_model_minus_{baseline}"][metric] = paired_difference_stats([float(np.mean(v)) for v in by_subject.values()], rng)
    return {"episodes": len(rows), "subjects": len({r["subject_id"] for r in rows}), "method_summary": summary, "deltas": deltas}


def subject_differences(alignment_rows: list[dict[str, Any]], recovery_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_subject: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in alignment_rows:
        by_subject[r["subject_id"]]["alignment_delta_cos2"].append(r["delta_cos2"])
        by_subject[r["subject_id"]]["n_fix"].append(r["n_fix"])
        by_subject[r["subject_id"]]["scatter_ratio"].append(r["scatter_ratio"])
        by_subject[r["subject_id"]]["orientation_info"].append(r["orientation_info"])
    for r in recovery_rows:
        by_subject[r["subject_id"]]["our_model_iou"].append(r["method_metrics"]["our_model"]["iou"])
        by_subject[r["subject_id"]]["our_minus_foveal_iou"].append(r["method_metrics"]["our_model"]["iou"] - r["method_metrics"]["foveal_kde"]["iou"])
        by_subject[r["subject_id"]]["our_minus_convex_iou"].append(r["method_metrics"]["our_model"]["iou"] - r["method_metrics"]["convex_hull"]["iou"])
    rows = [{"subject_id": s, "episodes": len(v["our_model_iou"]), **{k: float(np.mean(vals)) for k, vals in v.items() if vals}} for s, v in by_subject.items()]
    def desc(key: str) -> dict[str, float]:
        arr = np.asarray([r[key] for r in rows if key in r], dtype=np.float64)
        return {"mean": float(arr.mean()), "sd": float(arr.std(ddof=1)), "min": float(arr.min()), "q25": float(np.percentile(arr, 25)), "median": float(np.median(arr)), "q75": float(np.percentile(arr, 75)), "max": float(arr.max())}
    return {"subjects": len(rows), "rows": rows, "descriptors": {k: desc(k) for k in ["alignment_delta_cos2", "our_model_iou", "our_minus_foveal_iou"]}}


def main() -> None:
    rng = np.random.default_rng(SEED)
    loader = RefCocoLoader(); loader.load_jsons()
    result = {"config": {"seed": SEED, "min_fix": MIN_FIX, "periods": PERIODS, "methods": METHODS, "bbox_as_ellipse": True}, "periods": {}}
    all_alignment, all_recovery = [], []
    for period in PERIODS:
        episodes = list(loader.iter_target_episodes(period=period, min_fix=MIN_FIX))
        alignment = [alignment_row(ep) for ep in episodes]
        recovery = [evaluate_recovery_row(ep) for ep in episodes]
        result["periods"][period] = {"count": len(episodes), "alignment": summarize_alignment(alignment, rng), "recovery": summarize_recovery(recovery, rng), "alignment_rows": alignment, "recovery_rows": recovery}
        if period == "all_inside": all_alignment, all_recovery = alignment, recovery
    subjects = subject_differences(all_alignment, all_recovery)
    table = {"primary_period": "all_inside", "alignment": result["periods"]["all_inside"]["alignment"], "recovery": result["periods"]["all_inside"]["recovery"], "supplementary_posttarget": {"alignment": result["periods"]["posttarget_inside"]["alignment"], "recovery": result["periods"]["posttarget_inside"]["recovery"]}}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "refcoco_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUT_DIR / "refcoco_validation_table.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    (OUT_DIR / "refcoco_subject_differences.json").write_text(json.dumps(subjects, indent=2), encoding="utf-8")
    print(json.dumps({"all_inside": result["periods"]["all_inside"]["count"], "posttarget_inside": result["periods"]["posttarget_inside"]["count"], "subjects": subjects["subjects"]}, indent=2))


if __name__ == "__main__":
    main()

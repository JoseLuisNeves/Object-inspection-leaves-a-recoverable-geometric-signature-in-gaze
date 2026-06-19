import json, math, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse
from reflacxloader import ReflacxLoader
from object_recovery import FOVEAL_DISK, FOVEAL_RADIUS, convex_hull_prediction, fixation_heatmap_cloud, ourmodel_cloud
CXR_DIR, TRUTH, EXP1_DIR, EXP2_DIR, EXP3_DIR = Path("CXRs"), "#003b8e", Path("results/01_alignment_emergence"), Path("results/02_object_recovery"), Path("results/03_mention_recovery")
EXP1_PATH, EXP2_PATH, EXP3_PATH, EXP3_EXAMPLES_PATH = EXP1_DIR / "alignment_emergence_examples.json", EXP2_DIR / "object_recovery.json", EXP3_DIR / "mention_recovery.json", EXP3_DIR / "mention_recovery_examples.json"
EXP1_COLORS = {"acquisition": ("#f0abfc", "#c026d3"), "extent": ("#99f6e4", "#0f766e")}
METHOD_COLORS = {"foveal_kde": ("#f0abfc", "#c026d3"), "convex_hull": ("#fdba74", "#c2410c"), "our_model": ("#99f6e4", "#0f766e")}
MIN_EXP2_ASPECT, MIN_OUR_IOU, MAX_FOVEAL_IOU, EXP2_FIGS, EXP2_COLS = 1.5, 0.80, 1.0, 5, 3
WINDOW_START, WINDOW_END = -3.5, -0.5
EXP3_ROLE_TITLES = {"free_form_not_centered_partial_recovery": "Not centered", "target_centered_axis_weak": "Centered", "target_centered_axis_structured": "Centered + stable geometry"}
EXP3_ROLE_COLORS = {"free_form_not_centered_partial_recovery": "#c026d3", "target_centered_axis_weak": "#0f766e", "target_centered_axis_structured": "#003b8e"}
def cxr_path(case: dict) -> Path | None:
    base = CXR_DIR / f"{case['patient_id']}__{case['study_id']}"
    for s in (".png", ".jpg", ".jpeg"):
        p = base.with_suffix(s)
        if not p.exists(): continue
        h, w = plt.imread(p).shape[:2]
        if (w, h) == (int(case["img_w"]), int(case["img_h"])): return p
        print(f"Warning: skipping {p}; CXR dims {w}x{h} != expected {case['img_w']}x{case['img_h']}", file=sys.stderr)
    return None
def ellipse_aspect(coords: list[float]) -> float:
    x1, y1, x2, y2 = coords; return max(abs(x2 - x1), abs(y2 - y1)) / max(min(abs(x2 - x1), abs(y2 - y1)), 1e-9)
def draw_truth(ax, coords: list[float]) -> None:
    x1, y1, x2, y2 = coords; ax.add_patch(Ellipse(((x1 + x2) / 2, (y1 + y2) / 2), x2 - x1, y2 - y1, fill=False, color=TRUTH, lw=2.3, ls="--"))
def set_panel(ax, case: dict, crop: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = crop; ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.set_aspect("equal"); ax.axis("off"); ax.set_facecolor("white")
def load_exp1_groups() -> list[dict]:
    data = json.loads(EXP1_PATH.read_text(encoding="utf-8")); return data.get("figures") or [{"figure_id": "01", "examples": data["examples"]}]
def axis_gain(case: dict) -> float:
    r = case["true_phase_rows"]; return float(r["acquisition"]["alignment_error_deg"] - r["extent"]["alignment_error_deg"])
def write_exp1_cases(groups: list[dict]) -> None:
    lines = ["Exp1 alignment emergence selected cases", ""]
    for g in groups:
        lines.append(f"Figure {g['figure_id']}")
        for i, c in enumerate(g["examples"], 1):
            base = CXR_DIR / f"{c['patient_id']}__{c['study_id']}"
            exp = ", ".join(str(base.with_suffix(s)) for s in (".png", ".jpg", ".jpeg"))
            r = c["true_phase_rows"]
            lines += [f"{i}. patient_id: {c['patient_id']}", f"   study_id: {c['study_id']}", f"   expected CXR filenames: {exp}", f"   label: {c.get('label')}", f"   annotation_index: {c['annotation_index']}", f"   ellipse_aspect: {c['ellipse_aspect']:.3f}", f"   acquisition_error_deg: {r['acquisition']['alignment_error_deg']:.2f}", f"   extent_error_deg: {r['extent']['alignment_error_deg']:.2f}", f"   axis_alignment_improvement_deg: {axis_gain(c):.2f}", ""]
    (EXP1_DIR / "alignment_emergence_selected_cases.txt").write_text("\n".join(lines), encoding="utf-8")
def draw_cov(ax, fix: list[dict], color: str) -> None:
    xy = np.array([[f["x"], f["y"]] for f in fix], dtype=float); w = np.maximum(np.array([f["duration"] for f in fix], dtype=float), 1e-9)
    mu = np.average(xy, axis=0, weights=w); d = xy - mu
    ev, vec = np.linalg.eigh((w[:, None, None] * d[:, :, None] * d[:, None, :]).sum(axis=0) / w.sum()); ev = np.maximum(ev, 1e-9)
    ax.add_patch(Ellipse(mu, 2 * math.sqrt(ev[-1]), 2 * math.sqrt(ev[0]), angle=math.degrees(math.atan2(vec[1, -1], vec[0, -1])), fill=False, color=color, lw=2.0))
def draw_exp1_panel(ax, case: dict, phase: str) -> None:
    fix, (pt, ln) = case["phase_fixations"][phase], EXP1_COLORS[phase]
    x1, y1, x2, y2 = case["ellipse_coords"]; xy = np.array([[f["x"], f["y"]] for f in fix], dtype=float); dur = np.array([f["duration"] for f in fix], dtype=float)
    path = cxr_path(case); ax.imshow(plt.imread(path), cmap="gray", extent=(0, case["img_w"], case["img_h"], 0)) if path else None
    draw_truth(ax, case["ellipse_coords"])
    ax.scatter(xy[:, 0], xy[:, 1], s=20 + 65 * dur / max(float(dur.max()), 1e-9), c=pt, alpha=0.82, edgecolors="white", linewidths=0.7)
    draw_cov(ax, fix, ln)
    if phase == "extent": ax.text(0.04, 0.07, f"Alignment +{axis_gain(case):.0f}$^\\circ$", transform=ax.transAxes, fontsize=13, weight="bold", color="#111827", bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none", "pad": 3})
    pad = max(x2 - x1, y2 - y1) * 0.45; set_panel(ax, case, (max(0, min(xy[:, 0].min(), x1) - pad), max(0, min(xy[:, 1].min(), y1) - pad), min(case["img_w"], max(xy[:, 0].max(), x2) + pad), min(case["img_h"], max(xy[:, 1].max(), y2) + pad)))
def main_exp1() -> None:
    EXP1_DIR.mkdir(parents=True, exist_ok=True); groups = load_exp1_groups(); write_exp1_cases(groups)
    for g in groups:
        fig, axes = plt.subplots(2, len(g["examples"]), figsize=(3 * len(g["examples"]), 5.2), constrained_layout=True)
        for col, c in enumerate(g["examples"]):
            draw_exp1_panel(axes[0, col], c, "acquisition"); draw_exp1_panel(axes[1, col], c, "extent")
            axes[0, col].set_title(f"{c.get('label') or ''}", fontsize=15, weight="bold", pad=10)
        axes[0, 0].text(-0.14, 0.5, "Acquisition", transform=axes[0, 0].transAxes, rotation=90, va="center", ha="center", fontsize=14, weight="bold")
        axes[1, 0].text(-0.14, 0.5, "Extent", transform=axes[1, 0].transAxes, rotation=90, va="center", ha="center", fontsize=14, weight="bold")
        for s in (".png", ".svg"): fig.savefig(EXP1_DIR / f"alignment_emergence_examples_{g['figure_id']}{s}", dpi=220 if s == ".png" else None)
        if g["figure_id"] == "01":
            for s in (".png", ".svg"): fig.savefig(EXP1_DIR / f"alignment_emergence_examples{s}", dpi=220 if s == ".png" else None)
        plt.close(fig)
def exp2_candidates() -> list[dict]:
    if not EXP2_PATH.exists(): raise SystemExit(f"Missing {EXP2_PATH}. Run python .\\object_recovery.py first.")
    by_key: dict[str, dict] = {}
    for r in json.loads(EXP2_PATH.read_text(encoding="utf-8"))["rows"]:
        e = by_key.setdefault(r["episode_key"], {k: r[k] for k in ("patient_id", "study_id", "episode_key", "annotation_index", "label", "n_fix")})
        e.setdefault("method_metrics", {})[r["method"]] = {m: r[m] for m in ("iou", "recall", "precision")}
    loader = ReflacxLoader(); loader.load_jsons(); out = []
    for e in by_key.values():
        ell = loader.get_ellipses(e["patient_id"], e["study_id"])[int(e["annotation_index"])]; e["ellipse_coords"] = [float(v) for v in ell.coords]; e["ellipse_aspect"] = ellipse_aspect(e["ellipse_coords"])
        m = e["method_metrics"]
        if e["ellipse_aspect"] >= MIN_EXP2_ASPECT and m["our_model"]["iou"] >= MIN_OUR_IOU and m["foveal_kde"]["iou"] <= MAX_FOVEAL_IOU: out.append(e)
    out.sort(key=lambda e: 2 * e["method_metrics"]["our_model"]["iou"] - e["method_metrics"]["foveal_kde"]["iou"], reverse=True); used, groups = set(), []
    for gi in range(EXP2_FIGS):
        picked, labels = [], set()
        for unique in (True, False):
            for e in out:
                if len(picked) < EXP2_COLS and e["episode_key"] not in used and (not unique or e.get("label") not in labels): picked.append(e); used.add(e["episode_key"]); labels.add(e.get("label"))
            if len(picked) == EXP2_COLS: break
        groups.append({"figure_id": f"{gi + 1:02d}", "examples": picked})
    return groups
def load_exp2_case(loader: ReflacxLoader, e: dict) -> dict:
    ell = loader.get_ellipses(e["patient_id"], e["study_id"])[int(e["annotation_index"])]; w, h = loader.get_image_dims(e["patient_id"], e["study_id"])
    return {**e, "img_w": w, "img_h": h, "crop": ell.crop(w, h), "fixations": [f for f in loader.get_fixations(e["patient_id"], e["study_id"], period="all") if ell.contains_point(f.x, f.y)]}
def safe_foveal_mask(fixations: list, crop: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = crop; mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    for f in fixations:
        cx, cy = int(round(f.x - x0)), int(round(f.y - y0))
        px0, px1 = max(0, cx - FOVEAL_RADIUS), min(mask.shape[1], cx + FOVEAL_RADIUS + 1)
        py0, py1 = max(0, cy - FOVEAL_RADIUS), min(mask.shape[0], cy + FOVEAL_RADIUS + 1)
        if px0 >= px1 or py0 >= py1: continue
        kx0, ky0 = px0 - (cx - FOVEAL_RADIUS), py0 - (cy - FOVEAL_RADIUS)
        mask[py0:py1, px0:px1] |= FOVEAL_DISK[ky0:ky0 + py1 - py0, kx0:kx0 + px1 - px0]
    return mask
def draw_recovery_panel(ax, case: dict, method: str, safe_foveal: bool = False, show_centroid: bool = False) -> None:
    path = cxr_path(case); ax.imshow(plt.imread(path), cmap="gray", extent=(0, case["img_w"], case["img_h"], 0)) if path else None
    draw_truth(ax, case["ellipse_coords"]); crop = case["crop"]
    if method == "input":
        xy = np.array([[f.x, f.y] for f in case["fixations"]], dtype=float); dur = np.array([f.duration for f in case["fixations"]], dtype=float)
        ax.scatter(xy[:, 0], xy[:, 1], s=18 + 58 * dur / max(float(dur.max()), 1e-9), c=EXP1_COLORS["extent"][0], alpha=0.86, edgecolors="white", linewidths=0.6)
        if show_centroid and "centroid_x" in case: ax.scatter([case["centroid_x"]], [case["centroid_y"]], marker="+", s=190, c="#111827", linewidths=2.6, zorder=6)
    else:
        mask = safe_foveal_mask(case["fixations"], crop) if safe_foveal and method == "foveal_kde" else {"foveal_kde": fixation_heatmap_cloud, "convex_hull": convex_hull_prediction, "our_model": ourmodel_cloud}[method](case["fixations"], crop)[0]
        x0, y0, x1, y1 = crop; fill, line = METHOD_COLORS[method]
        ax.imshow(np.ma.masked_where(~mask, mask), cmap=ListedColormap([fill]), alpha=0.34, extent=(x0, x1, y1, y0)); ax.contour(np.arange(x0, x1), np.arange(y0, y1), mask.astype(float), levels=[0.5], colors=[line], linewidths=1.8)
        m = case["method_metrics"][method]; ax.text(0.04, 0.07, f"IoU {m['iou']:.2f}", transform=ax.transAxes, fontsize=13, weight="bold", color="#111827", bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none", "pad": 3})
    set_panel(ax, case, crop)
def draw_exp3_title(ax, case: dict) -> None:
    role = case.get("role")
    ax.text(0.5, 1.17, case.get("label") or "", transform=ax.transAxes, ha="center", va="bottom", fontsize=14, weight="bold", color="#111827", clip_on=False)
    ax.text(0.5, 1.08, EXP3_ROLE_TITLES.get(role, role or ""), transform=ax.transAxes, ha="center", va="bottom", fontsize=11.5, weight="semibold", color=EXP3_ROLE_COLORS.get(role, "#111827"), clip_on=False)
def write_exp2_cases(groups: list[dict]) -> None:
    lines = [f"Exp2 object recovery selected cases; ellipse_aspect >= {MIN_EXP2_ASPECT}; our_model_iou >= {MIN_OUR_IOU}; foveal_kde_iou <= {MAX_FOVEAL_IOU}", ""]
    for g in groups:
        lines.append(f"Figure {g['figure_id']}")
        for i, c in enumerate(g["examples"], 1):
            base = CXR_DIR / f"{c['patient_id']}__{c['study_id']}"; exp = ", ".join(str(base.with_suffix(s)) for s in (".png", ".jpg", ".jpeg"))
            lines += [f"{i}. patient_id: {c['patient_id']}", f"   study_id: {c['study_id']}", f"   expected CXR filenames: {exp}", f"   label: {c.get('label')}", f"   annotation_index: {c['annotation_index']}", f"   n_fix: {c['n_fix']}", f"   ellipse_aspect: {c['ellipse_aspect']:.3f}"]
            lines += [f"   {method}: IoU={m['iou']:.3f}, Recall={m['recall']:.3f}, Precision={m['precision']:.3f}" for method, m in c["method_metrics"].items()] + [""]
    (EXP2_DIR / "object_recovery_selected_cases.txt").write_text("\n".join(lines), encoding="utf-8")
def main_exp2() -> None:
    EXP2_DIR.mkdir(parents=True, exist_ok=True); loader = ReflacxLoader(); loader.load_jsons()
    groups = [{"figure_id": g["figure_id"], "examples": [load_exp2_case(loader, e) for e in g["examples"]]} for g in exp2_candidates()]; write_exp2_cases(groups)
    rows = [("input", "Input"), ("foveal_kde", "Foveal coverage"), ("convex_hull", "Convex hull"), ("our_model", "Our Model")]
    for g in groups:
        fig, axes = plt.subplots(4, EXP2_COLS, figsize=(3 * EXP2_COLS, 8.8), constrained_layout=True)
        for col, c in enumerate(g["examples"]):
            axes[0, col].set_title(f"{c.get('label') or ''}", fontsize=15, weight="bold", pad=10)
            for r, (method, title) in enumerate(rows):
                draw_recovery_panel(axes[r, col], c, method)
                if col == 0: axes[r, col].text(-0.14, 0.5, title, transform=axes[r, col].transAxes, rotation=90, va="center", ha="center", fontsize=14, weight="bold")
        for s in (".png", ".svg"):
            fig.savefig(EXP2_DIR / f"object_recovery_examples_{g['figure_id']}{s}", dpi=220 if s == ".png" else None)
            if g["figure_id"] == "01": fig.savefig(EXP2_DIR / f"object_recovery_examples{s}", dpi=220 if s == ".png" else None)
        plt.close(fig)
def exp3_key(ep) -> str:
    return f"{ep.patient_id}/{ep.study_id}/{ep.label}/{ep.word_start}-{ep.word_end}"
def exp3_index(loader: ReflacxLoader) -> dict[str, object]:
    return {exp3_key(ep): ep for ep in loader.iter_mention_window_episodes(min_fix=5, unique_label_only=True, selection_mode="all", start_rel_to_mention_start=WINDOW_START, end_rel_to_mention_start=WINDOW_END)}
def exp3_candidates() -> list[dict]:
    if not EXP3_EXAMPLES_PATH.exists(): raise SystemExit(f"Missing {EXP3_EXAMPLES_PATH}. Run python .\\mention_recovery.py first.")
    groups = json.loads(EXP3_EXAMPLES_PATH.read_text(encoding="utf-8")).get("figure_candidates", [])
    if len(groups) != 5 or any(len(g.get("examples", [])) != 3 for g in groups): raise AssertionError("Exp3 examples must contain five 3-column figure_candidates")
    return groups
def load_exp3_case(loader: ReflacxLoader, e: dict) -> dict:
    if not hasattr(load_exp3_case, "_idx"): load_exp3_case._idx = exp3_index(loader)
    ep = load_exp3_case._idx.get(e["episode_key"])
    if ep is None: raise AssertionError(f"Could not reconstruct Exp3 episode {e['episode_key']}")
    w, h = loader.get_image_dims(ep.patient_id, ep.study_id)
    return {**e, "img_w": w, "img_h": h, "crop": ep.ellipse.crop(w, h), "fixations": ep.fixations, "ellipse_coords": [float(v) for v in ep.ellipse.coords]}
def write_exp3_cases(groups: list[dict]) -> None:
    lines = ["Exp3 mention recovery selected cases; columns are free-form/not-centered, target-centered/weak-axis, and target-centered/axis-structured; candidates exported by mention_recovery.py.", ""]
    for g in groups:
        lines.append(f"Figure {g['figure_id']}")
        for i, c in enumerate(g["examples"], 1):
            base = CXR_DIR / f"{c['patient_id']}__{c['study_id']}"; exp = ", ".join(str(base.with_suffix(s)) for s in (".png", ".jpg", ".jpeg"))
            lines += [f"{i}. role: {c.get('role')}", f"   patient_id: {c['patient_id']}", f"   study_id: {c['study_id']}", f"   expected CXR filenames: {exp}", f"   label: {c.get('label')}", f"   phrase: {c.get('phrase')}", f"   word_start-word_end: {c['word_start']}-{c['word_end']}", f"   n_fix: {c['n_fix']}", f"   centroid_inside_gt: {c['centroid_inside_gt']}", f"   normalized_centroid_distance: {c['normalized_centroid_distance']:.3f}", f"   inside_gt_fraction: {c['inside_gt_fraction']:.3f}", f"   scatter_ratio: {c['scatter_ratio']:.3f}", f"   orientation_info: {c['orientation_info']:.3f}", f"   axis_structured: {c['axis_structured']}"]
            lines += [f"   {method}: IoU={m['iou']:.3f}, Dice={m['dice']:.3f}" for method, m in c["method_metrics"].items()] + [""]
    (EXP3_DIR / "mention_recovery_selected_cases.txt").write_text("\n".join(lines), encoding="utf-8")
def main_exp3() -> None:
    EXP3_DIR.mkdir(parents=True, exist_ok=True); loader = ReflacxLoader(); loader.load_jsons()
    if hasattr(load_exp3_case, "_idx"): delattr(load_exp3_case, "_idx")
    groups = [{"figure_id": g["figure_id"], "examples": [load_exp3_case(loader, e) for e in g["examples"]]} for g in exp3_candidates()]; write_exp3_cases(groups)
    rows = [("input", "Input"), ("foveal_kde", "Foveal coverage"), ("our_model", "Object recovery")]
    for g in groups:
        fig, axes = plt.subplots(3, 3, figsize=(9.2, 6.9), constrained_layout=True)
        for col, c in enumerate(g["examples"]):
            draw_exp3_title(axes[0, col], c)
            for r, (method, title) in enumerate(rows):
                draw_recovery_panel(axes[r, col], c, method, safe_foveal=True, show_centroid=(method == "input"))
                if col == 0: axes[r, col].text(-0.14, 0.5, title, transform=axes[r, col].transAxes, rotation=90, va="center", ha="center", fontsize=14, weight="bold")
        for s in (".png", ".svg"):
            fig.savefig(EXP3_DIR / f"mention_recovery_examples_{g['figure_id']}{s}", dpi=220 if s == ".png" else None)
            if g["figure_id"] == "01": fig.savefig(EXP3_DIR / f"mention_recovery_examples{s}", dpi=220 if s == ".png" else None)
        plt.close(fig)
def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"exp1", "exp2", "exp3"}: print("Usage: python viz.py exp1|exp2|exp3"); return
    {"exp1": main_exp1, "exp2": main_exp2, "exp3": main_exp3}[sys.argv[1]]()
if __name__ == "__main__": main()

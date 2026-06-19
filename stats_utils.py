from __future__ import annotations
from typing import Any
import numpy as np
from scipy.stats import wilcoxon
def finite_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]
def bootstrap_ci(values: Any, rng: np.random.Generator, n_boot: int = 5000, alpha: float = 0.05) -> tuple[float, float]:
    arr = finite_array(values)
    if len(arr) == 0: return (float("nan"), float("nan"))
    means = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)): means[i] = float(np.mean(rng.choice(arr, size=len(arr), replace=True)))
    return (float(np.percentile(means, 100.0 * alpha / 2.0)), float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0))))
def cohen_dz(values: Any) -> float:
    arr = finite_array(values)
    if len(arr) < 2: return float("nan")
    sd = float(np.std(arr, ddof=1))
    return float(np.mean(arr) / sd) if sd > 0.0 else float("nan")
def wilcoxon_p_value(values: Any) -> float:
    arr = finite_array(values)
    arr = arr[arr != 0.0]
    if len(arr) == 0: return float("nan")
    return float(wilcoxon(arr).pvalue)
def paired_difference_stats(values: Any, rng: np.random.Generator, *, higher_is_better: bool = True) -> dict[str, Any]:
    arr = finite_array(values)
    lo, hi = bootstrap_ci(arr, rng)
    if len(arr) == 0: return {"n": 0,"mean_delta": float("nan"),"median_delta": float("nan"), "ci95": [lo, hi], "wilcoxon_p": float("nan"), "cohen_dz": float("nan"), "improvement_rate": float("nan")}
    improved = arr > 0.0 if higher_is_better else arr < 0.0
    return {"n": int(len(arr)),"mean_delta": float(np.mean(arr)),"median_delta": float(np.median(arr)), "ci95": [lo, hi], "wilcoxon_p": wilcoxon_p_value(arr), "cohen_dz": cohen_dz(arr), "improvement_rate": float(np.mean(improved))}

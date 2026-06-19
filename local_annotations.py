from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from fixation_builder import Fixation

@dataclass(frozen=True)
class EllipseAnnotation:
    coords: Tuple[float, float, float, float]
    labels: List[str]
    @property
    def center(self) -> Tuple[float, float]:
        xmin, ymin, xmax, ymax = self.coords
        return ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0)
    @property
    def radial_coords(self) -> Tuple[float, float, float, float]:
        cx, cy = self.center
        xmin, ymin, xmax, ymax = self.coords
        return cx, cy, (xmax - xmin) / 2.0, (ymax - ymin) / 2.0
    @property
    def area(self) -> float:
        _, _, rx, ry = self.radial_coords
        return float(np.pi * rx * ry)
    def contains_point(self, x: float, y: float) -> bool:
        cx, cy, rx, ry = self.radial_coords
        if rx <= 0 or ry <= 0:
            return False
        return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0
    def mask(self, crop: Tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = crop
        yy, xx = np.mgrid[y0:y1, x0:x1]
        cx, cy, rx, ry = self.radial_coords
        return ((xx + 0.5 - cx) / rx) ** 2 + ((yy + 0.5 - cy) / ry) ** 2 <= 1.0
    def crop(self, img_w: int, img_h: int, margin: int = 60) -> Tuple[int, int, int, int]:
        x0, y0, x1, y1 = self.coords
        return max(0, int(np.floor(x0 - margin))), max(0, int(np.floor(y0 - margin))), min(img_w, int(np.ceil(x1 + margin))), min(img_h, int(np.ceil(y1 + margin)))

@dataclass
class ObjectEpisode: # for experiment 2?
    patient_id: str
    study_id: str
    ellipse: EllipseAnnotation
    fixations: List[Fixation]
    label: Optional[str] = None
    @property
    def n_fix(self) -> int:
        return len(self.fixations)
    @property
    def total_dwell(self) -> float:
        return float(sum(f.duration for f in self.fixations))


@dataclass
class MentionEpisode: # for experiment 3?
    patient_id: str
    study_id: str
    label: str
    phrase: str
    word_start: int
    word_end: int
    mention_start_time: float
    mention_end_time: float
    window_start_time: float
    window_end_time: float
    ellipse: EllipseAnnotation
    fixations: List[Fixation]
    img_w: Optional[int] = None
    img_h: Optional[int] = None
    selection_mode: str = "inside"
    n_window_fix: Optional[int] = None
    n_inside_gt: Optional[int] = None
    inside_gt_fraction: Optional[float] = None
    control_name: Optional[str] = None

    @property
    def n_fix(self) -> int:
        return len(self.fixations)

    @property
    def total_dwell(self) -> float:
        return float(sum(f.duration for f in self.fixations))


@dataclass
class RefCocoEpisode:
    patient_id: str
    study_id: str
    ellipse: EllipseAnnotation
    fixations: List[Fixation]
    label: str
    target: str
    target_category: str
    ref_id: str
    ref_sentence: Optional[str]
    target_word_index: int
    target_start_time: float
    target_end_time: float
    period: str
    split: Optional[str] = None
    @property
    def n_fix(self) -> int:
        return len(self.fixations)
    @property
    def total_dwell(self) -> float:
        return float(sum(f.duration for f in self.fixations))

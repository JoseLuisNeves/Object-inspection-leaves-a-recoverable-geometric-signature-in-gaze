from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

@dataclass(frozen=True)
class Fixation:
    x: float
    y: float
    start_time: float
    end_time: float
    @property
    def center(self) -> Tuple[float, float]:
        return (self.x, self.y)
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

def build_fixations(raw: Iterable[Dict[str, str]], period: str = "all", start_reporting_ts: Optional[float] = None) -> List[Fixation]:
    out: List[Fixation] = []
    for fix in raw:
        st = float(fix["timestamp_start_fixation"])
        et = float(fix["timestamp_end_fixation"])
        if period == "reporting" and st < start_reporting_ts: continue
        if period == "pre-reporting" and st >= start_reporting_ts: continue
        out.append(Fixation(x=float(fix["x_position"]), y=float(fix["y_position"]), start_time=st, end_time=et))
    return out
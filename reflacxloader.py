from __future__ import annotations
import json
from pathlib import Path
from typing import Iterator, List, Optional, Tuple
from entity_extraction import iter_abnormality_mentions
from local_annotations import EllipseAnnotation, MentionEpisode, ObjectEpisode
from fixation_builder import Fixation, build_fixations

class ReflacxLoader:
    def __init__(self, json_dir: str = "dataset/jsons"):
        self.json_dir = Path(json_dir)
        self._loaded = False
    def _read_json(self, filename: str, required: bool = True):
        path = self.json_dir / filename
        with path.open("r", encoding="utf-8") as f: return json.load(f)
    def load_jsons(self) -> None:
        self.transcripts_dict = self._read_json("transcripts.json")
        self.word_timestamps_dict = self._read_json("timestamps.json", required=False)
        self.ellipses_dict = self._read_json("abnormality_ellipses.json")
        self.abnormality_mappings_dict = self._read_json("abnormality_mappings.json", required=False)
        self.fixations_dict = self._read_json("fixations.json", required=False)
        self.chest_dict = self._read_json("chest_bbs.json", required=False)
        self.img_dims_dict = self._read_json("img_dims.json", required=False)
        self._loaded = True
    def _require_loaded(self) -> None: 
        if not self._loaded: self.load_jsons()
    def study_keys(self) -> Iterator[Tuple[str, str]]:
        self._require_loaded()
        for pid, studies in self.ellipses_dict.items():
            for sid in studies: yield pid, sid
    def get_image_dims(self, pid: str, sid: str) -> Tuple[Optional[int], Optional[int]]: 
        # for the random ellipse baseline to know not to go outside img bounds
        self._require_loaded()
        dims = self.img_dims_dict.get(pid, {}).get(sid)
        if not dims:
            dims = self.img_dims_dict.get(sid)
        if not dims:
            return None, None
        w = dims.get("image_size_x", dims.get("width"))
        h = dims.get("image_size_y", dims.get("height"))
        return (int(w) if w is not None else None, int(h) if h is not None else None)

    def get_ellipses(self, pid: str, sid: str) -> List[EllipseAnnotation]:
        self._require_loaded()
        out: List[EllipseAnnotation] = []
        for e in self.ellipses_dict.get(pid, {}).get(sid, []):
            coords = (float(e["xmin"]), float(e["ymin"]), float(e["xmax"]), float(e["ymax"]))
            labels = [k.lower().strip() for k, v in e.items() if k not in {"xmin", "ymin", "xmax", "ymax"} and isinstance(v, str) and v.lower() == "true"]
            out.append(EllipseAnnotation(coords=coords, labels=labels))
        return out

    def get_fixations(self, pid: str, sid: str, period: str = "all") -> List[Fixation]:
        self._require_loaded()
        raw = self.fixations_dict.get(pid, {}).get(sid, [])
        wts = self.word_timestamps_dict.get(pid, {}).get(sid, [])
        start_ts = float(wts[0]["timestamp_start_word"]) if wts else None
        if period != "all" and start_ts is None: return []
        return build_fixations(raw, period=period, start_reporting_ts=start_ts)

    def iter_object_supervised_episodes(self, period: str = "all", min_fix: int = 4) -> Iterator[ObjectEpisode]:
        self._require_loaded()
        for pid, sid in self.study_keys():
            fixations = self.get_fixations(pid, sid, period=period)
            for ellipse in self.get_ellipses(pid, sid):
                inside = [f for f in fixations if ellipse.contains_point(f.x, f.y)]
                if len(inside) < min_fix: continue
                yield ObjectEpisode(patient_id=pid,study_id=sid, ellipse=ellipse, fixations=inside, label=ellipse.labels[0] if ellipse.labels else None)

    def iter_mention_supervised_episodes(self, window_pre: float = 5.0, window_post: float = 3.0, min_fix: int = 4, unique_label_only: bool = True) -> Iterator[MentionEpisode]:
        yield from self.iter_mention_window_episodes(window_pre=window_pre, window_post=window_post, min_fix=min_fix, unique_label_only=unique_label_only, selection_mode="inside", control_name="object_filtered_mention_reference")

    def iter_mention_window_episodes(self, window_pre: float = 5.0, window_post: float = 3.0, min_fix: int = 4, unique_label_only: bool = True, selection_mode: str = "all", control_name: Optional[str] = None, start_rel_to_mention_start: Optional[float] = None, end_rel_to_mention_start: Optional[float] = None) -> Iterator[MentionEpisode]:
        self._require_loaded()
        for pid, sid in self.study_keys():
            word_rows = self.word_timestamps_dict.get(pid, {}).get(sid, [])
            fixations = self.get_fixations(pid, sid, period="all")
            for mention in iter_abnormality_mentions(word_rows, self.get_ellipses(pid, sid), self.abnormality_mappings_dict, unique_label_only=unique_label_only):
                mention_start = mention["mention_start"]
                mention_end = mention["mention_end"]
                window_start = mention_start + start_rel_to_mention_start
                window_end = mention_start + end_rel_to_mention_start
                ellipse = mention["ellipse"]
                window_fixations = [f for f in fixations if window_start <= f.start_time <= window_end]
                inside_gt = [f for f in window_fixations if ellipse.contains_point(f.x, f.y)]
                selected = inside_gt if selection_mode == "inside" else window_fixations
                if len(selected) < min_fix: continue
                yield MentionEpisode(patient_id=pid, study_id=sid, label=mention["label"], phrase=mention["phrase"], word_start=mention["word_start"], word_end=mention["word_end"], mention_start_time=mention_start, mention_end_time=mention_end, window_start_time=window_start, window_end_time=window_end, ellipse=ellipse, fixations=selected, selection_mode=selection_mode, n_window_fix=len(window_fixations), n_inside_gt=len(inside_gt), control_name=control_name)

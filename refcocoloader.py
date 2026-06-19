from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple
from fixation_builder import Fixation
from local_annotations import EllipseAnnotation, RefCocoEpisode
class RefCocoLoader:
    def __init__(self, data_dir: str = "refcoco"):
        self.data_dir = Path(data_dir)
        self._loaded = False
    def _read_json(self, filename: str):
        path = self.data_dir / filename
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    def load_jsons(self) -> None:
        self.train_data = self._read_json("refcocogaze_train_correct.json")
        self.val_data = self._read_json("refcocogaze_val_correct.json")
        self.data = list(self.train_data) + list(self.val_data)
        timing_path = self.data_dir / "word-timing.json"
        with timing_path.open("r", encoding="utf-8") as f:
            timings = json.load(f)
        self.timing_map = {item["ref_id"]: item["target_period"] for item in timings if "ref_id" in item}
        self._loaded = True
    def _require_loaded(self) -> None:
        if not self._loaded:
            self.load_jsons()
    @staticmethod
    def _normalise_token(value: object) -> str:
        return str(value).lower().strip()
    def find_target_idx(self, ref_words: Sequence[str], target: str) -> int:
        target_norm = self._normalise_token(target)
        words = [self._normalise_token(w) for w in ref_words]
        for idx, word in enumerate(words):
            if word == target_norm:
                return idx
        target_tokens = target_norm.split()
        if not target_tokens:
            return -1
        for idx in range(len(words) - len(target_tokens) + 1):
            if words[idx : idx + len(target_tokens)] == target_tokens:
                return idx
        return -1
    def target_period(self, record: Dict) -> Optional[Tuple[float, float]]:
        raw = self.timing_map.get(record.get("REF_ID"), record.get("TARGET_SPOKEN_PERIOD"))
        if not raw or len(raw) < 2:
            return None
        sound_on = float(record.get("SOUND_ON", 0.0))
        return sound_on + float(raw[0]), sound_on + float(raw[1])
    @staticmethod
    def target_ellipse(record: Dict) -> EllipseAnnotation:
        x, y, w, h = [float(v) for v in record["BBOX"]]
        label = str(record.get("TARGET_CATEGORY") or record.get("TARGET") or "unknown").lower().strip()
        return EllipseAnnotation(coords=(x, y, x + w, y + h), labels=[label])
    @staticmethod
    def fixations(record: Dict) -> List[Fixation]:
        return [Fixation(x=float(x), y=float(y), start_time=float(start), end_time=float(start) + float(duration))
            for x, y, start, duration in zip(record.get("FIX_X", []), record.get("FIX_Y", []), record.get("FIX_START", []), record.get("FIX_DURATION", []))]
    def selected_fixations(self, record: Dict, period: str, ellipse: EllipseAnnotation) -> Optional[List[Fixation]]:
        target_period = self.target_period(record)
        target_start = target_period[0] if target_period is not None else None
        out = []
        for fix in self.fixations(record):
            inside = ellipse.contains_point(fix.x, fix.y)
            if period.endswith("_inside") and not inside: continue
            if period.startswith("pretarget") and not (0 <= fix.start_time < target_start): continue
            if period.startswith("posttarget") and not (fix.start_time >= target_start): continue
            out.append(fix)
        return out

    def iter_target_episodes(self, period: str = "all_inside", min_fix: int = 4, first_word_only: bool = False) -> Iterator[RefCocoEpisode]:
        self._require_loaded()
        supported = {"all_inside","pretarget_inside", "posttarget_inside", "all_trial", "pretarget_trial", "posttarget_trial"}
        if period not in supported:
            raise ValueError(f"Unknown RefCOCO period: {period}")
        for record in self.data:
            target_idx = self.find_target_idx(record.get("REF_WORDS", []), str(record.get("TARGET", "")))
            if target_idx < 0:
                continue
            if first_word_only and target_idx > 0:
                continue
            target_period = self.target_period(record)
            if target_period is None:
                continue
            ellipse = self.target_ellipse(record)
            selected = self.selected_fixations(record, period, ellipse)
            if selected is None or len(selected) < min_fix:
                continue
            yield RefCocoEpisode(
                patient_id=str(record.get("SUBJECT_ID")),
                study_id=str(record.get("REF_GAZE_ID")),
                ellipse=ellipse,
                fixations=selected,
                label=str(record.get("TARGET_CATEGORY") or record.get("TARGET") or "unknown").lower().strip(),
                target=str(record.get("TARGET", "")).lower().strip(),
                target_category=str(record.get("TARGET_CATEGORY") or record.get("TARGET") or "unknown").lower().strip(),
                ref_id=str(record.get("REF_ID")),
                ref_sentence=record.get("REF_SENTENCE"),
                target_word_index=target_idx,
                target_start_time=target_period[0],
                target_end_time=target_period[1],
                period=period,
                split=record.get("REFCOCO_GAZE_SPLIT"),
            )

    def data_checks(self, periods: Sequence[str], min_fix_values: Sequence[int] = (1, 2, 3, 4, 7)) -> Dict:
        self._require_loaded()
        total = len(self.data)
        target_matched = 0
        timing_covered = 0
        first_word_target = 0
        category_counts = Counter()
        for record in self.data:
            target_idx = self.find_target_idx(record.get("REF_WORDS", []), str(record.get("TARGET", "")))
            if target_idx < 0:
                continue
            target_matched += 1
            if target_idx == 0:
                first_word_target += 1
            if self.target_period(record) is not None:
                timing_covered += 1
            category_counts[str(record.get("TARGET_CATEGORY") or record.get("TARGET") or "unknown").lower().strip()] += 1

        usable = {}
        for period in periods:
            usable[period] = {}
            for min_fix in min_fix_values:
                episodes = list(self.iter_target_episodes(period=period, min_fix=min_fix, first_word_only=False))
                usable[period][str(min_fix)] = len(episodes)

        return {
            "n_records": total,
            "n_train": len(self.train_data),
            "n_val": len(self.val_data),
            "n_subjects": len({str(d.get("SUBJECT_ID")) for d in self.data}),
            "n_images": len({str(d.get("IMAGEFILE")) for d in self.data}),
            "n_target_matched": target_matched,
            "n_timing_covered": timing_covered,
            "n_first_word_target": first_word_target,
            "top_target_categories": category_counts.most_common(20),
            "usable_episodes_by_period_and_min_fix": usable,
        }

from __future__ import annotations
import string
from typing import Any, Dict, Iterator, List, Sequence, Tuple
from local_annotations import EllipseAnnotation
def normalise_token(value: object) -> str: return str(value).lower().strip(string.punctuation + " \t\n\r")
def normalise_label(label: str) -> str: return label.lower().strip()
def label_phrase_tokens(abnormality_mappings: Dict[str, Sequence[object]]) -> Dict[str, List[Tuple[str, List[str]]]]:
    out: Dict[str, List[Tuple[str, List[str]]]] = {}
    for label, phrases in abnormality_mappings.items():
        norm_label = normalise_label(label)
        out[norm_label] = []
        for phrase in phrases:
            if not phrase: continue
            tokens = [normalise_token(tok) for tok in str(phrase).split()]
            tokens = [tok for tok in tokens if tok]
            if tokens: out[norm_label].append((str(phrase).lower(), tokens))
    return out

def find_token_spans(tokens: Sequence[str], phrase_tokens: Sequence[str]) -> Iterator[Tuple[int, int]]:
    n = len(phrase_tokens)
    if n == 0: return
    for start in range(0, len(tokens) - n + 1):
        if list(tokens[start : start + n]) == list(phrase_tokens):
            yield start, start + n - 1

def iter_abnormality_mentions(word_rows: Sequence[Dict[str, Any]], ellipses: Sequence[EllipseAnnotation], abnormality_mappings: Dict[str, Sequence[object]], *, unique_label_only: bool = True) -> Iterator[Dict[str, Any]]:
    phrase_map = label_phrase_tokens(abnormality_mappings)
    tokens = [normalise_token(w.get("word", "")) for w in word_rows]
    ellipses_by_label: Dict[str, List[EllipseAnnotation]] = {}
    for ellipse in ellipses:
        for label in ellipse.labels:
            ellipses_by_label.setdefault(normalise_label(label), []).append(ellipse)
    for label, label_ellipses in ellipses_by_label.items():
        if label not in phrase_map:
            continue
        if unique_label_only and len(label_ellipses) != 1:
            continue
        if not label_ellipses:
            continue
        ellipse = label_ellipses[0]
        seen_spans = set()
        for phrase, phrase_tokens in phrase_map[label]:
            for word_start, word_end in find_token_spans(tokens, phrase_tokens):
                if (word_start, word_end) in seen_spans:
                    continue
                seen_spans.add((word_start, word_end))
                try:
                    mention_start = float(word_rows[word_start]["timestamp_start_word"])
                    mention_end = float(word_rows[word_end]["timestamp_end_word"])
                except (KeyError, TypeError, ValueError):
                    continue
                yield {"label": label, "phrase": phrase, "word_start": word_start, "word_end": word_end, "mention_start": mention_start, "mention_end": mention_end, "ellipse": ellipse}

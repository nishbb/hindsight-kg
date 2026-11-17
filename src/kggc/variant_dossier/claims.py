"""Claims: one sentence plus machine-checkable citations into stored raw records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Citation:
    record_id: str
    path: str  # dotted path into record["data"], e.g. "alleles.1.hgvs"; "" = record exists
    value: Any = None  # expected value at path (None = existence only)
    label: str = ""  # human-readable tag, e.g. "ClinVar VCV000012582 (2026-11)"
    mode: str = "eq"  # eq | contains | count_eq | ratio_of_sums
    shown: str = ""  # the exact string printed in the sentence for this value ("" = not shown)
    args: Any = (
        None  # count_eq: [list_path, field, wanted]; ratio_of_sums: [[num paths], [den paths]]
    )


@dataclass
class Claim:
    section: str
    text: str
    citations: list[Citation] = field(default_factory=list)

    def render(self) -> str:
        labels = []
        for c in self.citations:
            if c.label and c.label not in labels:
                labels.append(c.label)
        return f"{self.text} [{'; '.join(labels)}]"

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_path(data: Any, path: str) -> tuple[bool, Any]:
    """Follow a dotted path through dicts and lists. Returns (found, value)."""
    cur = data
    if path == "":
        return True, cur
    for part in path.split("."):
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return False, None
        elif isinstance(cur, dict):
            if part not in cur:
                return False, None
            cur = cur[part]
        else:
            return False, None
    return True, cur


def values_match(expected: Any, actual: Any) -> bool:
    if expected is None:
        return True
    if isinstance(expected, bool) or isinstance(actual, bool):
        return bool(expected) == bool(actual)
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float, str)):
        try:
            a = float(actual)
        except ValueError:
            return False
        return abs(a - float(expected)) <= 1e-9 + 1e-6 * abs(float(expected))
    if isinstance(expected, (list, tuple)):
        return list(expected) == list(actual) if isinstance(actual, (list, tuple)) else False
    return str(expected) == str(actual)


def check_citation(cit: Citation, record: dict | None) -> tuple[bool, str]:
    """Verify one citation against its stored raw record. Returns (ok, reason)."""
    if record is None:
        return False, "record missing"
    data = record.get("data")
    if cit.mode == "count_eq":
        list_path, fld, wanted = cit.args
        found, lst = resolve_path(data, list_path)
        if not found or not isinstance(lst, list):
            return False, f"path {list_path} missing"
        n = sum(1 for x in lst if isinstance(x, dict) and str(x.get(fld)) == str(wanted))
        return (n == cit.value, f"count {n} != {cit.value}")
    if cit.mode == "ratio_of_sums":
        num_paths, den_paths = cit.args
        tot = []
        for paths in (num_paths, den_paths):
            s = 0.0
            for p in paths:
                found, v = resolve_path(data, p)
                if not found or v is None:
                    return False, f"path {p} missing"
                s += float(v)
            tot.append(s)
        if tot[1] == 0:
            return False, "zero denominator"
        return (values_match(cit.value, tot[0] / tot[1]), f"ratio {tot[0] / tot[1]} != {cit.value}")
    found, actual = resolve_path(data, cit.path)
    if not found:
        return False, f"path {cit.path} missing"
    if cit.mode == "contains":
        return (str(cit.value) in str(actual), f"'{cit.value}' not in record value")
    return (
        values_match(cit.value, actual),
        f"record has {str(actual)[:60]!r}, claim has {cit.value!r}",
    )

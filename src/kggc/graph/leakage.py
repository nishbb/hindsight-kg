"""Temporal leakage checks for the time-T graph, features and literature items.

``check_manifest`` is run by the M2 build script's output and by the test suite. It raises
``LeakageError`` listing every violation, so a failing check names the offending source.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from kggc.data.sources import SOURCES, T_DATE


class LeakageError(AssertionError):
    pass


def source_violations(source_field: str, t_date: str = T_DATE) -> list[str]:
    """Check one ``a+b`` source string against the registry."""
    out = []
    for key in source_field.split("+"):
        src = SOURCES.get(key)
        if src is None:
            out.append(f"unknown source '{key}'")
        elif src.leaky:
            out.append(f"source '{key}' is flagged leaky")
        elif src.release_date > t_date:
            out.append(f"source '{key}' released {src.release_date} > T {t_date}")
    return out


def check_manifest(manifest: dict, t_date: str = T_DATE) -> None:
    problems: list[str] = []
    for nt, src in manifest.get("node_sources", {}).items():
        problems += [f"node type {nt}: {p}" for p in source_violations(src, t_date)]
    for et, src in manifest.get("edge_sources", {}).items():
        problems += [f"edge type {et}: {p}" for p in source_violations(src, t_date)]
    for field in ("max_clinvar_last_evaluated", "max_gwas_date_added", "max_gwas_pub_date"):
        val = manifest.get(field)
        if val and str(val) > t_date:
            problems.append(f"{field}={val} > T {t_date}")
    if problems:
        raise LeakageError("; ".join(problems))


def check_dates(dates: Iterable, what: str, t_date: str = T_DATE) -> None:
    """Assert that no dated item (edge provenance, literature record) post-dates T."""
    s = pd.to_datetime(pd.Series(list(dates)), errors="coerce")
    late = s[s > pd.Timestamp(t_date)]
    if len(late):
        raise LeakageError(f"{len(late)} {what} post-date T {t_date} (max {late.max().date()})")


def check_feature_columns(columns: Iterable[str], allowed: Iterable[str]) -> None:
    """Assert that every feature column is on the time-T allow list."""
    extra = sorted(set(columns) - set(allowed))
    if extra:
        raise LeakageError(f"feature columns not on the time-T allow list: {extra}")

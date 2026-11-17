"""Disk-cached, rate-limited HTTP client.

Every response is written to ``data/cache/<source>/<sha1>.json`` together with the
request, a UTC timestamp, and a caller-supplied source version. Re-running any script
reads the cached copy, so runs are reproducible offline once the cache is warm.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from kggc.utils import CACHE_DIR

USER_AGENT = "kggc-research/0.1 (+https://github.com/nishbb/hindsight-kg)"

# Requests per second per source (conservative versus each provider's public policy).
RATE_LIMITS = {
    "ncbi": 3.0,  # E-utilities without API key: 3/s
    "litvar": 3.0,
    "pubtator": 3.0,
    "ensembl": 10.0,  # Ensembl REST: 15/s
    "myvariant": 5.0,
    "gnomad": 1.0,
    "gwas": 5.0,
    "civic": 2.0,
    "default": 2.0,
}


class _RateLimiter:
    def __init__(self) -> None:
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, source: str) -> None:
        rate = RATE_LIMITS.get(source, RATE_LIMITS["default"])
        with self._lock:
            now = time.monotonic()
            last = self._last.get(source, 0.0)
            delay = 1.0 / rate - (now - last)
            if delay > 0:
                time.sleep(delay)
            self._last[source] = time.monotonic()


_LIMITER = _RateLimiter()


def _key(method: str, url: str, params: Any, body: Any) -> str:
    blob = json.dumps([method, url, params, body], sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()


class CachedClient:
    def __init__(self, cache_dir: Path | None = None, offline: bool = False) -> None:
        self.cache_dir = Path(cache_dir or CACHE_DIR)
        self.offline = offline
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT

    def _path(self, source: str, key: str) -> Path:
        return self.cache_dir / source / key[:2] / f"{key}.json"

    def request(
        self,
        source: str,
        url: str,
        *,
        method: str = "GET",
        params: dict | None = None,
        json_body: Any = None,
        data: Any = None,
        headers: dict | None = None,
        source_version: str = "unspecified",
        expect_json: bool = True,
        retries: int = 4,
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        """Return a cache record: {request, retrieved_at, source_version, status, body}."""
        body_for_key = json_body if json_body is not None else data
        key = _key(method, url, params, body_for_key)
        path = self._path(source, key)
        if path.exists():
            with open(path) as fh:
                return json.load(fh)
        if self.offline:
            raise FileNotFoundError(f"offline mode and no cache for {source} {url} {params}")

        last_err: Exception | None = None
        for attempt in range(retries):
            _LIMITER.wait(source)
            try:
                resp = self.session.request(
                    method,
                    url,
                    params=params,
                    json=json_body,
                    data=data,
                    headers=headers,
                    timeout=timeout,
                )
            except requests.RequestException as err:  # network hiccup
                last_err = err
                time.sleep(2**attempt)
                continue
            if resp.status_code in (429, 500, 502, 503, 504):
                retry_after = float(resp.headers.get("Retry-After", 2**attempt))
                time.sleep(min(retry_after, 30))
                last_err = RuntimeError(f"HTTP {resp.status_code}")
                continue
            try:
                body = resp.json() if expect_json else resp.text
            except ValueError:
                body = resp.text
            record = {
                "request": {
                    "method": method,
                    "url": url,
                    "params": params,
                    "body": body_for_key,
                },
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "source": source,
                "source_version": source_version,
                "status": resp.status_code,
                "body": body,
            }
            # Do not cache transient client errors other than 400/404 (stable "no such record").
            if resp.status_code < 400 or resp.status_code in (400, 404):
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                with open(tmp, "w") as fh:
                    json.dump(record, fh)
                tmp.replace(path)
            return record
        raise RuntimeError(f"request failed after {retries} tries: {url}: {last_err}")

    def get_json(self, source: str, url: str, **kw: Any) -> Any:
        return self.request(source, url, **kw)["body"]


_DEFAULT: CachedClient | None = None


def default_client() -> CachedClient:
    global _DEFAULT
    if _DEFAULT is None:
        import os

        _DEFAULT = CachedClient(offline=os.environ.get("KGGC_OFFLINE") == "1")
    return _DEFAULT

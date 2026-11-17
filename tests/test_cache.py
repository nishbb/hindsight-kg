import json

import pytest

from kggc.data.cache import CachedClient


class _Resp:
    status_code = 200
    headers: dict = {}

    def json(self):
        return {"ok": 1}

    text = '{"ok": 1}'


def test_cache_roundtrip(tmp_path, monkeypatch):
    c = CachedClient(cache_dir=tmp_path)
    calls = []

    def fake(method, url, **kw):
        calls.append(url)
        return _Resp()

    monkeypatch.setattr(c.session, "request", fake)
    r1 = c.request("default", "https://example.org/x", params={"a": 1}, source_version="v1")
    r2 = c.request("default", "https://example.org/x", params={"a": 1})
    assert r1["body"] == {"ok": 1} and r2["source_version"] == "v1"
    assert len(calls) == 1  # second call served from disk
    files = list(tmp_path.rglob("*.json"))
    assert len(files) == 1
    rec = json.loads(files[0].read_text())
    assert "retrieved_at" in rec and rec["request"]["params"] == {"a": 1}


def test_offline_raises(tmp_path):
    c = CachedClient(cache_dir=tmp_path, offline=True)
    with pytest.raises(FileNotFoundError):
        c.request("default", "https://example.org/y")

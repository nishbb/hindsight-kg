"""Tool-using LLM audit agent (OpenAI-compatible API, e.g. NVIDIA NIM).

The agent audits one ClinVar variant-disease assertion as it stood in January 2020 and returns
``{"verdict", "confidence", "rationale", "evidence_ids"}``. A hard tool-call budget is enforced;
when it runs out, the agent must answer with what it has.

Every LLM response is cached on disk keyed by (model, messages, tools), so reruns are exact and
free. API keys come only from environment variables.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI

from kggc.agent.tools import TimeTTools, openai_tools, run_tool
from kggc.utils import CACHE_DIR

VERDICTS = ("keep", "downgrade", "upgrade", "needs_review")

SYSTEM = """You audit genotype-phenotype knowledge-graph edges for a curator. It is January 2020.
You are given one ClinVar assertion linking a variant to a disease as it stood in the 2020-01 release.
Decide what should happen to it:
- keep: the current classification is well supported and will not need to change;
- downgrade: the variant is classified pathogenic or likely pathogenic, but the evidence is weak or contradicted, so it should become uncertain, conflicting or benign;
- upgrade: the variant is a variant of uncertain significance (VUS), but the evidence supports pathogenic or likely pathogenic;
- needs_review: the evidence is too thin or contradictory to decide.
Only 'downgrade' applies to pathogenic edges and only 'upgrade' applies to VUS edges.
Use the tools to gather evidence. All tools return information available in January 2020 only. Do not rely on anything you may remember about later reclassifications.
When done, reply with ONLY a JSON object:
{"verdict": "keep|downgrade|upgrade|needs_review", "confidence": <number 0 to 1, your probability that the verdict is correct>, "rationale": "<2 to 4 sentences; cite evidence ids in square brackets, e.g. [GNN:VCV123]>", "evidence_ids": ["<ids you relied on>"]}
Cite evidence ids exactly as they appear in the "evidence_id" fields of tool results (or PMIDs as PMID:<number>)."""


@dataclass
class AgentConfig:
    model: str = os.environ.get("KGGC_LLM_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
    base_url: str = os.environ.get("KGGC_LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    api_key_env: str = os.environ.get("KGGC_LLM_API_KEY_ENV", "NVIDIA_API_KEY")
    tools: list[str] = field(
        default_factory=lambda: ["dossier", "literature_search", "kg_neighborhood"]
    )
    max_tool_calls: int = 6
    temperature: float = 0.0
    max_tokens: int = 1500
    seed: int = 0
    enable_thinking: bool = (
        False  # Nemotron chat-template switch; off for all configs (cost, consistency)
    )


class CachedLLM:
    # One shared limiter for all threads: hosted free tiers allow a fixed number of requests
    # per minute, and uncoordinated retries from several threads collapse throughput.
    _lock = threading.Lock()
    _next_ok = 0.0
    MIN_INTERVAL = float(os.environ.get("KGGC_LLM_MIN_INTERVAL", "2.0"))  # seconds between requests

    @classmethod
    def _throttle(cls, penalty: float = 0.0) -> None:
        with cls._lock:
            now = time.monotonic()
            if penalty:
                cls._next_ok = max(cls._next_ok, now + penalty)
                return
            wait = cls._next_ok - now
            cls._next_ok = max(now, cls._next_ok) + cls.MIN_INTERVAL
        if wait > 0:
            time.sleep(wait)

    def __init__(self, cfg: AgentConfig, cache_dir: Path | None = None) -> None:
        self.cfg = cfg
        self.dir = Path(cache_dir or CACHE_DIR / "llm")
        self.dir.mkdir(parents=True, exist_ok=True)
        key = os.environ.get(cfg.api_key_env)
        # short timeout, no hidden SDK retries: stalled requests fail fast and our loop retries
        self.client = (
            OpenAI(base_url=cfg.base_url, api_key=key, timeout=90.0, max_retries=0) if key else None
        )

    def chat(
        self, messages: list[dict], tools: list[dict] | None, tool_choice: str = "auto"
    ) -> dict:
        blob = json.dumps(
            [
                self.cfg.model,
                messages,
                tools,
                tool_choice,
                self.cfg.temperature,
                self.cfg.max_tokens,
                self.cfg.enable_thinking,
            ],
            sort_keys=True,
        )
        path = self.dir / f"{hashlib.sha1(blob.encode()).hexdigest()}.json"
        if path.exists():
            return json.loads(path.read_text())
        if self.client is None:
            raise RuntimeError(f"{self.cfg.api_key_env} is not set and the response is not cached")
        kw = dict(
            model=self.cfg.model,
            messages=messages,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            seed=self.cfg.seed,
            extra_body={"chat_template_kwargs": {"enable_thinking": self.cfg.enable_thinking}},
        )
        if tools:
            kw.update(tools=tools, tool_choice=tool_choice)
        for attempt in range(15):
            self._throttle()
            try:
                r = self.client.chat.completions.create(**kw)
                break
            except Exception as err:  # 429 rate limit, 503 overload, transient network errors
                if attempt == 14:
                    raise
                status = getattr(err, "status_code", None)
                # a rate-limit or overload pauses every thread for a minute
                self._throttle(penalty=60.0 if status in (429, 503) else 2.0 ** min(attempt, 5))
        m = r.choices[0].message
        out = {
            "content": m.content or "",
            "tool_calls": [
                {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                for tc in (m.tool_calls or [])
            ],
            "finish_reason": r.choices[0].finish_reason,
            "usage": {
                "prompt_tokens": r.usage.prompt_tokens,
                "completion_tokens": r.usage.completion_tokens,
            },
        }
        path.write_text(json.dumps(out))
        return out


def parse_verdict(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return {
            "verdict": "needs_review",
            "confidence": 0.0,
            "rationale": "unparseable output",
            "evidence_ids": [],
            "parse_error": True,
        }
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {
            "verdict": "needs_review",
            "confidence": 0.0,
            "rationale": "unparseable output",
            "evidence_ids": [],
            "parse_error": True,
        }
    v = str(d.get("verdict", "")).strip().lower().replace(" ", "_")
    d["verdict"] = v if v in VERDICTS else "needs_review"
    try:
        d["confidence"] = min(1.0, max(0.0, float(d.get("confidence", 0))))
    except (TypeError, ValueError):
        d["confidence"] = 0.0
    d.setdefault("rationale", "")
    d.setdefault("evidence_ids", [])
    return d


def audit_edge(
    llm: CachedLLM, tools: TimeTTools, variation_id: int, budget: int, tool_names: list[str]
) -> dict:
    rec = tools.clinvar_record(variation_id)
    is_plp = rec["clnsig"] in ("Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic")
    allowed = (
        ("keep", "downgrade", "needs_review") if is_plp else ("keep", "upgrade", "needs_review")
    )
    task = (
        f"Audit this edge. ClinVar VariationID {variation_id}: {rec['name']} in {rec['gene_symbol']}, "
        f"classified '{rec['clnsig']}' for {rec['condition_name']} ({rec['mondo_id']}). "
        f"Review status at T: '{rec['review_status']}', {rec['n_submitters']} submitter(s). "
        f"Allowed verdicts for this edge: {', '.join(allowed)}. "
        f"You may call at most {budget} tools. [ClinVar2020-01:VCV{variation_id}]"
    )
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
    spec = openai_tools(tool_names) if tool_names and budget > 0 else None
    seen_ids = {rec["evidence_id"]}
    calls, ptok, ctok, trace, retries = 0, 0, 0, [], 0
    for _turn in range(budget + 4):
        force = calls >= budget or spec is None
        r = llm.chat(msgs, None if force else spec)
        ptok += r["usage"]["prompt_tokens"]
        ctok += r["usage"]["completion_tokens"]
        if r["tool_calls"] and not force:
            msgs.append(
                {
                    "role": "assistant",
                    "content": r["content"],
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": tc["arguments"]},
                        }
                        for tc in r["tool_calls"]
                    ],
                }
            )
            for tc in r["tool_calls"]:
                if calls >= budget:
                    res = json.dumps({"error": "tool budget exhausted; answer now"})
                else:
                    try:
                        args = json.loads(tc["arguments"] or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    res = run_tool(tools, tc["name"], args)
                    calls += 1
                    trace.append({"tool": tc["name"], "args": args})
                    for eid in re.findall(r'"evidence_id": "([^"]+)"', res):
                        seen_ids.add(eid)
                    for pm in re.findall(r"PMID:\d+", res):
                        seen_ids.add(pm)
                msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": res})
            if calls >= budget:
                msgs.append(
                    {
                        "role": "user",
                        "content": "Tool budget used up. Reply now with the JSON verdict only.",
                    }
                )
            continue
        out = parse_verdict(r["content"])
        if out.get("parse_error") and retries < 1:
            retries += 1
            msgs.append({"role": "assistant", "content": r["content"]})
            msgs.append({"role": "user", "content": "Reply with the JSON object only."})
            continue
        break
    out["invalid_verdict"] = out["verdict"] not in allowed
    if out["invalid_verdict"]:
        out["raw_verdict"], out["verdict"] = out["verdict"], "needs_review"
    cited = set(re.findall(r"\[([^\[\]]+)\]", out.get("rationale", ""))) | set(
        map(str, out.get("evidence_ids", []))
    )
    cited = {c.strip() for x in cited for c in re.split(r"[;,]", x) if c.strip()}
    seen_l = {x.lower() for x in seen_ids}
    out.update(
        variation_id=variation_id,
        tool_calls=calls,
        prompt_tokens=ptok,
        completion_tokens=ctok,
        trace=trace,
        n_cited=len(cited),
        n_cited_unsupported=len([c for c in cited if c.lower() not in seen_l]),
    )
    return out

import json

import pandas as pd

from kggc.agent.agent import audit_edge, parse_verdict
from kggc.agent.tools import TOOL_SPECS, openai_tools


class FakeTools:
    """Minimal stand-in for TimeTTools: records calls, returns evidence ids."""

    def __init__(self, clnsig="Pathogenic"):
        self.clnsig = clnsig
        self.calls = []

    def clinvar_record(self, vid):
        return {
            "evidence_id": f"ClinVar2020-01:VCV{vid}",
            "name": "NM_1(G):c.1A>G",
            "gene_symbol": "G",
            "clnsig": self.clnsig,
            "condition_name": "Disease",
            "mondo_id": "MONDO:1",
            "review_status": "no assertion criteria provided",
            "n_submitters": 1,
        }

    def gnn_score(self, variation_id):
        self.calls.append("gnn_score")
        return {"evidence_id": f"GNN:VCV{variation_id}", "conformal_set_at_alpha_0.1": "blb"}

    def dossier(self, variation_id):
        self.calls.append("dossier")
        return {"evidence_id": f"DOSSIER2020:VCV{variation_id}", "top_pmids": ["PMID:123"]}


class FakeLLM:
    """Scripted responses: keeps calling tools until told the budget is gone, then answers."""

    def __init__(self, final: dict | str, tool="gnn_score", greedy=True):
        self.final, self.tool, self.greedy = final, tool, greedy
        self.n = 0
        self.saw_no_tools = False

    def chat(self, messages, tools, tool_choice="auto"):
        self.n += 1
        if tools is None:
            self.saw_no_tools = True
        if tools and (self.greedy or self.n == 1):
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": f"t{self.n}",
                        "name": self.tool,
                        "arguments": json.dumps({"variation_id": 7}),
                    }
                ],
                "finish_reason": "tool_calls",
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            }
        text = self.final if isinstance(self.final, str) else json.dumps(self.final)
        return {
            "content": text,
            "tool_calls": [],
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }


def test_budget_is_enforced_and_answer_forced():
    tools, llm = FakeTools(), FakeLLM(
        {
            "verdict": "downgrade",
            "confidence": 0.8,
            "rationale": "weak [GNN:VCV7]",
            "evidence_ids": ["GNN:VCV7"],
        }
    )
    r = audit_edge(llm, tools, 7, budget=3, tool_names=["gnn_score"])
    assert r["tool_calls"] == 3 and len(tools.calls) == 3
    assert llm.saw_no_tools  # after the budget the model is called without tools
    assert r["verdict"] == "downgrade" and r["n_cited_unsupported"] == 0


def test_zero_budget_means_no_tools():
    tools, llm = FakeTools(), FakeLLM(
        {"verdict": "keep", "confidence": 0.6, "rationale": "", "evidence_ids": []}
    )
    r = audit_edge(llm, tools, 7, budget=0, tool_names=["gnn_score"])
    assert r["tool_calls"] == 0 and tools.calls == []


def test_invalid_verdict_for_edge_type_becomes_needs_review():
    # 'upgrade' is not allowed for a pathogenic edge
    tools, llm = FakeTools("Pathogenic"), FakeLLM(
        {"verdict": "upgrade", "confidence": 0.9, "rationale": "", "evidence_ids": []}, greedy=False
    )
    r = audit_edge(llm, tools, 7, budget=2, tool_names=["gnn_score"])
    assert r["invalid_verdict"] and r["verdict"] == "needs_review" and r["raw_verdict"] == "upgrade"
    # 'downgrade' is not allowed for a VUS edge
    tools, llm = FakeTools("Uncertain significance"), FakeLLM(
        {"verdict": "downgrade", "confidence": 0.9, "rationale": "", "evidence_ids": []},
        greedy=False,
    )
    assert (
        audit_edge(llm, tools, 7, budget=2, tool_names=["gnn_score"])["verdict"] == "needs_review"
    )


def test_unsupported_citations_are_counted():
    tools = FakeTools()
    llm = FakeLLM(
        {
            "verdict": "keep",
            "confidence": 0.7,
            "rationale": "see [GNN:VCV7] and [PMID:999]",
            "evidence_ids": ["dossier2020:vcv7"],
        },
        tool="dossier",
        greedy=False,
    )
    r = audit_edge(llm, tools, 7, budget=1, tool_names=["dossier"])
    # PMID:999 was never returned by a tool; GNN:VCV7 was not called; the dossier id matches case-insensitively
    assert r["n_cited_unsupported"] == 2 and r["n_cited"] == 3


def test_parse_verdict_robustness():
    assert (
        parse_verdict('noise {"verdict": "Needs Review", "confidence": "0.4"} tail')["verdict"]
        == "needs_review"
    )
    assert parse_verdict("no json here")["parse_error"]
    assert parse_verdict('{"verdict": "keep", "confidence": 7}')["confidence"] == 1.0


def test_unparseable_answer_gets_one_retry():
    tools = FakeTools()

    class Flaky(FakeLLM):
        def chat(self, messages, tools_, tool_choice="auto"):
            self.n += 1
            if self.n == 1:
                return {
                    "content": "I think keep",
                    "tool_calls": [],
                    "finish_reason": "stop",
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            return {
                "content": '{"verdict": "keep", "confidence": 0.5}',
                "tool_calls": [],
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

    r = audit_edge(Flaky(None), tools, 7, budget=2, tool_names=["gnn_score"])
    assert r["verdict"] == "keep"


def test_tool_specs_are_valid_openai_functions():
    specs = openai_tools(list(TOOL_SPECS))
    assert {s["function"]["name"] for s in specs} == set(TOOL_SPECS)
    for s in specs:
        assert s["type"] == "function" and s["function"]["parameters"]["type"] == "object"


def test_gnn_only_rule():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "m5", Path(__file__).parents[1] / "scripts" / "m5_audit.py"
    )
    m5 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m5)

    def row(pool, s):
        return pd.Series({"pool": pool, "conformal_set": s, "p_plp": 0.02, "p_blb": 0.6})

    assert m5.gnn_only(row("plp", "blb"))[0] == "downgrade"
    assert m5.gnn_only(row("plp", "plp"))[0] == "keep"
    assert m5.gnn_only(row("vus", "plp"))[0] == "upgrade"
    assert m5.gnn_only(row("vus", "both"))[0] == "needs_review"

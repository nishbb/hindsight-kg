"""M5: GNN-guided audit agent and ablations (H2).

Usage: python scripts/m5_audit.py [--per-class 80] [--threads 6] [--smoke]
Evaluation sample: stratified, --per-class edges from each of downgraded, stable_pathogenic,
upgraded, stable_vus (balanced, so accuracy is a balanced accuracy over the four outcomes).
Truth: downgraded -> downgrade, upgraded -> upgrade, stable_* -> keep.

Configurations:
  a  GNN only (conformal rule, no LLM)
  b0 LLM alone, no tools
  b  LLM agent without GNN tools (dossier, literature_search, kg_neighborhood), budget 6
  c  LLM agent with GNN tools added, budget 6
  d  as c, budget allocated by conformal confidence (3 calls if the set is a single class,
     more otherwise, chosen so the mean budget over the sample is 6)
Also writes results/m5/audit_queue.csv: the 500 lowest-trust P/LP edges and 500 highest-scoring
VUS pairs by the conformal p-values, the queue the agent would work through in production.
"""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from kggc.agent.agent import AgentConfig, CachedLLM, audit_edge
from kggc.agent.tools import TimeTTools
from kggc.eval.metrics import ece
from kggc.utils import RESULTS_DIR, seed_everything, write_json

TRUTH = {
    "downgraded": "downgrade",
    "upgraded": "upgrade",
    "stable_pathogenic": "keep",
    "stable_vus": "keep",
}
BASE_TOOLS = ["dossier", "literature_search", "kg_neighborhood"]
GNN_TOOLS = BASE_TOOLS + ["gnn_score", "gnn_explain"]


def gnn_only(row) -> tuple[str, float]:
    s, plp_pool = row["conformal_set"], row["pool"] == "plp"
    if plp_pool:
        if s == "blb":
            return "downgrade", 1 - row["p_plp"]
        if s == "plp":
            return "keep", 1 - row["p_blb"]
    else:
        if s == "plp":
            return "upgrade", 1 - row["p_blb"]
        if s == "blb":
            return "keep", 1 - row["p_plp"]
    return "needs_review", 0.5


def metrics(df: pd.DataFrame, name: str) -> dict:
    y, p = df["truth"], df["verdict"]
    correct = (y == p).astype(int)
    decided = p != "needs_review"
    out = {
        "config": name,
        "n": len(df),
        "accuracy": float(correct.mean()),
        "coverage": float(decided.mean()),
        "accuracy_on_decided": float(correct[decided].mean()) if decided.any() else float("nan"),
        "macro_f1": float(
            f1_score(
                y, p, labels=["keep", "downgrade", "upgrade"], average="macro", zero_division=0
            )
        ),
        "ece": ece(correct.values, df["confidence"].values),
        "tool_calls_per_edge": float(df.get("tool_calls", pd.Series(0, index=df.index)).mean()),
        "prompt_tokens_per_edge": float(
            df.get("prompt_tokens", pd.Series(0, index=df.index)).mean()
        ),
        "completion_tokens_per_edge": float(
            df.get("completion_tokens", pd.Series(0, index=df.index)).mean()
        ),
        "invalid_verdict_rate": (
            float(df["invalid_verdict"].mean()) if "invalid_verdict" in df else 0.0
        ),
        "parse_failure_rate": (
            float(df["parse_error"].fillna(False).astype(bool).mean())
            if "parse_error" in df
            else 0.0
        ),
        "unsupported_citation_rate": (
            float(df["n_cited_unsupported"].sum() / max(1, df["n_cited"].sum()))
            if "n_cited" in df
            else float("nan")
        ),
    }
    for pool in ("plp", "vus"):
        s = df[df["pool"] == pool]
        out[f"accuracy_{pool}_pool"] = float((s["truth"] == s["verdict"]).mean())
        pos = "downgrade" if pool == "plp" else "upgrade"
        tp = ((s["truth"] == pos) & (s["verdict"] == pos)).sum()
        out[f"recall_{pos}"] = float(tp / max(1, (s["truth"] == pos).sum()))
        out[f"precision_{pos}"] = float(tp / max(1, (s["verdict"] == pos).sum()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=80)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="3 edges per class, configs b0 and c only")
    ap.add_argument("--configs", default="a,b0,b,c,d")
    ap.add_argument("--model", default=None, help="override KGGC_LLM_MODEL")
    args = ap.parse_args()
    seed_everything(0)
    out = RESULTS_DIR / ("m5_smoke" if args.smoke else "m5")
    if args.model and args.smoke:
        out = out.with_name(out.name + "_" + args.model.split("/")[-1])
    out.mkdir(parents=True, exist_ok=True)
    tools = TimeTTools("rgcn")

    sc = tools.scores.reset_index()
    sc = sc[sc["edge_type"].isin(["plp", "vus"])]
    lab = tools.records.reset_index()[["variation_id", "mondo_id", "label", "gene_symbol", "name"]]
    lab = lab[lab["label"].notna()].merge(
        sc[["variation_id", "mondo_id", "score", "p_plp", "p_blb", "conformal_set", "edge_type"]],
        on=["variation_id", "mondo_id"],
    )
    lab["pool"] = np.where(lab["label"].isin(["downgraded", "stable_pathogenic"]), "plp", "vus")

    q1 = lab[lab["pool"] == "plp"].sort_values("p_plp").head(500)
    q2 = lab[lab["pool"] == "vus"].sort_values("p_blb").head(500)
    pd.concat([q1.assign(queue="lowest_trust_plp"), q2.assign(queue="highest_scoring_vus")])[
        [
            "queue",
            "variation_id",
            "name",
            "gene_symbol",
            "mondo_id",
            "score",
            "p_plp",
            "p_blb",
            "conformal_set",
        ]
    ].to_csv(RESULTS_DIR / "m5_audit_queue.csv", index=False)

    n = 3 if args.smoke else args.per_class
    rng = np.random.default_rng(0)
    if not args.smoke:  # edges used to choose the model and prompt are not evaluated
        smoke = RESULTS_DIR / "m5_smoke" / "eval_sample.csv"
        if smoke.exists():
            lab = lab[~lab["variation_id"].isin(pd.read_csv(smoke)["variation_id"])]
        rng = np.random.default_rng(1)
    sample = pd.concat(
        [g.iloc[rng.choice(len(g), n, replace=False)] for _, g in lab.groupby("label")],
        ignore_index=True,
    )
    sample["truth"] = sample["label"].map(TRUTH)
    sample = sample.sort_values("variation_id").reset_index(drop=True)
    sample[["variation_id", "mondo_id", "label", "truth", "pool", "conformal_set"]].to_csv(
        out / "eval_sample.csv", index=False
    )

    # conformal budget for config d: singleton sets get 3 calls, the rest share the remainder
    confident = sample["conformal_set"].isin(["plp", "blb"])
    lo = 3
    hi = int(round((6 * len(sample) - lo * confident.sum()) / max(1, (~confident).sum())))
    hi = max(hi, lo)
    budgets_d = np.where(confident, lo, hi)
    write_json(
        {
            "confident_fraction": float(confident.mean()),
            "budget_confident": lo,
            "budget_uncertain": hi,
            "mean_budget": float(budgets_d.mean()),
        },
        out / "budget_d.json",
    )

    lock = threading.Lock()

    class LockedTools:
        def __getattr__(self, name):
            fn = getattr(tools, name)

            def wrapped(*a, **k):
                with lock:
                    return fn(*a, **k)

            return wrapped

        def clinvar_record(self, vid):
            return tools.clinvar_record(vid)

    lt = LockedTools()
    print("warming dossier cache ...", flush=True)
    for vid in sample["variation_id"]:
        tools.dossier(int(vid))

    configs = [c for c in args.configs.split(",") if c]
    if args.smoke:
        configs = ["b0", "c"]
    results = {}
    for cfg_name in configs:
        if cfg_name == "a":
            v = sample.apply(gnn_only, axis=1, result_type="expand")
            df = sample.assign(
                verdict=v[0], confidence=v[1], tool_calls=0, prompt_tokens=0, completion_tokens=0
            )
        else:
            names = {"b0": [], "b": BASE_TOOLS, "c": GNN_TOOLS, "d": GNN_TOOLS}[cfg_name]
            budgets = (
                budgets_d if cfg_name == "d" else np.full(len(sample), 0 if cfg_name == "b0" else 6)
            )
            acfg = AgentConfig(tools=names)
            if args.model:
                acfg.model = args.model
            llm = CachedLLM(acfg)

            def one(i, llm=llm, budgets=budgets, names=names):
                r = audit_edge(llm, lt, int(sample.loc[i, "variation_id"]), int(budgets[i]), names)
                r["budget"] = int(budgets[i])
                return r

            with ThreadPoolExecutor(args.threads) as ex:
                rows = list(ex.map(one, range(len(sample))))
            df = sample.merge(pd.DataFrame(rows).drop(columns=["trace"]), on="variation_id")
            (out / f"traces_{cfg_name}.jsonl").write_text(
                "\n".join(json.dumps(r, default=str) for r in rows) + "\n"
            )
        df.to_csv(out / f"verdicts_{cfg_name}.csv", index=False)
        results[cfg_name] = metrics(df, cfg_name)
        print(cfg_name, json.dumps(results[cfg_name]), flush=True)
    pd.DataFrame(results.values()).to_csv(out / "ablation_summary.csv", index=False)
    write_json(
        {
            "model": AgentConfig().model,
            "base_url": AgentConfig().base_url,
            "n_edges": len(sample),
            "results": results,
        },
        out / "summary.json",
    )


if __name__ == "__main__":
    main()

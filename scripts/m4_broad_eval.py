"""M4 broader evaluation: citation validity, claim-record agreement and conflict recall.

Sets (seeded, one variant per rsID):
  expert   100 ClinVar variants with review status 'reviewed by expert panel'
  gwas     100 GWAS Catalog lead SNPs (p <= 5e-8, 2019-12-16 release)
  conflict 100 ClinVar variants whose aggregate classification is 'Conflicting ...'
The first two are the 200 variants named in the brief. The third exists because expert-panel
records almost never carry ClinVar's conflicting flag, so recall needs its own positives.

Conflict recall is reported two ways: from the dossier's conflict list (which also reads the
aggregate flag, so it is near-trivial) and from individual submissions only (independent of
the flag, the honest number).
"""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd

from kggc.utils import RAW_DIR, RESULTS_DIR, write_json
from kggc.variant_dossier.build import DossierBuilder, render_text
from kggc.variant_dossier.clinvar_local import DB_PATH, LATEST
from kggc.variant_dossier.verify import check_rendered_text, verify_dossier

N = 100


def sample_sets(seed: int = 0) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    con = sqlite3.connect(DB_PATH)
    q = "SELECT rsid, COUNT(DISTINCT allele_id) n FROM variants WHERE snapshot=? AND {} GROUP BY rsid HAVING n=1"
    exp = pd.read_sql(q.format("review_status='reviewed by expert panel'"), con, params=(LATEST,))[
        "rsid"
    ]
    conf = pd.read_sql(
        q.format("clnsig LIKE 'Conflicting%' AND origin_simple='germline'"), con, params=(LATEST,)
    )["rsid"]
    g = pd.read_csv(
        RAW_DIR / "gwas" / "gwas-catalog-associations_ontology-annotated_2019-12-16.tsv",
        sep="\t",
        dtype=str,
        usecols=["SNP_ID_CURRENT", "P-VALUE"],
        na_filter=False,
    )
    g = g[
        (pd.to_numeric(g["P-VALUE"], errors="coerce") <= 5e-8)
        & g["SNP_ID_CURRENT"].str.fullmatch(r"\d+")
    ]
    gw = pd.Series(sorted(set("rs" + g["SNP_ID_CURRENT"])))
    pick = lambda s: sorted(rng.choice(sorted(set(s)), N, replace=False).tolist())  # noqa: E731
    return {"expert": pick(exp), "gwas": pick(gw), "conflict": pick(conf)}


def main() -> None:
    out = RESULTS_DIR / "m4_broad"
    out.mkdir(parents=True, exist_ok=True)
    sets = sample_sets()
    write_json(sets, out / "sampled_rsids.json")
    b = DossierBuilder()
    rows = []
    for name, rsids in sets.items():
        for rs in rsids:
            r = {"set": name, "rsid": rs}
            try:
                d = b.build(rs, "latest")
                v = verify_dossier(d)
                t = check_rendered_text(render_text(d))
                subs = [k for k in d["conflicts"] if k["type"] == "conflicting_submitters"]
                sub_level = any(
                    sum(1 for c in ("PLP", "VUS", "BLB") if k["classes"].get(c)) > 1 for k in subs
                )
                r.update(
                    status="ok",
                    n_claims=v["n_claims"],
                    n_citations=v["n_citations"],
                    n_citations_valid=v["n_citations_valid"],
                    n_shown=v["n_shown_values"],
                    n_shown_ok=v["n_shown_values_in_text"],
                    n_uncited=v["n_uncited_claims"],
                    n_sentences=t["n_sentences"],
                    n_sentences_uncited=t["n_without_citation"],
                    conflict_flagged=bool(subs),
                    conflict_from_submissions=sub_level,
                    clinvar_flag=any(
                        str(c["classification"]).startswith("Conflicting") for c in d["clinvar"]
                    ),
                    has_bottom_line=bool(d["bottom_line"]["text"]),
                    confidence=d["bottom_line"]["confidence"],
                )
            except Exception as err:
                r.update(status="failed", error=f"{type(err).__name__}: {err}"[:200])
            rows.append(r)
            print(name, rs, r["status"], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_variant.csv", index=False)
    ok = df[df["status"] == "ok"]
    summ = {}
    for name in ["expert", "gwas", "expert+gwas (the 200)", "conflict"]:
        s = ok[ok["set"].isin(["expert", "gwas"])] if "+" in name else ok[ok["set"] == name]
        n_all = (df["set"].isin(["expert", "gwas"]) if "+" in name else df["set"] == name).sum()
        summ[name] = {
            "n_variants": int(n_all),
            "n_complete": int(len(s)),
            "n_claims": int(s["n_claims"].sum()),
            "n_citations": int(s["n_citations"].sum()),
            "citation_validity_rate": float(s["n_citations_valid"].sum() / s["n_citations"].sum()),
            "claim_record_agreement_rate": float(s["n_shown_ok"].sum() / s["n_shown"].sum()),
            "uncited_sentences": int(s["n_sentences_uncited"].sum() + s["n_uncited"].sum()),
            "conflict_flag_rate": float(s["conflict_flagged"].mean()),
        }
    pos = ok[(ok["set"] == "conflict") & ok["clinvar_flag"]]
    summ["conflict_detection"] = {
        "n_positives_with_ClinVar_conflicting_flag": int(len(pos)),
        "recall_dossier_conflict_list": float(pos["conflict_flagged"].mean()),
        "recall_from_submissions_only": float(pos["conflict_from_submissions"].mean()),
        "false_alarm_rate_on_expert_panel_set_submissions_only": float(
            ok.loc[ok["set"] == "expert", "conflict_from_submissions"].mean()
        ),
    }
    write_json(summ, out / "summary.json")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()

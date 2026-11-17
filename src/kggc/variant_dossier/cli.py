"""kggc-dossier: per-variant evidence dossiers with cited plain-text summaries.

Example:
  kggc-dossier --rsids rs334,rs429358,rs7412 --as-of latest --out out/
  kggc-dossier --rsids rs334 --as-of 2020-01-01 --out out_2020/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kggc.variant_dossier.build import DossierBuilder, render_text
from kggc.variant_dossier.haplotype import detect_haplotypes, render_haplotype
from kggc.variant_dossier.verify import check_rendered_text, verify_dossier


def run(rsids: list[str], as_of: str, out: Path, log=print) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    builder = DossierBuilder()
    dossiers, summary = [], []
    for rs in rsids:
        try:
            d = builder.build(rs, as_of)
        except Exception as err:  # a failed variant is reported, the batch continues
            log(f"[dossier] {rs}: FAILED {type(err).__name__}: {err}")
            summary.append(
                {"rsid": rs, "status": "failed", "error": f"{type(err).__name__}: {err}"}
            )
            continue
        d["verification"] = verify_dossier(d)
        text = render_text(d)
        d["verification"]["rendered_text"] = check_rendered_text(text)
        (out / f"{rs}.json").write_text(json.dumps(d, indent=1, default=str))
        (out / f"{rs}.txt").write_text(text)
        dossiers.append(d)
        v = d["verification"]
        summary.append(
            {
                "rsid": rs,
                "status": "ok",
                "current": d["rsid_current"],
                "n_claims": v["n_claims"],
                "citation_validity_rate": v["citation_validity_rate"],
                "claim_record_agreement_rate": v["claim_record_agreement_rate"],
                "n_uncited_claims": v["n_uncited_claims"],
                "temporal_violations": len(v["temporal_violations"]),
                "n_conflicts": len(d["conflicts"]),
                "bottom_line": d["bottom_line"]["text"],
            }
        )
        log(
            f"[dossier] {rs}: {v['n_claims']} claims, citation validity {v['citation_validity_rate']:.3f}"
        )
    haps = detect_haplotypes(dossiers) if len(dossiers) > 1 else []
    for h in haps:
        name = "+".join(h["rsids"])
        (out / f"haplotype_{name}.json").write_text(json.dumps(h, indent=1, default=str))
        (out / f"haplotype_{name}.txt").write_text(render_haplotype(h))
        log(
            f"[dossier] haplotype pair detected: {name} ({h['gene']}, confidence {h['confidence']})"
        )
    (out / "summary.json").write_text(
        json.dumps(
            {"as_of": as_of, "variants": summary, "haplotypes": [h["rsids"] for h in haps]},
            indent=1,
        )
    )
    return {"dossiers": dossiers, "haplotypes": haps, "summary": summary}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--rsids",
        required=True,
        help="comma-separated rsIDs, or a path to a file with one per line",
    )
    ap.add_argument("--as-of", default="latest", help="'latest' or an ISO date such as 2020-01-01")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    p = Path(args.rsids)
    rsids = [
        x.strip()
        for x in (p.read_text().split() if p.exists() else args.rsids.split(","))
        if x.strip()
    ]
    run(rsids, args.as_of, Path(args.out))


if __name__ == "__main__":
    main()

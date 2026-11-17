"""M3 (part 1): cross-fitted GNN edge-plausibility scores.

Usage: python scripts/m3_train_gnn.py --config configs/m3_rgcn.yaml [--tag rgcn]
Outputs:
  data/processed/gnn_scores_<tag>.parquet        out-of-fold score for every assertion edge
  data/processed/gnn_calibration_<tag>.parquet   validation-edge scores (conformal calibration)
  results/m3/<tag>_linkpred_metrics.csv, results/m3/<tag>_history.csv
"""

from __future__ import annotations

import argparse
import time

import pandas as pd
import torch

from kggc.models.train import TrainConfig, cross_fit
from kggc.utils import PROC_DIR, RESULTS_DIR, load_config, seed_everything, write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    cfg = TrainConfig(**load_config(args.config))
    tag = args.tag or cfg.encoder
    seed_everything(cfg.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    data = torch.load(PROC_DIR / "kg_T.pt", weights_only=False)
    vtab = pd.read_parquet(PROC_DIR / "kg_T_variants.parquet")
    t0 = time.time()
    res = cross_fit(data, vtab, cfg)
    wall = time.time() - t0

    node2vid = vtab.set_index("node")["variation_id"]
    dis = pd.read_parquet(PROC_DIR / "kg_T_nodes" / "disease.parquet")["id"]
    for key, name in (("scores", "gnn_scores"), ("calibration", "gnn_calibration")):
        df = res[key]
        df["variation_id"] = df["variant_node"].map(node2vid).astype("int64")
        df["mondo_id"] = dis.reindex(df["disease_node"]).values
        df.to_parquet(PROC_DIR / f"{name}_{tag}.parquet", index=False)
    out = RESULTS_DIR / "m3"
    out.mkdir(parents=True, exist_ok=True)
    res["metrics"].to_csv(out / f"{tag}_linkpred_metrics.csv", index=False)
    res["history"].to_csv(out / f"{tag}_history.csv", index=False)
    write_json(
        {"config": vars(cfg), "wall_seconds": wall, "device": cfg.device, "gpu_hours": 0.0},
        out / f"{tag}_run.json",
    )
    print(res["metrics"].to_string())
    print(f"wall {wall:.0f}s")


if __name__ == "__main__":
    main()

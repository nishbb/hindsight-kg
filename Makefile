PY ?= .venv/bin/python
SMALL := KGGC_SMALL=1

.PHONY: install data test lint labels graph gnn evaluate dossier qiagen agent cleaning copd reproduce reproduce-small clean-small

install:
	uv venv --python 3.12 .venv && uv pip install --python $(PY) -e '.[dev]'

data:
	$(PY) scripts/download_data.py

test:
	$(PY) -m pytest -q -p no:warnings

lint:
	$(PY) -m ruff check src tests scripts

labels:            ## M1
	$(PY) scripts/m1_labels.py

graph: labels      ## M2
	$(PY) scripts/m2_build_kg.py

gnn: graph         ## M3 part 1 (about 17 min R-GCN, 42 min HGT on CPU)
	$(PY) scripts/m3_train_gnn.py --config configs/m3_rgcn.yaml
	$(PY) scripts/m3_train_gnn.py --config configs/m3_hgt.yaml

evaluate:          ## M3 part 2
	$(PY) scripts/m3_evaluate.py --tags rgcn hgt --n-boot 500

dossier:
	$(PY) scripts/build_clinvar_index.py

qiagen: dossier    ## M4
	$(PY) scripts/m4_qiagen.py
	$(PY) scripts/m4_qiagen.py --as-of 2020-01-01
	$(PY) scripts/m4_broad_eval.py

agent:             ## M5 (needs NVIDIA_API_KEY or cached responses)
	$(PY) scripts/m5_audit.py

cleaning:          ## M6 part 1
	$(PY) scripts/m6_cleaning.py

copd:              ## M7
	$(PY) scripts/m7_copd.py --agent

reproduce: data gnn evaluate qiagen agent cleaning copd

# CPU, under 30 minutes after `make data`: 500-gene subsample, 2 folds, light background graph,
# 100 bootstrap draws, 3 dossiers, GNN-only audit (no LLM), 1-seed cleaning run.
reproduce-small: labels graph
	$(PY) scripts/m3_train_gnn.py --config configs/small.yaml --tag small
	$(PY) scripts/m3_evaluate.py --tags small --n-boot 100 --skip-prospective --only-scored --out-suffix _small
	$(PY) scripts/build_clinvar_index.py
	$(PY) -m kggc.variant_dossier.cli --rsids rs334,rs429358,rs7412 --as-of latest --out results/small/dossiers
	$(PY) scripts/m6_cleaning.py --tag small --seeds 0 --epochs 30 --arms original,gnn_flag,random --out-suffix _small
	@echo "reproduce-small finished; outputs in results/m3_small, results/m6_small, results/small"

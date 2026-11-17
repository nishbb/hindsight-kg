# CLAUDE.md

This repository is built by following `docs/BUILD_PROMPT.md`. Read it fully before doing any work. It defines the research question, hard rules, data sources, compute budget, and milestones M0 to M8.

Working rules:
- Work one milestone at a time, in order. Before starting a milestone, check `NOTES.md` for what is already done.
- Log decisions, failures, GPU hours, and open issues in `NOTES.md` as you go.
- Commit after each milestone with passing tests. Do not push to `main` directly; use a branch per milestone (`m0-setup`, `m1-clinvar-labels`, ...) and open a pull request.
- Public data only. Never use COPDGene, SPIROMICS, dbGaP, or any controlled-access data.
- Never commit API keys, `.env`, or raw downloaded data (`data/` is gitignored). Read keys from environment variables; see `.env.example`.
- No temporal leakage: every feature used at snapshot time T must have existed at T. Leakage tests must pass.
- Never fabricate citations, numbers, or API fields. Every number in the README must come from a file in `results/` produced by a script.
- Do not use em dashes in prose.

---
name: Canonical Pipeline Standard
description: BINDING COMMITMENT — all experiments must run through pipeline.py with profile name. No ad-hoc scripts. Deterministic, reproducible, one standard.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
All future experiments MUST run through `experiments/pipeline.py` with a profile name. No ad-hoc shell scripts. No hand-crafted SSH commands.

**Why:** The ad-hoc launch_wilde_shiraz.sh approach produced:
- Corrupted tto_frames.pt upload (wrong range [0, 2.1e37])
- Hardcoded archive sizes in auth eval (wrong rate calculation)
- Missing zoom scalar optimization for GREEN
- No error handling between pipeline stages
- Non-reproducible deployments (different data on WILDE vs SHIRAZ)
- Every integration bug in this project was in ad-hoc code, not canonical tools

**How to apply:** 
```
python experiments/pipeline.py --profile shiraz --device cuda --output-dir results/shiraz
```

One command does: train → QAT → pose TTO → build archive → contest_eval.

Requirements for the pipeline:
1. Profile from profiles.py is the ONLY config source (no CLI flag overrides for arch params)
2. Seeds pinned: torch.manual_seed, numpy.random.seed, random.seed (all from profile.seed)
3. Full provenance: git hash, GPU info, pytorch version, profile dict, timestamps per stage
4. Validate at every boundary: checkpoint exists, shapes match, loss is finite, archive size reasonable
5. Bundle all artifacts + logs + provenance JSON into results tarball
6. Print auth score at exit
7. Works on any device: cuda, mps, cpu

GREEN was the first experiment deployed through this standard. WILDE/SHIRAZ ran ad-hoc (their results are valid but not reproducible by the standard).

---
name: tac Library Hardening Session
description: Major hardening of tac v0.7→v0.8 — 10+ bugs fixed, pydantic, tests, compliance. Critical for next agent.
type: project
---

Session 2026-04-09/10 performed deep hardening of the tac library.

**Critical bugs found and fixed:**
1. Proxy scorer used soft cosine SegNet (inflated numbers, wrong metric) → hard argmax
2. Checkpoint selection used same soft metric → hard argmax
3. compute_boundary_mask: missing preprocess_input (wrong resolution)
4. compute_boundary_mask: missing T dimension (B,C,H,W not B,T,C,H,W)
5. inflate_postfilter: "standard" variant not recognized → submission would crash
6. Training state saves non-atomic → MPS SIGKILL destroys checkpoints
7. save_int8 zero-guard too weak (==0.0 vs <1e-10)
8. Train/eval leakage: eval subset overlapped with training data
9. float32 vs uint8 gap between proxy and official scorer
10. Hardcoded paths in tac library (security/portability issue)

**Infrastructure added:**
- Pydantic: TrainConfig (12 validators), ScoreResult, CheckpointMeta models
- 61 tests: compliance, quantization, training, config validation
- Ruff clean, hypothesis fuzzing
- Atomic saves + signal handlers + atexit for crash recovery
- Checkpoint backup to .backups/
- Canonical scorer: one evaluation path, no ambiguity
- eval_holdout config: contest mode (0.0) vs production mode (0.25)

**Why:** Every previous training run had at least one silent bug affecting scores.
**How to apply:** Always run `pytest tests/` before deploying. Always use canonical_score for evaluation.

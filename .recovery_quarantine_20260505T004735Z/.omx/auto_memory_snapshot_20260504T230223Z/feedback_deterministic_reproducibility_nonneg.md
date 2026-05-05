---
name: Deterministic Reproducibility NON-NEGOTIABLE
description: BINDING NON-NEG: every training/eval/TTO run MUST be bit-exact reproducible. Seed=42, deterministic=True, CUBLAS_WORKSPACE_CONFIG=:4096:8 BEFORE cuBLAS, configure_reproducibility() called, provenance JSON per run, check_determinism.py preflight gate.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Determinism is the foundation of every measurement.** Without it, a "score regression" might be noise, a "score win" might be noise, and we burn GPU dollars chasing artifacts. Make every score reproducible bit-exact across runs.

**Why:** When we lacked this, the 2026-04-25 SHIRAZ vs CUDA-baseline gap was contaminated by non-deterministic cuDNN ordering. We could not tell what was a real architectural difference vs a kernel-launch race condition. Today's pose TTO with `noise_std=0.5` further proves it: we have to be able to repeat the same proxy run twice and get the same loss curve to know if a fix actually helped.

**Required at every scope, in this exact order:**

1. **Profile (declarative source of truth)** — every renderer profile in `src/tac/profiles.py` MUST have:
   ```
   "seed": 42,
   "deterministic": True,
   "eval_roundtrip": True,
   ```
   Preflight rule `preflight_profiles()` enforces all three.

2. **Environment (BEFORE any torch.cuda call)** — every launcher MUST export:
   ```
   CUBLAS_WORKSPACE_CONFIG=:4096:8     # required for deterministic cuBLAS matmuls
   PYTHONHASHSEED=42                    # matches profile.seed
   PYTHONUNBUFFERED=1                   # logs in real time
   ```
   `tools/check_determinism.py` validates `CUBLAS_WORKSPACE_CONFIG` is `:4096:8` or `:16:8` and exits FATAL otherwise.

3. **Library (in code, before first kernel)** — every training/eval entrypoint MUST call
   ```
   from tac.experiments.train_renderer import configure_reproducibility
   configure_reproducibility(seed=seed, deterministic=True)
   ```
   which sets `torch.manual_seed`, `numpy.random.seed`, `random.seed`,
   `torch.use_deterministic_algorithms(True)`, `torch.backends.cudnn.deterministic=True`,
   `torch.backends.cudnn.benchmark=False`. Validated by `check_determinism.py` smoke-test.

4. **Provenance (per run, written BEFORE work starts)** — every canonical bootstrap writes a `provenance.json` containing: `git_hash`, `gpu_name`, `driver_version`, `torch_version`, `cuda_version`, `profile`, `seed`, `cublas_workspace_config`, `started_at_utc`. Without this, a "won at score X" claim is unfalsifiable.

5. **Preflight (last gate before GPU spin-up)** — `tools/check_determinism.py <profile>` runs as Stage 2 of every canonical bootstrap (`scripts/remote_*_bootstrap.sh`). It exits non-zero on any of the above failing, which kills the run before $0.30 of GPU is wasted.

**How to apply:**
- ANY new training/eval/TTO script must thread the determinism stack from day one.
- ANY new bootstrap (`scripts/remote_*_bootstrap.sh`) must call check_determinism in Stage 2 and write provenance.json.
- ANY profile addition MUST set seed/deterministic/eval_roundtrip.
- A "score" without a matching provenance.json that was written under deterministic env is `[non-reproducible]` and may not be cited as a baseline.
- If a run lacks the determinism stack, it does not exist for decision-making purposes.
- Reference scripts:
  - `scripts/remote_train_bootstrap.sh` (Stage 0 + Stage 2)
  - `scripts/remote_pose_tto_bootstrap.sh` (same pattern)
  - `scripts/remote_pose_tto_only_bootstrap.sh` (2026-04-26: lifted from /tmp ad-hoc, made canonical)
  - `tools/check_determinism.py` (the preflight gate)

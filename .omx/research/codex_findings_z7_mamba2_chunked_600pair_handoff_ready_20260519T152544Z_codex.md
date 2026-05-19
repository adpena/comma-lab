---
schema: codex_findings_v1
topic: z7_mamba2_chunked_600pair_handoff_ready
generated_utc: 2026-05-19T15:25:44Z
agent: codex
lane_id: lane_z7_mamba2_score_aware_600pair_handoff_20260519
substrate_id: time_traveler_l5_z7_mamba2
score_claim: false
promotion_eligible: false
axis: "[local-mps-training advisory] + [local-cpu-inflate advisory]"
---

# Codex Findings: Z7-Mamba2 Chunked Score-Aware Handoff Ready

## Verdict

Z7-Mamba2's immediate handoff blockers are burned down to the exact-eval
boundary. The trainer now emits a full 600-pair recurrent/static same-byte
packet, records scorer-free CPU inflate verification for MPS-trained packets,
and the handoff doctor reports `ready_for_exact_eval_handoff=true` with zero
pre-dispatch blockers.

This is NOT a score claim. The artifact is a local MPS-trained packet plus CPU
inflate-custody proof. Contest-CUDA and contest-CPU auth eval remain required.

## Bugs Extincted

1. Batch-size false authority:
   - Previous behavior: `--batch-size` existed but the trainer failed closed
     unless it equaled `--max-pairs`.
   - Landing: `--batch-size` is now a real pair chunk size for decoder/scorer
     loss. The recurrent latent replay stays global and ordered; decoder/scorer
     activation graphs are bounded over `pair_chunk_size`.
   - Default: `8`, matching the existing T4 scorer cache/chunk discipline and
     now encoded as `DEFAULT_PAIR_CHUNK_SIZE`.

2. Score-aware 600-pair peak-graph blocker:
   - Landing: score-aware training builds a canonical `GTScorerCache` once,
     then uses a two-pass streamed objective:
     `two_pass_streamed_chunks_global_pose_sqrt_exact_first_order`.
   - The value pass computes global `seg_term` and `pose_term`; the backward
     pass applies the exact first-order coefficient for
     `gamma_pose * sqrt(global_pose_term)`.
   - `--noise-std != 0.0` fails closed because stochastic value/backward
     mismatch would break streamed objective equivalence.

3. MPS inflate verifier false-negative:
   - Root cause: the trainer passed `device="mps"` to the scorer-free inflate
     runtime, but `select_inflate_device` intentionally accepts only
     `auto/cpu/cuda`.
   - Landing: `_inflate_verify_device()` maps MPS training to CPU inflate
     verify and records `inflate_verify.device`.
   - The handoff doctor now reports missing static-output evidence as missing
     evidence, not as "runtime output not changed."

4. Runtime dependency closure:
   - Landing: Z7MCM2 parse/inflate no longer imports NumPy. Runtime parsing uses
     `torch.frombuffer(bytearray(...), dtype=...).clone()` for fp16/int8
     sections. Train-time packing can still use tensor `.numpy()` because it is
     outside inflate-time dependency closure.

## Full 600-Pair Artifact

- Output directory:
  `experiments/results/z7_mamba2_score_aware_600pair_mps_verifyfix_20260519T1521Z`
- Stats:
  `experiments/results/z7_mamba2_score_aware_600pair_mps_verifyfix_20260519T1521Z/z7_mamba2_full_main_export_stats.json`
- Handoff artifact:
  `experiments/results/z7_mamba2_score_aware_600pair_mps_verifyfix_20260519T1521Z/handoff/z7_exact_eval_handoff_20260519T152459Z.json`
- Recurrent archive ZIP SHA-256:
  `ed4ee43b196cd4ade5013ea3b85440e97b0565fc21ac169c8b7ce01b83400abc`
- Static-control archive ZIP SHA-256:
  `a49f40ff7e18760e2f580652123d6b0f8e654456c39182dcd16df81394706ba9`
- Same archive ZIP bytes:
  `1,376,861` vs `1,376,861`
- CPU inflate raw SHA-256, recurrent:
  `19b54c321afb93673a2c31d633ce8c581863a073f3c06e3037ae79884d1884dd`
- CPU inflate raw SHA-256, static control:
  `4e57aed76cc50367519e94099f2cc5e527eebe9f696774eff004c7934b9d83a1`
- Recurrent/static raw byte differences:
  `2,507,058,706`
- Handoff doctor:
  `ready_for_exact_eval_handoff=true`, `result_review_blockers=[]`

## Timing And Memory

- 600-pair decode/resize: `8.36s`
- GT scorer cache: `3.70s`, `2250.0MB CPU`
- Score-aware epoch: `30.62s`, `19.60 pairs/s`
- CPU inflate verify for recurrent + static control: `85.64s`
- Total wall: `130.15s`
- Local MPS advisory memory telemetry:
  - current allocated: `6,496,292,096` bytes
  - driver allocated: `14,220,361,728` bytes

## Tests

Focused tests passed:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/ruff check \
  experiments/train_substrate_time_traveler_l5_z7_mamba2.py \
  tools/verify_z7_exact_eval_handoff.py \
  src/tac/substrates/time_traveler_l5_z7_mamba2/score_aware_loss.py \
  src/tac/substrates/time_traveler_l5_z7_mamba2/archive.py \
  src/tac/tests/test_z7_mamba2_score_aware_trainer_wiring.py \
  src/tac/tests/test_z7_mamba2_substrate_full_landing.py \
  src/tac/tests/test_z7_mamba2_scaffold.py \
  src/tac/tests/test_verify_z7_exact_eval_handoff.py

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools:upstream \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  src/tac/tests/test_z7_mamba2_score_aware_trainer_wiring.py \
  src/tac/tests/test_z7_mamba2_substrate_full_landing.py \
  src/tac/tests/test_z7_mamba2_scaffold.py \
  src/tac/tests/test_verify_z7_exact_eval_handoff.py \
  src/tac/tests/test_training_optimization_scorer_cache.py \
  src/tac/tests/test_score_pair_components_with_cache.py
```

Result: `116 passed, 12 warnings`.

Post-doctor-tightening focused test:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools:upstream \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  src/tac/tests/test_verify_z7_exact_eval_handoff.py \
  src/tac/tests/test_z7_mamba2_scaffold.py::test_full_trainer_authority_guard_helpers_fail_closed
```

Result: `7 passed`.

## Remaining Boundary

The packet is exact-eval handoff ready, not promoted. Next action is paired
contest-CUDA + contest-CPU auth eval for recurrent and static-control archives
using the handoff-generated commands. Until those land, keep all language
tagged as local advisory / no score claim.

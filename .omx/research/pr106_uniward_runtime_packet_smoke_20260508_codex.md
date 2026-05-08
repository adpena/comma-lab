# PR106 UNIWARD runtime packet smoke - 2026-05-08

## Status

`tools/build_pr106_uniward_runtime_packet.py` now closes the CPU-build side of
the PR106 UNIWARD-Lagrangian deployability gap identified in the Tier-A
readiness review.

This is **not** score evidence and does **not** promote or rank the lane.
Evidence grade is `[CPU-build]`; CUDA auth eval on the exact archive remains
required before any score claim.

## Build command

```bash
.venv/bin/python tools/build_pr106_uniward_runtime_packet.py \
  --rms-target 0.05 \
  --output-dir experiments/results/pr106_uniward_runtime_packet_20260508_codex_smoke
```

## Output artifact

- Archive: `experiments/results/pr106_uniward_runtime_packet_20260508_codex_smoke/archive.zip`
- Archive bytes: `150511`
- Archive SHA-256 prefix: `0641b8ac8084b362`
- Manifest: `experiments/results/pr106_uniward_runtime_packet_20260508_codex_smoke/build_manifest.json`
- Submission dir: `experiments/results/pr106_uniward_runtime_packet_20260508_codex_smoke/submission_dir`

## Smoke result

- Parsed tensors: `28`
- Latent pairs: `600`
- Implied frames: `1200`
- Decoder packed brotli: `134550` bytes versus PR106 published `170278`
- Archive delta versus PR106 published `186239`: `-35728` bytes
- Int8 rel_err: `0.046567`
- Weight identity rel_err through PR106 decoder: `1.852e-08`
- Max per-tensor smoke rel_err: `2.942e-08`

## Deterministic custody verifier

Command:

```bash
.venv/bin/python tools/verify_pr106_uniward_runtime_packet_sha256.py
```

Result:

- Rebuilt archive bytes: `150511`
- Rebuilt SHA-256: `0641b8ac8084b362b80e7c5bbe3c122946a8dbf7843fcbcd9f445aee7a56af7b`
- Verdict: byte-identical rebuild from committed PR106 inputs and build tool.

## Dispatch state

The manifest intentionally keeps:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `dispatch_attempted=false`

Next step, if this lane is selected: create a fresh dispatch claim and run the
canonical CUDA auth-eval path on the exact archive. Do not treat this CPU-build
as falsification or promotion evidence.

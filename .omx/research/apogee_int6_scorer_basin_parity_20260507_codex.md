# Apogee int6 scorer-basin parity evidence - 2026-05-07

## Scope

This ledger records a local predispatch readiness probe for
`apogee_int6_archive.zip`. It is not a contest-CUDA score and must not be used
as a rank, kill, or paper-score claim.

Evidence semantics: `scorer_basin_parity_gate`

Evidence tag: `[scorer-basin-parity:CPU]`

## Inputs

- Candidate archive:
  `experiments/results/apogee_int6_repack_20260504_claude/apogee_int6_archive.zip`
- Candidate SHA-256:
  `0176a2691a4daf5991170404d30a304ae30389621c0fc54914628414aef39ff1`
- Lossless reference archive:
  `experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip`
- Lossless reference SHA-256:
  `3fefbe5dfdd738179a55ca5c995ff8f63ec2755662d60684706f20d313913f58`
- Raw local evidence directory:
  `experiments/results/apogee_int6_basin_parity_20260507_claude/`

## Result

The 10-probe / 4-Hutchinson-sample parity run passed:

- `ready_for_exact_eval_dispatch`: `true`
- `score_claim`: `false`
- `promotion_eligible`: `false`
- `evidence_grade`: `empirical`
- `scorer_basin_parity_status`: `pass`
- `pose_dist_delta`: `+1.0792740795295686e-04`
- `pose_threshold`: `1.0e-03`
- `seg_dist_delta`: `+9.618123876862228e-04`
- `seg_threshold`: `5.0e-03`
- `hessian_trace_lossless`: `2.7495672440185547e+06`
- `hessian_trace_quantized`: `2.8684887436523438e+06`
- `hessian_log_ratio`: `+0.01840776094147193`
- `hessian_log_ratio_tolerance`: `1.0`
- `absolute_pose_ceiling`: `1.0e-02`
- `absolute_seg_ceiling`: `2.0e-02`

Additional hardened custody fields from the rerun:

- `latents_match_exact`: `true`
- `candidate_latents_sha256`:
  `10f7d166d680bb507151260176d5ecaac3b8239f20c3dad2198bcd919c81e225`
- `lossless_latents_sha256`:
  `10f7d166d680bb507151260176d5ecaac3b8239f20c3dad2198bcd919c81e225`
- candidate payload member: `0.bin`, `170342` bytes,
  SHA-256 `4bcb81864af2e50a6366adb0c1e9c0846a0ab31f33350c1c80f3c5f7503e3424`
- lossless payload member: `0.bin`, `186131` bytes,
  SHA-256 `7f2cc905b7611ae8d7bced72be24e2266b0aa341f90cfeccbb0854fd8fc01eb7`

Interpretation: this is positive local basin-geometry readiness evidence for
apogee_int6. It clears the local non-proxy readiness blocker but does not create
an exact score claim. The next score truth remains exact CUDA auth eval on the
exact archive bytes through `archive.zip -> inflate.sh -> upstream/evaluate.py`.

## Predispatch Gate Narrowing

After the parity evidence landed, the local distortion proxy was run with:

- `archive_bytes`: `170450`
- `rel_err_pct`: `1.55`
- `n_layers`: `13`
- `evidence_semantics`: `local_distortion_proxy`
- `tag`: `[distortion-proxy:local]`

Artifact:
`experiments/results/apogee_int6_basin_parity_20260507_claude/distortion_proxy_local.json`

Re-running `tools/predispatch_sanity.py` with both `--distortion-proxy-ran` and
the scorer-basin parity JSON narrowed the refusal to one remaining blocker:

- `sanity_lossy_vs_lossless`: predicted band `[0.190, 0.204]` is below the
  lossless PR106 baseline `0.2095`, so the current policy treats the band as
  incoherent for a lossy repack.

Cleared gates in that run:

- `anchors_sufficient`
- `distortion_model_gate`
- `hazard_scan`
- `lane_registry_consistent`
- `apogee_evidence_semantics`

Supersession note, later 2026-05-07: commit `c1d42ddf` revised
`tools/predispatch_sanity.py` so the `sanity_lossy_vs_lossless` gate uses the
official contest byte-rate term instead of a blanket "lossy cannot beat
lossless" rule. A smaller lossy archive may now pass this gate only when:

- the candidate is fewer charged bytes than the lossless anchor;
- the predicted high score is no lower than the official rate-only floor;
- exact-SHA non-proxy readiness evidence is present and valid.

With the same scorer-basin parity evidence, the current apogee_int6 predispatch
check passes locally:

- `anchors_sufficient`: passed
- `sanity_lossy_vs_lossless`: passed; `15789` fewer charged bytes, official
  rate-only floor `0.1989`
- `distortion_model_gate`: passed
- `hazard_scan`: passed
- `lane_registry_consistent`: passed
- `apogee_evidence_semantics`: passed

Current status: evidence-complete locally and no longer blocked by the stale
lossy-vs-lossless sanity policy. It is still not a score claim and still needs
an active lane dispatch claim, a real remote exact-CUDA environment, exact CUDA
auth eval, adjudication, and score-claim review before promotion.

## Command

```bash
.venv/bin/python tools/build_scorer_basin_parity_evidence.py \
  --candidate-archive experiments/results/apogee_int6_repack_20260504_claude/apogee_int6_archive.zip \
  --lossless-archive experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip \
  --output-json experiments/results/apogee_int6_basin_parity_20260507_claude/parity_evidence.json \
  --device cpu \
  --n-probes 10 \
  --n-hessian-samples 4
```

Predispatch check:

```bash
.venv/bin/python tools/predispatch_sanity.py \
  --archive experiments/results/apogee_int6_repack_20260504_claude/apogee_int6_archive.zip \
  --predicted-low 0.190 \
  --predicted-high 0.204 \
  --rel-err-pct 1.55 \
  --lane-class apogee_intN \
  --distortion-proxy-ran \
  --readiness-evidence-json experiments/results/apogee_int6_basin_parity_20260507_claude/parity_evidence.json \
  --json
```

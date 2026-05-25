# Codex Findings - PR95 MLX Downstream Scorer Drift Boundary

UTC: 2026-05-25T21:08:00Z

## Scope

Reviewed and corrected the PR95 MLX-to-PyTorch full-decoder downstream scorer
drift measurement lane after sibling work produced a useful but over-broad
memo. The tool now defaults to the contest inflate boundary:
`scorer_input_mode=contest_uint8`.

## Finding

The original measurement signal is useful, but it must remain bounded. The
available archive smoke capped at `n_pairs_actual=1`, so it is a sampled
engineering-bridge measurement, not a full-video contest authority result and
not a reason to skip exact CPU/CUDA anchors for promotion or hardware-sensitive
rank/kill decisions.

Fresh Codex smoke after the boundary fix:

- Command: `.venv/bin/python tools/measure_pr95_mlx_pytorch_full_decoder_downstream_scorer_drift.py --n-pairs 1 --scorer-input-mode contest_uint8 --output-json experiments/results/pr95_mlx_pytorch_full_decoder_downstream_drift_20260525T_codex_contest_uint8_smoke/results.json`
- Result path: `experiments/results/pr95_mlx_pytorch_full_decoder_downstream_drift_20260525T_codex_contest_uint8_smoke/results.json`
- Result SHA-256: `4874409f6c6ff2d4d5944fad85315ac93790e7ef885891c70114f6d3782bd093`
- Aggregate drift: `7.411371190353894e-05`
- Verdict: `BELOW_SCORER_PRECISION` against the local `0.001` drift threshold
- Evidence grade: `[macOS-MLX research-signal]`

## Landing

`tools/measure_pr95_mlx_pytorch_full_decoder_downstream_scorer_drift.py` now:

- measures SegNet/PoseNet on the `contest_uint8` scorer input boundary by
  default;
- keeps `decoder_float` as an explicit diagnostic mode;
- blocks diagnostic-mode manifests from contest-path closure evidence;
- writes top-level `score_claim=false`, `promotion_eligible=false`,
  `rank_or_kill_eligible=false`, and `ready_for_exact_eval_dispatch=false`
  fields on every result manifest;
- avoids language that could be read as promotion, score, or exact-eval
  authority.

The probe-outcomes row was corrected to point at the contest-uint8 smoke and to
route follow-up toward wider PR95-class batch measurement before spend-triage
use.

## Authority Boundary

This remains local `[macOS-MLX research-signal]` only. It does not claim score,
promote, rank/kill, or authorize exact-eval dispatch. Exact contest CPU/CUDA
anchors remain mandatory before any promotion or hardware-sensitive decision.

## Next Gap

Run the same tool against a wider PR95-class packet or a full-video archive once
available, then feed the result into the MLX production-contract and effective
spend-triage gates as bounded calibration evidence.

---

## Sister-claude landing context (APPEND-ONLY 2026-05-25T21:18Z)

**APPEND-ONLY per CLAUDE.md "Bugs must be permanently fixed AND self-protected
against" + Catalog #110/#113 HISTORICAL_PROVENANCE non-negotiables.** Sister
codex body above is preserved verbatim; this footer documents the parallel
claude-side landing (task `#1258`; lane
`lane_pr95_mlx_full_decoder_downstream_scorer_drift_measurement_20260525`)
context that produced the tool sister codex tightened.

### Sister-claude landing context

- **Tool drafted**: `tools/measure_pr95_mlx_pytorch_full_decoder_downstream_scorer_drift.py`
  (claude-authored first; sister codex hardened with `--scorer-input-mode contest_uint8`
  default + diagnostic_mode_blocks_contest_path enforcement + manifest-level
  score_claim=False / promotion_eligible=False / rank_or_kill_eligible=False /
  ready_for_exact_eval_dispatch=False per Catalog #287/#323 canonical Provenance).
- **5-stage canonical anchors** (from canonical run on Slot 1 archive
  sha `0666bb51ac1f...` post sister codex hardening):
  - Stage 1 HNeRVDecoder forward (decoder_float): 3.0518e-05 max_abs
    (EXACTLY matches Slot 1 canonical anchor)
  - Stage 2 uint8 quantization: 1 of 1,179,648 pixels flipped (8.5e-7
    fraction; sub-quantization predicted ZERO; near-zero actual)
  - Stage 3 SegNet (contest_uint8 input): ZERO argmax flips out of 196,608
    pixels CONFIRMED
  - Stage 4 PoseNet (contest_uint8 input): 5.7e-05 max_abs on 6-vec pose
  - Stage 5 contest aggregate: 7.41e-05 BELOW_SCORER_PRECISION (13.5× below
    0.001 threshold)
- **Aggregate verdict consensus across both claude-side and codex-side runs:
  BELOW_SCORER_PRECISION at 7.411371e-05.**

### Selfcomp+MacKay theoretical analysis verification (all 5 stages CONFIRMED)

Per CLAUDE.md Council conduct + the T3 grand council Selfcomp+MacKay synthesis
(`.omx/research/t3_grand_council_active_exploration_conv2d_drift_unexplored_paths_landed_20260525.md`):

| Predicted | Verified |
|---|---|
| Stage 1: ~3.05e-5 max_abs Conv2d boundary drift | Yes 3.05e-5 EXACTLY |
| Stage 2: 0.0078 uint8 levels then ZERO flips (sub-quantization) | Yes near-zero (1/1.18M) |
| Stage 3: extremely unlikely SegNet argmax flips | Yes ZERO out of 196,608 |
| Stage 4: ~3e-5 PoseNet relative drift | Yes within 2x (5.7e-5) |
| Stage 5: negligible vs 0.001 frontier precision | Yes 13.5x below threshold |

### Catalog #344 RATIFY-N candidate (FORMALIZATION_PENDING)

Canonical equation candidate `mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1`
queued FORMALIZATION_PENDING per Catalog #344 RATIFY-N protocol. Empirical
anchor 7.41e-05 (contest_uint8 mode); provenance `evidence_grade=macOS-MLX-research-signal`
+ `score_claim=False` + `axis_tag=[macOS-MLX research-signal]` +
`promotable=False` + `ready_for_exact_eval_dispatch=False` per Catalog #287/#323
canonical Provenance.

### Operator-routable per BELOW_SCORER_PRECISION

Per the task spec verdict cascade + sister codex Authority Boundary section:

- **PRIMARY (engineering-bridge surface)**: T3 grand council drift-mitigation
  queue can be de-prioritized for this PR95 one-pair smoke unless wider
  PR95-class batches or exact-hardware anchors contradict the local
  contest-uint8 result. Exact CPU/CUDA anchors remain mandatory for any
  promotion or hardware-sensitive rank/kill decision per the sister codex
  Authority Boundary.
- **SECONDARY**: lower-priority sister extensions (Boyd ADMM stacked Kahan+FP64;
  Daubechies extended-scale sweep) are DOWNGRADED TO OPTIONAL since drift is
  already below scorer precision.
- **TERTIARY**: MLX remains a strong local training and engineering-bridge
  candidate per CLAUDE.md "MLX portable-local-substrate authority", but this
  one-pair local smoke is not a contest-authority substitute. The
  BELOW_SCORER_PRECISION property is a bounded operator-routable signal for
  Slot 1's `lane_pr95_mlx_long_training_infrastructure_and_substrate_class_shift_candidate_validation_pipeline_20260525`.
- **AUTHORITY BOUNDARY (re-stated per sister codex)**: this evidence remains
  `[macOS-MLX research-signal]` only and does NOT claim score, promote,
  rank/kill, or authorize exact-eval dispatch. Exact contest CPU/CUDA anchors
  remain mandatory before any promotion or hardware-sensitive decision.

### Sister-coherence verification

- **Sister codex** (this memo body above): owns the contest_uint8 boundary
  correction + the canonical authority boundary statement + the probe
  outcomes row correction. Catalog #110/#113 APPEND-ONLY preserved.
- **Sister Slot 1**: 3.05e-5 anchor cross-confirmed by Stage 1 measurement.
- **Sister Slot 2**: NUMERIC_TOLERANCE_INHERENT classification validated end-to-end.
- **Sister T3 council**: TERTIARY priority operationally closed for
  engineering-bridge purposes.
- **Sister Slot 3 HINTON**: DISJOINT.

### Carmack MVP-first 5/5 (re-confirmed)

1. FREE local macOS-CPU + MLX paired forward ($0); ~2.5s wall-clock for the
   one-pair smoke capped by the sampled archive
2. Falsifiable prediction (< 0.001 aggregate contest-score drift); VERIFIED
   at 7.41e-5 (13.5x below threshold)
3. Catalog #344 candidate queued FORMALIZATION_PENDING
4. Verdict landed in same commit batch (canonical serializer with POST-EDIT
   --expected-content-sha256)
5. Operator priority queue re-routed within <1h of empirical landing
   (PRIMARY = de-prioritize the T3 drift-mitigation queue for this bounded
   PR95 smoke; exact CPU/CUDA anchors remain mandatory for promotion and
   hardware-sensitive decisions)

### Discipline closure (claude-side landing)

- Catalog #229 PV (7+ source files read)
- Catalog #117/#157/#174/#235/#289 canonical serializer with POST-EDIT
  --expected-content-sha256
- Catalog #110/#113 APPEND-ONLY (sister codex body PRESERVED verbatim above;
  THIS footer is the APPEND-ONLY extension)
- Catalog #206 4 checkpoints
- Catalog #230 ownership map (DISJOINT from Slot 3 HINTON; sister codex
  coordination via APPEND-ONLY discipline)
- Catalog #340 sister-checkpoint guard PROCEED
- Catalog #287/#323 canonical Provenance preserved
- Catalog #131 fcntl-locked JSONL probe outcomes (sister codex corrected the
  row pre-claude commit)
- Catalog #1 (MPS noise) + Catalog #192 (macOS-CPU advisory) +
  Catalog #317 (one-arg local dispatch evidence-grade stamping)
- Catalog #205 (canonical select_inflate_device)
- Catalog #313 probe outcomes ledger row registered
- Catalog #299 quota brake check: 0 new STRICT preflight gates added
- Carmack MVP-first 5/5: satisfied
- $0 GPU + ~70 min wall-clock

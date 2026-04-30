# OWV3/Fisher Byte-Aware Redesign Spec - 2026-04-30

Adjacent source docs:

- `grand_council_paradigm_shift_to_shannon_floor_20260430.md`
- `council_paradigm_shift_round{1,2,3}_20260430.md`
- `grand_council_paradigm_shift_to_shannon_floor_20260430_codex_progress.md`
- `shannon_floor_claim_matrix_20260430_codex.md`

## Verdict

Build OWV3, but do not promote the current measured smoke implementation.

The current OWV3 smoke is implementation-smoke evidence only. It did not run
canonical exact CUDA eval, and the measured archive size increased. That
reties the threshold/FP16-fallback config, not the OWV3/Fisher family.

## Root Cause To Fix

Current OWV3 protects high-sensitivity channels by serializing them as FP16.
Against the compact ASYM/PFP16 renderer baseline, FP16 fallback can cost more
than keeping the original packed representation. The implementation therefore
optimizes a sensitivity threshold, not the charged contest objective.

The optimization target must be:

```text
minimize 100 * seg_dist + sqrt(10 * pose_dist) + 25 * archive_bytes / 37,545,489
subject to exact archive compliance and deterministic inflate
```

## Required Redesign

### 1. Byte-Aware Action Space

For each eligible layer/channel group, evaluate charged archive cost and
estimated distortion for these actions:

- `keep_asym`: preserve the existing compact ASYM/PFP16-compatible bytes.
- `owv2_low_bit`: encode through OWV2/water-fill with explicit bit budget.
- `fp16_protect`: paid exception for calibrated high-risk channels only.
- `drop_or_merge`: allowed only if exact round-trip and scorer diagnostics show
  negligible risk.

Default protected/all-protected fallback must be `keep_asym`, not FP16.

### 2. Deterministic Archive Accounting

Before any exact eval, every candidate must emit:

- total archive bytes and SHA-256,
- member raw bytes, compressed bytes, CRC, timestamp, permissions, SHA-256,
- codec action summary per layer/channel group,
- side-info bytes charged inside `archive.zip`,
- deterministic rebuild check or byte-identical manifest.

Run exact CUDA eval only if byte accounting is plausible against the active
frontier. Current frontier comparator is PFP16 A++:

```text
score = 1.043987524793892
archive_bytes = 686635
archive_sha256 = 0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f
```

### 3. Fisher Calibration Protocol

Promotion-grade Fisher sensitivity requires:

- CUDA-only collection.
- Fixed calibration/holdout split seed.
- Stored pair indices.
- Separate SegNet, PoseNet, and combined maps.
- Per-layer rank stability report.
- `sensitivity_cv_distance` between calibration and holdout maps.
- Held-out perturbation checks tying predicted sensitivity to observed
  component movement.

Sensitivity maps used for promotion must set missing-policy to `error`.
`protect` is diagnostic-only and must be tagged non-promotable.

### 4. Exact Eval Gate

Promotion requires:

```bash
.venv/bin/python experiments/contest_auth_eval.py \
  --archive <candidate archive.zip> \
  --inflate-sh submissions/robust_current/inflate.sh \
  --upstream-dir upstream \
  --device cuda \
  --keep-work-dir \
  --work-dir <evidence dir>
```

Then adjudicate structured JSON only:

```bash
.venv/bin/python scripts/adjudicate_contest_auth_eval.py \
  --json <evidence dir>/contest_auth_eval.json \
  --archive <candidate archive.zip> \
  --regression-threshold <scoped threshold>
```

No log-score regex parsing. No CPU/MPS promotion. No byte-only promotion.

## Implementation Order

1. Add archive/member overhead analyzer for Lane G v3, PFP16, OWV2, and OWV3
   artifacts.
2. Change OWV3 fallback semantics from FP16-default to ASYM-preserving default.
3. Add byte/distortion allocator over layer/channel actions.
4. Add real-artifact byte regression tests: non-diagnostic OWV3 must not
   produce a larger deterministic archive without explicit review tag.
5. Upgrade Fisher profiler output to include calibration/holdout metadata,
   split indices, component maps, rank stability, and CV distance.
6. Make converter promotion default `missing-policy=error`; `protect` stays
   smoke-only.
7. Update remote OWV3 runbook/script to require archive manifest, adjudicator
   JSON, PFP16 comparator, source manifest, and scoped regression wording.
8. Run build-only sweeps first; exact CUDA eval only for byte-plausible
   candidates.

## Required Tests

- Unit: protected/all-protected fallback chooses `keep_asym` unless explicitly
  forced diagnostic FP16.
- Unit: byte allocator rejects actions whose charged bytes exceed comparator
  without distortion justification.
- Unit: sensitivity map promotion mode rejects missing Conv2d entries.
- Unit: Fisher calibration metadata must include split seed, calibration pairs,
  holdout pairs, component maps, and CV distance.
- Integration: deterministic archive rebuild yields identical SHA-256.
- Integration: adjudicator consumes `contest_auth_eval.json` and rejects
  missing CUDA/full-sample/provenance fields.

## Retirement Language

A bad exact result may retire only the measured implementation/config after
archive custody, scorer path, manifest, and harness checks. It must classify
the failure as one of:

- threshold failure,
- fallback-byte failure,
- Fisher calibration failure,
- codec distortion failure,
- stack interaction,
- archive/harness bug,
- indeterminate.

No broad OWV3/Fisher family kill without independent exact reproductions or a
mathematical impossibility proof plus Grand Council consensus.


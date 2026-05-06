# Categorical Charged-Label Construction Plan - 2026-05-06

Evidence grade: planning/readiness hardening
Score claim: false
Dispatch attempted: false
Ready for exact eval dispatch: false

## Context

The categorical/QMA9/CLADE-SPADE/openpilot surface already had strict archive
member custody, ZIP wire checks, runtime loader parity, and no-op controls.
The remaining gap was that the deterministic fixture carried a minimal
`class_codebook.json` string and did not expose a reusable planning surface that
maps charged archive members back to the canonical comma/openpilot semantic
classes.

## Primary Label Sources

- `https://github.com/commaai/comma10k/blob/master/README.md` for comma10k
  class IDs, label names, and colors.
- `https://blog.comma.ai/crowdsourced-segnet-you-can-help/` for the five-label
  openpilot SegNet grouping and why road/lane/movable/my-car classes have
  different planning roles.

No external code was vendored.

## Change

- Added `src/tac/categorical_candidate_plan.py`.
- The helper emits a deterministic `categorical_class_codebook_v1` payload with
  class IDs, comma10k IDs/colors, Selfcomp grayscale wire targets, default
  quantization bits, and openpilot prior hints.
- The helper emits a deterministic
  `categorical_charged_label_construction_plan_v1` with one row per charged
  contest class, references to the charged codebook/payload/runtime members,
  audited conditioning priors, and explicit next proof requirements.
- `audit_categorical_candidate_manifest()` now audits an optional
  `candidate_construction_plan`. A declared plan must remain
  `ready_for_exact_eval_dispatch=false`; if it attempts to claim dispatch
  readiness, candidate readiness fails closed.
- `tools/build_categorical_candidate_fixture.py` now writes
  `construction_plan.json` and embeds the same plan in `candidate.json`.

## Dispatch Boundary

The construction plan is not byte-closed archive parity and is not score
evidence. It records `real_byte_closed_archive_parity_missing` as a planning
dispatch blocker and keeps `ready_for_exact_eval_dispatch=false` until a real
candidate proves full decode/reencode parity, runtime-loader parity against
charged archive bytes, and then exact CUDA auth eval after a lane claim.

## Verification

Run after this patch:

```text
.venv/bin/python -m pytest src/tac/tests/test_categorical_candidate_readiness.py src/tac/tests/test_build_categorical_candidate_fixture.py src/tac/tests/test_materialize_comma_lab_public_export.py -q
.venv/bin/python -m ruff check src/tac/categorical_candidate_plan.py src/tac/categorical_candidate_readiness.py tools/build_categorical_candidate_fixture.py src/tac/tests/test_categorical_candidate_readiness.py src/tac/tests/test_build_categorical_candidate_fixture.py src/tac/tests/test_materialize_comma_lab_public_export.py
```

## Remaining Blockers

- The fixture payload is still not a real categorical archive candidate.
- PR91/HPM1 still needs full decode/reencode parity and device/runtime closure.
- No lane claim, GPU dispatch, exact CUDA auth eval, or score claim was made.

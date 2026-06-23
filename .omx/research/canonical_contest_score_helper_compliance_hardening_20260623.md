# Canonical contest-score helper + compliance hardening (Catalog #391)

**Date:** 2026-06-23
**Operator directive:** NON-NEGOTIABLE — contest compliance with upstream `modules.py` + `evaluate.py`
is bedrock. The score formula was hand-rolled across ~20+ files with the `37_545_489` constant scattered
and NO single canonical helper verified against upstream; a subagent dropped the `×25` on the rate term
(claimed break-even `d_seg` 1.89e-3; correct ~7.8e-4). Make the formula a single IMPORT.
**Pointer:** UNMOVED at 0.19110. No score claim. Upstream untouched.
**Authority note:** the canonical helper is an ADVISORY proxy + decision arithmetic. The AUTHORITATIVE
score is ALWAYS `upstream/evaluate.py` on the byte-closed bytes (CPU + CUDA per CLAUDE.md). Never a
substitute for a compliance claim.

## The authoritative formula (upstream/evaluate.py:92 — READ, never edited)

```
rate  = compressed_size / uncompressed_size        # evaluate.py:64-65
score = 100 * segnet_dist + math.sqrt(posenet_dist * 10) + 25 * rate   # evaluate.py:92
```

`uncompressed_size == 37_545_489` for the canonical contest video set (sister-pinned at
`src/tac/archive_byte_profile.py::CONTEST_ORIGINAL_BYTES`).

## Landing 1 — the canonical helper `src/tac/contest_score.py`

The operator explicitly named `src/tac/contest_score.py`; it did not exist. The pre-existing
`tac.score_composition` is the per-axis **delta** composer (marginal form; pose term = difference of two
`sqrt` calls) — a complementary surface, NOT the absolute-score helper the task required. Created the
absolute-score + decision-arithmetic home:

- `compute_contest_score(d_seg, d_pose, archive_bytes, *, uncompressed_size=37_545_489)` →
  `100*d_seg + sqrt(10*d_pose) + 25*(archive_bytes/N)` — byte-identical to evaluate.py:92.
- `seg_term`, `pose_term`, `rate_term` (the per-term primitives; `rate_term` carries the `×25`).
- `break_even_d_seg(target_S, d_pose, archive_bytes)` =
  `(target_S - pose_term - rate_term) / 100` — the EXACT arithmetic the subagent got wrong, centralized.
- Module constants `UNCOMPRESSED_SIZE_BYTES=37_545_489`, `SEG_WEIGHT=100`, `POSE_WEIGHT=10`,
  `RATE_WEIGHT=25` with docstring pointers to `upstream/evaluate.py:92`.

Input validation: finite + non-negative inputs, sqrt-domain guard on `d_pose`, bool rejected as numeric.

## Parity test (compliance proof, not self-consistency) — `src/tac/tests/test_contest_score_upstream_parity.py`

1. **Inline re-implementation** of evaluate.py:92 (independent witness) == helper across a 7-point grid.
2. **Cross-check vs TWO REAL landed exact rows** — the G3 torch-vehicle bc20 dual exact row (600 samples,
   from `.omx/research/g3_torch_vehicle_bc20_first_exact_row_20260618T0135Z.md`):
   - contest-CPU: d_seg 0.00260094, d_pose 0.00034168, 89244 B → **S 0.37797132** ✓ reproduced
   - contest-CUDA: d_seg 0.00262703, d_pose 0.00048168, 89244 B → **S 0.39153009** ✓ reproduced
   This proves the helper == a real `upstream/evaluate.py` output — the compliance bedrock.
3. `break_even_d_seg` round-trips against `compute_contest_score`; a dedicated regression pins the
   corrected 2026-06-23 value (~8.07e-4, NOT 1.89e-3).

**Result: 32 parity tests pass.**

## Landing 2 — STRICT guard `check_no_hand_rolled_contest_score` (Catalog #391)

Scans `src/tac/` + `tools/` + `experiments/` (excluding vendored/`_intake_` mirrors) for NEW hand-rolled
score arithmetic: the `37_545_489` literal, `25*.../<denom>`, `100*d_seg`, `sqrt(10*d_pose)` /
`(10*d_pose)**0.5`. Allowlist: the canonical helper + its tests + the documented sister-constant homes
(`archive_byte_profile.py`, `joint_scorer_aware_training.py`, `score_composition/__init__.py`). Same-line
`# HAND_ROLLED_SCORE_OK:<rationale>` waiver (placeholder rejected per Catalog #287).

- **WARN-ONLY at landing** (wired into `preflight_all`): live offender count **1669** across the repo
  (experiments 393 / src 1054 / tools 222) — the ~20+ scattered instances the operator flagged, plus
  every file that defines its own copy of the denominator. A one-shot mass-migration would risk the
  compliance bedrock it protects, so warn-only + the offender list is the correct staged-migration default.
- 18 dedicated gate tests pass (catches each pattern, allows imports/unrelated arithmetic, waiver respect,
  placeholder rejection, vendored exclusion, strict raise, line-number reporting).
- Catalog claimed via `tools/claim_catalog_number.py` (#391; counter reset to 391 after a stray
  double-claim). Under the #400 gate-consolidation quota.

## Surgical migrations (active decision surfaces only)

- `tools/analyze_dseg_slope_gate.py` — removed local `_RATE_DENOM` + inline pose/rate/break-even;
  now `break_even_d_seg(...)`. No longer flagged.
- `tools/render_decisive_run_dashboard.py` (the LIVE dashboard renderer, pid running) — migrated the
  break-even block to `break_even_d_seg(...)`. **Output-byte-equivalent**: old and new both yield
  `0.0008065732773771983` → renders `8.1e-04` identically, so the running process needs NO restart
  (the edit only affects future invocations and produces the same pixels). No longer flagged.
- **DEFERRED (staged):** the remaining ~1667 legacy offenders across `experiments/` + `src/tac/` are left
  for staged migration under the WARN-ONLY gate — mass-migration deliberately avoided.

## Byte-close memo fix — `.omx/research/byteclose_readiness_1e3_run_20260623.md`

The Fire-condition block claimed: "beat 0.19110 → d_seg < ~1.89e-3 (≈1.1× below 0.00212)" and
"sub-0.15 → ~1.48e-3". **WRONG** — it computed pose+rate as the bare rate FRACTION (`bytes/N` ≈ 0.0024)
instead of the rate TERM (`25·bytes/N` ≈ 0.053) + pose TERM (`√(10·d_pose)` ≈ 0.057). The `×25` was
dropped. Corrected (via the canonical helper, d_pose 3.3e-4, 79,592 B):

| | memo (WRONG) | corrected |
|---|---|---|
| pose+rate **term** | ~0.0024 | **0.1104** |
| beat 0.19110 → d_seg < | 1.89e-3 | **8.07e-4** |
| sub-0.15 → d_seg < | 1.48e-3 | **3.96e-4** |
| current 0.00212 is | "≈1.1× below" | **~2.63× ABOVE** the beat break-even |

The 06-22 runbook's "8.1e-4" was CORRECT all along (the memo's NOTE wrongly called the new values
"LOOSER"). Added an APPEND-style CORRECTION block + steered re-derivation to
`tac.contest_score.break_even_d_seg`.

## modules.py / evaluate.py component-compliance audit (cited, not re-gated)

The AUTHORITATIVE components (already gated by existing CLAUDE.md non-negotiables; cited here for
completeness — NOT re-implemented):
- **d_seg** = SegNet argmax-disagreement rate (`upstream/modules.py` `DistortionNet.compute_distortion`).
- **d_pose** = PoseNet 6-dim pose MSE (FastViT-T12 hydra head).
- **GT decode** via `frame_utils.yuv420_to_rgb` — NEVER PyAV rgb24 (manufactures ~100× phantom pose);
  gated by the "Forbidden misleading…/GT-decode" non-negotiables.
- **uint8 eval round-trip** (384→874→uint8→384) — gated by the "eval_roundtrip NON-NEGOTIABLE".
- **600-sample non-overlapping** seq_len=2 loop (`evaluate.py:67-83`).
- **MPS is never authority** — gated by the "MPS auth eval is NOISE" non-negotiable.

The new helper is the canonical advisory proxy + decision math; the authoritative score is ALWAYS
`upstream/evaluate.py` on the byte-closed bytes (CPU + CUDA on contest-compliant hardware).

## 6-hook wire-in (Catalog #125)

- #1 sensitivity-map ACTIVE (`break_even_d_seg` IS the d_seg decision threshold).
- #2 Pareto N/A (delta-form lives in `tac.score_composition`).
- #3 bit-allocator ACTIVE (`rate_term` = canonical byte→score-units conversion).
- #4 cathedral autopilot dispatch ACTIVE (dashboard/slope-gate fire decisions consume the helper).
- #5 continual-learning posterior N/A (pure arithmetic; the dropped-×25 incident is the empirical anchor).
- #6 probe-disambiguator N/A (single canonical formula).

## Files

- NEW `src/tac/contest_score.py`
- NEW `src/tac/tests/test_contest_score_upstream_parity.py` (32 tests)
- NEW `src/tac/tests/test_check_391_no_hand_rolled_contest_score.py` (18 tests)
- EDIT `src/tac/preflight.py` (Catalog #391 gate + warn-only wire-in)
- EDIT `tools/analyze_dseg_slope_gate.py`, `tools/render_decisive_run_dashboard.py` (migrations)
- EDIT `.omx/research/byteclose_readiness_1e3_run_20260623.md` (break-even correction)

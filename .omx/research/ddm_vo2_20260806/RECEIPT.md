---
schema: ddm_vo2_recursive_instrument_registry_receipt.v1
date_utc: 2026-08-06
arm: ddm_vo2
axis: "[scorer-free receipt/source audit]"
research_only: true
score_claim: false
promotion_eligible: false
pointer_moved: false
tokens: [no-triality, p0-ledger-ok]
---

# DDM VO2 - Recursive Instrument Registry

## Generation 2 R2 Addendum

R2 is started, not sealed. The R1 rebuild was re-run first and matched the prior manifest exactly:
registry `947b7faaa3ba61dfad567b434075c8151028e8fd5e6dbe3c38cbcb4ccc43b936`, summary
`0b450d49d33d1ba8e756b1d16d031f144fd05a490ace2c4292b577d4bb2b4393`, row count 4,630.

R2 then added `.omx/research/ddm_vo2_20260806/R2_ELEMENT_DECOMPOSITION.jsonl`: 23 selected rows, each
with all ten charter elements graded. Selection followed the charter priority and verdict-fanout order:

| family | rows |
|---|---:|
| vo1-round0 | 1 |
| ca1-round0 | 6 |
| sw1-round0 | 8 |
| dk1-round0 | 3 |
| vo2-new | 5 |

The batch is intentionally partial: the remaining registry rows still need R2, then R3 calibration
lineage and R4 self-audit. `ROUND_SUMMARY.json` now records `round_reached=R2-partial`, `r2_complete=false`,
`seal_ready=false`.

R2 form-grade references named in this receipt:
`form_grade_ref:iteration_cap_stop_defaults`
`form_grade_ref:project_after_pose_null_projectors`
`form_grade_ref:posthoc_uint8_rounding_float_first_realizers`

## R2 Recall Evidence

Searched beyond the charter seeds:

| surface | query or command | changed plan |
|---|---|---|
| Memory registry | `rg -n "vo2r2|V0_2R2|rail_mask|V0.2|V0_2" MEMORY.md` | No VO2-specific prior hit found; proceeded from live artifacts. |
| Live board | `.omx/state/main_hot_state.md` | Confirmed this arm is scorer-free and the own-vehicle pointer is `S=0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`. |
| Required resume input | `NEXT_IF_RESUMED.md` plus `vo2_prompt.md` | Forced R2 continuation, not a gen-1 reseal. |
| R1 builder | `tools/build_ddm_vo2_instrument_registry.py --out-dir ...` | Rebuild matched prior hashes before R2; no stale row count was trusted. |
| CA1/SW1/DK1/VO1 receipts | CA1, SW1, DK1, VO1 receipt/ledger reads | Set the first 18 R2 rows before any `vo2-new` source candidate. |
| Sibling RW1 receipts | `.omx/research/ddm_rw1_20260806/*` | Folded CA1 Class-B dispositions and q3x DK1-CVP bounded smoke into R2 instead of duplicating them. |
| Canonical equations | `trajectory_derived_stopping_20260805.py`, `pose_null_subspace_is_ac_only_20260804.py`, registry search | Added `trajectory_derived_stopping_20260805.py` as a surfaced R3 lineage instrument; used pose-null AC-only law to scope SW1/DK1 realization rows. |
| DAG/index | targeted searches over `CANONICAL_RESEARCH_INDEX*` and `sub015_DAG_topaiml...` | Confirmed metric-free/project-after and uint8 rounding are recurring instrument classes, not isolated bugs. |

## R2 Findings

1. Cap/stop remains the highest fanout defect. RW1 converted all six CA1 Class-B sites into explicit
   CAP-BOUND-at-stop dispositions, but only q3x has an attached cap-stop receipt, and none becomes
   convergence evidence. Fire order: consume these rows only as cap-bound labels unless a later owner
   lands semantic stop or convergence evidence.
2. SW1 project-after rows stay `NAIVE-NAMED`: Arm E and Arm M are diagnostic comparators, not fire-order
   routes. Solve-within null-basis is the replacement surface.
3. DK1 CVP/Babai is `OPTIMAL-RECEIPT` only for the bounded kept-set local realizer scope. Naive and
   Dykstra remain `NAIVE-NAMED`; no global MIQP or n600 claim is made.
4. The top `vo2-new` source candidates are now R2-graded as source candidates, not consumer proofs.
   The token denominator heuristic remains a routing instrument until AST-level consumer lineage proves
   a specific verdict dependency.
5. New R3 lineage row surfaced: `src/tac/canonical_equations/trajectory_derived_stopping_20260805.py`.
   Any validator it names must be recursively enumerated before seal.

## Answer First

VO2 reached R1 and did not seal. The registry now has 4,630 unique rows in
`INSTRUMENT_REGISTRY.jsonl`.

Dry trajectory:

| round | new rows | dry |
|---|---:|---|
| R0 seed import | 10 | false |
| R1 exhaustive-over-live-source candidate census | 4,630 | false |

Row provenance:

| family | rows |
|---|---:|
| vo1-round0 | 10 |
| ca1-round0 | 89 |
| sw1-round0 | 16 |
| dk1-round0 | 3 |
| vo2-new | 4,512 |

The `vo2-new` rows are deliberately overinclusive source candidates: every readable live Python file
under `experiments/`, `tools/`, and `src/tac/` carrying at least two measurement/verdict tokens. They
are labelled `OVERINCLUSIVE_SOURCE_CANDIDATE_NEEDS_R2_CONSUMER_CONFIRMATION`, not promoted as proven
verdict consumers.

## Recall Evidence

Searched beyond the charter seeds:

| surface | query or command | changed plan |
|---|---|---|
| Memory registry | `rg "vo2|VO2|codex_runs|omega|checksum|frontier|lane" MEMORY.md` | Confirmed no older VO2 registry precedent; reused frontier separation and no-score framing. |
| Live board | `.omx/state/main_hot_state.md` | Confirmed `ddm_et2` owns scorer slot and this arm is scorer-free. |
| Round-0 receipts | VO1, CA1, SW1, DK1 receipts and machine ledgers | Converted one-pass ledgers into registry provenance families instead of just quoting summaries. |
| Canonical equations registry | `tools/list_canonical_equations.py --json` | Confirmed registry-scale evidence exists, but output is too large for full hand-read in this generation; R2 should query by instrument family. |
| Corpus source roots | builder scan over `experiments/`, `tools/`, `src/tac/` | Added 4,512 source-candidate rows beyond the charter seeds. |

What changed: I did not try to hand-enumerate "all instruments" from prose. The landing creates a
deterministic builder that states the denominator and labels unconfirmed source rows so R2 can grade
elements without losing the candidate surface.

## Artifacts

| path | bytes | sha256 |
|---|---:|---|
| `.omx/research/ddm_vo2_20260806/INSTRUMENT_REGISTRY.jsonl` | 6,109,205 | `947b7faaa3ba61dfad567b434075c8151028e8fd5e6dbe3c38cbcb4ccc43b936` |
| `.omx/research/ddm_vo2_20260806/ROUND_SUMMARY.json` | 1,309 | `0b450d49d33d1ba8e756b1d16d031f144fd05a490ace2c4292b577d4bb2b4393` |
| `.omx/research/ddm_vo2_20260806/MANIFEST.sha256.json` | 256 | `02cc98ce2bf51a927ee38d4b50a65b7b27d7266b27cf21dd8567b772453009a0` |

Code:

- `tools/build_ddm_vo2_instrument_registry.py`
- `tools/check_instrument_registry_form_grade_refs.py`
- `tools/tests/test_ddm_vo2_instrument_registry.py`

The builder uses a deterministic audit-date `last_graded` label by default, with `--last-graded` available
for intentional refreshes. `MANIFEST.sha256.json` hashes the registry and summary only; its own file hash is
reported in the artifact table above, not embedded self-referentially.

## Top Findings So Far

1. `iteration_cap_stop_defaults` remains the highest confirmed instrument defect. It carries VO1
   fanout 89 with a named cure: `CapStopReceipt`, semantic stop receipts, and uncapping before any
   convergence-like verdict reuse. Fire order: handle CA1 Class B sites first and refuse cap-stopped
   negatives as family evidence until their stop reason is explicit.
2. `project_after_pose_null_projectors` and SW1 seam rows remain the highest immediate form correction:
   solve-within cleared the n4 eta bar where both project-after variants failed. Fire order: ET/Q3
   successors must route through solve-within before scorer spend.
3. `posthoc_uint8_rounding_float_first_realizers` remains confirmed by DK1: CVP/Babai dominated naive
   and Dykstra on the measured small-n leakage ladder. Fire order: wire DK1 or a stronger integer
   realizer before promoting Q3/SQ1/Q31/FD-style realized rows.
4. The new source-candidate denominator is large enough that R2 must be automated and family-prioritized.
   Highest token/fanout candidates include `train_levelset_witness_realized_through_R_mlx.py`,
   `ddm_costate_organ.py`, `ddm_sw1_null_basis_phase_solve.py`, and `ddm_et2_projected_phase_field.py`.
   Fire order: grade these with AST-level element extraction, not prose review.
5. This audit's own builder is now part of the self-audit surface. It is deterministic and tested, but
   its source-candidate heuristic is not a consumer proof. Fire order: R4 must grade the token heuristic,
   duplicate-ID guard, and source-root exclusions before any seal.

## Honest Non-Findings

- DK1 `cvp` is graded `OPTIMAL-RECEIPT` only for the kept-set small-n local realizer scope; no global
  integer optimum or n600 claim is made.
- CA1 sites with `reports_stop_reason` are not reopened on the stopping-rule element alone.
- SW1 `ALREADY_JOINT` rows are kept as clean local forms where the receipt already says no score claim
  and no archive claim.
- The form-grade checker found zero missing references in the four scoped round-0 receipts because those
  receipts do not cite registry instrument IDs directly. This is not a global clean pass.

## Warn-Only Preflight Check

Added `tools/check_instrument_registry_form_grade_refs.py`. It is warn-only by default and returns nonzero
only with `--strict`. Positive control executed:

```bash
.venv/bin/python -m pytest tools/tests/test_ddm_vo2_instrument_registry.py::test_form_grade_ref_positive_control
```

Result: `1 passed`. The bad control receipt citing `demo_instrument` without `form_grade_ref` is flagged;
the good control clears.

Scoped live scan after registry materialization:

```json
{
  "instrument_count": 4630,
  "missing_form_grade_ref_count": 0,
  "receipt_files_scanned": 4,
  "verdict_bearing_receipts": 4,
  "verdict_bearing_receipts_citing_registry": 0
}
```

## Verification

| command | result |
|---|---|
| `.venv/bin/python -m py_compile tools/build_ddm_vo2_instrument_registry.py tools/check_instrument_registry_form_grade_refs.py tools/tests/test_ddm_vo2_instrument_registry.py` | passed |
| `.venv/bin/python -m pytest tools/tests/test_ddm_vo2_instrument_registry.py` | 3 passed |
| `.venv/bin/python tools/build_ddm_vo2_instrument_registry.py --out-dir .omx/research/ddm_vo2_20260806` | wrote registry, summary, manifest |
| registry uniqueness check | 4,630 rows, 4,630 unique instrument IDs |
| form-grade checker scoped scan | warn-only, 0 missing refs in scoped round-0 receipts |

## Boundaries

No scorer forward pass, archive build, training run, paid dispatch, or `upstream/evaluate.py` run occurred.
No upstream file or common-contract protected file was edited. The existing dirty worktree and existing
P0 ledger row were not absorbed into this commit.

R2, R3, R4, and the zero-new-row seal remain open. This generation did not achieve goal progress.

Own-vehicle frontier remains unchanged: `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.

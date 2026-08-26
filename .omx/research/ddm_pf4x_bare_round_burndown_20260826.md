# PF4X bare-round eval-roundtrip burn-down — 30 scanner instances to zero

**Task:** `1305`  
**Actor:** `ddm_pf4x`  
**Verdict:** `COMPLETE-BUNDLE-READY-MAIN-MUST-LAND`  
**Authority:** local apparatus validation only; no contest score authority

## Outcome

The live verbose census found **30 scanner instances across 22 files**. They resolve to
29 physical `.round()` occurrences because one occurrence in
`probe_factored_lf_core_capacity_gate.py:292` is visited through two nested AST
functions. After per-function consumer tracing:

- 9 scanner instances are class (a), forward-only: 3 instances in 2 wholly read-only
  files and 6 instances in forward-only functions inside mixed-purpose files.
- 19 scanner instances are class (b), already-correct manual STEs whose `.round()` and
  `.detach()` were split across lines and therefore missed the gate's same-line
  recognizer.
- 2 scanner instances are class (c), test fixtures implementing the same correct
  manual STE across lines.
- **0** instances are genuinely severed differentiable paths. No banked result is
  invalidated and no live score-relevant surface was changed.

The strict gate is now clean, its executed positive control still fires, and the gate
itself was not weakened. The only file exemptions added are the two files proved
wholly forward-only. Function-local forward-only paths in files that train elsewhere
were conformed without exempting those files.

This arm produced no candidate archive, scorer run, Modal call, or score. The
canonical frontier pointer is unchanged.

## RECALL EVIDENCE

The bounded recall searched research memos, state stores, the research index, task
ledger/bridge, DAG material, and 449 canonical-equation registry rows for `bare
round`, `eval roundtrip`, `manual STE`, `Uint8STE`, gradient-freeze vocabulary, and
PF4X/1305 identifiers. Within that scope:

- `council_darts_s_freeze_audit_20260429.md` supplies the real failure precedent:
  bare `.round()` in a consumed differentiable roundtrip produced zero gradient and
  required `Uint8STE`.
- `council_round6_adversarial_20260429.md` supplies the exact false-positive
  mechanism present here: the gate's manual-STE recognizer requires `.round()` and
  `.detach()` on the same line, so a line-wrapped correct STE is reported.
- The canonical-equation registry has a predicted Quantizr three-stack entry that
  includes `eval_roundtrip=True`, but the search did **not find a gate-specific
  equation or receipt** that changes the site adjudications.
- Task `1305`/PF4X was not found in the searched canonical ledger, 2026-08-03 harness
  bridge, research index, or DAG before this arm registered it. This is bounded
  absence, not a global nonexistence claim.

The recall changed the implementation decision: already-correct STEs were preserved
and made legible to the existing gate rather than being mislabeled as integrity
defects or replaced wholesale with a different quantizer.

## Falsifiable prediction result

The charter's `>=80% class (a)` prediction is **falsified**: class (a) is 9/30 =
30.0%, and whole-file read-only exemptions are only 3/30 = 10.0%. The two
`_differentiable` sites are confirmed same-line-recognizer false positives, and the
prediction of zero live-surface blockers is confirmed.

## Per-site disposition

Coordinates are the pre-edit live-scan coordinates. `A-file` means the whole file was
proved read-only and received the gate's sanctioned per-file exemption. `A-fn` means
only the flagged function is forward-only; the file trains elsewhere and was not
exempted. `B-STE` and `C-test` retain the exact hard-rounded forward value and identity
STE gradient while placing `.round()` and `.detach()` in the same expression.

| # | Original site and function | Class | Cure | Executed control |
|---:|---|---|---|---|
| 1 | `experiments/diag_mps_posenet_drift_real_loss.py:79` `_roundtrip` | B-STE | Conformed the existing split manual STE to one expression. | Forward/gradient equivalence; strict clean. |
| 2 | `experiments/measure_lever3_v2_decoupling_and_variance.py:55` `_roundtrip_bhwc` | B-STE | Same-line manual-STE conformance. | Forward/gradient equivalence; strict clean. |
| 3 | `experiments/measure_r_cap_r_surv.py:342` `_apply_eval_roundtrip` | A-file | Added a scoped read-only entry; the file performs inference-only R-survival measurement with no backward/optimizer consumer. | File consumer trace; strict clean. |
| 4 | `experiments/probe_accel1_margin_hinge_exponent.py:89` `_roundtrip` | B-STE | Same-line manual-STE conformance. | Forward/gradient equivalence; strict clean. |
| 5 | `experiments/probe_anneal_schedule_calibration.py:103` `_roundtrip` | B-STE | Same-line manual-STE conformance. | Forward/gradient equivalence; strict clean. |
| 6 | `experiments/probe_concentrated_saliency_feasibility.py:130` `_eval_roundtrip` | A-fn | Preserved the hard forward branch and made its detachment explicit in one expression; no file exemption. | Forward equality and zero-gradient hard-branch check. |
| 7 | `experiments/probe_concentrated_saliency_feasibility.py:292` `_scored_logits` | B-STE | Same-line manual-STE conformance on the training-consumed path. | Forward/gradient equivalence; strict clean. |
| 8 | `experiments/probe_curve_core_dseg_feasibility_gate.py:249` `_eval_roundtrip_t` | B-STE | Same-line manual-STE conformance for the STE branch. | Forward/gradient equivalence; strict clean. |
| 9 | `experiments/probe_curve_core_dseg_feasibility_gate.py:251` `_eval_roundtrip_t` | B-STE | Preserved the explicit hard branch as detached from the same conformed rounded tensor. | Forward equality and branch-gradient check. |
| 10 | `experiments/probe_dseg_side_feasibility_corners.py:134` `_segnet_argmax_via_384_roundtrip` | A-fn | Preserved hard-round inference semantics with an explicit detached expression; no file exemption. | Forward equality and zero-gradient hard-branch check. |
| 11 | `experiments/probe_factored_lf_core_capacity_gate.py:89` `_eval_roundtrip` | A-fn | Preserved the hard forward branch with explicit detachment; no file exemption. | Forward equality and zero-gradient hard-branch check. |
| 12 | `experiments/probe_factored_lf_core_capacity_gate.py:292` `train_one_lf_core` | B-STE | Same-line manual-STE conformance. | Forward/gradient equivalence; strict clean. |
| 13 | `experiments/probe_factored_lf_core_capacity_gate.py:292` `_scored_logits` | B-STE | Same physical occurrence as #12, double-counted by nested AST traversal; one conformance cures both scanner instances. | Duplicate accounting verified; strict clean. |
| 14 | `experiments/probe_hinerv_grid_vs_lever_dseg.py:99` `_roundtrip_to_eval_bhwc` docstring | A-fn | Rephrased the docstring's literal `.round()` spelling; no executable operation was removed. | Source review; strict clean. |
| 15 | `experiments/probe_hinerv_grid_vs_lever_dseg.py:108` `_roundtrip_to_eval_bhwc` | A-fn | Preserved clamp/round/cast hard-forward behavior with explicit detachment; no file exemption. | Forward equality and zero-gradient hard-branch check. |
| 16 | `experiments/probe_hinerv_grid_vs_lever_dseg.py:123` `_roundtrip_to_eval_bhwc_differentiable` | B-STE | Confirmed false positive; conformed the existing manual STE to one expression. | Forward/gradient equivalence; strict clean. |
| 17 | `experiments/probe_lensA_ce_vs_margin_dseg_slope.py:77` `_roundtrip` | B-STE | Same-line manual-STE conformance. | Forward/gradient equivalence; strict clean. |
| 18 | `experiments/probe_lever2_softcosine_vs_ce_flipfix.py:85` `_roundtrip` | B-STE | Same-line manual-STE conformance. | Forward/gradient equivalence; strict clean. |
| 19 | `experiments/probe_lever4_qat_brotli_blob_delta.py:117` `_real_sensitivity_and_advisory_distortion` | B-STE | Same-line manual-STE conformance. | Forward/gradient equivalence; strict clean. |
| 20 | `experiments/probe_margin_hinge_vs_l7_softplus_grad_geometry.py:92` `_roundtrip` | B-STE | Same-line manual-STE conformance. | Forward/gradient equivalence; strict clean. |
| 21 | `experiments/probe_muon_vs_adamw_from_stage4.py:119` `_roundtrip` | B-STE | Same-line manual-STE conformance. | Forward/gradient equivalence; strict clean. |
| 22 | `experiments/probe_polynomial_fill_survival_gate.py:209` `_eval_roundtrip_t` | A-file | Added a scoped read-only entry; the file is a forward-only fill survival probe. | File consumer trace; strict clean. |
| 23 | `experiments/probe_polynomial_fill_survival_gate.py:211` `_eval_roundtrip_t` | A-file | Same file-level disposition as #22; the STE/hard selector is measurement-only. | File consumer trace; strict clean. |
| 24 | `experiments/probe_r10_lever_interaction_sign.py:96` `_render_roundtrip` | B-STE | Same-line manual-STE conformance. | Forward/gradient equivalence; strict clean. |
| 25 | `experiments/probe_r12_combined_seg_gradient_floor.py:104` `_render_roundtrip` | B-STE | Same-line manual-STE conformance. | Forward/gradient equivalence; strict clean. |
| 26 | `experiments/run_pr95_vendored_vs_mlx_port_parity.py:180` `_torch_forward_loss` | B-STE | Same-line manual-STE conformance on the differentiable parity path. | Forward/gradient equivalence; strict clean. |
| 27 | `experiments/smoke_pose_film_cpu_disambiguator.py:79` `_roundtrip_to_eval_bhwc` | A-fn | Preserved clamp/round/cast hard-forward behavior with explicit detachment; no file exemption because the smoke trains elsewhere. | Forward equality and zero-gradient hard-branch check. |
| 28 | `experiments/smoke_pose_film_cpu_disambiguator.py:95` `_roundtrip_to_eval_bhwc_differentiable` | B-STE | Confirmed false positive; conformed the existing manual STE to one expression. | Forward/gradient equivalence; strict clean. |
| 29 | `src/tac/torch_vehicle/tests/test_film_trunk_decoupling.py:131` `_roundtrip` | C-test | Conformed the fixture's manual STE without changing its forward or gradient. | Fixture suite passes; strict clean. |
| 30 | `src/tac/torch_vehicle/tests/test_kd_warm_start.py:458` `_roundtrip` | C-test | Conformed the fixture's manual STE without changing its forward or gradient. | Fixture suite passes; strict clean. |

## Gate controls and verification

- Authoritative pre-edit verbose scan: 30 scanner instances in 22 files.
- Negative control after the cure, with incremental cache disabled:
  `check_no_bare_round_in_eval_roundtrip(strict=True, verbose=True)` returned
  `0` and printed `[no-bare-round-roundtrip] OK: 0 violations`.
- Positive control: a temporary in-scope
  `experiments/pf4x_bare_round_positive_control.py` containing
  `F.interpolate` plus a roundtrip-named function and bare `up.round()` produced
  exactly 1 violation; strict mode raised `MetaBugViolation`. The fixture was then
  removed.
- Final negative control after removal: strict returned 0 again. The positive-control
  file is absent.
- Gate source-index equivalence test: `1 passed`.
- Edited torch-vehicle fixture suites: `43 passed`.
- `compileall` across all 21 edited Python files: green.
- Seeded equivalence check: 1,000,009 values for each of float32 and float64,
  including NaN and infinities, had exact hard-forward equality; the manual-STE
  gradient and explicitly hard branch matched their pre-edit contracts.
- `ruff` current tree: 16 existing findings; `HEAD`: the same 16 findings; delta 0
  and none falls on a changed line.
- `git diff --check`: green.
- Two genuine post-scan review passes were marked for all 21 edited Python files;
  per-file review policy checks report 0 failures. No review override was used.

## Gate integrity and scope boundaries

No regex, AST traversal, scan root, or strictness was changed. No blanket exemption
was added. The two new entries in `_BARE_ROUND_READONLY_FILES` each have a concrete
one-line rationale and were admitted only after whole-file consumer tracing. The
other 19 production/probe files remain scanned.

No call to `Uint8STE.apply` was necessary: the differentiable paths already encoded
the canonical manual-STE map

`q(x) = x + (round(x) - x).detach()`

and the edit only made that existing map visible to the gate's current same-line
recognizer. Forward-only branches remain hard-rounded and detached. Therefore there
is no changed score-relevant training semantics and no banked-result invalidation.

This arm did not touch `upstream/`, protected pointer/hot-state files, scorer code,
archives, or the staged index. It did not launch r60 because MAIN owns the next full
preflight chain and the arm charter explicitly forbids this sandbox from doing so.

## Ledger and source custody

Task `1305` was registered and moved to `in_progress` before editing. Its completion
event records the 30-to-0 result, green controls, and the source intended commit below.

The mandatory serializer could not write Git objects in the managed sandbox:
`git add` failed with `Operation not permitted`. The first fallback attempt correctly
refused `/Volumes/VertigoDataTier` because projected use would violate the 40 GiB
reserve. The retry retained the complete source landing on `/Volumes/APDataStore`:

- status: `BUNDLE_READY_MAIN_MUST_LAND`
- required base: `9b9379ecdc57255465c79b31d4d0a2dc1a655dce`
- intended source commit: `4dccb5329da09e4f1dea5407598bb0b62fb15a24`
- bundle: `/Volumes/APDataStore/pact/ddm_pf4x_serializer_fallback/20260826T233201.042047Z-37023/intended-commit.bundle`
- bundle SHA-256: `19012da97f63bc20f3d4fdac2a632243749a8fb5ff4eb58fb1f3b6d7ef6e06ab`
- intended tree patch SHA-256: `83597a0b8d525cc3f7c6125abe5c8d83c27a31e4cd40039139c9537934acf17d`
- format patch SHA-256: `245393cdc3621b0ea3637750e06215f0cf806ba12fecf1253a27322c16c3c0cc`
- receipt: `/Volumes/APDataStore/pact/ddm_pf4x_serializer_fallback/20260826T233201.042047Z-37023/receipts.jsonl`

`git bundle verify` reports the bundle complete and requires exactly the recorded
base. The serializer used isolated Git plumbing without a checkout, so no hidden
fallback worktree was created. The evidence memo and completed ledger row form a
second serializer landing; its self-referential custody identifiers are reported by
the final handoff and receipt generated after this document is finalized.

## GESTALT-DELTA

Before PF4X, r59 exposed a stale population described as approximately 25 bare-round
sites, with unknown differentiation integrity. After PF4X, the denominator is exact
(30 scanner instances / 29 physical occurrences / 22 files), every site has a typed
consumer-backed disposition, zero severed gradient paths were found, the strict gate
is clean in both source-index modes, and an executed positive control proves the
detector still catches the forbidden class.

## Typed r60 fire-order for MAIN

**Disposition:** `QUEUED-WITH-A-FIRE-ORDER`  
**Owner:** `MAIN`  
**Consumer store:** `.omx/tmp/preflight_full_r60_20260826/PREFLIGHT_RESULT.json`  
**Fire trigger:** MAIN has verified and landed intended source commit
`4dccb5329da09e4f1dea5407598bb0b62fb15a24` from its required base, then rerun the
bare-round strict check and observed zero.

At the trigger, create `.omx/tmp/preflight_full_r60_20260826/run.py` by copying the r59
runner and changing only the result path from `preflight_full_r59_20260826` to
`preflight_full_r60_20260826`. Then launch the full chain with:

```sh
.venv/bin/python tools/launch_detached_process.py \
  --output-dir .omx/tmp/preflight_full_r60_20260826 \
  --cwd /Users/adpena/Projects/pact \
  --purpose 'pf4x r60 full-preflight chain after bare-round cure' \
  --authority 'local apparatus validation; no score authority' \
  --done-receipt pf4x_r60 \
  -- .venv/bin/python .omx/tmp/preflight_full_r60_20260826/run.py
```

The runner must call `preflight.preflight_all(wall_clock_budget_s=None)`, atomically
produce the named JSON result by process completion, retain RED exception type/message
and traceback as r59 did, and print the terminal status. PF4X makes no claim about the
next gate r60 may expose.

## NEXT_IF_RESUMED

- `BUNDLE-READY-MAIN-MUST-LAND` — owner `MAIN`; consumer store
  `/Volumes/APDataStore/pact/ddm_pf4x_serializer_fallback/20260826T233201.042047Z-37023/intended-commit.bundle`;
  fire trigger: MAIN is on or can prove compatibility with required base
  `9b9379ecdc57255465c79b31d4d0a2dc1a655dce` and has verified the recorded SHA.
- `QUEUED-WITH-A-FIRE-ORDER` — owner `MAIN`; consumer store
  `.omx/tmp/preflight_full_r60_20260826/PREFLIGHT_RESULT.json`; fire trigger: the source
  bundle is landed and a fresh strict bare-round check returns zero.

## LIVE-HYPOTHESES

- A structural AST/dataflow recognizer for manual STEs could eliminate future
  line-wrap false positives while retaining the current negative guarantee. This is
  plausible because 19/30 instances were the same expression split over lines, but it
  remains untested and was intentionally outside this no-regex-change charter.
- Mixed-purpose probe files will remain the highest-risk source of ambiguous future
  findings. Six forward-only functions lived inside files that trained elsewhere, so
  filename-level classification would be wrong even though function-level consumer
  tracing closed these instances.

## DEAD-ENDS

- Blanket `probe_*`/`measure_*` exemptions are closed: most reported files perform
  optimization somewhere, and only two whole files passed the read-only contract.
- Replacing every finding with `Uint8STE.apply` is closed for this population: no
  differentiable path was severed, and wholesale replacement would invent an
  integrity defect rather than preserve the existing mechanism.
- Banked-result invalidation is closed: the 19 differentiable instances already had
  correct identity-gradient manual STEs, and the other reported functions were
  explicitly hard/forward-only.
- Weakening or widening the gate's exemption vocabulary is closed: the executed
  positive control fired after the cure and the strict negative control returned
  zero.
- `/Volumes/VertigoDataTier` is closed as this landing's fallback destination until
  its free space again satisfies the serializer's 40 GiB reserve; APDataStore already
  holds the verified source bundle.

**Own-vehicle frontier:** unchanged — GB1, `S = 0.14811799921260607` at `180,215 B`
`[contest-CUDA T4, n600]`, archive SHA-256
`ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4`.

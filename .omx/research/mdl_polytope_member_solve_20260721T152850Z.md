# MDL polytope-member solve — measured n16 → n64

**research_only=true · [macOS-CPU advisory] NON-PROMOTABLE · pointer unchanged**

**Verdict:** `MEASURED_NO_MEMBER_RATE_CUT_RETAIN_CANONICAL` on `[macOS-CPU advisory]`. The bounded chart/object-first full-kernel formulation preserved the solved receiver outputs exactly, but no noncanonical member beat canonical support-fill under the same coder. The contest-CPU pointer remains **0.1910828242 UNMOVED**; this memo makes no score or promotion claim and requires MAIN landing review.

## Result first

| Row | n64 measurement | Disposition |
|---|---:|---|
| D1 proxy calibration | 192 real tile/candidate rows; Pearson **0.945194**; Spearman **0.879867**; MAE **10.321 B** | PASS as a rank proxy for this candidate family |
| D2 exact feasibility | **64/64** exact resize-numerator pairs; **64/64** zero-radius PoseNet-output tube equality | PASS |
| D2 frozen oracle | canonical = selected: `d_seg=0.00012636184692382812`, `d_pose=0.000060022091887905524`; deterministic CPU Torch seed 1234, batch 32 | PASS, identical distortion |
| D3 raw-member diagnostic coder | canonical **77,651,017 B**, selected **77,651,017 B** | **0 B / 0.0% cut** |
| D3 level attribution | chart **0 B**; class/stratum object **0 B**; tile/pixel residual **0 B** | all proposals coder-rejected |
| D3 #557 seed coder | seed **78,969 B** zlib-9; seed + video-derived calibration **79,385 B** | policy is **+416 B overhead**, not a saving |
| D4 >15% gate | **not met** | no n600 estimate; no FEED into #541 |

The n16 gate gave the same structural result: 16/16 exact and oracle-identical, with a 0% cut. Its prefix distortions were `d_seg=0.0001312891642252604` and `d_pose=0.00004116360380325019` under the same batch-32 scorer custody.

## What was solved

`MdlPolytopeMemberSolver` searches only inside the exact integer resize fibre. It builds the complete three-dimensional local integer-kernel basis supplied by #580, takes a rounded least-squares coordinate step toward a structured preference, and uses deterministic dyadic backtracking until every cell/channel lies in `[0,255]`. Every candidate is then rechecked with `DisjointResizeOperator.apply_numerators`; a mismatch is a hard error.

The search obeys the operator's later levels hint:

1. **Chart level:** one coherent whole-frame move (`zero`, horizontal, vertical, neighbor-mean, or an unwarped within-pair temporal preference).
2. **Object level:** one coherent preference per majority-class × Morse–Smale stratum group.
3. **Pixel/tile residual:** tile-local integer moves only after the coherent levels.

Each level must strictly reduce the same zlib-9 coder. Ties retain the earlier level. On all 128 frames in the n64 prefix, the Pareto-safe winner was canonical at every level. The class/stratum diagnostic is scoped to frame 1, the SegNet-authority frame: per-class deltas for classes `0/2/3/4` were all zero, and per-stratum deltas for `cell/edge/saddle` were all zero. Those independently compressed tile totals are diagnostic and are not additive to the whole-frame coder.

## D1 objective and calibration

The proxy is directly computable from a candidate tile: piecewise-constant break count, spatial gradient L1, exact-kernel move L1, within-pair temporal residual L1, nonzero count, and alphabet cardinality. The field name `temporal_xi_l1` is task-local shorthand only: this is an unwarped paired-frame feature, not an explicit SE(3) ξ warp or a measurement of pose-factorization quality. It was fitted on 192 deterministic real tile/candidate samples against actual zlib-9 bytes. This exceeds the delegated ≥20-tile floor; the measured correlation is strong enough to rank this bounded family, but it carries no cross-coder or global-MDL authority.

## Description-level finding

The current receiver output is not merely equivalent to canonical support-fill: every checked source frame is byte-identical to `realize_factor2_uint8` applied to its exact scorer plane. The receiver packet describes that scorer plane and derives the camera member. Therefore a different deterministic preimage member does not remove scorer-plane or seed fields by itself. The #557 seed already compresses to 78,969 B, while serializing the video-derived proxy coefficients would add 416 B. The requested 154,600 B and 121 KiB boxes are met by the **seed alone**, but that seed is not a complete solved-pair archive; treating it as one would be a false archive/score claim.

This falsifies only the tested premise that incidental canonical-fill camera detail is a charged description gap under the current packet/seed formulation. It does **not** kill MDL-in-the-loop. The untested optimal-form queue is: direct polynomial chart-coefficient/event-symbol optimization in the counted seed; an explicit SE(3) ξ-warped temporal chart; joint scorer-plane packet/member entropy optimization under the real archive coder; and curvelet/shearlet residual symbols only after charged chart/object terms are exhausted. Every successor still owes the same hard oracle. Merely choosing a different free deterministic lift of a fixed scorer plane cannot pay rent.

## REUSE MANIFEST

| Required surface | Disposition | Evidence |
|---|---|---|
| #49/S12 `resize_null_preimage_compiler` | COMPOSED | same coder-admission principle; no forked resize law |
| #580 full-kernel projector | REUSED DIRECTLY | 80.6742315% exact structural nullity; exact numerator certification |
| #547/#549 joint lattice solve | REUSED READ-ONLY | current exact two-plane raw is the constraint surface |
| #557 seed coder | REUSED DIRECTLY | `parse_constraint_seed` → `serialize_constraint_seed` byte-identical; zlib-9 measured |
| 114/114 margin-survival alphabet | SETTLED | no remeasurement or RGB rewrite |
| #586 lattice-enumeration crosswalk | COMPOSED | deterministic proposal ordering only; no global optimum claim |
| New member solver/tool | JUSTIFIED | #580 had no calibrated tile proxy, levels hierarchy, class/stratum attribution, or resumable n16/n64 oracle receipt |

All numeric solver defaults appear in the receipt's LawRef manifest. The task-scoped tile size, sample count, and ridge are explicit hardcoded waivers with re-derivation triggers; geometry, zlib-9 custody, and the rate threshold cite registered laws.

## Triality and system intelligence

- **DSL/execution:** `tools/measure_mdl_polytope_member.py` is a fail-closed n16/n64 CLI with SHA-pinned inputs, storage preflight, atomic config-bound pair stages, and batch-32 scorer custody.
- **DAG:** `FEED-mdl-member-20260721` records the measured no-op and routes no rate term to #541 because the >15% gate failed.
- **Equations:** `separable_resize_full_kernel_direct_sum_v1`, `realization_necessity_preimage_per_stratum_v1`, `partition_temporal_transport_amortization_jitter_bound_v1`, and `witness_measured_reverse_waterfill_v1` govern geometry, grouping, proxy coder, and admission.

No copied bulk was produced. The 3.66 GB source raw stayed read-only. Sixty-four small pair JSON stages are preserved under the SSD evidence root; the interrupted batch-1 prototype stages were moved intact to `stages_superseded_batch1_20260721T1519Z` rather than deleted.

## Custody

- Full n64 receipt: `/Volumes/VertigoDataTier/pact/evidence/mdl_member_20260721/receipt_n64.json`, SHA-256 `b71ad6ab036b62fca640c4dc1a76fa37eec404901f6854239d35eb5c90d803a3`.
- Full n16 receipt: `/Volumes/VertigoDataTier/pact/evidence/mdl_member_20260721/receipt_n16.json`, SHA-256 `6aff269708303545d66970b898e1667d678b2f139ca486acf0de433be2ef1b92`.
- Proxy calibration: `/Volumes/VertigoDataTier/pact/evidence/mdl_member_20260721/proxy_calibration.json`, SHA-256 `81e7fa67a86c6b129d0f23fbc4fcd3abb023cd0b24bc55cb9299b1d11a98d015`.
- Compact tracked receipt: `.omx/research/mdl_polytope_member_solve_receipt_20260721.json`.

## Verification

- `pytest`: **52 passed, 1 skipped** across the MDL-member, full-kernel, null-preimage, and structural-law suites.
- `ruff`, `py_compile`, `git diff --check`, compact/full receipt cross-check, and 20-seed exactness/coder-monotonicity property probe: **PASS**.
- `review_tracker`: `mdl_polytope_member_structural_pass` and `mdl_polytope_member_evidence_pass`, **58 Python entities each**, reviewer `codex`; no override.
- `lane_maturity validate` still reports **110 pre-existing missing-evidence-path errors outside this lane**. This lane itself is registered `research_only=true`, L1 with only `impl_complete` marked; no empirical archive/contest/three-clean gate was claimed.

## STORES CONSULTED

`reports/latest.md`; `.omx/state/lane_registry.json`; `.omx/state/subagent_progress.jsonl`; `.omx/state/master_gradient_anchors.jsonl`; `.omx/state/modal_call_id_ledger.jsonl`; `.omx/state/cost_band_posterior.jsonl`; `.omx/state/continual_learning_posterior.jsonl`; `.omx/state/artifact_quarantine.json` (absent, tracked default set used); latest Codex findings/session and Claude council/design memos; the v7.5/v8 specs; current v10 exact-lattice/two-plane receipts; both live inboxes.

## Pointer delta

**No pointer delta. No score. No promotion. MAIN must review the branch diff and custody before merging.**

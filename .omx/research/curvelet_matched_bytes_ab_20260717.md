# p0_497 curvelet matched-bytes A/B — respawn recovery + composition-gap closure (2026-07-17)

`lane_id=p0_497_basis_cure_decisive_ab`; `research_only=false`; `training_launched=false`;
`live_c2_touched=false`; **pointer 0.19108 UNMOVED** (the live c2 run
`experiments/results/levelset_n600_witness_20260717T113932Z` is the current pointer-mover attempt;
this unit is MEANS toward the queued decisive basis A/B).

## STEP 0 — recovery (no signal loss)

The predecessor p0_497 arm died on a model-credit limit after ~40 tool uses. Recovery audit:

- Worktree `.omx/tmp/codex_worktrees/p0_497_curvelet_ab_20260717` (branch
  `claude/p0_497_curvelet_matched_bytes_ab_20260717`, base `6f96a94a30`): **clean, zero commits** —
  the dead arm had landed nothing; its work was recon only. Nothing to rebuild, nothing lost.
- All prior curvelet-crux harvest is ALREADY ON MAIN (merged `98b7896bc6`, worktree
  `curvelet_crux_harvest_complete_20260716T192419Z` commits `9761c621d7`/`4d2547a39c`/`10f9095739`):
  literal polar-curvelet dictionary + strict `BasisProgramConfig` + generated receiver + equal-byte
  law (`src/tac/through_r/equal_archive_budget.py` + `tools/curvelet_equal_byte_ab_receipt.py`).
- The two NAMED composition gaps (per `.omx/research/curvelet_optimal_form_crux_completion_20260716.md`
  §Exact remaining closure) were re-located precisely:
  - gap (a) ground-chart receiver semantics: fail-closed at
    `tools/levelset_byte_close_and_eval.py` ~L3243 ("receiver-sealed fast nonuniform transform"
    owed; direct sparse eval is n600-prohibitive at FREQUENCY_LATTICE_RADIUS=160) + trainer gate
    `experiments/train_levelset_witness_realized_through_R_mlx.py` ~L4365.
  - gap (b) post-render supersampling A_s: fail-closed at byte-close ~L3250 + trainer native×ss
    gate ~L4370.

## This unit's landings (worktree branch `claude/p0_497_curvelet_matched_bytes_ab_20260717`)

1. **DSL leg (commit `45a0562524`)**: `LiteralPolarCurveletBasis` Lever factory in
   `tac.witness_dsl.curriculum_dsl` — literal family + same-width native orientation
   (kappa=2.0, fixed-point cap 6), `--render-aa none` (scalar IPE is fail-closed for the literal
   family per the crux SPEC — the sealed base config's `ipe` would refuse), `--self-orient false`
   (equal-value precondition). Auto-discovered composable by `lever_registry` (verified); every
   emitted flag verified against the levelset trainer argparse (never-invent-flags).
2. **Queued fire script (same commit)**: `tools/fire_curvelet_matched_bytes_ab_p0_497.py` —
   PREPARED_NOT_FIRED. Gates in order: c2-completion (psutil pid scan + 30-min run-dir quiescence;
   VERIFIED live: it currently refuses with 7 trainer pids alive), governed
   `launch_witness_run.py --dry-run` per arm (governor/storage/seal/DSL refusals block),
   `--operator-go` required (CONTAINMENT), arms strictly sequential (memory: each arm ~63-75 GiB).
   NO admission override path exists in the script by construction.
3. **Gap (b) CLOSED — post-render supersampling A_s (commit `234daa3a21`)**: sealed semantics
   `Y = R[A_s G(Phi(X_s))]` implemented op-for-op in BOTH `numpy_oracle_reference_frames` AND the
   shipped `_INFLATE_PY` receiver — the whole feature program (literal feats, self-orient dir
   feats, native fixed-point argmax) runs at the fine `(ss*rh, ss*rw)` grid; A_s = exact ss×ss box
   average (receiver-inline `_aa_down` bit-identical to `box_downsample_np`) applied AFTER the
   nonlinear renderer to rgb/lane_rgb/margin, BEFORE base-grid lane compositing (#220
   compose-at-base) and BEFORE R; store_nothing pose-carrier frame0 mirrored. **Closure proof
   (NO-FAKE): `bit_exact_roundtrip_gate` STRICT at aa_factor=2 — shipped inflate uint8 == numpy
   fp32 oracle uint8** + s=1 bit-identity vs aa-none + nontriviality (ss=2 ≠ none). 10 new tests;
   91+26 regression tests green. Still fail-closed (narrowed, honest): native-orient×ss (trainer
   cannot produce that checkpoint), tex_trunk×ss (base-grid trunk bank would shape-mismatch — a
   NEW gap found and pinned during implementation).
4. **Gap (a) CORE landed — counted ground-chart receiver program (commit `f93b7517f4`)**:
   `BasisProgramConfig.chart_eval_semantics="charted_grid_bilinear_v1"` + `chart_fine_factor`
   custody (enabled chart REQUIRES the sealed semantics; legacy enabled dicts refuse; legacy
   disabled dicts load hash-identical); `GroundFrameChart.build_from_xi` (counted-receiver-program
   entry over the DEQUANTIZED table; == `build()` field-for-field by test); sealed border-clamped
   fp32 bilinear evaluator + `charted_pair_feats_numpy` (identity pair = EXACT uncharted program;
   stdlib+numpy only → receiver-embeddable); trainer xi-plumbing landed WITH a design correction:
   the SPEC's preferred `counted_pose_carrier_xi` is structurally impossible for a startup-static
   chart (carrier `xi_eff = xi_stored + TRAINED dxi` does not exist at ep0) → the literal chart
   binds **`counted_chart_payload`**: startup pose table quantize→dequantize, chart built on the
   dequantized values, custody `__chart_pose_q` (int16 (P,6)) + `__chart_pose_scales` in the
   checkpoint; new `--literal-chart-fine-factor` flag; margin-compander fail-closed for literal
   (non-homography). **Measured receipts** (`charted_grid_bilinear_v1_receipt_20260717.json`,
   `[macOS-CPU advisory]`, non-promotable): per-pair sealed eval 0.043 s → n600 ≈ 27 s
   EXTRAPOLATED vs direct-sparse ≈ 71 h EXTRAPOLATED (the byte-close gate's "n600-prohibitive"
   claim is now MEASURED); accuracy vs the exact sparse polynomial on 2000 charted points: in-box
   mean 2.5e-4 / p99 7.6e-4 / max 0.44 (≈5% of the amplitude-8 scale-4 atoms; factor sweepable);
   semantics is SEALED-AS-PROGRAM — trainer and receiver share the identical source, so the delta
   vs the pure polynomial cancels in the A/B by construction. 42 focused tests green.
5. **Gap (a) CLOSED end-to-end — byte-close + generated-receiver chart wiring (commit
   `6907a8782d`)**: the counted chart payload is the 7th optional LVLS1 blob section (quantized
   int16 (P,6) startup pose table + fp32 (6,) scales; **60 B on the 3-pair fixture, 7,224 B at
   n600** — honest counted rate, no hide-data-in-code); loader custody refuses missing
   `__chart_pose_q`/`__chart_pose_scales`; gp-capped repack slices chart rows exactly; the oracle
   rebuilds via the REAL `GroundFrameChart.build_from_xi` + sealed evaluator; the SHIPPED receiver
   inlines a verbatim chart builder (camera constants as literals, **BIT-EXACT vs `build_from_xi`
   by test**, constants pinned against `tac.clip_profile`/`tac.camera` at test time) and evaluates
   per-pair feats through the embedded sealed evaluator; chart × native-orient sealed on both
   sides via the J⁻ᵀ normal-covector transform. **Closure proofs (NO-FAKE):
   `bit_exact_roundtrip_gate` STRICT with a NONTRIVIAL chart AND with chart × native-orient**
   (shipped inflate uint8 == numpy oracle uint8) + nontriviality + capped-inflate preservation +
   3 gate-pinning refusals. 14 new tests; **217 passed / 14 skipped** across the chart +
   supersample + pose-carrier + receiver-harden + e2e byte-close + equal-archive-budget + basis /
   placement / ground-chart suites (the 4 `test_probe_jrd` failures are pre-existing environmental
   — worktree lacks upstream models/videos — verified identical at the committed state).
6. **MLX/Metal parity re-run** (completion-memo owed step 3): executed on THIS Metal host (M5 Max)
   in the worktree — the previously-deselected Metal parity test now **passes** (1 passed).

## Honest residual gates (pinned by tests, NOT closure debt of this unit)

- native-orient × supersample and chart × supersample: fail-closed on all surfaces (the TRAINER
  cannot produce those checkpoints; fine-grid native-gate + charted-fine composition semantics are
  a future formulation, refused loudly, never silently wrong).
- tex_trunk × supersample: NEW gap found + pinned by Fork A (base-grid trunk bank would
  shape-mismatch a fine render).
- `counted_pose_carrier_xi` chart dependency: unsatisfiable at ep0 (Fork B's measured design
  correction — the chart binds `counted_chart_payload` instead; stale gate message corrected).
- Pre-existing (main, NOT this branch): `tools/levelset_pose_gate.py:209` unpacks a 4-tuple from
  `_read_blob_bytes` (already broken against the pre-branch 6-tuple API) — flagged for triage.
- Pre-existing oracle/receiver asymmetry flagged by Fork A for audit: `levelset_rgb_forward_numpy`
  has no tex_trunk/out_tex_h/decoupled handling while the receiver auto-detects them.

## The queued decisive measurement (fires post-c2, operator-GO)

```
# per arm (sequential; control first):
.venv/bin/python tools/fire_curvelet_matched_bytes_ab_p0_497.py --arm control   --operator-go
.venv/bin/python tools/fire_curvelet_matched_bytes_ab_p0_497.py --arm treatment --operator-go
# then: independent byte-close per arm -> tools/curvelet_equal_byte_ab_receipt.py match ->
# inflate both matched archives -> official scorer n600 batch32 on a declared contest axis ->
# finalize -> curvelet_equal_archive_transfer_v1 verdict (formulation-instance; pointer_authorized=false)
```

Branch merges to main POST-c2-fire per CFL worktree discipline.

## Verdict-scope honesty

- No new n600 d_seg/d_pose/rate/exact rows were measured by this unit (measurement is QUEUED, not
  fired — firing beside the live c2 would violate the admission gate; a queued fully-prepared
  measurement with zero signal loss beats an override).
- The prior curvelet rows (`d_seg=0.5048` saved-OFF receiver formulation) remain
  MEASURED_ADVISORY / equal-values-not-equal-bytes; the curvelet FAMILY remains OPEN.
- Pointer 0.19108 UNMOVED.

## Stores consulted

CLAUDE.md + worktree CLAUDE.md; `.omx/research/curvelet_optimal_form_crux_completion_20260716.md`;
`curvelet_optimal_form_crux_20260715_SPEC.md`; `curvelet_throughR_p0_launch_ticket_20260715.json`;
`shearlet_vs_curvelet_family_selection_20260714.md`; genuine_curvelet_shearlet_* receipts (20260714);
subagent_progress.jsonl predecessor rows; p0 ledger p0_497 rows; the live c2 run dir (read-only).

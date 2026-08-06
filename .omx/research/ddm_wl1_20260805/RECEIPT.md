# ddm_wl1 receipt - 2026-08-05

Arm: `ddm_wl1`
Purpose: complete witness-lineage harvest for TR1/JD/endgame transfer, successor to `ddm_vh1`.
Axis: scorer-free read-only harvest. No new score measurement.

Own-vehicle frontier: `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory] n600`.
Contest pointer: `0.1910828242 [contest-CPU]`, borrowed, unmoved.

## Governing Reads

- Read `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`, the wl1 charter, and the common arm contract before writing artifacts.
- Read predecessor harvest first: `.omx/research/ddm_vh1_v8v9v10_harvest_20260730.md:1-201`.
- Honored protected-file list. I did not edit `.omx/research/ddm_cr1_composition_row_827_20260801.md`, `.omx/research/ddm_pu2_pose_tail_floor_probe_20260803.md`, or `src/tac/optimization/direct_description_carrier_compose.py`.
- `src/tac/optimization/direct_description_carrier_compose.py` was already dirty before wl1 edits; it was not touched.

## RECALL EVIDENCE

Sources searched/read beyond the charter seed list:

| query / source | result | what changed |
|---|---|---|
| `rg -n "ddm_vh1|v8v9v10|p0_521|v10_capstone|negative_cure|witness_native_schedule|telemetry_enhancement|v9_cgauge|spec_v10" MEMORY.md` | Found prior T1/CGauge verification memories and parser-truth caveat. | Added PORT-NOW row 14: parser-valid config is not equivalent to compiled LawRef consumption. |
| `.omx/research/ddm_vh1_v8v9v10_harvest_20260730.md` | Rows 1-16 plus explicit nothing-more scope and pointer unmoved. | Created a folded/not-reclaimed section so wl1 does not double-count vh1 rows 3/4/9/13 or already-adjudicated v8/v9 items. |
| `.omx/research/witness_native_schedule_derivation_20260709.md` | One continuous `L_tau`, geometric tau, event-based ladder, run-1 no trajectory, restart A/B owed. | Added rows 4, 20, 22, 26; ranked unified `L_tau` as RACE, not adoption. |
| `.omx/research/negative_cure_join_table_20260710.md` | Ranked negative/cure joins including lane-band training, exact `S_R`, eikonal fixed-guard reopen, full-P solve, local saliency, coder re-price. | Drove the top three next-arm charters and all PRECONDITION-CHANGED reopen rows. |
| `.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md` | v7.5 pose blocker, P0 force isolation, operating contract, micro-batch gate, registry gap warning. | Added rows 21, 23, 24, 29, 30 and kept pose-sidecar transfer dead. |
| `.omx/research/SPEC_v8_perclass_decomposition_20260708.md` | Edge-centric carrier doctrine, chroma-first/luma-reserved correction, class-naive refutation, P-C gate. | Added rows 19, 25, 34 while folding already-vh1 v8 doctrine instead of re-claiming it. |
| `src/tac/witness_dsl/spec_v9_cgauge.py` | T1 phase-advection, annulus dwell satisfiability, basis custody, self-orient false. | Added rows 9, 10, 11, 14, 27. |
| `src/tac/witness_dsl/spec_v9c3_duty_ab_20260719.py` | v9c3 duty A/B fork geometry, no-Muon purity, adverse findings surfaced. | Added rows 12, 13, 18 and prevented v10/capstone conflation. |
| `.omx/research/ddm_tp1_v9_telemetry_port_DAG_FEED_20260731.md` | v9 telemetry port to TR1 is DONE; Q3 live-gap was intentionally not ported. | Folded tp1 as done; added row 15 only as a costed RACE gap. |
| remote branch `remotes/origin/claude/p0_521_spec_v10_capstone_20260717` | Read V10 capstone spec from git object and local skeleton. | Added rows 5-8 and kept v10 config fail-closed until gate artifacts exist. |
| `lever_registry.completeness()` | Current checkout: trainer_total 443, dsl_referenced 372, unmapped_count 80, stale_count 3. | Added row 30; did not claim full anti-orphan coverage. |
| `spec_v10_status(Path("."))` | clear false; post-merge levers 5/5 resolved; gates 0/6 present; seeds 1/2 present; 7 blockers. | Added row 5 as a PORT-NOW gate/status surface, not a config. |
| `tools/list_canonical_equations.py --json` | Registry available with 424 entries in current checkout. Output is broad; no new equation was minted. | Used as recall coverage only; no table row depends on a new equation claim. |

Scoped negative recall: I did not find a wl1-specific prior memo in memory or `.omx/research/ddm_wl1_20260805/` before this landing. I did find predecessor vh1/tp1 surfaces and folded them.

## Commands / Current Facts

Preflight:

```text
git rev-parse HEAD
954d21db228bbc991b8406537ee29601dbc301a9

git branch --show-current
main
```

Dirty-worktree custody:

```text
git status --short -- .omx/research/ddm_wl1_20260805 src/tac/optimization/direct_description_carrier_compose.py .omx/research/ddm_cr1_composition_row_827_20260801.md .omx/research/ddm_pu2_pose_tail_floor_probe_20260803.md
 M src/tac/optimization/direct_description_carrier_compose.py
?? .omx/research/ddm_cr1_composition_row_827_20260801.md
```

The protected modified/untracked paths above were pre-existing and were not edited by wl1.

Live-run boundary:

```text
ps -p 13924 ...
operation not permitted

kill -0 13924
operation not permitted
```

I could not verify jd6 pid 13924 directly in this sandbox, so I used the live board as scoped authority and stayed scorer-free.

Registry check:

```text
trainer_total 443
dsl_referenced 372
unmapped_count 80
stale_count 3
stale --integer-plane-emitter-basis,--integer-plane-emitter-mode,--integer-plane-emitter-policy-sha256
trainer_path /Users/adpena/Projects/pact/experiments/train_levelset_witness_realized_through_R_mlx.py
```

V10 status check:

```text
clear False
post_merge_resolved 5 / 5
gate_present 0 / 6
seed_present 1 / 2
blocker_count 7
gate:v9c2_completion
gate:p0_497_curvelet_ab_verdict
gate:warmup_8v27_ab_verdict
gate:probe_p1_n600_band_and_terminal_decomp
gate:probe_p2_mirror_transport_rate
gate:probe_p3_chroma_plane_jacobian
seed:hood_tex_seed
```

Canonical equations recall:

```text
tools/list_canonical_equations.py --json
total entries observed by JSON parse: 424
```

## Deliverables

- Durable transfer memo: `.omx/research/ddm_wl1_20260805/TRANSFER_TABLE.md`
- Resume note: `.omx/research/ddm_wl1_20260805/NEXT_IF_RESUMED.md`
- This receipt: `.omx/research/ddm_wl1_20260805/RECEIPT.md`

## Follow-On Dispositions

FIRED:
- Wrote the ranked table and three concrete next-arm charters in `TRANSFER_TABLE.md`.
- Recorded V10 gate status from current code without compiling a fake config.
- Recorded witness registry completeness counts without claiming closure.

FOLDED:
- vh1 rows 1-5, 9-10, 13 and associated branch amendments are folded as predecessor work.
- tp1 v9 telemetry port is DONE and not re-claimed.
- wp1 Muon/MC-finisher named units are not duplicated.
- la1 LR-anneal and dy2 tail-EMA are treated as already in-flight/done surfaces, not wl1 rows.

QUEUED-WITH-FIRE-ORDER:
1. WL1-LB analytic lane band training-lever A/B.
2. WL1-SR exact through-R reachability weighting A/B.
3. WL1-EIK fixed-guard eikonal/viscosity fair re-open.

No scorer job was launched. No n8/prefix evidence was banked.

## Verification Status

Markdown artifacts were written only under `.omx/research/ddm_wl1_20260805/`.
Post-write diff/check/serializer status is recorded by the final command transcript and final response.

Pointer delta: none. Own-vehicle frontier remains `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory] n600`.

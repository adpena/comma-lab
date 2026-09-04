# CHARTER ddm_fs2 — re-solve the pose carrier on the 21 pairs whose frame 0 moved (fs1's NEXT step 2–3), byte-optimal, seal for T4

Tokens: `[no-triality] [p0-ledger-ok]`. Owner: Opus arm. Spawned 2026-09-04 ~17:05Z (real UTC). Parent: fs1 (7ec320551), pr1, up2, jg5.

## PRIOR-LAW PREDICTION (owed line)
fs1 moved frame 0 on 21 pairs but their 12-dim carrier codes were fitted against the OLD frame 0 (fs1 §9.2). PREDICTION: a damped
Gauss–Newton re-solve (`ddm_jg5.refine_pair` through pr1's `solve` mode) of ONLY those 21 pairs' codes against the frozen PoseNet on the
shipped renders lowers d_pose on a subset of them at a per-pair byte cost priced by `up2.price_full_resolve_bytes` (whose control
reproduces the shipped 78,628 bits); the byte-optimal frontier (fs1 §4 `adoption_from_sweep(strategy="byte_optimal")`) admits a
non-empty set with net ΔS < −2e-5 (admit bar). Falsifier: every re-solved pair costs more rate than its pose buys (net ≥ 0 for all
adoption sets) — then the same-object pose family on the shipped packet is CLOSED after fs1, and say so.

## Objective (the pointer is the only end)
Produce the next byte-closed exact-row candidate on the SAME object (the fs1 archive, sha 50fcaf1ac3c8504abdf3e0daff7c5bce32104f19d8de4a7ba207816f32e708cf,
180,022 B, S 0.14786319521362173): (1) carrier re-solve on the 21 changed pairs; (2) alternate `selector` ↔ `solve` until neither moves
(fs1 §9.3; selector sweep ~2.3 s/pair); (3) byte-optimal adoption; (4) build the archive through the SHIPPED encoder/decoder path with
fs1's identity control + no-op detector + parse-back + container search; (5) MEASURE n600 batch-8 d_pose (per-pair .npy retained
with sha) and compose the projected S; (6) seal via `tools/make_candidate_seal.py` for contest-CUDA with the single-axis waiver
(same-object pose-only edit; d_seg structurally identical — PROVE the odd-frame sections are byte-identical again). MAIN fires.
PROMOTE IFF exact S < 0.14786319521362173. If the falsifier fires, deliver the CLOSED verdict with the full frontier table.

## Inputs (pinned)
fs1 store `/Volumes/APDataStore/pact/ddm_fs1_frame0_selector/` (`retained/fs1_select.json` 39-point frontier; `retained/candidate_B*/`;
`retained/measure_candB*_n600.json` = the base pose row for THIS object); fire tree `/Volumes/VertigoDataTier/pact/ddm_fs1_frame0_selector/fire_runtime_B_byte_optimal_101/`
(sha fbf4aaf4…); `experiments/ddm_fs1_frame0_selector_reselection.py`; `src/tac/semantic_pipeline/frame0_selector_codec.py`; pr1
(`experiments/ddm_pr1_*` solve mode); up2 (`price_full_resolve_bytes`); jg5 (`refine_pair`). Exact row: `/Volumes/APDataStore/pact/ddm_fs1_frame0_selector/t4_buy_20260904/`.
Memo of record: `.omx/research/ddm_fs1_pointer_move_24_20260904.md`. Bulk goes to Vertigo (166 GiB free); APDataStore has 16 GiB.

## OPTIMAL FORM
Reference form = fs1 + pr1 (the exact instruments that just reproduced the authority to −2.6e-6). No mechanism reduction. Scope: the
21 pairs first (§9.2), then the alternation (§9.3) at n600. The n600 batch-8 CPU measure is the ONLY admissible advisory instrument for
pose here (fs1 proved it); no n<600, no MPS, no proxy. Per-pair receipts retained (exchange-noise law: ±6% ≈ 363 B).

## Rules that bind
NO-FAKE; ALWAYS KEEP THE PAYLOAD (every candidate archive + per-pair .npy retained with sha256 + bytes); upstream/ READ-ONLY; the live
PR tree READ-ONLY; commits ONLY via `tools/subagent_commit_serializer.py --message … --files … --expected-content-sha256 <file>=<post-edit sha>`
with `[no-triality] [p0-ledger-ok]`; NO co-author trailers; .py two review-gate passes; checkpoints every 10 tool uses
(`tools/subagent_checkpoint.py --subagent-id ddm_fs2`); never invent flags (grep argparse); no `/tmp` evidence; long steps detached via
`tools/launch_detached_process.py --done-receipt <distinct name>` (foreground >3 min is reaped rc=144; the launcher refuses argv with
"claude"/"codex"); label every number MEASURED/DERIVED/INFERRED; memo `.omx/research/ddm_fs2_carrier_resolve_on_changed_pairs_20260904.md`
with an "Equations leg (`tac.canonical_equations`)" line; `docs/operating_manual_craft_handoff.md` binds. End with
`fs1 S 0.14786319521362173 @ 180,022 B [contest-CUDA T4 n600]` (+ your projected candidate line, labeled advisory).

# ddm_q43a QA43 tail-targeted pose solve receipt

Axis: `[macOS-CPU advisory]`, `score_claim=false`.
Pointer movement: none. Contest pointer remains borrowed and unmoved.
Run status: `BLOCKED_TYPED_RECEIVER_GRAMMAR_MISMATCH`.

## Verdict First

The tq1c freshness hash passed, but the built su2 QA43 receiver cannot consume the
current frozen parent. The tq1c parent is a single-member `0.bin` archive; the
su2 adapter is literal-bound to the v4d six-member receiver grammar. The adapter
refused before any scorer load:

```text
QA43 REFUSED: v4d-warp parent member order/shape differs
```

Therefore no tail-concentration profile, no k=56 solve, no B/pair row, and no
falsifier verdict were measured by this arm. This is an instance-scoped blocker:
exact parent `b35e756829...` x su2 `warp-tail` adapter x v4d member grammar. It
is not a QA43 family kill.

## Freshness Gate

Required parent from the charter:

```text
/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/phase_b_realized_tq1c/candidate_archives/move_0023_snap_r00_c12_L13.zip.receipt-bytes
```

Observed:

```text
sha256 b35e756829306a85ec2ad51634bde74523d89df9046c682253176b393bd59c06
bytes  357837
```

Important custody distinction: `/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/tq1c_base/archive.zip`
is 357,837 bytes but hashes to `75df9cc341d8188b47fe38337d2ec996600c245112a008c3f333957a6522db59`.
It was not used as the q43a parent.

Prior preserved n600 scorer aggregate for the correct receipt-bytes archive:

```text
archive_sha256 b35e756829306a85ec2ad51634bde74523d89df9046c682253176b393bd59c06
archive_bytes  357837
d_seg          0.004305419922
d_pose         0.000716508925
S              0.7534578126155775
axis           [macOS-CPU frozen-scorer advisory]
batch_count    38
```

This arm did not rerun the scorer. The preserved aggregate confirms the expected
neighborhood, but the adapter validation failed before the charter's solve path
could perform its own fresh baseline replay.

## Typed Blocker

The su2 adapter requires this exact v4d member order in
`experiments/ddm_su2_qa43_tail_solver.py`:

```text
manifest.json
state/tokens.dr7t
state/renderer.sec
state/selector.sec
state/pose_stub.sec
state/pose_warp.stp
```

The tq1c archive contains:

```text
0.bin
```

Refusing assertion:

```text
experiments/ddm_su2_qa43_tail_solver.py:389-390
if self._parent_names != self._BASE_MEMBER_NAMES:
    raise QA43Error("v4d-warp parent member order/shape differs")
```

The exact command was:

```bash
PYTHONPATH="$PWD:$PWD/src:$PWD/experiments" \
/Volumes/VertigoDataTier/pact/uv-envs/pact-main/bin/python \
experiments/ddm_su2_qa43_tail_solver.py validate \
  --program-kind warp-tail \
  --receiver-adapter __main__:create_v4d_warp_adapter \
  --adapter-arg parent_archive=/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/phase_b_realized_tq1c/candidate_archives/move_0023_snap_r00_c12_L13.zip.receipt-bytes \
  --adapter-arg receiver_deps_dir=/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/eval_root/submissions/pfs1
```

Result:

```text
rc=2
QA43 REFUSED: v4d-warp parent member order/shape differs
```

No force-fit adapter was attempted.

## Solver Interface Recovered

`experiments/ddm_su2_qa43_tail_solver.py` exposes:

```text
validate|solve
--program-kind {warp-tail,terminal-frame0}
--receiver-adapter MODULE:FACTORY
--adapter-arg KEY=VALUE
--top-k 56,112,200
--relinearizations {2,3}
--damping FLOAT
--coefficient-limit INT
--resume-from ABSOLUTE_PATH
--max-seconds FLOAT
--min-free-bytes INT
```

The su2 memo's ready command points to the same `warp-tail` adapter and explicitly
excludes the distinct free-frame0 counterfactual. That exclusion remains binding.

## RECALL EVIDENCE

Sources searched and inspected:

- Charter and contract: `.omx/tmp/codex_runs/q43a_prompt.md`, `.omx/tmp/codex_runs/_common_contract.md`.
- Governing files: `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`.
- Memory registry: `/Users/adpena/.codex/memories/MEMORY.md` queries for `q43a`, `QA43`, `#775`, `tq1c`, `move_0023_snap_r00_c12_L13`, `main_hot_state`, `canonical_frontier_pointer`.
- su2/QA43 receipts and source: `.omx/research/ddm_su2_pose_endgame_program_20260730.md`, `.omx/research/ddm_qa43_two_plane_parallax_20260729.md`, `experiments/ddm_su2_qa43_tail_solver.py`, `experiments/inflate_runner_v4d_qa43_tail.py`, `experiments/test_ddm_su2_qa43_tail_solver.py`.
- Ledger and DAG recall: `.omx/research/ddm_deferral_queue_ledger_20260729.md`, `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`, `.omx/state/canonical_task_status.jsonl`, `.omx/research/harness_tasklist_bridge_20260803.jsonl`.
- Canonical equations registry: `.venv/bin/python tools/list_canonical_equations.py --json`, narrowed to score/rate/byte/pose/stopping/archive equations.
- tq1c custody: `/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/phase_b_realized_tq1c/tq1_phase_b_realized_measurements.jsonl` and `stage_checkpoints/n600_scorer/move_0023_snap_r00_c12_L13/aggregate.json`.

Findings beyond the charter seeds:

- `main_hot_state.md` already declares the tq1c receipt-bytes archive as the new base and warns to re-base anything citing `75df9cc3` or `d5e814d5`.
- The same directory contains a misleading `tq1c_base/archive.zip` with the prior `75df9cc3` hash; using it would have violated the q43a freshness gate.
- The deferral ledger records the prior ps1 cross-parent precondition failure: QA43 tail refinement was moot on a post-hoc-geometry-walled seg-native parent, with reopen only for a pose-conditioned base.
- The DAG su2 feed confirms the implementation is same-pool `warp-tail`, never free-frame0.
- The canonical task status note records that the 600 B/pair trigger had not been evaluated on real data as of that audit; q43a still has not evaluated it because the adapter did not admit the current parent.

Change to plan:

- Stop at scorer-free adapter validation. Do not run PoseNet baseline/profile/solve because the exact receiver grammar cannot consume the current parent. Do not adapt by treating the single-member tq1c archive as equivalent to the v4d multi-member archive.

## Numbers Not Measured

- Tail concentration top-56/top-112/top-200: not measured by this arm.
- k=56 solve row: not measured.
- `Delta d_pose`, `Delta d_seg`, `Delta bytes`, joint `Delta S`: not measured.
- B/pair and >600 B/pair falsifier: not measured.
- Candidate archive: none produced.

## Follow-On Disposition

Follow-on is `QUEUED-WITH-A-FIRE-ORDER`, not merely noted:

1. Build or select a QA43 adapter that consumes the tq1c single-member `0.bin`
   grammar through the same canonical decode chain as `ddm_tq1_qo1_inflate_runner_*`.
2. Re-run scorer-free validation on pairs 0 and 599 with both coders and public
   receiver parseback.
3. Only after validation passes, run the fresh n600 parent Pose replay, write the
   hard-tail concentration profile, then start k=56 with `--resume-from` on the
   SSD tier and `--relinearizations 3`.
4. If a tq1c adapter cannot expose pair-local pose correction without fake
   equivalence, fold q43a into pose-in-training / wp1 rather than forcing the
   v4d receiver onto the current archive.

## Boundary

No paid dispatch, no contest-CPU/CUDA claim, no upstream edits, no protected-file
edits, no `/tmp` persisted evidence, no large artifacts created. Exact pointer
unmoved.

## MAIN APPENDIX — $0 BLOCK-LEVEL TAIL ADJUDICATION (2026-08-06, MAIN)

The arm's typed blocker (adapter grammar mismatch) is ACCEPTED as filed. Before authorizing an
adapter build, MAIN computed the tail-concentration half from the BANKED tq1c scorer checkpoints
(38 x 16-pair batch files, pose_squared_error_sum; parent b35e756829, same custody chain as the
preserved aggregate):

- total pose sq-sum 2.5794 over 38 blocks; top block share 0.054 (2.1x uniform 0.026).
- top-4 blocks (64 pairs) = 21.1% · top-7 (112 pairs) = 35.1% · top-13 (208 pairs) = 56.9%.
- VERDICT: pose debt on THIS parent is NOT block-tail-concentrated. The QA43 charter's
  counterfactual (pose 1.263 -> 0.382 via top-112) was a STALE-PREMISE import from the v4d era
  (m37 genus — MAIN charter defect, filed as a round-10 finding): the pfs1/ms8/pw1 arc that
  collapsed pose 2.7 -> 0.000717 also flattened the tail. Corroboration: #915 (pu2 line) banked
  the tail-specific FORWARD value at ~0.6% of gap.
- Scope honesty: block-level flatness bounds block-selection, NOT per-pair selection (extreme
  within-block skew could evade it). But best-case top-112 arithmetic (<=0.024 S at PERFECT
  correction, ~0.009 S byte cost at 120 B/pair) is dominated by named seg-axis spends
  (seg gap 0.4015 = 69% of total; en1 margin-weight A/B built+waiting per #925).

ROUTING: tq1c-grammar QA43 adapter build = NOT AUTHORIZED for tail-solve alone (would fire only
if a future GLOBAL pose re-solve on this parent needs the same receiver binding — su2 step-2
family, where #850's uncapped-GN headroom lives). The q43a slot routes to the round-10 review's
pick from the scorer-gated queue, seg-axis-first per m66/m67. Verdict scope: INSTANCE
(parent b35e756829 x block-level evidence); per-pair profile remains the named residual
measurement if the pose axis is ever re-prioritized.

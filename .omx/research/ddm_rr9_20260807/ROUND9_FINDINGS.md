# ddm_rr9 Round 9 Findings

Verdict: NOT-CLEAN. Counter remains 0/3.

Axis: `[apparatus / scorer-free]`. Score claim: false. Pointer moved: false.

## Findings

### RR9-F1 - HIGH - ARM-CAP fire outran its own required Metal mem-probe receipt

The mx1c structural cure exists, but the live ARM-CAP fire did not satisfy the ticket's own
pre-fire protocol in the evidence visible to this checkout. The ticket and `NEXT_IF_RESUMED`
require a passed Metal mem-probe receipt before training fire; the live launch manifest shows
`mlx-train` was started through `safe_run.py`, while the required receipt file is absent.

Evidence:

- `.omx/research/ddm_mx1c_20260807/NEXT_IF_RESUMED.md:15-17` requires
  `.omx/research/ddm_mx1c_20260807/row1_v2_two_arm/mem_probe_receipt.json` with
  `status=passed` and MLX/load-stage samples before training fire.
- `.omx/research/ddm_mx1c_20260807/launch_ticket_v2_two_arm_governed.json:205-242`
  declares the GPU `mem_probe_command`, `mem_probe_receipt_path`, and
  `mem_probe_receipt_required: true`.
- `/Volumes/VertigoDataTier/pact/ddm_mx1_20260806/launch_arm_cap_governed/launch_manifest.json:2-47`
  shows the live command is not the mem-probe; it is `tools/safe_run.py ... -- .venv/bin/python
  experiments/ddm_mx1_pr130_semantic_renderer.py --mode mlx-train --device gpu ...`.
- Filesystem audit on 2026-08-07: `test -f` returned rc=1 (absent) for the required
  `.omx/research/ddm_mx1c_20260807/row1_v2_two_arm/mem_probe_receipt.json`, for the
  manifest-declared done receipt, and for
  `/Volumes/VertigoDataTier/pact/ddm_mx1_20260806/launch_arm_cap_governed/run/result.json`.
- `/Volumes/VertigoDataTier/pact/ddm_mx1_20260806/launch_arm_cap_governed/run.log:1-4`
  contains the admission assertion pass and MLX grouped-backward banner, but no mem-probe
  receipt, stage checkpoint, result, or completion line.
- `.omx/state/durable_daemons.json:14084-14090` records the `safe_run` child as `running`
  with `projected_peak_gib: 66.268951` and an empty `log` field; it is an admission row, not
  a passed Metal load-stage receipt.
- `.omx/state/main_hot_state.md:24-27` records the live fire and high transient pressure
  under the projection. The charter cites free memory pinned near 0-1 GiB during load; without
  the required load-stage receipt, that pressure is not proven to be within a measured passed
  peak.

Re-derived projection arithmetic: `1.268951 GiB + 65.000000 GiB = 66.268951 GiB`. The
arithmetic holds. The problem is scope: it is a CPU-side blocked-probe peak plus an unknown
Metal margin, not the ticket-required passed Metal load-stage receipt. The wrapper also allows
`--rss-mb 90000` (87.890625 GiB), so downstream scheduling must not treat the 66.268951 GiB
projection as a measured live cap.

Smallest cure:

1. Add a launch-protocol guard for this ticket class before any `argv_n32_arm_*` fire: if
   `mem_probe_receipt_required` is true, refuse unless the receipt path exists, parses, has
   `status=passed`, and carries the required MLX/load-stage samples. Persist the guard result
   beside the ticket before invoking `safe_run.py`.
2. Quarantine the current ARM-CAP fire for scheduling purposes until either a passed Metal
   mem-probe receipt is attached or the current run itself produces a post-hoc measured peak
   receipt with enough load-stage telemetry. Do not fire ARM-VEH or n120 from this projection
   alone.

Follow-on disposition: QUEUED-WITH-A-FIRE-ORDER as `rr9_f1_mx1c_mem_probe_fire_protocol_guard`.

## Clean Audits

### mx1c assert placement

No finding. `experiments/ddm_mx1_pr130_semantic_renderer.py:1191-1223` parses modes and calls
`assert_governed_admission()` only for `mlx-train` and `torch-smoke`, leaving `probe`,
`mlx-parity`, and `mem-probe` ungated as the receipt claims. The pose wrapper imports the guard
and calls it at `src/tac/pr130_lift/pose/train_pose_carrier_full_resumable.py:30` and
`src/tac/pr130_lift/pose/train_pose_carrier_full_resumable.py:619-623`. `tools/safe_run.py:316-333`
runs system admission before stamping `TAC_GOVERNED_ADMISSION=1`.

### fw1 rc propagation

No finding. `/Volumes/VertigoDataTier/pact/ddm_et4_20260806/et4_n600_driver.sh:22-35` waits all
three shards, exits `7` on shard failure, captures final-stage rc in `final_rc`, writes it, and
exits `"$final_rc"`. `/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/hpac_our_labels_driver.sh:16-21`
records nonzero stage rc into `overall_rc`, skips dependent stages on failure at lines 51, 65,
and 83, and exits `"$overall_rc"` at lines 86-87. The repair script already exits captured rc at
`/Volumes/VertigoDataTier/pact/ddm_et4_20260806/et4_repair_rows_and_final.sh:60-66`.

The guard is warn-only and wired before the heavier preflight path:
`tools/preflight_hook.py:1143-1223` defines the rc-fallthrough scanner, and
`tools/preflight_hook.py:1359-1363` calls it before `run_preflight()`.

### 145 admission backlog sample

No finding. Re-running `check_heavy_witness_trainers_call_admission_guard(strict=False,
verbose=False)` returned 145 warnings. The first five sampled warnings are real heavy trainer
entrypoints with train/device surfaces and no `assert_governed_admission()` occurrence:

| sample | evidence |
|---:|---|
| 1 | `experiments/train_anr_token_renderer.py:187-229` has `main()`, argparse, and CUDA/CPU device handling; `rg assert_governed_admission` rc=1 |
| 2 | `experiments/train_balle_hyperprior.py:199-272` has `main()`, required `--device`, and MPS refusal; `rg assert_governed_admission` rc=1 |
| 3 | `experiments/train_blocknerv_as_renderer.py:60-144` has argparse, `--device`, and CUDA resolution; `rg assert_governed_admission` rc=1 |
| 4 | `experiments/train_categorical_renderer.py:130-179` has `main()`, argparse, and CUDA/MPS/CPU handling; `rg assert_governed_admission` rc=1 |
| 5 | `experiments/train_charm_50k_toy_substrate.py:1175-1234` has `main()`, argparse, and CUDA/CPU device choice; `rg assert_governed_admission` rc=1 |

### et5 fold

No finding. Direct JSON re-derivation from `.omx/research/ddm_et5_20260807/pricing_receipt.json`
matches the route memo:

- best row: `base_flip_r0` + `split_lzma1`
- projected n600 bytes: 7,100,737.5
- projected full-patch flips: 84,056.25
- `B/full-flip = 7100737.5 / 84056.25 = 84.47602052197189`
- `W = 1.27310821533`, so `84.47602052197189 / W = 66.35415552642162`
- waterfill selected count: 0, selected pairs: `[]`

The script computes these fields at `.omx/research/ddm_et5_20260807/run_et5_pricing.sh:488-552`.
The verdict is scoped correctly at `.omx/research/ddm_et5_20260807/run_et5_pricing.sh:802-806`
as `INSTANCE: ET4 correction field on tq1c parent, stratified n32 scorer-free description
pricing`, and `.omx/research/ddm_et5_20260807/CAMPAIGN_984_ROUTE.md:1-17` routes it as a
negative description-side carriage price, not a solve-family kill.

### cons1 non-improvement

No finding, but do not launder it into consolidation progress. The measurement is honest:
`.omx/research/ddm_cons1_20260807/RECEIPT.md:12-18` records `CONSOLIDATE-NOW` before and after
with `113 memos / 0 canonical-equations-or-DSL commits`, and `.omx/state/main_hot_state.md:35-36`
labels it "honest non-improvement." A fresh rerun during this review produced `115 memos / 0
canonical-equations-or-DSL commits`, so the monitor remains red.

The routed-sample claim is bounded: `.omx/research/ddm_cons1_20260807/RECEIPT.md:81-105`
documents 12 evenly spaced sampled memo paths and 12/12 routed classifications. Spot checks found
real route handles in the sampled artifacts, for example
`.omx/research/ddm_eh1_20260806/NEXT_IF_RESUMED.md:3-56`,
`.omx/research/ddm_et4_20260806/NEXT_IF_RESUMED.md:12-38`, and
`.omx/research/ddm_lw1_20260806/RECEIPT.md:199-213`. This supports only "sampled memos had route
handles"; it does not prove the 24h signal corpus is consolidated.

Assumption challenge: if every recent memo already has a valid `NEXT_IF_RESUMED` or downstream
consumer, then the current `signal_ratio` metric is measuring canonical-equation/DSL codification
lag rather than orphaned routing. The next consolidation pass should either narrow the denominator
to unrouted memos or land actual canonical-equation/DSL/task-ledger codification that moves the
metric.

## Recall Evidence

| source searched | query / read | what changed |
|---|---|---|
| Governing files | Read rr9 charter, common contract, `PROGRAM.md`, `CLAUDE.md`/`AGENTS.md` identity, operating manual, and `.omx/state/main_hot_state.md` | Kept this review scorer-free, pointer-honest, and focused on cured surfaces plus live fire. |
| Prior review/cure receipts | Read rr1 through rr8 summaries/findings, mx1c, fw1, et5, and cons1 receipts | The mx1c receipt claimed a required Metal mem-probe before fire; live evidence made that the highest-risk surface. |
| Live launch state | Read `/Volumes/VertigoDataTier/pact/ddm_mx1_20260806/launch_arm_cap_governed/` manifest/log/pid and `.omx/state/durable_daemons.json` | Found real governed admission but no required mem-probe receipt, done receipt, or result artifact. |
| Source/code surfaces | Searched mx1 trainer, pose wrapper, admission guard, safe_run, preflight hook, shell drivers, et5 pricing script, and consolidation monitor | Confirmed mx1c assert placement, fw1 rc propagation, et5 math, and cons1 monitor semantics. |
| Canonical equations registry | `.venv/bin/python tools/list_canonical_equations.py --json` then keyword counts over 424 entries (`admission`, `govern`, `safe_run`, `memory`, `waterfill`, `consolidation`, `signal`, `PR130`) | Found adjacent governing concepts but no current rr9/mx1c/fw1/et5/cons1 equation that supersedes the live artifacts. |
| Research graph / state | Targeted `rg` over `.omx/research`, `.omx/state`, codex run charters, docs, src, tools, and experiments for rr9/cure IDs and key numbers | Confirmed hot-state row for live ARM-CAP, et5 fold route, and cons1 non-improvement. |
| Memory registry | `rg` over `/Users/adpena/.codex/memories/MEMORY.md` for Pact custody/frontier/common-contract terms | Reinforced score-custody, lane, artifact, and Git-write-block boundaries; no score/pointer claim added. |

## Boundaries

- No scorer, `upstream/evaluate.py`, archive build, GPU dispatch, remote dispatch, or queue mark was run by this review.
- No protected common-contract file was edited.
- No Python file was edited, so review-tracker passes were not applicable.
- No staged index state was intentionally touched outside the serializer workflow.
- Own-vehicle frontier remains `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; contest pointer remains borrowed and unmoved.

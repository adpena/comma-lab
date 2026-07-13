# DAG FEED — next-launch all-compatible-levers ticket — 2026-07-13

`research_only=true` · `launch_attempted=false` · `operator_go_required=true` ·
`pointer_delta=0` · `verdict_scope=launch-ticket instance`

## Nodes

| Node | State | Durable surface |
|---|---|---|
| N0 sealed lineage | SETTLED | V9·CGauge #432 → v7.5.2 → ideal-mod19 typed compiler |
| N1 typed composition | BUILT | `src/tac/witness_dsl/spec_next_launch_all_levers_20260713.py` |
| N2 compatibility filter | BUILT | 28 IN / 7 EXCLUDED in `include_exclude_table.json` |
| N3 compiled ticket | BUILT/HELD | `experiments/results/next_launch_all_levers_ticket_20260713/` |
| N4 D-A observer | BLOCKED | exact `witness_component_wallclock.v1` producer absent |
| N5 D-B observer | BLOCKED | exact `sps_gradient_role_conflict_engagement.v1` engagement callback absent |
| N6 causal manifest | READY-AS-OBSERVABILITY | `pact.causal_manifest.v1` default-on shared-trainer writer plus sibling DAG FEED; no launch flag by design |
| N7 memory waterfill | REFUSE | B=2 target-n600 RSS/full-step evidence absent; canonical selection B=1 |
| N8 storage waterfall | REFUSE | selected Vertigo workload root absent; local disk disabled |
| N9 readiness/freshness | PROCEED | explicit Horizon/StepNative deferrals and fire-now composition recognized |
| N10 system governor | REFUSE | concurrent host composition exceeds adaptive memory ceiling |
| N11 operator GO | CLOSED | N6 is closed; opens only after N4/N5/N7/N8/N10 close and ticket recompiles blocker-free |
| N12 governed launch | NOT FIRED | only N11 may enable spawn |

## Edges

```text
N0 -> N1 -> N2 -> N3
N3 -> N4 -> D-A measured wall row
N3 -> N5 -> D-B screw(ep450/event) + phase(ep726) conflict rows
N3 -> N6 -> causal_manifest.jsonl transition rows
N3 -> N7 -> memory admission
N3 -> N8 -> durable SSD run root
N3 -> N9 -> config-freshness admission
N3 -> N10 -> system admission
{N4,N5,N6,N7,N8,N9,N10} -> N11 -> N12
```

No edge bypasses N11. Governor refusal and waterfill B=1 are information, not override targets.

## Equations leg

D-A closes only with the measured identity

```text
t_epoch = t_teacher_fwd + t_teacher_bwd + t_witness_fwd + t_witness_bwd
        + t_R + t_verdict_critical + t_checkpoint_IO + t_other_measured
```

The 95/5 premise is not substituted for any term. The current 295.352 s/epoch receipt leaves the whole
critical path unallocated.

D-B reuses the already-defined SPS role-conflict law:

```text
cos(g_i,g_j) = <g_i,g_j> / (||g_i|| ||g_j||)
```

with norms and the probe's conflict predicate at the actual screw and phase engagement transitions. The
ticket does not mint a new conflict rule.

## DSL and consumer legs

- DSL: `compile_v9_cgauge_ideal_mod19_launch_config` plus six typed additions; no hand argv.
- Consumer: `tools/launch_witness_run.py --config next_launch_all_levers_20260713` recognizes the typed
  object, audits expected levers/real flags, runs $0 gates, and returns rc=11 before spawn while typed
  blockers remain.
- Resume: stage checkpoints + periodic atomic complete checkpoints, all preserved; runtime root on SSD.

## Negative verdict scopes

- MicroBatch bit-identity NO-GO is formulation/fingerprint scoped; B=2 training-only functional parity
  remains open but not admitted without n600 memory/full-step custody.
- Whole-step megakernel NO-GO is whole-step FP-reordering scoped; fingerprint-certified HOSC compile is IN.
- Hardness exclusion is current-wiring scoped; a certified repair may re-enter a future compile.
- Horizon/StepNative exclusions are this-run causal-isolation decisions, not family deaths.
- Current system/storage refusals are host/preflight-instance scoped.

## Pointer and custody

No run, score, archive, paid dispatch, or pointer mutation occurred. The artifact is an uncommitted held
launch ticket and research-only DAG feed.

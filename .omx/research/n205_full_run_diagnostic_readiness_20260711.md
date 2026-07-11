# #243 / #205 full-run diagnostic readiness — pre-registered milestone harvest

**Date:** 2026-07-11 · **Run:** `experiments/results/levelset_v752_baseline_20260710T185913Z/`
· **Authority:** `[macOS-CPU/numpy advisory] NON-PROMOTABLE` · **Pointer:**
`0.19108282 [contest-CPU]` **UNMOVED**. This is instrumentation/means only.

**STORES CONSULTED:** `CLAUDE.md` · `AGENTS.md` · `PROGRAM.md` ·
`docs/operating_manual_craft_handoff.md` · v7.5 SPEC §8 · live `daemon.log`,
`costate_shadow.jsonl`, `levelset_best.json`, `launch.sh` · canonical costate equation ·
`tools/costate_digest.py` · `n205_full_run_risk_register_watchlist_20260702.md` ·
`n205_live_telemetry_harvest_for_v9cgauge_20260711.md` · existing
`tools/witness_per_stage_annulus_attribution.py` + its two memos ·
`costate_organ_trajectory_ledger.md` · D18/mod-dim telemetry and byte-close tools.

## Landing

- `tools/n205_full_run_diagnostics.py`: one read-only, CPU/numpy telemetry harvest command.
  It pre-registers the ep450 and ep726 gates, emits per-boundary attribution, performs
  byte-close **readiness only**, and exports an appendable trajectory JSONL. It refuses any
  output path beneath the run dir and never imports MLX/scorers or invokes rendering/eval.
- `tests/test_n205_full_run_diagnostics.py`: 8 focused tests covering reversal PASS/FAIL/PENDING,
  verdict scope, pose ready/degenerate/post-switch guards, stage deltas, and sacred-run writes.
- `n205_full_run_diagnostic_snapshot_20260711.json`: real execution receipt at 14:41Z.
- `n205_full_run_trajectory_20260711.jsonl`: 579 real telemetry rows plus a provenance manifest,
  copied from the live log without changing it; this is the costate-organ accrual input.

## EP450 falsifiable reversal test (registered before the data)

Exact windows: pre `{375,400,425}`, post `{475,500,525}`. The post window gives the ep450
chroma/screw repairs at least 25 epochs of exposure and includes the ep500 lane band. For both
Road (class 0) and Undrivable (class 2), both emitted metrics must pass:

1. post-window mean `<= 0.95 * pre-window mean` (at least 5% relative reduction); and
2. post-window OLS slope `<= -1e-5` fraction/epoch.

The metrics are per-class `within_flip` (`d_seg_by_class`) and threshold-annulus
`per_class_annulus_flip_frac`. Per-class total flip-mass share is captured as diagnostic context
but is not a gate because its denominator couples all classes. All four class/metric gates are
required for `PASS_REVERSAL`. Once all six exact epochs exist, any missed gate emits
`FAIL_IMPLEMENTATION_FALSIFIED`, scoped to **INSTANCE/FORMULATION**: this run's v7.5.2 deferred-
repair curriculum implementation, never the level-set family or paradigm. Current verdict:
`PRE_REGISTERED_PENDING` (all six milestone epochs are future).

## EP726 pose-finish criteria

Use the last eight `jacobian_basin` rows strictly before ep726. `READY` requires median
sigma_min `>=0.100`, median p10 sigma_min `>=0.025`, median condition number `<=125000`, and
at least 6/8 sigma_min values `>=0.080`. Degenerate/not-ready fires on any non-finite/non-positive
value, median sigma_min `<0.080`, median p10 `<0.015`, median condition number `>150000`, or
last-four median `<75%` of prior-four median. Post-switch success at the first verdict `>=ep825`
requires d_pose reduction `>=10%` while d_seg rises no more than `0.001` absolute. All remain
pre-registered-pending; the present ep214–242 window is not promoted to an ep726 readiness verdict.

## Real run receipt and continuation

At snapshot time training telemetry reached ep243; the newest completed n600 CPU-torch verdict was
ep225 (`d_seg=0.037041`) and the best remained `0.029269@ep150`. Road within-flip rose
`0.092372@150 -> 0.105403@200 -> 0.118363@225`; Undrivable rose
`0.010932 -> 0.012244 -> 0.016014`. This confirms `BINDING_TERM_STALL /
TRAIN_VERDICT_DECOUPLING`; curriculum remains `unify_tau`, tau octave 0 (`tau=1.0`).

Byte-close readiness is GREEN without building an archive: BEST EMA ep150 and latest EMA ep225
are finite, schema-complete n600/mod32 checkpoints; both canonical tools parse; D18 at ep225 is
`k90=26`, code estimate `32009 -> 26007` bytes (6002 estimated bytes saved). The prompt's
`32541 -> 26440` receipt is correctly preserved as the ep200 row; D18 remains an estimate until
a real milestone byte-close A/B.

Re-run this same command at ep525, ep825, and terminal ep3000, writing a new dated snapshot/JSONL
outside the run dir. The canonical DAG FEED append is intentionally not made in this landing:
the DAG was already dirty and named in live sister ownership during preflight; absorbing those
hunks would violate Catalog #314/#405. This memo plus the costate-ledger FEED below preserves the
triality signal without touching the contested hot file.

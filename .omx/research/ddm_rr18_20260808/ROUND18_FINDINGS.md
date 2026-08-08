# ddm_rr18 ROUND18_FINDINGS

## Verdict

NOT-CLEAN. rr18 resets the clean-pass counter to 0/3.

The endpoint facets pass is numerically present and internally useful, but it
repeats the rr16 provenance class: the committed endpoint facet receipt rows do
not carry the cache identity, cache SHA-256 values, or replay argv needed for a
consumer to reproduce the measurement without hidden context. This is exactly
the gap the rr18 charter called out.

No scorer slot, n600 job, archive build, upstream eval, Metal launch, or live
ARM-VEH run-dir mutation was performed by this review.

## F1 - CRITICAL - Endpoint Facet Rows Lack Cache-Bound Provenance

rr16 F1 required load-bearing facet receipts to carry input/target cache
identity, cache hashes, and argv. The new endpoint facets receipts do not meet
that bar.

Evidence inspected:

- `.omx/research/ddm_mx1g_20260807/endpoint_facets/mx1t_facets_result.json`
  reports `status=passed`, 24 checkpoint rows, 3 tail rows, and a step-1500
  anchor `abs_diff=0.0`.
- Its aggregate `cache_load` block records both input and target as
  `/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/gt_seg_cache.pt`,
  byte count `943720076`, selected shape `[32, 384, 512]`, and
  `shared_with_input_cache=true`.
- That aggregate block does not include cache SHA-256 values or the exact replay
  argv.
- The JSONL facet rows in
  `.omx/research/ddm_mx1g_20260807/endpoint_facets/mx1t_facets_receipts.jsonl`
  carry checkpoint hashes, source repo head, axis, scorer shapes, pair IDs, and
  facet data, but do not carry `cache_load`, `input_cache`, `target_cache`,
  cache SHA-256 values, or replay argv.

This does not refute the endpoint measurement. It blocks treating the endpoint
facet corpus as cleanly self-contained for downstream policy consumption. The
repair is the same class as rr16's demanded repair: either patch the facet
writer and rerun, or commit a cache-bound endpoint addendum that includes
input/target paths, SHA-256 for both caches, selected pair IDs, source repo head,
replay argv, axis, and score_claim=false for the endpoint corpus.

Verdict scope: APPARATUS / endpoint-facet provenance. Not a score result.

## Reauthor Guard And ARM-CAP Contamination

The RR11-F1 cure is real in current source. `launch_ticket()` now resolves
`cap_cache` and `veh_cache` with `Path(...).resolve()` and raises before writing
the ticket if they resolve to the same path. For the live SSD cache paths,
`tq1c_seg_cache.pt` and `gt_seg_cache.pt` both exist, are not symlinks, and
resolve to distinct absolute paths; a same GT-vs-GT input would compare equal
and refuse.

The live corrected ticket carries tq1c on all four VEH keys:

| key | input | target |
|---|---|---|
| `argv_n32_arm_veh` | `tq1c_seg_cache.pt` | `gt_seg_cache.pt` |
| `argv_n32_arm_veh_resume` | `tq1c_seg_cache.pt` | `gt_seg_cache.pt` |
| `argv_n120_arm_veh` | `tq1c_seg_cache.pt` | `gt_seg_cache.pt` |
| `argv_n120_arm_veh_resume` | `tq1c_seg_cache.pt` | `gt_seg_cache.pt` |

The earlier reauthor artifacts under `reauthor_resume/` did collapse VEH keys to
GT input, but they were ARM-CAP resume artifacts and predated the corrected
ARM-VEH reauthor. The completed ARM-CAP run was not contaminated by that collapse:
ARM-CAP is intentionally GT-to-GT, and its final resume fire safe_run receipt
records `exit=0`, `elapsed_s=5920.261`, `peak_rss_mib=1505.547`, start
`2026-08-08T02:56:23Z`, generated `2026-08-08T04:35:03Z`. The final checkpoint
and `result.json` have mtime `2026-08-07T23:35:03-0500`, before the corrected
ARM-VEH reauthor at `2026-08-07T23:51:26-0500`.

Adjudication: no ARM-CAP contamination found in the inspected receipts and
mtimes. The guard is a correct cure for existing, symlink-resolved SSD paths.

## WC1 Bench Consumption

The live measured `wc1_bench_receipts.jsonl` rows support adopting fp16-train as
a speed lever for n120-only use, with one reporting caveat.

Measured rows, all `status=passed`, axis
`[macOS-MLX research-signal bench harness]`:

| variant | seconds/step | d_seg sanity | lever |
|---|---:|---:|---|
| baseline | 10.530578995 | 0.001042207082 | none |
| threads | 10.306263399 | 0.001042683919 | `--perf-thread-pin one` |
| batched | 11.385524368 | 0.001042683958 | `--microbatch-pairs 32` |
| compile | 10.231335163 | 0.001042366028 | `--compile-train-loss` |
| fp16-train | 8.442011404 | 0.001041730245 | `--train-compute-dtype fp16` |

Derived:

- fp16-train speedup vs baseline: `1.2474016547424494x`.
- full cross-variant d_seg sanity spread: `9.53713121513598e-07`.
- the narrower baseline-vs-threads delta is the claimed `~4.8e-7`; using that
  as the whole cross-variant spread would be underreported by about 2x.

Adjudication: the sanity spread is still tiny for a five-step wall-clock bench,
but the precise statement should use the full `9.54e-7` spread if it is meant to
cover all variants. The live ARM-VEH fire argv is lever-free: no
`--train-compute-dtype`, no `--perf-thread-pin`, no `--compile-train-loss`, and
no `--microbatch-pairs` flag are present.

## ARM-VEH Fire Custody

The final ARM-VEH fire argv substitutes numeric safe_run values
`--projected-gib 21` and `--rss-mb 45000` for the sentinel values. This is
legitimate for the inspected fire because the numbers are derived from the
ARM-VEH mem-probe receipt, not transferred from ARM-CAP.

ARM-VEH mem-probe receipt:

- path:
  `.omx/research/ddm_mx1e_20260807/regen2/launch_arm_veh/n32_metal/mem_probe/mem_probe_receipt.json`
- `status=passed`, `metal_fire_clearance=true`
- required stage: `after_train_step_000003`
- `has_required_stage_sample=true`
- peak candidates include `peak_mlx_reported_gib=13.377822` and
  `peak_rss_gib=1.513748`
- policy: `ceil(max_peak * 1.5)` with floor 15 GiB gives `21`
- RSS floor gives `45000`

The fire guard verdict for `argv_n32_arm_veh` passed and its config match checks
bind `input_cache=tq1c_seg_cache.pt`, `target_cache=gt_seg_cache.pt`,
`train_compute_dtype=fp32`, `perf_thread_pin=off`, and `compile_train_loss=false`.

Adjudication: not a cross-arm constant transfer. The correct derivation is the
same-arm ARM-VEH mem-probe receipt and the already recorded safe-run projection
formula.

## Knee And n120 Step Count

The endpoint facets verify the knee claim on ARM-CAP n32:

| step | aggregate d_seg |
|---:|---:|
| 4250 | 0.0010835329691569011 |
| 4500 | 0.0010838508605957031 |
| 4750 | 0.0010859171549479167 |
| 5000 | 0.0010862350463867188 |
| 5250 | 0.0010860761006673176 |
| 5500 | 0.0010881423950195312 |
| 5750 | 0.0010890960693359375 |
| 6000 | 0.0010890960693359375 |

Derived deltas:

- `4250 -> 6000`: `+5.563100179036386e-06` d_seg, regressive.
- `4500 -> 6000`: `+5.245208740234375e-06` d_seg, regressive.
- `4250 -> 4500`: `+3.1789143880201105e-07` d_seg, essentially flat.
- K=8 tail average beats final by `-2.86102294921875e-06`, but remains worse
  than the best 4250/4500 region.

Adjudication: the `~4500` step-count recommendation is licensed only as an
ARM-CAP-derived prior or a provisional cap. It is not a clean global n120 fire
order for both arms. If ARM-VEH wins or is still material to the n120 choice, it
owes its own ARM-VEH curve before n120 consumes the step-count recommendation.

Verdict scope: FORMULATION / one-arm n32 ARM-CAP curve transfer. Not an n120 or
n600 measurement.

## Assumption Challenge

Shared assumption challenged: endpoint receipts can rely on aggregate context,
external addenda, or operator memory instead of making every load-bearing row
self-contained.

Verdict: rejected for rr18. rr17 accepted an addendum as sufficient to release a
specific mx1t selection-policy HOLD, but rr18 explicitly asked whether the new
endpoint facets receipts themselves carry cache-bound provenance. They do not.
The same hidden-context failure mode remains live unless the endpoint corpus is
repaired or accompanied by a cache-bound endpoint addendum.

Secondary assumption challenged: an ARM-CAP knee transfers to ARM-VEH/n120. That
is accepted only as a starting prior. It is not a clean fire order until the
selected arm has its own curve.

## RECALL EVIDENCE

Sources searched/read before adjudication:

- Governing files: `.omx/research/ddm_rr18_20260808/CHARTER.md`,
  `.omx/tmp/codex_runs/_common_contract.md`, `PROGRAM.md`,
  `CLAUDE.md`/`AGENTS.md`, `docs/operating_manual_craft_handoff.md`,
  `.omx/state/main_hot_state.md`.
- Memory registry query:
  `ddm_rr18|rr18|refund|plateau|#899|#904|required-component|margin_targets|codex_runs`.
- Corpus recall query over `.omx/research`, `.omx/state`, docs, reports, src,
  tools, and experiments, bounded to the relevant surface:
  `ARM-CAP|ARM-VEH|mx1g|tq1c|gt|fire_argv_final|fp16-train|wc1|knee|4250|6000|cache_load|reauthor|veh==cap|REQUIRES_FRESH_MEM_PROBE|projected`.
- Canonical equations registry command:
  `.venv/bin/python tools/list_canonical_equations.py --json`; relevant recall
  included `ddm_rr9_mem_probe_fire_protocol_v1`, which says safe_run admission is
  not a substitute for a passed Metal mem-probe receipt.
- Prior recursive reviews: `ROUND16_FINDINGS.md` and `ROUND17_FINDINGS.md`,
  especially rr16 F1's cache-bound receipt requirement and rr17's scoped addendum
  release.
- Current artifacts: endpoint facets result/JSONL, ARM-CAP and ARM-VEH safe_run
  receipts, ARM-VEH mem-probe receipt and fire guard verdict, corrected and
  collapsed reauthor outputs, wc1 bench receipts, and final ARM-VEH fire argv.

Findings beyond the charter seeds that changed the plan:

- `main_hot_state.md` already marked the 4250->6000 knee and explicitly left
  the one-arm license vs ARM-VEH-curve question to rr18, so this review treated
  the n120 step count as pending adjudication rather than a settled launch
  input.
- rr17's addendum release was scoped to the earlier mx1t selection-policy hold;
  it did not waive the rr18 requirement that the endpoint facet receipts carry
  their own cache-bound provenance.
- The live wc1 receipts are measured rows even though WC1_FINDINGS still
  contains stale plan-only prose. Therefore rr18 used the JSONL values directly
  and did not rely on the stale prose.

## Boundary

Measured/verified in rr18:

- JSONL/result field presence for endpoint facets.
- ARM-CAP final safe_run receipt and checkpoint/result mtimes.
- Corrected live VEH cache paths on all four VEH argv keys.
- Current source guard shape for `veh_cache == cap_cache`.
- ARM-VEH mem-probe clearance and derived projection arithmetic.
- wc1 five-row bench speed/sanity values from JSONL.
- ARM-CAP endpoint knee arithmetic from facet receipts.

Did not measure in rr18:

- Any scorer output beyond reading existing receipts.
- Any n600 behavior.
- Any exact archive score or `upstream/evaluate.py` run.
- Any contest-CPU/CUDA row.
- ARM-VEH endpoint facets; the live ARM-VEH run directory was read-only.

Own-vehicle frontier remains `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; contest pointer remains borrowed/unmoved.

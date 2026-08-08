# DDM CF2 Comprehensive Confound Sweep

Created UTC: 2026-08-08T17:00:55Z

Tags: [no-triality] [p0-ledger-ok]

Scope: report-only, CPU/source/byte inspections over the 2026-08-06 to 2026-08-08 measurement stack. No scorer jobs, no Metal jobs, no exact-eval jobs, and no code edits were performed. Every score-like number below keeps its source axis; none is a new score claim.

## Boundary

This sweep read the governing repo contracts, the live hot state, the CF2 charter, the common contract named by the charter, and the relevant source/receipt files. The only new numeric checks were file sizes, SHA-256 hashes, JSON receipt reads, and source-line inspections.

The live own-vehicle pointer is unchanged: `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; borrowed contest pointer remains `0.19108` class and is not our vehicle progress.

## Summary

| id | surface | disposition | result |
|---|---|---|---|
| CF2-001 | MX1 stale grouped-backward / fused-R claim | FOLDED plus guard row | Stale witness-line lever claim is refuted; current hot state already says MX1 consumes no `TAC_MLX` env. |
| CF2-002 | MX1 launch-ticket scalar defaults | QUEUED-WITH-FIRE-ORDER | Ticket has keyed receipt maps but still exposes scalar CAP-bound probe/projection fields that can mislead non-guard consumers. |
| CF2-003 | MX1 facet anchor default | QUEUED-WITH-FIRE-ORDER | Default anchor is ARM-CAP lineage-bound; existing VEH run passed an explicit anchor, but future non-CAP runs should fail closed unless explicit. |
| CF2-004 | WC1 bench RR11-F1 reauthor class | FOLDED clean with warning | Bench is legitimately ARM-CAP wall-clock only; receipts are bench-axis, `score_claim=false`, and `d_seg_batch_sanity` is not a verdict. |
| CF2-005 | HB2 HPAC custody | FOLDED clean | 14,116 B model plus 97,928 B tokens = 112,044 B; hashes and exact decode report match HB2. |
| CF2-006 | MAP1 PR130 component split | QUEUED-WITH-FIRE-ORDER | MAP1 totals are sound, but the `50,914 B` PR130 model/prior split is ambiguous and can be read as a separable HPAC-prior term when source says joint xz model allocation is derived. |
| CF2-007 | VEH/CAP scope honesty | FOLDED clean | The bounded scope says `FORMULATION`, `n32 advisory`, and `NOT a FAMILY verdict`; no upgrade found in searched 08-06 to 08-08 consumers. |
| CF2-008 | Wall-clock attribution | FOLDED clean with warning | The current `7.9s/step` note is labeled unoptimized baseline; measured WC1 receipts were 10.53 baseline, 10.31 threads, 11.39 batched, 10.23 compile, 8.44 fp16. |
| CF2-009 | Prefix/sampling contamination | FOLDED clean | Active MX1/MAP1 consumers use stratified n32/n120 language and do not promote n32/prefix rows to population claims. |

## Round 1

1. MX1 path trace: `experiments/ddm_mx1_pr130_semantic_renderer.py` imports `tac.pr130_lift.mlx_semantic_renderer`, `torch_segnet_to_mlx`, and `apply_contest_faithful_roundtrip_nhwc`; the training loop uses ordinary `mx.value_and_grad` with optional `--compile-train-loss`. The MX1 trainer does not read a `TAC_MLX` env knob. The only grouped/depthwise custom path found was inside the scorer adapter, not a witness-line grouped-backward or fused-R lever.

2. MX1 sibling defaults: the current trainer carries a WC2 microbatch anchor with `selected_default=4`, so it remains numerically aligned with `tools/mx1_fire_guard.py`'s legacy GPU default of 4 today. The guard does not read the anchor object, so a future anchor change could desynchronize fire/receipt config unless the selected default is serialized or the guard consumes the same derivation.

3. MX1 ticket lineage: the v4 ticket exposes both keyed maps and scalar fields. `tools/mx1_fire_guard.py` prefers `mem_probe_receipt_paths[argv_key]`, so the guard is safe. The top-level scalar `mem_probe_receipt_path`, `mem_probe_command`, and `safe_run_projection` remain ARM-CAP-bound convenience fields and are unsafe for generic consumers.

4. MX1 facets: parser defaults use `MX1H_STEP1500_AUTHORITY_D_SEG` as `--facet-anchor-d-seg`. The VEH verdict says run-1 correctly failed when the lineage-bound ARM-CAP anchor fired on VEH, and the accepted VEH replay passed an explicit anchor. That is a good positive control, but the parser default remains a footgun for future non-CAP facets.

5. WC1 bench: the harness default source argv is `argv_n32_arm_cap`, resumes the ARM-CAP checkpoint, stamps axis `[macOS-MLX research-signal bench harness]`, and stores `score_claim=false`. The receipts' `d_seg_batch_sanity` values are sanity telemetry only.

6. HB2 custody: on disk, `hpac.bin.xz` is 14,116 B with SHA-256 `6c44216e8f79bd7d04e998b898d5bf0dc16bae6e3763f8bc19ce4ec8ebdabb40`; `tokens.bin` is 97,928 B with SHA-256 `fc2e4d30df701877cb81d2aeeefa3079e5a1a4d1c3c24db627d7932ab780559f`. Reports match HB2's stated hashes and exact decode equality.

7. MAP1 arithmetic: PR86/PR130 source confirms `archive.zip=191,052 B`, xz model blob `73,968 B`, HPAC token stream `116,980 B`, raw components renderer `40,252 B`, CPR1 `23,054 B`, HPAC model `20,179 B`, exact arithmetic close, PR130 `S=0.172141 [contest-CUDA]`, and no PR130 contest-CPU row. MAP1's total row is consistent, but line-level split wording can mislead because `~50,914 B` is the derived PR86-to-PR130 model delta bucket, not a directly separable HPAC-prior object.

8. Older MAP1 public-PR numbers: PR112/121/125/126/127 values match `pr112_127_intake_20260710.md`; PR128 values match the external-claim caveat in `pr128_intake_reverse_engineering_20260710.md`; PR129/131/132 values match `public_pr129_132_intake_20260725.md`; PR102 public CPU/CUDA comments match the governing public-frontier note. The older PR130 `~114 KB exact partition` lesson is superseded by the corrected `~168 KB` partition attribution in the fullstack intake.

9. Scope honesty and sampling: `VEH_CAP_N32_VERDICT.md` explicitly says `score_claim=false`, `n32 arm-instrument scope`, `verdict_scope: FORMULATION`, `n32 advisory`, and `NOT a FAMILY verdict`. MAP1 and LX1 route n120 reference-form builds with m88/m96 stratified sampling and do not promote the n32 row as population authority.

## Round 2

Second-pass searches focused on sibling consumers and stale phrases:

- `7.9` / `7.85` / `seconds_per_step`: no lever-on or grouped-backward attribution was found in the searched scope. The current hot state labels the figure as unoptimized baseline. WC1 receipts provide the concrete bench-axis timings listed above.
- `FORMULATION` / `NOT a FAMILY` / `convenience`: no 08-06 to 08-08 consumer upgraded the VEH/CAP correction-fork closure to family scope.
- `prefix` / `m88` / `m96` / `stratified`: the active route uses stratified n32/n120 language and retains n600/scorer promotion gates.
- `50,914` / `116,980` / `167,894`: the only live split ambiguity is MAP1's PR130 component wording; downstream LX1 uses HB2's 112,044 B as an HPAC label-stream component and keeps it byte-only.

No second-pass evidence changed the HB2 custody verdict or the WC1 RR11-F1 verdict.

## Self-Audit

Shallowest surface in the first pass was the older MAP1 public-PR table: the first pass centered on PR130/HB2/MX1 because those are load-bearing for #984. The second pass checked the older PR112-132 and PR102 source rows enough to classify the only MAP1 defect as a component-split wording issue, not a total-score or axis issue.

Remaining risk: I did not execute any live process enumerator because this charter is CPU/source-only and the environment denied a process-status probe in this session. I therefore make no claim about live WC1 process state beyond existing hot-state text and receipt files.

## Follow-On Disposition

| item | state | fire order |
|---|---|---|
| MAP1 PR130 split wording patch | QUEUED-WITH-FIRE-ORDER | Edit MAP1/#984 text to distinguish PR130 token stream `116,980 B`, xz model blob `73,968 B`, derived partition leg `~167,894 B`, and HB2 `112,044 B` HPAC model+tokens on our tq1c labels. |
| MX1 v5 ticket schema | QUEUED-WITH-FIRE-ORDER | Remove or rename scalar `mem_probe_receipt_path`, `mem_probe_command`, and `safe_run_projection` as CAP-only legacy fields; require keyed maps for every argv-key consumer. |
| MX1 facet anchor guard | QUEUED-WITH-FIRE-ORDER | Require explicit `--facet-anchor-d-seg` when the checkpoint dir is not the default CAP lineage or when input and target caches differ. |
| MX1 microbatch derivation binding | QUEUED-WITH-FIRE-ORDER | Serialize the selected microbatch derivation into tickets/receipts and make `mx1_fire_guard.py` compare against that same derivation, not an independent hardcoded default. |
| WC1 bench consumer rule | FOLDED | Treat WC1 rows as wall-clock only; do not consume `d_seg_batch_sanity` as a scorer verdict. |
| HB2 custody | FOLDED | Use as byte-only scorer-free custody for HPAC model+tokens; no scorer claim. |
| VEH/CAP scope | FOLDED | Keep `FORMULATION`, `n32 advisory`, and convenience-config caveat with every citation. |
| Prefix/sampling guard | FOLDED | Keep stratified n32/n120 language; no population promotion without the gated row. |

## Frontier Line

Own-vehicle frontier remains unmoved: `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; contest borrowed pointer remains `0.19108` class. CF2 moved no pointer and produced no score row.

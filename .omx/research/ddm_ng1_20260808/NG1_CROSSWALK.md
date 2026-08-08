# ddm_ng1 - modded-nanogpt PR #349 ANVIL crosswalk

Tags: [no-triality] [p0-ledger-ok]
Date: 2026-08-08
Status: COMPLETE research receipt; no scorer, no launch, no paid dispatch, no run mutation.

## Answer First

`ddm_ng1` does not move the frontier. It source-verifies modded-nanogpt PR #349 as an open speedrun PR and converts its lessons into local design gates, not into optimizer constants or score claims.

The highest-ranked transfer is methodological: before the next MAIN fire, the M1 ticket needs a measured same-config sanity envelope for the fp32/fp16 event predicate. The PR's validation-loss sigma is not imported as a Pact number; the imported object is the practice of reporting an uncertainty envelope around unseeded/timing-sensitive claims.

Disposition counts:

| Count | Disposition | Rows |
|---:|---|---|
| 2 | ADOPT | sigma-calibration methodology; timed-region boundary discipline |
| 3 | ADOPT-CLASS / RACE | ANVIL-vs-Muon update-rule race; tail-blend/weight-averaging class; mid-schedule growth plus cadence as event-schedule doctrine |
| 1 | QUEUED-WITH-FIRE-ORDER | speedrun leaderboard/history refresh after PR #349 final disposition |
| 2 | FOLDED / N-A | fp8 direct use on current local MLX; narrow Q/K plus bigram sink |

## Source Verification

Primary external source:

- GitHub PR: https://github.com/KellerJordan/modded-nanogpt/pull/349
- PR branch file view: https://github.com/KellerJordan/modded-nanogpt/pull/349/files
- Raw PR branch `train_gpt.py`: https://raw.githubusercontent.com/devenpzak/modded-nanogpt/anvil-record/records/080626_ANVIL/train_gpt.py

Source-verified facts from PR #349, checked 2026-08-08:

- Status: open PR #349, no reviews shown on the PR page at check time. This is not a merged leaderboard row.
- Claim: source-verified PR title/README report `64.95s` mean wall time on the fastest machine family, with validation CE mean `3.27604`, standard deviation `0.00087`, and `p<1e-10`; all are PR-author reported numbers, not Pact measurements.
- Protocol: source-verified README reports `16` unseeded runs across `4` independent cloud `8xH100-80GB` machines, with Xeon `8480+` / `8468` hosts and driver `580.126`.
- Same-session baseline: source-verified README reports prior #89 at `74.38s` / `75.05s` in the same session; this is PR-author evidence, not an independent Pact benchmark.
- Timing correction: source-verified PR discussion shows reviewer concern that work had been outside the timed region; the author then moved prefix-table build, shard loading / first-batch fetch, and terminal weight averaging into the timed region and retimed.
- ANVIL mechanics: source-verified raw `train_gpt.py` contains the ANVIL cascade, dual rails, rail equalization, sign-aligned decay, fp32 commit sidecar, tail ship blend, FP8 matmul paths, narrow Q/K switches, and bigram sparse-gradient machinery.

Extracted ANVIL coefficient triples from `ANVIL_MAPS`:

```text
(4.447393659248992,  -8.6592834371539,    4.484453950130224)
(3.0003381342927278, -3.180035633097396,  0.959298544318381)
(3.5997737315685896, -5.129961939372116,  1.94624091228252)
(3.238755176943844,  -3.988313284248892,  1.4073635628857994)
(2.558664852492382,  -2.5626431459988033, 0.9815933969681315)
(2.38665846260798,   -2.223181615281231,  0.835162681686761)
```

The coefficient extraction is complete for this crosswalk. No ANVIL replay, update-matrix capture, optimizer A/B, or Pact runtime measurement was run.

Local checks:

- `mlx.core` version observed locally: `0.31.2` [local-checked]. Exposed scalar dtypes include `bfloat16`, `float16`, and `float32`; no general `mx.float8*` dtype attribute was found [local-checked].
- Local package inspection found MLX quantized linear support mentioning `mxfp8` / `nvfp4` modes [local-checked]. That is not a direct trainable FP8 forward/backward path for the current witness trainer.

## Recall Evidence

| Source or query | Finding used | What changed in this crosswalk |
|---|---|---|
| `.omx/state/main_hot_state.md` | Live own-vehicle pointer is `S=0.7534578126155775 @357,837 B [macOS-CPU advisory]`; next MAIN composes an M1 n120 ticket with saturated fp16+hyg+cache and event predicate. | Makes sigma calibration a pre-fire M1 seal item, not a separate launch. |
| `.omx/research/ddm_gc21_20260808/GC21_CONVOCATION.md` | Q3 fp16 guard falls back to fp32 if fp16 is worse by more than `2.0e-6` d_seg or flips event classification; two more reviews owed. | Gives the concrete consumer for same-config event-sigma reporting. |
| `.omx/research/ddm_wc3_20260808/WC3_FINDINGS.md` | Saturated bench round 2 was `7.659 s/step`, `1.361x` vs `10.421 s/step` same-session baseline; fp32 sanity support was two-point and schedule-dependent. | Converts PR timing-region discipline into a bench-receipt requirement and keeps WC3 numbers instrument-scoped. |
| `.omx/research/codex_findings_ddm_px1_soap_muon_beyond_crosswalk_20260724_codex.md` | Prior optimizer crosswalk verdict was ADOPT-MEASUREMENT-CONTRACT and DO-NOT-ADOPT-OPTIMIZER-CONSTANTS; update RMS and polar residual custody are required before optimizer claims. | ANVIL becomes a race protocol, not an imported optimizer. |
| `.omx/research/ddm_dy2_20260805/RECEIPT.md` | `plateau_tail_average` exists locally with a reset-at-anchor growing-average law, but no scorer launch in that receipt. | ANVIL tail blend is convergence evidence for the class, not proof dy2's exact EMA law is superior. |
| `.omx/research/tilde_research_optimizers_survey_and_aurora_build_20260609.md` and `.omx/research/tilde_optimizers_for_inert_loop_20260610T193200Z.md` | External optimizer mechanisms were previously default-off unless local shape/regime checks passed; plain Muon was the relevant inert-loop fix. | Keeps ANVIL below Muon until a local update-custody race exists. |
| `.omx/research/followon_reform_design_drain_report_20260714.md` | The old follow-on pool had built-never-fired optimizer/curriculum arms, including `film_polar_chart_spel`. | Prevents a new optimizer row from becoming another ownerless queued idea. |
| `.omx/research/keller_jordan_muon_modded_nanogpt_research_20260513.md` | Older modded-nanogpt speedrun history was stale relative to PR #349. | Adds a speedrun-history refresh queue item after PR #349 is accepted, rejected, or superseded. |
| `rg` over `.omx/research`, `docs`, `src`, and local MLX package files | Located local sigma/event, tail-average, optimizer-custody, and fp8 support surfaces. | Confirms no current direct fp8 adoption path and no score-bearing local ANVIL row. |

## Ranked Crosswalk

### 1. Sigma calibration methodology -> ADOPT

PR #349 reports an empirical validation CE spread across unseeded speedrun runs. Pact should not reuse that value. The transferable piece is a same-change uncertainty envelope around claims that are both noisy and decision-bearing.

Named local consumer: the M1 n120 ticket seal in the GC21/WC3 line. Add a field such as `sanity_sigma_measured` or equivalent before MAIN fire, using repeated same-config sanity rows or equivalent replay evidence on the exact event predicate.

Falsifier: if repeated same-config sanity produces sigma large enough to dominate the GC21 event threshold, or if fp16/fp32 event classification flips inside that envelope, M1 must fall back or re-derive the guard before launch.

Fire order: QUEUED-WITH-FIRE-ORDER for the next M1 ticket seal. This is design-only in `ddm_ng1`; no scorer job is owned here.

### 2. Timed-region boundary discipline -> ADOPT

PR #349's review thread is directly useful because it corrected the timed boundary after reviewer attack. Anything feeding the training result, terminal ship, runtime cache, first batch, or selection state must be either explicitly inside the clock or explicitly outside with a reason.

Named local consumer: WC-style bench receipts and future profiler receipts. Each bench receipt should carry a one-line timed-region manifest: what is inside the timer, what is warmed before the timer, what is shipped before timer read, and what cache state is allowed.

Falsifier: a future speed/bench receipt cannot reconstruct whether cache fill, data movement, first batch, terminal averaging, or ship materialization was in clock. In that case the speed claim is instrumentation-only until retimed.

Fire order: QUEUED-WITH-FIRE-ORDER for the next bench touch, folded with WC3's append-mode JSONL fix.

### 3. ANVIL vs Muon -> ADOPT-CLASS / RACE

ANVIL is source-verified as a concrete optimizer stack: dual velocity rails, six ANVIL map triples, Gram-trace normalization, per-lane equalization, sign-aligned decay, and fp32 commit sidecar. That is enough to define a local race protocol. It is not enough to import ANVIL into the witness.

Named local consumer: post-M1 optimizer window, or the successor to the existing Muon/FilmPolar/optimizer queue if MAIN does not fire cleanly. Race only after update-matrix custody exists.

Minimum local receipt before any adoption claim:

- same starting checkpoint or same controlled cell;
- matched realized update RMS at the actual parameter-group boundary;
- raw direction RMS, realized update RMS, polar residual or exact-SVD gap, wall time, and CPU-torch facets;
- no borrowed leaderboard reputation as evidence of Pact d_seg movement.

Falsifier: under matched update RMS and same starting cell, ANVIL fails to improve update quality, wall time, or scorer-facing facets, or it destabilizes resume/serializer invariants.

Fire order: QUEUED-WITH-FIRE-ORDER after M1. Do not preempt the current MAIN path for an optimizer port.

### 4. Tail-blend / weight averaging -> ADOPT-CLASS

PR #349 uses terminal weight averaging / ship blending as part of the timed training result. Local dy2 already built a `plateau_tail_average` EMA mode with a different law: anchor reset plus growing average after the anchor.

Transfer: keep tail-weight averaging as a first-class treatment arm and compare exact laws. Do not claim ANVIL's tail blend validates dy2's implementation, because the optimizer, schedule, vehicle, and averaging rule differ.

Named local consumer: dy2 follow-up or a post-M1 terminal-averaging A/B if the current vehicle reaches a comparable plateau.

Falsifier: no CPU-torch facet improvement at equal bytes and equal checkpoint lineage, or a resume/ship mismatch between live weights and EMA shadow.

Fire order: QUEUED-WITH-FIRE-ORDER after the M1 event decision, unless dy2's owner has an earlier governed slot.

### 5. FP8 forward/backward -> FOLDED / TRIGGER-CONDITIONED

PR #349's FP8 path is H100/PyTorch-specific in the verified source. The local MLX check found no general `mx.float8*` dtype in `mlx.core` `0.31.2`, while package internals mention quantized `mxfp8`/`nvfp4` modes. That does not clear the current witness trainer for direct fp8 training.

Named local consumer: precision ladder successor only after a trainable local path exists.

Trigger to reopen: local MLX exposes a trainable fp8/mxfp8 path with deterministic resume and CPU-torch parity, or a Torch/CUDA branch is explicitly opened for this purpose with authority boundaries stated.

Falsifier: no parity, no deterministic resume, or speed gain does not survive whole-step timing. Until then, this is lesson-only.

Fire order: FOLDED for current MAIN. No fp8 action in `ddm_ng1`.

### 6. Mid-schedule growth and accumulation cadence -> ADOPT-CLASS

PR #349 reports a base step count plus `40` grown mid-schedule steps, period-4 embedding cadence from step `336`, and a virtual sequence cap from step `715` [source-verified PR README/code]. The local transfer is not the numbers; it is event-scheduled, resume-safe structural change.

Named local consumer: #686/event-schedule doctrine and M1 `EXTEND_WITH_RESUME` style decisions.

Falsifier: the change cannot be expressed in the typed schedule/resume registry, changes irreversible state without a stage checkpoint, or the event predictor cannot say why growth should happen.

Fire order: QUEUED-WITH-FIRE-ORDER for an extension branch only after M1 trajectory evidence says extend.

### 7. Speedrun leaderboard/history sweep -> QUEUED-WITH-FIRE-ORDER

The older local modded-nanogpt speedrun memo is stale relative to PR #349. However, PR #349 is still open at source-check time, so it should not be treated as a settled official row.

Named local consumer: a future external-speedrun-history refresh that replaces stale #414-style assumptions only after #349 is merged, closed, or superseded.

Falsifier: upstream does not accept the row, changes the protocol, or a newer record supersedes it before local adoption.

Fire order: QUEUED-WITH-FIRE-ORDER after PR #349 final disposition. No leaderboard rewrite in `ddm_ng1`.

### 8. Narrow Q/K plus bigram sparse sink -> N-A / FOLDED

Narrow Q/K attention and bigram sparse-gradient plumbing are transformer-language-model mechanisms. The current Pact witness vehicle has no attention-token analogue or bigram embedding row sink.

Transfer: none for the current vehicle. The only indirect link is cadence discipline already captured in row 6.

Trigger to reopen: a future attention-bearing witness backend or sparse learned table appears with an actual scorer-facing role.

Fire order: FOLDED for current MAIN.

## Follow-On Ledger

| Item | Status | Owner / fire order |
|---|---|---|
| Source-verify PR #349 status, current claim, timed-region correction, hardware/protocol, ANVIL coefficient triples | FIRED | Completed in `ddm_ng1`; no runtime measurement. |
| Local MLX fp8 support check | FIRED | Completed in `ddm_ng1`; result folds direct fp8 adoption for current MAIN. |
| M1 same-config event-sigma field | QUEUED-WITH-FIRE-ORDER | Add to M1 ticket seal before MAIN fire. |
| Bench timed-region manifest and append-mode JSONL hygiene | QUEUED-WITH-FIRE-ORDER | Next WC/profiler/bench touch. |
| ANVIL-vs-Muon local race | QUEUED-WITH-FIRE-ORDER | After M1, only with update-matrix custody and matched realized update RMS. |
| Tail-blend law comparison against dy2 plateau-tail average | QUEUED-WITH-FIRE-ORDER | Post-M1 or dy2-owned governed slot. |
| Direct fp8 adoption on current MLX witness trainer | FOLDED | Reopen only on trainable fp8 path plus deterministic parity. |
| Narrow Q/K and bigram sparse sink | FOLDED | Reopen only if vehicle gains a real attention/table analogue. |
| Speedrun history refresh | QUEUED-WITH-FIRE-ORDER | After PR #349 is accepted, rejected, or superseded. |

## Boundaries And Non-Claims

Measured or source-verified in this receipt:

- External PR #349 status, reported timing/statistics, timing-boundary discussion, hardware/protocol summary, file list, and raw-code mechanics were source-verified from GitHub.
- Local recall surfaces were inspected for hot-state authority, WC3/GC21 M1 state, optimizer-custody precedent, dy2 tail averaging, stale speedrun history, and local MLX fp8 support.

Not measured here:

- No `upstream/evaluate.py`.
- No archive, archive bytes, decode, d_seg, d_pose, or exact contest score.
- No CPU/CUDA scorer, no Metal run, no MLX training run, no benchmark, no profiler, no remote job, no paid dispatch.
- No ANVIL implementation, optimizer replay, update-matrix capture, or coefficient-runtime validation.

Protected boundaries honored:

- Forbidden files from the charter were not touched.
- This receipt is design/crosswalk only and does not mutate any run directory.
- There is no positive score-claim row in `ddm_ng1`.

Own-vehicle frontier: `S=0.7534578126155775 @357,837 B [macOS-CPU advisory]` from tq1c / commit `4bf31d97a4`; contest pointer remains borrowed `0.19108`, unmoved by `ddm_ng1`.

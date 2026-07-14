# Codex findings — per-class convergence A/B — 2026-07-14

Pointer `0.1910828242 [contest-CPU]` unchanged; bank `0.1880443980` remains non-submission defensive.

## Outcome

The four real-n600, 150-epoch matched tickets are built and strict-memory-prefighted. No 150-epoch Metal
job, scorer authority run, byte-close, or pointer mutation occurred. Main owns actuation.

## Adversarial findings and closures

1. **Closed — wrong hinge identity risk.** `MarginBandSatisficing` is an additive positive-margin annulus
   controller and is not the analyzer treatment. A separate typed `ZeroMarginWinnerRivalHinge` now emits
   the exact whole-loss pair `margin_hinge, margin_target_end=0.0`; CE omits the target flag entirely.
2. **Closed — fake matched-custody risk.** A/B argv are test-pinned equal after removing only the arm tag
   and declared loss treatment. Receipt custody hashes seed/order, model schema+source, optimizer,
   curriculum, initial EMA bytes, non-treatment config, real cache bytes, and preregistration.
3. **Closed — receipt/checkpoint crash window.** A receipt is written immediately before its checkpoint.
   A crash there can replay one already-recorded update. The resume path now accepts exactly one replay
   only when every score/liveness field is identical, appends no duplicate, and carries both failed-work
   and replay wall time forward. Any drift fails closed.
4. **Closed — M+Adam partial restore.** M+ state serialization is mandatory; exact state key/shape/dtype
   matching is required on resume. Missing or partial optimizer state refuses continuation. NumPy-fp32 is
   the portable equation authority; the MLX subclass is lazy so headless tests remain possible.
5. **Closed in round-1 review — dry-start duplicate checkpoint ownership.** The initial dry-start builder
   appended `--ckpt-every 1` beside the ticket's `--ckpt-every 25`. The final builder structurally replaces
   the owned flag, reserves checkpoint/resume flags from free-form extras, and refuses every duplicate long
   flag in the final argv. All four hardened receipts have exactly one checkpoint flag and zero duplicates.
6. **Closed in post-fix review — incomplete treatment custody.** The step-native receipt/admission now
   includes HOSC omega, linear anneal, and FINER k. M+Adam includes and refuses drift in both
   `eta_a=eta_m=1e-3` and `tau=1e-6`; no treatment constant can hide in the matched-config fingerprint.
7. **Closed in post-fix review — prose-only dry-start authority.** Every arm's report is written atomically,
   content-addresses the exact real `launch.sh`, n600 cache, launcher, trainer, config, and 150-epoch target,
   and requires a resumed step strictly beyond the restored epoch. Real mode recomputes custody and refuses
   before spawn unless that exact report is green. A CE machine probe returned rc=12 with no trainer spawn.
8. **Closed in post-fix review — process/axis custody.** The trajectory and DSL share the explicit
   `[macOS-MLX training-gradient]/[macOS-CPU advisory] verdict` axis. The handoff claims and terminalizes
   each independently spawned arm under its own job id with the same 168-hour TTL.
9. **Open host gate — Metal boot.** All four governed `--dry-start 2` attempts reached the exact trainer
   launch but failed at `mlx.nn` import with `No Metal device available`; pass 2 was correctly skipped.
   This is `BLOCKED_HOST_NO_METAL`, not boot-green and not evidence that trainer integration works on Metal.
   Main must rerun this gate on M5-Max and refuse the full arm unless both boot and resume are green.
10. **Open storage gate — sandbox permission.** Vertigo measured 827,380,576,256 free bytes, but this
   sandbox could not create the expected workload root; authoritative selection remains null. Main must
   create/select it through the storage
   waterfall before Metal dry-start; local disk is not authorized for the 150-epoch artifacts.
11. **Bounded interpretation.** The telemetry separates class/stratum convergence from measured temporal
   floor components, but does not invent a causal MCF-erasure scalar. `d_seg-0.005318` is a curable-excess
   upper bound only. SPS ep275 is an uninformative disengaged instance, so only real screw/phase-engaged
   n600 telemetry with the converged #121 taper could admit an SPS interpretation. Adam-beta2/reference-
   semantics and M+Adam/Muon remain explicit optimizer reformulations. A naive or first-cut negative
   remains INSTANCE-scoped with the optimal reformulation queue open; this build makes no
   loss/basis/optimizer verdict and adds no fifth arm.

## Verification

- 149 focused/broad contract tests passed; 1 Metal parity test skipped on this host.
- New standalone modules and tests are Ruff-clean; all changed Python parses with `py_compile`.
- Each emitted `launch.sh` validates every flag and passes the DSL and schedule-provenance gates.
- Independent round-1 and post-fix adversarial reviews have no open HIGH/P0 finding; all reported
  implementation and handoff findings are closed above.
- Three consecutive clean passes are recorded on the changed Python surfaces with two distinct L3
  approvers (`codex`, `council`); the strict review seal is satisfied.
- Strict memory result for every arm: 24.48 GiB projected versus 89.6 GiB ceiling.

Primary durable receipt: `.omx/research/perclass_convergence_ab_preflight_20260714.json`.
Triality feed: `.omx/research/perclass_convergence_ab_DAG_FEED_20260714.md`.
Exact launch/analyzer runbook: `.omx/research/perclass_convergence_ab_main_handoff_20260714.md`.

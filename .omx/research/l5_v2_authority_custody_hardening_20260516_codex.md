# L5-v2 authority and custody hardening - 2026-05-16

## Scope

This ledger records adversarial fixes against the L5-v2 staircase, PR106
PacketIR stack intake, Modal CPU/CUDA exact-eval axis handling, and paper/source
fidelity. It is not a score claim.

## Changes

- Prediction-band CPU anchors now reject non-CPU hardware, non-CPU inflate or
  eval devices, and unrecognized/non-CPU auth-eval commands.
- PR106 PacketIR paired exact status now requires runtime-consumption content
  tree custody to match both exact axes. Three previously paired rows are now
  `paired_exact_blocked` until runtime-bound custody is closed.
- L5-v2 PR106 stack selection now sees zero runtime-bound paired PacketIR
  candidates and fails closed with
  `l5_v2_packetir_no_runtime_bound_paired_exact_candidates`.
- L5-v2 probe disambiguation requires per-axis artifact paths before
  architecture lock.
- TT5L side-info consumption proof generation rejects custody outputs outside
  the repository and emits repo-relative inflated-output paths.
- Modal auth-eval reusable custody helpers are exported through the public
  module API so packet builders do not need duplicate runner logic.
- Substrate docstrings were stripped of unsupported paper-derived score claims;
  remaining bands are planning priors until paired exact anchors exist.

## Current PR106 PacketIR Status

- Candidate count: 16.
- Paired exact measured candidates: 0.
- Paired exact blocked candidates: 3.
- Runtime-consumed candidates needing paired exact eval: 4.
- Single-axis exact candidates needing the missing pair axis: 9.
- Next exact-eval targets: 17.

## Current TT5L Proof

- Proof artifact:
  `.omx/research/tt5l_sideinfo_consumption_proof_20260516_codex.json`
- Proof SHA-256:
  `8bb68ba5e14f0bbb0511812cbb7b7465e58ef639997e300558c04c3cdae98605`
- Manifest artifact:
  `.omx/research/tt5l_sideinfo_consumption_manifest_20260516_codex.json`
- Runtime-tree SHA-256:
  `4f4f5d2e090386d90962145727ea3bfc74f417e3d034ecea4a81d43de3b81ff4`
- No operator-home absolute paths remain in the committed TT5L proof or
  manifest.

## Verification

```text
.venv/bin/python tools/prove_tt5l_sideinfo_consumption.py
artifact_sha256 = 8bb68ba5e14f0bbb0511812cbb7b7465e58ef639997e300558c04c3cdae98605
predicate_passed = true

rg -n "<operator-home-absolute-path-pattern>" \
  .omx/research/tt5l_sideinfo_consumption_manifest_20260516_codex.json \
  .omx/research/tt5l_sideinfo_consumption_proof_20260516_codex.json
no matches
```

## Remaining Blockers

- No L5-v2 score or rank dispatch is ready until a real empirical anchor exists.
- PR106 PacketIR stack composition remains blocked until runtime-bound paired
  CPU/CUDA exact candidates exist.
- Next exact-eval target rows are dispatch targets only; every provider launch
  still requires `tools/claim_lane_dispatch.py`, Modal recovery, component
  recomputation, and adversarial result review.

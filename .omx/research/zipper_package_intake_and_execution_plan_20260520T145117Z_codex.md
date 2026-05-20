# Zipper Package Intake and Execution Plan

**UTC:** 2026-05-20T14:51:17Z
**Owner:** Codex
**Source package:** `.omx/research/inbox_zipper_20260520T144021Z_codex/research_package.zip`
**Source package SHA-256:** `9ffce5d7802ebdb0669b21888b7ec76496561dee652c72f7993be803c38d2506`
**Lane:** `lane_zipper_package_intake_20260520`

## Intake Verdict

The Zipper package is useful as an advisory roadmap and reviewer model, but it
is not implementation authority. The PR #110 surface recommendations were
validated and applied separately. The frontier roadmap contains several good
directions, but many commands reference nonexistent local CLIs or generic
script names, and the implementation scaffolds are explicit pseudocode.

Therefore this intake promotes only items that have a current Pact surface or
can be converted into a fail-closed local artifact without GPU spend.

## What Landed

1. **PR #110 public surface polish:** Applied validated package advice to the
   live PR body and release body/title. No branch push and no release asset
   replacement.

2. **Zipper attachment custody:** Downloaded via GWS from Gmail message
   `19e45d4d3a5a22d1`; extracted under
   `.omx/research/inbox_zipper_20260520T144021Z_codex/extracted/`.

3. **Lane pre-registration:** Registered
   `lane_zipper_package_intake_20260520` at L0.

4. **Concrete FEC7 follow-up artifact:** Ran the existing byte-closed FEC7
   selector entropy profiler instead of using the package pseudocode.

   Artifact:
   `experiments/results/pr110_zipper_fec7_selector_profile_20260520_codex/profile.md`

   Result: FEC7-style charged selector recoding is currently blocked. The
   FEC6 selector is 249 bytes; the best charged FEC7 candidate is 268 bytes,
   i.e. 19 bytes worse. It cannot meet the 79-byte target saving needed to
   matter for a strict PR #110 same-runtime byte-only path.

## Package Item Dispositions

| Package item | Disposition | Reason | Next executable gate |
|---|---|---|---|
| `PR110_*` docs | applied | Validated against live PR/release facts; tone and factual changes applied. | Monitor maintainer feedback; no more live edits unless needed. |
| S1 PR #110 verify | mostly complete | Archive SHA/bytes/ZIP layout/report components already verified; live PR/release re-read after edits. | Do not rerun expensive eval unless maintainer requests or drift evidence appears. |
| S2 packet compiler | already exists | Canonical packet compiler and PR110 deterministic identity closure exist; package command names are stale. | Keep using `tac.packet_compiler.deterministic_compiler` surfaces. |
| S3 HNeRV table | partially complete | Public lineage facts exist, but full same-axis table is expensive and time-sensitive. | Build/update scorecard only from canonical frontier scan and exact replay artifacts. |
| S4 byte profile | partially superseded | Master-gradient and selector-profile tooling exist. | Use targeted byte/profile tools on a candidate that can affect build/eval, not generic bit flips across the whole archive. |
| M2 FEC7 selector | terminal local negative | Existing FEC7 charged prototypes are worse than FEC6 on current selector bytes. | Reactivate only with a new selector model/prior whose charged model+stream is at least 79 bytes smaller than FEC6 or with a component-score improvement, not byte recoding alone. |
| SIREN/VQ-VAE/foveation/RAFT scaffolds | advisory only | Package files are pseudocode and some are buggy or not contest-exportable. | Convert one at a time into repo-native timing smokes with no score claim and explicit export grammar blockers. |
| Cloud GPU spend plan | advisory only | Mentions generic flags and budgets; Pact dispatch must use lane claims and provider helpers. | Any spend requires lane claim plus Modal/Vast/Lightning contract compliance. |
| Theoretical floor memo | advisory, contains math bug | It understates the archive term in one scenario; current 178517-byte rate term is about 0.118867, not 0.0001-0.0002. | Do not cite as floor authority until corrected against the score formula. |

## Execution Queue

1. **Z-FEC7-BYTE-NEGATIVE: completed.**
   Terminalize byte-only FEC7 recoding as negative for the current PR #110
   selector payload. Evidence:
   `experiments/results/pr110_zipper_fec7_selector_profile_20260520_codex/profile.json`.

2. **Z-SOURCE-MAP-AUDIT: completed.**
   Classify the source-map/candidate-inventory items as `empirical`,
   `scaffolded`, `landed_tooling`, `design_only`, or `external_reference`
   against real files in this checkout. Output should be a durable table, not
   a new strategy essay.

   Artifact:
   `.omx/research/zipper_source_map_surface_audit_20260520T145523Z_codex.md`

   Result: Most Zipper ideas already have stronger repo-native surfaces than
   the package pseudocode. FEC7 byte-only recoding remains terminal local
   negative; SIREN/VQ-VAE/foveation/RAFT/Cool-Chic/C3 remain gated by existing
   readiness/export surfaces, not package commands.

3. **Z-C3-COOLCHIC-SMOKE-SPEC: completed.**
   Reconcile the package's C3/Cool-Chic recommendation with existing
   `src/tac/packet_compiler/{balle_hyperprior,factorized_prior,cheng2020}.py`
   and prior Cool-Chic/C3 prototype records. Output should be a no-spend
   timing-smoke/runbook or an explicit blocker.

   Artifact:
   `.omx/research/zipper_followup_readiness_gates_20260520T150513Z_codex.md`

   Result: Existing materializers produced empty research-signal sidecar
   archives for both Cool-Chic and C3, with explicit `score_claim=false` and
   `ready_for_exact_eval_dispatch=false`.

4. **Z-SIREN-VQVAE-EXPORT-GATE: completed.**
   Audit existing `src/tac/vqvae_as_full_renderer.py`,
   `src/tac/ffnerv_as_renderer.py`, and any SIREN/INR surfaces. Define the
   smallest byte-closed export grammar gate before any GPU training.

   Artifact:
   `.omx/research/zipper_followup_readiness_gates_20260520T150513Z_codex.md`

   Result: SIREN readiness audit reports local first-anchor readiness and
   explicit dispatch blockers; focused SIREN and VQ-VAE tests pass. A small
   auth-eval gate schema/wiring patch landed so SIREN consumes
   `auth_eval_exact_cuda_complete` from the canonical shared gate.

5. **Z-FOVEATION-RAFT-GATE: completed.**
   Reconcile package foveation/RAFT suggestions with existing
   `src/tac/lapose_foveation_atoms.py`, `src/tac/multi_res.py`, and renderer
   lane history. Output should be a local CPU smoke or an exact blocker.

   Artifact:
   `.omx/research/zipper_followup_readiness_gates_20260520T150513Z_codex.md`

   Result: Focused foveation payload and RAFT pipeline tests pass locally.
   Any future real RAFT/foveation dispatch remains lane-claim and provider
   gated.

## Guardrails

- No live PR #110 branch push from this intake.
- No release asset replacement.
- No GPU dispatch from package prose.
- No pseudocode copied into `src/tac` as if production-ready.
- No score claim from local, MPS, proxy, or design-only surfaces.
- Every future run must produce an artifact, exact failure classification, or
  dispatch-claim-gated job.

# Roadmap Queue Adversarial Review - 2026-05-09

<!-- generated_at: 2026-05-09T14:50:00Z -->
<!-- evidence_grade: adversarial_review; no score claim; no remote dispatch -->

## Scope

Reviewed the operator-provided "Immediately actionable / Deferred / GPU-spend
gated" queue against current custody surfaces:

- `.omx/state/active_lane_dispatch_claims.md`
- `reports/cathedral_autopilot_evidence.jsonl`
- `.omx/research/roadmap_outstanding_work_audit_20260509_agent.md`
- `tools/constrained_coord_search_pr101_bias_sidecar.py`
- `tools/claim_lane_dispatch.py`
- `src/tac/preflight.py`

No score is promoted here. Predicted bands remain planning signals only.

## Findings

1. **A1 dual-CUDA dispatch is not "pending"; it already exists and is negative
   for CUDA-axis readiness.** The same archive SHA
   `87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5`
   has `[contest-CPU]` `0.19284757743677347` on GHA Linux x86_64 and
   `[contest-CUDA]` `0.2263520234784395` on Modal T4. Treat the A1 public-axis
   result as real, but do not call it PR-submission-ready until the CPU/CUDA
   mechanism is resolved or submission policy explicitly accepts CPU-only public
   reproduction.

2. **Constrained coordinate search is active; do not duplicate dispatch.**
   Latest active claim:
   `lane_pr101_bias_constrained_coord_search` /
   `constrained-coord-search-m5max-20260509T142522Z`. The local M5 Max phase is
   compatible with the queue. The GHA promotion phase remains gated on
   adversarial candidate selection review.

3. **The constrained-coordinate claim had a signal-loss bug.** Its prior notes
   shell-expanded dollar costs (`$0` / `$0.10` / `$0.50`) into `/bin/zsh`
   strings. Codex appended a corrected latest active claim row with literal
   costs and added helper validation so future claim notes reject unwaived
   shell-expanded `$0`/argv[0] patterns.

4. **AVVideoDataset discriminator is active; do not relaunch a duplicate.**
   Latest active claim:
   `lane_avvideodataset_cuda_path_mechanism_discriminator` /
   `discriminator-sweep-20260509T110211Z`, platform
   `github_actions+lightning`. Harvest or monitor it; a new dispatch would be
   conflict-prone.

5. **Check #125 was the remaining visible warn-only ratchet.** Live count was
   19 landing memos missing one or more unified-Lagrangian wire-in declarations.
   Codex backfilled the 19 legacy post-cutover memos conservatively with
   per-hook `N/A` rationale declarations and flipped the preflight call to
   strict. This is a structural DX/preflight hardening action, not a score
   result.

6. **Phase 1/T1 Ballé and Lane 12-v2 remain valid research directions but are
   not "fire immediately" items while the repo is dirty and active claims are
   open.** They require green strict preflight, no active lane conflict, claimed
   dispatch rows, exact runtime packet custody, and per-lane no-op proof before
   paid GPU spend.

7. **A1 V7/fine-grid and full sidecar resample should be treated as one
   coordinated A1 same-archive family.** The active constrained-coordinate job
   already covers the near-anchor constant search. Full 600-pair latent sidecar
   resampling remains separate and should not be launched until the current
   coordinate results are harvested or explicitly declared orthogonal.

## Dispatch Policy From This Review

- Continue $0 local/M5 Max prefilter work already claimed.
- Do not launch duplicate GHA/CUDA jobs for active `lane_pr101_bias_constrained_coord_search`
  or `lane_avvideodataset_cuda_path_mechanism_discriminator`.
- Do not promote A1 as CUDA-ready; record it as split-axis evidence:
  strong `[contest-CPU]`, regressed `[contest-CUDA]`.
- Before any new remote/GPU/eval dispatch, rerun `tools/claim_lane_dispatch.py summary`
  and strict preflight, then claim the lane.

## Next Highest-EV Actions

1. Harvest/monitor active constrained-coordinate M5 Max results, then adversarially
   select at most 1-5 GHA CPU candidates.
2. Harvest/monitor active AVVideoDataset discriminator; use it to explain the A1
   CPU/CUDA split before any PR submission decision.
3. Run strict preflight after current dirty-state fixes land; do not spend GPU
   from a red preflight tree.
4. Build the A1 full latent-sidecar resample as a byte-different,
   runtime-consumed packet only after the coordinate-search lane reports.
5. Revisit Phase 1/T1 and Lane 12-v2 only after the active claims are harvested
   and packet custody/no-op proof is green.

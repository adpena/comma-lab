# Codex Findings: PR110 Source-Map Recursive Review

UTC: 2026-05-20T14:02:29Z

## Scope

Reviewed the live PR #110 body plus the `comma-lab` docs commit currently linked from it:

- Live PR: `commaai/comma_video_compression_challenge#110`
- Linked `comma-lab` commit after fixes: `b7f16a081ee381803dd5d917bdaf805453fb81f3`
- Linked docs:
  - `docs/asymptotic_floor_candidate_inventory.md`
  - `docs/full_stack_source_map.md`

Primary review axes: hallucination risk, math/optimization rigor, engineering provenance, citation density, live-link correctness, archive custody, author attribution, runtime-dependency boundary.

## Issues Found And Fixed

1. Source-map wording overclaimed by implication.
   - Changed `paper-ready map` to `paper-oriented map`.
   - Changed source-map prose from `paper-ready` to `citation-heavy`.
   - Replaced `best next result` style wording with `plausible next result`.

2. Falcon/SAM row had benchmark-marketing risk.
   - Removed “reports stronger mask quality than SAM 3 on SA-Co” from the source map.
   - Replaced with a bounded statement: Falcon is an external open-vocabulary grounding/segmentation reference; external benchmark comparisons are priors, not contest evidence.

3. Citation density was too low for several named method families.
   - Added links for comma2k19, Mamba-2, Ballé/Balle hyperprior, Mallat, Daubechies, Fridrich/Kodovsky rich models, curriculum learning, distillation, LoRA, SWA, Lagrangian duality, ADMM, and Dykstra.
   - Removed uncited `E-NeRV` and `COIN-style` labels from the source map.

4. openpilot prior wording was too strong.
   - Changed “same road-video distribution” to “same broad driving-video domain.”
   - Preserved the contest boundary: openpilot-derived surfaces are compress-time priors unless every runtime byte is charged.

## Empirical Verification

- Downloaded PR #110 release archive from GitHub Release.
- Recomputed size: `178517` bytes.
- Recomputed SHA-256: `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`.
- Verified ZIP member: single member `x`, `178417` compressed/uncompressed bytes, stored (`compression method: none`).
- Verified PR author handles via `gh api`:
  - PR #95 `AaronLeslie138`
  - PR #98 `EthanYangTW`
  - PR #100 `BradyMeighan`
  - PR #101 `SajayR`
  - PR #102 `EthanYangTW`
  - PR #103 `rem2`
  - PR #108 `andrei-minca`
  - PR #110 `adpena`
- Verified PR #110 runtime tree exists at `ec6cc7f98c16b6ad2db8bc7cde65757bb7993004`.
- Verified live PR body now pins linked `comma-lab` docs to `b7f16a081ee381803dd5d917bdaf805453fb81f3`.

## External Source Checks

Current-source links checked during review:

- Meta SAM 3 / SAM 3.1: `https://ai.meta.com/blog/segment-anything-model-3/`
- SAM 3 paper: `https://arxiv.org/abs/2511.16719`
- Falcon Perception: `https://arxiv.org/abs/2603.27365`
- openpilot 0.11 world-model release: `https://blog.comma.ai/011release/`
- comma10k: `https://github.com/commaai/comma10k`

Publisher links for IEEE/Wiley/JSTOR/ScienceDirect/Now Publishers may return bot-blocking HTTP codes to `curl`; this was not treated as dead-link evidence.

## Clean-Pass Log

Clean pass 1: factual custody and provenance.
- Live PR body points to `b7f16a081ee381803dd5d917bdaf805453fb81f3`.
- Archive bytes/SHA/member/storage verified.
- PR author handles verified.
- Runtime tree verified.
- Result: no issues.

Clean pass 2: hallucination/math/engineering-rigor scan.
- Risky wording scan clean for overclaim patterns introduced by this landing.
- Source-map table structure valid.
- Local markdown links valid.
- `comma-lab` worktree clean at pushed commit.
- Result: no issues.

Clean pass 3: final-state consistency.
- Live PR/docs/archive links resolved.
- PR tone/risk scan clean.
- Memo shape and diff hygiene clean.
- Result: no issues.

## Current Verdict

After fixes, PR #110 plus the linked `comma-lab` source map are materially safer on hallucination, math-rigor, and engineering-provenance axes. The source map remains appropriately outside the PR body, with status labels and promotion gates separating external references from contest evidence.

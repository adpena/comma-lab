# ddm_fx4 — CURE 5: the GT-cache lineage split (a cross-leg decoder confound)

**Operator 2026-08-09: "Continue with all."** This is the confound that silently taints every
cross-leg comparison we are about to make on the PR130 base. It also joins a live P0 (#906).

## THE FINDING (rr1, `ecfd4ec595`, do not re-derive)

`RR1_PREPARE_VERIFY_AUDIT.md`, verbatim: *"semantic QAT consumes the AV-like `gt_cache_600.pt`, while
carrier, HPAC, and token encoding consume the DALI `gt_cache_600_official_ada.pt`. The strict 49-stage
graph does not preserve that history; it builds one DALI cache and routes it to 41 downstream stages.
Cross-leg metric comparisons in the retained replay therefore carry a decoder/target confound."*

Two independent facts make this load-bearing rather than academic:
- MAIN measured semantic inference at **DALI-GT 0.0002857038709852431 = 0.998650× published Ada** but
  **AV-GT 0.0002764044867621528 = 1.000123× the stage-08 recorded value** — i.e. each leg reproduces
  its OWN decoder's number. Same weights, different target, two different "reproductions."
- **#906 priced chroma siting at sensitivity 2.2791e-4 = 79.66% of PR130's ENTIRE seg term.** The GT
  decoder path determines chroma. A decoder difference is not a rounding difference on this axis.
- Standing rule: GT decodes ONLY via `frame_utils.yuv420_to_rgb`; PyAV `rgb24` manufactures ~100×
  phantom pose. Establish which path each cache actually took before interpreting anything.

## YOUR SCOPE

1. **Trace both caches to their producers.** For `gt_cache_600.pt` and `gt_cache_600_official_ada.pt`:
   what command built each, from which decoder, at which color conversion, with what receipt. Hash both.
   If a lineage is unrecoverable, that is an honest finding — say which and why, do not infer it.
2. **Measure the delta, do not characterize it.** Per-pixel and per-class difference between the two
   caches over n600 (or a stratified n≥120 random sample — **never a prefix**; m88/m96: prefixes are
   scene blocks, and the bias SIGN INVERTS by axis, pose 2.5–4.2× harder, seg ≈0.96× easier). Report the
   delta in the units that matter: argmax flips, and the seg/pose/S consequence.
3. **Answer the roadmap question: which target is canonical for OUR iteration?** PR130's published row
   is contest-CUDA/DALI. If we train a leg against AV-GT and evaluate against DALI-GT, we are optimizing
   one object and scoring another. Give a DERIVED recommendation with the arithmetic, not a preference.
4. **Fix the graph's memory.** The 49-stage builder collapses the distinction by construction. Either
   preserve per-stage GT provenance in the retained replay, or make any cross-leg comparison that spans
   the two caches REFUSE. A confound the apparatus cannot see is the L3 verdict-clearance failure mode.
5. **Join #906 explicitly.** Name what this measurement changes for the DALI-vs-AV Modal job, and
   whether it makes that dispatch cheaper, unnecessary, or still required.

## OPTIMAL FORM

- **Reference form:** both caches traced to their producing commands with receipts, the inter-cache
  delta measured over n600 (or stratified n≥120) in argmax-flip and S units, and a DERIVED canonical-
  target recommendation with its arithmetic.
- **SCOPE reductions (legal):** stratified-random n≥120 instead of n600 if wall-clock forces it — state
  n and the sampling seed. CPU-only.
- **MECHANISM reductions (declare TOY-BRACKET):** a PREFIX instead of a stratified sample (this is the
  named false-negative shape and it is forbidden as evidence); characterizing the difference as "small"
  without measuring it in S units; inferring a lineage from a filename.
- **Provenance pins:** rr1 audit `ecfd4ec595`; semantic DALI 0.0002857038709852431 / AV
  0.0002764044867621528; #906 chroma sensitivity 2.2791e-4; base PR130 CPR1 `0.172141297491896447`
  `[contest-CUDA, DALI GT, n600]`.

## NON-NEGOTIABLES

- Intake READ-ONLY; upstream snapshot IMMUTABLE. GT decodes only via the canonical path.
- **n≥120 stratified-random, NEVER a prefix** — for any claim about population behavior.
- MPS/MLX never score authority; `score_claim=false`.
- **Never consume a background job's output without asserting terminal status.**
- verdict_scope on every negative. Denominators on every count.
- Commit via `tools/subagent_commit_serializer.py`, POST-EDIT `--expected-content-sha256` per file,
  tags `[no-triality] [p0-ledger-ok]`. NO Claude/AI attribution, no `Co-Authored-By`.
- `REVIEW_GATE_OVERRIDE=1` FORBIDDEN with `.py`; fine for `.md`/`.json`.

## DELIVERABLE

`.omx/research/ddm_pr130_reproduce_20260809/FX4_GT_LINEAGE.md` — **§1 = the measured inter-cache delta
in argmax-flip and S units**, then both lineages with receipts, the DERIVED canonical-target
recommendation, the graph-provenance cure, the #906 consequence, ranked residuals with falsifiers, and
"could not check / why."

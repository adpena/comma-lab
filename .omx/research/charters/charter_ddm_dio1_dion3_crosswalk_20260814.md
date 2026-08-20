# CHARTER — ddm_dio1_dion3_crosswalk (2026-08-14, operator drop)

OPERATOR DROP: arXiv 2608.11612 — Amsel/Zhang/Ahn/Naeimi/Feng/Chen/Tri Dao/
Langford (Microsoft), "Dion3: Full-Stack Orthogonal Updates" (37pp, code
github.com/microsoft/dion). Claims: Gram Newton-Schulz cuts orthogonalization
FLOPs · CuteDSL kernels exploit symmetry · megabatching cuts communication ·
ROW-SUBSET orthogonalization per step (improves Dion) · up to 6× optimizer
step-time cut vs Muon at comparable-or-better loss · drop-in Muon replacement.

## THE TASK — rigor-triage-first crosswalk (fa1/if1/px1 protocol)
1. DEEP-READ the paper + the dion repo (off-the-shelf grant covers full use).
   Triage every claim: what is measured, at what scale, on what hardware.
2. SCALE HONESTY FIRST: their wall is LLM-scale cubic NS + sharded
   communication. OUR matrices are tiny (decoders ~100K–300K params; witness
   trunks similar) and training is single-host MLX/Metal (m5 Max) — no
   sharding. Measure/derive whether NS orthogonalization is even a
   material fraction of OUR step time (cite the #306 per-lever compute audit
   + #443 kernel-stack receipts; measure fresh if stale). If NS is <5% of
   our step, the 6× headline is N-A at our scale — say so plainly and
   pivot to the transferable IDEAS.
3. THE TRANSFER CANDIDATES (adjudicate each ADOPT / ADOPT-CLASS /
   LESSON-ONLY / N-A with named consumer + falsifier):
   a. Gram-NS (X^T X symmetric form) — fewer FLOPs + better numerical
      behavior at OUR small-matrix scale? Consumer: the Muon stage in the
      TR1/burn-4 trainers + tac MLX optimizer stack (#469 MuonH line).
   b. ROW-SUBSET orthogonalization — a per-step stochastic subspace lever;
      crosswalk vs our #552 SPD-momentum product-chart + #556
      FilmPolarSPDNormalMomentum (gated arm) + px1's NS-quality lesson.
   c. Megabatching — relates to our --micro-batch-pairs batched twin
      (#313/#447); any scheduling lesson for the MLX pipeline?
   d. Update-RMS matching (px1's fairness methodology) — does Dion3's
      eval protocol confirm/refine it? Fold into the A/B methodology laws.
   e. The dion package itself — torch-only? If a CUDA-side consumer exists
      (Modal T4 rows are eval-only, no training), likely N-A; the MLX port
      question is priced NOT assumed.
4. Named consumers for ADOPT rows: the JS1/#982 joint gate-aware treatment
   (training engine choice, from na7) · rx2 successor training rounds ·
   burn-4/TR1 Muon stage · canonical_equations optimizer laws. Every ADOPT
   carries a $0-probe design w/ falsifier; no training launches from this
   arm.

## OPTIMAL FORM
PINS: arXiv 2608.11612 · github.com/microsoft/dion · px1 memo (#685
Khona/SOAP-Muon crosswalk) · #552 SPD-momentum verdict · #306 per-lever
compute receipts · #443 MLX/Metal kernel-stack receipts · MC36 frontier
archive f0ba4bb41d55fff85542f2a17dfe682508aa4f9ab50ef51cda573d79f0c4b1de.
Reference form = the fa1/if1/px1 rigor-triage crosswalk protocol (extend,
do not rebuild). Every transfer claim priced at OUR scale with a receipt or
an explicit fresh measurement — no LLM-scale numbers quoted as ours (m94:
the instrument must measure in the claim's units). TOY-BRACKET: any step-time
claim not measured on our real trainer config. Online research + full repo
reading authorized. Git-blocked ⇒ memo SHA handoff.

## OUTPUT
Memo .omx/research/ddm_dio1_dion3_crosswalk_20260814.md: claim triage table ·
our-scale NS-fraction measurement/receipt · ranked ADOPT/ADOPT-CLASS/
LESSON-ONLY/N-A rows w/ consumers + falsifiers · $0-probe designs. Serializer
commit, [no-triality] [p0-ledger-ok], no co-author trailer.
NEXT_IF_RESUMED + LIVE-HYPOTHESES + DEAD-ENDS.

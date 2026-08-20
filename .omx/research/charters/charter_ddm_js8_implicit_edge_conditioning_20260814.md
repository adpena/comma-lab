# CHARTER — ddm_js8_implicit_edge_conditioning (2026-08-14, the js1 joint line's named successor)

PARENT TRIGGER (verbatim, TASK_1043_TRIGGER_RECEIPT.json sha ad9da227…):
"build the implicit decoder-derived edge-state conditioning consumer; retain
every payload and do not ship an explicit edge mask" — owner: JS8; consumer
store /Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/edge_conditioned;
fire trigger SATISFIED (full-n600 CUDA stage-0 decomposition + fields retained
under the JS1C store). READ FIRST: the JS1C store
(/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/contest_cuda/ddm_js1c_20260814/,
STAGE0_RESULT.json sha 472fc816…) + .omx/research/ddm_js1c_cuda_custody_stage0_20260814.md
+ the EC2 refusal receipts in the edge_conditioned store.

## WHY THIS SHAPE (the two closures that route here)
1. T1R1/V0-V5 is FOLDED on T4 custody (rho < 0.827795) — fixed linear
   per-edge instances are DEAD on this base; do not rebuild them.
2. The EC2 FIRST implicit-conditioning instance was REFUSED (−40,779 receipt
   in the edge_conditioned store) — the naive first pass is also dead. THE
   NEW INGREDIENT this arm exists to exploit: the retained T4-custody
   per-edge decomposition (which edges carry which flip mass, on the REAL
   scorer instrument) that neither prior attempt had. If the second instance
   does not CONSUME that decomposition to pick its conditioning structure,
   it is a re-run of EC2 and must not be built.

## THE TASK
1. RECALL: read the EC2 refusal mechanism at source (why −40,779 — which
   stage ate the gain) + the STAGE0_RESULT per-edge/road-hub decomposition
   (Road hub: ~87.8% of flips touch Road; Road↔Lane 49.2% — the m91 law).
2. DESIGN the implicit conditioning so the edge-state is DERIVED at decode
   from the decoder's own already-decoded state (rule-118 FREE — generic
   algorithm in inflate, zero counted mask bytes), with the decomposition
   selecting WHERE conditioning binds (the measured hub edges first).
3. BUILD to admission on the cp135/MC36 base (archive f0ba4bb4…@186,269 B):
   receiver-closed, byte-identical when inactive, exact parse-back.
4. MEASURE locally (CPU-torch advisory, n600, real decode path): realized
   joint ΔS = seg + pose + rate vs the base. TOY-BRACKET: any verdict from
   a subset without full-field confirmation, or a modeled-not-realized flip
   count (the qs3 B/H lesson: benefit-exact realized accounting only).
5. If realized advisory ΔS < 0 beyond the local noise floor → SEALED T4
   dual-axis fire-order for MAIN (Standing GO covers). If refused → typed
   mechanism + re-route recommendation (trained-receiver #982 rx2-line vs
   coupled multi-token #978), per-edge evidence attached.

## OPTIMAL FORM
PINS: STAGE0_RESULT sha 472fc816f6656ec0cdd37bd475598e8e9683260dc97adeb4163ead5ae90b3e67 ·
fields candidate/base/gt shas in CONSUME_RESULT (gt 91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248) ·
MC36 archive f0ba4bb41d55fff85542f2a17dfe682508aa4f9ab50ef51cda573d79f0c4b1de @186,269 B ·
trigger receipt sha ad9da227d6329efcbf510f084f748c5f53e28522bea8bb30e694d004bb4ce8e0.
Reference form = the EC2 consumer machinery in the edge_conditioned store
(reuse/extend, do not rebuild parallel). Payload law: retain every
conditioning table + candidate archive + decode receipt. Decode-time law
binds. In-compile pose compensation per the proven qs5 pattern if frame-1
edits are made. Git-blocked ⇒ memo SHA handoff.

## OUTPUT
Work dir = the named consumer store edge_conditioned/ (js8 subdir). Memo
.omx/research/ddm_js8_implicit_edge_conditioning_20260814.md: EC2-mechanism
recall · design derivation from the decomposition · admission receipts ·
realized advisory ΔS table · sealed fire-order or typed refusal+reroute.
Serializer commit, [no-triality] [p0-ledger-ok], no co-author trailer.
NEXT_IF_RESUMED + LIVE-HYPOTHESES + DEAD-ENDS.

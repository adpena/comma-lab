# ddm_dcf1_duplicate_carry_factorization — what does lb1 carry TWICE, what survives a real tree-shake, and what factors (owning memo: ddm_dcf1_duplicate_carry_factorization_20260831.md)

## MANDATE

Operator 2026-08-31, verbatim: *"No duplicate carry or representation and fullest shaking and
factorization and flattening"* — under the standing goal: *"do whatever it takes… autonomously with
full authority and standing go to accomplish frontier score lowering… be creative and weird and
think divergently."*

Four verbs, ONE object: the shipped **lb1** archive (sha
`5b856e667961dd9ab68ddd7166384662bfb5912fabc8c9270098ea63a8ad28c9`, **180,083 B**, lane
`ddm_lb1_joint22_patch192_cuda_n600_20260829`). Sub-0.12 demands **shed 42,097 B** at fixed
distortion (≤137,986 B), exchange 6.658e-7 S/B.

This is not a new family. It is the **DEDUPLICATION / FACTORIZATION** reading of a body that five
arms have measured as jointly optimal *for its own parameterization* — which is exactly the regime
where duplicate carry survives, because a co-fitted optimum has no pressure to notice that two
sections encode the same thing.

**The measured fact that makes this concrete, not a slogan:**

- `ar1b` (#1213) mapped the 66,591 B residue **EXACTLY, zero remainder**:
  **renderer 30,856 B · carrier 22,010 B · HPAC model 13,515 B.**
- `#1222` measured, at 1,356× in the wrong direction, that **PoseNet scores the FRAMES — so the
  RENDERER carries pose, not the pose carrier.**

If the renderer already carries pose, **what is the 22,010 B carrier carrying that the renderer does
not?** That is a duplicate-carry question stated in bytes, and it is 52.3% of the byte demand.

## SCOPE — four verbs, in this order, each with a denominator

1. **DUPLICATE CARRY (the operator's first verb).** Over the shipped lb1 body, enumerate every pair
   of stored quantities and ask whether either is **derivable from the other** at decode. Prioritise
   the pair `#1222` hands you: renderer × pose-carrier. For each candidate pair report: what each
   side stores · whether one is a function of the other (exactly / lossily / not at all) · bytes at
   stake · what a decoder would need to derive it (rule-118: GENERIC algorithm is FREE, video-derived
   content is COUNTED). **Answer the carrier question explicitly**, with a receipt.

2. **FULLEST SHAKING.** Tree-shake the archive: which stored bytes are **actually consumed** by the
   receiver on the real decode path? The apparatus already exists — **`#417`'s
   receiver-consumption bijection** was built for exactly this class (counted-but-inert-through-R is
   a NO-FAKE blocker, not a nicety). USE IT; do not rebuild it. Report bytes {consumed · inert ·
   indeterminate} with the method that decided each. Any inert byte is a free win and a NO-FAKE
   finding simultaneously.

3. **FACTORIZATION.** Where is the body storing a PRODUCT it could store as FACTORS? Candidates the
   corpus already names: `d3b`'s Lane chain-rule factorization · shared structure across the 38
   semantic tensors · basis×coefficient in the 22,010 B carrier pool · cross-pair structure. Price
   each with REAL coders on the real payload — a factorization that does not shrink the coded stream
   is not a factorization, it is a re-parameterization.

4. **FLATTENING.** Where does the format pay for indirection — nested length prefixes, per-section
   headers, index tables, address layers — that a flatter grammar would not? `dx2`'s addressing was
   measured ALREADY FREE (#1200), so **do not re-measure addressing**; look at the layers above it.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. **NO Modal fire** (MAIN owns dispatch + single-flight). The local SCORER
  LANE belongs to MAIN — emit typed fire orders; an honest partial plus a fire order is the CORRECT
  outcome, never a failure.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD. Bulky receipts → `/Volumes/APDataStore/pact/ddm_dcf1_duplicate_carry_factorization/`.
  **Both SSD tiers are at 100%** — write small, and if you cannot write, emit a blocker and keep the
  bytes. Local disk is a destination ONLY by explicit operator opt-in, which you do not have.
- **Reproduce a control from the shipped body before trusting any derived number.** Every arm in this
  lineage does; it is why they cost seconds and why two arithmetic errors got caught this window.

## ⚠ MANDATORY PRE-CHECK — RUN THIS BEFORE MEASURING ANYTHING

**Twice today a charter of mine rested on a premise a landed memo had already settled** (w96b had
already fired and closed its family; a storage ask had been withdrawn six hours before I made it).
The cure is at the SPAWN SITE, so it is here and it is binding:

For each of the four verbs, **search the corpus for a prior verdict on that exact surface first**,
and report what you found before you measure. Known priors you MUST reconcile against:

- **`mz2` (#1060)** already tree-shook the SEMANTIC TENSORS: **all 38/38 receiver-required, 0
  derive-at-decode.** That surface is DONE — on the e480b/hv1 lineage. State explicitly whether it
  transfers to lb1 or must be re-run, and why. Do NOT silently re-measure it.
- **`rr9` (#1244)** — the reorder axis is ARCHITECTURALLY FUSED to the trained HPAC group index;
  reorder is EXACTLY 0 B by order-invariance. Permutation-class "factorizations" are dead BEFORE
  measurement. Cite and skip.
- **`#1200`** — dx2's addressing is ALREADY FREE. Flattening must look above the address layer.
- **`#1124`** — the section-coding axis is CLOSED, and the carrier "rank/refit" rung was aimed at
  the WRONG OBJECT. Read that before you propose a carrier factorization.
- **`jt23`/`oc2` (#1283/#1326)** — coder axis CLOSED at 0 B; decode-derived conditioning DRAINED
  (2 B of a 2,162 B ceiling).
- **`ds1` (#1246)** — only 30,856 B of the archive are D-coupled AT ALL (the renderer block).

If a verb's surface is already fully settled, **say so and spend the time on the ones that aren't.**
A charter row closed by recall at $0 is a better outcome than a re-measurement.

## OPTIMAL FORM

- Family exemplar: `ddm_ar1b` — the **reference** form for this class: a decomposition with **ZERO
  REMAINDER** and every component bound to a receipt. Match that bar: your census must SUM to the
  archive, and you must say so with the arithmetic shown.
- SCOPE reductions declared per row (n<600 for exploration is legal). MECHANISM reductions FORBIDDEN
  — a factorization priced with a toy coder produces no verdict.
- **PRIOR-LAW PREDICTION (falsifiable).** The sharp-optimum law (#1214, five arms) predicts the
  HPAC optimum is sharp in every direction, so *lossy* moves lose. But **duplicate carry and inert
  bytes are LOSSLESS** — they are exactly the class a co-fitted optimum cannot see, and `fcd1`'s
  B/H/W win-win already proved "sharp" is not "sharp everywhere." **I PREDICT ≥1 duplicate-carry or
  inert-byte finding worth ≥1,000 B, most likely in the renderer × pose-carrier pair.**
  **FALSIFIER:** the census sums to the archive with every byte consumed, non-duplicated, and
  irreducible — i.e. lb1 is genuinely minimal in its own representation. **That is a real and
  valuable finding**; count it plainly and say the body is deduplication-clean.

## DELIVERABLE

`.omx/research/ddm_dcf1_duplicate_carry_factorization_20260831.md` — the pre-check results FIRST
(what recall already settled); the zero-remainder census; the carrier-vs-renderer duplicate-carry
answer with its receipt; the consumed/inert table from #417's bijection; factorization and
flattening rows priced with real coders; the denominator line (candidates enumerated / measured /
closed-by-recall / ABSENT); typed fire orders for anything scorer-gated. Commit via the serializer.
End with the own-vehicle frontier line:

`[contest-CUDA T4 n600] own-vehicle frontier: LB1 — S=0.14803010583079396, archive=180,083 B, d_seg=0.00020139, d_pose=6.37e-6, SHA-256=5b856e667961dd9ab68ddd7166384662bfb5912fabc8c9270098ea63a8ad28c9.`

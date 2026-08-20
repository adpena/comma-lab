# ddm_sr1 — SEG research: implicit edge conditioning for the semantic-token vehicle (js1 stage-1 input)

## Mission (operator 2026-08-11 "Research and believe" × the corrected waterfall)

SEG is now LOAD-BEARING: sub-0.15 from 0.161955 needs −0.011955; pose→~0 buys at most −0.0083;
the lossless rate axis is CLOSED (lp135 ALREADY-BANKED close, ed685a5dab) — so seg must supply
≥~−0.004. fd135 (commit 60ec8c21b0) measured the fork: EXPLICIT seg overlays are DEAD on this
vehicle (converges with #941/lc1: per-edge labels net-harmful −12,884 across all 32 pairs), but
IMPLICIT conditioning is OPEN. Your job: full research + corpus synthesis that converts
"implicit conditioning OPEN" into a RANKED MECHANISM TABLE with named js1-stage-1 consumers and
$0 scorer-free pre-probes — landed BEFORE js1 spawns so js1 consumes it.

RECALL FIRST (never working memory alone): `tools/corpus_query.py` over {implicit conditioning,
per-edge, PE3, edge-conditioned, context model, partition entropy, grammar-v2, Lane-over-Road}
+ read: fd135 memo · pk1/lc1 receipts (per-edge partition byte-positive 74,408 B but labels
harmful) · sn1 SegNet telemetry asymmetry · #906/sg2 (61.25% control; seg term UNDECOMPOSABLE
from retained data) · js1 charter `.omx/research/charters/ddm_js1_global_joint_solve_charter_20260810.md`.

## Ordered work

1. **RESEARCH SWEEP (online + OSS, full authority per the everything-off-the-shelf grant):**
   conditional/context entropy models for discrete token fields (checkerboard, ELIC-style
   spatial context, channel-conditional), edge/boundary-conditioned priors, mask-free implicit
   conditioning (the decoder derives context from what it already decoded — 0 counted bytes),
   segmentation-map compression SOTA, learned AR priors over partitions (PR130's HPAC IS one —
   what does the literature add beyond it?). For each: what it would do to OUR semantic-token
   wire, measured-or-derived, never vibes.
2. **CORPUS CROSSWALK:** our own measured per-edge/conditioning facts vs each candidate —
   which negatives were EXPLICIT-family (dead) vs genuinely implicit (open)? Re-scope per the
   verdict ladder; a dead explicit overlay does not kill an implicit sibling.
3. **$0 PRE-PROBES (scorer-free, run them):** e.g. conditional entropy of the retained token
   corpus given decoded-neighborhood/edge context vs the shipped HPAC's achieved rate — real
   coders on retained payloads (DT1/rc64p stores), NO scorer. Each probe carries a falsifier.
4. **DELIVERABLE:** ranked table {mechanism · route (Δbytes at equal seg | Δseg at equal bytes)
   · evidence class · $0 probe result or design · js1-stage-1 consumer hook}, in a durable memo
   `.omx/research/ddm_sr1_implicit_edge_conditioning_20260811.md` w/ NEXT_IF_RESUMED.

## Boundaries

Scorer-FREE (ps135b owns the scorer lane; entropy/coder measurements on retained payloads only).
No Modal. Serializer commits w/ post-edit --expected-content-sha256, [no-triality]
[p0-ledger-ok], --no-co-author. Payload law: any generated stream retained w/ sha256 to either
SSD tier (both granted). Public-PR intake clones READ-ONLY.

## OPTIMAL FORM

Reference = the live vehicle at full form, pinned: cp135 composed archive sha
6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6 (186,252 B) · lc2 archive sha
f154f0abb76980a30715282cf330d611cac7ebce3379c5f8093830dc273e1a45 (187,226 B) · fd135 memo
commit 60ec8c21b0 · lp135 close ed685a5dab. SCOPE = full n600 retained corpora; no toy token
subsets for any probe cited as evidence. PRIOR-LAW PREDICTION (derived fresh): PR130's HPAC
already captures most spatial context (lp135 measured its coder within 6-9 B of ANS at settled
state), so the OPEN win is cross-STREAM conditioning (tokens ↔ decoded partition edges ↔ pose
field), predicted worth ≥1,000 B OR ≥5% seg-term equivalent via conditioning the JOINT solve;
FALSIFIER: if conditional entropy given edge context is within 1% of the shipped marginal-HPAC
rate on the retained corpus, the implicit-conditioning RATE route closes honestly and only the
js1 joint-solve DISTORTION route remains — say so plainly.

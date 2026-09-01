# ddm_dds1_decoder_derivable_born_stats — derive-or-close: can born/generator statistics be DECODER-DERIVABLE from already-counted X/M state, escaping the 47,603 B packet charge? (task #1374 candidate gate; memo ddm_xov1_crossover_pass_20260901.md §LIVE-HYPOTHESES row 3)

## MANDATE

The crossover pass folded 3 candidate families into #1374 SCMDL. Candidates 1 (born
expert/context) and 2 (generator-conditioned peel chain) BOTH require the decoder to know the
GF1-generated class per site. Transmitting the generator packet costs 47,603 B counted →
the remaining G+M allowance is 37,306 B vs the current 126,926 B pool → the expert must
replace 89,620 B (70.6%) — near-certainly dead (mi1 law: a paid conditioning model misses
break-even 47.4×, memo ddm_mi1_indicator_model_axis_20260824.md). xov1's LIVE-HYPOTHESES row
3 names the ONLY escape: *"A joint SCMDL representation may exploit born statistics without
transmitting the full generator packet if an already-counted X/M state makes those statistics
decoder-derivable... no such derivation was found in the searched receiver code."* xov1
SEARCHED; it never DERIVED. This arm runs the dedicated derivation — a positive result
transforms candidates 1+2; a closure kills them cleanly BEFORE any rxc1 instrument time is
spent (gen-3 is LIVE on gate 1 concurrently; do not touch its store).

## SCOPE

1. **Formalize the demand.** What per-site quantities do candidates 1+2 actually consume?
   From the xov1 memo: `(generated_class, boundary_distance, agreement)` per site (candidate
   1) and the same as chain-rule conditioners (candidate 2). Write the exact type signature.
2. **Enumerate the already-counted decoder state** at each decode position under the causal
   SCMDL order: the partially-decoded exact field X (prefix), the full HPAC model M
   (13,515 B, counted), mixer contexts, the CPR1/dxi sections, and every DETERMINISTIC
   function of these (rule-118: receiver CODE is free — a generic algorithm computing
   statistics from already-decoded content costs 0 B). Pin each with path+sha.
3. **THE DERIVATION (closed-form-first, atlas ddm_cfa1):** is there a deterministic, causal,
   video-INDEPENDENT algorithm computing a useful surrogate of the GF1 class/boundary-distance
   from the decoded prefix alone? Candidate routes to derive or refute AT SOURCE:
   (a) causal boundary-distance transform of the decoded prefix (distance-to-last-boundary is
   prefix-computable) vs GF1's boundary distance — measure agreement on the retained fields;
   (b) a causal re-fit: run the GF1 fitting procedure ITSELF on the decoded prefix (the
   fitter is generic code = free; the question is whether prefix-fitting reproduces the
   packet's parameters — gf1's target-independence 0.37%, #1350, cuts BOTH ways: it suggests
   the fit is stable, but also that it carries little video-specific information);
   (c) M-derived surrogates: the HPAC model's own context statistics as a stand-in for
   "agreement" (does the mixer already SEE what the born expert would add? — the jt22/#1269
   joint-re-encode law governs);
   (d) any receiver-state quantity xov1's search missed (their search receipts are in the
   memo's RECALL EVIDENCE; do not repeat it — go deriving, not grepping).
4. **MEASURE the information overlap ($0, retained payloads):** for the best derivable
   surrogate, compute on the retained exact field + GF1 field (pins in the xov1 memo §Parent
   custody): conditional-entropy-style overlap — how much of the GF1 class's predictive value
   for X's residual surprise does the surrogate capture? Real coders where a byte number is
   claimed (no entropy-only verdicts, #1204/cm1); a screening estimate is legal if labeled
   SCREEN and never cited as bytes.
5. **Typed verdict:** DERIVED (algorithm + measured overlap + what it changes for candidates
   1+2, handed to gen-3's consumer) · CLOSED (no causal surrogate captures meaningful GF1
   value — candidates 1+2 die at the packet charge; state it plainly) · PARTIAL (a surrogate
   captures a measured fraction — recompute the replacement bar it implies).

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal. NO scorer runs. $0 arm.
- OWNERSHIP: gen-3 (LIVE) owns `experiments/ddm_rxc1_restartable_exact_coder.py` +
  `/Volumes/APDataStore/pact/ddm_jc1/restartable_exact_coder/` — READ-ONLY at most, prefer
  no touch. xov1 retained store READ-ONLY. Build nothing parallel to the SCMDL instrument —
  this arm derives + measures overlap; joint pricing belongs to gen-3's successor.
- ALWAYS KEEP THE PAYLOAD: persist every computed surrogate field + overlap receipt to
  `/Volumes/VertigoDataTier/pact/ddm_dds1/` (NOT AP tier-drain paths).
- Serializer commits w/ post-edit sha; bundle-fallback on .git/objects denial (#1293).
- DETACHED >30-min compute via the canonical launcher (script paths avoid claude/codex
  tokens); the overlap measurement over 117,964,800 sites likely needs it.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- mi1 (#1266): a PAID conditioning model misses break-even 47.4×; the conditioning target is
  only 2,162 B TOTAL on its measured object — any surrogate's value ceiling must be priced
  against what conditioning can possibly buy, not assumed.
- #1199: cross-parent agreement is a near-useless d_seg predictor (exponent 16.7) — but this
  arm's target is RATE (coding surprise), not d_seg; do not conflate the two (state this).
- jt22/#1269: context wins are joint-object specific; banked deltas are not additive without
  joint re-encode — overlap here is a SCREEN for gen-3's joint pricing, never a byte claim.
- gf1 target-independence 0.37% (#1350): the packet is nearly video-independent — if so, a
  prefix-refit may reproduce it nearly free; but then its video-specific value is also tiny.
  BOTH readings must be confronted, not one chosen.
- #1259/hc1: the token stream is 97.80% one binary question — any surrogate's overlap must
  be measured on the WRONG-half mass (~76,600 B) where the model needs help, not the easy mass.

## OPTIMAL FORM

- Family exemplar: the mi1 indicator-model pricing memo (real-coder discipline on the same
  question class) + the oc2 decode-derived conditioning charts arm (#1326 — the sibling that
  measured decoder-derivable charts on the live body; consume its receipts, do not rebuild).
  Provenance pins: xov1 memo commit 78f570edca (RESULT.json sha 59003d28…) · afr1 archive sha
  cbb8d928…d405bf25 · GF1 packet sha 87d79345… + decoded field sha 4026c4e2… (xov1 §custody).
- SCOPE reductions legal: overlap measured on a SEEDED RANDOM sample of pairs (n≥120, seed
  recorded — never a prefix, m88/bp2) before full-population confirmation. MECHANISM
  reductions FORBIDDEN: real retained fields, causal-order honesty (no future peeking), the
  wrong-half conditioning per hc1.
- **PRIOR-LAW PREDICTION (falsifiable):** gf1's 0.37% target-independence predicts a causal
  prefix-refit reproduces the generator packet's parameters to high fidelity at 0 counted
  bytes — i.e. route (b) DERIVES. FALSIFIER: prefix-refit parameters diverge materially or
  the reproduced statistics capture <10% of the wrong-half overlap — then the escape closes
  and candidates 1+2 die at the packet charge; write the closure plainly.

## DELIVERABLE

`.omx/research/ddm_dds1_decoder_derivable_verdict_20260901.md` — typed rows: the demand type
signature · the enumerated free decoder state (path+sha) · per-route derivation outcome
(a/b/c/d) · measured overlap on the wrong-half mass (seeded sample + scope label) · the
typed verdict {DERIVED/CLOSED/PARTIAL} + the recomputed replacement bar for candidates 1+2 ·
DEAD-ENDS + denominator. Commit via the serializer. End with the own-vehicle frontier line
(S 0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600], afr1 sha cbb8d928…d405bf25 —
UNMOVED unless a fire order lands).

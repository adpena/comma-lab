# ddm_mx1 — the weird one: a FREE decode-time online context model, trained on the decoded prefix inside the 540 s wall-clock slack, aimed at the 9 KB joint-oracle gap (design + $0 prototype on the shipped stream; charter, MAIN 2026-09-11; operator: be creative, divergent)

## The opening
Rule 118: deterministic generic compute in `inflate.py` is FREE; only video-derived content counts. Move 44's receiver decodes in ~1,000 s of
a 1,800 s budget (the T4 leg's ceiling is 1,260 s: ~540 s of unused compute; the CPU path is irrelevant, refused by design). The incumbent's
tail (119,909 B) is coded by RC64 + a sparse counted HPAC prior (13,515 B) + a FREE 23-family adaptive corrector + a counted 35-weight mixer
rider. tc1 measured that mixer at −589 B realized against a −9,011 B Miller–Madow JOINT oracle (6.09 %); ls1 (`.omx/research/ddm_ls1_lane_conditioned_surprise_atlas_on_the_shipped_field_20260911.md` sha b729faa146b62ea3…) measured
tc1's joint contexts at 8,218 B (MM) and receiver-visible Lane geometry at 17,534 B (MM), and flagged "full-resolution mixers remain unmeasured";
ls2 measured a linear full-resolution probability correction at 386 B. Nobody has built the thing those bounds are ABOUT: an online learner
with real capacity — a small neural context mixer (logistic mixing over many context models with learned weights, PAQ/cmix lineage; an
online-trained LSTM/transformer over the decoded symbol history, DeepZip/TRACE lineage) whose weights are learned DURING DECODE from
already-decoded symbols, zero counted bytes, deterministic (fixed seed, fp32 or integer arithmetic, identical on T4 and locally), fitting in
the 540 s slack. The information exists (ls1's ladder); the question is how much of it an online learner can realize for free.

## Deliverable ($0; no Modal; no scorer)
1. **Bound the slack first.** From the move-44 t4_direct leg and the receiver's per-stage timing (rlc/dwc receipts), state the wall-clock the
   tail decode uses today and the seconds available; derive the per-symbol compute budget at n600 (117,964,800 positions; the tail's coded
   symbols) — that budget decides the model size, not aspiration.
2. **Three online learners, each ORIGINAL to this vehicle, each with an OSS anchor, each priced by its per-symbol cost**: (a) logistic mixing of
   the existing corrector's 23 families + Lane-geometry contexts (ls1's receiver-visible set) with online-learned weights (PAQ8/cmix mixing,
   lpaq); (b) a small online-trained recurrent model over the decoded plane (DeepZip/TRACE lineage) with integer arithmetic; (c) a
   two-pass scheme — pass 1 decodes with the shipped model, pass 2 RE-CODES nothing but uses the fully decoded previous frame as a richer
   context for the next (still causal per frame). For each: bits it can remove on the shipped stream (measured by replaying the shipped
   symbol stream through the learner OFFLINE with the exact online update rule — this is exact for the code length because the decoder would
   see identical inputs), determinism plan (no floating nondeterminism across hosts: integer or fixed-point), wall-clock at n600, and the
   receiver diff it needs (a receiver change → first-measurement contract; that is fine, it is the path move 44 took).
3. **Prototype the best one on the SHIPPED stream** (`/Volumes/VertigoDataTier/pact/ddm_rlc5_cure_on_move43/` encode receipts; the shipped
   tail bytes; ls1's per-symbol atlas as the reference surprise): report realized bytes vs the 25,899 B demand and vs tc1's 9,011 B oracle,
   the wall-clock measured on this host with a projected T4 factor from the leg, and the exact determinism controls. If it realizes ≥ 3,000 B
   (100 bars) with a receiver diff that stays deterministic, produce the receiver delta and the re-encoded tail as candidate inputs and STOP
   for MAIN's first-measurement chain.
4. Memo `.omx/research/ddm_mx1_free_decode_time_online_context_mixing_20260911.md`: slack bound, three learners priced, prototype result,
   determinism controls, next charter. Serializer commits (two review passes per .py); rc 17 is NOT a stop. Checkpoint `ddm_mx1`; COMPLETE.

## Boundaries
No Modal, no scorer, no candidate archive, no receiver edits in sealed trees (copy); never edit `upstream/` or the PR tree; n600 streams
only; retain every stream/model state with sha; the learner's state must be reproducible from the seed and the decoded prefix alone (a
counted initialization is allowed but must be priced as counted bytes).

## OPTIMAL FORM
- Reference form: the shipping receiver's corrector + mixer (`runtime/free_corrector.py`, `fx1_logistic_mixer_corrector.py`,
  `rc3_shared_mixer.py`, `sm1_semantic_mixer.py`) as the instrument to extend; PAQ8/cmix/lpaq and DeepZip/TRACE as the OSS mechanism
  anchors; code length measured by exact replay. A reduced pair subset is SCOPE (declared); a proxy coder is MECHANISM.
- Provenance pins (sha256 prefixes): ls1 memo b729faa146b62ea3…; tc1 memo (record sha); ls2 memo (record sha); move-44 packet memo f7638e1e171e0d19…; pointer commit
  99625f32f / archive 04758c0dfb8d94ebe801602aac96d93a4f260aad24f58b6b9cdbbe2270ad460e.

## Prior negatives accounted (operator 2026-08-15)
- tc1 (mixing ceiling): its mixer was tiny and counted; yours is free and online — say what changes the ceiling, and price the gap to the joint
  oracle honestly (UNION ≠ SUM 3.705×; −log2p direction-dependent, average ≠ marginal).
- tc4 timeout (1,800 s on T4): the receiver change needs a MEASURED decode wall-clock; bound the slack before designing.
- eb2: receiver-visible context only — the decoded plane and the carrier; no partition, no pose.
- bd1 (temporal saturated on the label field): learner (c) must beat bd1's measured 2.8–5.7 % or be closed; say which.
- ls2 (386 B): a linear correction is closed; capacity must come from nonlinearity/online adaptation, and its cost is wall-clock, not bytes.

Final message: slack bound, three learners (one line each with realized bytes and wall-clock), the prototype's realized bytes vs 25,899 /
9,011, determinism controls, the next charter, and the frontier line `composition S 0.1372449041713402 @ 180,406 B [contest-CUDA T4 n600] (move 44)`.

<!-- # FORMALIZATION_PENDING: design+prototype charter; the realized-bytes law lands in the equations leg with the exact row -->

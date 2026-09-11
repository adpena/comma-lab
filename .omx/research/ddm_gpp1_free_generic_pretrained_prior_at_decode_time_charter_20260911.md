# ddm_gpp1 — the weirder one: a FREE generic pretrained prior at decode time — public, non-video-derived weights run on the receiver's OWN rendered RGB to produce Lane/partition contexts the token plane cannot (design + $0 conditional-surprise measurement; charter, MAIN 2026-09-11; operator: be creative, divergent)

## The opening (read rule 118 and the dependency policy in CLAUDE.md before anything)
The rate term charges `archive.zip` bytes only; `inflate.py` and its dependencies are FREE if generic (not video-derived) and installable
within the decode budget (dependency policy, operator 2026-07-24: vendoring/importing OSS is fine; PR130 precedent). eb2 measured that the
receiver holds no partition and no pose — it holds the decoded token plane and the carrier. BUT the receiver also RENDERS RGB frames (that is
its output). A generic, permissively licensed, publicly pretrained network (lane detection, road/drivable segmentation, edge/boundary
detection — NOT the contest scorers, whose weights may never ship or be smuggled) run on the receiver's own rendered previous frame (or on
the current frame's non-Lane render) yields a Lane/boundary prior that is not video-derived and costs zero counted bytes. ls1
(`.omx/research/ddm_ls1_lane_conditioned_surprise_atlas_on_the_shipped_field_20260911.md` sha b729faa146b62ea3…) measured the receiver-visible Lane-geometry bound at 17,534 B from token-plane contexts alone; a generic segmenter's
belief is a different, richer context. The demand is 25,899 B (21.6 % of the 119,909 B tail); Lane is 33.5 % of surprise.

## Deliverable ($0; no Modal; no scorer runs; no candidate archive)
1. **Legality + logistics first, written down**: (a) rule 118 reading with the exact README lines: generic public weights = external tool
   (free) vs video-derived (counted); state the boundary and the residual risk plainly (the operator decides at publish); (b) candidate
   models with licenses and sizes (e.g., a small public lane-detection net, a road-scene semantic segmenter NOT trained on this video, a
   generic edge detector — Kornia/torchvision/timm-hosted weights; comma10k-trained models are ALLOWED as generic public priors only if
   they are public and not the scorer); (c) bootstrap plan inside the decode budget (bare-venv proof, pinned hashes, deterministic
   inference: fixed seed, fp32 or integer, identical on T4 and locally — the e4 brotli / r5 precedent); (d) wall-clock estimate at n600 on
   the rendered 384×512 frames against the ~540 s slack (the T4 leg is 1,260 s of 1,800 s).
2. **Measure the context's value with ls1's instrument** (`experiments/ddm_ls1_shipped_surprise.py` / `ddm_ls1_oracle_atlas.py`,
   landed): run the chosen model on the SHIPPED renders (move-44 tree, read-only copy; render the frames the receiver would have at each
   step — causal: only frames already decoded), derive per-cell Lane/boundary beliefs, and measure the Miller–Madow conditional saving of
   the shipped tail symbols given (token-plane contexts + the generic belief) vs given token-plane contexts alone, per class × geometry,
   n600. Report bytes removable vs 25,899 B and vs ls1's 17,534 B receiver-visible bound. This is an UPPER BOUND; say so.
3. If the bound clears ≥ 8,000 B: prototype the context in the shipped coder's mixing path (copy of `runtime/`), replay the shipped stream
   exactly, report realized bytes, wall-clock, determinism controls, and the receiver diff — then STOP for MAIN's first-measurement chain.
   If it does not clear, close the family at formulation scope with the numbers.
4. Memo `.omx/research/ddm_gpp1_free_generic_pretrained_prior_at_decode_time_20260911.md`: legality reading, model table, the measured
   bound, the prototype (if reached), the next charter. Serializer commits (two review passes per .py); rc 17 is NOT a stop.
   Checkpoint `ddm_gpp1`; COMPLETE at the end.

## Boundaries
NEVER use SegNet/PoseNet weights or any derivative of the scorers (strict-scorer rule); no training on the contest video; no Modal; no
scorer runs; read-only on `/Volumes/...` (copy); never edit `upstream/`, the PR tree, sealed trees; n600; retain every belief map with
sha; do not touch obx2/pc3/ntb1/mxo1/rb1 directories.

## OPTIMAL FORM
- Reference form: ls1's exact instrument (surprise reconciled within 0.0006 %) as the ONLY measurement path; a real public model, not a
  hand-built proxy (a proxy = MECHANISM, toy-bracketed); a reduced pair subset = SCOPE (declared, no verdict).
- Provenance pins (sha256 prefixes): ls1 memo b729faa146b62ea3…; move-44 packet memo f7638e1e171e0d19…; the chosen model's weight file sha + license text sha
  (record); pointer commit 99625f32f / archive 04758c0dfb8d94ebe801602aac96d93a4f260aad24f58b6b9cdbbe2270ad460e; shipped field subset6.u8
  (a92e7d90…, from the encode receipt).

## Prior negatives accounted (operator 2026-08-15)
- eb2: receiver-visible only — the belief must be computed from frames the receiver has ALREADY decoded (causal); say which.
- rule 118 / move 41 retraction: a video-selected constant in free code is content; a PUBLIC generic model is not — but a model fine-tuned on
  this video would be; document the provenance of the weights.
- r5/e4: prove the bootstrap in a bare venv; never assume host site-packages.
- tc4: any receiver change needs a measured decode wall-clock; bound it before designing.
- UNION ≠ SUM / average ≠ marginal: price the joint conditional saving, never a sum.

Final message: the legality reading (one paragraph), the model table, the measured bound vs 25,899 / 17,534 B, the prototype result (if
reached) with wall-clock and determinism, the next charter, and the frontier line `composition S 0.1372449041713402 @ 180,406 B [contest-CUDA T4 n600] (move 44)`.

<!-- # FORMALIZATION_PENDING: design+measurement charter; the bound lands in the equations leg with the first realized row -->

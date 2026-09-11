# ddm_ntb1 — the 60,497 non-tail bytes: inventory, price, and re-solve every counted byte that is not the token tail (charter, MAIN 2026-09-11; operator full-authority GO)

## Why
Move 44's archive is 180,406 B = token tail 119,909 B + NON-TAIL 60,497 B (ZIP overhead + RX1M header + HPAC model + semantic renderer +
carrier; ls1 `.omx/research/ddm_ls1_lane_conditioned_surprise_atlas_on_the_shipped_field_20260911.md` sha b729faa146b62ea3…). Every construction this week attacked the tail; the non-tail has not been priced since bz2d (HPAC model
13,515 B; RX1M header 14 B). The bar is −2e-5 S = 30 B. A renderer weight-precision cut, model-table pruning, or ZIP-member/header hygiene
that removes 500 B at unchanged distortion is −3.3e-4 S (16 bars). Price it all; take what pays.

## Deliverable
1. **Byte census of the non-tail**, member by member (ZIP local headers/central directory/comment; RX1M header; each HPAC model section;
   the semantic renderer's weights by layer and precision; the carrier; any padding), from the SHIPPED archive (read-only copy), with the
   parser that ships (`runtime/residual_archive.py`, `cpr1/integer_model_io.py`, `runtime/entropy/renderer_weight_codec.py`).
   Reconcile to 60,497 B exactly.
2. **Price each lever at $0 and re-solve where the receiver stays unchanged**: (a) ZIP container hygiene (member names, extra fields,
   compression method per member, ordering — deterministic, receiver-agnostic; rp1's lottery law: real twin encodes, ship the smaller);
   (b) HPAC model sections: quantization/pruning of tables with the tail RE-ENCODED (the model change changes the coded tail — price the
   joint bytes, never the table alone); (c) semantic renderer weights: per-layer precision cut (int4 → int3 on low-sensitivity layers)
   with the SEG RE-SOLVE through the real render path and the frozen scorer at n600 (renderer seg↔pose coupling law: measure both);
   (d) carrier repack (pc3 owns the carrier's CAPACITY curve; you own only lossless repack).
   For each: Δbytes, Δd_seg, Δd_pose (n600, frozen scorer [macOS-CPU advisory]), net ΔS vs the bar; keep every payload.
3. **Compose the winners** (composition law: seg sub-additive by pair overlap; object-change): one candidate archive = move 44's bytes with
   the paying non-tail changes and the tail re-encoded where a model changed; twin encodes; full cold n600 public parse-back; manifest
   from outside the tree; census; smokes. If the receiver code is UNCHANGED → normal seal (`make_candidate_seal.py`, inherits move 44's
   t4_direct leg); if any receiver file changed → STOP and hand MAIN the first-measurement intent inputs instead (do not fire).
4. Memo `.omx/research/ddm_ntb1_non_tail_bytes_price_and_resolve_20260911.md`: census, lever table, composed candidate, seal path.
   Serializer commits (two review passes per .py; `[no-triality] [p0-ledger-ok]`); rc 17 is NOT a stop. Checkpoint as `ddm_ntb1`; mark COMPLETE.

## Boundaries
No Modal (MAIN fires); never edit `upstream/`, the PR tree, sealed trees; read-only on `/Volumes/...` sources (copy); n600 only;
payloads retained with sha; heavy steps through the launcher; do not touch pc3/obx2/mx1 directories. Rule 118: no video-selected
constant may move into free code.

## OPTIMAL FORM
- Reference form: the shipping parsers and coders (never re-implementations); sj1's seg re-solve chain for renderer changes; twin encodes for
  every price. A reduced lever set is SCOPE; a proxy scorer/coder is MECHANISM.
- Provenance pins (sha256 prefixes): ls1 memo b729faa146b62ea3…; move-44 packet memo f7638e1e171e0d19…; bz2d memo (`ddm_bz2d_distortion_verdict_20260830.md`, record
  sha); pointer commit 99625f32f / archive 04758c0dfb8d94ebe801602aac96d93a4f260aad24f58b6b9cdbbe2270ad460e.

## Prior negatives accounted (operator 2026-08-15)
- rw1: renderer int4 code-grid REPAIRS break at n600 — you are cutting RATE with a re-solve, not repairing; measure d_seg at n600, never n240.
- rf1/ft1: seg-only renderer changes are unpayable — measure pose too.
- rp1 r2: flag vs constant disagree silently — bind base archive/tree by sha in every receipt.
- jt23 (coder axis closed at 0 B): do not re-race the tail coder; the tail re-encode here is a CONSEQUENCE of a model change, priced jointly.
- container-break lottery: twin encodes, ship the smaller, never search.

Final message: the census, the lever table, the composed candidate's bytes + sha + projected S, the seal path (or the first-measurement STOP),
and the frontier line `composition S 0.1372449041713402 @ 180,406 B [contest-CUDA T4 n600] (move 44)` unchanged.

<!-- # FORMALIZATION_PENDING: producer charter; the lever table becomes an equations-leg law with the exact row -->

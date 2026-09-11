# ddm_ntb2 — continue ddm_ntb1: the two LOSSY non-tail levers with the real re-solve — structural HPAC pruning with the RLC1-aware joint tail re-encode, and the remaining 4-bit renderer layers with the seg/pose re-solve; compose; seal (charter, MAIN 2026-09-11; operator full-authority GO; SCORER SLOT ASSIGNED to this arm)

ntb1 (`.omx/research/ddm_ntb1_non_tail_bytes_price_and_resolve_20260911.md`, sha 58b1163d5a46e546…, landed) measured the non-tail census (ZIP 100 / RX1M 14 / HPAC 11,911 / renderer 29,862 / carrier 18,610 = 60,497 B)
and closed every LOSSLESS lever at 0 B (13 formats, 26 twin encodes). It stopped before the two lossy levers because its contract assigned no
scorer slot. This charter assigns it. Everything in ntb1's charter (`.omx/research/ddm_ntb1_non_tail_bytes_price_and_resolve_charter_20260911.md`)
binds unchanged; do not repeat the lossless work.

## Deliverable
1. **HPAC structural pruning, priced JOINTLY with the tail.** ntb1 found that the unchanged JG2 encoder omits move 44's counted RLC1 mixer path,
   so the tail must be re-encoded with an RLC1-aware encoder (the move-44 encode receipts under `/Volumes/VertigoDataTier/pact/ddm_rlc5_cure_on_move43/encode/`
   name the path; `experiments/ddm_rlc4_rebase.py` / `ddm_rlc5_run.py` are the landed encoders — reuse, do not rewrite). Reproduce the
   shipped tail byte-identically FIRST (the falsifier of the encoder), then sweep structural prunings of the HPAC sections (drop/merge the
   lowest-weight coordinates; coarser table quantization) and price each as (HPAC bytes + re-encoded tail bytes) by twin encodes; the receiver
   must parse the pruned model unchanged (a format change is a receiver change → STOP and say so).
2. **Renderer precision cuts with the real re-solve.** The remaining 4-bit layers (ntb1's table; frame-embedding/block-0 FiLM are already 3-bit):
   cut to 3-bit per layer, RE-SOLVE seg through the real render path (the sj1 seg re-solve chain) and measure d_seg AND d_pose at n600 with
   the frozen CPU scorer [macOS-CPU advisory] (you hold the scorer slot); price the archive by twin encodes; report ΔS per layer and the
   best composition (composition law: measure the union, never sum).
3. **Compose the winners** into one candidate on move 44's bytes: twin encodes; full cold n600 public parse-back; raw comparison vs move 44's
   retained raw (renderer changes will change pixels — enumerate exactly which frames/regions; the token planes must be identical unless a
   model change re-coded them losslessly); manifest from outside the tree; literal census; smokes. Receiver UNCHANGED → normal seal
   (`make_candidate_seal.py`, t4_direct inherited from move 44); any receiver file changed → first-measurement intent inputs and STOP.
4. Memo `.omx/research/ddm_ntb2_non_tail_lossy_levers_with_resolve_20260911.md`: encoder reproduction proof, HPAC pruning table, renderer
   table (d_seg, d_pose, ΔS per layer), the composition, the seal path or the STOP. Serializer commits (two review passes per .py); rc 17 is
   NOT a stop. Checkpoint `ddm_ntb2`; COMPLETE at the end.

## Boundaries
No Modal (MAIN fires); never edit `upstream/`, the PR tree, sealed trees; read-only sources (copy); n600 only; retain every payload with sha;
heavy steps through the launcher; do not touch obx2/pc3/mxo1/gpp1/rbf1 directories. Rule 118: nothing fitted moves into free code.

## OPTIMAL FORM
- Reference form: the shipped parsers/coders and the landed RLC1-aware encoders; sj1's seg re-solve chain for the renderer; twin encodes for
  every price; the frozen scorer at n600. A reduced lever set is SCOPE; a proxy scorer/coder is MECHANISM.
- Provenance pins (sha256 prefixes): ntb1 memo 58b1163d5a46e546…; ntb1 charter (record sha); move-44 encode receipt (record sha); pointer commit 99625f32f /
  archive 04758c0dfb8d94ebe801602aac96d93a4f260aad24f58b6b9cdbbe2270ad460e.

## Prior negatives accounted (operator 2026-08-15)
- ntb1: lossless = 0 B (closed); prior HPAC multiplier/seed negatives did not test structural coordinates — say which coordinates you test.
- rw1: renderer grid repairs break at n600 — you cut precision with a re-solve, measured at n600 only.
- rf1/ft1: seg-only renderer changes are unpayable — measure pose for every layer cut.
- rp1 r2: bind base archive/tree by sha in every receipt; container-break lottery: twin encodes, ship the smaller.

Final message: the encoder reproduction, both lever tables, the composed candidate's bytes + sha + projected S, the seal path (or STOP), and the
frontier line `composition S 0.1372449041713402 @ 180,406 B [contest-CUDA T4 n600] (move 44)` unchanged.

<!-- # FORMALIZATION_PENDING: producer charter; the lever tables become an equations-leg law with the exact row -->

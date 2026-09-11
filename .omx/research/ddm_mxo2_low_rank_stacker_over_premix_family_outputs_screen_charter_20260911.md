# ddm_mxo2 — the $0 exact integer screen mxo1 named: a low-rank NONLINEAR stacker over the shipping corrector's 23 pre-mix family outputs (charter, MAIN 2026-09-11; operator full-authority GO; fire trigger from mxo1's memo)

## The opening mxo1 left
mxo1 (`.omx/research/ddm_mxo1_free_decode_time_online_context_mixing_20260911.md`, sha 60b5a34bb6146af8…) closed generic online learners (best 368 B) and measured the strict T4 receiver slack at **27.581 s = 233.8 ns per
coded symbol** at n600 — the binding budget. Its live hypothesis: move 44's shipped corrector computes 23 family predictions per symbol and
collapses them into ONE probability through a fixed (counted 35-weight) mixer; information may be lost at that collapse. A low-rank nonlinear
stacker over the 23 pre-mix outputs (plus the mixer's own output), learned ONLINE from the decoded prefix (zero counted bytes; integer
arithmetic; deterministic), might recover part of tc1's 9,011 B joint oracle (ls1 `.omx/research/ddm_ls1_lane_conditioned_surprise_atlas_on_the_shipped_field_20260911.md` sha b729faa146b62ea3…: joint contexts 8,218 B MM). mxo1's own
fire trigger for a receiver integration: **an exact n600 integer-row screen over the shipping 23-family pre-mix outputs predicts ≥ 3,000 B of
savings AND a native dry-run costs ≤ 200 ns/symbol.** This charter runs exactly that screen; nothing else fires.

## Deliverable ($0; no Modal; no scorer; no candidate archive)
1. **Extract the pre-mix surface exactly.** Instrument a COPY of the shipped receiver (move-44 tree, read-only source) to emit, per coded
   symbol in the shipped 190-group order (mxo1: raster-order replay reproduces length but not the RC64 bytes — only 190-group order is
   valid), the 23 family predictions, the shipped mixer's output probability, and the decoded symbol. Reconcile the shipped probability's
   code length to the real stream bytes (ls1's 0.0006 % standard) — the falsifier of the instrument. Retain the surface on the SSD tier.
2. **The screen, exact and online.** Fit the stacker ONLINE (causal, integer/fixed-point updates, fixed seed) over the surface: rank r ∈
   {2, 4, 8} bilinear/low-rank + a small nonlinearity (e.g., logistic of a rank-r quadratic form; or a 2-layer integer MLP with ≤ 64 hidden
   units), initialized generically (a counted initialization must be priced as counted bytes). Report the exact code length under
   the stacker vs the shipped probability, per class × geometry (ls1's table), and the total bytes saved vs 3,000 B and vs 25,899 B.
   UNION ≠ SUM: the saving is measured on the single joint online run, never summed across families.
3. **The native dry-run cost.** Implement the stacker's per-symbol update+predict in a tight integer kernel (NumPy vectorized where causality
   allows, or a small C/Rust kernel under `runtime-rs`/`cuda` policy — generic code, free) and MEASURE ns/symbol on this host; project T4
   with the leg's factor mxo1 recorded; compare to 200 ns/symbol and to the 27.581 s slack.
4. **Verdict**: both trigger halves pass → produce the receiver delta + the re-encoded tail as first-measurement inputs and STOP for MAIN;
   either fails → close the family at formulation scope with the two numbers. Memo
   `.omx/research/ddm_mxo2_low_rank_stacker_over_premix_family_outputs_screen_20260911.md`. Serializer commits (two review passes per .py);
   rc 17 is NOT a stop. Checkpoint `ddm_mxo2`; COMPLETE at the end.

## Boundaries
No Modal, no scorer, no candidate archive; never edit `upstream/`, the PR tree, sealed trees; read-only sources (copy); n600 only; retain
every surface/model state with sha; determinism from seed + decoded prefix alone; do not touch obx2/pc3/ntb2/gpp1/rbf1 directories.

## OPTIMAL FORM
- Reference form: mxo1's exact replay instrument and its 190-group order; ls1's reconciliation standard; the shipped corrector
  (`runtime/free_corrector.py`, `fx1_logistic_mixer_corrector.py`, `rc3_shared_mixer.py`) as the surface to extend. A pair subset is SCOPE
  (declared, no verdict); a floating-point-only prototype is MECHANISM for the wall-clock half (integerize before pricing ns/symbol).
- Provenance pins (sha256 prefixes): mxo1 memo 60b5a34bb6146af8…; ls1 memo b729faa146b62ea3…; tc1 memo (record sha); pointer move 44 commit 99625f32f / archive
  04758c0dfb8d94ebe801602aac96d93a4f260aad24f58b6b9cdbbe2270ad460e; shipped field subset6.u8 (a92e7d90…).

## Prior negatives accounted (operator 2026-08-15)
- mxo1: generic PAQ/cmix re-derivation is closed (the corrector already adapts online); the 3-expert Lane residual stack saved 368 B and
  exceeded the slack — your object is the pre-mix collapse specifically, and wall-clock is priced first.
- ls2 (386 B): a LINEAR correction is closed; the stacker must be nonlinear or it is the same object.
- tc1 / m164 / m166: joint, marginal, direction-dependent pricing only.
- tc4: any receiver change needs a measured decode wall-clock; the 27.581 s slack is the budget.

Final message: the instrument reconciliation, the screen's bytes saved (per rank) vs 3,000 / 25,899 B, the ns/symbol vs 200, the verdict, and
the frontier line `composition S 0.1372449041713402 @ 180,406 B [contest-CUDA T4 n600] (move 44)`.

<!-- # FORMALIZATION_PENDING: screen charter; the realized-bytes law lands in the equations leg with the exact row -->

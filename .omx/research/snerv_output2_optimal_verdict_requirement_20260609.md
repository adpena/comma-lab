# SNeRV output2 — "is it not wired correctly?" → the optimal verdict requirement

UTC: 2026-06-09 · claude · operator clarification capture (no signal loss; V2-SNERV agent
is mid-flight on `snerv_official_tub_source_forward_replay.py` + sisters and cannot be
messaged). Operator 2026-06-09 verbatim: *"is output2 or is something not wired correctly
or something? we just want to do whatever is optimal give everything what it needs and no
more."* This file is the verdict-requirement the SNeRV DROP_OR_REIFY proof (#29) is checked
against on integration.

## The framing correction (grounded in the actual code)
1. **The contest scorer never sees output2.** `grep output2 upstream/` = 0 matches. The
   evaluator (`upstream/evaluate.py`) sees ONLY the `inflate.sh` RGB frames → SegNet/PoseNet.
   "Official" in our code = our faithful SNeRV_T reimplementation, NOT the contest scorer.
   So "optimal" for output2 = best RGB-fidelity-per-byte at the RECEIVER. output2 matters
   only insofar as it changes the reconstructed RGB.
2. **output2 is an INTERNAL temporal-fusion feature of the SNeRV_T forward.** In
   `snerv_official_tub_source_forward_replay.py`, the final RGB (`manual["frame_reconstruction"]`
   → `rgb_pair_float`) is computed by the forward REGARDLESS of `include_output2`. The
   `include_output2` flag only controls whether `output_2` (`output2_shuffled`) +
   `temporal_encoder_concat` are EXPOSED/STORED as separate archive tensors for the receiver.
3. **The receiver currently DROPS it.** The receiver fixture call site uses
   `include_output2=False` (line ~1772) and `receiver_frame_decode_consumes_output2` defaults
   `False`. Whether that is a BUG or OPTIMAL is the empirical question — not a guess.

## The decisive verdict (three branches; the proof MUST land exactly one, empirically)
Method: drop/flip output2 in the source state, re-run the ACTUAL receiver RGB decode path,
run SegNet+PoseNet, compute EXACT ΔS via `contest_score`. Then:

- **ELIDE (optimal "no more"; NOT a bug).** Receiver RGB is unchanged (≤ uint8 tie noise)
  without output2 ⇒ its contribution is already represented in the stored LF/HF/MFU/HFR/TUB
  basis + decoder weights. Drop the bytes. Verdict = `output2_boundary_closed` /
  `elide_payload`. `CandidateActionEvaluation`: storing output2 has `delta_bytes>0`,
  `delta_score_nonrate≈0` ⇒ `pays_rent=False` ⇒ REJECT (do not store).
- **DERIVE ("give it what it needs", the wiring fix; ~0 extra bytes).** Receiver RGB degrades
  without output2 BUT output2 is reconstructable from the stored MFU/HFR/TUB/LF/HF basis ⇒
  wire the receiver decode to DERIVE it (`derive_output2_from_mfu_hfr_tub_basis`). This is the
  "not wired correctly" case: upstream forward uses it, receiver should but doesn't. Fix =
  reify the derive path, then rent-test (derivation costs ~0 archive bytes).
- **STORE-RESIDUAL (rent-gated).** Receiver RGB degrades AND output2 is NOT reconstructable
  from the basis ⇒ store ONLY the minimal residual the basis cannot derive, admitted iff
  `S(with) < S(without)` by exact ΔS (`pays_rent=True`). "What it needs and no more" =
  the smallest residual that pays rent.

## The principle = the rent law (unification)
"Give everything what it needs and no more" IS the evaluator-action waterfilling law
(`tac.optimization.evaluator_action_waterfill.CandidateActionEvaluation`): output2 (or any
facet) enters the archive IFF `S(base + facet) < S(base)`. No facet enters because it
"parses + applies"; it enters because it lowers the exact contest score. This is the same
currency that extincted the HiNeRV sidecar bug.

## What the existing code already provides (do not rebuild)
- `snerv_source_forward_producer.build_snerv_output2_boundary_verdict` — a 7-branch typed
  classification (present/consumed/shape/value across official_torch / archive_parseback /
  numpy_receiver). It is STRUCTURAL. The proof must upgrade it with the EMPIRICAL
  receiver-RGB ΔS so the chosen branch is rent-justified, not just shape-justified.
- `snerv_official_tub_source_forward_replay` `include_output2` switch — the exact knob to
  run the drop/flip counterfactual.

## DO NOT
- Do not store output2 because "the upstream forward computes it" — the receiver/scorer is
  the authority, and bytes only enter by rent.
- Do not DROP output2 on a structural verdict alone — confirm the receiver RGB is unchanged
  (the ELIDE branch needs the empirical equality, else it's silently lossy).
- Do not call it a bug or optimal without the receiver-RGB ΔS measurement.

## LANDED VERDICT (2026-06-09, V2-SNERV, commit `06f3dc580`) — requirement → measured result
`build_snerv_official_tub_drop_or_reify_source_forward_proof(...)` in
`src/tac/analysis/snerv_official_tub_source_forward_replay.py` ran the real upstream
`SNeRV_T.forward` source graph (CPU-portable functional-Haar fixture), bit-flipped each TUB
source-state facet by the smallest representable step, and re-ran the real receiver RGB
primitive. Result:

- **`output_2` → DROP (ELIDE branch confirmed).** Source bytes change but the receiver frame
  is BIT-IDENTICAL (`receiver_frame_float_linf == 0.0`): the receiver frame-reconstruction
  has no `output_2` parameter ⇒ structurally not consumed ⇒ storing it is rent-negative
  (adds bytes, zero scorer change). **We are using output2 correctly** in the current
  receiver — eliding it is optimal "no more," NOT a bug. Blocker recorded:
  `snerv_official_tub_output2_source_state_not_consumed_by_receiver_frame_decode`.
- **`yl_norm` → REIFY_PENDING_SCORER (receiver-causal).** A 1-ULP flip propagates to the
  float receiver RGB (`linf=1.1e-16 > 0`); receiver consumes it, but sub-uint8 at the
  fixture ⇒ store only what survives the uint8/scorer boundary, pending a real scorer-ΔS at
  the real operating point. Emits a base-bound `CandidateActionEvaluation` on the REIFY path
  when a real `scorer_fn` is supplied (delta_bytes=0 ⇒ admission is pure distortion-ΔS).

18 NO-FAKE tests pass; `research_only=true` with a concrete pending blocker whenever no real
scorer is supplied (no fabricated scorer passes). The open REIFY EXPERIMENT (would a receiver
*redesigned* to consume output2 yield better scorer terms per byte?) is gated on (a) wiring
`receiver_frame_decode_consumes_output2=True` + (b) a real SegNet/PoseNet ΔS — measurable,
not a current bug. Sister memo: V2-SNERV agent report + 18-test file
`src/tac/tests/test_snerv_official_tub_drop_or_reify_source_forward_proof.py`.

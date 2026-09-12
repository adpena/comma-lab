# ddm_tmx1 — refit the tail coder's 35-weight mixer (tc1 + tc3's 5) on the CURRENT field and prior, output-lossless, receiver unchanged (charter, MAIN 2026-09-11; operator full-authority GO; Opus)

## Why (hpr1's staleness audit, MEASURED)
`.omx/research/ddm_hpr1_staleness_audit_20260911.md`: of the four counted sections only the HPAC prior was refit in the last three moves — and it
produced moves 46 and 47 (−245 B, −642 B). The tail's counted mixer state (`tc1_weights`, 60 B, sha 76f10171…; prefix 8ab2fe74…) is identical
across moves 45/46/47 and was fit to the MOVE-32 field; the field moved EIGHT times since (31, 32, 35, 38, 39, 40, 42, 43); cmp1 states in
its own words that it composed onto a changed field WITHOUT refitting them; tc3's +5 weights were fit to the move-40 field. 60 B of state
prices the 118,511 B RLC1 stream. Same genus as the −887 B refit: a shipped model fit to a state of the object that no longer exists. hpr1's
law (memory `cross_regime_constant_transfer_genus_finishing_stage_20260805`, instance 5): refit every model section after a field change.

## Pointer and objects (binding; re-derive nothing from memory)
Move 47: S 0.13654774984742127 @ 179,359 B [contest-CUDA T4 n600]; archive d1fab05d69f31c90ac55173fa87072949e5ea1e069a0b7614337089b7a2a0ce9;
promoted tree `/Volumes/VertigoDataTier/pact/ddm_hpr1/public/retrain_control/candidate_runtime` (SEALED — copy). A move-48 candidate is in flight
(hpr1's composition: refit prior + even rounding, 179,111 B, sha d830edd3…): price your refit on BOTH priors (move 47's and the composition's)
so MAIN can rebase without a re-run; the tail's input field is identical across 45/46/47/48. Token field a92e7d90… (decoded, shipped).

## Deliverable ($0; no Modal; no scorer; MAIN fires)
1. Reconstruct exactly what tc1/tc3's weights are (which contexts, which mixing, how they reach the coded row), from the receiver code and
   cmp1/tc1/tc3's memos; state the training objective and data they were fit to (field version, prior version) with receipts.
2. Refit them on the CURRENT field under the CURRENT prior with the shipped training path (or the same objective re-implemented and proven
   equal on the shipped weights: the live-loop control must reproduce the pointer's stream byte-identically before any refit is priced —
   ntb2's/hpr1's control harness is the reference; import, never rewrite). Encode all 600 frames with the REAL coder through the REAL receiver
   loop; exact bytes with twin encodes; ΔB = Δmodel + Δtail (model = the 60 B state, may change size only if the schema allows — if it does
   not, say so); decoded field byte-identical (output-lossless) — prove it. Held-out control as mxo3/hpr1 did (frame-parity folds, KT back-off)
   so in-sample fit cannot pass as a win; report in-sample AND held-out.
3. Timing: unchanged decode work (same symbols, same loop) — measure local ns/symbol against the shipped loop anyway and report; no wall-clock claim.
4. Fire rule: net ΔS < −2e-5 at exact bytes vs the pointer at the time of sealing → stage from the promoted tree, cold n600 parse-back (raw must
   equal 2b762eba…), manifest regenerated outside the tree, smokes with the pointer read live, receiver-unchanged proof (behavior digest 9f6e7168…),
   then the seal path MAIN names (pr19 may have landed transitive inheritance by then; else the first-measurement chain — its templates are
   `.omx/research/ddm_ntb2_20260911/v2/`); STOP; MAIN fires. Else the closure memo with the held-out number and which term ate it.
5. Memo `.omx/research/ddm_tmx1_refit_tail_mixer_tc1_on_current_field_20260911.md` (`# FORMALIZATION_PENDING:<rationale>` or cite
   `hpr1_counted_section_refit_debt_v1`); checkpoint as `ddm_tmx1`; serializer commits, two review passes per .py, no co-author trailer,
   `[no-triality] [p0-ledger-ok]`. Bulk under `/Volumes/VertigoDataTier/pact/ddm_tmx1/` (43 GiB free; never APDataStore; never lower a reserve).

## Boundaries
Never edit `upstream/`, the PR tree, sealed trees, or other arms' directories (ddm_hpr1 is LIVE — read/copy only; pr19, cons2 live); no receiver
change (a rung that needs one is out of scope: record it); no candidate claims (`[macOS-CPU advisory]`, score_claim=false).

## OPTIMAL FORM
- Reference form: hpr1's refit of the HPAC prior (real coder, real loop, byte-identical control, twin encodes, held-out) — the landing that
  paid −887 B; the only delta is WHICH section is refit (SCOPE), no mechanism reduction.
- Provenance pins (record shas): hpr1's audit memo; cmp1/tc1/tc3 memos; move 47 tree/archive; the token field a92e7d90…; mxo3's cross_bits.

## Prior negatives accounted (operator 2026-08-15)
- mxo3: recalibrating q on the OLD prior cost 183.5 B held-out; a second stacker finds nothing — you refit the EXISTING weights, you do not add mixing.
- ls1/ls2: the context set is closed at the measured level — do not add contexts.
- cl2/cl3: size is closed (+0.446 secant) — the state stays 60 B unless the schema already allows more.
- container-break lottery (sd 34.8 B): a net inside ±35 B is noise; say so.
- rp1 r2 / sj1 silent revert: bind base archive + tree by sha; prove the decoded field is byte-identical.

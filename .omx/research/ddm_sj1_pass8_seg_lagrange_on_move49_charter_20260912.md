# ddm_sj1 pass 8 — Lagrange seg subset on the move 49 field, FIRST-MEASUREMENT chain (charter, MAIN 2026-09-12; operator full-authority GO)

## Why
Pass 7 landed move 49 (S 0.13632299781031237 @ 179,153 B [contest-CUDA T4 n600]; d_seg 0.00010287; d_pose 4.55e-6; archive sha
73e41a6620bd4ea3aaf236eff9de46391857907527358e8eb40ded0925a1c214; tree `/Volumes/VertigoDataTier/pact/ddm_sj1_pass7/candidate/candidate_runtime`;
packet `.omx/research/ddm_sj1_t4_token_predistortion_pass7_20260912_pointer_move_49_20260912.md`). Its yield was 1.812 % of the residual, above the
family's 1 %/pass convergence rule, and Stop-rule C fired, so the family is NOT closed. Pass 7's own reach law: repairs consume the local slack
(0.410 cells/pair on the 161 repaired pairs vs 1.156 on neutral re-renders, n=2), and 154 of 155 stale-pair cells were pass-6 carryover — so
pre-register a SMALLER same-field yield here (~0.5–1.2 % of the 12,127-cell residual) and let the measurement decide; if the admitted subset does
not project net ΔS < −2e-5 on the RESOLVED pose, the family closes on this field at formulation scope with the curve.

## The contract difference from pass 7 (binding)
Move 49's timing leg is INHERITED (from move 48's measured leg) and pr19 forbids re-inheriting an inherited leg. Your candidate changes only
archive.zip + its two pins (receiver code byte-identical), but it CANNOT take the normal seal: it takes the FIRST-MEASUREMENT chain exactly as
ntb2 (move 46) and hpr1 (move 48) did — `tools/make_candidate_seal.py --first-fire-intent … --timing-risk-evidence …` with risk mode
`measured_t4_identity_class_envelope` (projection = max measured T4 inflate over the identity class's declared t4_direct legs: move 46's 1,140.8 s
and move 48's 1,023.3 s; local ratio kept as ceiling × (1 + fraction) ≤ 1,800 s; fraction 0 REFUSED), the frozen contract
`.omx/research/ddm_ffi1_20260910/PREFIRE_CONTRACT_FROZEN.json` (11 rows; drift refuses emission), and pr18's behaviour digest v2
(9f6e7168… must match). Then the normal validator's typed refusal is recorded, the intent JSON is COMMITTED, and MAIN alone claims the lane,
authorizes (`tools/authorize_candidate_first_measurement.py`), fires, harvests, completes (`--complete-first-fire-intent`), and packets move 50.
Read `.omx/research/ddm_hpr1_20260911/v2/` (PREFIRE_INTENT, TIMING_RISK, REAL_DOOR_CONTROL, SEAL v3) and hpr1's `experiments/ddm_hpr1_seal_inputs.py`
modes as the worked example; `.omx/research/ddm_pr19_20260911/TIMING_RISK_FIELD_SPEC_FOR_HPR1.json` for the risk fields.

## Resume surface
Your pass-7 memo `.omx/research/ddm_sj1_t4_token_predistortion_pass7_20260912.md` + receipts under `/Volumes/VertigoDataTier/pact/ddm_sj1_pass7/`
(read-only; the pricer `experiments/ddm_sj1_rlc1_price.py` proves itself by byte-identical repack of the pointer — reuse it, now against move 49's
tail 118,938 B); base = move 49's SHIPPED bytes by parse-back; receiver env as inflate.sh exports (RLC1_GEOMETRY_LIBRARY etc.); laws unchanged
(resolve pose on the edited field, admit only on the resolved pose, frame-0 repair inside the admission, bind base + ranking by sha, twin-price by
real encode, seg composes by re-verification). Fix `patch_inflate_pins` → MANIFEST.sha256 for YOUR tree (a sibling codex arm lands the class guard;
do not wait for it). Heavy steps via `tools/launch_detached_process.py --output-dir /Volumes/VertigoDataTier/pact/ddm_sj1_pass8/<stage> --nice 0
--done-receipt …`; waits as background receipt-only until-loops; no `pkill` by receipt name.

## Deliverable
Pass 8 on the move-49 field → Lagrange subset under the three legs → twin price → falsifiers pre-registered → timing-risk receipt → COMMITTED
intent (`CANDIDATE_PREFIRE_INTENT.json` with file sha/bytes/digest in the memo) + the normal validator's typed refusal + all seal inputs staged
(census, manifest regenerated from outside the tree, raw identity NOT expected — the field changes — but the decoded output's scorer numbers must
equal the in-loop numbers, smokes candidate + frontier, retention ≤ 8 GiB with shas). NO authorization, NO fire, NO completion, NO packet (MAIN's).
Memo `.omx/research/ddm_sj1_t4_token_predistortion_pass8_20260912.md`; serializer commits (two review passes per .py; `[no-triality] [p0-ledger-ok]`;
never a co-author trailer or AI attribution); lane id `ddm_sj1_t4_token_predistortion_pass8_first_measurement_20260912` (claim it); checkpoint `ddm_sj1`.

## Boundaries
No Modal; no `authorize_*`; no `fire_modal_auth_eval.py`; never edit `upstream/`, the PR tree, sealed trees, contract code (`candidate_seal.py`,
`decode_wall_clock.py`), or the receiver; do not touch the renderer/realization (ren2/rq1 closed both ways); never lower a reserve (Vertigo ~40 GiB
free after pass 7's 7.3 GB retention — check `df` before every heavy step; APDataStore is the overflow tier); no ScheduleWakeup.

## OPTIMAL FORM
Reference form: pass 7 exactly (machinery, admission, instrument, pricer) with the base moved to move 49 (SCOPE) and the seal path changed to the
contract's first-measurement route (a CONTRACT requirement, not a mechanism change). Provenance pins: HEAD (record), pass-7 memo sha (record), move
49 packet + seal (above), frozen contract sha (record), field sha of move 49 (read from the tree by parse-back and record).

## Prior negatives accounted (operator 2026-08-15)
Same-field decay (pass 5 closed at 0.634 %); pass 7's reach law (repairs consume slack); pr19 (no transitive inheritance — this charter obeys it);
the env KeyError of pass 7's harness (export the four native libraries as inflate.sh does); the tc1 pricer defect (use the RLC1 pricer);
container lottery sd 34.8 B (quote margins in its units); ntb2's prize falsified (never touch the renderer).

Final message: yield fraction vs the pre-registered band, subset size, three-leg decomposition and projected S, the intent path with file
sha/bytes/digest and its commit, the typed normal-validator refusal, seal-inputs path, every boundary, and the frontier line
`composition S 0.13632299781031237 @ 179,153 B [contest-CUDA T4 n600] (move 49)` — a new number only if MAIN's fire moved it.

<!-- # FORMALIZATION_PENDING: charter, not a finding; the pass-8 row, if any, carries its equations leg through tools/pointer_move_packet.py --equations-leg at harvest -->

# ddm_cr1 — the deeper carrier re-solve on all 600 pairs (pc3 ITEM 4, cb1's lead) composed with pass 8's admitted subset; price on move 49 (charter, MAIN 2026-09-12; operator full-authority GO; Opus)

## Why (measured facts; read at source)
- cb1 (`.omx/research/ddm_cb1_carrier_basis_refit_to_pose_residual_subspace_20260912.md`, §control re-solve; `/Volumes/VertigoDataTier/pact/ddm_cb1/VERDICT.json`):
  re-solving the SHIPPED basis's per-pair coefficients with up2's solver (`up2.solve_pair_realized`, its warm start / neighbourhood) improved
  24 of 48 seeded-random pairs for **−0.32 % of d_pose at 0 B** (keep-12 rebuild byte-identical). pass 7's control (`jg5.refine_pair`, lighter)
  had called the carrier converged (2 of 600 improve, Δ0) — the verdict was solver-relative. pc3 named this "ITEM 4, unowned".
- pass 8 (`.omx/research/ddm_sj1_t4_token_predistortion_pass8_20260912.md`; receipts `/Volumes/VertigoDataTier/pact/ddm_sj1_pass8/admission_pass8/`):
  admitted subset 11 pairs / 19 cells / 19 tokens; seg −1.612e-5, pose −4.86e-6, rate +6.74e-6 ⇒ net −1.424e-5 = 0.71 bars; nothing built.
- Arithmetic (DERIVED): −0.32 % of d_pose 4.55e-6 → pose leg −1.08e-5 S; + pass 8's −1.42e-5 ≈ −2.5e-5 ≈ 1.25 bars IF the two compose
  (the re-solve gain lives on pairs pass 8 does not edit; pass 8's 11 pairs are re-solved inside its own admission). Pre-registered band:
  −1.5e-5 … −3.5e-5 S; fire bar net < −2e-5 on the RESOLVED pose with EXACT bytes; falsifier: the n600 deeper re-solve alone gains < 0.15 %
  of d_pose or costs > +20 B, or the composition realizes < 0.8 of the sum.

## Deliverable
1. **Deeper re-solve, n600, on move 49's shipped basis and field** (tree `/Volumes/VertigoDataTier/pact/ddm_sj1_pass7/candidate/candidate_runtime`,
   read-only, copy): cb1's exact solver configuration (cite its producer + config sha; reuse, do not rewrite), all 600 pairs, frame-0 repair
   inside; accept per pair only if the RESOLVED d_pose improves (pp1's measured-tolerance gate). Report per-pair improvements, the n600 mean
   d_pose (MEASURED, frozen CPU-torch PoseNet, DALI GT lineage), and the EXACT carrier bytes through Rice/DX2 (the solver may move codes; "zero
   bytes" is a claim about the container — measure it, twins). Seg unchanged by construction (frame 0 only) — verify on the decoded output.
2. **Compose with pass 8's admitted subset** by re-verification (never additive): apply pass 8's 19 token edits (`field_admitted.npz`,
   `keep_mask.npy`), re-render those 11 pairs, re-solve their carriers with the SAME deeper solver, then the three legs on the composed object:
   seg on the shipped-mode decode, RESOLVED pose n600, real-encode rate (RLC1 pricer `experiments/ddm_sj1_rlc1_price.py`, proved by
   byte-identical repack of move 49's tail 118,938 B; carrier through the shipped chain). Report alone / subset alone / composed and the
   realized fraction of the sum.
3. **If the composed row nets** (net ΔS < −2e-5, exact bytes): byte-close on the move-49 base (token stream + carrier; semantic member and
   prior byte-identical; receiver code byte-identical), twins, cold n600 public parse-back (decoded scorer numbers = in-loop), manifest via the
   Catalog #420 producer, census, smokes candidate + frontier (four native-library exports as inflate.sh sets them), retention ≤ 8 GiB with
   shas (Vertigo ~38 GiB free; APDataStore overflow), seal on the NORMAL path inheriting move 49's adopted leg
   (`/Volumes/VertigoDataTier/pact/ddm_sj1_pass7/SEAL_ddm_sj1_token_predistortion_pass7_contest_cuda.json.decode_wall_clock.json`; pr18 behaviour
   digest 9f6e7168… must match). `tools/make_candidate_seal.py`; NO Modal, NO fire, NO packet (MAIN fires). If only the re-solve alone nets,
   seal that. If neither nets, close with the table.
4. Memo `.omx/research/ddm_cr1_deeper_carrier_resolve_n600_composed_with_pass8_subset_20260912.md`: per-pair improvement histogram, the
   three-leg table (alone / subset / composed), exact bytes + sha, margins in units of the 34.8 B lottery and the instrument's pose
   reproduction, falsifier verdicts, every boundary; serializer commits (two visible review passes per .py; `[no-triality] [p0-ledger-ok]`;
   NEVER a co-author trailer or AI attribution); lane `ddm_cr1_deeper_carrier_resolve_n600_composed_pass8_20260912` (claim it); checkpoint
   `ddm_cr1`. Heavy steps via `tools/launch_detached_process.py --output-dir /Volumes/VertigoDataTier/pact/ddm_cr1/<stage> --nice 0
   --done-receipt …`; waits as background receipt-only until-loops; `sys.dont_write_bytecode` against the read-only tree.

## Boundaries
No Modal, no `authorize_*`, no `fire_modal_auth_eval.py`; never edit `upstream/`, the PR tree, sealed trees, contract code, receiver code,
the renderer, the basis, or the prior; never lower a reserve; sj1/cb1/pp1/ren/dpi1/rq1/so1 directories read-only; no ScheduleWakeup.
Label MEASURED / DERIVED / INFERRED / ASSUMED.

## OPTIMAL FORM
Reference forms: cb1's exact solver configuration; pass 8's admission receipts and pricer; the receiver's carrier coders. Declared deltas: n
(48 → 600, SCOPE) and the composition (the rung). Provenance pins: HEAD (record); cb1/pass-8 memo shas (record); move 49 packet + seal + leg.

## Prior negatives accounted (operator 2026-08-15)
- pc2/pc3: the carrier lattice/rank/basis are closed — this rung moves only coefficient CODES inside the shipped lattice.
- cb1: a warm start off the shipped codes is catastrophic — start from the shipped codes (cb1's control did).
- lb1 (authority substitution priced by the incumbent): pass 7's "converged" was under a lighter solver; state which solver each number uses.
- Composition law (sy2/m148): compose by object change and re-verify; report the realized fraction of the sum.
- Container lottery sd 34.8 B and Rice bits: "zero bytes" must be measured on 600 pairs, twins.
- Pose re-solve law (op 09-10): admit on the RESOLVED pose; frame-0 repair inside.

Final message: the per-pair histogram, the three-leg table with exact bytes + sha, fraction of the sum, the seal path with file sha or the
typed blocker, commit shas, retained bytes, every boundary, ending with `composition S 0.13632299781031237 @ 179,153 B [contest-CUDA T4 n600]
(move 49)` — a new number only if MAIN's fire moved it.

<!-- # FORMALIZATION_PENDING: charter, not a finding; the row, if any, carries its equations leg through tools/pointer_move_packet.py --equations-leg at harvest -->

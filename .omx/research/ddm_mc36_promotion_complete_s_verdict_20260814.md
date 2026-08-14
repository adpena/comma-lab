# MC36 Variant C complete-S promotion verdict — THE FRONTIER POINTER MOVED (2026-08-14)

**VERDICT: PROMOTED. Canonical S = 0.1619344578804448 [contest-CUDA T4, n600,
full `archive.zip → inflate.sh → upstream/evaluate.py --device cuda` chain].**
effective_frontier moved 0.16195513827824176 → **0.16193445788044480**
(Δ = −2.0680e-5, ~6× the ±3.5e-6 report-quantization band). First frontier
pointer move bought by the micro-edit campaign, and the promotion the named
component row (memo a3a2196128, −1.99799e-5) owed.

## The row

- Archive f0ba4bb41d55fff85542f2a17dfe682508aa4f9ab50ef51cda573d79f0c4b1de,
  186,269 B (MC36 Variant C: mc35 union DROP pair 532 + FRESH-solve pair 105
  with qs5 in-compile pose compensation, on the cp135 composed base).
- Call fc-01M00RBQM4RMGEG2GXK4H8MEVX, comma-auth-eval T4, ~7 min wall, ~$0.25.
  Dispatched via the canonical paired launcher
  (tools/dispatch_modal_paired_auth_eval.py --execute), pair_group
  ddm_mc36_promotion_paired_modal_auth_20260814T182512Z.
- Components (contest_auth_eval.json sha 87cdfb49…): d_seg 0.00029611 ·
  d_pose 6.88e-6 · 186,269 B. Canonical S recomputed from components per the
  rounded-display rule (printed "0.16" is display only):
  0.029611 + 0.008294576541331089 + 0.12402888133911373 = 0.1619344578804448.
- Component-row cross-check: the a3a2196128 prediction (0.16193516) agrees to
  −7.0e-7 — within one report-quantization ULP. The component adjudication and
  the complete-S authority CONCUR.
- Custody: MODAL_REMOTE_RESULT sha fa2cc066… + contest_auth_eval.json +
  25 KB stdout log under
  experiments/results/modal_auth_eval/ddm_mc36_promotion_paired_modal_auth_20260814T182512Z_cuda/.
  Posterior ingestion via posterior_update_locked_from_auth_eval_json
  (accepted, axis=cuda); pointer refreshed via
  tools/refresh_canonical_frontier.py.

## CPU axis: STRUCTURALLY INFEASIBLE (recorded, not skipped)

The paired CPU leg (fc-01M00RY87TDDE5N556HXNQ1JZX) failed at 3.2 s (~$0):
`runtime/f26_inflate.py:106 → InflationError("F26 inflation requires a
CUDA-capable GPU")`. This is the KNOWN lineage property (m05; #998): the
PR130/PR135 runtime family is CUDA-locked by its own code. The cp135 floor
holds effective_frontier on identical CUDA-only custody, so MC36-vs-cp135 is
same-axis apples-to-apples. Both CPU-leg ledgers closed terminal
(failed / failed_cpu_infeasible_cuda_locked_runtime).

## Gates + defects encountered (all recorded)

1. Bare ::main fire REFUSED by the paired-by-default gate → re-routed through
   the canonical paired launcher (correct refusal).
2. The paired launcher's CUDA leg wrote spawn metadata but did NOT register
   the call in the call-id ledger (late-registered by MAIN; sister of the mt1
   write_spawn_metadata defect — the paired launcher needs the same
   fail-closed registration audit; dt1-census seed).
3. Concurrent CPU-sibling fire: the modal-single-flight env override engaged
   correctly, but the claim tool's own conflict check refused (rc=5, no
   --override threading in the CPU wrapper). Resolution: sequential pairing —
   which then surfaced the structural CPU infeasibility anyway.
4. AC1 closer on the CUDA leg: REFUSED_DUAL_LEDGER (no ledger row existed —
   consequence of defect 2), but its terminal_decision + result
   materialization were CORRECT; MAIN closed the ledgers manually.

## Waterfall at the new floor (exact)

S 0.1619344579 = seg 0.029611 + pose 0.0082946 + rate 0.1240289
(25·186,269/37,545,489). Sub-0.15 needs −0.0119345 from here: pose→0 buys at
most −0.0082946; seg remains load-bearing (needs ≥ ~−0.0036 beyond pose-zero,
or rate representation-changing work). Base instrument for successor micro-edit
rows: 34,933 flips · d_pose 6.88e-6 (report precision) · 186,269 B — the
same-instrument full-precision base is the retained T4 custody on
comma-ddm-js1b-argmax-retained /ddm_js1b_retained/ddm_mc36_dual_axis_t4_r1/.

## Follow-ons

1. mt1 #978 sign gate re-fire (fixed dispatcher 369d0bbb63; seal c9d6d62c
   valid) — last open seg-family custody item; T4-queue-starvation risk noted.
2. Bank note unchanged: qs2 (−4.375e-6) + re1 (−1.207e-6) do NOT auto-compose
   onto MC36; a joint rebuild on the NEW floor is a new candidate.
3. Paired-launcher call-ledger registration audit (defect 2) → dt1 census.
4. MEMORY GOAL-row update at the next quiet boundary: floor is now MC36
   0.1619344579 @ 186,269 B, archive f0ba4bb4….

Own-vehicle line: lc2 S 0.16959899569230852 @ 187,226 B [contest-CUDA T4,
n600]; **effective floor NOW MC36 Variant C 0.1619344578804448 @ 186,269 B
[contest-CUDA T4, n600] — moved this turn, Δ −2.068e-5.** Modal ≈ $5.85/$20.

# ddm_mt1 — #978 coupled multi-token T4 sign gate: NEGATIVE (2026-08-14)

**VERDICT: `positive_t4_sign = false` on real T4 population custody.** The
sealed second-train order does not exist and nothing fires. The #978 coupled
multi-token family closes at this formulation.

## The measurement

Call `fc-01M01D2MCYVF87X2A0NKPSVT28` (run_id `ddm_mt1_t4_sign_gate_20260814`,
T4, rc=0, ~$0.10). Three arms, 24 batches, n32 stratified-random heldout of
the hidden-4, max-support-mass-0.25 local simplex formulation (the receipt's
own `verdict_scope`; `score_claim=false`, `promotion_eligible=false`).

| arm | d_seg | d_pose |
|---|---|---|
| cp135_hard (base) | 0.00024302800496419272 | 0.00011792784789577127 |
| mt1_multitoken | 0.00024302800496419272 | 0.00021997121802996844 |
| hc1_direct_c1 | 0.00041151046752929690 | 0.00576400756835937500 |

The multitoken arm holds seg EXACTLY (identical to base) and makes pose
1.865× WORSE. The direct-C1 arm is worse on both axes. The T4 population
authority CONFIRMS the local n32 screen's negative — the sign did not flip.

## Scope and routing

- verdict_scope: FORMULATION-level screen (fixed-seed n32 heldout, hidden-4,
  support-mass cap 0.25). It kills THIS formulation's train fire; it is not
  a family-wide kill-authority row (per the charter's own typing).
- Convergence consequence: with pk4 + ps135b + pz4a terminal and #978 now
  negative at its screen, the live majors are exactly TWO: rx2 (Route C rate
  learner, training live toward the terminal QAT serialization test) and the
  JS1/#982 gate-aware joint treatment (queued on rx2 harvest).

## Custody (ALWAYS KEEP THE PAYLOAD)

- FINAL_RESULT.json harvested + sha-verified: `fc2b5938a45549…f8632` (81,052 B)
  at `<mt1 store>/harvest/FINAL_RESULT.json`.
- COMPLETE remote custody (790 files, 2.5 GB incl. all three arms' retained
  pose fields) mirrored to
  `/Volumes/APDataStore/pact/ddm_mt1_t4_sign_gate_20260814_custody/`
  (Vertigo hit ENOSPC at 100% mid-pull; partial removed, pointer manifest
  `COMPLETE_REMOTE_CUSTODY_MOVED.json` left in the mt1 store). The remote
  Modal volume `comma-ddm-js1b-argmax-retained` retains everything.

## The 5-round infra saga (all cures landed, receipts on disk)

r1/r2: $0 queue starvation. r3: $0 — sealed-argv signature drift + container
partial-package-mount ImportError → ddm_dr1 repaired all three receipts
(10c81bd3c26e) incl. the closer call-ID registration. r4: ~$0.04 remote rc=1 —
missing `CUBLAS_WORKSPACE_CONFIG` under `torch.use_deterministic_algorithms`
(Catalog #244 class; the CPU smoke could not see a CUDA einsum) → one-line
env fix (3f60af504a) + reseal via dr1's signature-checking builder. r5: rc=0,
clean harvest, closer end-to-end (its closure-manifest step returned
REFUSED_MANIFEST after harvest — cosmetic, ledger+claim closed terminal).
Total mt1 spend ≈ $0.15.

## NEXT_IF_RESUMED
- rx2 terminal-QAT harvest = the live serialization test (potential ≈0.134 S
  IF the 144,906 B surrogate survives identity closure).
- JS1/#982 gate-aware joint treatment fires after rx2 harvest.
- #978 reopens ONLY with a new formulation (e.g. trained-receiver coupling
  or support-mass >0.25) — not by re-firing this screen.

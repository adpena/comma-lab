# ddm_mc36 dual-axis T4 verdict — FIRST NAMED CANONICAL ROW of the micro-edit campaign (2026-08-14)

**VERDICT: ADMITTED + NAMED. Net realized ΔS = −1.99799e-5 [contest-CUDA T4, n600,
same-instrument composed] — 2.0× the 1e-5 naming bar.** Per the mc35/mc36
pre-registered rule ("canonical row NAMED only if it survives the worker"), MC36
Variant C is the first named canonical row of the micro-edit campaign.

## The object

- Candidate archive: `f0ba4bb41d55fff8...` (full sha
  f0ba4bb41d55fff85542f2a17dfe682508aa4f9ab50ef51cda573d79f0c4b1de), 186,269 B.
  Built by ddm_mc36 (commit 2e4abc6210): mc35 union DROP pair 532 + FRESH-solve
  pair 105 with qs5 in-compile pose compensation.
- Base: cp135 composed floor, matched instrument (34,970 flips ·
  d_pose 6.885642960696714e-6 · 186,252 B).
- Dispatch: call `fc-01M00KSF1VB1636W0P82FB25DN`, ~$0.16, 716 s, Tesla T4,
  qs1 dual-axis worker on fs1's conformant seal (2a311813..., MAIN-verified
  loader acceptance + all 3 input SHAs before fire).
- Worker result sha:
  5bd6044cad503c5585659376882d9e39dc10b1f0d2a3796b5bca6d728a9e0e00
  (QS1_T4_REMOTE_RESULT.json, schema ddm_re1t_t4_measurement_result.v1).

## Component adjudication (exact, same instrument)

| leg | measured | ΔS |
|---|---|---|
| seg | base 34,970 → candidate 34,933 flips vs GT (−37; 180 px changed vs cp135) | −3.136529e-5 |
| rate | 186,269 − 186,252 = +17 B; 25·17/37,545,489 | +1.131959e-5 |
| pose | d_pose 6.885752100060927e-6 vs 6.885642960696714e-6 (+1.09139e-10); marginal 602.56 S/unit | +6.576e-8 |
| **NET** | | **−1.99799e-5** |

- **Instrument integrity:** base flips 34,970 EXACT match to the pinned base
  instrument. Seg transfer was pixel-exact: the local advisory predicted 37 net
  flips; T4 realized exactly −37.
- **Pose micro-sign flipped** (local advisory −1.463e-10 → T4 +1.091e-10) but
  the magnitude is one part in ~60,000 of the seg win — negligible either way.
  Pose repeat bit-identical (repeat_noise_mse 0.0); first/repeat first6 vectors
  share sha bf339749....
- All local gates and the T4 realization agree; projection −2.013e-5 vs
  realized −1.99799e-5 (Δ = the pose micro-sign).

## New composed-row candidate

cp135 composed floor 0.16195513827824176 @ 186,252 B → **MC36 Variant C
composed S ≈ 0.16193516 @ 186,269 B [contest-CUDA T4, n600, COMPONENT-composed
same-instrument]**. Honesty boundary (worker's own): "contest-CPU or upstream
evaluate.py; complete-S adjudication remains local" — score_claim=false,
promotion_eligible=false. **The complete-S `upstream/evaluate.py` row on the
exact archive remains the promotion step** before the canonical frontier
pointer moves; this memo names the row on the campaign's pre-registered
instrument, exactly as qs2/re1 admissions were adjudicated.

## Custody

Everything retained on durable Modal volume `comma-ddm-js1b-argmax-retained`
under `/ddm_js1b_retained/ddm_mc36_dual_axis_t4_r1/`: exact-receiver raw decode
(3.66 GB, sha a41ca69d...), candidate argmax field (sha ebcded1d...), all
logits + seg inputs, all pose inputs/outputs + first6 vectors (3 sources),
batch receipts. Local: ENDPOINT_CLOSURE.receipt.json + MODAL_REMOTE_RESULT.json
+ QS1_T4_REMOTE_RESULT.json under
/Volumes/VertigoDataTier/pact/ddm_mc35_successor_drop532_pair105/dispatch/ddm_mc36_dual_axis_t4_r1/.

## AC1 closer — first live dogfood: FLAWLESS

Armed detached at dispatch (--allow-legacy-manifest, agent=MAIN after the
reaper-predicate refused "claude-main"); polled 716 s, drove claim →
completed_endpoint_harvested + call ledger → harvested (both terminal,
invariant held), storage-preflighted, materialized the result with SHA
verification. Zero manual harvest steps. The endpoint-closure-must-be-automatic
law is now PROVEN in production.

Second dogfood is orphan-rescue mode: the mt1 dispatcher spawned
fc-01M00MQ1S6ZV0E7AKD5514YR78 then crashed at write_spawn_metadata (signature
mismatch, a second latent arm-written-dispatcher defect fs1's fix did not
cover — the spawn + fail-closed ledger registration had already succeeded).
Closer armed on the live call (pid 43935); dt1-census seed: dispatcher
local_entrypoint chains never executed end-to-end by arms (they cannot run
Modal) — mc36 avoided this class by REUSING the proven qs1 dispatcher.

## Follow-ons

1. mt1 #978 sign-gate harvest (closer live) → custody closure of the last open
   seg family.
2. Promotion step: complete-S evaluate.py row on archive f0ba4bb4... (queues at
   the next authority window; component row stands NAMED regardless).
3. Bank note: qs2 (−4.375e-6, +34 B) and re1 (−1.207e-6, 0 B) do NOT auto-compose
   onto MC36 (different objects on the same base); a joint rebuild would be a new
   candidate.
4. mt1 dispatcher write_spawn_metadata signature fix — trivial, rides the next
   commit touching that file; the closer made it non-blocking.

Own-vehicle frontier line: lc2 S 0.16959899569230852 @ 187,226 B [contest-CUDA
T4, n600]; effective floor cp135 composed 0.16195513827824176 @ 186,252 B —
MC36 Variant C is the named −1.99799e-5 improvement candidate on that floor.
Modal ≈ $5.55/$20.

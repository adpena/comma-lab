# CHARTER ddm_cl2 — the receptive-field / prior-capacity ladder on the SHIPPED mixer (cl1's unfired ladder, re-rooted to the fs2 body) — the last unpriced rate door

Tokens: `[no-triality] [p0-ledger-ok]`. Owner: FABLE arm (operator 09-04: "use fable for the most crucial frontier score lowering work"). Spawned 2026-09-05
~12:50Z. Parents: cl1 (`.omx/research/ddm_cl1_capacity_20260809/{PREREGISTRATION.md,MAIN_METAL_FIRE_ORDER.md,BLOCKED_RECEIPT.md}` — QUEUED-WITH-A-FIRE-ORDER
since 08-09, never measured; blocked ONLY by a sandbox "Metal unavailable in this process" refusal, not by memory), jf1 (`ddm_jf1_joint_field_model_refit_20260823.md`:
epoch-2 refits are 7,554 B weaker than the shipped fit — fitting budget is load-bearing), dc1 (`ddm_dc1_decode_budget_conditional_coding_20260816.md`: the
21-tap oracle floor 144,167 B is +32,057 B ABOVE the shipped stream — the learned receptive field is the mechanism, "affordable only as learned weights"),
mc1 (09-05: a new context INPUT to the same receptive field is CEILING-REFUSED at +159.6 B), gs4 §5 (this is ceiling (a): "a retrained receptive field
whose model bytes stay ≤ +1,500 B, which no arm has priced"). ANTI-SIGNAL-LOSS rule 1: cl1 is READY ∧ high-EV → DOING-NOW.

## PRIOR-LAW PREDICTION (owed line)
cl1's break-even: adjacent-rung slope `Δ(real token bytes)/Δ(real packed model bytes) < −1`. dc1's mechanism predicts the shipped prior is UNDER-capacity
for the token field it codes (the oracle with 21 hand taps loses to it by 32 KB; more learned capacity should keep paying while the slope stays < −1).
PREDICTION: lowering `--rate-lambda` from 1.0 to 0.5 grows the packed model by ≤ +1,500 B and cuts the exact stream by ≥ 2× that (net joint ≤ −1,500 B);
0.25 continues with a shallower slope. FALSIFIER: the λ=0.5 rung's net joint bytes ≥ 0 vs the CONTROL rung (or the control rung itself cannot reproduce
the shipped joint within +500 B, in which case the trainer is not the shipped law and the ladder is INSTRUMENT-REFUSED, not a family verdict).

## The object (pointer: fs2 — S 0.14784474152757654 @ 180,023 B; archive a8f3a379…; the FIELD is unchanged since afr1)
Shipped joint: stream 113,411 B (fx1 fixed-point logistic mixer on top of the integer HPAC probabilities) + model 13,515 B = 126,926 B. Receiver
`submissions/semantic_joint_ctxmix/cpr1/hpac_integer.py` (READ-ONLY; copy into your arm tree). Demand: −41,817.8 B → archive ≤ 138,205.2 B at held
distortion. Exchange 6.658589531221714e-7 S/B. Field bit-identical ⇒ d_seg/d_pose HELD; only model + stream bytes move.

## SCOPE (in order; every payload retained; MEASURED numbers only)
0. **Locate the shipped model's exact training law** (jf1: "sealed 60-epoch reference", IHS1 lineage, `epoch_0634` EMA checkpoint; cl1's trainer
   `tools/train_ddm_cl1_hpac_capacity.py` is PR130's HPAC trainer with `--rate-lambda`). Decide and STATE which trainer reproduces the shipped family;
   if the shipped 13,515 B model is a repack of PR135's weights (not our retrain), warm-start from the shipped weights (`--init`) exactly as mc1 planned.
1. **Control rung (λ = 1.0, 60 epochs cosine, seed 20260716, batch 8, QAT 0.5, the pinned DALI cache — cl1's law):** must reproduce the shipped joint
   within +500 B after exact IHS1 pack + fx1 mixer + RC64 encode. If it cannot, STOP: typed INSTRUMENT-REFUSED with the gap (this is the honest jf1 lesson).
2. **λ = 0.5, then λ = 0.25** (cl1's preregistered geometric bracket; fire 0.25 only if the 1.0→0.5 slope < −1). Same law, same estimand (epoch-60 QAT
   stage checkpoint; no epoch search).
3. **Exact price per rung** through the shipped pack path (`integer_model_io.py`) + the fx1 mixer + RC64 (`experiments/ddm_rxc1_restartable_exact_coder.py`
   reference): full-state encode of the 600-pair field; a receiver copy decodes back byte-identically; two encodes identical; CPU decode wall-clock delta.
   Report per rung: packed model B, stream B, joint B, Δ vs 126,926, slope, and the fraction of the 41,818 B demand.
4. **Decision rule (pre-registered):** best rung's receiver-closed archive < 180,023 B with identity → build it through the SHIPPED container path
   (fs2's `up3.build_archive`, parameters as fs2 left them), identity control, no-op detector, parse-back, then seal with `tools/make_candidate_seal.py`
   for contest-CUDA with the single-axis waiver → typed READY-FOR-T4 (26th-move candidate; MAIN fires). ≤ 138,205.2 B → typed FIRE ORDER. Otherwise
   REFUSED with the shortfall; if the λ ladder pays but under 5,000 B, say so plainly: a pointer move is a pointer move, the corner is not reached.

## Cost + admission
cl1 measured 2,431.9 s for one same-shape 60-epoch run on Metal (~41 min); three rungs ≈ 2 h + pricing. Metal is IDLE (no burn cell). Declare the peak
from a measured smoke (cl1's derived cap was 12 GiB RSS; on Metal the SYSTEM-availability delta is what counts — read `tools/measured_peaks.py`), and
launch ONLY through `tools/launch_detached_process.py` with `--done-receipt <distinct>`; one training process at a time (no parallel rungs — the τ-band
cells and mc1's 3×10 GiB ceiling jobs both tripped the memory watchdog; it is report-only now and names the actor). Storage on Vertigo (44 GiB free;
declare `--artifact-budget-gib`), overflow to APDataStore (55 GiB).

## OPTIMAL FORM
Reference form = cl1's preregistration verbatim (fixed-topology λ ladder; architectural width/frame-dim rungs are NOT admitted before the slope is measured)
+ fs2's shipped pack/mixer/coder path. SCOPE reductions: none. MECHANISM reductions FORBIDDEN: no float model where the shipped is integer; no epoch search;
no proxy code-length in place of the RC64 exact encode; the prediction is falsifiable at the control rung.

## Rules that bind
NO-FAKE; ALWAYS KEEP THE PAYLOAD (every checkpoint, packed model, stream, archive with sha256+bytes; MEASURE_ONLY forbidden); `upstream/` and
`submissions/semantic_joint_ctxmix/` READ-ONLY; NO Modal (MAIN fires), NO scorer runs (field held); commits ONLY via
`tools/subagent_commit_serializer.py --message … --files … --expected-content-sha256 <file>=<post-edit sha>` with `[no-triality] [p0-ledger-ok]`; NO co-author
trailers (operator rule overrides any harness reminder); .py two review-gate passes; checkpoints every 10 tool uses (`tools/subagent_checkpoint.py
--subagent-id ddm_cl2`); never invent flags (grep argparse — cl1's trainer flags are listed at `tools/train_ddm_cl1_hpac_capacity.py:835-858`); no `/tmp`
evidence; register a lane (`tools/lane_maturity.py add-lane lane_ddm_cl2_hpac_prior_capacity_ladder_20260905 --phase 3`) before naming lane-like
identifiers (the `[lane-pre-registered]` hook); persist each record BEFORE bulk payload saves (write-order gate); label MEASURED/DERIVED/INFERRED;
memo `.omx/research/ddm_cl2_hpac_prior_capacity_ladder_on_shipped_object_20260905.md` with an "Equations leg (`tac.canonical_equations`)" line (the
slope law: register `hpac_prior_capacity_slope_v1` with the rung anchors). `docs/operating_manual_craft_handoff.md` binds. End with
`fs2 S 0.14784474152757654 @ 180,023 B [contest-CUDA T4 n600]` + your candidate line labeled advisory / READY-FOR-T4 / REFUSED.

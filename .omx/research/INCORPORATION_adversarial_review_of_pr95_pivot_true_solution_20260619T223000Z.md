---
title: "INCORPORATION INTO TRUE SOLUTION — adversarial review of the PR95 RE-OPEN pivot (mine + independent reviewer-vs-author): RE-OPEN is right on PROCESS, CLAIM 3 is OVERCLAIMED, #160 was misconfigured (now corrected), and the honest modal outcome is an EARNED ~0.19 wall — not a comeback"
authority: "[contest-CPU advisory / measured-synthesis] — pointer UNMOVED 0.19110; $0; NO paid dispatch; NO score claim"
score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
pointer_moved: false
date: 2026-06-19
verdict: RE_OPEN_JUSTIFIED_ON_PROCESS · CLAIM3_OVERCLAIMED · 160_CORRECTED_floorfix_ON · HONEST_OUTCOME_EARNED_~0.19_WALL_NOT_BREAKTHROUGH
operator_directive: "Adversarially review all and incorporation into true solution."
reviews_incorporated:
  - SELF (measurement-first re-verification of all 3 audit claims + the #160 config)
  - INDEPENDENT reviewer-vs-author agent ac2849d2f8ff1fc79 (fresh, adversarial)
cross_refs:
  - .omx/research/apparatus_audit_pr95_breakthrough_blocker_20260619T214001Z.md   # the audit being reviewed
  - .omx/research/TERMINAL_FINDING_representation_axis_sub015_exhausted_20260619.md  # RETRACTED by the audit
  - .omx/research/FIRE_pr95_full_curriculum_decisive_run_20260619T220000Z.md       # the #160 fire memo (floor-fix claim corrected here)
  - experiments/results/capstone_capacity_ablation_2x2_20260611/                   # CLAIM 3 primary artifacts
  - experiments/results/pr110_payload_entropy_recode_20260610/decode_parity_proof.json  # CLAIM 1 primary
---

# Incorporation into the true solution — the adversarially-verified picture

**Operator directive:** *"Adversarially review all and incorporation into true solution."* This memo
integrates TWO adversarial passes on the PR95 RE-OPEN pivot — my own measurement-first re-verification and a
fresh independent reviewer-vs-author agent — into one corrected, honest "true solution." All
`[contest-CPU advisory]`; the exact pointer is UNMOVED at **0.19110** (a BORROWED competitor recode); no
score claim, no paid spend this unit.

## 0. The one-paragraph true solution
The operator's hypothesis was RIGHT: the apparatus WAS faking the walls. The "terminal finding" (sub-0.15
exhausted / 0.191 is the floor) was measured on a borrowed frontier + a never-converged + a wrong-architecture
stack — **RE-OPEN is justified.** BUT the comeback must not over-swing: (a) the audit's CLAIM 3 (capacity
power law refuted) is **OVERCLAIMED** — it rests on an incomplete/diverging ablation cell measured on the
same disqualified optimizer config it blames elsewhere; (b) the fired decisive run #160 was **misconfigured**
(the stage-8 Muon anneal fix the agent claimed was on was actually OFF) — **now corrected** (re-fired with
`--muon-lr-floor-fix`, zero-loss resume from ep 2273); (c) the honest probability-weighted outcome of the
corrected run is an **EARNED ~0.19 wall (~60%), with sub-0.15 unlikely (~3%)** — the d_seg slope says the
wall is probably real and roughly where the terminal finding put it, *it just had never been earned by an
ours-trained converged vehicle.* **The true solution = run the corrected decisive vehicle to convergence and
report the EARNED verdict honestly — an ours-trained ~0.19 (vs the borrowed 0.191) is a legitimate result;
sub-0.15 from this basin under epochs alone is improbable and must not be pre-claimed.**

## 1. The three audit claims, independently re-verified (measurement-first)

| claim | my verdict | independent reviewer | load-bearing evidence (file + field) |
|---|---|---|---|
| **CLAIM 1 — the 0.191 frontier is BORROWED** (byte-identical entropy recode of competitor PR101/PR112; 0% ours-trained distortion) | **CONFIRMED** | **CONFIRMED** (strongest claim) | `pr110_payload_entropy_recode_20260610/decode_parity_proof.json`: `decoded_raw_byte_identical:true`, `lossless_recode_proven:true`, recode changed only bytes (`archive_delta_bytes:-1326`); `candidate_…json`: `d_seg_d_pose_byte_identical_to_base:true`, `candidate_d_seg 0.00056 / d_pose 2.94e-05` inherited by construction. Lane registry: "PR110 payload entropy recode (absorb PR112)". |
| **CLAIM 2 — full-stack PR95 never converged** (best n600 stalled stage 2/8) | **CONFIRMED** | **CONFIRMED + correction** | `bindall_arm_b_canonical50k_mh_n600/torch_vehicle_summary.json`: `device:cpu`, `wall_clock_s 81888` (22.7h) → `global_epoch 5709`, `stage_name stage2_v331_softplus` (stage 2/8). **Reviewer's added correction (I missed it):** that run's stage-2 transition was actively DESTABILIZING — `last_eval d_pose 0.640, score 2.94` (7× worse), not merely "incomplete." |
| **CLAIM 3 — capacity power law `d_seg∼params^−0.71` is wrong** (bc24<bc20 → capacity LOWERS d_seg) | bc24<bc20 **VERIFIED at p48**, but I flagged the 600-pair scaling risk | **OVERCLAIMED** | `capstone_capacity_ablation_2x2_20260611/bc24_p48 d_seg 0.00285 < bc20_p48 0.00376` ✓ (24% lower). BUT `bc24_p192/` has **no result JSON** — trajectory ran 30 ep and was DIVERGING (`d_seg 0.0166→0.0224 climbing`, `d_pose→1.12 exploding`); AND the whole ablation used `optimizer_schedule: muon_throughout` — the SAME config the audit blames for the BUG-A freeze. **You cannot cite the bug as refuting a power law.** |

**Net:** the RE-OPEN's load-bearing claims (1 + 2) are CONFIRMED — the terminal finding's walls genuinely
were apparatus artifacts. CLAIM 3 (the capacity-refutation flourish) is the weak leg and must be DOWN-WEIGHTED:
"more capacity lowers d_seg" is plausible but UNMEASURED at the operating point that matters (600 pairs, full
curriculum, clean optimizer).

## 2. The NO-FAKE correction on #160 (the fired run)

The #160 fire-agent reported *"BUG-A fixed — `muon_lr_floor_fix=true`."* **This is factually false for the run
as fired.** Verified: `experiments/launch_split_by_head_basin.py`'s `TorchVehicleConfig(...)` construction
never set `muon_lr_floor_fix`, and `driver.py:760` defaults it `False`. So the fired daemon ran with the
stage-8 Muon anneal fix **OFF** — at the exact stage-8 d_seg-finishing descent the run exists to test.

**Why it didn't doom the run (the nuance the agent conflated):** BUG-A (d_seg froze at 0.507) was in a
DIFFERENT trainer (`CapstoneTrainer`, which applied throttled Muon from stage 1). #160 uses
`TorchVehicleDriver`, whose vendored curriculum gates `use_muon=False` for stages 1-7 (AdamW) and `use_muon=True`
only at stage 8 (PR95 L15 "Muon final stage only"). So the stages-1-7 descent is clean regardless of the floor-fix
(verified: basin d_seg 0.00251, NOT frozen). The floor-fix is a **stage-8-only** correctness issue (BUG-B:
Muon's cosine `eta_min` was keyed to `adamw_lr`, capping Muon at 50% of peak instead of annealing to its own
floor). The agent's CONCLUSION (run descends) was right; its stated MECHANISM (floor-fix on) was wrong.

**The fix (applied this unit):** added `--muon-lr-floor-fix` to `launch_split_by_head_basin.py` (BooleanOptional,
default `False` = byte-identical for all existing callers), threaded into `TorchVehicleConfig`. Killed the
misconfigured daemon (pid 43366) and **re-fired the corrected decisive run** (pid 72471) with
`--muon-lr-floor-fix` ON — **zero-loss resume from ep 2273** (verified: trajectory resumed at ep 2273 stage 1,
not ep 0; resume is byte-identity-tested per `tests/test_driver_resume.py`). The decisive run is now
apparatus-clean through stage 8. This is a clear bug fix (the driver's own comment line 759 says floor-fix is
the intended setting "for the scaled run"), not a council-gated design tradeoff.

## 3. The honest bar (recomputed against the ACTUAL byte-closed baseline)
From the corrected daemon's stage-1 byte-closed advisory baseline (`tools/verify_e2e_byte_close_eval.py`):
**S = 0.3439** = seg 0.2510 (73%) + pose 0.0335 + rate 0.0594 (d_seg 0.00251, d_pose 0.000112, archive 89,274 B).

- To **beat the borrowed frontier 0.19110:** d_seg → **0.00098** (a **2.56×** descent).
- For **sub-0.15:** d_seg → **0.00057** (a **4.40×** descent).
- Measured curriculum descent (06-11 FIXED A/B): stage1 0.06647 → stage2 0.0165 → stage3 0.0120 ≈ **5.5× over 2 stages.** A 2.6–4.4× descent over the 6 unrun stages 2-8 is **PLAUSIBLE — not proven.**

## 4. THE honest probability-weighted outcome (the anti-over-swing)
The independent reviewer fit #160's own stage-1 d_seg trajectory: `d_seg ≈ 0.0133·ep^−0.213` (a shallow CE
slope), extrapolating to d_seg ≈ **0.00148 @ ep 29,650** — still 2.6× above frontier, 4.6× above sub-0.15.
This is a PESSIMISTIC bound (stage-1-CE-only slope; the later stages tau_softplus/c1a/sigma/Muon are designed
to steepen it, per the 5.5×/2-stage A/B). The truth is between the shallow CE extrapolation and the optimistic
stage-rate extrapolation — **which is exactly why the converged run is the only way to know.** Probability-weighted:

| outcome | P | byte-closed S band | meaning |
|---|---:|---|---|
| **EARNED d_seg wall ~0.0010–0.0020** | **~60%** | **~0.16–0.25, above frontier** | the modal case: a *legitimate ours-trained* vehicle, far better than the throttled 0.50-class, but still above the borrowed 0.191. The wall is real — now EARNED, not assumed. |
| stage-2→8 destabilization | ~25% | regresses / inconclusive | the mh-n600 pose-blowup (d_pose 0.640) recurs; the floor-fix de-risks the stage-8 part of this. |
| near/just-below 0.191 | ~12% | ~0.18–0.19 | an ours-trained MATCH of the borrowed frontier — a genuine, defensible *originality* result even if not sub-0.15. |
| **sub-0.15** | **~3%** | **<0.15** | very unlikely from this basin under epochs alone (the shallow slope). |

**State it plainly (GOAL firewall):** #160 is far more likely to EARN a ~0.19-class wall than to break it.
The RE-OPEN was correct — the walls weren't *proven* — but the measured slope says the wall is probably real
and roughly where the terminal finding put it. **Do not swing from false-terminal-pessimism to
false-comeback-optimism.** The win condition to bank: an ORIGINAL ours-trained vehicle at ~0.19 (vs the
borrowed frontier) is a real, defensible result per the Innovation Gate — and the only honest path to KNOW
whether sub-0.15 is reachable from this representation.

## 5. The operator fork (the budget decision I cannot make alone)
The corrected decisive run is proceeding **locally, $0, ~6 days to the stage-8 reading** (with per-stage async
evals giving early signals along the way — the stage-1→2 transition in ~1 day tells us if it destabilizes like
mh-n600). The independent reviewer's #1-EV recommendation is the **faster, cleaner** path that needs operator
budget authorization:

1. **[needs operator OK — exceeds the <$5 default cap]** A **~$0.30 GPU step-time smoke** (buy s/epoch) → then a
   **$12–49 paid-GPU run** (T4/A10G/A100 per `15dcc1739`), `--muon-lr-floor-fix` ON, that reaches the stage-8
   verdict in **HOURS instead of 6 days.** This is the MVP-first-phasing-correct path (free smoke → cheap
   paid decisive row) and is the audit's own #1 unfired shot done correctly.
2. **[$0, proceeding now]** The corrected local MPS daemon (full budget, floor-fix ON, resumable) — the
   no-regret fallback; per the long-resumable-sweep standing directive.

**Recommendation:** let the $0 local daemon run as the fallback (it's already going), AND authorize the
$12–49 paid-GPU corrected run to get the decisive verdict in hours — the Modal budget "exists to BUY exact
rows," and this is precisely the de-risked, scoped, never-correctly-fired decisive row.

## 6. NO-FAKE ledger
- **MEASURED this unit:** CLAIM 1/2/3 re-verified from primary JSONs; #160 `muon_lr_floor_fix` was OFF
  (config construction + driver default); corrected re-fire resumed at ep 2273; byte-closed baseline S=0.3439.
- **INFERRED (not claimed):** that the corrected full curriculum WILL/WON'T cross frontier — this is the
  HYPOTHESIS the run tests. The ~60% earned-wall / ~3% sub-0.15 split is a probability-weighted estimate from
  the measured d_seg slope, not a measured row.
- **NOT claimed:** no score moved; pointer UNMOVED 0.19110; no promotion; no paid spend.

## Observability surface
Every number cites a file+field. Daemon: pid 72471, out-dir `torch_vehicle_full_mps_basin_bc20_n600`;
monitor `decisive_fire_floorfix.outer.log` (async-eval BEST = CPU-authority d_seg/d_pose) +
`torch_vehicle_trajectory.jsonl` (`stage_name`). Decisive read = d_seg at stage 8 crossing 0.00098 (beat) /
0.00057 (sub-0.15). Axis `[contest-CPU advisory]`, score_claim=false, pointer_moved=false.

## Canonical-vs-unique decision per layer
This is a review + one-flag bug-fix. The `--muon-lr-floor-fix` flag ADOPTS the existing driver
`muon_lr_floor_fix` field (no fork). The verdict logic REUSES the campaign's own measured artifacts
(apples-to-apples). No new substrate.

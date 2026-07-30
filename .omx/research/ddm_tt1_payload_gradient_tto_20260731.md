---
schema: ddm_tt1_payload_gradient_tto.v1
date_utc: 2026-07-31
arm: ddm_tt1 (joint payload gradient-TTO pilot — QA71, operator-directed)
lane_id: "lane_ddm_tt1_joint_payload_gradient_tto_20260731"
research_only: true
score_claim: false
pointer: "0.1910828242 [contest-CPU] UNMOVED"
axis: "[macOS advisory — realized CPU decode + frozen CPU-torch PoseNet; MPS = gradient/proposal device ONLY, NEVER a score]"
operator_verbatim: "pick up that old research we did and the idea... to not use a proxy, but actually update in real time against the frozen scorer gradient."
operating_state: "v4c frozen baseline MEASURED S 0.992972 (evaluate.py: d_seg 0.00431179 · d_pose 0.01038450 · 359,750 B), archive b6365270"
tokens: "[no-triality] [p0-ledger-ok] [magnitude-ok]"
tools:
  - "experiments/ddm_tt1_twin.py (WarpTwin: differentiable torch/MPS twin of the v4c continuous decode + the numpy realized acceptor)"
  - "experiments/ddm_tt1_joint_tto.py (Adam-propose / realized-accept harness; joint | pose_only | ab_only)"
  - "experiments/launch_tt1_detached.sh (ppid-detached, tac-hijack guarded, PATH-exported, resumable, all 3 modes)"
data: "SSD ddm_tt1_20260731/{archive_v4c/, tto_{joint,pose_only,ab_only}.partial.jsonl, tto_*_receipt.json, run.log}"
---

# ddm_tt1 — joint payload gradient-TTO: the #350 revival under the ph3 realization doctrine

**POINTER HONESTY FIRST: `0.1910828242 [contest-CPU]` UNMOVED.** Every number
below is `[macOS advisory]`, `score_claim=false`. This pilot lowers the **v4c
advisory vehicle** (measured `evaluate.py` S 0.992972), which is FAR from the
pointer — it does NOT move the pointer. MPS is used ONLY as the gradient/proposal
device (fp32, 104× scorer law); the ONLY acceptor is the real numpy decode + the
frozen CPU-torch PoseNet (proven bit-exact with the gate instrument, §2).

## §1 The revived form (ph3 §1 doctrine, binding)

The operator's "old research" (#350 payload-space TTO) historically walled on the
**proxy-auth gap** (the tangent/gradient over-predicts the realized response 3–10×
— pi2; 2–11× on PoseNet without the yuv6 patch). The ph3 doctrine is the cure:
**the frozen-scorer gradient is the PROPOSAL ENGINE, never the acceptor.** Every
proposed step is realized through the uint8/decode/PoseNet path and accepted iff
realized ΔS < 0 at the shipped f16 quanta. This pilot is that loop, applied to the
v4c shipped payload — gradient descent ON THE SHIPPED PARAMETERS (per-pair pose
6-vector + photometric gains a,b) through the deterministic decode, with realized
monotone acceptance. Compress-time ONLY (strict scorer rule: the receiver ships no
scorers; we optimize the shipped description).

## §2 The instrument (twin + acceptor) — BOTH controls PASS

**Acceptor control (the load-bearing one): the numpy v4c Decoder + frozen CPU-torch
PoseNet reproduces the gate's per-pair d_pose to `|res| = 0.0e0` on 5 stratified
pairs** (0, 44, 100, 300, 500; incl. the d=0.78 hard core and the b=8.75 big-
exposure pair). The acceptor IS the gate instrument, bit-exact — so realized
per-pair acceptance is gate-grade (the fidelity law's 3rd anchor, v4c, held at
`evaluate.py` residual 1.38e-4).

**Twin fidelity control: the differentiable torch twin matches the numpy decode**
to `|Δd_pose| ≤ 9e-6` (CPU) / `≤ 8e-6` (MPS) at the shipped values (band 1e-3) —
pixel `mean|Δ| ≈ 2e-4`. **Twin gradient correctness: the pure differentiable path
(no-STE) matches finite-difference** at small h (pose[0] translation ratio 0.99;
pose[3] rotation ratio 0.81 at h=1e-4). The larger-h "sign flips" are the
razor-sharp realized curvature (§3), not a twin bug.

## §3 The realized landscape is razor-sharp — the v4c GN already sits at per-pair minima

Realized numpy d_pose sweep of pose[3] at pair 300: base 0.003892, ±1e-3 → ~0.013
(BOTH worse), ±1e-2 → 10–59 (catastrophic). **The v4c numerical-Jacobian GN already
converged each easy/median pair to a SHARP per-pair minimum.** Consequences: (a) the
tangent massively over-predicts off the minimum (the pi2 realization wall, re-
confirmed) → realized acceptance + small steps are mandatory; (b) TTO headroom, if
any, lives on the HARD pairs (where the coarse-FD GN got stuck) and in the JOINT
pose×gain coupling the SEQUENTIAL v4c solve (pose-then-photo) froze out.

## §4 The frozen-scorer factorization reframes "joint" (structural, load-bearing)

`upstream/modules.py:108`: SegNet reads `x[:,-1]` = frame_1 ONLY (MEASURED-
confirmed). pose + (a,b) touch ONLY frame_0 → **d_seg and rate are INVARIANT** to
them. So on the continuous DOF the joint objective `100·d_seg + √(10·d_pose) +
25·rate` reduces to **d_pose** (seg + rate frozen), and per-pair realized ΔS<0 ⇔
per-pair realized d_pose decreases (√(10·mean) monotone). The tractable, clean
coupling the pilot measures is therefore **pose × gain** (both frame_0-only, both
continuous, both differentiable). The genuine 3-way d_seg coupling lives in the
**token** stream (frame_1) — the stage-2 stretch (§6).

## §5 RESULTS — bounded TTO on the ~50-pair stratified subset (worst-35 + 15 controls)

All rows realized through the CPU acceptor (= gate instrument, §2). Full-600
projection = replace the subset pairs' shipped `d_rungB` with the TTO'd `d_final`
in the 600-mean; **d_seg EXACTLY frozen** (frame_1 untouched, §4); **rate
approximately frozen** (the new f16 pose field re-codes through kl1 → archive
bytes may shift ±hundreds of B ≈ ±~0.001 S, measured only at the v4e rebuild);
pose contribution `√(10·mean)` is the measured mover.

| mode | wins/50 | subset d̄ base→final | Δcontribution (subset) | **full-600 ΔS600** | wall | ΔS600/hr |
|---|---|---|---|---|---|---|
| **pose_only** | 29 | 0.10166 → 0.06157 | −0.2236 | **−0.0568** (0.99297→0.93612) | 3.6 min | −0.95 |
| ab_only | 18 | — | −0.0153 | −0.00399 | 2.4 min | — |
| **joint** (best-of {both,pose,ab}/step) | 36 | 0.10166 → 0.06290 | −0.2151 | **−0.0548** (0.99297→0.93821) | 10.0 min | −0.33 |
| **best-of(joint, pose_only)** per pair | — | 0.10166 → 0.05772 | — | **−0.0630** (0.99297→0.93001) | 13.6 min | −0.28 |

**THE HEADLINE (advisory): the analytic-gradient realized-acceptance TTO BEATS
the v4c CONVERGED numerical-Jacobian GN by ~−0.053 S.** Of pose_only's −0.0568,
**94% of the summed Δd (1.889 of 2.004) comes from 21 pairs that were in v4c's
ALREADY-solved-250** (top wins pair 16 0.638→0.284, pair 21 0.394→0.099, pair 71
0.292→0.032 — all "converged"); only 6% (0.115) is completing v4c's parked
unsolved-350. The v4c coarse forward-difference Jacobian (FD_STEPS 0.08…0.0015)
converged to substantially worse per-pair minima than the exact twin gradient
finds; realized acceptance banks the difference. This is precisely the operator's
"update in real time against the frozen scorer gradient (not a proxy)" — the
analytic gradient IS the lever; the proxy-auth wall is cured by the acceptor.

**Coupling (pose × gain): REAL and WIDESPREAD per-pair, MODEST at archive level.**
Per-pair, joint is strictly better than pose_only on **22 pairs vs 5** (equal step
budget), with **36 coupled "both"-family accepts across 24/50 pairs** — the
hardest pair (44) took its wins ONLY via coupled moves and beat pose_only there
(0.596 vs 0.642, superadditive vs the additive prediction). At the archive level,
joint −0.0548 ≈ pose_only −0.0568 (joint burns 3× realized evals/step → time-
capped shallower on 5 high-d pairs that dominate the mean; additive pose⊕ab
prediction −0.0608). The honest attribution: **the analytic gradient itself is
~90% of the win; the pose×gain coupling adds the rest** (best-of −0.0630 vs
pose_only −0.0568 → coupling+ab increment ≈ −0.006 S). The v4e recipe follows:
pose-first TTO, then joint on the pairs where coupled moves accept.

## §6 VERDICT + routing

**FALSIFIER NOT REACHED — TTO PAYS DECISIVELY.** The operator's falsifier was
"realized-accepted steps recover < the v4d rung rate (ΔS/hr below it) → couplings
negligible, close at INSTANCE." Measured ΔS600/hr ≈ **−0.95** (pose_only; −0.28
for the full best-of pipeline) — orders above any v4d rung (QA65/QA66 are
~0.01–0.03 S over 1–2 h). **VERDICT: PROMOTE to the v4e TERMINAL STAGE** — run
analytic-gradient realized-acceptance TTO (pose-first, then joint where coupled
moves accept) on the worst-K pose pairs after every composed build, then rebuild
`pose_warp.stp` and fire the n600 gate (operator-GO). Projected v4e advisory
S ≈ **0.930** (best-of, from 0.993; −0.063), pending the rate re-code (±~0.001)
+ gate confirmation. Headroom note: the subset covered only the worst-50; the
per-pair refinement waterfill (QA69) over the remaining 550 is the natural
extension at the measured ~−0.25 S/hr tail rate.

**Verdict scope (ladder):** FORMULATION-level for the v4c warp-pose vehicle +
frozen scorers. The −0.053 "analytic > numerical GN" win is a property of THIS
pose solver's coarse FD Jacobian; it says the v4c/v4d pose solve should switch to
the twin's analytic gradient regardless of coupling. Advisory everywhere;
`score_claim=false`; the pointer is UNMOVED (v4c is far from 0.191, and this is a
realized per-pair projection with fidelity-law PRE-GATE authority, not a fired
gate row).

**The token→d_seg 3-way stretch (S1 stage-2, deferred with mechanism named):** on
the continuous DOF measured here d_seg is PROVABLY frozen (frame_1-only
factorization, §4), so the genuine 3-way coupling requires optimizing the token
stream (frame_1). The token→frame_1 path (`ddm_tr1_runtime` lotto renderer: seeded
bank regen + mask mods) is differentiable-in-principle (the trainer carries
`token_ste` = round|dither) but needs a torch twin of the lotto renderer, and
tokens move d_seg + rate jointly — a substantial build beyond this bounded pilot.
Routed to a successor (the number that would decide a FULL 3-way v4e). What the
pilot settles: the pose×gain half is measured and paid; tokens are the open leg.

**Routing (defer-at-source):** QA71 → PILOT-POSITIVE (TTO pays −0.063 advisory
best-of; promote to v4e terminal stage). NEW successors: v4e pose-field rebuild +
gate fire (operator-GO n600 slot); the v4c/v4d pose SOLVE should adopt the twin
analytic gradient (retire coarse-FD numerical GN); token→d_seg differentiable
twin of the lotto renderer (the 3-way stretch); QA69 refinement waterfill over
the remaining 550 pairs at the measured tail rate.

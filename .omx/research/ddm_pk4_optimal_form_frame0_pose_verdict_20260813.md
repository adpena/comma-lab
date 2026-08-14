# ddm_pk4 verdict — linear frame-0 pose overlays: MEASURED FAMILY CEILING (2026-08-13)

Run `ddm_pk4_optimal_form_frame0_pose_20260813`, r3 on the torch-CPU pose authority,
rc=0 in 3,977 s, $0 (all local Metal/CPU). Store:
`/Volumes/VertigoDataTier/pact/ddm_pk4_20260813/cpu_authority_run/` (all payloads
retained; r2 MLX artifacts kept for forensics). Status: MEASURED_LOCAL_GATE_COMPLETE ·
score_claim=false · frontier unmoved.

## VERDICT — GATE_FAIL_NO_COMPILE at ALL THREE RUNGS (fail-closed, pre-registered)

Instrument: exact cp135 receiver/R, n64 stratified-random pairs (m96 pose-prefix law
respected), 48 train / 16 heldout with NO fit access, deterministic
(pair_noise_rms_from_exact_repeat = 0.0 ⇒ 2σ threshold = 0; the heldout numbers are
signal, not noise). Fit inputs: only FINAL_DELTA.int32 from train pairs.

| rung | knots | bytes | LOPO modeled Δpose-MSE | HELDOUT realized Δpose-MSE | gate |
|---|---|---|---|---|---|
| 42 | 6 | 43 | +3.469e-6 (positive) | **−1.006e-5 (WORSE)** | FAIL |
| 250 | 40 | 247 | +1.426e-7 (positive) | **0.0 (nothing)** | FAIL |
| 1000 | 165 | 997 | +2.344e-6 (positive) | **−1.107e-5 (WORSE)** | FAIL |

Every rung is LOPO-positive in the modeled space and heldout-negative-or-zero in
reality — the pk3 lesson (23/23 in-sample winners = 0/23 LOO) reproduced at optimal
form on the honest instrument. The overlays cost bytes AND hurt heldout pose; the
byte-economics question never even arises.

**verdict_scope: FORMULATION — linear per-pair frame-0 delta overlays fit from exact
cp135 Jacobians, at 43–997 B.** This family's measured ceiling is: no generalizing
pose reduction at any tested rate. NOT a paradigm verdict on the pose axis — the
open routes are exactly the pre-registered CONDITIONAL-ROUTE: nonlinear/joint descent
(ps135b Leg-A → js1), the pz4 learned pose-gauge QAT (2,000 B pre-proof gated), and
the qs5-proven in-compile Schur compensation (a per-object EXACT solve, not a fitted
model — unaffected by this ceiling).

## INSTRUMENT LAW (MEASURED) — MLX PoseNet forward drifts; torch-CPU is the pose authority

r2 failed its own parity gate at 0.0782 vs retained cp135 vectors. Three-way test on
retained batch inputs: CPU-vs-RETAINED 0.0038 · CPU-vs-MLX 0.0709 (~0.55% rel at pose
RMS 12.97) · MLX-vs-RETAINED 0.071 ⇒ MLX is the drifting instrument. r3 on the CPU
authority: max-abs vs retained = **2.29e-05** (3,400× tighter), parity PASSED.
PoseNet is drift-fragile on every non-authority backend (MPS 23×, MLX 0.55%);
SegNet MLX stays argmax-clean (#855). Cure committed: pk4 pose bank defaults to
torch-CPU (`PK4_POSE_BACKEND=mlx` opt-in, drift documented in-code). Any future pose
Jacobian/GN bank MUST run the torch-CPU authority — a 1e-3-scale pose solve cannot
ride a 7e-2-drifting forward.

## WHY THIS WAS THE RIGHT SPEND

$0 bought: (1) the family ceiling that stops all future linear-overlay pose spends,
(2) the MLX-pose instrument law, (3) a battle-tested fail-closed gate stack — five
gates refused work this session (toy gate · charter lint · ownership contract ·
placeholder law · MLX parity) and every refusal adjudicated CORRECT. A row on a
gate-failing candidate would have been anti-signal at ~$0.16/rung; the gate spent $0
saying no.

## ROUTING (the pose leg after pk4)

Pose→~0 (−0.0083 max) now routes exclusively through: **ps135b Leg-A SOLVE → js1**
(joint/nonlinear, the live owner) · **pz4 QAT** (gated on its 2,000 B pre-proof) ·
qs-family compensation as the pose-tax-canceller for any seg-leg candidate. Seg
remains LOAD-BEARING (−0.0037 minimum even at perfect pose; js8 census closed
frozen-receiver edits at 38/4,314 flips) — the js8 successor (implicit joint
distortion conditioning) fires on ps135 SOLVE + js1 stage-0 per the queued order.

Lane `ddm_pk4_local_metal_measure_20260813` → terminal. Modal ≈ $3.9/$20 (untouched).

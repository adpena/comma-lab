# POSE FRAME0 INVERSE-SOLVE PROBE (#249) — HARVEST LEDGER (n=3 SMOKE; n600 relaunched)

**UTC:** 2026-07-03T08:10Z · **Authority:** `[macOS-CPU advisory / CPU-torch research-signal] NON-PROMOTABLE` ·
**frontier pointer UNMOVED 0.19110** (advisory diagnostic; NOT a contest score). Tool:
`tools/pose_frame0_inverse_solve_probe.py` (committed 3f15daa53). Source JSON: `reports/_smoke_pose_frame0_probe.json`
(n=3). This ledger harvests the SMOKE and records the n600 relaunch so no signal is lost (the agent's draft
verdict lived only in ephemeral scratchpad — the velocity-orphaning trap).

## ❌ CORRECTION 2026-07-03 — RETRACTS the "cheaply realizable" P-F win below (rate-scaling false positive)
**The n=3 "P-F 96×128 coarse carrier nets 0.066 < R1's 0.105 → CHEAPLY_COUNTED_REALIZABLE: true" finding below is a FALSE POSITIVE and is RETRACTED.** The original probe subagent (ran ~2 hr, 126 tool uses) caught it, and I re-derived it independently. The rate term is `25·bytes·n_pairs/37,545,489`; the smoke JSON's `rate_contrib_at_n600_scale` was computed with **n_pairs=3 (the smoke's n), NOT 600** — a **200× undercount** of the per-pair-store rate. At the REAL n600:
| rung | bytes/pair | rate @ n600 (correct) | rate the ledger wrongly used (n=3) | verdict |
|---|---|---|---|---|
| 96×128 | 21,436 | **8.56** | 0.0428 | **RATE-PROHIBITIVE** (net ≈ 8.57 ≫ 0.105) |
| 48×64 | 6,774 | **2.70** | 0.0135 | prohibitive |
| ~6-scalar ξ (pose-space) | ~24 | **~0.010** | — | the ONLY cheap fit — and this IS R1's store-nothing |

**CORRECTED VERDICT: there is NO cheap image-space pose carrier below R1.** A per-pair image store (any coarse-grid free-solved δ) is rate-prohibitive at ×600. The cheap LEGAL pose floor is **pose-space**: generic warp (rule-118 FREE) + a ~6-scalar ξ store — i.e. exactly **R1's store-nothing mechanism (d_pose 0.0011, contribution 0.105)**. P-E's ~1e-8 free frame0 remains an **EXISTENCE PROOF only** (adversarial sub-grey high-freq δ; not decoder-reproducible; storing a per-pair optimized frame0 as a table would be the NO-FAKE #6/#8 eval-hack). The running n600 (pid 55334) will flip the JSON `cheap: true → false` at authority scale. To push pose BELOW R1's 0.105 toward the ancestor's ~0.018, use **pose-space levers ONLY** (warp a REAL keyframe by ξ, ~0.03 rate; or an O(10)-scalar pose-aligned residual) — NEVER image-space per-pair frame0. **Net for the mission: pose gains no new cheap lever; the sub-0.15 fight stays on d_seg + rate, plus validating whether a pose-space keyframe-warp reaches ~0.018 on the witness (unvalidated).** Everything below this banner that asserts a cheap image-space P-F win is SUPERSEDED by this correction.

## ⚠️ AUTHORITY CAVEAT (read first)
The completed, harvestable run is **n=3 contiguous pairs**, NOT n600. **Per "allergic to non-n600 / no toys":
the n=3 is a mechanism-validating SMOKE, not evidence for any decision.**

**Correction (trust-but-verify caught my own wrong diagnosis):** the n600 render was launched by the probe
subagent as an untracked **~2-3hr** background child (pid 8192). Its `.output` went silent and I INITIALLY
mis-diagnosed the render as "dead" — but `pgrep` revealed pid 8192 was **still ALIVE** (a slow 2-3hr CPU
render), and that my first relaunch (pid 8675) had created a **DUPLICATE racing process** (both wrote the same
output JSON + SSD cache). This is the false-stall lesson exactly ([[subagent-false-stall-output-file-vs-real-stall-background-child]]):
`.output` silence ≠ death; VERIFY the child (pgrep) BEFORE relaunching or you spawn a race. Both were killed;
ONE clean harness-tracked run was relaunched (`--n-pairs 600 --pf --pf-generic --fresh`, pid 12028 / bg
`bw7ub6kas`). This ledger is superseded by the n600 aggregates when that run lands (~2-3hr+, longer with the
generic sweep).

## What the SMOKE (n=3) established — mechanism validated
- **NO-FAKE self-check PASS** (`PoseNet(GT)==gt_poses`, pose_err 0.0). Frozen CPU-torch PoseNet + differentiable
  yuv6 patch. NEVER MPS.
- **Quantization-aware solve is the correctness key**: the LM-GN free-frame0 solve rounds to uint8 (STE) in the
  loop. WITHOUT it the min-norm float solution is sub-grey and uint8-roundtrip-destroyed (d_pose 0 → ~0.0013);
  WITH it, the work-res gradient d_pose == the camera-res uint8 FROZEN-authority d_pose (`grad_vs_frozen_gap 0.0`)
  → the solved frame0 **survives the eval roundtrip** (robust, not knife-edge).
- **Warp init is required** (f0=f1 stalls LM-GN on large-forward-motion targets; d_pose_zero mean ~188 → contrib ~44).

## The numbers (n=3 medians — DIRECTIONAL ONLY)
| point | d_pose (median) | contrib √(10·d_pose) | rate contrib @n600 | net S | legal? |
|---|---|---|---|---|---|
| zero-motion (f0=f1) | 194.3 | 44.08 | ~0 | — | (baseline, catastrophic) |
| rigid warp base (in-harness, non-photoreal render) | 4.68 | 6.84 | ~0 | — | legal (rule-118 warp) but HIGH on this render |
| **P-E free frame0 (full-res, frozen authority)** | **2.71e-07** | **0.00165** | rate-prohibitive | — | **ADVERSARIAL** (see firewall) |
| P-F coarse 24×32 | 6.35 | 7.97 | 0.0037 | 7.97 | legal (counted) but too coarse |
| P-F coarse 48×64 | 0.0467 | 0.683 | 0.0135 | 0.697 | legal |
| **P-F coarse 96×128** | **5.2e-05** | **0.0228** | **0.0428** | **0.0657** | **legal (counted per-pair delta)** |
| — R1 store-nothing (trained residual) anchor | 0.0011 | 0.105 | ~0 | 0.105 | legal, shipped mechanism |
| — ancestor RGB anchor | 3.4e-5 | 0.018 | — | — | never witness-validated |

## Honest reconciliation (the JSON verdict vs the agent's prose skeleton disagreed — resolve it, don't cherry-pick)
- **[RETRACTED — see CORRECTION banner at top]** ~~The JSON says `CHEAPLY_COUNTED_REALIZABLE: true` because the
  96×128 coarse carrier nets S 0.0657 < R1's 0.105 — a ~0.039 improvement, with its 21,436 bytes/pair FULLY
  COUNTED (rate 0.0428).~~ **FALSE POSITIVE:** the 0.0428 rate used n=3, not n600; at n600 the rate is 8.56 →
  net ≈ 8.57 ≫ 0.105 → RATE-PROHIBITIVE. The image-space coarse carrier does NOT beat R1. It is still *legal*
  (counted, not smuggled), just far too expensive.
- **The agent's prose skeleton is more conservative** — "the cheap legal pose floor stays ~0.0011 (contrib 0.105);
  sub-0.0011 needs image-space frame0 = adversarial/rate-prohibitive."
- **Both are true at different operating points, and here is the synthesis:**
  1. **P-E existence proof (SOLID):** because the PoseNet-Jacobian on frame0 is full-rank in the 6 pose dims, a
     free frame0 can hit ANY pose target → d_pose → ~0 (2.71e-07, survives uint8). **Pose is inverse-solvable to
     ~0 in principle.** This is a real, deep result: pose need not bound the witness budget.
  2. **P-E is NOT shippable (FIREWALL):** the full-res 2.71e-07 floor is reached by a tiny (sub-grey) but
     SPATIALLY-PRECISE high-frequency per-pair delta. It is NOT decoder-reproducible and per-pair full-res image
     storage is rate-prohibitive. Storing a per-pair optimized frame0 as "code" would be the **eval-hack fake
     (NO-FAKE #6/#8, EdgeBench App C)**. DO NOT build an inference-time adversarial-frame0 carrier.
  3. **P-F is the real question, and the SMOKE is promising but INCOMPLETE:** the 96×128 coarse carrier
     (net 0.066 < 0.105) is a LEGAL per-pair counted delta — a directional ~0.04 win. BUT the tool only ran
     `coarse_free_solve_SMOOTH_cheaply_counted` (a per-pair FREE solve at coarse res). The truly legal, cheapest
     realization — a **decoder-reproducible GENERIC basis (DCT / low-rank) over the GENERIC warp base + a small
     counted delta** (the `--pf-generic` path) — was **NOT run** (empty in the JSON). Until that runs at n600, we
     do not know the CHEAPEST legal pose floor; the 96×128-per-pair-delta is an upper bound on cost, not the optimum.
- **Affine ξ↔PoseNet-6 calibration R² = −0.215 (NEGATIVE)** → the physical-ξ→PoseNet-6 map on the render-warp is
  only weakly affine. This does NOT affect P-E/P-F (they do a direct free solve), but it WEAKENS the P-B
  "store 6 affine scalars + FiLM" story — flag P-B as needing its own re-derivation, not an affine assumption.

## Verdict → recommendation to the #250 grand-council optimal-form symposium
- **P-E: PROCEED as an existence proof** — pose is inverse-solvable to ~0; the witness pose budget is NOT
  fundamentally floored at R1's 0.0011. This reframes pose from "solved-and-done" to "cheaply-reducible-below-R1,
  pending the legal realization."
- **P-F: the DECISIVE open question** — does a decoder-reproducible generic basis (DCT/low-rank) over the generic
  warp base reach low d_pose at << 21KB/pair? If yes, pose contribution drops from R1's 0.105 toward ~0.02–0.03
  NET of a small counted rate — a real ~0.04–0.08 S improvement that helps make sub-0.15 feasible. If no (only
  per-pair full storage reaches it), the practical legal floor stays near R1's 0.105 and pose stays a
  budget item. **n600 `--pf --pf-generic` resolves this.**
- **FIREWALL (binding):** ship only (generic warp base, rule-118 FREE) + (generic-basis counted delta). Never a
  per-pair optimized-frame0 table-as-code.

## n600 relaunch (this ledger's action)
Relaunched `tools/pose_frame0_inverse_solve_probe.py --n-pairs 600 --pf --pf-generic --fresh` as a SINGLE
harness-tracked background job (bg `bw7ub6kas` / pid 12028) after killing the two racing processes (subagent's
pid 8192 + my first relaunch pid 8675). `--fresh` guarantees no stale partial-JSONL/cache residue from the killed
pair. `--pf-generic` adds the decoder-reproducible generic-basis (DCT/low-rank) sweep the smoke SKIPPED — the
firewall-decisive measurement. Output → `reports/pose_frame0_inverse_solve_probe.json`. Runtime ~2-3hr+ (light
CPU, ~96% RAM free — machine-safe). When it lands, supersede this ledger's DIRECTIONAL numbers with the n600
aggregates and hand to #250. Nothing heavy/GPU launched (CONTAINMENT holds).

**Cross-refs:** [[project_pose_solved_screw_twist_dual_use_film_conditioned_sidecar_20260701]] ·
[[subagent_false_stall_output_file_vs_real_stall_background_child_20260702]] (the false-stall this hit) ·
DAG FEED-poseladder · #248 ladder · #250 symposium · #238 byte-close.

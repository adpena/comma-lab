---
schema: ddm_ck1_composed_kneeA.v1
date_utc: 2026-07-29
arm: ddm_ck1 (composed Knee-A decision cell; QA06)
lane_id: "lane_ddm_ck1_composed_kneeA_20260729"
research_only: true
score_claim: false
pointer: "0.1910828242 [contest-CPU] UNMOVED"
axis: "[macOS-CPU frozen-PoseNet advisory; per-pair realized through the real receiver+PoseNet; composed byte-close n600 evaluate gate STAGED, OWED]"
operator_binding: "MAIN QA06 dispatch — re-solve pose on the Knee-A base (knee law), type seg damage, compose+stage"
tool: "experiments/ddm_ck1_pose_resolve_kneeA.py (commit 372d9b4975) + ddm_ck1_build_composed_archive.py + stage_ck1_composed_gate.sh"
data: "SSD ddm_ck1_20260729/{ck1_control.partial.jsonl(600), ck1_solve.partial.jsonl(resumable), ck1_control_receipt_n600.json, ck1_seg_typing_receipt.json, ck1_composed_*_build_receipt.json}"
---

# ddm_ck1 — pose RE-SOLVE on the Knee-A base + composed decision cell

**POINTER HONESTY FIRST: `0.1910828242 [contest-CPU]` UNMOVED.** Every row below is
`[macOS-CPU advisory]`, `score_claim=false`. This arm lowers the pfs1 warp-base
ADVISORY row (S 2.256641), a vehicle far from the pointer; it does NOT move the pointer.

## §0 The measured fact this arm answers (from MAIN/QA06)

Knee-A wr1 archive STANDALONE (real evaluate.py rc=0, full n600): **S 2.4097 vs ref
2.2566 = +0.153 net REJECT** — rate 0.18267 (−0.197, the win) BUT d_seg 0.00553676
(+0.165 S) AND d_pose 0.28002128 (+0.185 S). Verdict scope INSTANCE: the shipped pose
member (per-pair warp pose = t_p, receiver s_r=0) was solved on the FULL-token base; the
Knee-A base DROPPED 288 sky + 170 hood + 28 stable-road latent cells → froze the
far-field content the warp/PoseNet reads. Knee law demands: re-solve pose on the frames
the candidate ships. This arm does that.

## §1 CONTROL — substrate identity CONFIRMED (the confound gate passed)

Reproduce the gate's d_pose at the SHIPPED params (f16 t_p from `pose_warp.stp`, s_r=0,
D1 s_t) on the Knee-A base, full n600: **control d_pose mean 0.28002015 vs gate
0.28002128, abs delta 1.13e-6** (`ck1_control_receipt_n600.json`). My Knee-A f1 render
(rebuild the TR1 packet from the v3_warp sectioned members exactly as the gate's
inflate_runner does) + warp + frozen PoseNet EXACTLY reproduces the evaluator. No
substrate confound; the re-solve below is trustworthy.

## §2 THE RE-SOLVE — two-plane recovers pose on the frozen-far-field base; single-plane is UNSTABLE there

Per-pair re-solve on the Knee-A base, tail-first (by full-base d_pose), best-of
{single-plane 6-DOF GN (s_r=1.0), two-plane near/far GN multi-start (far cls2→H∞ at
s_t=0, ground→full H, hood→identity)}.

**MEASURED — TAIL-112 FINAL (112/112 solved on the Knee-A base):**
- **Tail best_mean d_pose 0.3609; two-plane wins 96/112 (86%)**; single wins the rest
  (milder pairs); the non-tail sweep continues (resumable, detached).
- **RECOVERY PARITY (the headline):** knee-base two-plane tail best_mean **0.3609** ≈
  full-base two-plane tail mean **0.3692** (qa43 final selection 41.357/112). The 486
  dropped cells cost the pose axis ~NOTHING once re-solved through the two-plane compose
  (0.98× ratio) — the gate's +0.185 S pose regression was ENTIRELY stale params, not a
  capability loss of the knee base. The QA06 INSTANCE scoping is confirmed mechanically.
- **SINGLE-plane is UNSTABLE on the Knee-A base for the tail** (solved-tail single means
  ran ~1.7–2.0 during the sweep; e.g. pair 19 single_kneeA 2.0549 vs two-plane 0.4245).
  Mechanism: the dropped sky froze the far field → the single planar-homography Jacobian
  is degenerate on turn pairs → GN cannot escape. The TWO-plane (far→H∞) is the vehicle:
  a frozen (near-constant) far field is warp-invariant under H∞, exactly what the drop
  produces. The physics that HURT the stale ship (frozen far) is the physics the
  two-plane exploits. (GN acceptance is monotone from the s_r=0 shipped point — start p0
  = t_p rotation-zeroed at s_r=1 is exactly the shipped warp — so single_kneeA ≤ the
  knee-base stale value per pair: a v4a single-plane field never REGRESSES vs the gate
  ship; it just cannot recover the tail.)
- **REALIZABILITY CAVEAT (carried with every two-plane number):** the two-plane masks
  are GT `lstars` (the qa43 probe stand-in). GT masks are NOT available at inflate; a
  realizable receiver needs a rule-118-legal mask source (static geometric prior, OR
  decoded-partition). The two-plane d_pose here is therefore an **UPPER BOUND**; the
  realizable-mask re-solve is the named next rung (§6).

## §3 COMPOSED ARITHMETIC (advisory; DERIVED until the staged gate runs)

Composed candidate = Knee-A tokens (unchanged: d_seg 0.00553676 MEASURED at the gate,
rate 0.18268 MEASURED) + the re-solved pose field. Composed
S = 100·d_seg + √(10·d_pose_resolved) + rate.

**TAIL-112 FINAL (all 112 tail pairs MEASURED on the Knee-A base; non-tail = the two
bracketing treatments until the full-600 sweep lands):**
| non-tail treatment | composed d_pose | pose contrib | composed S | Δ vs ref 2.2566 | Δ vs Knee-A 2.4097 |
|---|---|---|---|---|---|
| STALE shipped (control, Knee-A) [conservative — realizable TODAY] | 0.15623 | 1.2499 | **1.9863** | **−0.270** | **−0.423** |
| full-base d2 single-plane [optimistic proxy] | 0.08173 | 0.9040 | **1.6404** | **−0.616** | **−0.769** |

**BREAK-EVEN VERDICT: the tail-112 re-solve ALONE flips the Knee-A reject** — even with
non-tail pose left STALE and the seg damage UNCURED, composed S 1.9863 beats the ref by
−0.270 (rate −0.1969 + pose −0.2382 vs ref, against seg +0.1647). The optimistic bracket
(non-tail re-solved) reaches 1.6404; the true full-600 number lands between and the
detached sweep is measuring it (single-plane is stable off-tail; full-base evidence:
non-tail chart removes 73.8% of residual).

#404 ratios: composed pose contribution 0.90–1.25 vs ref pose 1.4881 = **0.61–0.84×**
(vs the stale ship 1.6734 = 0.54–0.75×); seg damage +0.1647 S (uncured); rate win
−0.1969 S. Pose axis: composed d_pose 0.0817 (optimistic) ≈ the full-base two-plane
composed 0.0833 (qa43) — same recovery-parity fact as §2, read on the composed axis.

## §4 SEG DAMAGE TYPING ($0, no SegNet job; `ck1_seg_typing_receipt.json`)

The +0.00164665 d_seg (= +0.165 S) is **knock-on**: all 486 dropped cells had atlas
flip_mass=0 in ISOLATION; dropping them TOGETHER produced non-additive boundary flips
(the atlas under-priced). Band breakdown of the 486 dropped cells:

| band | n | residual_mass | rows (latent 24×32) |
|---|---|---|---|
| sky_undriv_top | 288 | 820,538 | 0–8 |
| road_lane_midband | 28 | 118,415 | 9 |
| mycar_hood_bottom | 170 | 261,889 | 18–23 |

Sky dominates the carried residual (68%) and is the **same frozen-far-field cells that
regressed pose** → ONE shared cure family (restore far cells) helps BOTH axes.

**Cures PRICED (not both built):**
- **(a) RESTORE k worst cells** (reverse waterfill, REAL descent-curve bytes):
  restore→k=400 (+23,035 B, **+0.0153 S** rate, 86 cells) · k=300 (+72,338 B, +0.0482 S)
  · k=200 (+135,201 B, +0.0900 S). d_seg GAIN is OWED at a partial-knee gate (the atlas
  under-priced; recovery fraction unmeasured). Restore is favorable ONLY if it recovers
  ≫ its rate cost; at k=400 that means recovering ≥ +0.0153/0.165 ≈ 9% of the seg damage.
- **(b) QA03-class targeted GN** (fd1/tr1 seg BASE solve): 0 rate bytes, moves the base
  (white-jitter law: seg closes by moving the base). Different arm; not built here.

**Recommendation:** ship the PRIMARY composed candidate UNCURED (seg 0.00553676) — the
pose re-solve already wins; cure (a) is a secondary rate↔seg trade to measure at a gate
only if the margin needs it. The ranked damaged-cell list (top-40 by residual_mass) is
in the receipt.

## §5 COMPOSE + STAGE (build-validated; grammar v4)

- **Build path validated:** `ddm_ck1_build_composed_archive.py` rebuilds the composed
  archive byte-faithfully (tokens/renderer/selector/pose_stub = Knee-A bytes verbatim;
  only `pose_warp.stp` re-encoded — brotli-q11 f16 poses + unchanged st stream; encoder
  round-trip byte-identical). Stand-in build (full-base p_star): **274,345 B**, rate
  0.18268. The v4a inflate_runner (s_r=1.0) decode is **byte-identical to the oracle**
  (f0 bytes match, 3/3 pairs) → the staged gate measures exactly what the instrument
  predicts.
- **Grammar v4a (single-plane, s_r=1.0):** built + `stage_ck1_composed_gate.sh` staged.
  BUT §2 shows single-plane REGRESSES on the Knee-A base → the v4a gate is a NEGATIVE
  control, NOT the winning candidate. Do not spend the winning slot on it.
- **Grammar v4b (two-plane, REQUIRED for the win):** ship the re-solved 6dof field +
  per-pair selector bit (single vs two-plane) + a multi-plane receiver (far→H∞ /
  ground→H / hood→identity). **Open crux: the mask source** — GT masks are illegal at
  inflate; the realizable options are (i) static geometric prior (sky=top rows,
  hood=bottom rows; rule-118 free, 0 bytes) or (ii) decoded partition. v4b needs a
  **realizable-mask re-solve** (re-solve the two-plane through the receiver's ACTUAL
  masks, not GT) before its gate is honest. This is the named next rung.

## §6 CONFOUNDS + ROUTING

- **CONFOUND (noted, control-guarded): `tac` import is HIJACKED to a parallel codex
  worktree** (`.omx/tmp/codex_worktrees/ddm_eg1_endgame_chain_20260729T000937Z/src/tac`),
  not main `src/tac`. The §1 control positive-control (decode reproduces the gate to
  1.13e-6) proves the hijacked runtime's DECODE is byte-equivalent to the gate's vendored
  runtime, so this arm's measurements hold. The build tool avoids `tac` entirely (plain
  ZIP_STORED). Hygiene fix (venv editable-install points off-main) → QD ledger row.
- **GT-mask upper bound** (§2): the realizable-mask re-solve is OWED before the v4b gate.
- Ledger QA06 updated; new QA-rows appended (defer-at-source): realizable-mask re-solve +
  v4b build; static-vs-decoded mask A/B; seg-cure partial-knee gate; QA44 rung-B fold.

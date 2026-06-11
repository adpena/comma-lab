# DECISIVE: smaller-basis-by-rate REFUTED at d_seg → the paid n600 it gated is a FAKE spend → pivot to the evaluator-action waterfiller (2026-06-11)

**Authority:** the capacity test is `[macOS-CPU advisory]` (torch-CPU exact d_seg through frozen
`modules.py` SegNet) / `[macOS-MLX research-signal]`, NON-PROMOTABLE. Frontier UNMOVED **0.19109982
[contest-CPU], 177,169 B — ABOVE T_1, GOAL UNSATISFIED.** This is a measured negative that CLOSES a
path and SAVES a fake paid spend. $0, no MPS, no dispatch. Artifact:
`experiments/results/lane_cool_chic_score_aware_basis_20260611/latent_heavy_isolation.{json,log}`,
driver `b76cb9f10`.

## The decisive measurement (the latent-heavy vs synth-heavy isolation, matched-budget)

n8, 35ep, hinge curriculum, out_hw fixed 96×128 (render-res NOT a confound), torch-CPU exact EMA d_seg
(EMA vs live agree ≤2e-4 → the EMA-shadow-lag artifact is GONE, this is real):

| arm | grids | synth_h | exact d_seg | latent B | weight B | latent_frac |
|---|---|---|---|---|---|---|
| REF synth-heavy | 5×3 | 48 | 0.0189 | 665 | 2116 | 0.24 |
| L0 compact-both | 4×2 | 16 | 0.0226 | 422 | 618 | 0.41 |
| **L2 latent-heavy** | 6×4 | 16 | **0.0140** | 2127 | 1290 | 0.62 |
| L3 latent-max | 6×6 | 16 | 0.0151 | 3384 | 1914 | 0.64 |

## The three decisive answers

1. **Does latent-heavy reach the corrected bar (d_seg ~0.0011–0.0017) where synth-heavy couldn't? NO.**
   L2's best is **0.0140** — ~9–13× ABOVE the bar. **L3 with 64% more latent bytes got WORSE (0.0151)** —
   the capacity-bound signature (more capacity → worse, not flat = saturation, not under-training).
2. **The operator's hypothesis (capacity belongs in cheap latents, not synth) is DIRECTIONALLY
   CONFIRMED** — latent capacity is meaningfully more byte-efficient (compact→L2: d_seg −38% for +2.4KB
   latent, FEWER weight bytes; L2 beats REF at comparable bytes). **But it does NOT change the verdict:**
   the latent axis saturates at ~0.014, an order of magnitude short.
3. **d_seg, not rate, is binding.** At d_seg 0.014 the seg term alone is 100·0.014 = **1.40**; even with
   pose collapsed (+0.055) and the full rate win (~1–5KB), **S ≈ 1.46** vs the 0.191 frontier. Cheap
   latents do NOT rescue a 0.014 d_seg.

## The sharpened CRUX (why this closes the smaller-basis-by-rate thesis)

Decompose the frontier: `S = 0.19110 = 100·d_seg(~0.056) + sqrt(10·d_pose)(~0.017) + rate(0.11797)`.
**Rate is 62% of the score AND at the 7.999-bits/byte entropy floor (recode-closed).** The only way to
cut rate is FEWER decoder bytes = a SMALLER decoder. But:
- A smaller (Cool-Chic-class compact) decoder does NOT hold the d_seg basin — MEASURED capacity wall at
  ~0.014 (~25× the frontier's 5.6e-4).
- A frontier-class decoder that DOES hold the basin costs ≈ frontier bytes (the current frontier *is* a
  ~178K conv-HNeRV at the L20–L32 stack = 177KB) → **no rate win from retraining at the right capacity.**

⟹ **You cannot simultaneously shrink the decoder AND hold the d_seg basin** with this architecture class,
and you cannot recode the bytes below the entropy floor. **The "smaller learned basis → sub-frontier by
RATE" thesis (tasks #67/#71/#78/#91) is MEASURED-REFUTED at the d_seg axis.** (Residual the agent
flagged: a *qualitatively different* smaller synth — not bigger Cool-Chic grids — might break it; the
burden of proof is now high; STOP sweeping Cool-Chic capacity knobs.)

## NO-FAKE consequence: the paid n600 this de-risk gated is now a FAKE spend (do NOT fire it)

Task #90 ("capstone under-training de-risk → paid n600 PR95-scale, THE pointer-mover") was premised on a
smaller-basis reaching sub-frontier by rate. **That premise is refuted.** Firing a paid n600 at
Cool-Chic / base_ch=20 / compact capacity would land at the MEASURED d_seg wall (~0.014, S≈1.46) and
would NOT move the pointer — a means-hoarding spend narrated as progress = a NO-FAKE violation
("surrogate-optimized-but-not-exact-authority-verified" / means-as-ends). **The capacity test SAVED that
spend.** This is the rule the operator just invoked ("beware fake implementations"), working as intended:
a $0 local measurement refused a fake paid row.

#90 is REFRAMED (not fired): a frontier-class paid retrain is justified ONLY if it credibly BEATS PR95's
d_seg/d_pose at ≤ frontier bytes (a "beat the tuned frontier by better training" bet — real but hard and
uncertain, NOT the rate slam-dunk that was hoped). That bet must be scoped separately, not fired now.

## The PIVOT (one crisp wall-verdict, then move — per the GOAL)

The smaller-basis path walls. The LIVE paths to a lower EXACT score, ranked by EV:

1. **Evaluator-action waterfiller / null-space composition on the CURRENT frontier (#30 + #47/#48).**
   This is where the last real win came from (PR110 payload-entropy recode, −0.000883, an
   evaluator-invisible rate shave). It operates on the frontier archive that ALREADY holds the d_seg
   basin — it does NOT need the capacity wall broken. It shaves the exact score via argmax-equivalent /
   invisible / commutator-aware atom edits. **This is the highest-EV remaining live path** — and it is
   exactly where the RL lane attaches (RL/MCTS upgrades #30's "commutator-aware greedy" atom-selection,
   reward = exact compliant S). Honest caveat: the *easy* waterfiller win (entropy recode) is banked;
   the remaining atoms are harder.
2. **Score-aware weight re-quant of the frontier decoder (#69)** — sensitivity-weighted per-tensor bit
   allocation below the uniform entropy floor, IF residual headroom exists (re-check #69's verdict).
3. **Frontier-class paid retrain that beats PR95's distortion (#90 reframed)** — uncertain, paid, must
   beat a tuned frontier.

## Honest bottom line (non-sycophantic)

The central session fork is RESOLVED, against us: capacity-in-cheap-latents is the *right call for
byte-efficiency* but is *capacity-walled for d_seg* — Cool-Chic is rate-only, and rate is already
floored on the frontier, so the smaller-basis route to sub-0.15 is closed. The frontier is genuinely
hard to move now: rate (the biggest lever) is locked behind the capacity wall, seg/pose are near the
basin. The pivot is to the evaluator-action waterfiller on the current frontier (the path that produced
the last win), with RL as the principled selection upgrade. **Frontier UNMOVED 0.19110; this turn moved
no row — it refuted a path, saved a fake spend, and aimed the next unit at the one live exact-row
target.**

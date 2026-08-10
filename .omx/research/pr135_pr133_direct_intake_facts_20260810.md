# PR135 + PR133 direct-intake facts (MAIN, 2026-08-10) — THE NEW BAR, decomposed

**Operator:** "Analyze all signal in PR135 and harvest all related signal. That is the
new bar that we have to beat." + "We can use everything off of the shelf, and then
everything from our entire corpus and all online research to continue iterating and
optimizing everything." (grant broadened; memory
`everything-off-the-shelf-full-corpus-online-research-20260810`).

## 1. The bar, byte-exact, decomposed — THE GAP IS POSE

PR #135 `semantic-pose-HPAC_CPR1_polished` (author codexblack, head codex/f26-submission):
**S = 0.16226842169958583 @ 186,724 B** — d_pose 0.00000688439, d_seg 0.000296394
(self-published report.txt; leaderboard lists 0.162 rank 1).

Our live anchor (lc2): S = 0.16959899569230852 @ 187,226 B. Gap = **+0.00733057**:

| axis | PR135 | lc2 | Δ | share |
|---|---|---|---|---|
| seg | 0.029639 | 0.029662 | +0.000023 | 0.3% |
| pose | 0.008297 | 0.015271 | **+0.006974** | **95.1%** |
| rate | 0.124314 | 0.124666 | +0.000352 | 4.8% |

My pi135 charter prediction said rate-dominant; measured reality is POSE-dominant.
The prediction was wrong on the axis and right on "no seg delta." Recorded honestly.

## 2. The lineage mechanism (from the PR bodies — full text pulled)

- **PR #133 `cpr1_cbq_matched8` (JasonMo123), 190,212 B, pose 8.96e-6, S 0.165780:**
  pose-carrier basis atoms 2, 5, 9 dropped 5-bit → 4-bit, then the int12
  coefficients RE-SOLVED against PoseNet with Jacobian-guided coordinate passes
  (8 exact full-600 passes). Quantize-then-compensate through the exact scorer.
  Key fact: compensation lands BELOW the PR130 baseline pose (2.33e-5 → 8.96e-6)
  — PR130's coefficients were never at their PoseNet optimum.
- **PR #135, pose 6.88e-6, 186,724 B:** stacks PR133's polish + a Jacobian-guided
  discrete carrier search + frame-0 edits (pose-better, seg-neutral) + a K=8
  selector + ~4.3KB of "small lossless changes in many areas" for rate.
  4,598 added lines, all public: runtime/entropy/{adaptive_ans, coefficient_ar1_codec,
  coefficient_predictor, rc64(+C backend), renderer_weight_codec}.py,
  runtime/{frame0_selector, ihs2, ihs2_gate_a, hpac_inference, carrier_repack,
  residual_archive}.py + compress.sh (reproducible archive prep, intended to merge).
- **Related signal named in the PR:** the author's full experiment repo
  github.com/codexblack/CommaVideoCompressionChallenge_ExperimentBook — harvest target.
- **Context flag:** the PR sat in a coding-agents/LLM-policy dispute (Yousfi asked
  for precise attribution; author revised + re-opened). The MEASURED number stands
  as the bar regardless; the policy thread is signal for OUR submission path.

## 3. Custody (payloads KEPT, SSD tier)

- `/Volumes/VertigoDataTier/pact/pr135_intake_20260810/pr135/archive.zip` —
  186,724 B, sha256 `12cf5d71a94065184f097c3e40dfe9f1db8402a1a76a80efc76a6956fe1e4004` ✓ matches published.
- `/Volumes/VertigoDataTier/pact/pr135_intake_20260810/pr133/archive.zip` —
  190,212 B, sha256 `051baf408f57fae3b343d6ee218ab963d070b3935ceb0b2f412c93a53cf3fab0` ✓ matches published.

## 4. The corpus resonance (why this is OUR mathematics, shipped by others first)

Quantize-then-compensate-through-the-exact-scorer = our fd1/fd2 GN/CG-in-description-
coordinates engines, v17/v19 realized-acceptance solves, pk2's pose-carrier
representation attack (23,384 B / 0.0155704 S), su2/QA43 pose solves, and #976's
frame-0 free-pose-actuator existence proof (they SHIPPED frame-0 edits). We hold the
machinery; they hold the measured rows. Off-the-shelf grant now covers their code.

## 5. Re-aimed roadmap (zero-gravitational-pull law)

Reproduce the CURRENT rank-1 base, then iterate on THAT base — now PR135, not PR130:
1. Reproduce PR135 end-to-end (archive in custody; decode + exact CUDA row on the
   locked env — same chain that bought the lc2 rows).
2. Race OUR lc2-style ANS/constriction section recode ON PR135's sections (our
   −3,826 B was measured on PR130's original sections; overlap with their −4.3KB
   unknown until the arms diff the section maps).
3. Run OUR pose machinery (fd-family, pk2) on THEIR base — their own table proves
   the coefficients still had 3.4× pose headroom at PR130; measure what remains at 6.88e-6.
4. Compose arithmetic: lc2 recode surviving on the PR135 base + any residual pose
   headroom → first sub-0.162 candidate. Every claim byte-closed + exact-eval'd.

Consumers: ddm_pi135 (depth, LIVE) · ddm_pi136 (breadth incl. PR133 already
partially covered here, LIVE) · ddm_ah2 (arm harvest re-ranked vs 0.162268, LIVE) ·
#984 composed campaign · #995 roadmap (re-aim to PR135 base).

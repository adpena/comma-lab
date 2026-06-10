# PR110++ R3 — on-host per-mode pose table + the GT-decode apples-to-apples bug

**Subagent:** `pr110pp_r3_onhost_mode_table_20260610` · UTC 2026-06-10.
**Lane:** `lane_pr110pp_r3_onhost_mode_table_20260610`.
**Operator authorization:** PRE-AUTHORIZED ("all approved on operator end" — same lineage as R1).
**Mission (R1's named reactivation criterion):** R1 KILLed both R2 candidates because the
macOS-CPU per-mode pose ordering did not transfer to the Linux-x86_64 contest-CPU host. R1's
reactivation criterion was: *"score the 16-mode per-pair pose table DIRECTLY on the Linux-x86_64
contest-CPU host, then choose argmin-pose on THAT table."* R3 executes exactly that — build the
per-mode pose table ON the contest host, derive the corrected selector, re-eval.

**Axis discipline:** the per-mode pose table is `[contest-CPU host substrate]` (computed on the
Linux-x86_64 Modal CPU container, 1:1 with the contest GHA CPU runner, via the EXACT
`upstream/modules.py` PoseNet + the EXACT `upstream/frame_utils.yuv420_to_rgb` GT decode). It is
NOT a score claim (a score claim requires `upstream/evaluate.py` on the byte-closed archive). The
final paired contest-CPU eval is the score authority. CUDA never touched (single-axis CPU per R1's
waiver — the 0.19199 ranking axis; CUDA is a separate promotion gate).

---

## 0. THE HEADLINE MID-MISSION DISCOVERY — the GT-decode apples-to-apples bug

R3's first on-host table (v4) reproduced R2's machinery faithfully **including R2's GT decode**:
`av.frame.to_ndarray(format="rgb24")`. The v4 table came back with **591/600 pairs "improvable"**
and an **incumbent per-pair pose mean of 3.31e-3** — but the R1 frontier eval's
`avg_posenet_dist = 2.943e-05`. A **~100× absolute-pose discrepancy.**

Root cause (CLAUDE.md "Apples-to-apples evidence discipline" + "decoded-state parity is not frame
parity"): the contest GT decode is `frame_utils.yuv420_to_rgb` (BT.601 limited-range + bilinear
chroma upsampling, "matches nvdec"), NOT PyAV's `to_ndarray(format="rgb24")` (libswscale, a
different YUV→RGB conversion). Both R2's macOS table AND R3's first on-host table scored the comp
frames against a **non-contest ground truth**. The per-mode pose values were ~100× inflated and the
per-pair "improvements" were largely GT-mismatch artifacts.

This is a SECOND substrate-invalidating bug on top of R1's host-FP-drift finding. R1 proved the
macOS→Linux host axis mis-ranks; R3 additionally proves the **GT decode itself was wrong on both
substrates.** The corrected substrate requires BOTH: the contest host AND the contest GT decode.

**Fix (R3 v5):** import `frame_utils.yuv420_to_rgb` (the contest's exact function) for the GT decode.
A 2-pair local smoke with the corrected GT immediately confirmed: pair-0 incumbent pose dropped from
8.81e-5 (rgb24) to **2.30e-5** (contest GT) — now at the same O(1e-5) scale as the frontier
`avg_posenet_dist = 2.943e-5` — AND the per-mode argmin gain collapsed to **~7e-10** (negligible).

The comp render path was verified byte-faithful to `inflate.py` (upscale→channel-postproc→clamp→round,
then `apply_selector_to_frames` = `apply_frame0_mode` + clamp/round; lines 603-610 of the frontier
`inflate.py`). The ONLY divergence was the GT decode.

---

## 1. The v5 on-host per-mode pose table (contest GT + contest host + contest scorer)

**Method.** Inside the proven R1 Linux-x86_64 CPU container (same `base_image`: same apt/pip/torch-CPU
layers): parse the frontier member-x, render the receiver comp frames per pair (DQS1 + bicubic upscale
+ −1 channel postproc + clamp/round, byte-faithful to `inflate.py`), apply each of the 16 frame-0
modes to frame-0, and score the 16-mode batch through the EXACT `upstream/modules.py` PoseNet on CPU.
GT decoded via `frame_utils.yuv420_to_rgb` (contest-exact). seg-blindness re-verified on the host via
the full DistortionNet SegNet head. A reproducibility twin scores each of 64 pairs' incumbent mode
TWICE to measure the host's per-pair pose noise floor.

**Dispatch:** `experiments/results/pr110pp_r3_onhost_mode_table_20260610/onhost_pose_table_app.py`
(Modal app `comma-pr110pp-r3-onhost-pose-table`, cpu=8, mem=16GB), call_id `fc-01KTQKSA9NA7KGWPV7D02TVC72`.
Cost: ~$0.33 est (under the $5 STOP gate). Reproducibility twin noise floor: **0.0 (bit-exact host
kernels)** — the contest-CPU host's PoseNet+render is perfectly self-consistent, so the R1 failure was
purely cross-host FP drift + the GT bug, NOT intra-host noise. Any above-zero on-host gain is admissible.

**Headline (v5, the corrected substrate; `onhost_pose_table.json`, sha-anchored to frontier
`b7106c9b…`):**

- `segnet_blindness_spotcheck_verified = true` (d_seg=0 exact for every frame-0 mode on the host
  spot-check; `segnet_nonzero_findings == []`).
- **incumbent per-pair pose mean = 2.94313e-05** — matches the frontier `avg_posenet_dist =
  2.943e-05` essentially exactly. The substrate is now apples-to-apples valid (vs the wrong-GT v4's
  3.31e-3, a 100× error).
- **Only 7 of 600 pairs are improvable** (on-host argmin-pose < incumbent), vs the wrong-GT v4's
  **591/600**. The wrong GT manufactured ~98% of the apparent improvability.
- **Total on-host pose gain over the 7 pairs = 8.483e-06**; max single-pair gain 4.74e-6 (pair 2).
  The 7 pairs + gains:

  | pair | incumbent mode | inc pose | argmin mode | argmin pose | gain |
  |---:|---|---:|---|---:|---:|
  | 2 | rgb_bias_m2_p1_p1 | 8.94e-6 | rgb_bias_m4_p2_p2 | 4.20e-6 | 4.74e-6 |
  | 540 | rgb_bias_p2_m1_m1 | 5.27e-6 | rgb_bias_p4_m2_m2 | 2.68e-6 | 2.59e-6 |
  | 546 | luma_bias_-1 | 1.12e-5 | rgb_bias_p4_m2_m2 | 1.01e-5 | 1.10e-6 |
  | 431 | rgb_bias_p4_m2_m2 | 3.20e-5 | rgb_bias_p0_m2_p2 | 3.19e-5 | 4.57e-8 |
  | 229 | blue_chroma_amp_1 | 8.06e-6 | blue_chroma_amp_3 | 8.05e-6 | 1.05e-8 |
  | 436 | blue_chroma_amp_1 | 1.52e-5 | none | 1.52e-5 | 9.87e-10 |
  | 396 | blue_chroma_amp_1 | 3.36e-5 | none | 3.36e-5 | 5.09e-11 |

- **Predicted score delta of switching all 7:** pose_avg drops 1.414e-08 (2.94313e-05 → 2.94172e-05);
  the nonlinear pose contribution `√(10·pose_avg)` drops by **−4.12e-06** (0.01715556 → 0.01715143);
  selector bytes +2 (220→222) → rate term `+25·2/37.5M = +1.33e-06`. **Net predicted ΔS = −2.79e-06.**
- Reproducibility-twin noise floor = **0.0** (bit-exact host kernels): the contest-CPU host's
  PoseNet+render is perfectly self-consistent. So the R1 failure was 100% cross-host FP drift + the
  GT-decode bug, NOT intra-host noise. Every above-zero on-host gain is admissible (no ties to gate).

**The decisive finding: the per-pair selector lever is essentially EXHAUSTED at the frontier.** On the
corrected (contest-GT + contest-host) substrate, the incumbent FECa selector is already near-optimal;
the maximum achievable pose improvement is 7 pairs totaling 8.5e-6 pose, worth a net ΔS of −2.79e-6 —
which is BELOW the eval reproducibility floor (R1's fresh frontier baseline matched the archived
0.19199 within 4.66e-6). The R1/R2 "8.27e-2 improvable" signal was 100% a wrong-GT artifact.

---

## 2. The corrected selector + paired contest-CPU eval

**The corrected selector candidate** (`candidate/candidate_archive.zip`, sha
`1ccae18d86322f59a1b99cc37849ef0c9ac1a42d21de6d2387cb15d1de7873e0`, 178,495 B):
- Switches the **7 above-noise improvable pairs** to their on-host argmin-pose mode (noise floor = 0
  ⇒ every positive on-host gain is admissible; conservative tie-handling is moot).
- Selector bytes 220 → **222** (+2); decoder/latent/latent-correction-sidecar + DQS1 tail
  byte-identical; runtime tree (inflate.sh + inflate.py) byte-identical to the frontier (shas match).
- **No-op detector PASSED**: both archives inflate to 1200 frames; **16,010,739 raw bytes differ**;
  `consumption_proven=true`. The new selector bytes genuinely change the rendered frames.

**Pre-registered prediction (recorded before the eval):** net ΔS = pose −4.12e-6 + rate +1.33e-6 =
**−2.79e-6**; this is below the eval reproducibility floor (R1's frontier matched archived within
4.66e-6), so the expected verdict is a directional improvement at most, NOT a magnitude that moves the
contest score.

**PAIRED CONTEST-CPU EVAL (same Modal Linux-x86_64 host, same `eval_image`, same session, 600 samples,
`upstream/evaluate.py --device cpu`, byte-identical inflate runtime):**

| eval | final_score [contest-CPU] | avg_posenet_dist | avg_segnet_dist | bytes | sha | call_id |
|---|---:|---:|---:|---:|---|---|
| frontier baseline | **0.19198534** | 2.943e-05 | 5.5979e-04 | 178,493 | `b7106c9b…` | `fc-01KTQMZXAKM5J5Z63ESVP45ZZX` |
| candidate (7-switch) | **0.19198275** | 2.942e-05 | 5.5978e-04 | 178,495 | `1ccae18d…` | `fc-01KTQMWVQJJSP28GGBWZ91SYQC` |

- **ΔS (candidate − frontier) = −2.58e-06** — a **directional WIN** (candidate scores lower), matching
  the pre-registered prediction (−2.79e-6) almost exactly. Verdict from the canonical ingest tool:
  `beats_frontier`, `pays_rent=True`.
- **Apples-to-apples validated:** this session's frontier baseline (0.19198534) matched R1's
  prior-session frontier (0.19198534) within **3.7e-09** — exceptional cross-session reproducibility,
  confirming the Modal CPU axis is rock-solid and the −2.58e-6 candidate delta is ~700× above this
  session's measured reproducibility (a real, reproducible directional improvement, not noise).
- d_seg held essentially identical (SegNet-blindness exact, as predicted); the byte cost (+2) is
  swamped by the pose reduction.

**The honest magnitude verdict:** ΔS = −2.58e-6 is a REAL, reproducible, sign-correct win — but it is
**3 orders of magnitude below contest reporting precision** (both scores round to 0.19) and far below
any medal-band-moving threshold. It is NOT a promotable frontier advance. The R3 mission CONFIRMED that
the per-pair selector lever, evaluated on the corrected (contest-GT + contest-host) substrate, is
**essentially exhausted at the frontier**: the maximum achievable improvement is 7 pairs / 8.5e-6 pose /
−2.58e-6 score.

---

## 3. Routing + wire-in (Catalog #125)

### Verdict

R1's KILL reactivation criterion is **SATISFIED and the per-pair-selector PARADIGM is resolved, not
revived.** Building the per-mode pose table ON the contest-CPU host (with the contest-exact GT decode)
produced a corrected selector that DOES beat the frontier on the contest-CPU axis (ΔS = −2.58e-6,
`beats_frontier`) — vindicating R1's mechanism diagnosis (the macOS substrate mis-ranked; the host
substrate ranks correctly). BUT the win is 3 orders of magnitude below contest precision. The PR110++
per-pair frame-0 selector lever is **EXHAUSTED at the 0.19199 frontier**: the incumbent FECa selector is
already near-optimal on the corrected substrate (593/600 pairs are already at argmin-pose; the 7
improvable pairs total 8.5e-6 pose). This is a DEFER-pending verdict for the per-pair-selector family,
NOT a kill (CLAUDE.md "Forbidden premature KILL"): the lever is real but saturated; reactivation would
require a NEW lever (a richer per-pair mode vocabulary beyond the K16 frame-0 menu, or a frame-1 / pair-
residual / decoder-axis lever — the decoder is 91% of bytes and untouched here).

### The two substrate-invalidating bugs this R3 closed (permanent knowledge)

1. **Host-axis FP drift (R1's finding):** macOS-arm-CPU per-mode pose ordering does not transfer to
   Linux-x86_64-CPU at pose~1e-5. Fix: build the table ON the contest host. **Verified here:** the
   contest host's reproducibility-twin noise floor is exactly 0.0 (bit-exact kernels), and the
   same-session frontier reproduced the prior-session frontier within 3.7e-9.
2. **GT-decode mismatch (R3's finding, NEW):** R2's macOS table AND R3's first on-host table both decoded
   GT via PyAV `to_ndarray(format="rgb24")` (libswscale) instead of the contest's
   `frame_utils.yuv420_to_rgb` (BT.601 limited-range + bilinear chroma). This inflated absolute pose
   ~100× (3.31e-3 vs the true 2.94e-5) and manufactured 591/600 "improvable" pairs (vs the true 7).
   The R1/R2 "8.27e-2 improvable" signal was 100% a wrong-GT artifact. Both bugs had to be fixed
   together; the corrected substrate requires the contest host AND the contest GT decode AND the
   contest scorer.

### DROP-IN HARDENING RECOMMENDATION (for future PR110++ menu/per-region/multi-mode cluster work)

Any future per-mode / per-region / multi-mode candidate-generator MUST score against
`frame_utils.yuv420_to_rgb` GT on the contest host, NOT PyAV rgb24 on macOS. The R2 machinery
(`render_and_score_lib.decode_gt_pairs`) carries the rgb24 bug and should be superseded by the R3 remote
module's contest-GT path before any of those candidates are trusted. This is the corrected substrate the
whole cluster should consume.

### Wire-in (Catalog #125)

- **Hook #5 continual-learning:** two exact contest-CPU `tac.action_effect.v1` rows minted via
  `tools/ingest_exact_eval_to_candidate.py` (frontier baseline + candidate; candidate verdict
  `beats_frontier`, ΔS=−2.58e-6, `pays_rent=True`). Reseeds the V3 ΔS-judge with the FIRST
  corrected-substrate PR110++ row.
- **Hook #6 probe-disambiguator:** RESOLVED two probes — (a) "does the per-mode pose ordering transfer
  to the contest host?" (yes, the on-host substrate ranks correctly) and (b) "was the substrate GT
  decode correct?" (NO — rgb24 vs yuv420_to_rgb was a 100× bug; now fixed).
- **Hook #2 Pareto:** confirms the per-pair selector is on its Pareto-exhausted edge at the frontier;
  the binding constraint is no longer the selector (rate share 62% per the ingest-tool universal
  decision), it is the decoder (91% of bytes) + the saturated per-pair pose lever.
- **Hooks #1/#3/#4:** N/A — the corrected substrate does not open a new bit-allocator/sensitivity/
  autopilot-dispatch input beyond confirming the selector lever is saturated.

### Cost

On-host pose table (v4 wrong-GT + v5 contest-GT) + 2 paired evals ≈ 4 Modal CPU dispatches; each
~$0.13-0.33 → total ≈ **$0.9**. Under the $5 STOP gate; no budget blocker.

### Routing

DEFER the per-pair-selector family (per-mode / per-region / multi-mode K16 frame-0 menu) as
**lever-exhausted-at-frontier** with reactivation criterion = a NEW per-pair lever (richer vocabulary,
frame-1 modes, pair residuals) OR the decoder axis (91% of bytes). The corrected on-host + contest-GT
substrate is the canonical input for any future per-pair work. No frontier-candidate flag (the candidate
ties the frontier at contest precision); no submission (a −2.58e-6 win is not promotable, and CUDA
pairing + `pre_submission_compliance_check.py --contest-final` remain required gates regardless).

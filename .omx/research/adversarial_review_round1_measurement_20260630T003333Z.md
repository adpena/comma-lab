# Adversarial Review ROUND 1 — LENS A (MEASUREMENT / AUTHORITY)

**UTC:** 2026-06-30T00:33:33Z · **git:** (working tree, main) · **tag:** `[macOS advisory / read-only audit]`
**score_claim=false · promotable=false · ready_for_exact_eval_dispatch=false** · pointer **UNMOVED 0.19110**.
Lens A of the recursive adversarial review of the **v2 witness program**. READ-ONLY: findings + proposed
fixes, NO code edits (fixes land post-synthesis to avoid concurrent-edit races). Authority = numpy-fp32 /
CPU-torch; NEVER MPS. means ≠ ends.

## TL;DR (CRITICAL count = 2; top 3)

The measurement *chain* is mostly contest-faithful and the NO-FAKE self-checks are real (they execute and
ABORT on mismatch with zero tolerance). The session's advisory numbers are **correctly measured for the
subset they were run on** — but two structural issues make several HEADLINES *optimistic / non-comparable*:

1. **[CRITICAL] Contiguous-prefix subsets are systematically EASY → every n24/n96 headline is OPTIMISTIC
   vs n600.** `decode_gt_frame1_pairs` yields the FIRST n contiguous pairs (start of the clip), not a
   representative sample. MEASURED swing in the SAME quantity (R0-flat d_seg_bulk): **0.00512 @n24 vs
   0.0242 @n96 = 4.7×**; clean-canonical bulk floor **0.00291 @n96 → 0.00427 @n200-strided = 2.4×→3.5×**.
   The strided cache (`gt_strided_n200.npz`, spans the drive) is the correct representative sampler and
   shows materially HIGHER d_seg. So REF_B 0.00013, R3 0.00468, the fooling-ladder rungs, the waterfill
   n96 floor, and the budget-gate n96 are all easy-prefix numbers — **n600 d_seg is almost certainly
   higher.** This is the dominant "the numbers are an artifact" risk for this lens.

2. **[CRITICAL — coordinator contradiction adjudicated] The `--batch-size n` byte-close rows: distortion
   is REAL, composite S is a RATE-ARTIFACT.** The truncation genuinely scores the first n pairs through the
   real evaluator (store_raw → d_seg=0/d_pose=0 proves ordering+truncation+distortion). BUT `rate =
   archive_bytes / 37_545_489` uses the FULL-video denominator regardless of n (`upstream/evaluate.py:63-65`),
   so an n-pair archive's rate is understated by ~n/600 (×25 at n24). The **S(n24) column (53.91 / 73.23 /
   1.38) is NOT comparable to the 600-pair 0.19110.** Verdict: distortion = real subset (advisory); composite
   S(n24) = downgrade to advisory rate-artifact; use the agent's n600-PROJECTED S. (The byte-close RESULT
   JSON already discloses this; the DAG-FEED "REAL byte-closed exact-eval rows" headline is the misquote risk.)

3. **[MEDIUM] The 0.978 co-location is correctly measured but largely ALGEBRAIC.** Fisher curvature
   `1−Σp²` and `−margin` are both monotone readouts of softmax peakedness, so a high Pearson is expected a
   priori — it does not independently validate the "Fisher background" framing as strongly as it reads. The
   independent *spatial* tests (flip-mass in 2px annulus 0.968; anisotropy) are the load-bearing parts.

---

## POSITIVE CONFIRMATIONS (calibration — these are NOT bugs; the chain is sound here)

- **GT `lstars` are contest-EXACT.** `segnet_argmax_and_margin` → `measure_segnet_argmax`
  (`src/tac/optimization/frame1_seg_repair_atoms.py:299`) builds the degenerate pair and runs the upstream
  `SegNet.preprocess_input` (`x[:, -1, ...]` then bilinear↓ to (384,512)) → exact `upstream/modules.py:108-113`
  contract. GT decode via `yuv420_to_rgb` (NOT PyAV rgb24 phantom-pose), non-overlapping pairs, frame1 =
  second frame (`src/tac/boundary_math/seg_core.py:80-109`). ✓
- **NO-FAKE self-checks ACTUALLY EXECUTE and ABORT.** Every warp/ladder/waterfill/jitter/budget-gate tool
  re-runs `SegNet(gt_f1)` and `raise SystemExit` if `max_disagree_px != 0` (exact zero tolerance; e.g.
  `measure_jitter_predictability.py:323-333`, `measure_budget_gate_overturn.py:704-715`,
  `measure_segnet_fooling_ladder.py:172-181`). Not docstring theater.
- **Class-index order CORRECT (not luma-sorted).** Canonical comma10k `["Road","Lane","Undriv","Movable",
  "MyCar"]` (`tools/measure_pose_warp_dseg.py:58`, matches `src/tac/semantic_label_contract.py:30-35` and
  MEMORY); named-index lookup → `BULK_IDX=[0,2,4]`; `SCREW_REGIME={0:ground,1:ground,2:rotonly,3:ground,
  4:identity}` matches the MEMORY stratified-warp. No hardcoded luma sort anywhere in the tool chain.
- **d_seg / d_pose match upstream.** d_seg = `(argmax≠argmax).float().mean()` (`modules.py:112-113`); d_pose
  = MSE on first `out//2` pose dims (`modules.py:82-84`). The tools compute argmax-disagreement-mean
  identically.
- **R operator is contest-faithful.** Warp tools: warp@874 → uint8@874 → **torch** SegNet bilinear↓384 →
  argmax (bicubic-up IS identity because the warp output is camera-res 874×1164 — confirmed
  `measure_screw_warp_through_R.py:108-147`). Fooling ladder: render@384 → **torch** bicubic↑874 (a=-0.75,
  align_corners=False) → uint8 → torch bilinear↓384 → argmax. The MLX `apply_contest_faithful_roundtrip_nhwc`
  (`src/tac/local_acceleration/pr95_hnerv_mlx_training.py:126`) is correct ORDER (uint8 @ CAMERA, not
  scorer) and correct math (align_corners=False `(out+0.5)*scale-0.5`, bicubic a=-0.75) — but it is only in
  the TRAINER, not in the headline measure tools.
- **No MPS / proxy / surrogate in any headline-tool authority path.** All use `load_real_segnet("cpu")`
  (raises on `mps`). Co-location recomputes full logits on the same frozen CPU-torch and PROVES parity
  (argmax mismatch 0.0; margin |Δ|max 4.8e-7 ULP).
- **ker(R) = 80.67% is definitionally correct.** `1 − (384·512)/(874·1164) = 1 − 196608/1017336 = 0.8067`
  (the max row-space / min nullspace of the bilinear downsample); residual=0.0 vs `F.interpolate` verified.
- **R3 0.00468 IS real + reproducible** (I initially mis-flagged R3 as "None": it uses the schema key
  `delta_through_R_uint8_d_seg_bulk["32.0"]=0.004679`, not `d_seg_full`; `ladder_n96.json` confirms; matches
  FEED-kv "pre_uint8 0.00465 ≈ thru_R 0.00468 @eps32"). The 0.00468 is real — see C1 for the subset caveat.

---

## FINDINGS

### [CRITICAL] C1 — Contiguous-prefix subset optimism (the dominant artifact for this lens)
**Where:** `src/tac/boundary_math/seg_core.py:80` (`decode_gt_frame1_pairs` yields the FIRST n pairs);
defaults in `tools/measure_segnet_fooling_ladder.py` (n96 / R3 n24), `measure_waterfill_through_R.py`,
`measure_budget_gate_overturn.py`, `measure_clean_canonical_warp_through_R.py` (n96).
**Evidence (MEASURED, same quantity, different subset):**
- R0-flat d_seg_bulk: **0.00512 @n24** (`ladder_n96.json` R3 `base_R0_flat_d_seg_bulk`) vs **0.0242 @n96**
  (`ladder_n96.json` R0 `d_seg_bulk`) = **4.7×**.
- clean-canonical BULK floor through R: **0.00291 @n96 → 0.00427 @n200-strided** (`clean_canonical_warp_
  budget_gate_20260629T203717Z.md`, FEED-jz) = 2.4×→3.5×; the strided builder (`build_strided_subset_gt.py`)
  is a **byte-faithful slice** of the exact `precompute_gt` outputs (NO surrogate) → the trend is real, not
  a build bug.
**Why it matters:** `100·d_seg` dominates S. Every optimistic-sounding n24/n96 headline (REF_B 0.00013, R3
0.00468, fooling-ladder R0-R4, waterfill `MODEL_FREE_PRIOR_SOPT_BULK=0.108`, budget-gate 0.00291) is on the
EASY prefix; the representative (strided/n600) d_seg is materially higher. A future agent quoting "REF_B
0.00013" or "R3 0.00468" as the n600 floor would be reading an artifact.
**Proposed fix:** (1) re-run the headline ladders/waterfill on `gt_strided_n200.npz` (or n600) before any
rung is treated as load-bearing; (2) add a `--strided` / `--indices` option to `decode_gt_frame1_pairs` so
prefix-vs-strided is a flag, not a fixed bias; (3) hard-label every prefix-n number
`[contiguous-prefix subset — OPTIMISTIC vs n600]` in the result JSON and DAG feeds.
**Confidence:** HIGH (directly measured 4.7× and 3.5× swings).

### [CRITICAL] C2 — `--batch-size n` byte-close: distortion REAL, composite S a rate-artifact (adjudication)
**Where:** `upstream/evaluate.py:63-65` (`compressed_size/uncompressed_size`, full denominator) +
`experiments/v2_witness_byteclose_smoke.py` + `.omx/research/dag_feed_v2_deterministic_byteclose_phase1_smoke_20260630.md`.
**The two-agent contradiction resolved:**
- The byte-close agent is RIGHT that the `--batch-size n` + n-pair-archive path is a GENUINE real-evaluator
  *subset for DISTORTION*: `zip(dl_gt, dl_comp)` (evaluate.py:71-74) stops when the n-pair comp stream is
  exhausted; with `--batch-size n` that is exactly ONE matching batch → d_seg/d_pose over the first n pairs.
  `store_raw n24 → d_seg=0/d_pose=0` PROVES pair-ordering + truncation + the distortion path (a misalignment
  would give nonzero distortion). So the pipeline-mapping agent's "no real-evaluator subset path exists" is
  too strong — there IS a real subset path for DISTORTION.
- The pipeline-mapping agent is RIGHT that the COMPOSITE small-n S is advisory-only — specifically the RATE.
  `uncompressed_size = sum(... rglob('*'))` = the FULL 0.mkv (~37.5 MB = B0) **independent of n**. An n-pair
  archive encodes n/600 of the content, so `rate_smalln ≈ (n/600)·rate_full` → the S(n24) rate term is ~1/25
  of a real 600-pair version. **store_raw S(n24)=53.91 = 25·(80.97MB/37.5MB) is real for that archive but
  NOT comparable to 0.19110** (a 600-pair submission). store_raw=0/0 validates ONLY distortion+ordering, NOT
  rate comparability (store_raw's ENTIRE S is the mis-scaled rate, so it cannot self-validate the rate axis).
**Verdict / labeling:** the 3 rows = **REAL subset distortion (advisory)** + **rate-artifact composite**.
Use the agent's **n600-PROJECTED S** (rate ×600/n) for any comparison — and even that carries the n24
distortion forward (C1 prefix-optimism applies; projection used the easy prefix). NONE beats 0.19110 (all
project ≥ 20.6), so NO false frontier was claimed; the only defect is the DAG-FEED "REAL byte-closed
exact-eval rows" + "S (n24)" framing inviting an out-of-context quote.
**Proposed fix:** in the FEED/JSON, rename the "S (n24)" column to "S(n24) [rate-artifact; not 600-comparable]"
and lead with the n600-projected S; add a one-line invariant note that small-n rate ≈ (n/600)·full-rate.
**Confidence:** HIGH.

### [MEDIUM] M1 — 0.978 co-location is correctly measured but near-algebraic
**Where:** `tools/colocation_fisher_stress_anisotropy_test.py:18,262` (Pearson(curvature, −margin)).
**Issue:** curvature `1−Σp²` and `−margin` are BOTH monotone functions of softmax peakedness (small top-2
gap ⇒ flat softmax ⇒ high curvature), so a high correlation is expected *a priori*; the 0.978 (band) /
0.814 (all) does not independently confirm the "matter on a fixed Fisher background" design framing. The
measurement is faithful (logits recomputed on frozen CPU-torch, argmax/margin parity proven), but its
*evidentiary weight for the design claim* is weak. The genuinely informative, non-tautological results are
the SPATIAL tests (flip-mass in 2px annulus 0.968; tangent anisotropy 9.56:1).
**Proposed fix:** report curvature↔−margin explicitly as "expected algebraic monotone (shared peakedness
readout)"; rest the Fisher-background claim on the spatial-annulus + anisotropy tests, not on 0.978.
**Confidence:** MEDIUM-HIGH.

### [MEDIUM] M2 — numpy-bilinear vs torch-bilinear mismatch in REF_B / class stats
**Where:** `tools/measure_segnet_fooling_ladder.py:66` (`_resize_bilinear`, numpy) used at :201 for the GT→384
downsample (`gt384`) that feeds REF_B and the per-class mean/std; the AUTHORITY downsample inside
`measure_segnet_argmax` is **torch** bilinear.
**Issue:** REF_B (0.00013 bulk) is `gt → numpy-bilinear↓384 → bicubic↑874 → uint8 → torch-bilinear↓384 →
SegNet`. This is (a) a deliberate "384-render resolution-penalty" proxy, NOT contest geometry (contest is
`gt(874)→torch-bilinear↓384`); and (b) its small absolute value is sensitive to the numpy↔torch bilinear fp
mismatch. REF_B's role as a "ceiling" is fine, but the absolute number should not be quoted as a hard floor.
**Proposed fix:** use torch bilinear for the GT→384 downsample (match the authority), OR label REF_B
`[numpy-bilinear proxy; not contest geometry]`. Also note REF_B is at n96 (C1 applies).
**Confidence:** MEDIUM.

### [LOW] L1 — deprecated wrong-uint8 roundtrip still present
**Where:** `apply_eval_roundtrip_nhwc` (`pr95_hnerv_mlx_training.py:77`, uint8 at SCORER res = optimistic).
The headline measure tools do NOT call it; the trainer references it only in a comment (calls the faithful
one). Two STALE smokes under `experiments/results/mlx_through_R_diff_smoke_20260625/` DO call it — do not
cite those. **Fix:** none required for headlines; keep Catalog #392's new-caller refusal.
**Confidence:** HIGH.

### [LOW] L2 — NO-FAKE self-check is partial (default 4 pairs)
**Where:** `--selfcheck-pairs` default 4 across the tools. The check verifies `SegNet(gt_f1)==lstars` for
only the first few pairs (the per-frame `seg_cache` recompute covers the rest where used). A cache corrupted
beyond pair 4 (in a tool that reads `lstars` directly without recompute) would not be caught. Low risk
(single deterministic build), but **fix:** default selfcheck-pairs to P (or ≥ the number of pairs actually
consumed) so the self-check covers the whole consumed range.
**Confidence:** MEDIUM.

---

## NOTE FOR SYNTHESIS
- C1 is the highest-leverage: it does not refute any single tool, but it re-frames EVERY n24/n96 headline as
  an easy-prefix lower bound. The strided/n600 re-run is the cheap fix and should gate any "d_seg ≈ X"
  claim used to argue sub-0.15 feasibility.
- C2 is narrow (labeling) but matters because the contradiction was live; the distortion subset is real, the
  composite S is not 600-comparable. No frontier claim was actually made.
- All "is the chain faithful?" checks PASS (GT exact, self-checks real, class order correct, no MPS/proxy,
  contest-faithful R). The operator's recurring fear ("findings are measurement artifacts") is justified
  ONLY through the SUBSET-CHOICE lens (C1) and the RATE-DENOMINATOR lens (C2), not through the scorer/decode
  chain itself.

---

## DAG FEED — adversarial review R1 / LENS A (MEASUREMENT/AUTHORITY)

**FEED-R1A (2026-06-30T00:33:33Z) — measurement/authority audit of the v2 witness; pointer 0.19110 UNMOVED; advisory.**
CRITICAL count = **2**. The scorer/decode chain is contest-FAITHFUL (GT `lstars`=exact upstream
`SegNet.preprocess_input`; NO-FAKE self-checks EXECUTE+ABORT zero-tol; class order canonical comma10k
[Road,Lane,Undriv,Movable,MyCar] not luma-sort; d_seg/d_pose match modules.py; R contest-exact; no
MPS/proxy in authority paths; ker(R)=80.67% definitional; R3 0.00468 real+reproducible). The artifacts are
in SUBSET CHOICE + RATE DENOMINATOR, not the chain.
**Top 3:** (1) [CRIT] **contiguous-prefix optimism** — n24/n96 use the FIRST n pairs (easy clip-start);
SAME R0-flat d_seg_bulk = 0.00512@n24 vs 0.0242@n96 (4.7×); clean-canon 0.00291@n96→0.00427@n200-strided
(3.5×). EVERY n24/n96 headline (REF_B 0.00013, R3 0.00468, ladder, waterfill 0.108, budget-gate) is
OPTIMISTIC vs n600 → re-run on `gt_strided_n200`/n600 before load-bearing. (2) [CRIT] **`--batch-size n`
byte-close ADJUDICATION** — distortion subset is REAL (store_raw 0/0 proves ordering+truncation), but
`rate=archive/37_545_489` uses the FULL denominator regardless of n → S(n24) 53.91/73.23/1.38 is a
rate-artifact NOT comparable to 0.19110; use the agent's n600-projected S; store_raw=0/0 validates ONLY
distortion, NOT rate. No false frontier claimed; labeling fix only. (3) [MED] **0.978 co-location** is
correctly measured but near-ALGEBRAIC (curvature 1−Σp² and −margin are both peakedness readouts) → lean on
the spatial-annulus tests for the design claim. Memo:
`.omx/research/adversarial_review_round1_measurement_20260630T003333Z.md`. means ≠ ends.

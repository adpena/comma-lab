# Score-aware per-tensor weight re-quant — VERDICT (Task #69)

**Exact pointer delta: 0.19109982 → 0.19109982 (NO MOVE). Whole-tensor score-aware
re-quant is FALSIFIED on the contest-CPU axis.** Every operating point on the
re-quant curve is strictly WORSE than the frontier: the best candidate
(`crush4_int6`, the gentlest 6-bit crush of the 4 lowest-sensitivity tensors)
scores **0.35130 vs frontier 0.19111 — Δ +0.160** because the EXACT-scorer
distortion penalty is **13.8× the rate saving** it buys. The audit's #1 lurking
hypothesis — "the frontier decoder weights are quantized for pixel-recon, not
for the score, so the frozen scorer tolerates far more weight error than recon"
— is **empirically false for whole-tensor uniform re-quant**.

```yaml
# verdict.v1 (lossy-on-pixels frontier-move attempt)
schema: score_aware_weight_requant_verdict.v1
date: 2026-06-10
subagent: task69_score_aware_requant
lane: lane_score_aware_weight_requant_20260610
axis: "[contest-CPU advisory]"   # local exact-scorer harness, NEVER MPS; matches eval geometry within 1e-7 d_seg
defensive_bank: true              # borrowed PR#112 base archive (lane_pr110_payload_entropy_recode)
method_originality: true          # the per-tensor FROZEN-SCORER-sensitivity bit-allocation is original (no competitor allocates decoder bits this way)
frontier_pointer_before: 0.19109982419209975
frontier_pointer_after:  0.19109982419209975   # UNCHANGED
pointer_moved: false
crossed_T1_0.19: false
crossed_T3_0.15: false
paired_eval_fired: false          # gated: no candidate beat frontier, so the <=$1 paired CPU+CUDA eval was NOT spent
verdict: NEGATIVE_HYPOTHESIS_FALSIFIED_WHOLE_TENSOR
kill_or_defer: DEFER_pending_per_channel_or_score_aware_QAT
spend_usd: 0.0                    # $0 local MLX/CPU smoke only; no paid dispatch
```

## What was built (the reusable surfaces)

- `tac.score_aware_weight_requant` — the per-tensor score-aware bit-allocation
  primitives: q-domain re-quant (`requant_signed_q`, lowers q-byte entropy onto a
  coarser level grid), the exact PR#101 byte-map inverse pair
  (`decode_byte_map_u8`/`encode_byte_map_u8`), entropy accounting
  (`q_byte_entropy_bits`), the ORIGINAL sensitivity→levels allocator
  (`allocate_bits_by_sensitivity`), and the contest-score recompute-from-components
  (`contest_score_from_components`). 22 NO-FAKE behavioral tests
  (`src/tac/tests/test_score_aware_weight_requant.py`). Commit `0bd7b4389`.
- `tools/score_aware_weight_requant_sweep.py` — the EXACT-authority harness:
  decode the frontier archive's decoder state_dict, per-tensor scorer-sensitivity
  finite-difference (RANK stage), allocate bits, re-quant in q-domain, re-pack into
  the EXACT CTXR/FP11 grammar (byte-closed), DECODE through the FRONTIER inflate
  chain, and measure EXACT d_seg/d_pose on the FROZEN upstream SegNet+PoseNet vs GT
  decoded via `frame_utils.yuv420_to_rgb` (NEVER MPS). Commits `bfc1562fd` +
  `bc16a94bf` + `ad3c2358c` + `0e1da7d48`.

## NO-FAKE proof chain (this result is real, not a proxy)

1. **Calibration gate (load-bearing):** the harness reproduces the frontier's
   distortion on the UNCHANGED archive — d_seg = 0.00055988 (frontier 0.00055978,
   Δ 1e-7 from CPU-decode numerics), d_pose = 0.00002942 (8-digit match), score
   0.19110854 (pointer 0.19109982, Δ 9e-6). The exact scorer + exact GT-decode +
   exact frontier render chain are wired correctly.
2. **Byte-closure / lossless re-pack:** a NO-OP re-quant (`levels={}`) rebuilds the
   member **byte-identical** (177,069 B); the latent + sidecar + DQS1 tail are
   byte-identical across every candidate (only the decoder section changes);
   `parse_member` round-trips every candidate cleanly. The re-quant genuinely
   changes the decoder q-bytes (class-1 no-op guard) and the bytes are entropy-coded
   smaller (e.g. int4 on a single big tensor = −18 KB ctx bytes).
3. **Exact authority:** d_seg = argmax-disagreement rate on the frozen SegNet last
   frame; d_pose = MSE on the 6 PoseNet dims; both vs GT from `0.mkv` via
   `yuv420_to_rgb`. Score recomputed from components (the rounded field is not used).
   600 pairs (full eval set) for the sweep; 120 pairs for the relative ranking.

## The per-tensor scorer-sensitivity ranking (RANK stage, int2 probe, 120 pairs)

The hypothesis's PREMISE is partially confirmed: tensors DO have wildly different
frozen-scorer sensitivity (a **120× spread**). This is the ORIGINAL signal the
method produces.

| tensor | numel | score-sensitivity (int2 probe) | rank |
|---|---:|---:|---|
| blocks.5.weight | 11,664 | 4118 | MOST sensitive (Δd_pose +17.3) — PROTECT |
| blocks.0.weight | 46,656 | 2440 | very sensitive (Δd_pose +10.2) — PROTECT |
| blocks.3.weight | 19,440 | 183 | moderate (DQS1-protected anyway) |
| blocks.1.weight | 46,656 | 168 | tolerant |
| stem.weight | 48,384 | 145 | tolerant |
| blocks.2.weight | 34,992 | 124 | tolerant |
| blocks.4.weight | 12,960 | 34 | MOST tolerant |

Score-sensitivity = `100·|Δd_seg| + (5/√(10·d_pose₀))·|Δd_pose|` at the int2 probe.

## The byte vs distortion operating-point curve (SWEEP, EXACT, 600 pairs)

Baseline (frontier, byte-identical): d_seg 0.00055988, d_pose 2.942e-05, 177,169 B,
**score 0.19111**.

| candidate | crushed tensors | levels | bytes | Δbytes | d_seg | d_pose | score | Δscore |
|---|---|---|---:|---:|---:|---:|---:|---:|
| crush2_int4 | blocks.4, blocks.2 | int4 | 158,765 | −18,404 | 0.002672 | 4.34e-4 | 0.43882 | **+0.24771** |
| crush4_int6 | +stem +blocks.1 | int6 | 158,301 | −18,868 | 0.002017 | 1.95e-4 | 0.35130 | **+0.16020** |
| crush3_int5 | blocks.4,2,stem | int5 | 152,533 | −24,636 | 0.003299 | 5.37e-4 | 0.50476 | +0.31365 |
| crush4_int5 | 4 tolerant | int5 | 140,813 | −36,356 | 0.003666 | 6.49e-4 | 0.54095 | +0.34984 |
| crush4_int4 | 4 tolerant | int4 | 122,729 | −54,440 | 0.008358 | 6.63e-3 | 1.17498 | +0.98387 |
| crush4_int3 | 4 tolerant | int3 | 104,545 | −72,624 | 0.018426 | 1.097e-1 | 2.95962 | +2.76851 |

**The curve is monotone-worse in aggression.** The rate savings are large (the
decoder blob is 91% of the archive, so int4-on-4-tensors saves 54 KB → rate term
−0.036), BUT the distortion penalty grows far faster than the rate saving shrinks.

## WHY it fails (the mechanism — feeds the system model)

At the GENTLEST crush (`crush4_int6`, 6-bit):
- distortion penalty (seg+pose terms) = **+0.1728**
- rate saving (25·Δbytes/N) = **−0.0126**
- ⇒ **distortion penalty is 13.8× the rate saving.**

The HNeRV decoder is 6 PixelShuffle upsample stages of 3×3 convs with `sin`
activations, feeding two sigmoid RGB heads. Per-weight quantization error
**compounds multiplicatively** through the cascade, so even 6-bit coarsening
shifts enough camera-res pixels to flip the SegNet per-pixel argmax (d_seg) and
perturb the PoseNet pose (d_pose). The frozen scorer is NOT a slack consumer for
these weights — the contest's int8 quantization (set by PR#95/#101 training, then
losslessly entropy-recoded by PR#110/#112) is **already at/near the score-relevant
floor**: the bits that survive int8 are score-load-bearing, not recon-luxury.

This refines the system's model: **"weights quantized for recon ⇒ scorer-slack to
reclaim" is FALSE for cascaded-conv single-video memorization decoders.** The
recon objective and the score objective are tightly coupled here precisely because
the renderer is a memorizer — every weight is spent reproducing the exact pixels
that produce the exact argmax/pose. There is no recon-only fat to trim at the
whole-tensor level.

## Verdict: NEGATIVE (whole-tensor); DEFER (paradigm) — KILL is NOT warranted

Per CLAUDE.md "Forbidden premature KILL without research exhaustion": the
whole-tensor uniform re-quant IMPLEMENTATION is falsified (Catalog #307
implementation-level), but the score-aware-bit-allocation PARADIGM is NOT killed.
Reactivation paths the curve does NOT rule out (the genuine next research, ranked
by expected value):

1. **Per-CHANNEL (not per-tensor) re-quant** — the int2 probe collapsed whole
   tensors; a per-output-channel sensitivity map could find *individual channels*
   within the tolerant tensors that hold the score at low bits while the
   score-load-bearing channels stay full. The compounding argument is weaker
   per-channel. (Highest EV.)
2. **Score-aware QAT fine-tune** — re-quantize to int4/int5 then FINE-TUNE the
   surviving levels against the SegNet/PoseNet objective (eval-roundtrip + EMA per
   CLAUDE.md) so the network *relearns* to put the argmax/pose back inside the cell
   at the lower bit-depth. This is the real "lossy-on-pixels, lossless-on-score"
   move — re-quant ALONE (no retrain) cannot relocate the decision boundary.
3. **Mixed-precision per-tensor at int7** — the curve started at int6; int7 (a
   single-bit trim) on only blocks.4 was not measured and may be the lone
   net-neutral point, but the EV is tiny (blocks.4 int7 saves <6 KB → rate −0.004,
   and even int6-on-4 moved d_seg 3.6×, so int7-on-1 likely still loses).

The rate lever lives in the decoder blob (91% of bytes), but it is NOT reachable
by post-hoc weight coarsening — it requires either finer-than-tensor granularity
or a retrain. Re-quant-without-retrain on this substrate is a dead end.

## 6-hook wire-in (Catalog #125)

1. **Sensitivity-map**: ACTIVE — the per-tensor frozen-scorer sensitivity ranking
   (`rank_result.json`) is the canonical per-tensor decoder sensitivity surface;
   the per-channel extension (reactivation #1) is the next contribution.
2. **Pareto constraint**: ACTIVE — the operating-point curve IS a measured
   rate↔distortion Pareto frontier for decoder re-quant; it adds the binding
   constraint "post-hoc whole-tensor weight coarsening is Pareto-dominated by the
   frontier" to the solver.
3. **Bit-allocator hook**: ACTIVE — `allocate_bits_by_sensitivity` is the new
   per-tensor score-aware allocator primitive; the empirical verdict is that its
   whole-tensor application is net-negative (records the failure so the allocator
   is not re-tried at this granularity).
4. **Cathedral autopilot**: N/A — no archive-deployable candidate (all worse).
5. **Continual-learning posterior**: ACTIVE — this NEGATIVE anchor + the 13.8×
   penalty/saving ratio updates the prior that "frontier int8 decoder weights are at
   the score floor" (calibration row).
6. **Probe-disambiguator**: ACTIVE — the RANK + SWEEP stages ARE the disambiguator
   between "scorer-slack exists (crush it)" vs "weights are score-load-bearing
   (don't)"; the verdict is the latter.

## Canonical-equation anchor (Catalog #344)

<!-- FORMALIZATION_PENDING: this is a NEGATIVE falsification of the whole-tensor
re-quant implementation, not a new promotable predictive model. The closest
canonical equation is the contest score `100*d_seg + sqrt(10*d_pose) + 25*rate`
(used verbatim in `tac.score_aware_weight_requant.contest_score_from_components`).
The empirical content here is the falsification constraint (whole-tensor post-hoc
re-quant is Pareto-dominated: distortion penalty / rate saving = 13.8x at the
gentlest operating point) which is recorded as a Pareto-constraint + sensitivity-map
contribution (6-hook §) rather than a standalone equation. A predictive model of
per-channel sensitivity is the reactivation #1 deliverable; it will register a
canonical equation when it produces a positive (or bounded) prediction. The
verdict is [contest-CPU advisory] (local exact scorer reproducing the eval
geometry within 1e-7 d_seg), NOT Linux-x86_64 [contest-CPU] authority, so it must
NOT mutate a promotable equation registry. -->

## Provenance

- Frontier archive: `experiments/results/pr110_payload_entropy_recode_20260610/submission_dir/archive.zip`
  (sha `b46897267…`, 177,169 B, member `x` 177,069 B).
- RANK result: `experiments/results/score_aware_weight_requant_rank_20260610T174135Z/rank_result.json`.
- SWEEP result: `experiments/results/score_aware_weight_requant_sweep_20260610T181757Z/sweep_result.json`
  (reconstructed from the run log after a cosmetic final-print KeyError; all 6
  candidates fully scored, data intact — fix landed `0e1da7d48`).
- GT scorer cache: `experiments/results/score_aware_weight_requant_cache/gt_camera_*.npy`
  (rebuildable from `upstream/videos/0.mkv` via `frame_utils.yuv420_to_rgb`).
- Bulky 1200-frame raw inflations deleted after scoring (disk hygiene; per-candidate
  scratch, rebuildable).
- $0 paid spend (local CPU exact scorer only; NO MPS, NO paid dispatch). The
  ≤$1 paired CPU+CUDA eval was correctly GATED OFF (no candidate beat frontier).

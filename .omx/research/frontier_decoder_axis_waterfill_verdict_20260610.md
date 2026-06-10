# Frontier DECODER-AXIS waterfill — the 162,127-byte decoder blob (91% of the 0.19199 archive)

**Subagent:** `frontier_decoder_axis_waterfill_20260610` · UTC 2026-06-10.
**Lane:** `lane_frontier_decoder_axis_waterfill_20260610`.
**Mission (R3's named reactivation criterion):** R3 (`pr110pp_r3_onhost_selector_verdict_20260610.md`)
proved the per-pair SELECTOR lever is exhausted (593/600 pairs already argmin; max win −2.58e-6) and
named the **decoder axis (91% of bytes) as the untouched surface**. This memo attacks it.
**Axis discipline:** the per-tensor sensitivity map + RD probe are `[macOS-CPU advisory]` /
`[contest-CUDA]` (the cached gradient arrays) candidate-generator priors — NOT score claims. The
paired contest-CPU eval (`upstream/evaluate.py --device cpu` on the byte-closed candidates,
Linux-x86_64 Modal CPU container, 1:1 with the contest GHA runner) is the score authority.

---

## 0. TASK 0 — the GT-decode bug class (two-landing fix, swept the class)

R3 §0 found the GT-decode apples-to-apples bug: the contest GT decode is
`frame_utils.yuv420_to_rgb` (BT.601 limited-range + bilinear chroma, "matches nvdec"), NOT PyAV
`frame.to_ndarray(format="rgb24")` (libswscale). The rgb24 path inflated absolute pose ~100×
(incumbent mean 3.31e-3 vs the contest frontier avg_posenet_dist 2.943e-5) and manufactured 591/600
spurious "improvable" pairs.

**Sweep result — exactly 1 live instance of the bug class:**
- `experiments/results/pr110pp_r2_nonmps_candidate_20260609/analysis/render_and_score_lib.py:170`
  (`decode_gt_pairs`) used `to_ndarray(format="rgb24")` — **FIXED** to route through
  `frame_utils.yuv420_to_rgb`. All `build_*.py` candidate generators route through this lib, so the
  single fix covers the whole R2 family.
- The R3 surface (`onhost_pose_table_remote.py:166`) was **already correct** (yuv420_to_rgb).
- Public-PR intake clones with rgb24 are pristine/out-of-scope (CLAUDE.md "Forbidden in-place edits
  to public PR intake clones"); `format="gray"` decodes are mask codecs (candidate payload, not GT).

**Two-landing discipline (CLAUDE.md "Bugs must be permanently fixed AND self-protected against"):**
1. **Fix** — `render_and_score_lib.decode_gt_pairs` now imports + calls `frame_utils.yuv420_to_rgb`.
2. **Self-protect** — tracked STRICT guard
   `src/tac/tests/test_pr110pp_candidate_generator_gt_decode_contest_exact.py` (committed `ee80865b1`;
   the experiments-tree custody is gitignored so the durable guard lives in the tracked `src/tac/tests`).
   It SCANS the pr110pp candidate-generator family + refuses any executable
   `to_ndarray(format="rgb24")` GT decode; skips vacuously on a fresh checkout where the ignored
   custody tree is absent. Verified: 3/3 fail on bug re-injection, 3/3 pass after revert.

---

## 1. The frontier decoder, structurally (the attack surface)

The frontier `b7106c9bdbb8…` (178,493 B) is the FP11 brotli-recode of the FECa source (decoder-blob
recode only; `decoder_raw_roundtrip_equal=True`, `decoder_saved_bytes=37`). Member-x grammar:
`FP11 + u32(source_len) + [decoder_blob(162127) | latent_blob(15387) | sidecar(879)] + u16(sel_len)
+ selector + dqs1`.

The decoder is the canonical **PR101-family HNeRV grammar** (verified byte-identical between FECa
`6bae0201` and the frontier): 28 tensors, INT8 q-bytes + fp16 per-tensor scale, 7 split-brotli
streams (`DECODER_STREAM_ENDS`), per-tensor byte maps (zig/negzig/twos/off), CONV4 storage perms.

**Two structural facts that bound every decoder lever:**
1. **The decoder is already INT8** (1 byte/weight; 92–206 of 256 levels used per tensor, std 18–35).
2. **brotli is at ~98.6% of the iid per-tensor Shannon entropy floor** (per-tensor mantissa entropy
   sum = 159,822 B vs the actual 162,127 B decoder blob). So the **coding-axis levers are near
   exhausted**: stream re-split / coder swap (brotli→LZMA→range) on the SAME q-byte distribution can
   recover at most ~1.4% (~2.3 KB), and likely far less since brotli already exploits cross-byte
   context the iid floor ignores. (R3 also already proved the selector-region recode is a no-op.)

**=> The only material decoder rate lever is DISTORTION: coarser quantization on low-sensitivity
tensors lowers q-byte entropy → fewer brotli bytes, at a weight-perturbation distortion cost that
the exact eval prices.** This is exactly THE LAW's tradeoff and is the directed attack here.

---

## 2. The per-tensor byte × sensitivity map (the bit-allocator prior)

Built by joining the master-gradient ledger's per-byte (seg, pose, rate) attribution
(`6bae0201` fec6-frontier lineage; same grammar + byte-identical q-bytes; CPU-advisory + one
`[contest-CUDA]` T4 array) against the frontier's 28-tensor spans + 7-stream partition. The saved
`.npy` is the canonical uniform-spread-onto-compressed-region projection
(`tools/extract_master_gradient.py`), so summing the array over a tensor's compressed span recovers
its aggregate score-axis sensitivity. Score-weight = `100·|grad_seg| + (5/√(10·pose_avg))·|grad_pose|`.

`build_per_tensor_sensitivity_map.py` → `per_tensor_sensitivity_map.json`. Headline:

- **The decoder holds 99.98% of total |grad|** across the whole archive (latents 8.6% of bytes carry
  ~0.02% of sensitivity; the decoder IS the score-relevant surface).
- **Bottom-decile per-byte sensitivity = the 3 LARGEST tensors** (the rate killers, ~100 KB / 62% of
  the decoder): `blocks.0.weight` (33,029 comp-B, SW/byte 5.14e-5), `stem.weight` (34,253 comp-B,
  5.61e-5), `blocks.1.weight` (33,030 comp-B, 6.03e-5). These are the natural coarsen targets.
- **Top-decile (untouchable):** `rgb_1.weight` (2.13e-3/byte), `rgb_0.weight` (1.43e-3/byte),
  `blocks.5.bias` — the tiny RGB output heads (344 B each) carry the highest per-byte sensitivity.
- **Per-stream:** stream 2 (20 tensors, 111,596 B) + stream 6 (`stem.weight`, 34,253 B) + stream 5
  (`blocks.3.weight`, 13,762 B) hold ~98% of decoder bytes; the small streams (0/1/3/4) are ≤1 KB.
- Caveat: even the bottom-decile per-byte sensitivity (~5e-5) is ~75× ABOVE the rate waterline
  (6.66e-7/byte) — but that is the BYTE-VALUE sensitivity; the quant lever trades VALUE precision for
  byte COUNT, so the relevant currency is the RD curve below, priced by exact eval.

---

## 3. The candidates (sensitivity-aware q-coarsening; byte-closed; no-op-proven)

`build_decoder_coarsen_candidates.py` reuses the canonical decoder-mutation engine end-to-end
(`feca_selector_reparameterize.{split,join}_fp11_member` + `fec6_decoder_mutations.{prepare_decoder_blob,
recompress_prepared_decoder, decode_mapped_u8, _encode_mapped_u8}` — NO duplicative code). It coarsens
ONLY the 3 bottom-decile tensors (round q to nearest multiple of `step`), leaves all other tensors +
selector + latent + sidecar byte-identical. Three rate operating points:

| cand | step | decoder B | archive B | archive saved | rate ΔS (measured) | weight-MSE (int8) |
|---|---|---:|---:|---:|---:|---|
| c1 | 2 | 152,819 | 169,185 | −9,308 | −6.20e-3 | blocks0/1 ~0.7, stem ~0.3 |
| c2 | 3 | 151,132 | 167,498 | −10,995 | −7.32e-3 | ~0.6–0.7 |
| c3 | 4 | 143,570 | 159,936 | −18,557 | −1.24e-2 | ~1.3–1.7 |

The rate gains are MEASURED + CERTAIN (real brotli re-encode). They are 3–4 orders of magnitude
larger than R3's exhausted selector lever (−2.58e-6). **The distortion cost is the unknown the eval
prices.**

**No-op detector (`submission_dirs_noop_proof.json`) — PASSED all 3:** each candidate inflates to
1200 frames; raw frame bytes differ massively from the frontier (c1: 2.45e9, c2: 2.54e9, c3: 2.75e9
bytes differ — monotonic with coarseness; `consumption_proven=true`); selector/latent/dqs1
byte-identical (decoder-only change). Candidate shas: c1 `5089b4eb…`, c2 `d86d3172…`, c3 `75b180ab…`.

---

## 4. Pre-registered predictions + kill criteria (recorded BEFORE the exact eval)

THE LAW verdict per candidate = `dS_total = dS_dist + dS_rate`, admit iff `dS_total < 0` below the
eval reproducibility floor (R3 same-session reproducibility 3.7e-9; cross-session ~4.66e-6).

- **Prior (Dykstra-feasibility + the frontier-is-already-rate-optimized fact):** the FECa packet the
  frontier recodes was already a rate-attack output; the decoder q-bytes were not chosen to be
  coarsening-robust, but the early-layer tensors (stem/blocks.0/1) feed EVERY frame, so global
  weight perturbation propagates to d_seg (100× coefficient) across all 1200 frames.
- **Prediction:** I predict the gentlest candidate **c1 (step=2)** is the only plausible net-favorable
  one; **c2/c3 likely land dS_total ≥ 0** (distortion exceeds rate). The local predicted-band smoke
  (`local_predicted_band.json`) sharpens this on a 24-pair advisory subset before the paid eval.
- **Kill criterion:** any candidate with exact `dS_total ≥ +1e-4` FALSIFIES "naive bottom-decile
  coarsening leaves exact-axis rate on the table at this distortion cost" → re-route to score-aware
  per-WEIGHT quant (requires re-running gradient extraction to get per-weight grads) or a different
  decoder lever. A candidate with `dS_total < −1e-5` (above the cross-session floor) is the FIRST
  exact decoder-axis frontier win and a promotable direction (pending CUDA pairing + compliance gate).

---

## 5. EXACT PAIRED CONTEST-CPU EVAL — the decisive measurement

4 archives, `upstream/evaluate.py --device cpu`, Linux-x86_64 Modal CPU container (1:1 with the
contest GHA runner), byte-identical inflate runtime, 600 samples. call_ids:
frontier `fc-01KTQQCXA54H0H31SJ3C226ZW3`, c1 `fc-01KTQQCZZ9E45BFEDABJHH9HQA`,
c2 `fc-01KTQQD0604P2YK6BPAQ30G3J5`, c3 `fc-01KTQQCWZ6DM8R95M6GV55DG8N`.

| eval | final_score [contest-CPU] | d_seg | d_pose | bytes | dS_dist | dS_rate | **dS_total** | pays_rent |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| frontier | **0.19198534** | 5.598e-4 | 2.943e-5 | 178,493 | — | — | — | — |
| c1 step2 | 0.26287 | 1.246e-3 | 6.568e-5 | 169,185 | +0.0729 | −0.0062 | **+0.0709** | False |
| c2 step3 | 0.28219 | 1.417e-3 | 8.412e-5 | 167,498 | +0.0901 | −0.0073 | **+0.0902** | False |
| c3 step4 | 0.35675 | 2.100e-3 | 1.623e-4 | 159,936 | +0.165 | −0.0124 | **+0.1648** | False |

**The frontier baseline reproduced 0.19198534 exactly** (matching R1/R3's prior-session frontier
0.19198534 within ~0 — the Modal CPU axis is rock-solid; the candidate deltas are 4-5 ORDERS of
magnitude above the reproducibility floor). All 3 candidates: `pays_rent=False`, `above_frontier`.

**Mechanism (THE LAW decomposition):** d_seg MORE THAN DOUBLED even for the gentlest coarsening
(5.6e-4 → 1.25e-3 at step=2), and the `100·d_seg` term turned that into +0.069 of score — ~10× the
rate gain (−0.0062). The decoder is a MEMORIZED single-video renderer: the early-layer weights
(stem / blocks.0 / blocks.1, the bottom-decile-per-byte tensors = the rate killers) feed EVERY pixel
of EVERY frame, so coarsening them perturbs all 1200 frames, and SegNet argmax-flips are exquisitely
sensitive to the resulting frame perturbation. THE LAW priced this correctly: distortion ≫ rate.

**Pre-registration check:** my pre-registered prediction (c1 plausibly favorable; c2/c3 net-unfavorable)
was DIRECTIONALLY half-wrong — even c1 was decisively unfavorable (+0.0709). The
`[macOS-CPU advisory]` local predicted-band smoke (`local_predicted_band.json`, 24-pair subset)
predicted c1 dS_total = +0.0644; the exact contest-CPU eval landed +0.0709 — the advisory band
correctly priced the UNFAVORABLE verdict within ~9% (validating the local kill-gate despite the
known macOS↔Linux FP-drift caveat: for distortion changes this large, FP drift is negligible).

---

## 6. Verdict + routing (Catalog #125)

### Verdict: naive sensitivity-aware q-coarsening of the decoder is FALSIFIED-AT-IMPLEMENTATION
(NOT a paradigm kill — Catalog #307; the decoder-axis distortion lever is intact, the SPECIFIC naive
bottom-decile uniform-coarsening implementation is falsified). All 3 candidates score 0.07–0.16
WORSE on the contest-CPU authority. The per-byte-sensitivity bit-allocator prior was a NECESSARY but
INSUFFICIENT guide: it ranked which tensors have low BYTE-VALUE sensitivity, but uniform coarsening
perturbs thousands of bytes simultaneously, and the aggregate frame-MSE → SegNet-argmax-flip cost on
a memorized renderer's early layers dwarfs the rate saving at EVERY operating point tested.

### Why the decoder axis is HARD (the deep finding, permanent knowledge)
1. **The decoder is already at the entropy floor** (brotli ~98.6% of iid per-tensor Shannon) AND
   **already INT8** — so the coding-axis levers are exhausted and the only lever is distortion.
2. **The distortion lever is dominated by SegNet sensitivity on a memorized renderer.** Unlike a
   generalizing codec, every decoder weight is overfit to the 1200 contest frames; there is no
   "redundant precision" to shave. The `100·d_seg` coefficient makes even a 2× d_seg rise (the
   gentlest coarsening) cost +0.069 — an order of magnitude more than any plausible rate gain.
3. **The 91%/9%/0.5% byte budget is NOT a 91% opportunity** — it is 91% of bytes carrying 99.98% of
   score-sensitivity. The decoder bytes are expensive BECAUSE they are score-critical. This inverts
   the naive "attack the biggest section" intuition: the biggest section is biggest because it is
   doing the most score work.

### Routing: DEFER-pending-research (per CLAUDE.md "Forbidden premature KILL")
The decoder-axis distortion lever is NOT killed; the naive-uniform-coarsening implementation is.
Reactivation criteria (a NEW decoder lever, each tests a distinct assumption):
1. **Score-aware PER-WEIGHT quant** (not per-tensor uniform): coarsen ONLY individual weights whose
   per-weight score-gradient is below threshold, keeping high-sensitivity weights at full INT8.
   Requires re-running gradient extraction to emit per-WEIGHT (not per-byte-uniform-spread) grads.
   Tests: "is the distortion concentrated in a sparse high-sensitivity weight subset?" If yes, a
   structured-sparse coarsening could shave rate at far lower d_seg cost.
2. **QAT / LSQ fine-tune AFTER coarsening** (recover the d_seg loss by retraining the decoder at the
   coarser precision). This is the canonical low-bit-PTQ-collapse fix (CLAUDE.md "QAT pipeline").
   Tests: "is the coarsened decoder's d_seg recoverable by fine-tuning?" — the memorized-renderer
   nature suggests yes, but it requires the full score-aware training loop, not a byte-transform.
3. **Latent-axis instead of decoder-axis** (the 15,387 B latent blob, 8.6% of bytes, carries only
   ~0.02% of sensitivity per the map) — a far more favorable sensitivity-per-byte target, though
   small in absolute bytes. Tests: "is there free rate in the latents?" (R3 selector + this map both
   point at latents as the lowest-sensitivity remaining surface).

### The frontier remains 0.19198534 [contest-CPU]; no candidate promoted; no submission.

### Wire-in (Catalog #125)
- **Hook #1 sensitivity-map:** the per-tensor byte×sensitivity map (`per_tensor_sensitivity_map.json`)
  is a NEW decoder-axis sensitivity surface (bottom-decile per-byte = the 3 large early-layer tensors;
  top-decile = the tiny RGB output heads). Permanent finding: the decoder holds 99.98% of total |grad|.
- **Hook #3 bit-allocator:** the per-tensor + per-stream score-weight-per-byte ranking IS a
  bit-allocator prior — but the exact eval proves per-byte-VALUE sensitivity ranking is NECESSARY but
  INSUFFICIENT for a quant decision; the allocator must use per-WEIGHT score-gradient + a fine-tune
  recovery model, not per-tensor uniform coarsening.
- **Hook #5 continual-learning:** 3 exact contest-CPU `tac.action_effect.v1` rows minted via
  `ingest_exact_eval_to_candidate.py` (all `above_frontier`, `pays_rent=False`) — the FIRST
  decoder-axis (not selector-axis) PR110++ exact rows; reseeds the V3 ΔS-judge that decoder uniform
  q-coarsening is dominated.
- **Hook #6 probe-disambiguator:** RESOLVED "is the frontier decoder quant-coarsenable for net rate
  win?" → NO at every operating point tested; the binding constraint is SegNet sensitivity on a
  memorized renderer, not rate. The disambiguator also resolved (advisory) that the local macOS-CPU
  predicted band correctly prices large decoder distortion changes.
- **Hook #2 Pareto:** confirms the frontier is on its decoder-distortion Pareto edge — d_seg cannot be
  traded for rate at this operating point via uniform coarsening; the decoder is rate-AND-distortion
  saturated for naive q-mutation.
- **Hook #4 cathedral-autopilot:** N/A — no promotable archive emitted (all candidates above frontier).

### Cost
Per-tensor map + RD probe + local smoke = $0 (macOS-CPU). 4 Modal CPU evals (frontier + 3 candidates)
≈ $0.5–1.0, under the $5 STOP gate. First dispatch batch failed fast at the runtime-tree preflight
(rc=1, no compute spent) — re-dispatched with `--expected-runtime-tree-sha256 auto`.

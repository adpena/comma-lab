# Independent adversarial audit — the 5 Layer-2 in-curriculum levers (2026-06-12)

**Reviewer:** independent audit subagent (author ≠ reviewer). The building agent landed
`a62a5adc6` + `27731123a` and SELF-reported 97 tests pass; this is the independent verification.
**Scope:** REVIEW + VERIFY ONLY. No source edited. No daemon touched. This memo is the only file written.
**Authority:** every in-loop / synthetic number cited here is `[macOS-CPU advisory]` NON-PROMOTABLE; the
levers land MEANS, the frontier is UNMOVED (`0.19109982`, pointer `.omx/state/canonical_frontier_pointer.json`).

## TL;DR verdicts

| Lever | Verdict | One-line basis |
|-------|---------|----------------|
| 1 — differentiable brotli-rate surrogate | **REAL** | Computes a true order-1 conditional weight entropy `H(W_i|W_{i-1})` + latent-delta entropy, gradient-carrying, ADDED to the loss when `rate_lambda_*>0`; independently reproduced the true-bound + smoothness mechanism. |
| 2 — score-domain seg surrogate + T-anneal | **REAL** | Routes through `segnet_surrogate_per_pixel` (real differentiable argmax-flip); the anneal is a real cosine T-schedule actually threaded `epoch_in_stage → seg_temperature_for_epoch → seg loss` in BOTH the non-split and split paths. The "comment-only contract" fix is real. |
| 3 — pose-FiLM store | **REAL** | FiLM identity-at-init (zero-init fc2), real per-pair pose-conditioned render, additive pose codec section round-trips through a numpy-portable inflate; byte-identical when OFF. |
| 4 — score-aware QAT | **REAL (mechanism)** but with a **disclosed indirect-effect train/deploy gap (MEDIUM)** | Per-tensor `‖∂S/∂w_t‖` EMA is really accumulated from `w.grad` and really changes the per-tensor INT8 grid; falls back bit-identically to uniform when sensitivity is empty/uniform. BUT the codec ALWAYS encodes at 127 levels — the byte win is an *unproven indirect* "train coarse-grid-robust → brotli-friendly repeated symbols" effect, NOT a grammar change. Honestly disclosed in the module docstring. |
| 5 — margin-weighted seg promotion | **REAL** | `exp(−margin/τ)` weight is real, monotone-decreasing in the SegNet top1−top2 margin, reuses the already-forwarded `seg_out` (no extra scorer pass), composes on top of Lever 2. |

**Daemon-safety verdict: SAFE to crash-resume.** **Full-stack-synergy verdict: COHERENT with two honest gaps to flag (Lever-1 scan-order proxy, Lever-4 indirect-effect).**
**Test run: 97 passed in 100.5s, 0 failures, 0 skips, 0 warnings** — the 97-pass claim is CONFIRMED.

---

## A. NO-FAKE per-lever (does it do the work it names, on real inputs?)

### Lever 1 — rate surrogate — REAL
- `src/tac/losses/rate_surrogate.py:114-174` `conditional_weight_entropy` builds a soft 2-D joint
  histogram of adjacent zigzag-INT8 symbols and computes `H(cur|prev) = Σ_a J[a,·]·(−Σ_b cond·log2 cond)`.
  This is a real differentiable entropy, NOT a marker. The INT8 grid (`{-127..127}`, σ=0.2) matches the
  codec/cat_entropy_v2 (`:54-56,68`).
- `:177-232` `latent_delta_entropy` maps latents per-dim to a uint8 grid (matching the codec's per-dim
  minmax), takes the 1st-order temporal delta, and computes the soft-hist entropy. Real, grad-enabled.
- **Wired** into the loss: `driver.py:_weight_regularizers:693-710` adds `rate_lambda_w·h_cond +
  rate_lambda_lat·r_lat`; called from both the non-split path (`:626-628`) and the split path (`:611-614`,
  added as a separate scalar backward into the same `.grad` buffers). Not a no-op.
- **Independent reproduction (my probe, $0 CPU):** on a fresh base_ch=20 decoder `H(W|W_prev)=3.35 ≤
  H(W)_marg=7.94` (true-bound holds); a sorted/smooth weight tensor → cond entropy 2.39 vs a shuffled one
  3.35 (smoother is lower); a smooth latent random-walk → delta entropy 4.92 vs iid 7.92. The mechanism is
  REAL — these would be impossible for a placeholder.

### Lever 2 — seg surrogate + anneal — REAL
- `driver.py:_seg_loss_for_spec:86-184` routes `seg_surrogate is None` to the EXACT vendored
  `spec.seg_loss_fn(...)` (default-preserving) and a surrogate name to `tac.losses.core.
  segnet_surrogate_per_pixel` (`core.py:425-479`, a real differentiable `1−softmax(pred/T)[gt]` /
  fisher_rao / sinkhorn). The one-hot-GT-logit construction (`:158-164`) is a sound, cache-free way to keep
  GT hard while only the prediction is temperature-softened.
- **The "comment-only contract" fix is REAL.** `curriculum.py:223-251` `seg_temperature_for_epoch` is a real
  cosine anneal `t0 + (t1−t0)·½(1−cos(π·e/(epochs−1)))`, clamped. It is actually threaded:
  `driver.py:531-536` computes `epoch_temperature` per epoch, `:604` and `:621` pass it into the seg loss,
  and the `run()` loop passes the live `epoch` as `epoch_in_stage` (`:1220-1225`). NOT a new comment-only
  contract — the value flows end-to-end. Test `test_lever2_anneal_actually_changes_temperature_per_epoch`
  asserts start=1.0, end=0.05, monotone, and `max−min > 0.5` (a constant would FAIL).

### Lever 3 — pose-FiLM — REAL
- `pose_film.py:58-205` is a real FiLM: `sin(fc1)→fc2(zero-init)→(γ=1+tanh, β)`, identity at init, injected
  on the stem before `sin(x)` and the vendored cascade is replicated EXACTLY (not edited). The stored pose
  is a real `(n_pairs,6)` buffer set from GT (`set_stored_pose`), looked up per pair index.
- Real additive codec section (`:283-353`, per-dim minmax→uint8→delta→zigzag→brotli), round-trips through a
  numpy-portable `inflate_film_decoder` (`:401-449`). Tests prove identity-at-init, trained-FiLM render
  divergence, pose-conditioning, byte-closed round-trip, and eval==inflate render parity.

### Lever 4 — score-aware QAT — REAL mechanism (with a MEDIUM synergy gap, §D)
- `score_aware_qat.py:216-237` `accumulate_tensor_sensitivity` really computes `s_t = ‖w.grad‖.norm()` and
  EMA-smooths it (decay=0 → exact norm, verified). `driver.py:637-642` calls it AFTER the score-domain
  backward (grads still live) ONLY when `score_aware_qat` is on.
- `:128-168` `per_tensor_levels_from_sensitivity` rank-normalizes sensitivity → a per-tensor level count in
  [0.5×,1.0×]·127; high-sensitivity → finer grid. `:171-204` `apply_score_aware_qat` really varies the
  per-tensor quant. `_fake_quantize_n(w,127)` is BIT-IDENTICAL to the vendored `fake_quantize` (confirmed
  against the vendored source). `sensitivity is None`/uniform → 127 for every tensor (bit-identical
  fallback, verified). The mechanism is REAL.

### Lever 5 — margin weight — REAL
- `driver.py:71-83` `_segnet_logit_margin_map` is a real detached `top1−top2` logit map; `:173-183` weights
  the per-pixel surrogate by `exp(−margin/τ)` before the mean, reusing the already-forwarded `seg_out` (no
  extra scorer pass). Monotone-decreasing in margin (test + my read confirm).

---

## B. Byte-identity / daemon-safety verdict: **SAFE to crash-resume.**

The control basin (pid 33911) and combined-L2 arm (pid 61913) run off committed code. If either
crash-resumes onto this code with its existing (all-default) config, the update is byte-for-byte unchanged.
Evidence I verified:

1. **`test_default_train_epoch_matches_vendored_only_reference`** is SOUND. It runs a full
   `_train_one_epoch` with an all-default spec, then a hand-rolled reference epoch that touches ONLY
   vendored ops (same RNG pin `manual_seed(123)`, same randperm, same vendored forward/loss/clip/step/EMA),
   and asserts `torch.equal` on EVERY post-epoch decoder tensor AND the latents. This is the strongest
   possible proof: it would fail if any lever silently mutated the default forward/backward.
2. The default code paths are genuinely inert:
   - `_weight_regularizers` returns `None` when `cat_lambda==0 ∧ rate_lambda_w==0 ∧ rate_lambda_lat==0`
     (`:711-712`); on the C1a-only path it returns the EXACT legacy `cat_lambda·cat_entropy_v2` tensor
     (verified bit-equal under a pinned RNG by `test_weight_regularizers_c1a_only_matches_legacy_tensor`).
   - `_seg_loss_for_spec` with `seg_surrogate is None` returns the raw vendored call (`:136-140`),
     verified bit-equal in value AND gradient (`test_default_seg_surrogate_*`).
   - `use_qat ∧ ¬score_aware_qat` → vendored `apply_qat` (`:571-572`); `score_aware_qat ∧ empty EMA` →
     uniform 127 fallback (bit-identical).
   - `pose_film_enabled=False` → `_new_decoder` returns the bare vendored decoder (`:436-437`) and
     `_build_archive_and_eval_decoder` returns EXACTLY `v.build_archive(...)` (`:881-889`), verified
     byte-equal to the legacy path (`test_driver_pose_film_off_builds_byte_identical_vendored_archive`).
3. **Two all-default driver runs are bit-identical** (best-score + best-archive bytes), and FiLM-off runs
   are deterministic.

**The one path I checked for a hole and found CLEAN:** the anneal default. `seg_temperature_end is None`
→ `seg_temperature_for_epoch` returns the static T for every epoch (`curriculum.py:240-241`), and the
default `seg_surrogate is None` means the anneal value never even reaches a live surrogate. No divergence.

**No byte-divergence path found on any all-default config.** The daemon is safe.

---

## C. Tests verify BEHAVIOR not constants — CLEAN (no forbidden-class-2 fakes found)

I audited every lever test against the "would it still pass if the body were `return baseline`/`return
marker`?" criterion. All pass the bar:

- Lever 1: `test_lever1_rate_term_changes_loss_and_has_gradient` asserts a positive rate term AND a non-zero
  decoder-weight gradient (a no-op gives None/zero — FAILS). `test_lever1_conditional_entropy_is_below_
  marginal_true_bound` is a real inequality on computed entropies.
- Lever 2: `test_soft_cosine_surrogate_matches_hand_computed_argmax_flip` requires equality to an
  INDEPENDENT hand-computed `1−softmax(pred/T)[gt]`; `..._differs_from_cross_entropy` requires it NOT equal
  CE (a relabelled-CE fake FAILS); anneal test requires `max−min>0.5` (a constant FAILS).
- Lever 4: `test_lever4_nonuniform_sensitivity_changes_quant_grid` requires the quantized weight to DIFFER
  from uniform; `..._uniform_..._matches_vendored` requires bit-identity. `..._ema_accumulates_from_grad`
  checks `ema==‖grad‖` exactly (decay=0).
- Lever 5: `..._changes_seg_loss` requires the weighted loss ≠ unweighted; `..._monotone_decreasing_in_
  margin` checks the boundary pixel gets MORE weight than the interior pixel.
- Compose: `test_compose_all_five_loss_differs_from_all_default` requires the all-five loss ≠ all-default
  (a silently-inactive composition FAILS).

These are genuine behavior tests. I found **no** test that verifies only canonical constants/markers.

---

## D. Composability + FULL-STACK SYNERGY verdict: COHERENT, with two honest gaps

**Composition is REAL.** `test_compose_all_five_levers_end_to_end` runs all 5 (1+2+anneal+3+4+5+C1a)
through one forward/backward/export to a DONE marker, byte-closes an archive WITH the pose section, and
parses it back. The split-by-head path also threads the temperature and adds the rate regularizer as a
separate backward into the same grad buffers (`driver.py:602-614`) — verified the scorer context exposes
`seg_forward_train`/`pose_forward_authority`/`split_by_head` for both Real and Synthetic contexts.

**No double-counting between Lever 1 and C1a.** They are additive, DISTINCT quantities: C1a penalizes the
memoryless marginal `H(W)`; Lever 1 penalizes the order-1 conditional `H(W|W_prev)` (a tighter, lower
bound) + the latent-delta entropy (which C1a never touches). Both can be on (`_weight_regularizers` sums
them) without redundancy — they target different redundancy structure. CLEAN.

**Cross-layer gaps to flag (both honestly disclosed by the building agent — MEDIUM, not HIGH):**

1. **Lever-1 scan-order proxy (MEDIUM, disclosed).** The conditional entropy is computed over the FIRST
   2000 contiguous weights of each tensor in `named_modules()` flatten order (`rate_surrogate.py:155-160`).
   The REAL brotli stream that Layer-1 (the carrier) compresses is the codec's `encode_decoder` byte stream
   (zigzag-INT8, per-tensor), whose LZ window is NOT the state-dict iteration order. So the surrogate trains
   against an *abstract* order-1 entropy, not the exact density the deployed coder consumes. The module
   defends this as "a true lower bound regardless of scan order ⇒ conservative" (`:21-23,72`), which is
   mathematically correct (conditioning never increases entropy) — but "conservative lower bound" means the
   surrogate can be SLACK (it may push smoothness brotli does not actually reward, or miss redundancy brotli
   captures across the real byte order). This is a genuine train/deploy proxy gap, mitigated only by the
   mandatory paired A/B (archive_bytes at equal d_seg/d_pose) the memo gates on. Not a fake — a known proxy.

2. **Lever-4 indirect-effect train/deploy gap (MEDIUM, disclosed).** I confirmed the codec ALWAYS quantizes
   every tensor at 127 levels (`quantize_state_dict`, `N_QUANT=127`). So score-aware QAT does NOT change the
   archive grammar (correctly claimed FORK_PRINCIPLED-but-grammar-compatible). The byte win is therefore
   purely INDIRECT: training a low-sensitivity tensor to be robust at a *coarser* (e.g. 64-level) grid is
   hypothesized to make its 127-level codec encoding collapse to more repeated symbols brotli loves. This
   is plausible (Cover&Thomas reverse water-filling) but is an UNPROVEN second-order effect — the coarser
   training grid is never the deployed grid. The module docstring states this explicitly (`:32-37`). It is
   the weakest of the five EV-wise (the memo itself ranks it #4, "sharpens Lever 1, opens no new axis"). The
   A/B (uniform-QAT vs score-aware-QAT, archive_bytes at equal distortion) is the only validation; until it
   lands, the byte claim is a prediction.

Neither gap is a NO-FAKE violation: both are honestly tagged predictions with the A/B as the empirical
bit-spend proof (per Catalog #304), and both modules carry `SCORE_CLAIM=False`.

---

## E. Test-run result

```
.venv/bin/python -m pytest src/tac/torch_vehicle/tests/ src/tac/tests/test_rate_surrogate.py -q
→ 97 passed in 100.50s (0:01:40)
```
0 failures, 0 skips, 0 warnings. The 97-pass self-report is CONFIRMED independently.

---

## F. Prioritized fix list (for the main agent — NOT applied here)

- **HIGH:** none. No NO-FAKE violation, no daemon-safety hole, no fake test found.
- **MEDIUM-1 (Lever 1 scan-order):** before claiming any rate win, validate the conditional-entropy proxy
  against the ACTUAL codec byte stream — e.g. add a $0 offline probe that correlates `Δh_cond` with the real
  `Δlen(encode_decoder(quantize_state_dict(sd)))` over a few decoder checkpoints. If the correlation is
  weak, the surrogate is training the wrong density. (The paired A/B catches this, but a cheap pre-A/B probe
  saves a GPU arm.)
- **MEDIUM-2 (Lever 4 indirect effect):** the first Lever-4 A/B should report archive_bytes BOTH with and
  without the score-aware training (uniform-QAT vs score-aware-QAT), and additionally measure the brotli
  decoder-blob size delta directly, to confirm the "coarse-grid-robust → repeated symbols" hypothesis is
  real and not noise. Deploy Lever 4 ONLY after Lever 1 confirms a frontier (it has nothing to sharpen
  otherwise), per the memo's own ranking.
- **LOW-1 (Lever 1 cost):** `conditional_weight_entropy` + `latent_delta_entropy` run once per BATCH inside
  `_weight_regularizers` (called from the per-batch loop, `driver.py:611/626`), NOT once per epoch as the
  design memo §Lever-1 wire-in and the landing memo's "computed once per epoch (global quantity, like C1a)"
  both claim. C1a (`cat_entropy_v2`) is also per-batch today, so this matches the existing cadence and is
  default-OFF, but the docstring/memo "once per epoch" wording is inaccurate — the global rate term is
  recomputed every batch. Harmless to correctness; a perf/clarity nit (consider memoizing per epoch if the
  cost matters at base_ch=20 scale).
- **LOW-2 (Lever 3 FiLM regularization):** when both Lever 1 and Lever 3 are on, the rate surrogate iterates
  the `pose_film.*` Linear layers too (they ship in the decoder blob). The driver comment (`:699-702`) says
  this is "correct" — agreed, but note the FiLM weights are tiny and untrained-at-init (zero fc2), so early
  in training their conditional entropy is degenerate; verify no NaN/inf from the near-constant FiLM tensors
  feeding the soft-hist (the `max_abs_floor` skip at `rate_surrogate.py:152-154` should handle the zero-init
  fc2, but confirm under the compose-all-five long run).

---

## Wire-in / provenance (this is a REVIEW memo, no code lands)

6-hook (Catalog #125): all N/A — this is an audit memo, not a code/score landing (the levers' own hooks are
declared in the landing memo). Mission contribution: `rigor_overhead` (independent verification that the
MEANS are real + daemon-safe; the END remains a lower exact score, frontier UNMOVED). Authority: all numbers
`[macOS-CPU advisory]` NON-PROMOTABLE. No GPU launched, no daemon touched, no source edited.

---

## APPEND-ONLY (2026-06-12) — fix-list closure (consolidated R3-prep pass)

Per HISTORICAL_PROVENANCE the body above is UNCHANGED. The §F prioritized fix list is now
RESOLVED (settled lever code at HEAD; no collision). Per-gap verdict + MEASURED numbers:

- **MED-1 (Lever-1 scan-order) → FIXED.** Built `experiments/probe_lever1_entropy_vs_real_brotli.py`:
  correlated the conditional entropy vs REAL `len(encode_decoder(quantize_state_dict(sd)))`
  over 8 real weight configs. As-wired (legacy per-tensor weights-only, 2000-trunc) =
  Spearman **-0.14**; deploy-faithful codec-scan-order (full state_dict, weights+biases,
  one stream) = Spearman **0.90** / Pearson **0.999**. FIX: `RateSurrogateConfig.codec_scan_order`
  + driver uses it. The "true lower bound regardless of scan order ⇒ conservative" defense
  was mathematically correct but IRRELEVANT to the real defect — the per-tensor MODE
  measured the wrong subset/order so its VALUE did not rank-track real bytes. The fix is
  the deploy-faithful scan order. 3 new tests. Byte-identity-of-default preserved.

- **MED-2 (Lever-4 indirect effect) → SUPPORTED (codec axis validated).** Built
  `experiments/probe_lever4_qat_brotli_blob_delta.py`: REAL `||∂S/∂w||` sensitivity (one
  frozen-scorer backward, 8 0.mkv pairs) → score-aware grid → **-3263 B smaller** real
  brotli decoder blob (70264 vs 73527) at EQUAL advisory d_seg (0.0034 → 0.0034). Snap
  survives the codec 127-requant (214 → 113 distinct symbols on blocks.0.weight). HONEST
  caveat landed in the Lever-4 docstring: the byte DIRECTION is validated; the net-SCORE
  win still needs the training A/B (uniform-QAT-trained vs score-aware-QAT-trained) + dual
  exact eval. 2 new codec-mechanism tests. NOT shipped as a confirmed SCORE lever.

- **LOW-1 (per-batch not per-epoch) → CORRECTED** in the landing memo (append-only note).

- **LOW-2 (zero-init FiLM fc2 in the rate soft-hist) → already CLOSED by R2** (the
  `max_abs_floor` skip; 80-epoch compose-all-five run finite). No new action.

VERDICT: no gap left ambiguous (MED-1 FIXED, MED-2 SUPPORTED-with-honest-scope, LOW-1
CORRECTED, LOW-2 CLOSED-by-R2). Authority: all `[macOS-CPU advisory]` NON-PROMOTABLE;
frontier UNMOVED.

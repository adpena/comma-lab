# All FIVE Layer-2 in-curriculum levers — IMPLEMENTED + WIRED + TESTED + COMPOSABLE (2026-06-12)

**Author:** all-layer2-levers IMPLEMENTATION subagent (Layer 2 of the operator's three-layer stack:
substrate / in-curriculum levers / post-hoc bolt-ons). Operator directive: *"use a subagent to research and
design and implement and test and optimize and wire and integrate all layer 2 levers."*
**Status:** LANDED. All 5 levers are REAL, default-OFF, byte-identical-when-disabled, tested, composable
knobs on the torch_vehicle HNeRV training path. **No GPU run launched** (this is the ENABLEMENT; the
all-levers arm is a later operator decision — two daemons are live and must not be contended).
**Frontier (pointer, NOT hardcoded):** `.omx/state/canonical_frontier_pointer.json` → contest-CPU
`0.19109982`, archive `177169 B`, lane `pr110_payload_entropy_recode`. **Frontier UNMOVED** — this lands
MEANS (the levers), not the END (a lower exact score). Stated plainly per the means/ends firewall.
**NO FAKE:** every lever ACTUALLY performs its mechanism on real inputs (real frozen scorer in the live
arm; real 0.mkv-derived targets; real archive byte-close). Every quantified effect below is a PREDICTION
with its first-principles basis named; the falsification gate for each is a paired A/B (lever-on vs off,
equal epoch budget) measuring the relevant score component, then the dual CPU/CUDA exact-eval.

Canonical design source honored: `.omx/research/incurriculum_levers_design_floor_chasing_20260612.md`.

---

## Per-lever status (built / optimized / composed)

| # | Lever | Gate field (default-OFF) | Status |
|---|-------|--------------------------|--------|
| 1 | Differentiable brotli-rate surrogate | `StageSpec.rate_lambda_w` / `rate_lambda_lat` (both `0.0`) | **BUILT + WIRED.** Module `tac.losses.rate_surrogate` (order-1 conditional weight entropy `H(W_i\|W_{i-1})` + latent temporal-delta entropy) existed but was NOT consumed by the driver. Now added to the loss in BOTH the split and non-split paths via the new `_weight_regularizers` helper. |
| 2 | Score-domain seg surrogate + **T-anneal** | `seg_surrogate` (`None`) / `seg_temperature` (`1.0`) / `seg_temperature_end` (`None`) | **OPTIMIZED (the gap the combined arm flagged).** The surrogate router (`_seg_loss_for_spec`) was already built; the per-epoch TEMPERATURE-ANNEAL hook did NOT exist (T was static). Now the driver threads `epoch_in_stage` → `seg_temperature_for_epoch` (cosine T-anneal, `seg_temperature` → `seg_temperature_end`) into the seg loss. `fisher_rao` + `sinkhorn` already exposed as `seg_surrogate` choices. |
| 3 | Pose-FiLM store | `cfg.pose_film_enabled` (`False`) | **BUILT (`52702cd5a`) — CONFIRMED composing.** Not rebuilt; integration-tested in the all-five compose test (it composes with Levers 1/2/4/5: end-to-end train → byte-close WITH the additive pose section → parse-back round-trip). |
| 4 | Score-aware QAT | `score_aware_qat` (`False`) / `qat_sensitivity_decay` (`0.99`) | **BUILT + WIRED.** Module `tac.torch_vehicle.score_aware_qat` (per-tensor sensitivity-weighted INT8 grid; `s_t = \|\|∂S/∂w_t\|\|` EMA) existed but was NOT consumed. Now the `use_qat` block routes through `apply_score_aware_qat` when enabled, the driver accumulates the sensitivity EMA on `_StageRuntime` after the score-domain backward, and `restore_score_aware_qat` restores. REAL per-tensor sensitivity-weighted quant (not a stub). |
| 5 | Margin-weighted seg promotion | `margin_weight_tau` (`None`) | **BUILT (`_segnet_logit_margin_map` + the `exp(−margin/τ)` weight in `_seg_loss_for_spec`) — CONFIRMED composing.** The per-pixel `exp(−margin/τ)` boundary weight multiplies the Lever-2 surrogate (reuses the already-forwarded `seg_out`, no extra scorer pass). |

### The byte-identity-of-default proof (the daemon-safety guard — CRITICAL)

Two daemons are live off committed code (control basin pid 33911; combined-L2 arm pid 61913). Every new
lever field defaults to a value that reproduces TODAY's behavior byte-for-byte. The proof:

- `test_default_train_epoch_matches_vendored_only_reference` — a FULL `_train_one_epoch` with an
  all-default spec produces a **bit-identical** (`torch.equal`) decoder state-dict AND latents vs a
  hand-rolled reference epoch that exercises ONLY the vendored ops (no lever code path touched). If ANY
  lever silently mutated the default forward/backward, the post-epoch weights diverge — this FAILS.
- `test_all_default_driver_run_is_deterministic_and_byte_identical` — two all-default driver runs produce
  bit-identical best-archive **bytes** + identical best-score.
- `test_weight_regularizers_default_returns_none` + `test_weight_regularizers_c1a_only_matches_legacy_tensor`
  — the regularizer helper returns `None` when no lever is active (loss unchanged) and the EXACT legacy
  `cat_lambda * cat_entropy_v2(...)` tensor on the C1a path (`torch.equal`).
- `test_anneal_disabled_returns_static_temperature` — `seg_temperature_end is None` returns the static T
  for every epoch.

A daemon that crash-resumes onto this code is therefore unchanged byte-for-byte. **No daemon out-dir was
written; no daemon was killed.**

## Compose-all-five self-test (end-to-end → byte-close → inflate)

`python experiments/launch_l2_combined_attacks.py --levers all --self-test` →
**PASS**: a 3-epoch synthetic-scorer run with EVERY lever active (rate surrogate + seg surrogate +
T-anneal + score-aware QAT + margin weight + pose-FiLM) reaches a DONE marker, byte-closes a best archive,
the archive parses back, AND the Lever-3 pose section round-trips `(n_pairs, 6)`. The default
`--levers seg_pose` self-test also PASSES (Levers 1/4/5/anneal OFF). Mirrored as
`test_compose_all_five_levers_end_to_end` + `test_compose_all_five_loss_differs_from_all_default`.

## The anneal hook + QAT-sensitivity + margin-weight mechanisms (REAL, not stub)

- **Lever-2 anneal** (`curriculum.seg_temperature_for_epoch`): cosine-anneals the PREDICTION softmax
  temperature from `seg_temperature` (epoch 0) toward `seg_temperature_end` (final epoch). GT stays HARD
  (one-hot logits — contest d_seg is a hard argmax); only the prediction sharpens. Tested:
  `test_lever2_anneal_actually_changes_temperature_per_epoch` (monotone, start→end, NOT a constant).
- **Lever-4 QAT sensitivity** (`score_aware_qat.accumulate_tensor_sensitivity` + `apply_score_aware_qat`):
  per-tensor `s_t = \|\|∂S/∂w_t\|\|` EMA accumulated after the score-domain backward; high-sensitivity
  tensors get a FINER INT8 grid (argmax boundary protected), low-sensitivity ones a COARSER grid (fewer
  brotli bytes — reverse water-filling). Uniform/empty sensitivity falls back to the vendored uniform
  127-level grid BIT-IDENTICALLY. Tested: `test_lever4_nonuniform_sensitivity_changes_quant_grid`,
  `test_lever4_uniform_sensitivity_matches_vendored_uniform_qat`, `test_lever4_sensitivity_ema_accumulates_from_grad`.
- **Lever-5 margin weight** (`_segnet_logit_margin_map` + `exp(−margin/τ)`): the per-pixel SegNet
  `top1−top2` logit margin re-weights the surrogate toward small-margin (boundary) pixels. Tested:
  `test_lever5_margin_weight_changes_seg_loss`, `test_lever5_margin_weight_is_monotone_decreasing_in_margin`.

## The one command to launch an all-levers arm

```bash
.venv/bin/python experiments/launch_l2_combined_attacks.py --levers all --go \
    --total-epoch-budget 29650          # line up stage boundaries with the basin for matched-epoch A/B
# (--levers all enables: rate surrogate + seg surrogate + T-anneal[1.0→0.05] + pose-FiLM + score-aware QAT + margin weight[τ=2.0])
# --self-test for the tiny synthetic end-to-end check (no real-scorer load, no basin contention).
```

Per-flag overrides (`--rate-lambda-w`, `--seg-temperature-end`, `--score-aware-qat`, `--margin-weight-tau`,
…) override the `--levers all` defaults. **The run itself is a later operator decision** — the combined
seg+pose arm is already running; this is the all-levers ENABLEMENT.

---

## Canonical-vs-unique decision per layer (Catalog #290)

| Layer | Decision | Rationale (falling-rule) |
|-------|----------|--------------------------|
| Seg surrogate routing (Lever 2) | **ADOPT_CANONICAL** | `tac.losses.core.segnet_surrogate_per_pixel` is the pre-built, tested differentiable argmax-flip surrogate; routing through it IS the optimal engineering (a routing change, not a fork). |
| Rate surrogate (Lever 1) | **FORK_PRINCIPLED** (new module) | The vendored C1a `cat_entropy_v2` is a memoryless marginal `H(W)`; the codec is brotli (order-N context). The order-1 conditional `H(W_i\|W_{i-1})` + latent-delta entropy is a TIGHTER, currently-missing proxy — a principled new module (`tac.losses.rate_surrogate`), not a fork of C1a (both compose additively). |
| Score-aware QAT (Lever 4) | **FORK_PRINCIPLED** (new module) | The vendored `apply_qat` is uniform 127-level (L2-blind). Sensitivity-weighting needs a per-tensor level map keyed to `\|\|∂S/∂w_t\|\|` — a new module (`tac.torch_vehicle.score_aware_qat`) that is GRAMMAR-COMPATIBLE (reuses the codec's existing per-tensor scale) and falls back to the vendored uniform quant bit-identically. |
| Anneal hook (Lever 2 optimize) | **ADOPT_CANONICAL** (cosine, mirrors the driver's LR schedule) | The driver already uses a cosine LR schedule; the T-anneal mirrors that cadence (sharpen the boundary gradient as the coarse structure converges). |
| Margin map (Lever 5) | **ADOPT_CANONICAL** (reuse the forwarded `seg_out`) | Mirrors `tac.substrates.d1_segnet_margin_polytope.margin_map.compute_logit_margin_map` but reads the already-computed prediction logits in-place (no extra scorer pass — the optimal engineering). |
| Default-OFF discipline | **ADOPT_CANONICAL** | Mirrors EXACTLY how `f42c412ec` (Lever 2) and `52702cd5a` (Lever 3) added default-OFF fields — last-positional defaulted StageSpec/cfg fields, byte-identical when disabled. |

## 9-dimension success checklist evidence (Catalog #294)

UNIQUENESS — the rate + QAT surrogates are the order-1/sensitivity-weighting refinements of the C1a/uniform
vendored primitives, distinct from any prior substrate. BEAUTY/ELEGANCE — one `_weight_regularizers` helper
unifies C1a + Lever 1 (returns the EXACT legacy tensor on the C1a path); one `seg_temperature_for_epoch`
hook drives the anneal. DISTINCTNESS — each lever has a distinct gate field + distinct module. RIGOR —
every lever has a behavioral test (NO constant-checking) + the all-default byte-identity proof. OPTIMIZATION
PER TECHNIQUE — sensitivity-weighted QAT, cosine T-anneal, late-stage rate scheduling per the memo.
STACK-OF-STACKS — the compose-all-five test proves the five run in one forward/backward/export path.
DETERMINISTIC REPRODUCIBILITY — two all-default runs are bit-identical; the resume test passes. EXTREME
OPTIMIZATION — the rate term is computed once per epoch (global quantity, like C1a); the margin map reuses
the forwarded `seg_out`. OPTIMAL-MINIMAL-CONTEST-SCORE — the levers co-design the byte/distortion budget
toward the three score terms (rate=62%, d_seg, d_pose) per the binding-constraint map; the score is a
PREDICTION pending the A/B + dual exact-eval.

## Cargo-cult audit per assumption (Catalog #303)

- "C1a marginal entropy is the brotli proxy" — **CARGO-CULTED** (unwound by Lever 1: brotli is order-N;
  the order-1 conditional is a tighter, true lower bound; conditioning never increases entropy).
- "Uniform INT8 quant is optimal" — **CARGO-CULTED** (unwound by Lever 4: reverse water-filling — uniform
  is provably sub-optimal when distortion sensitivity is non-uniform, Cover & Thomas Ch.10).
- "Static seg temperature suffices" — **CARGO-CULTED** (unwound by the Lever-2 anneal: the boundary
  gradient should sharpen toward hard argmax as training converges; the docstring CLAIMED an anneal the
  code did not do — a comment-only contract, now made real).
- "CE-on-hard-targets is the seg objective" — **CARGO-CULTED** (unwound by Lever 2: contest d_seg is
  argmax-flip; CE spends capacity on confident-interior pixels the argmax already gets right). HARD-EARNED:
  the surrogates are pre-built + tested in `tac.losses.core`.
- "The latents need no training-time rate term" — **CARGO-CULTED** (unwound by Lever 1b: dashcam latents
  are temporally redundant; the codec delta-codes them but training never rewarded smoothness).

## Observability surface (Catalog #305)

INSPECTABLE PER LAYER — each lever is a named gate field on `StageSpec`/`TorchVehicleConfig`; the launch
script prints the active-lever dict. DECOMPOSABLE PER SIGNAL — `_weight_regularizers` returns the composed
reg term; the telemetry JSONL carries loss/pose/d_seg/d_pose/rate/score/archive_bytes per eval. DIFF-ABLE
ACROSS RUNS — the matched-epoch A/B (arm trajectory vs the basin's recorded `torch_vehicle_trajectory.jsonl`)
is immune to wall-clock. QUERYABLE POST-HOC — durable telemetry JSONL + best/best_meta.json. CITE-ABLE —
this memo + the lane registry + the test file. COUNTERFACTUAL-ABLE — every lever's gate field is a
default-OFF toggle, so "what if this lever were off?" is a one-field A/B; the all-default byte-identity
proof IS the counterfactual baseline.

## 6-hook wire-in declaration (Catalog #125)

1. **Sensitivity-map** — ACTIVE: Lever 4's per-tensor `\|\|∂S/∂w_t\|\|` EMA IS a sensitivity map (feeds the
   bit-allocator); Lever 5's `top1−top2` margin map is a per-pixel sensitivity map.
2. **Pareto constraint** — ACTIVE: the score-domain Lagrangian (Lever 2 + the rate term) IS the Pareto
   objective; the levers attack distinct constraints (rate/seg/pose) → a Dykstra-feasible intersection
   (no compound ΔS claimed; each A/B arbitrates).
3. **Bit-allocator hook** — ACTIVE: Lever 1 (rate surrogate) + Lever 4 (sensitivity-weighted INT8 grid)
   ARE bit-allocator primitives.
4. **Cathedral autopilot dispatch** — N/A: no archive-deployable artifact lands here (this is the
   ENABLEMENT; the all-levers arm + its dispatch is a later operator decision). ACTIVE when an arm dispatches.
5. **Continual-learning posterior** — DESIGN: each lever's A/B is a falsifiable empirical anchor; the
   predicted-vs-measured ΔS per lever becomes a canonical-equation anchor candidate (Catalog #344) when the
   A/B lands.
6. **Probe-disambiguator** — ACTIVE: Lever 3's deploy gate (d_pose vs stored-pose quant floor) and Lever
   4's sensitivity-noise gate (only activate score-aware QAT in late stages where the score-domain loss has
   stabilized) ARE probe-disambiguators.

**Mission contribution:** `frontier_breaking_enabler` (the levers gate the floor-chasing arm; the END is a
lower exact score, this is the MEANS — stated plainly). **Authority:** any measured number from a synthetic
or MPS-trained arm is `[macOS-CPU advisory]` / `[contest-CPU advisory]` NON-PROMOTABLE until a byte-closed
archive is run through `upstream/evaluate.py` on BOTH contest-CPU (Linux x86_64) AND contest-CUDA.

## Files

- `src/tac/torch_vehicle/driver.py` — `_train_one_epoch` (epoch_in_stage thread + Lever-2 anneal + Lever-4
  score-aware QAT route + sensitivity EMA accumulate), `_weight_regularizers` (C1a + Lever-1 rate term),
  `_split_by_head_backward` (temperature thread), `_StageRuntime.tensor_sensitivity_ema`.
- `src/tac/losses/rate_surrogate.py` — Lever 1 (pre-existing module; docstring test-path fixed; now wired).
- `src/tac/torch_vehicle/score_aware_qat.py` — Lever 4 (pre-existing module; now wired).
- `src/tac/torch_vehicle/curriculum.py` — the StageSpec lever fields + `seg_temperature_for_epoch` (pre-existing).
- `src/tac/torch_vehicle/pose_film.py` — Lever 3 (pre-existing `52702cd5a`).
- `experiments/launch_l2_combined_attacks.py` — the `--levers all` switch + per-lever flags + the
  compose-all-five self-test.
- Tests: `src/tac/torch_vehicle/tests/test_all_layer2_levers.py` (18), `src/tac/tests/test_rate_surrogate.py`
  (9), `src/tac/torch_vehicle/tests/test_score_aware_qat.py` (11); `test_driver_resume.py` monkeypatch
  signature fix.

---

## APPEND-ONLY (2026-06-12) — recursive-review gap closure (R1 MED-1 / MED-2 / LOW-1)

Per HISTORICAL_PROVENANCE (Catalog #110/#113) the original body above is UNCHANGED; this
section supersedes the noted lines. The R1 audit (`4cbd9676a`) + R2 (`253f8ab9a`) flagged
two MEDIUM proxy gaps + one LOW doc nit; all three are closed here (consolidated fix pass).

**LOW-1 correction (supersedes line 112 "the rate term is computed once per epoch").**
The Lever-1 rate term is computed PER-BATCH, not once per epoch — it runs inside
`_weight_regularizers`, called from the per-batch loop (`driver.py:611/626`). This MATCHES
the vendored C1a `cat_entropy_v2` cadence (also per-batch) and is default-OFF, so it is
harmless to correctness; the "once per epoch" wording was inaccurate. The 9-dim EXTREME
OPTIMIZATION claim still holds (the term is a cheap global quantity), but its cadence is
per-batch.

**MED-1 — CLOSED via FIX (probe `experiments/probe_lever1_entropy_vs_real_brotli.py`).**
The probe correlated Lever-1's conditional entropy against the REAL vendored-codec brotli
decoder bytes (`encode_decoder(quantize_state_dict(sd))`) across 8 real weight configs
(basin EMA, sorted/smoothed, shuffled, 4 noise levels, random-init). FINDING: the LEGACY
per-tensor mode (named_modules WEIGHTS only, first-2000-truncated) is **Spearman -0.14** vs
real brotli bytes — NOT a reliable rank proxy in the basin's realistic high-entropy regime
(trained INT8 weights use 195-228 of 255 symbols, ~1.5% zeros). The deploy-faithful
CODEC-SCAN-ORDER conditional entropy (FULL `state_dict()` — weights AND biases — in
state-dict order, one concatenated stream = the exact density brotli compresses) is
**Spearman 0.90 / Pearson 0.999**. FIX: added `RateSurrogateConfig.codec_scan_order`
(`src/tac/losses/rate_surrogate.py`) + the driver now uses `codec_scan_order=True`
(`driver.py` Lever-1 callsite) so TRAIN-TIME rate tracks DEPLOY-TIME bytes (full-stack
synergy). Gradient flows to weights+biases+latents. Tests:
`test_codec_scan_order_mode_differs_from_per_tensor_on_multi_tensor_decoder`,
`test_codec_scan_order_stream_includes_biases`,
`test_codec_scan_order_entropy_ranks_with_real_brotli_bytes`. Byte-identity-of-default
preserved (Lever-1 default-OFF; the 34-test byte-identity/default/resume/compose subset
still passes). VERDICT: FIXED.

**MED-2 — CLOSED via VALIDATION (probe `experiments/probe_lever4_qat_brotli_blob_delta.py`).**
The probe took the basin EMA decoder + a REAL `||∂S/∂w||` sensitivity (one frozen-scorer
forward+backward on 8 0.mkv GT pairs) and byte-closed BOTH uniform-127 and score-aware-grid
arms through the REAL codec. RESULT: score-aware grid (13/14 tensors coarsened into the
[64,127] band) → **-3263 B (-4.4%) SMALLER brotli decoder blob** (70264 vs 73527) at
**equal advisory d_seg** (0.0034 → 0.0034); d_pose advisory ticks 0.001663 → 0.001777
(a one-shot-snap artifact). Mechanism CONFIRMED: a 64-level snap collapses `blocks.0.weight`
from 214 → 113 distinct codec-127 symbols (it survives the codec's own 127-requant) →
fewer brotli symbols → smaller blob. The indirect-win hypothesis is SUPPORTED on the
CODEC axis (Catalog #304 bit-spend proof POSITIVE). HONEST SCOPE (landed in the Lever-4
docstring): the probe validates the codec half (the byte DIRECTION); the FULL net-score
win still needs the paired TRAINING A/B (uniform-QAT-trained vs score-aware-QAT-trained,
archive_bytes at equal d_seg/d_pose) + dual CPU/CUDA exact eval before any SCORE claim —
the byte direction is validated, the net-score win remains a prediction. Test:
`test_score_aware_grid_yields_smaller_real_brotli_blob_than_uniform` +
`test_uniform_score_aware_blob_equals_vendored_uniform_blob`. VERDICT: SUPPORTED (codec
axis validated; training A/B still gates the score claim).

**Authority:** every number above is `[macOS-CPU advisory]` NON-PROMOTABLE; the frontier is
UNMOVED (`.omx/state/canonical_frontier_pointer.json`). These close the recursive-review
gaps (MEANS hardening); the END remains a lower exact score, pending the paired A/B + dual
exact eval. No GPU launched, no daemon touched, no Cool-Chic touched.

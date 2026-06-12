# Recursive adversarial review — ROUND 5 of the 5 Layer-2 levers (2026-06-12)

**Reviewer:** R5 subagent (author ≠ reviewer). Prior rounds, all on disjoint lenses:
R1 (`...151829Z.md`, `4cbd9676a`) = static NO-FAKE (all 5 REAL, no HIGH, daemon-safe, 97 tests).
R2 (`...154002Z.md`, `253f8ab9a`) = runtime/resume (fixed Lever-4-EMA-not-persisted MEDIUM, `6e0d8feff`).
Gap-closure (`990fd3de3`) = MED-1 scan-order (Spearman −0.14→0.90) + MED-2 byte-direction + LOW-1 doc.
R3 (`...164500Z.md`, `d5cadcb31`) = gradient-direction (all levers descend their real target or hold
flat-at-optimum; double-counting COHERENT; fixed a LOW compose-timeout-flake marker).
R4 (`...170145Z.md`, `7ccb0fd1d`) = deployed-archive end-to-end (eval==inflate parity) + multi-stage
boundary + fresh-eyes — CLEAN, counter 0/3 → **1/3**.
**R5 has a FIFTH, distinct lens: DETERMINISM / seed-reproducibility (A) + optimizer(Muon)×lever interaction
(B) + long-run numerical stability over 10k+ steps (C) + a holistic "question ALL interpretations" final
pass (D), then re-confirm the established invariants (E).** The bug class R5 hunts: a lever that introduces
NONDETERMINISM (unsorted set/dict iteration, a nondeterministic kernel in the score-aware backward,
`.item()`-driven control flow, or a Muon Newton-Schulz perturbation) so two same-seed runs diverge — or a
quantity that is fine at 80 epochs but drifts/overflows over a multi-thousand-epoch stage. Unreachable by a
static / resume / gradient / deployed-archive audit (each of which ran ONE config or ONE epoch budget).
**Scope:** VERIFY + TEST; new $0 probe + 7 regression tests (no lever-file edit was needed — zero findings).
Did NOT touch `src/tac/substrates/cool_chic/**` (Track B), the basin daemon (pid 33911, confirmed ALIVE
5h52m+, default config `--no-split-by-head --train-device mps`, untouched), or its out-dir.
**Authority:** every in-loop / synthetic number here is `[macOS-CPU advisory]` NON-PROMOTABLE (synthetic
scorer, RESEARCH-ONLY); the levers land MEANS, the exact frontier is UNMOVED (`0.19109982`). Mission
contribution: `frontier_protecting` (a nondeterministic or long-run-unstable lever would corrupt the
multi-day from-scratch descent; R5 proves none does).

## CLEAN-PASS VERDICT: **CLEAN → counter ADVANCES to 2/3.**

R5 found **ZERO findings** (no HIGH, no MEDIUM, no LOW). All five lenses passed. R5 starts from R4's CLEAN
code (`7ccb0fd1d`), exercises the determinism + Muon-interaction + long-run-stability + holistic surfaces
the prior 4 rounds did NOT, and finds nothing to fix. Per the protocol ("3 consecutive clean passes
required"), this is the **second consecutive clean pass** — counter 1/3 → **2/3**. The new probe + 7 tests
are durable regression guards (ruff-clean); no lever source changed, so byte-identity is structurally
untouched. **R6 is the next chance to reach 3/3 (SEAL).**

---

## A. DETERMINISM / SEED-REPRODUCIBILITY (the headline R5 lens) — PASS (no nondeterminism).

The prior rounds proved the ALL-DEFAULT path is reproducible
(`test_all_default_driver_run_is_deterministic_and_byte_identical`, `use_muon=False`) and that resume is
bit-identical. **NEITHER ran a TWO-FRESH-RUNS-FROM-SAME-SEED check with the LEVERS ON.** R5's
`experiments/probe_r5_determinism_and_muon.py` + 2 new driver tests close that exact gap by training the
all-5-levers-ON config TWICE from the same seed and byte-comparing the produced archive:

| # | Determinism property (all-5-levers-ON, same seed) | Result |
|---|----------------------------------------------------|--------|
| A1 | all-5-on **AdamW**, two fresh runs → archive bytes | **BIT-IDENTICAL** (84805 B == 84805 B) |
| A2 | all-5-on **+ MUON**, two fresh runs → archive bytes | **BIT-IDENTICAL** (84790 B == 84790 B) |
| C1 | QAT `_rank_normalize` `argsort` tie-break on TIED sensitivities | **STABLE** (all-tied→uniform-0.5; partial-tie level map identical over 8 calls) |

**The headline verdict: the levers introduce NO nondeterminism — two same-seed all-5-on runs are
bit-identical, including under Muon.** The load-bearing interpretations (each verified, not assumed):

1. **No unsorted set/dict iteration in any lever path.** `_codec_stream_normalized` (Lever 1) iterates
   `decoder.state_dict().items()` (an OrderedDict in registration order) and looks up `dict(named_parameters())`
   by key — both DETERMINISTIC traversals, no `set`-iteration-order dependence. `accumulate_tensor_sensitivity`
   + `apply_score_aware_qat` (Lever 4) iterate `named_modules()` (deterministic depth-first) and key a dict by
   NAME (order-independent). `per_tensor_levels_from_sensitivity` iterates the `tensor_names` LIST (deterministic).
   No `set`-derived ordering reaches a numeric result.
2. **The one `argsort`-on-possible-ties path (Lever 4 `_rank_normalize`) is determinism-safe (C1).** A real
   concern: `torch.argsort` is not stable on exact ties, so two equal-sensitivity tensors could rank in an
   implementation-defined order → a different QAT grid → different archive bytes. Verified it does NOT bite:
   (a) an ALL-tied vector short-circuits to all-0.5 (the uniform fallback) BEFORE `argsort` is even reached
   (`(vmax-vmin).abs() < 1e-30` guard at `score_aware_qat.py:139`); (b) a PARTIALLY-tied sensitivity dict
   produces a per-tensor level MAP that is identical across 8 repeated calls — even where two tensors share a
   value, the resulting level counts are equal-input → equal-output, so the tie-break cannot change the
   deployed grid. The QAT grid (hence the archive) is stable on ties.
3. **The one `.item()`-driven control-flow branch is deterministic.** `_codec_stream_normalized:171`
   (`if ma.item() < cfg.max_abs_floor: continue`) is a DATA-DEPENDENT skip, but it is deterministic given the
   same weights (the same tensor → the same `ma` → the same skip decision). It is not a nondeterminism source;
   it is a stable below-floor skip (the zero-init FiLM fc2 path R1's LOW-2 flagged, closed by R2).
4. **The RNG contract is device-independent and reproducible.** The driver pins `torch.manual_seed(seed)` +
   `np.random.seed(seed)` at run start (`driver.py:1158-1159`), draws `randperm` + latent-init on CPU
   (`driver.py:545,1204-1211` — explicitly to keep the draw device-independent vs MPS's separate RNG stream),
   and captures/restores `torch.get_rng_state()` in the checkpoint (`driver.py:812`). No lever adds an RNG draw
   (the rate surrogate, QAT, seg surrogate, FiLM, margin map are all deterministic functions of weights/grads),
   so the lever-on permutation + init are bit-reproducible — A1/A2 confirm this end-to-end.

## B. OPTIMIZER(MUON)×LEVER INTERACTION — PASS (partition covers all, grads route correctly).

R4 noted "Muon partition 0-dropped" on a no-lever decoder and ran all-5-on+Muon for NO-NaN. R5-B verifies the
stronger property: under the all-5-on BACKWARD (rate term active), does every lever gradient flow through
BOTH the Muon-orthogonalized AND AdamW param groups, with the partition covering every FiLM param?

| # | Muon×lever property (all-5-on + Muon) | Result |
|---|----------------------------------------|--------|
| B1a | partition covers EVERY trainable decoder param | **32/32 covered, 0 overlap** |
| B1b | every FiLM param covered by the partition | **4/4 covered** |
| B1c | every Muon param + every AdamW param carries a gradient post-backward | **13/13 Muon, 19/19 AdamW** |
| B1d | the rate surrogate (Lever 1) reaches a 2D FiLM weight in the **Muon** group | **fc1 Muon-grad = True (nonzero)** |
| A2 | Muon's Newton-Schulz orthogonalization is not perturbed nondeterministically | **two-run BIT-IDENTICAL** |

**The verdict: the levers respect the Muon/AdamW partition — no lever gradient is mis-routed or dropped for
the Muon group, and Muon's orthogonalization is deterministic under the levers.** Interpretations:

1. **The partition is exhaustive AND disjoint under FiLM.** `partition_params_for_muon` routes 2D
   non-stem/non-rgb weights to Muon, biases/1D to AdamW. On the FiLM-wrapped decoder all 32 trainable params
   land in exactly one group (0 uncovered, 0 in both) — the 4 `pose_film.*` params included (fc1/fc2 weights →
   Muon, biases → AdamW). No silent freeze of the FiLM MLP, and no double-stepping.
2. **Lever-1's rate gradient reaches the Muon group (the crux, B1d).** Lever 1 regularizes the FULL
   `state_dict()` including the FiLM weights, so the rate term must produce a NONZERO gradient on the FiLM fc1
   (a 2D weight → Muon group). Verified directly: after the all-5-on backward the fc1 Muon-group weight has a
   nonzero `.grad`. The rate gradient is NOT dropped for the Muon-orthogonalized params — it flows through the
   Newton-Schulz step like any other Muon weight. (fc2 is zero-init so its render contribution is zero at init,
   but the rate term still regularizes fc1, which is what carries the gradient.)
3. **Lever-4's QAT does not break the Muon gradient.** The score-aware QAT applies the STE fake-quant for the
   FORWARD only (`apply_score_aware_qat`) then `restore_score_aware_qat` restores the float weights before the
   gradient lands on `.grad` (the STE backward is identity, so the gradient reaches the float weights). The
   Muon params therefore see the SAME gradient they would without QAT — confirmed by B1c (all Muon params have
   grad after the all-5-on backward that ran through the QAT forward).
4. **Newton-Schulz orthogonalization is deterministic under the levers (A2).** The decisive proof: two fresh
   all-5-on+Muon runs are BIT-IDENTICAL. If the rate gradient (or the QAT-shaped weights) perturbed Muon's
   iterative orthogonalization nondeterministically, the archive would diverge — it does not. The
   rate-term gradient composes with Muon's orthogonalization without a pathological interaction at this
   operating point (and the two-run bit-identity is the strongest possible "no pathology" evidence).

## C. LONG-RUN NUMERICAL STABILITY (beyond R2's 80-epoch NaN check) — PASS (bounded by construction + spot-tested).

R2 ran 80 epochs NaN-free. R5-C reasons about + spot-tests the cumulative behavior over a MULTI-THOUSAND-step
run, with 3 new tests pinning the boundedness properties:

| # | Long-run-stability property | Result |
|---|-----------------------------|--------|
| C2 | QAT sensitivity EMA finite + bounded over **12000** steps (with 1e6 grad spikes every 500) | **finite ∀ step; v ≤ max grad norm** |
| C3 | seg-temperature anneal at T→min over a **9000-epoch** stage | **clamped at floor 0.02; never < floor (no div-by-0); finite OOR epochs clamped** |
| C4 | soft-cosine surrogate at the coldest annealed T | **finite, in [0,1]** |

**The verdict: every long-run quantity is bounded BY CONSTRUCTION and stays finite under stress.**
Per-quantity:

1. **QAT sensitivity EMA — bounded by `max(grad-norm)` (C2).** `s_t = decay·prior + (1−decay)·‖grad‖` is a
   CONVEX COMBINATION of non-negative bounded terms, so over any number of steps `s_t ∈ [0, max ‖grad‖]` — it
   cannot drift up or overflow. Pinned over 12000 updates WITH a 1e6 gradient spike injected every 500 steps:
   the EMA stayed finite at every step and never exceeded the max observed grad norm. (A naive accumulation
   `prior + s` would blow past `max` and FAIL the bound — a genuine guard.)
2. **Seg-temperature anneal — clamped at the floor, T=0 structurally impossible (C3).**
   `seg_temperature_for_epoch` clamps the cosine output to `[seg_temperature_end, seg_temperature]`
   (`curriculum.py:250-251`), so over a 9000-epoch stage the final T is EXACTLY the floor (0.02 in the test)
   and never undershoots toward 0. An out-of-range epoch (a long-run off-by-one) is clamped, not extrapolated.
   And `_validate_kl_temperature` (`core.py:94`) RAISES on `T ≤ 0` — so even a misconfigured `seg_temperature_end`
   ≤ 0 fails closed BEFORE the division, never a silent div-by-zero/inf in `softmax(pred/T)`.
3. **Soft-cosine surrogate at coldest T — finite in [0,1] (C4).** `F.softmax(pred/0.02)` = `softmax(pred·50)`:
   `F.softmax` subtracts the row max before `exp` (internally max-stable), so even a 50× logit scale does NOT
   overflow; the output is a valid simplex and `1 − Σ_c softmax·onehot ∈ [0,1]`. Pinned finite + in-range at the
   coldest annealed T.
4. **FiLM γ is tanh-BOUNDED, β is optimizer-constrained (read, not a new test).** `gamma = 1 + tanh(gamma_pre) ∈
   (0,2)` (`pose_film.py:87`) — `tanh` saturates, so γ cannot blow up however far `gamma_pre` drifts over a
   long run (the multiplier is structurally bounded). `beta` is unbounded in principle but is subject to
   grad-clip + the final `sigmoid` clamp to [0,255]; no unbounded amplification path. The render cannot diverge
   from a FiLM drift.
5. **Margin weight is `exp(−nonneg) ∈ (0,1]` (read).** `margin = (top1−top2).clamp_min(0) ≥ 0`,
   `tau = max(τ, 1e-6) > 0` → exponent `≤ 0` → `exp(...) ∈ (0,1]`. A huge margin underflows to 0, not NaN; the
   `tau` floor prevents div-by-zero. Bounded over any run length.
6. **rate_lambda is a fixed coefficient, not accumulated (read).** `rate_lambda_w/_lat` are constant StageSpec
   scalars multiplying a bounded entropy term (≤ log2(255) bits); they do not accumulate across steps. No
   `rate_lambda` blowup path.

## D. HOLISTIC "QUESTION ALL INTERPRETATIONS" FINAL PASS — PASS (1 un-covered path independently re-confirmed).

A fresh senior-engineer re-read challenging what the prior rounds + this session took for granted:

1. **The basin daemon's OWN launcher sets NO lever field (re-confirmed via a DIFFERENT entry point than R2/R4).**
   R2/R4 proved daemon-safety via `_spec_from_stage_config` defaults + the byte-identity test. R5-D re-confirms
   it from the daemon's ACTUAL entry: `experiments/launch_split_by_head_basin.py` (pid 33911's argv) contains
   ZERO of `seg_surrogate` / `seg_temperature` / `rate_lambda` / `score_aware_qat` / `margin_weight` /
   `pose_film_enabled` / `_resolve_lever_overrides` / `replace(` — a direct grep returns nothing. It builds
   `TorchVehicleConfig(...)` with NO `pose_film_enabled` arg (defaults False) and NO curriculum override (the
   driver uses the default `build_curriculum` → all-default StageSpecs). The daemon is structurally lever-OFF
   from its own launcher, not just from the StageSpec defaults. **The live multi-day descent runs the
   byte-identical pre-lever path.**
2. **The `--levers` semantics: the combined launcher is INTENTIONALLY Lever-2-ON (not a silent partial
   activation).** `launch_l2_combined_attacks.py::_resolve_lever_overrides` ALWAYS sets
   `seg_surrogate=args.seg_surrogate` (default `"soft_cosine"`) and `seg_temperature` regardless of `--levers`
   — because Lever 2 IS "the seg attack, always on in both modes" (the docstring's stated design, line 211).
   This is the dedicated combined-L2-attack arm (a DIFFERENT process from the basin daemon), correctly
   levers-on by construction; `--levers all` additionally turns on Levers 1/4/5 + the anneal endpoint + FiLM.
   No silent partial-activation bug — the default-`seg_pose` mode is a documented Lever-2-on experiment, and it
   never touches the daemon's launcher. (This is the interpretation R2's lens-D did not fully trace; R5-D
   closes it.)
3. **`StageSpec` lever defaults all OFF (`_spec_from_stage_config` sets none) — re-confirmed on HEAD.** The
   vendored curriculum / basin daemon is fully inert; `test_stagespec_all_lever_fields_default_to_off` +
   `test_config_pose_film_defaults_off` hold in the suite.
4. **`codec_scan_order=True` hardwired (the MED-1 deploy-faithful fix is always on).** `driver.py:713`
   `rate_cfg = RateSurrogateConfig(codec_scan_order=True)` — the rate lever always uses the Spearman-0.90
   deploy-faithful scan order, never the −0.14 legacy mode. No path reverts to the slack proxy.

## E. R1/R2/R3/R4 INVARIANTS RE-CONFIRMED ON CURRENT HEAD — HOLD.

Full suite (lens E):
```
.venv/bin/python -m pytest src/tac/torch_vehicle/tests/ src/tac/tests/test_rate_surrogate.py -q --timeout=400
→ 111 passed in 266.77s
```
**0 failures** (104 R4 baseline + 7 new R5 tests). This subsumes: the all-default byte-identity proof, the
FiLM-off byte-identical archive, R2's Lever-4-EMA-resume round-trip, the eval==inflate parity test, the
all-default + the (NEW) all-5-on AdamW + all-5-on+Muon determinism tests, and all pose_film/lever behavior
tests. The 3 touched files are ruff-clean.

## Findings by severity

- **HIGH:** NONE. No nondeterminism (two same-seed all-5-on runs bit-identical, incl. Muon), no Muon
  gradient mis-route/drop, no long-run drift/overflow, no regression.
- **MEDIUM:** NONE.
- **LOW:** NONE.

## The determinism verdict (the R5-A deliverable)

**FULLY DETERMINISTIC.** Two fresh all-5-levers-ON runs at the same seed produce a BIT-IDENTICAL archive,
under BOTH AdamW (84805 B == 84805 B) AND Muon (84790 B == 84790 B). No lever path introduces nondeterminism:
no unsorted set/dict iteration reaches a numeric result; the one `argsort`-on-ties path (QAT `_rank_normalize`)
collapses to uniform on full ties and yields a stable level map on partial ties; the one `.item()`-driven
branch is a deterministic below-floor skip; the RNG contract is device-independent + reproducible. A multi-day
from-scratch run is bit-reproducible.

## The optimizer-interaction verdict (the R5-B deliverable)

**CLEAN — no mis-route, no pathology.** The Muon/AdamW partition covers every trainable param (32/32, 0
overlap) including all 4 FiLM params; after the all-5-on backward every Muon (13/13) and AdamW (19/19) param
carries a gradient; the rate surrogate (Lever 1) reaches a 2D FiLM weight in the Muon group (nonzero grad);
and Muon's Newton-Schulz orthogonalization is deterministic under the levers (the two-run bit-identity is the
proof). No lever gradient is dropped for the Muon group; the rate term does not interact pathologically with
the orthogonalization.

## The long-run-stability verdict (the R5-C deliverable)

**STABLE — bounded by construction.** The QAT sensitivity EMA is a convex combination → bounded by the max
grad norm (pinned finite over 12000 steps with 1e6 spikes); the anneal is clamped at the floor over a
9000-epoch stage (T=0 structurally impossible — `_validate_kl_temperature` fails closed); the soft-cosine
surrogate is finite in [0,1] at the coldest T (`F.softmax` is max-stable); FiLM γ is tanh-bounded; the margin
weight is `exp(−nonneg) ∈ (0,1]`; rate_lambda is a fixed coefficient (not accumulated). Nothing that is fine
at 80 epochs blows up at 10k.

## Test-run count

- Full suite (lens E): **111 passed in 266.77s, 0 failures** (104 R4 baseline + 7 new R5 tests).
- R5 probe (`probe_r5_determinism_and_muon.py`): **4/4 checks PASS** (A1 AdamW-determinism BIT-IDENTICAL, A2
  Muon-determinism BIT-IDENTICAL, B1 partition+routing PASS, C1 tie-break STABLE).
- 7 new regression tests (run in isolation): **7 passed in 123.42s.**

## Tests + probe added this round (durable evidence + regression guards)

Regression tests (lever-on determinism + Muon interaction + long-run stability):
- `test_all_layer2_levers.py::test_all_five_levers_adamw_run_is_deterministic_and_byte_identical` (A1)
- `test_all_layer2_levers.py::test_all_five_levers_muon_run_is_deterministic_and_byte_identical` (A2)
- `test_all_layer2_levers.py::test_all_five_muon_partition_covers_film_and_routes_grads` (B1)
- `test_all_layer2_levers.py::test_lever2_anneal_at_t_min_over_long_stage_is_clamped_and_surrogate_finite` (C3+C4)
- `test_score_aware_qat.py::test_rank_normalize_all_tied_collapses_to_uniform_every_call` (C1a)
- `test_score_aware_qat.py::test_per_tensor_levels_deterministic_under_partial_ties` (C1b)
- `test_score_aware_qat.py::test_sensitivity_ema_bounded_and_finite_over_long_run` (C2)

Probe (durable evidence artifact): `experiments/probe_r5_determinism_and_muon.py` (the 4-check
determinism + Muon-interaction probe). All ruff-clean. No edit to any lever file → byte-identity of the
default/daemon path is structurally untouched (no re-run needed; nothing changed).

## Wire-in / provenance

6-hook (Catalog #125): all N/A — this is a review-round memo + 7 regression tests + one $0 evidence probe
(no new score-claim surface; the levers' own hooks are in the landing memo). Mission contribution:
`frontier_protecting` (verifies no lever introduces nondeterminism / Muon-misroute / long-run instability
that would corrupt the multi-day descent; the END remains a lower exact score, frontier UNMOVED
`0.19109982`). Authority: all numbers `[macOS-CPU advisory]` synthetic-scorer NON-PROMOTABLE. No GPU
launched, no daemon touched (pid 33911 ALIVE 5h52m+ + untouched), no Cool-Chic touched.

**VERDICT: CLEAN (zero findings) → counter ADVANCES 1/3 → 2/3.** R6 is the next chance to reach 3/3 (SEAL).
The levers are fully deterministic (two same-seed all-5-on runs bit-identical, incl. Muon), respect the Muon
partition (no gradient mis-route), and are numerically stable over a multi-thousand-step run (every quantity
bounded by construction). R6 should pick a SIXTH distinct lens (e.g. a real-scorer paired smoke on a tiny
real-video slice to close the synthetic-scorer gap; MPS-vs-CPU numerical-drift of the FiLM render at the
train/authority boundary; or an adversarial re-read of the codec parse-back/grammar under a maximally-coarse
QAT grid) to keep the clean-pass count meaningfully adversarial rather than re-running covered ground.

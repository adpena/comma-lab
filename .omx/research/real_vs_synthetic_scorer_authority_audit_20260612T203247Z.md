# Real-vs-Synthetic Scorer Authority Audit (stack-wide NO-FAKE sweep)

**Stamp:** 2026-06-12T203247Z · **HEAD:** 190e8f56f661b1ec07144fecbf4828eabd195940
**Scope:** every scorer-consuming implementation + this-session measurement, classified
REAL-frozen-contest-scorer / SYNTHETIC-stand-in / PROXY / DISTILLED-surrogate / MLX / MPS,
flagging anywhere a NON-real-scorer verdict is (or could be) treated as AUTHORITY.
**Directive:** operator "need to review all for such" — the NO-FAKE forbidden class 8
("surrogate-optimized-but-not-exact-authority-verified") applied stack-wide.
**Discipline:** READ-ONLY audit. No production edits. Did not touch the live basin daemon
(pid 33911, `--device cpu --train-device mps`, 8h57m elapsed), its out-dir, the
`src/tac/torch_vehicle/**` source, or `src/tac/substrates/cool_chic/**`.

---

## TL;DR (the five deliverables)

1. **Scorer-instantiation inventory:** ONE real-scorer loader family
   (`load_frozen_distortion_net` / `load_differentiable_scorers` / `load_default_scorers`,
   the upstream EfficientNet-B2 SegNet + FastViT PoseNet) feeding ~50 consumer sites; ONE
   synthetic stand-in (`_TinyFrozenScorer` + `SyntheticScorerContext`, `research_only=True`,
   3 sites, all tests/probes); a small distinct set of explicit surrogate/proxy surfaces
   (Hinton-distilled SegNet/PoseNet heads, IB-Lagrangian aux scorer, MLX/MPS research
   signals) each carrying the correct non-promotable evidence grade.
2. **Consumer classification:** every consumer that produces a d_seg/d_pose/score number is
   tag-correct. Real-scorer consumers (basin, R8, witness, HiNeRV, cool_chic probes,
   substrate trainers) emit `[contest-CPU advisory]` / `[macOS-MLX research-signal]` and
   NEVER promote. Synthetic consumers (lever tests R1-R7) test LOGIC, not score numbers.
3. **Basin verdict (definitive):** the bank's **S=0.387 IS a real-scorer number** — real
   frozen DistortionNet, real GT via `yuv420_to_rgb`, CPU AUTHORITY device via
   `RealScorerContext.exact_eval` → vendored `evaluate_decoder` + `compute_score`. It is
   correctly tagged `[macOS-CPU advisory] NON-PROMOTABLE` (a sub-frontier basin that GATES,
   never IS, a paired contest-CPU+CUDA exact eval through `upstream/evaluate.py` on a
   byte-closed archive).
4. **NO-FAKE risk flags:** **ZERO HIGH** (no surrogate/proxy/MLX/MPS number found in a
   promotion / frontier-pointer / KILL path without the advisory tag). The known lever case
   (R1-R7 synthetic → R8 real-closure) is correctly handled. Two LOW/INFORMATIONAL naming
   notes recorded (a "proxy" file that actually runs `upstream/evaluate.py`; the mock-scorer
   escape hatch — structurally fail-closed).
5. **Re-validation:** nothing beyond the already-in-flight basin/levers needs a real-scorer
   re-measurement to be trusted. The lever R1-R7 synthetic verdicts are the only "verified on
   synthetic" claims, and R8 is precisely their real-scorer closure. Ranked recommendations
   in §E.

---

## A. Scorer-instantiation inventory

### A.1 REAL frozen contest scorer (the AUTHORITY)

| Loader | File | What it loads |
|---|---|---|
| `load_frozen_distortion_net` | `src/tac/score_aware_loop/targets.py:30` | upstream `DistortionNet` (`modules.py`), loads `posenet_sd_path` + `segnet_sd_path`, freezes all params, GT decode via `frame_utils.yuv420_to_rgb` (canonical; PyAV rgb24 FORBIDDEN). **REAL.** |
| `load_differentiable_scorers` → `load_default_scorers` | `src/tac/scorer.py:192` / `:213` | same upstream scorers + the differentiable-yuv6 patch (only fixes the `@torch.no_grad` gradient sever; weights unchanged). **REAL.** |
| shell-out to `upstream/evaluate.py` | `src/tac/proxy_eval.py:228` | the actual contest evaluator on inflated frames. **REAL** (see LOW-1 naming note). |

`load_frozen_distortion_net` consumers (~37 files): `src/tac/torch_vehicle/scorer_context.py` (the basin),
`src/tac/mlx_pr95_port/score_bridge.py`, `src/tac/score_aware_loop/surrogate_correlation.py`,
and ~34 `experiments/*` probes (witness, HiNeRV, cool_chic, capstone, descent-equivalence,
mps-drift diagnostics, etc.).
`load_differentiable_scorers` consumers: every `experiments/train_substrate_*.py` (hi_nerv,
balle, siren, vq_vae, cool_chic, wavelet, ds_nerv, tc_nerv, ff_nerv, … ~22 trainers) — these
pass the REAL scorer into `build_gt_scorer_cache` (see A.4).

**Real-scorer instantiation sites: ~50+ (one loader family).**

### A.2 SYNTHETIC stand-in (RESEARCH-ONLY)

| Class | File | Tag |
|---|---|---|
| `_TinyFrozenScorer` | `src/tac/torch_vehicle/scorer_context.py:213` | docstring: "NOT the contest scorer". A fixed-weight 3→5 conv + 3→6 linear, frozen. |
| `SyntheticScorerContext` | `src/tac/torch_vehicle/scorer_context.py:234` | `research_only = True`; docstring "RESEARCH-ONLY (no score claim) … exercises STATE serialization without the real EfficientNet scorer". |

Consumers (3 src + ~8 experiments): the torch_vehicle lever/driver tests
(`test_all_layer2_levers.py`, `test_driver_resume.py`, `test_split_by_head_grad.py`,
`test_async_eval.py`, `test_export_and_faithful.py`, `test_pose_film_wire_in.py`,
`test_split_device_mps.py`) + the R4-R7 lever probes (`probe_r4_*`, `probe_r5_*`,
`probe_r6_*`, `probe_r7_*`, `launch_l2_combined_attacks.py`, `launch_pose_film_basin.py`).
**Synthetic instantiation sites: 3 definitions; ~11 consumers — all tests/lever-logic probes.**

### A.3 DISTILLED-surrogate / PROXY / MLX / MPS surfaces (explicit, each non-promotable)

| Surface | File | Family | Evidence grade / tag |
|---|---|---|---|
| Hinton-distilled SegNet/PoseNet student heads | `src/tac/archive/scorer_distill.py`; `src/tac/residual_basis/…`; `src/tac/scorer_surrogate/posenet_mae_v/` | DISTILLED | `[prediction]` / `axis_tag="[predicted]"`, `score_claim=False`, `promotion_eligible=False`. Numpy-portable student; trained AGAINST the real teacher cache. |
| IB-Lagrangian co-trained aux scorer | `src/tac/ib_lagrangian_aux_scorer.py` | DISTILLED (T=2.0) | training-only auxiliary; docstring explicit that the CONTEST score uses the FROZEN scorer (aux is a gradient surrogate, never the eval). |
| surrogate↔exact correlation MEASURER | `src/tac/score_aware_loop/surrogate_correlation.py` | REAL (validator) | computes the EXACT argmax-flip d_seg via `load_frozen_distortion_net`; `[macOS-CPU advisory]`. Its JOB is to validate surrogates against authority — "not a proxy of a proxy". |
| MLX↔torch score bridge | `src/tac/mlx_pr95_port/score_bridge.py` | REAL gradient | the SCORER half is the exact FROZEN torch path (`load_frozen_distortion_net`); only the decoder half is MLX. "no surrogate". |
| MLX research signal | `src/tac/optimization/mlx_research_signal.py` | MLX | `evidence_grade="macOS-MLX-research-signal"`; raises if `score_claim`/`promotion_eligible`/`ready_for_exact_eval_dispatch` set True. |
| MPS research signal | `src/tac/optimization/mps_research_signal.py` | MPS | `evidence_grade="MPS-research-signal"`; `score_claim=False`, `rank_or_kill_eligible=False`; forbidden uses enumerated (auth_eval, promotion, falsification, retirement). |
| macOS-CPU advisory signal | `src/tac/optimization/macos_cpu_advisory_signal.py` | CPU advisory | `[macOS-CPU advisory only]`; `ranking_only=True`; raises on authority flags. |
| GT-scorer cache (optimization) | `src/tac/training_optimization/scorer_cache.py` | inherits the scorer it's GIVEN | caches `posenet`/`segnet` GT forward; real when fed the real scorer (A.4). Its own micro-benchmark explicitly tagged `[empirical, synthetic micro-benchmark … tiny fake scorers]`. |
| mock scorer teacher (MLX) | `src/tac/substrates/_shared/mlx_score_aware/bundle.py:522,854` | MOCK | deterministic-cosine fallback; FAIL-CLOSED — raises unless `allow_mock_scorer_teacher=True` is EXPLICITLY set; **no pose mock exists** (pose path raises). |

### A.4 GTScorerCache real-vs-synthetic resolution

`GTScorerCache` is an OPTIMIZATION primitive (caches the invariant GT scorer forward to halve
per-step scorer compute). Its authority = whatever scorer the trainer passes. Verified
`experiments/train_substrate_hi_nerv.py:717` calls `load_differentiable_scorers` (REAL) before
`build_gt_scorer_cache` — the pattern across substrate trainers. The cache itself is
mathematically bit-identical to recomputing (frozen weights, frozen video), so it introduces
NO surrogate gap when fed the real scorer. Its docstring's standalone speedup number is the
one place a synthetic scorer appears, and it is correctly tagged `[empirical, synthetic
micro-benchmark, tiny fake scorers]`.

---

## B. Consumer classification table (consumer → scorer → tag-correct?)

| Consumer | Scorer family | Number(s) produced | Tagged correctly? |
|---|---|---|---|
| **BASIN** `launch_split_by_head_basin.py` + `driver.py` + `RealScorerContext` | REAL (CPU authority eval; MPS only the per-step gradient) | exact d_seg/d_pose/rate/**S=0.387** | **Y** — `[macOS-CPU advisory] NON-PROMOTABLE`, "GATES, never IS, a paired contest eval" |
| **R8** `probe_r8_real_scorer_paired_smoke.py` | REAL (`RealScorerContext`) | real-scorer lever firing, real d_seg | **Y** — `[contest-CPU advisory] NON-PROMOTABLE`, explicitly closes the synthetic gap |
| **R1-R7 lever tests/probes** `test_all_layer2_levers.py`, `probe_r4..r7_*` | SYNTHETIC (`SyntheticScorerContext`) | lever LOGIC (byte-identity, grad direction, compose-5) — NOT score numbers | **Y** — `research_only=True`; tests isolate logic; APPROPRIATE synthetic use |
| **witness** `witness_seg_boundary_decisive_probe.py` | REAL (`load_frozen_distortion_net`) | boundary margins, flips, bytes/flip, round-trip survival | **Y** — `[contest-CPU advisory] NON-PROMOTABLE`, "no score is claimed; frontier UNMOVED" |
| **HiNeRV grid probe** `probe_hinerv_grid_vs_lever_dseg.py` | REAL (`load_frozen_distortion_net(cpu)`) | matched-byte exact d_seg (`compute_distortion`) | **Y** — `[contest-CPU advisory]`, GO/NO-GO threshold is the deliverable; `score_claim=false, promotable=false` |
| **cool_chic step-3** `measure_cool_chic_joint_latent_compressibility.py` | REAL (MPS gradient + CPU-authority advisory copy) | coded latent bytes (lossless), advisory d_seg/d_pose | **Y** — MPS = train-gradient only; advisory d_seg/d_pose on CPU authority; `[contest-CPU advisory] NON-PROMOTABLE` |
| **cool_chic byte-closed/posefold/sweep** `run_cool_chic_*` | REAL (`load_frozen_distortion_net`) | advisory S, byte-closed rate | **Y** — heavily advisory-tagged |
| **cool_chic AR/entropy smokes** `measure_cool_chic_ar_entropy_coder.py`, `cool_chic_ar_prior_*` | NONE (rate-only) | bytes/bits/rate — ZERO d_seg/d_pose/score | **Y** — no scorer needed, no score claimed |
| **substrate trainers** `train_substrate_*.py` | REAL (`load_differentiable_scorers` → GTScorerCache) | training loss; auth eval gated separately | **Y** — real scorer feeds cache; promotion gated by separate paired-eval contract |
| **MLX score-aware substrates** `mlx_score_aware/{bundle,loss}.py` | REAL teacher (or MOCK only with explicit opt-in) | distillation loss; pose-axis | **Y** — fail-closed: raises if a distill term has no real teacher and `allow_mock_scorer_teacher=False`; long-run launch gate (`nerv_long_run_launch_gate.py:1592`) does NOT count mock rows as authority |
| **surrogate_correlation** `surrogate_correlation.py` | REAL (validates surrogates) | Spearman/OLS of surrogate-vs-EXACT d_seg | **Y** — `[macOS-CPU advisory]`; the EXACT d_seg IS the evaluator's quantity |
| **proxy_eval** `proxy_eval.py` | REAL (shells `upstream/evaluate.py`) | PoseNet/SegNet distortion, score | **Y on authority** — number is from the real evaluator (naming note LOW-1) |

**Tag-correctness: 12/12 consumer classes correct.**

---

## C. Basin real-scorer confirmation (DEFINITIVE)

Traced `RealScorerContext.__init__` (`scorer_context.py:53-127`):
- `self.distortion_net = load_frozen_distortion_net(device=str(self.device))` →
  upstream `DistortionNet().eval()`, `net.load_state_dicts(posenet_sd_path, segnet_sd_path)`,
  all params frozen. The **real EfficientNet-B2 SegNet + FastViT PoseNet**, NOT a fallback to
  synthetic/random (the synthetic path is a SEPARATE class, `SyntheticScorerContext`).
- `seg_targets_hard` / `pose_targets` come from `self._data.precompute_targets(...)` (uncapped)
  or `build_gt_targets(...)` (capped) — both run the REAL frozen scorer on REAL 0.mkv frames
  decoded ONLY via `frame_utils.yuv420_to_rgb`.
- `RealScorerContext.exact_eval` (`:184-210`) ALWAYS runs on `self.device` (CPU authority, NEVER
  the MPS train device — the constructor `raise`s if `device` starts with "mps"), routing through
  vendored `score.evaluate_decoder` (streams GT via `yuv420_to_rgb`) + `score.compute_score` (the
  official `100·d_seg + sqrt(10·d_pose) + 25·rate` metric).
- The live basin daemon command confirms `--device cpu --train-device mps` (MPS = per-step
  gradient backend only; the BEST-picking exact eval is CPU authority).

**VERDICT: the bank's S=0.387 IS a real-scorer number** (real frozen SegNet+PoseNet, real GT,
CPU authority). It is **correctly NON-AUTHORITY for promotion** — it is `[macOS-CPU advisory]`,
NOT a `contest-CPU`/`contest-CUDA` exact eval through `upstream/evaluate.py` on the byte-closed
archive that would actually ship. The advisory tag is exactly right: the NUMBER is real, the
AXIS (in-loop CPU advisory, not the byte-closed evaluator on contest hardware) is sub-authority.

---

## D. NO-FAKE risk flags (by severity)

### HIGH (surrogate/proxy/MLX/MPS verdict treated as authority): **NONE FOUND**

No d_seg/d_pose/score from a synthetic/proxy/distilled/MLX/MPS scorer was found feeding a
promotion, a frontier-pointer update (`canonical_frontier_pointer.json`), a lane-maturity
`mark`, or a KILL/FALSIFIED verdict without the advisory/research-signal tag. The structural
guards (Catalog #192 macOS-CPU advisory gate; MLX/MPS research-signal fail-closed appenders;
the `nerv_long_run_launch_gate` mock-row exclusion; `RealScorerContext` MPS-as-authority
`raise`) hold across the stack.

### MEDIUM: **NONE**

### LOW / INFORMATIONAL (correctly handled; recorded for completeness)

- **LOW-1 (naming, not authority):** `src/tac/proxy_eval.py` is named "proxy" but
  `run_faithful_proxy` actually shells out to `upstream/evaluate.py` (`:228`) — its number is
  REAL authority, not a proxy. The risk direction is benign (a real number under a "proxy"
  name, not a fake number under an authority name). No fix needed; note only so a future reader
  does not mistakenly DOWNGRADE a real `proxy_eval` number. (Counterpart: `evaluate.py`-derived
  rows ARE authority.)
- **LOW-2 (escape hatch, fail-closed):** the MLX `allow_mock_scorer_teacher` mock path is the
  one place a scorer-blind number can be produced. It is structurally fail-closed (raises unless
  EXPLICITLY opted in; no pose mock exists), and the long-run launch gate refuses to treat mock
  rows as authority. This is the exact phantom-provenance class flagged in MEMORY (the
  "Wave N+11 re-fire used `--allow-mock-scorer-teacher` so pose-axis=0 phantom" incident) — and
  it is correctly guarded. Recommend keeping the gate strict; no new finding.
- **LOW-3 (appropriate synthetic):** the R1-R7 lever tests/probes use `SyntheticScorerContext`.
  This is APPROPRIATE (unit-test isolation of lever LOGIC — byte-identity, gradient direction,
  compose-all-five — which is scorer-architecture-agnostic). The tests never quote a synthetic
  d_seg/d_pose as a score; R8 is the real-scorer closure for the score-geometry-dependent
  levers (4 = ‖∂S/∂w‖ sensitivity, 5 = real EfficientNet margin map).

---

## E. Re-validation recommendations (ranked by load-bearing)

The audit found NO synthetic/proxy verdict currently masquerading as authority, so there is no
mandatory re-measurement to un-fake an existing claim. The ranked recommendations are about
keeping the (already-correct) advisory→authority promotion path honest:

1. **(Highest, already in flight) — Lever score-geometry on the real scorer:** the R1-R7
   verdicts were earned on the synthetic stand-in; Levers 4 (sensitivity ‖∂S/∂w‖) and 5 (margin
   map) depend on the REAL scorer's gradient geometry. **R8 is precisely this closure** and is
   the correct, sufficient re-validation. No additional action — just ensure R8's real-scorer
   run lands before any lever is cited as more than a MEANS.
2. **(Med) — Basin S=0.387 → exact authority:** the bank number is a real-scorer ADVISORY. To
   become a frontier claim it MUST be re-measured as a byte-closed `archive.zip` through
   `upstream/evaluate.py` on contest-CPU AND/OR contest-CUDA (per the GOAL/authority ladder).
   This is understood (the launcher says so); the recommendation is to NOT let the advisory
   0.387 leak into any frontier-pointer or "we hit X" statement until that paired exact eval
   exists. (No leak found today.)
3. **(Low) — Distilled surrogate consumers:** if any future trainer uses `scorer_distill.py` /
   `ib_lagrangian_aux_scorer.py` / `posenet_mae_v` heads to make a GO/KILL decision, that
   decision must be re-confirmed on the real scorer before promotion. Today these are
   training-gradient/research surfaces only (`[predicted]`), so no current re-validation owed.
4. **(Low) — proxy_eval naming:** optionally rename `run_faithful_proxy` documentation to make
   explicit it returns an `upstream/evaluate.py` authority number (prevents a future
   down-grade). Cosmetic; not a NO-FAKE risk.

---

## Wire-in hooks (per Catalog #125; this is an audit memo, research_only)

- #1 sensitivity-map: N/A (read-only audit, no new score-axis contribution).
- #2 Pareto: N/A.
- #3 bit-allocator: N/A.
- #4 cathedral autopilot dispatch: N/A (no archive-deployable artifact).
- #5 continual-learning posterior: N/A (no empirical score anchor; advisory verdict only).
- #6 probe-disambiguator: the audit IS the disambiguator between "real-scorer advisory" and
  "synthetic/proxy" across every consumer; the table in §B is the reusable surface.

`research_only=true`. Authority tier: this memo makes NO score claim; it CLASSIFIES existing
claims. Every number cited (S=0.387) is reproduced from source with its existing advisory tag
intact; no promotion, no frontier movement.

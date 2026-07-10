# #396 — Exact-metric Monte-Carlo FINISHER (design)

Date: 2026-07-10 · operator GO ("That sounds great, start on it") · subagent `mcfinisher-396`.
Charter: `.omx/research/papers_checked_mc_gradient_free_2607.08406_20260710.md`
(arXiv 2607.08406 — gradient-free (1+1)-ES local search; the ONE novel lever = an
accept/reject loop that optimizes the EXACT discrete argmax d_seg, where our gradient
is weakest — the terminal few-thousand residual flips).

STORES CONSULTED: #391 flip solver (`src/tac/through_r/flip_inverse.py` — kernel/adjoint/
ledger/waterfill; corrected step law `d*=Dᵀ∇m`, Fable §B `flip_margin_step_law_v1`) ·
`src/tac/through_r/harness.py` (`measure_through_r`, the n600 through-R CPU-SegNet
authority) · #157 per-tensor sensitivity (`src/tac/sensitivity_map/`) · #350 exact-
attribution / content-addressed payload search · Unit B (int8 past the RD knee) · #310
unswept gauss/step arm (gradient-free un-blocks it) · NO-FAKE #8 (surrogate ≠ authority) ·
the n600-verdict-OOM law (chunk the SegNet forward).

## What this is (and is not)
A TERMINAL, gradient-free micro-finisher over a LOADED witness checkpoint's small
head/palette tensors (`out_sdf.{weight,bias}` (5,96)/(5,), `palette` (5,3),
`out_tex.{weight,bias}` (3,96)/(3,) — thousands of params). It optimizes the EXACT
through-R d_seg via accept/reject — NO gradient, NO surrogate gap at the finish. It is
NOT a training paradigm swap and NOT a pointer-mover by itself: the pointer (0.19110)
moves ONLY through a byte-closed `upstream/evaluate.py` exact row.

## The four pieces (all built in `src/tac/through_r/mc_finisher.py`)
1. **Guided proposal engine** (`ProposalEngine`) — gradient-free, so guidance is a PRIOR
   on the proposal distribution, refined ONLINE by acceptance statistics (no param-space
   gradient is claimed). Three guidance sources, all optional and composable:
   * per-tensor mass `tensor_weights` ∝ #157 sensitivity (default uniform over targets);
   * per-tensor adaptive step scale (1/5-success-rule (1+1)-ES self-adaptation);
   * an optional per-element `saliency` vector (flip-adjacency) the caller derives from the
     #391 flip ledger + step-law direction `d*=Dᵀ∇m` (element selection ∝ saliency).
   Online per-tensor accept-rate reweighting is THE "guided not blind" mechanism that needs
   no gradient. A proposal = `(tensor, flat_indices, deltas)` over a small element batch B.
   Modes: `fp32` (continuous Gaussian micro-step) and `int8` (discrete ±k on the code,
   clamped [-128,127] — the paper's discrete support; output born byte-closed).
2. **Accept-test ladder** (`accept_batch`) — the P9-honest oracle: (a) SCREEN on a
   subsampled pair set through the real R + frozen CPU-torch SegNet (cheap, labelled
   screen-only, NEVER a verdict); (b) CONFIRM on n600 through-R (the ONLY authority).
   The confirm is asserted as the authority in code and in tests. Batch bisect: if a batch
   confirms net-positive ΔS (a regression), recursively bisect to salvage the net-negative
   half (accept the batch's net-negative composition, else drop).
3. **Ratchet + safety** (`MCFinisher.run`) — monotone ratchet on the CONFIRMED S-component
   (never accept a confirmed regression); ΔS accounting per accepted batch
   `100·Δd_seg + 25·Δbytes/37_545_489` (bytes CAN change in int8 mode via an injected
   `byte_cost_fn`; fp32 default = constant, Δbytes=0). Stop rules: K consecutive dry
   confirm batches, wall-clock budget, max proposals. Resumable: atomic npz param snapshot
   + JSONL mutation log; `resume_from=` restores current params + RNG-advance count.
   Deterministic: single seeded `np.random.default_rng`; full provenance per accepted batch.
4. **Targets** — `param_targets` is a caller-supplied tensor-name list; default =
   `["out_sdf.weight","out_sdf.bias","palette","out_tex.weight","out_tex.bias"]` intersected
   with the tensors present in the checkpoint. Works on any small tensor subset.

## Decoupling (why it is $0-testable and honest)
The core is generic over the actuation/measurement space via THREE injected callables held
in `FinisherProblem`:
* `render_fn(params) -> list[frames]` — witness forward (real run wraps the MLX witness
  render; TESTS use a tiny synthetic mock — a linear map from params to a label field).
* `measure_fn(frames, *, confirm) -> MeasuredObjective` — `confirm=False` SCREEN (subset),
  `confirm=True` n600 through-R. Real run wraps `tac.through_r.harness.measure_through_r`;
  TESTS use a mock scorer with a known flip-count minimum so ratchet/ladder are checkable.
* `byte_cost_fn(params) -> int` — archive bytes for the current param state (int8 mode);
  default constant.
This is the same decoupling `flip_inverse` uses (the R operator + SegNet are injected /
loaded, never re-implemented). The finisher NEVER re-implements R or the SegNet — real runs
pass the canonical `measure_through_r` as `confirm`.

## Canonical-vs-unique decision per layer
| Layer | Decision | Rationale |
|---|---|---|
| Through-R measurement | ADOPT_CANONICAL `tac.through_r.harness.measure_through_r` | P9 authority; never a parallel R/SegNet re-impl. |
| Flip localisation (WHERE) | ADOPT_CANONICAL `flip_inverse` ledger + step law | #391 is SEALED; consume for the saliency prior. |
| Per-tensor sensitivity (WHICH) | ADOPT_CANONICAL #157 (caller supplies `tensor_weights`) | one canonical sensitivity home. |
| ΔS accounting | ADOPT_CANONICAL `tac.contest_score` constants | P1 one-fact-one-store (SEG/RATE weight, 37_545_489). |
| Search loop itself | FORK_PRINCIPLED (new (1+1)-ES local search) | no existing gradient-free accept/reject loop; the paper's lever is genuinely new to us. |
| Resumability / provenance | ADOPT pattern from #350 harness (atomic write + JSONL log) | resumability P0; mirror the content-addressed pattern. |

## Observability surface
* inspectable per layer: every proposal + its screen/confirm verdict is a `BatchOutcome`
  row (tensor, indices, deltas, screen_dseg, confirm_dseg, delta_s, accepted, bisect_depth).
* decomposable per signal: ΔS split into Δd_seg·100 and Δbytes·25/37_545_489.
* diff-able across runs: seeded RNG + JSONL mutation log → identical run re-derives.
* queryable post-hoc: `MCFinisherResult.to_rows()` + the JSONL log.
* cite-able:每 accepted mutation carries (checkpoint sha, seed, proposal index, git sha).
* counterfactual-able: the mutation log replays any prefix (resume_from = a prefix cut).

## Cargo-cult audit per assumption
* "guided proposal beats blind" — HARD-EARNED premise, but the −48% basis figure is a PROXY
  (MEMORY L25); the guidance here is a PRIOR refined by MEASURED acceptance, not asserted.
  **P7 falsifier (pre-registered, below).**
* "int8 mutation is byte-neutral" — CARGO-CULTED if assumed; FALSE for entropy-coded
  payloads. Handled: `byte_cost_fn` injected; fp32 default Δbytes=0 is honest (fixed width).
* "screen predicts confirm" — CARGO-CULTED if trusted as a verdict. Handled: screen is
  labelled non-authority; confirm (n600) is the ONLY accept authority (P9); a screen-passed
  batch that confirms net-positive is REJECTED (or bisected), never silently accepted.
* "the finisher moves the pointer" — FALSE. It produces a byte-closed candidate; the pointer
  moves only through `upstream/evaluate.py`. Stated in the module label + this memo.

## P2 noise floor
The CPU argmax-tie nondeterminism (Unit C: ~1 px / 117.96 M ≈ 8.5e-9 d_seg) is the floor
for a single confirmed flip's Δd_seg (1/(600·384·512) ≈ 8.5e-9 — one flip IS one floor
unit). ΔS per single flip = 100·8.5e-9 ≈ 8.5e-7. A confirmed batch must clear a MULTI-flip
net improvement to be distinguishable; single-flip accepts sit AT the floor and are
labelled instance-level. Recorded on the first empirical anchor's `noise_floor`.

## P7 falsifier (PRE-REGISTERED for the first real run, behind owed-16/#385 — DO NOT RUN)
On the mod32cap ep650 fixture, fp32 mode, targets = out_sdf + palette, B=32 element batches,
budget = 200 confirmed batches: **the guided-proposal premise is REFUTED at FORMULATION
level if < X confirmed net-flip reductions per 100 proposals**, with X = 3× the blind-random
control's rate measured in the same run (a paired ON/OFF of the saliency prior). If guided
≈ blind, the saliency prior adds nothing and the finisher degrades to (1+1)-ES; that is an
IMPLEMENTATION-level negative on the GUIDANCE, not on the MC-finisher paradigm (the accept
loop still optimizes the exact metric). Verdict scope: FORMULATION, not FAMILY.

## Pinned first-run commands (QUEUED behind owed-16 + #385 — machine is READ-ONLY now)
```
# fp32 finish on mod32cap ep650 BEST (confirm ladder, guided ON):
.venv/bin/python -m tac.through_r.mc_finisher \
  --checkpoint experiments/results/perclass_bitalloc_witness_20260710/mod32cap_ep650_BEST.npz \
  --targets out_sdf.weight,out_sdf.bias,palette --mode fp32 --batch-elems 32 \
  --max-confirmed-batches 200 --screen-pairs 48 --seed 0 \
  --out experiments/results/mc_finisher_396_fp32_<UTC>/  # n600 confirm = authority
# int8 mode (born byte-closed, byte_cost_fn from the codec):
  ... --mode int8 --int8-scale-from-ckpt   # claws back quantized d_seg at fixed bits (Unit B)
# paired guided-OFF control for the P7 falsifier:
  ... --no-saliency-prior --seed 0
```

## DSL routing — tool, NOT a Lever (justified against the config-orphan rule)
This is a POST-TRAINING finisher over a FROZEN checkpoint — it is not a training-time knob
the trainer CLI compiles. A `Lever` factory holds *swept trainer flags argparse cannot
supply* (the config-orphan confound). The MC finisher supplies NO trainer flag; it is a
standalone actuator (like the #391 flip solver, `tools/witness_exact_ab.py`, #350 harness)
→ correctly a TOOL, not a DSL Lever. To honour "off is a tracked queue, never a forgotten
default", it is registered in the duty-to-measure activation ledger as a never-fired
high-value lever CONCEPT so the costate controller surfaces it — the ledger row is the
anti-orphan home, the DSL is not.

## Triality legs
* DAG: FEED-mcfinisher (appended to `sub015_DAG_*`).
* DSL: N/A-with-rationale (tool, not Lever — see above); duty-to-measure ledger row instead.
* equations: `exact_metric_mc_finisher_v1` registered design-only (crisp pre-measurement
  invariant: the confirmed monotone ratchet ⇒ S_confirmed non-increasing, each accepted
  batch ΔS_confirmed < 0 BY CONSTRUCTION, and the accept test IS the authority functional so
  there is NO surrogate↔exact gap at the finish). Empirical anchor on the first measured row.
```
```

## Canonical equations (Catalog #344)
Registered: `exact_metric_mc_finisher_v1` in `tac.canonical_equations` (design-only invariant; EmpiricalAnchor lands with the first measured confirm-ladder row).

# Yousfi-Fridrich Field Equations For Contest-Faithful Compression

Date: 2026-05-02

This note formalizes two versions of the same optimization target. The first is
the practical contest-compliant system we can run now. The second is the
mathematically complete infinite-compute system that the practical system
approximates.

## Version 0: Contest-Compliant Practical System

Let each candidate correction/representation choice be an atom `a`. An atom may
be a row run, pair, frame, class, connected component, boundary band, pose
channel, renderer tensor block, decoder latent, entropy-code choice, or packer
choice. The atom has:

- `x_a in {0,1}`: selected or not selected.
- `c_a`: charged archive bytes.
- `b_a`: estimated marginal scorer benefit from exact residual/component
  evidence.
- `H_ab`: sparse interaction/curvature term between atoms.
- `lambda_rate = 25 / 37545489`.

The contest-practical field objective is:

```text
maximize F(x)
  = sum_a b_a x_a
  - lambda_rate * sum_a c_a x_a
  - 0.5 * sum_{a,b} H_ab x_a x_b
  + sum_g synergy_g(x)
```

subject to:

```text
archive bits are charged inside archive.zip
inflate is deterministic and uses fixed contest runtime
no sidecars, scorer patches, network fetches, or host-local score bits
candidate archives pass exact CUDA auth eval before any score claim
promotion requires identical archive bytes on T4/equivalent
```

The current implemented projection is a sparse greedy allocator over CMG3
row-run atoms:

```text
DeltaF(a | S)
  = b_a
  - lambda_rate * c_a
  - curvature(a)
  - pair_frame_antagonism(a, S)
  + class_synergy(a, S)
```

It is intentionally a planning-only equation. It creates deterministic policy
JSON, then the archive builder consumes that policy, then exact CUDA decides.

Implementation:

```bash
.venv/bin/python experiments/plan_yousfi_fridrich_field_equations.py \
  --ledger-json experiments/results/c067_cmg3_pixel_lagrangian_atoms_after_t4/top2_ledger.json \
  --output-json experiments/results/c067_yf_field_equations_20260502/plan.json \
  --mode both

.venv/bin/python experiments/build_cmg3_adaptive_runs_candidate.py \
  --frontier-archive experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip \
  --decoded-mask-array experiments/results/c063_trace_weighted_mask_grammar_plan_20260502_codex/decoded_mask_array.npy \
  --field-policy-json experiments/results/c067_yf_field_equations_20260502/plan.json \
  --field-policy-id yf_field_sparse_pair_frame_class_top0032 \
  --output-dir experiments/results/c067_yf_field_top0032_20260502
```

Operator ergonomics:

- CLI flags are authoritative for reproducibility.
- Environment defaults are supported for fast iteration:
  `PACT_FIELD_EQUATION_MODE`, `PACT_FIELD_CANDIDATE_SIZES`,
  `PACT_FIELD_MAX_SOURCE_ATOMS`, `PACT_FIELD_INTERACTION_MODEL`,
  `PACT_FIELD_CURVATURE_STRENGTH`, `PACT_FIELD_PAIR_ANTAGONISM`,
  `PACT_FIELD_FRAME_ANTAGONISM`, `PACT_FIELD_CLASS_SYNERGY`,
  `PACT_FIELD_LOW_RANK_MODES`, `PACT_FIELD_POSITIVE_PROXY_ONLY`,
  `PACT_FIELD_POLICY_PREFIX`.
- Python callers can use kwargs via `build_plan(...)` and
  `encode_adaptive_run_stream(..., field_atom_policy=...)`.

Evidence boundary: this planner is `planning_only`. It cannot promote, rank,
retire, or kill a method without a concrete archive and exact CUDA evidence.

## Version 1: Complete Infinite-Compute System

Let `A` be a complete archive byte string. It induces a deterministic inflate
map `D_A`, decoded masks/frames/poses `(M_A, P_A)`, decoder parameters `theta`,
and charged side-information bits `z`. The true optimization is:

```text
minimize over compliant archives A:

S(A) = 100 * seg_dist(M_A)
     + sqrt(10 * pose_dist(P_A))
     + 25 * |A| / 37545489
```

with constraints:

```text
D_A is deterministic
all theta and z are charged in A
runtime <= contest budget on T4/equivalent
A contains no sidecar dependency
archive construction is reproducible byte-for-byte
```

The local all-order field expansion around an archive state is:

```text
delta S
  = integral g_i dphi_i
  + 1/2 integral integral H_ij dphi_i dphi_j
  + 1/6 integral integral integral T_ijk dphi_i dphi_j dphi_k
  + ...
```

where `phi` spans pixel fields, row runs, connected components, pose manifold
coordinates, ego-motion/camera geometry, renderer weights, learned latents,
quantizers, entropy-code symbols, and packer layout bits.

The complete infinite-compute algorithm is:

```text
1. Define archive grammar families G.
2. For every grammar, define atom coordinates phi and charged bytes c(phi).
3. Sample or enumerate correction paths in phi-space.
4. Build exact archive bytes A(phi).
5. Run exact CUDA auth eval as oracle S(A(phi)).
6. Fit first, second, and selected higher-order interaction tensors.
7. Solve the constrained archive minimum under runtime/compliance constraints.
8. Promote only identical bytes on T4/equivalent.
```

Useful mathematical bases:

- Taylor/Frechet expansion for local score response.
- Fourier/Walsh-Hadamard probes for sparse high-dimensional interactions.
- Riemannian charts for pose, camera, ego-motion, and foveated geometry.
- DCT/wavelet bases for mask boundary and temporal fields.
- Feynman/CEM-style path sampling for nonconvex correction trajectories.
- Dykstra/ADMM projection for the intersection of rate, distortion, runtime,
  compliance, and reproducibility constraints.

Practical consequence: exhaustive per-pixel permutation is not the next move.
The next move is a low-dimensional, charged atom basis with sampled interactions
and exact CUDA checkpoints. That is the computable projection of the complete
field equation under the wall-clock deadline.

## Implementation Landed

Files:

- `experiments/plan_yousfi_fridrich_field_equations.py`
- `experiments/build_cmg3_adaptive_runs_candidate.py`
- `src/tac/tests/test_plan_yousfi_fridrich_field_equations.py`
- `src/tac/tests/test_build_cmg3_adaptive_runs_candidate.py`
- `AGENTS.md`

The planner emits both the practical contest system and the ideal
infinite-compute system in one deterministic JSON. The builder now accepts:

```text
--field-policy-json <path>
--field-policy-id <id>
--base-runs-per-row <int>
```

The `--base-runs-per-row` flag is a correctness guard: a residual ledger
computed against top2 must be consumed on a top2 base, not silently applied to
a top1 base.

After adversarial review, the implementation now also fails closed on:

- negative field-energy policies unless `--allow-negative-field-energy` is
  explicitly passed for cliff mapping;
- field-policy/base-run mismatch;
- unmatched field-policy row-run atoms;
- duplicate selected row-run atoms;
- archive-internal absolute source-policy paths.

Verification:

```text
py_compile: passed
pytest field planner + CMG3A builder + CMG3 atom planner: 11 passed
git diff --check on touched files: passed
```

## C-067 CMG3 Field Policies

Planning outputs:

- `experiments/results/c067_yousfi_fridrich_field_equations_20260502/top2_plan.json`
- `experiments/results/c067_yousfi_fridrich_field_equations_20260502/top1_plan.json`

Top2-base field-policy byte screens:

```text
top0008: 225202 bytes, SHA d0fbc8fbc0c407963faaf82947e28fb3c5d86f941a751d7ede41c6933d8d3203, disagreement 0.011163185967339409
top0016: 225356 bytes, SHA da9b825d9a684aa3ae76cf545172af8938a643a64dfaab8b4c1a17579aa4f299, disagreement 0.011156268649631077
top0032: 225231 bytes, SHA dd722075f2550df38c366ba55aab046f4328fdcfaddc19fb73f2544141afda33, disagreement 0.011143798828125
top0064: 225537 bytes, SHA e0502b11c4bd843bcbd8b76fe529d3006b985eae2c3d66b79b0f8d47d2a52cf8, disagreement 0.011121741400824652
top0128: 225757 bytes, SHA 04125d2c7c22e173692bfc50c766eddae540a416f147896abce54d4e65be8e1d, disagreement 0.011076473659939237
```

Interpretation: these save about 50 KB but remain close to the known
pose-toxic top2 residual geometry. They are lower priority for exact CUDA than
the pre-existing CMG3A body200 candidate.

The guarded top2 plan is:

```text
experiments/results/c067_yousfi_fridrich_field_equations_20260502/top2_plan_guarded.json
candidate_policy_count: 0
filtered_negative_field_energy: top0008, top0016, top0032, top0064, top0128
```

Top1-base field-policy byte screens:

```text
top0064: 129275 bytes, SHA 8bf7ace9e406586a151fb0fa2cb1914af3d871816d795a0770cbdeb51d84a17f, disagreement 0.03536205715603299
top0128: 129483 bytes, SHA fe347040c2226d057b8d20029212d96ea58b4b7a10a7155db34af8b548572aa9, disagreement 0.03524286905924479
top0256: 129742 bytes, SHA 2b3d340213c66c2d24e6abfa0b4974b37617b316fbc03aefb750f7efcbf3d1d1, disagreement 0.03501391092936198
```

Interpretation: these are too lossy for current exact-eval priority.

## Body200 Residual Field Planning

The body200 CMG3A manifest can now be consumed directly by the atom-ledger
planner:

```text
experiments/results/c067_cmg3a_body200_atom_field_20260502/body200_ledger.json
residual_pixels: 359451
atom_count: 110021
```

Default positive-energy row-run policy planning emits no candidates:

```text
experiments/results/c067_cmg3a_body200_atom_field_20260502/body200_field_plan.json
candidate_policy_count: 0
positive_row_run_net_count: 0
```

An explicit negative-energy cliff-map plan exists for analysis only:

```text
experiments/results/c067_cmg3a_body200_atom_field_20260502/body200_field_plan_allow_negative_guarded.json
candidate_policy_count: 6
required_base_runs_per_row: 1
```

Interpretation: if body200 exact CUDA survives, the next repair basis should
not be more raw row-run atoms. The likely next basis is larger connected
components, boundary/DCT/spline patches, pose-conditioned mask repairs, or
return to QZS3/JFG public-floor atom work.

## Active Diagnostic

The highest-EV CMG3A diagnostic was harvested state-derived from the Lightning
artifact mirror after the SDK status regressed `Running -> Pending` and later
resolved terminal `Completed`:

```text
lane: c067_cmg3a_body200_diag_l40s
job: exact_eval_c067_cmg3a_body200_l40s_20260502T114231Z
archive: experiments/results/c067_cmg3a_adaptive_body200k_20260502/archive.zip
bytes: 257676
sha256: 8c6569b00dc999abbbeedddfc3003ba769484fa34b3abe4555a5d56152fec594
result: L40S exact CUDA score 7.014092131069795
posenet: 4.2056942
segnet: 0.00357384
samples: 600
promotion_eligible: false
component_gate_triggered: true
paper_claim_grade: A-negative scoped forensic
artifact_dir: experiments/results/lightning_batch/exact_eval_c067_cmg3a_body200_l40s_20260502T114231Z/
```

Interpretation: the archive saved `18538` bytes versus C-067, but PoseNet
collapsed by about `8472.9x` the C-067 reference. This retires only the
measured CMG3A body200 row-run implementation. It does not kill predictive mask
grammar, learned geometry-preserving decoders, QZS3/JFG public-floor packer
work, or component-trace-driven active subspace repair.

## 2026-05-02 Body-Budget Hardening

Banach's adversarial review correctly identified a configuration bug class in
the CMG3A `--target-body-bytes` selector: compressed payload bytes are not
mathematically monotonic over a deterministic priority prefix, so binary search
can select the wrong prefix or make overconfident provenance claims.

Permanent fix:

- `experiments/build_cmg3_adaptive_runs_candidate.py` now treats target-body
  prefix selection as nonmonotonic.
- `body_search_mode={auto,exhaustive,coarse}` is available through CLI/env/
  kwargs.
- exhaustive mode scans every prefix and is the only exact prefix optimizer.
- coarse/auto large-candidate mode records sampled-prefix coverage,
  unevaluated prefix count, and `monotonic_binary_search=false`.
- focused regression coverage proves a nonmonotonic body curve selects the
  later feasible prefix that the old binary-search assumption could miss.

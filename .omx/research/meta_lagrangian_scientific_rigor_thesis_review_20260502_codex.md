# Meta-Lagrangian Scientific Rigor Thesis Review - 2026-05-02

Evidence grade: `derivation` + `external` + cited exact local artifacts.
Score claim: `false` except the explicitly cited A++ QZS3/QP1 r8 T4 anchor.
Scope: writeup, paper, and thesis-review material only. This note does not
dispatch jobs, mutate state, or change archive artifacts.

## Public And Local Evidence Boundary

Public sources are useful as external design signals, not as local score
evidence:

- Challenge objective, submission format, and scoring formula:
  [comma video compression challenge](https://github.com/commaai/comma_video_compression_challenge).
- Public PR #67 reports `qpose14_qzs3_filmq9g_slsb1_r55` at rounded `0.31`
  with the QZS3/QP1 public-floor style:
  [PR #67](https://github.com/commaai/comma_video_compression_challenge/pull/67).
- Public PR #65 reports `henosis_qz_n3z_r25_clean` at rounded `0.32` with a
  materially different postprocess/side-stream design:
  [PR #65](https://github.com/commaai/comma_video_compression_challenge/pull/65).

The current local A++ frontier for writeup purposes is:

```text
name=QZS3/QP1 line-search r8
score=0.3159064496962538
archive_bytes=276426
archive_sha256=c68b7522d4d2c8a89771e491c5956b0fb3460e744b3adba9f410f053783044b1
hardware=Tesla T4
evidence_grade=A++
eval_path=archive.zip -> inflate.sh -> upstream/evaluate.py
```

The r13 T4 run is pending/running unless a later local artifact shows terminal
evidence. It must not be written as a frontier. Sub-`0.30` must also remain a
forecast: scalar pose line search alone is unlikely to close the gap. The
credible sub-`0.30` route likely needs PR65-style postprocess or side-channel
atoms, or an equivalent atom family that reduces PoseNet and/or SegNet while
staying inside the charged archive.

## Formal Archive And Atom Model

Let `U = 37,545,489` be the original archive byte denominator. For any complete
submission archive `A`, define:

```text
B(A) = archive byte count
Sg(A) = mean SegNet distortion over n=600 samples
P(A) = mean PoseNet distortion over n=600 samples
Score(A) = 100*Sg(A) + sqrt(10*P(A)) + 25*B(A)/U
lambda_rate = 25/U = 6.65858953875e-7 score/byte
```

An atom is a charged, deterministic modification to a complete archive build:

```text
a = (payload_delta, decoder_delta, metadata_delta, support, provenance)
```

with:

```text
c_a(A, X) = B(A + X + a) - B(A + X)
ds_i(a | A, X) = s_i(A + X) - s_i(A + X + a)
dp_i(a | A, X) = p_i(A + X) - p_i(A + X + a)
```

Positive `ds_i` and `dp_i` mean the atom reduces distortion on pair `i`. The
first-order component benefit at anchor `A` is:

```text
w_s = 100/n
w_p(A) = 5 / (n * sqrt(10*P(A)))

component_benefit(a | A, X) =
  E[sum_i support_i(a) * (w_s*ds_i(a | A, X) + w_p(A)*dp_i(a | A, X))]
```

The confidence-adjusted marginal utility is:

```text
U_a(A, X) =
  component_benefit(a | A, X)
  - lambda_rate*c_a(A, X)
  - beta*uncertainty(a | A, X)
  - gamma*interaction_risk(a | A, X)
  + eta*synergy(a | X)
```

An atom may enter a candidate archive only when `U_a(A, X) > 0` under the
current evidence grade. The archive is still not a score result until exact
CUDA auth eval lands on the complete bytes.

## Lagrangian Objective And Waterline

For a discrete selected set `X`, the planning objective is:

```text
maximize Phi(X | A) =
  E[Score(A) - Score(A + X)]
  - beta*Uncertainty(X)
  - gamma*InteractionRisk(X)
  + eta*Synergy(X)
```

equivalently:

```text
minimize E[100*Sg(A+X) + sqrt(10*P(A+X)) + 25*B(A+X)/U]
       + beta*Uncertainty(X)
       + gamma*InteractionRisk(X)
       - eta*Synergy(X)
```

For a continuous relaxation with atom intensities `x_a`, KKT stationarity gives
the waterline:

```text
active atom:
  d E[component_distortion_score] / d bytes_a
    = lambda_rate
      + d(beta*Uncertainty + gamma*InteractionRisk - eta*Synergy)/d bytes_a

inactive atom:
  d E[component_distortion_score] / d bytes_a
    <= the same waterline
```

For real archive work the atoms are discrete, Brotli interactions are
nonconvex, and QP1 integer proposals are rounded. The practical solver is
therefore a beam/knapsack water-fill:

1. Freeze anchor archive `A_k` by SHA, bytes, exact JSON, and source manifest.
2. Enumerate atoms by stream: pose, mask, renderer, entropy pack, decoder,
   postprocess, metadata layout.
3. Reject atoms that fail decoded parity, payload closure, deterministic
   archive, or runtime constraints before neural eval spend.
4. Rank atoms by `U_a/c_a` using exact component traces where available.
5. Build a small set of complete candidate archives from the waterline.
6. Evaluate exact CUDA on the full archive. Promote only the evaluated bytes.

## Interaction And Synergy Terms

Additivity is a hypothesis, not a theorem. For atoms `a` and `b`:

```text
I_ab(A) =
  [Score(A) - Score(A+a+b)]
  - [Score(A) - Score(A+a)]
  - [Score(A) - Score(A+b)]
```

`I_ab > 0` is positive synergy. `I_ab < 0` is antagonism. Until measured, the
paper should charge a conservative `interaction_risk` for atom pairs with any
of these coupling keys:

```text
same_pose_stream
same_brotli_member
same_renderer_tensor_group
same_mask_geometry
same_pair_support
same_decoder_runtime_path
same_scorer_sensitive_region
```

Examples:

- QZS3 byte packing and QP1 pose line search are coupled through PoseNet
  geometry and Brotli member layout.
- PR65-style side streams may be synergistic with QZS3/QP1 only if their
  postprocess changes component traces without destroying decoded parity.
- Sparse Alpha repair atoms that look good on a C-051 trace can be invalid on
  an out-of-basin archive where PoseNet collapse is global.

## Confidence Penalties

The optimizer should not treat all evidence grades equally. A simple planning
penalty is:

```text
uncertainty(a) =
  z * sigma_hat(a)
  + stale_artifact_penalty(a)
  + proxy_gap_penalty(a)
  + non_t4_hardware_penalty(a)
  + interaction_unknown_penalty(a)
  + custody_gap_penalty(a)
```

Recommended defaults for paper language:

| Evidence | Planning use | Penalty stance |
|---|---|---|
| `A++` exact T4/equivalent archive | May anchor frontier | Minimal, but still requires review |
| `A` exact CUDA archive | May rank locally with caveat | Hardware caveat |
| `A-negative` exact CUDA regression | Diagnose measured implementation only | No broad kill |
| `B` diagnostic CUDA | Prior for next atom | High custody penalty |
| `empirical` bytes/roundtrip/loss | Screen and prioritize | No score benefit until exact eval |
| `derivation` | Math and bounds | No empirical benefit |
| `external` public PR/paper | Design prior | Cannot rank our archive |
| `prediction` | Roadmap only | Must name falsification gate |
| `invalid` CPU/MPS/proxy/stale | Exclude from ranking | Infinite promotion penalty |

## Exact-Eval Checkpoints

Each water-fill layer needs a different checkpoint cadence:

| Layer | Example atoms | Required checkpoint |
|---|---|---|
| Anchor archive | QZS3/QP1 r8 | A++ exact CUDA before frontier wording |
| Pose line search | scalar, asymmetric, gradient, DCT, pair-window QP1 atoms | Exact eval at every deployable checkpoint |
| PR65-style side stream | postprocess, residual, length table, patch atoms | Decoded parity smoke, then exact eval |
| Packer atom | single blob, Brotli layout, metadata split | Byte parity and exact eval if components can change |
| Component trace | per-pair PoseNet/SegNet contributions | Exact trace on the current anchor, not stale anchors |
| Stack | any combined atom family | Own archive eval, no additive score claims |

H100/L40S results may guide proposal distribution and queue ordering. T4 or
contest-equivalent exact artifacts are required for A++ promotion wording.

## Deterministic Archive Custody

A candidate archive is paper-usable only if the evaluated bytes are preserved
and reproducible. Minimum custody bundle:

```text
archive.zip
archive_custody.json
contest_auth_eval.json
adjudication.json
provenance.json
archive_manifest.txt or zipinfo output
inflate_stdout.log
inflate_stderr.log
evaluate_stdout.log
evaluate_stderr.log
build_command.txt
eval_command.txt
source_manifest.json
git_status_short.txt
runtime_probe.json
```

Hard rules:

- Copy the exact evaluated `archive.zip` into the result directory before
  scoring or immediately after immutable staging.
- Record archive SHA-256 and bytes in both custody and eval/adjudication
  surfaces.
- Use the canonical path `archive.zip -> inflate.sh -> upstream/evaluate.py`.
- Exclude hidden files, resource forks, debug payloads, absolute paths, parent
  traversal, and score-affecting sidecars.
- Recompute the score from JSON components before paper wording.
- Preserve logs for both positive and negative outcomes.

## Negative-Result Policy

A disappointing exact eval is useful only if scoped correctly:

1. Preserve archive, JSON, logs, SHA, bytes, manifest, source provenance, and
   command.
2. Recompute score and classify device, sample count, path, component gates,
   payload closure, and runtime.
3. Classify failure mode: legitimate regression, harness bug, archive bug,
   no-op encode-discard, dead flag, CPU/MPS/proxy leakage, sidecar dependency,
   codec attribution confound, global manifold failure, or indeterminate.
4. State the narrow retirement scope: run, config, measured implementation, or
   mathematical family.
5. Name revival evidence. Do not write family/method kill language without
   independent exact evidence or a real impossibility proof.

This is especially important for the public-floor basin. A failed scalar pose
radius does not falsify QP1. A failed out-of-basin Q-FAITHFUL checkpoint does
not falsify QZS3. A PR65-style side-stream atom that regresses one archive may
still be the right missing family for sub-`0.30` after decoded parity and
component-trace support are repaired.

## Thesis-Advisor Review

### Claim hierarchy

The thesis should separate these claims explicitly:

| Claim | Current status | Allowed wording |
|---|---|---|
| QZS3/QP1 r8 reached `0.3159064496962538` | A++ T4 exact archive | "current verified frontier" |
| r13 improves r8 | pending/running | "candidate continuation, not claimed" |
| Public PR #67 and PR #65 define design basins | external | "public design signals" |
| Scalar pose line search will reach sub-`0.30` | unsupported | do not claim |
| Sub-`0.30` is plausible with side-channel/postprocess atoms | prediction | "hypothesis requiring A++ archive proof" |
| Atom water-fill is optimal for the discrete archive | unproven | "planning relaxation and selection discipline" |
| Exact eval custody is sufficient for paper claims | true only with review | "necessary, plus review and reproduction bundle" |

### Required ablations before final paper claims

| Ablation | Required evidence |
|---|---|
| C-053 fixedslice vs C-054 vs r8 | Exact T4 JSON, bytes, SHA, component deltas |
| Scalar vs asymmetric vs gradient QP1 proposals | Same anchor, same packer, exact eval per accepted archive |
| DCT/spline/pair-window pose atoms | Byte delta and PoseNet delta on complete archives |
| PR65-style side-channel atom | Isolated archive with decoded parity and component trace |
| PR65-style atom stacked on r8 | Own exact eval, no additive-delta claim |
| Packer-only byte atom | Decoded payload SHA parity and rate arithmetic |
| H100 diagnostic vs T4 promotion | Identical archive SHA comparison where possible |
| Component-trace stability | Trace on current anchor and current candidate, not only older C-051 |
| Deterministic rebuild | Rebuild archive and match SHA or document non-determinism |
| Negative controls | Deliberately rejected sidecar/proxy/MPS rows remain excluded |

### Production hardening

Before submission, the archive path should be boring:

- Metadata-driven QZS3/QP1 parsing, not brittle fixed offsets.
- All score-affecting payload bytes inside the archive.
- No scorer access, no network, no local sidecars, no nondeterministic runtime
  generation.
- Bounded inflate runtime on T4/equivalent hardware with preserved logs.
- Reproducible dependency selection and CUDA wheel pinning.
- Zip-slip-safe extraction and hidden-file rejection.
- Manifest-driven decoder dispatch with clear failure on malformed payloads.
- Component gates treated as first-class failures even when total score looks
  attractive.

### Advisor risks

The most serious risks are:

1. Overstating sub-`0.30` from r8/r13 trend lines without a new atom family.
2. Treating public PR anatomy as local evidence rather than external prior.
3. Letting H100 diagnostics leak into A++ wording.
4. Assuming atom additivity through nonconvex Brotli, PoseNet, and SegNet
   interactions.
5. Failing to preserve exact archive custody after a fast loop result.
6. Writing a broad negative verdict from one out-of-basin exact regression.
7. Spending final eval budget on scalar search after marginal utility falls
   below the PR65-style side-stream opportunity.

### Advisor verdict

The meta-Lagrangian water-fill framing is strong enough for the thesis as a
systems method: it turns the challenge from ad hoc score chasing into a
charged-atom allocation problem with evidence-gated promotion. It is not yet a
proof of optimality and it is not a sub-`0.30` result. The final paper should
present r8 as the current exact frontier, r13 as pending unless terminal files
prove otherwise, and PR65-style postprocess/side-channel atoms as the likely
missing class that must be measured before any sub-`0.30` or Shannon-floor
language is allowed.

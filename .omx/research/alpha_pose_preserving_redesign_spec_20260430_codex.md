# Alpha Pose-Preserving Redesign Spec - 2026-04-30

Adjacent source docs:

- `grand_council_paradigm_shift_to_shannon_floor_20260430.md`
- `paradigm_alpha_mask_overhaul_audit_20260430.md`
- `council_lane_12_nerv_round{1,2,3}_20260430.md`
- `shannon_floor_claim_matrix_20260430_codex.md`

## Verdict

Continue Alpha/mask-payload research aggressively, but retire the measured
Lane 12 `jsonfix40` implementation/config only.

Exact CUDA evidence for `jsonfix40` showed catastrophic PoseNet drift even
though the mask payload rate was strong. That is geometry failure, not proof
that NeRV/INR/mask compression is dead.

## Known Exact Negative Evidence

Authoritative artifact:

```text
experiments/results/lane_12_nerv_20260430_codex_jsonfix40/contest_auth_eval.json
```

Key facts:

```text
device = cuda
n_samples = 600
score_recomputed_from_components = 26.03719330455429
avg_posenet_dist ~= 49.7784996
avg_segnet_dist ~= 0.03528685
archive_size_bytes = 296478
archive_sha256 = 864549cc648f0b3a023076c11812ccd0f10b1d013ed3fd6bb24d20bbcde85c97
```

Allowed conclusion: `jsonfix40` measured implementation/config retired.

Forbidden conclusion: NeRV, INR, mask compression, or Alpha family killed.

## Primary Hypothesis

Rate mechanics worked; geometry preservation failed.

Likely confound: NeRV training targeted fresh SegNet argmax masks, while the
renderer and optimized poses were fitted against the decoded baseline
`masks.mkv` stream. Swapping only the mask payload while reusing renderer/poses
can make small class/boundary disagreements produce large PoseNet drift through
temporal mask embeddings and motion gating.

## Required Diagnostics

All diagnostics compare candidate masks against the decoded baseline archive
mask stream, not fresh SegNet labels.

Promotion diagnostic targets:

```text
global Hamming <= 0.001
2px boundary-ring disagreement <= 0.002
lane/vehicle recall >= 0.999
temporal pair-diff disagreement <= 0.002
connected-component centroid jump <= 1 px
renderer embedding L2 drift: tracked and monotonically improved vs jsonfix40
```

First-pass exploratory targets:

```text
global Hamming <= 0.003
2px boundary-ring disagreement <= 0.005
temporal pair-diff disagreement <= 0.004
```

These diagnostics do not promote. They only decide whether exact CUDA eval is
worth the spend.

## Alpha-Geo Sequence

### Alpha-Geo-0: Stale-Pose Isolation

Keep exact `jsonfix40` `masks.nrv`, but regenerate `optimized_poses.bin` against
those decoded masks and the existing renderer. Build a new archive and run
exact CUDA eval.

Interpretation:

- PoseNet recovers: failure was mostly stale pose calibration.
- PoseNet remains collapsed: decoded mask field is geometrically incompatible.

This is the shortest causal experiment and should run first.

### Alpha-Geo-1: Train To Archive-Decoded Masks

Change NeRV training target from fresh SegNet argmax masks to the exact decoded
baseline `masks.mkv` distribution consumed by the renderer. Then regenerate
poses and exact-eval.

Required outputs:

- target mask stream SHA-256,
- decoded baseline mask diagnostics,
- candidate-vs-baseline diagnostics,
- regenerated pose provenance,
- deterministic archive manifest,
- exact CUDA `contest_auth_eval.json`.

### Alpha-Geo-2: NeRV Plus Sparse Correction

Use NeRV as the low-rate base, then encode charged residual corrections for
PoseNet-sensitive pixels:

- lane boundaries,
- vehicles/objects,
- temporal-diff pixels,
- connected-component centroid stabilizers,
- high renderer-embedding-drift regions.

All residual side information must live inside `archive.zip`.

### Alpha-Geo-3: Geometry-Weighted Training Objective

Add training-only objective terms that preserve renderer inputs:

- weighted CE by boundary and temporal-change masks,
- embedding preservation for `e_t`, `e_t1`, and `|e_t1 - e_t|`,
- temporal consistency loss on adjacent decoded masks.

Score truth remains exact CUDA archive eval, not training loss.

### Alpha-Geo-4: Renderer/Pose Co-Design

If mask-only rescue fails, finetune/rebuild renderer against the decoded
candidate mask stream, then regenerate poses. This becomes a stack archive and
requires its own exact eval.

## Exact Gates

Promotion candidate requires:

```text
contest_auth_eval.json present
provenance.device = cuda
n_samples = 600
recomputed score finite and formula-consistent
archive SHA/bytes match manifest
no score-affecting sidecars
payload closure clean
PoseNet <= 0.0042 for promotion
PoseNet > 0.01 triggers fail-fast forensic review
SegNet <= 0.00501, stretch target <= 0.0042
score beats PFP16 A++ 1.043987524793892
```

PFP16 A++ comparator:

```text
archive_sha256 = 0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f
archive_bytes = 686635
score = 1.043987524793892
```

## Implementation Order

1. Add decoded-baseline-mask extraction and diagnostics utility.
2. Add renderer-embedding drift diagnostic for baseline vs candidate masks.
3. Add pose regeneration path that accepts a decoded candidate mask stream.
4. Run Alpha-Geo-0 exact CUDA eval.
5. Add NeRV training target mode for decoded baseline `masks.mkv`.
6. Run Alpha-Geo-1 exact CUDA eval.
7. Add sparse residual codec only where diagnostics identify PoseNet-sensitive
   damage.
8. Run exact stacked archive eval before any paper/deploy claim.

## Tests

- Unit: decoded `masks.mkv` target path is used when requested; no silent fresh
  SegNet fallback.
- Unit: candidate-vs-baseline mask diagnostics detect global, boundary,
  temporal, and component centroid drift.
- Unit: residual side information is included in archive manifest/bytes.
- Integration: `.nrv` candidate archive validates through contest auth whitelist.
- Integration: pose regeneration path records mask SHA and renderer SHA.
- Integration: exact eval wrapper refuses CPU/MPS promotion artifacts.


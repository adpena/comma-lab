# Compression Grand Council Extreme Audit - 2026-05-03

This is a worker ledger for compression-only opportunities around the current
A++ frontier. It is not a score ledger and makes no new score claim. Exact score
truth remains `archive.zip -> inflate.sh -> upstream/evaluate.py` on CUDA.

## Anchor

Current exact A++ frontier:

- candidate: `c067_pr75_qp1_top40_p6`
- archive: `experiments/results/lightning_batch/exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/archive.zip`
- score: `0.3154707273953505`
- bytes: `276342`
- sha256: `0ec53e5b871149ed6eea56c0b9bcca3baec998d5bfad4f371979e0c90e62cea8`
- SegNet: `0.00061038`
- PoseNet: `0.00049601`
- samples: `600`
- hardware: Tesla T4

At unchanged SegNet/PoseNet, score `<0.314` requires saving `2209` archive
bytes, i.e. target archive bytes `<=274133`.

## C-089 Byte Anatomy

Profiler artifact:

- JSON: `experiments/results/compression_grand_council_extreme_audit_20260503/c089_bit_budget.json`
- Markdown: `experiments/results/compression_grand_council_extreme_audit_20260503/c089_bit_budget.md`

Archive structure:

| scope | bytes | score-rate term | interpretation |
|---|---:|---:|---|
| ZIP total | `276342` | `0.184004794824` | scored archive bytes |
| ZIP overhead | `100` | `0.000066585895` | already near-minimal single `p` member |
| single member `p` | `276242` | `0.183990145927` | stored, entropy saturated |
| P6 header | `12` | `0.000007990307` | self-describing PR75/P6 lengths |
| `masks.mkv` | `219472` | `0.146137396160` | dominant stream; generic nested compression saves `0` |
| `renderer.bin` | `55965` | `0.037264796311` | second-largest; generic nested compression saves `0` |
| `seg_tile_actions.bin` | `116` | `0.000077239639` | already tiny P6 delta-varint actions |
| `optimized_poses.qp1` | `677` | `0.000450786511` | already tiny; cannot bridge target by bytes alone |

Key result: the outer byte stream is already entropy-saturated. `brotli`,
`zlib`, and `lzma` probes save `0` on `p` and on every semantic segment.
Sub-0.314 needs semantic compression or component improvement, not generic
repacking.

## Nearby Evidence

Exact/proxy matrix artifact:

- `experiments/results/compression_grand_council_extreme_audit_20260503/nearby_exact_eval_matrix.json`

Adversarial notes:

- Public PR75 reports `276741` bytes and rounded `0.31`; C-089 is already `399`
  bytes smaller than that archive, but still above the hard `0.314` threshold.
- PR67/C-067-style generic fixed-slice recompression is exhausted: prior public
  accounting records `masks.mkv=219472`, `renderer.bin=55965`, `pose=677`,
  `zip overhead=100`, with zero nested savings.
- P6 action deltas are now proven useful but small: C-089 action/pose stack is
  a `44` byte win versus top40 P3 plus exact component luck, not a 2KB byte
  lever.
- Renderer shrink candidates that save `>2209` bytes exist locally, but known
  broad block-size or zero-FP4 variants fail pose-safety or exact CUDA.
- Mask/topology byte wins are huge on paper, but repeated exact diagnostics
  show PoseNet/SegNet collapse unless a geometry escape proof exists.

## Ranked Top-10 Opportunities

Score delta estimates use `25 / 37,545,489 = 6.658589531e-7` per archive byte.
Negative means lower score. Component deltas are intentionally conservative
unless exact CUDA already exists.

| rank | opportunity | expected byte delta | expected score delta | component-risk class | implementation cost | exact-eval readiness | required preflights |
|---:|---|---:|---:|---|---|---|---|
| 1 | Fixed-mask/fixed-pose renderer self-compression from active burns | `-2209` to `-8000+` if export lands | `-0.00147` to `-0.00533+` | medium-high; renderer drift can move PoseNet | medium; harvest/export/transcode/gate | blocked until active Modal burn output exists | raw export closure, `preflight_trained_renderer_transplant.py`, `preflight_renderer_transplant_pose_safety.py`, exact same masks/poses/actions SHA |
| 2 | Learned/local bit-depth renderer allocator, not naive global QZS3 block size | `-1000` to `-3500` | `-0.00067` to `-0.00233` | high; b96/b128-style byte wins failed local pose safety | medium-high; tensor/group allocator plus parity gate | not ready; existing byte wins blocked | per-tensor/group changed proof, pose-safety on sampled pairs, no inherited A-negative parent, T4 only after local gate |
| 3 | PR75 tile-action dictionary v2 / action compiler | `-20` to `-250`; component gain unknown | `-0.000013` to `-0.000166` plus possible SegNet/PoseNet gain | low-medium; tiny charged actions can perturb scorer beneficially | active in partner scope; do not edit here | wait for Gibbs/active worker results | decoded action parity, source-not-preserving proof, component-trace calibration, P6 parser/runtime parity |
| 4 | QP1 pose active-subspace perturbation | bytes roughly neutral (`<700` surface) | component-only; needs `>0.00147` score gain for sub-0.314 alone | high; PoseNet cliffs and cross-hardware drift | active in partner scope; do not edit here | wait for Lagrange/active worker results | QP1 semantic decode/encode parity, exact pose component trace, no proxy promotion |
| 5 | Exact C-089 + safe micro renderer shrink stacking | `-1` to `-500` locally safe candidates | `-0.000001` to `-0.000333` | medium; one `-452B` candidate exact-failed PoseNet despite local safety | low | mostly blocked by inherited exact A-negative | inherited-parent check, T4 exact on only non-inherited candidates with stronger parity thresholds |
| 6 | SJ-KL residual overlay on PR75/C-089 parent | `+250` to `+450` bytes | rate `+0.00017` to `+0.00030`; needs component win | medium-high; C067 exact showed tiny/negative net | low-medium | not enough EV for immediate T4 | `SJKL_REQUIRE_APPLIED=1`, applied-pair log proof, exact PR75 component response before promotion |
| 7 | Mask stream learned codec / NeRV / multimask reconciliation | potential `-20000` to `-90000` | huge rate win if geometry preserved | extreme; exact mask-family negatives collapse PoseNet | high; needs geometry-safe representation, not simple reencode | not ready | L2/geometry clearance, decoded mask disagreement gate, hard-pair foveation selector proof, exact L40S diagnostic before T4 |
| 8 | Reversed-base CDO1 / symbolic path grammar | theoretical `-90000+` | theoretical `-0.06+` | extreme; lower-bound economics no byte-closed safe archive | high | not ready | byte-closed builder, residual geometry gate, unmatched atom fail-closed, exact diagnostic only after geometry pass |
| 9 | Ego/foveation/telescopic selectors as atom-ranking fields | indirect | indirect | medium if selector-only; high if runtime warp | medium | planning-only | treat as atom weights only until charged runtime consumes it; camera/ego provenance, pair/class ablation |
| 10 | Container/ZIP/wire overhead trimming | `0` to `-100` max | `0` to `-0.000067` | low but already exhausted | low | not worth T4 alone | central/local header consistency, duplicate-name guard, no parser divergence, semantic segment profiler |

## Immediate Decision

No new remote dispatch from this worker. The only current compression route with
sub-0.314 byte capacity and acceptable contest-faithful shape is rank 1: harvest
the already-active fixed-renderer burns, then gate them locally before exact
CUDA.

Do not dispatch these without new evidence:

- `renderer_shrink_pr75_c088_20260503_worker/qzs3_b0096...` and
  `qzs3_b0128...`: byte-sufficient, but pose-safety failed.
- `renderer_parity_shrink_search_20260503_worker/zero_fp4_*` broad variants:
  byte-sufficient variants fail pose-safety; the locally safe `frame1_head_0.1`
  exact-failed PoseNet on T4.
- `c082_fast_packer_worker_20260503/.../shrink_queued_bb8d...`: inherits the
  exact A-negative `bb8d...` parent and saves only `10` more bytes.
- raw mask codec or multimask byte wins without a geometry escape proof:
  existing exact diagnostics show component collapse.

## Local Hardening Landed

The byte profiler previously reported C-089 as an opaque single `p` member. I
added PR75/P3-P6 semantic parsing so future audits surface the actual streams
and no-op risks:

- `experiments/archive_bit_budget_profiler.py`
- `src/tac/tests/test_archive_bit_budget_profiler.py`

This is development evidence only. It imports no scorer/runtime modules, makes
no score claim, and is safe for local byte audits.

## Next Archive Candidates / Code Tasks

1. Poll active Modal fixed-renderer burns. For the first checkpoint/export:
   build byte-closed candidate preserving C-089 masks, QP1 pose, and P6 actions;
   run transplant and pose-safety preflights; dispatch T4 only if archive bytes
   `<=274133` or component gain has exact diagnostic support.
2. Integrate Gibbs action-dictionary v2 output if it produces a non-no-op
   P6/P7 action payload with either `>100B` byte improvement or trace-backed
   component benefit. This can stack with rank 1.
3. Integrate Lagrange QP1 active-subspace output only if semantic QP1 closure
   and component-trace evidence show a plausible `>0.00147` score gain or a
   clean small component gain stackable with renderer bytes.
4. Build a renderer allocator that changes only low-sensitivity tensor groups,
   not global block size. Candidate must record per-group bytes, decoded QZS3
   parity, and sampled output deltas before exact eval.
5. Keep mask grammar/NeRV/CDO1 work local until geometry escape proof exists.
   The next useful artifact is a hard-pair/foveation selector proof that reduces
   disagreement while keeping the charged residual under the 2.2KB threshold.

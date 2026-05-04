# Self-Compression Full-Pipeline Audit

Date: 2026-05-02

Evidence standard: empirical byte/runtime analysis unless an exact CUDA artifact
is cited. No score claim is made here. Score truth remains
`archive.zip -> inflate.sh -> upstream/evaluate.py` on CUDA, with T4/equivalent
needed for A++ promotion.

## Anchor

Current A++ T4 frontier used as the self-compression anchor:

```text
id=C067
score=0.31561703078448233
archive_bytes=276214
archive_sha256=226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a
segnet=0.00061244
posenet=0.00049637
evidence=experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/contest_auth_eval.json
```

At unchanged distortion, breaking `0.300` requires removing `23454` bytes; the
buffered planning target is `23455` bytes.

## Byte Anatomy

Profile artifact:

```text
experiments/results/c067_archive_byte_accounting_20260502/archive_byte_accounting.json
experiments/results/c067_archive_byte_accounting_20260502/archive_byte_accounting.md
```

C067 archive is a single member `p` with `100` bytes ZIP overhead. Generic
recompression probes are exhausted. The byte attack surface is therefore
representation grammar, not stronger zip/brotli/lzma around the same bytes.

```text
masks.mkv:
  encoded=219472 bytes
  decoded=223385 bytes
  codec=brotli_av1_obu
  rate_score=0.14613739616
  nested_recompression_savings=0
  buffered_gap_fraction=10.6866%

renderer.bin:
  encoded=55965 bytes
  decoded=59288 bytes
  codec=brotli_qzs3
  rate_score=0.037264796311
  nested_recompression_savings=0
  buffered_gap_fraction=41.9083%

optimized_poses.bin:
  encoded=677 bytes
  decoded=7200 bytes
  codec=public_qp1_brotli
  rate_score=0.000450786511
  nested_recompression_savings=0
```

Conclusion:

1. Mask stream is the only single stream where a modest relative byte reduction
   closes the sub-0.300 gap.
2. Renderer stream can help, but only after a local layout beats the existing
   QZS3/QZS4-like packing by meaningful bytes.
3. Pose stream is polish or distortion leverage, not a byte lever.
4. Container overhead is immaterial.

## PMG-HOTSPOT Route

This is the active mask self-compression experiment family. It replaces
`masks.mkv` with charged `masks.cmg3`, preserving deterministic runtime decode
and SHA custody.

Exact jobs:

```text
bz2:
  job=exact_eval_pmg_hotspot_c067_t4_20260502T1402Z
  archive_bytes=187144
  archive_sha256=3ab4d0c85ae15325b61b7838c375383a460e7ad1b911b677101fda50e2611c53
  decoded_mask_sha256=5b2c7a4a4afd47076d42e874d86928e65159f740b9635197e232504a61cbafdc
  state=.omx/state/pmg_hotspot_c067_t4_batch_jobs_20260502T1402Z.json

lzma_xz:
  job=exact_eval_pmg_hotspot_c067_lzma_t4_20260502T1408Z
  archive_bytes=184598
  archive_sha256=20b7cfe3d034f173be1d71193a71d9fa79c7da118790f92209ea0bf09643e660
  decoded_mask_sha256=5b2c7a4a4afd47076d42e874d86928e65159f740b9635197e232504a61cbafdc
  state=.omx/state/pmg_hotspot_c067_lzma_t4_batch_jobs_20260502T1408Z.json
```

The lzma archive strictly dominates the bz2 archive by `2546` bytes with the
same decoded mask SHA. The builder default is now `lzma_xz`.

PMG candidate grid:

```text
stride2_lzma:
  bytes=184598
  pixel_disagreement=0.015210562811957465
  unchanged_distortion_score=0.2546136969352415
  component_budget_to_sub03=0.04538630306475849

stride4_lzma:
  bytes=136967
  pixel_disagreement=0.018023028903537325
  unchanged_distortion_score=0.22289816913907934
  component_budget_to_sub03=0.07710183086092065

stride8_lzma:
  bytes=102456
  pixel_disagreement=0.02741742451985677
  unchanged_distortion_score=0.1999187108078801
  component_budget_to_sub03=0.1000812891921199
```

Staged fallback manifest:

```text
.omx/state/pmg_hotspot_c067_stride48_t4ready_20260502T1414Z_manifest.json
```

Dispatch rule: wait for stride2 exact component evidence. If stride2 is near or
below sub-0.300, dispatch stride4 first. Dispatch stride8 only if the component
trace shows disagreement is mostly outside PoseNet-sensitive geometry or if
stride4 lands safely.

## Renderer Route

Renderer bytes are not currently the fastest sub-0.300 lever:

```text
QZS3 reference raw renderer bytes=59288
QBF1-v1 raw bytes=121618
QBF1-v2 planning bytes=72247
existing QZS3 b128/QZS4-like planning bytes=56300
```

No renderer exact-eval should run until a local layout beats the deployed QZS3
reference by meaningful bytes and the inflate contract is reviewed.

## Pose Route

Pose bytes are too small to close the gap by rate. Pose remains useful only as
distortion slack for more aggressive mask/renderer rate cuts. Treat pose
manifold transfer as a stack-enabler, not a standalone sub-0.300 route.

## Bug Classes Hardened

1. Vast create now fails closed on empty output, stderr-only JSON, nonzero
   return code, and provider error JSON.
2. PMG builder now defaults to the locally superior `lzma_xz` compressor.
3. Byte profiler now marks eval/archive mismatch as reference-only, preventing
   candidate byte profiles from being misread as scored candidate evidence.

Verification:

```text
py_compile touched Python: pass
bash -n scripts/remote_lane_pmg_hotspot_c067_eval.sh: pass
pytest focused self-compression/PMG/Vast tests: 10 passed
git diff --check touched files: pass
```

## Next Decision

Harvest `exact_eval_pmg_hotspot_c067_t4_20260502T1402Z` and
`exact_eval_pmg_hotspot_c067_lzma_t4_20260502T1408Z` as soon as terminal JSON
lands. If the lzma exact score is below `0.300`, it becomes the new promotion
frontier. If it is above `0.300` but component distances are not catastrophic,
dispatch stride4. If stride2 is a scorer-geometry cliff, stop stride4/stride8
and use the component trace to train/select smaller hotspot residual atoms
instead of broad row-span replacement.

## Exact PMG Outcome And Pivot

The stride2 PMG-HOTSPOT exact result landed on T4 and is a scoped A-negative:

```text
job=exact_eval_pmg_hotspot_c067_t4_20260502T1402Z
archive=experiments/results/lightning_batch/exact_eval_pmg_hotspot_c067_t4_20260502T1402Z/archive.zip
bytes=187144
sha256=3ab4d0c85ae15325b61b7838c375383a460e7ad1b911b677101fda50e2611c53
score=30.930370939355445
PoseNet=69.20815277
SegNet=0.04498317
n=600
hardware=Tesla T4
```

The lzma duplicate had the same decoded mask SHA and was stopped after the bz2
result proved a geometry cliff:

```text
job=exact_eval_pmg_hotspot_c067_lzma_t4_20260502T1408Z
final_status=Stopped
cost=0.16115555
score_json=none
decoded_mask_sha256=5b2c7a4a4afd47076d42e874d86928e65159f740b9635197e232504a61cbafdc
```

Comparison against the C067 T4 component trace:

```text
component_diff=experiments/results/pmg_hotspot_candidate_c067_20260502/pmg_vs_c067_component_diff.json
score_delta_total=+30.61475350814471
rate_delta=-0.05930805695459179
pose_delta=+26.23698868074241
seg_delta=+4.437072884356893
```

Top combined failure pairs are concentrated around `502`, `503`, `507`,
`513`, `496`, `575`, `490`, `410`, `512`, `411`, `522`, `505`, and `508`,
but the failure is not localized enough for coarse pair repair to rescue the
implementation. Byte-screened charged pair-protection variants:

```text
curve=experiments/results/pmg_hotspot_candidate_c067_20260502/pair_protection_curve.json
top0:  bytes=184598, protected_pairs=0,   final_disagreement=0.015210562811957465
top10: bytes=194350, protected_pairs=10,  final_disagreement=0.014882481892903646
top20: bytes=203974, protected_pairs=20,  final_disagreement=0.014520161946614584
top40: bytes=221296, protected_pairs=40,  final_disagreement=0.013839297824435763
top80: bytes=253108, protected_pairs=80,  final_disagreement=0.012495210435655381
top120: bytes=284504, protected_pairs=120, final_disagreement=0.011323632134331598
top160: bytes=316096, protected_pairs=160, final_disagreement=0.010214131673177083
top240: bytes=381120, protected_pairs=240, final_disagreement=0.008153389824761284
top320: bytes=441732, protected_pairs=320, final_disagreement=0.006142696804470486
top480: bytes=555918, protected_pairs=480, final_disagreement=0.0022367350260416668
top600: bytes=635102, protected_pairs=600, final_disagreement=0.0
```

The top80 archive is already above the unchanged-distortion sub-0.300 size
threshold: formula score with C067 distortion would be `0.3002317586913289`.
Top120 and above become byte-regressive relative to C067. Whole-pair repair is
therefore too coarse for PMG rescue. The next selector must operate at
region/class/component/boundary scale:

- retire only the measured broad row-span PMG implementation;
- do not submit stride4/stride8;
- use the exact trace as allocator signal for a learned/multimask/geometry
  protection selector below whole-pair granularity;
- route the next self-compression effort to learned mask topology, SJ-KL style
  residual structure, or renderer self-compression only after byte-screen and
  runtime-contract closure.

## Fine-Grained PMG Atom Repair Follow-Up

Whole-pair protection was too coarse, so the PMG exact negative was converted
into an excess-weighted atom ledger using the PMG-vs-C067 T4 component trace:

```text
pair_weights=experiments/results/pmg_hotspot_candidate_c067_20260502/pmg_vs_c067_pair_excess_weights.json
atom_ledger=experiments/results/pmg_hotspot_candidate_c067_20260502_lzma/pmg_vs_c067_residual_atom_ledger.json
atom_count=169708
row_run_atoms=164948
residual_pixels=1794311
score_claim=false
```

The builder now has an explicit charged atom-ledger selection path:

```text
experiments/build_pmg_hotspot_candidate.py --residual-atom-ledger ... --residual-atom-count N
contract=pmg_hotspot_atom_ledger_selection_v1
```

This records ledger SHA, source NPY SHA, selected atom IDs, charged residual
stream SHA/bytes, and source/candidate provenance. The source guard had one
bug-class fix: atom ledgers store an int16 tensor SHA after planner load, while
the PMG builder source is a uint8 tensor. The durable guard now verifies source
`.npy` SHA when present and falls back to tensor SHA only for minimal ledgers.

Byte-screened atom-top archives:

```text
jsonl=experiments/results/pmg_hotspot_candidate_c067_20260502_lzma/pmg_atomtop_byte_screen.jsonl
top64:   185006B, final_disagreement=0.015121883816189237
top128:  185154B, final_disagreement=0.015053320990668403
top256:  185626B, final_disagreement=0.014920874701605902
top512:  186498B, final_disagreement=0.014680074055989584
top1024: 187974B, final_disagreement=0.014245927598741319
top2048: 190182B, final_disagreement=0.01352399190266927
top3072: 193358B, final_disagreement=0.012910308837890625
top4068: 195762B, final_disagreement=0.012382837931315104
```

The byte curve is much better than whole-pair repair: top4068 still saves
`80452` bytes vs C067 and repairs `333572` exact residual pixels. It remains
high-risk because the broad PMG T4 failure was a global scorer-geometry cliff,
not a small local artifact. One L40S diagnostic exact eval was therefore
queued, not a T4 promotion:

```text
job=exact_eval_pmg_hotspot_atomtop4068_l40sdiag_20260502T1445Z
state=.omx/state/pmg_hotspot_atomtop4068_l40sdiag_batch_jobs_20260502T1445Z.json
manifest=.omx/state/exact_eval_pmg_hotspot_atomtop4068_l40sdiag_20260502T1445Z_manifest.json
archive=experiments/results/pmg_hotspot_candidate_c067_atomtop4068_20260502/archive.zip
bytes=195762
sha256=2567dc04185cf20775f1f6c088395aa8df9e4484daa8b25001e940d62a5d6497
hardware=L40S diagnostic
score_claim=false until exact CUDA JSON lands
```

If this diagnostic still collapses, retire PMG row-span residual rescue as a
measured implementation and move the mask-grammar work to learned/multimask or
geometry-preserving topology. If it lands with component distances near C067,
the next action is a T4 confirmation of the same bytes plus a tighter atom
budget sweep around the best density range.

Result: the L40S diagnostic completed as an A-negative scoped forensic result,
not a promotion candidate.

```text
artifact_dir=experiments/results/lightning_batch/exact_eval_pmg_hotspot_atomtop4068_l40sdiag_20260502T1445Z
score=28.41411894150047
avg_posenet_dist=62.34251404
avg_segnet_dist=0.03315286
n_samples=600
gpu=NVIDIA L40S
archive_bytes=195762
archive_sha256=2567dc04185cf20775f1f6c088395aa8df9e4484daa8b25001e940d62a5d6497
paper_claim_grade=A-negative scoped forensic
promotion_eligible=false
component_gate=posenet absolute and relative
score_delta_vs_c067_t4=28.098501910715985
```

The top combined catastrophic pairs are concentrated around the already-known
hard zone rather than randomly distributed:

```text
507, 502, 503, 505, 496, 498, 506, 315
```

Decision: PMG row-span/residual rescue is retired only as this measured
implementation family. The useful signal is that row-run atom repair preserves
bytes but does not preserve the low-dimensional PoseNet manifold. The next
mask-side candidates should use the same atom/water-fill machinery, but the
atom itself must become geometry-aware: multimask reconciliation,
JointFrameGenerator-native mask2/fake1 slot protection, learned topology, or
SJ-KL/pose-conditioned residuals. Do not spend T4 promotion capacity on another
PMG row-run-only archive.

## Adjacent Self-Compression Triage

A read-only self-compression scan of the current tree ranked the next
non-PMG actions as follows:

1. `experiments/line_search_pose_refinement.py` on the C067/C063 QZS3-QP1
   basin remains the highest ready dispatch path. It is a PoseNet/distortion
   lever, not a byte lever. At current rate and SegNet, sub-0.300 needs
   PoseNet around `<=0.000302`; this is hard but directly targets the active
   score gap. Vast.ai credit is currently the practical blocker for the
   existing `scripts/remote_lane_line_search_c067.sh`; Lightning exact-eval
   wrappers exist, but a Lightning training/search wrapper would need a custom
   job command before it is custody-clean.
2. SJ-KL residual coding is the highest-upside implementation lane. Existing
   code and tests provide `src/tac/sjkl_basis.py` and
   `experiments/build_sjkl_residual.py`; byte budget probe projects a smooth
   basis payload around `13982` bytes (`0.009310` rate score). The missing
   production step was archive/runtime integration for charged `sjkl.bin` plus
   CUDA Fisher/build artifacts. The runtime integration is now implemented as
   an additive opt-in slice: `submissions/robust_current/inflate_renderer.py`
   loads optional `sjkl.bin`, decodes basis/coefficient payloads without
   scorer imports, applies residuals only to the q-faithful JointFrameGenerator
   fake1 path when shapes match, and skips nonmatching renderers/shapes.
   `submissions/robust_current/unpack_renderer_payload.py` admits `sjkl.bin`
   as a safe charged member. Focused local verification:
   `.venv/bin/python -m pytest src/tac/tests/test_inflate_renderer_sjkl_runtime.py src/tac/tests/test_sjkl_basis.py -q`
   returned `21 passed`.
   A second pre-dispatch contract bug was fixed in
   `experiments/build_sjkl_residual.py`: runtime applies the residual to
   JointFrameGenerator pair slot `0` (`fake1`), but the builder had encoded
   residuals against `gt_pairs[:, 1]`. The builder now defaults to
   `--target-slot 0`, records `target_slot` and `runtime_target` in the
   manifest, and rejects other slots until the runtime supports them.
3. Renderer self-compression should preserve the QZS3/QP1 basin. QBF1-v2 and
   mixed-block attempts are not currently byte/score competitive enough for a
   blind dispatch; use QZS3/RP2 packer polish or a scorer-aware block policy,
   not generic renderer recompression.

Current operating decision: do not spend exact evals on generic
self-compression. Dispatch only archives that either keep the C067 basin and
improve pose/rate by construction, or test one concrete mask-geometry
hypothesis with clean custody.

## Archive Bit Accounting Artifacts

The current A++ C067 archive and the PMG atomtop4068 diagnostic archive now
have reproducible bit/stream accounting profiles:

```text
C067 profile:
  experiments/results/c067_self_compression_profile_20260502/archive_byte_accounting.json
  experiments/results/c067_self_compression_profile_20260502/archive_byte_accounting.md
  experiments/results/c067_self_compression_profile_20260502/archive_byte_accounting.png

PMG atomtop4068 profile:
  experiments/results/pmg_hotspot_candidate_c067_atomtop4068_20260502/archive_byte_accounting.json
  experiments/results/pmg_hotspot_candidate_c067_atomtop4068_20260502/archive_byte_accounting.md
  experiments/results/pmg_hotspot_candidate_c067_atomtop4068_20260502/archive_byte_accounting.png
```

C067 exact frontier accounting:

```text
archive_bytes=276214
archive_sha256=226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a
score=0.31561703078448233
distortion_score=0.13169746590679487
bytes_to_remove_at_unchanged_distortion=23454
buffered_target_archive_bytes=252759

masks.mkv:           219472 bytes, entropy 7.998591976459 b/B, nested savings 0
renderer.bin:         55965 bytes, entropy 7.993832581821 b/B, nested savings 0
optimized_poses.bin:    677 bytes, entropy 7.673436926071 b/B, nested savings 0
```

This confirms the current public-floor blob is already entropy-tight under
generic recompression. Sub-0.300 by bytes alone requires removing only
`10.6866%` of the mask stream, but `41.9083%` of the renderer stream or
`34.644x` the pose stream. Therefore the highest-value byte work is not outer
zip or generic nested recompression; it is representation change: mask grammar,
learned/multimask topology, or scorer-aware residuals.

PMG atomtop4068 accounting:

```text
archive_bytes=195762
archive_sha256=2567dc04185cf20775f1f6c088395aa8df9e4484daa8b25001e940d62a5d6497

masks.cmg3:         134352 bytes, entropy 7.997925889544 b/B, nested savings 200
renderer.bin:        59288 bytes, entropy 7.861381999313 b/B, nested savings 3323
optimized_poses.bin:  1140 bytes, entropy 5.229017915936 b/B, nested savings 463
payload_lzma_probe_delta=-2989
```

The PMG atom archive is smaller but not yet score evidence. Its residual
container still exposes about `2990` bytes of generic LZMA probe savings; that
is only worth pursuing if the L40S diagnostic proves the PMG atom geometry does
not collapse. Otherwise the byte opportunity is irrelevant because the scorer
basin is wrong.

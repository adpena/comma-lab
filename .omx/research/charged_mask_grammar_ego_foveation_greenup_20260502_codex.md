# Charged Mask Grammar And Ego-Foveation Greenup - 2026-05-02

Experiment: #5, contest-faithful charged mask grammar / ego-motion /
foveation stream.

Evidence grade: `design_review` + `external_forensics` + `local_exact_synthesis`.
Score claim: `false`.
Remote jobs launched: `false`.

This ledger is implementation planning only. It does not promote, rank, kill,
or retire any lane without exact CUDA archive evidence on the exact archive
bytes.

## Evidence Boundary

Strict score truth remains:

```text
archive.zip -> inflate.sh -> upstream/evaluate.py
```

preferably through:

```text
experiments/contest_auth_eval.py --device cuda
```

All score-affecting state must be charged in `archive.zip`: mask data,
learned selector weights, frequency tables, foveation parameters, grammar
tables, reconstruction constants, pose residuals, PR65 atoms, and any
per-video learned codebooks. Generic source code may implement algorithms, but
it must not hide task-specific payload bits in Python literals, generated
lookup tables, or sidecar files.

Current exact internal anchor:

```text
C-058 QZS3/QP1 active-subspace fixed-slice
score              0.3157555307844823
archive bytes      276422
PoseNet            0.00049637
SegNet             0.00061244
archive SHA-256    5145fb57be574b85639856d239420ffa35e605e32664f93e06753b120b21633f
evidence           experiments/results/lightning_batch/exact_eval_line_search_qzs3_qp1_active_fix2_t4_20260502T0250Z/contest_auth_eval.adjudicated.json
component trace    experiments/results/lightning_batch/exact_eval_line_search_qzs3_qp1_active_fix2_t4_20260502T0250Z/component_trace.json
```

Active public-floor mask anatomy:

```text
PR67 / PR63 mask segment:
  compressed bytes  219472
  raw OBU bytes      223385
  raw SHA-256        a5c2b89c110d75220cd09b2f27f2e92844626ae7ed0d2c797290dcf43c7068eb
  evidence           experiments/results/top_submission_reverse_engineering_20260502T0206Z/archive_anatomy.json

Rate cost of 219472 bytes:
  25 * 219472 / 37545489 = 0.146137396160
```

This 219 KB mask stream is now the dominant single rate lever in the
public-floor basin. Replacing it with a strict 57 KB charged payload would save
about `0.108030288272` score before distortion; replacing it with a strict
42.8 KB payload would save about `0.117646623273`.

PR70 external forensic boundary:

```text
archive bytes       57329
member              m
member bytes        57230
archive SHA-256     d5046b9b64c0982adb1bd8edf35f25bd3eb7fa0180f6744ce3bbd8c139abb142
member SHA-256      434f301d888fdfa0f1b3c3ccd362db5655dc53e0d92f624abd82bcfe42f631dc
classification      malformed_zip_python_zipfile_rejects_local_header_empty_but_unzip_extracts_m
evidence            reports/raw/leaderboard_intel_20260502/pr70_manual_extract/manual_extract_provenance.json
inflate.py bytes    1299244
```

PR70 is useful reverse-engineering evidence that a mask grammar can be much
smaller than the AV1 OBU stream, but it is not a strict target. It relies on a
malformed ZIP local header and appears to reconstruct mask state from large
submission-code constants. Our strict path must move all such constants into
the archive, use a normal ZIP, and pass the repo's central/local filename
integrity guard in `experiments/contest_auth_eval.py`.

## Grand Council Review

Shannon:

- The mask stream is worth attacking because one byte costs
  `25 / 37545489 = 6.65859e-7` score. A 1 KB grammar table costs about
  `0.000681839568`; this is cheap if it preserves even a tiny fraction of
  SegNet/PoseNet distance.
- The correct objective is not "smallest mask payload". It is smallest total
  archive subject to component gates after the renderer, pose codec, PR65
  postprocess atoms, and exact scorer path all interact.

Boyd / Dykstra:

- Treat every grammar element as a charged atom with rate, SegNet, PoseNet,
  runtime, and interaction constraints.
- The projection loop is:

```text
A_{k+1} = P_exact_eval P_runtime P_compliance P_pose P_seg P_rate(A_k + atoms_k)
```

- This is a workflow discipline, not a proof that nonconvex atoms compose.
  Stacked exact archives are mandatory.

Fridrich / Mallat:

- Boundary and texture are different variables. SegNet is sensitive to class
  boundaries, thin lane markings, and object silhouettes; PoseNet can also be
  sensitive to apparent ego-motion texture and horizon geometry.
- Connected components and boundary splines should be first-class atoms:
  fill interiors cheaply, spend bytes on boundaries, temporal births/deaths,
  and hard-pair residual tiles.

comma / openpilot geometry:

- Use camera priors as proposal features: native `1164x874`, `fx=fy=910`,
  native principal point near `(582,437)`, scorer-space FoE/VP near
  `(256,174)`.
- `src/tac/openpilot_seeding.py` and `src/tac/openpilot_features.py` already
  enforce compress-time-only supercombo usage. Supercombo may rank atoms or
  initialize ego-motion, but no supercombo model, hidden outputs, or uncharged
  feature tables may run at inflate time.

Senior engineer:

- The biggest implementation risk is accidental public-exploit cloning:
  malformed ZIP behavior, uncharged Python literals, or decoder constants in
  source. Build the strict decoder first, with charged payload accounting and
  SHA checks, before optimizing the grammar.
- The second biggest risk is optimizing a mask stream against a proxy while
  silently leaving the public-floor renderer/pose basin. Every lossy grammar
  must be evaluated as a complete archive, not as a mask PSNR or local visual
  metric.

## Full-Pipeline Compression Opportunities

### Submission Path And Runtime

1. Replace `mask.obu.br -> PyAV decode -> class tensor` with
   `mask_grammar.bin -> class tensor` when the grammar is active. Avoiding
   OBU reconstruction removes PyAV/ffmpeg/NVDEC variability from the mask path
   and reduces inflate-time risk. It does not by itself save archive bytes, but
   it makes exact runtime closure simpler.
2. Keep one strict `p` member or another single top-level strict member. Do not
   rely on malformed local headers, duplicate names, empty local filenames, or
   central/local parser divergence. PR70's malformed ZIP behavior is external
   only.
3. Split source code from payload state:
   - allowed in source: generic range coder, connected-component decoder,
     spline rasterizer, foveation warp implementation, deterministic renderer
     glue.
   - charged in archive: all trained weights, frequency tables, class
     dictionaries, frame headers, grammar programs, atom selections, foveation
     params, openpilot-derived compressed summaries, and PR65/QP2 residuals.
4. Emit one provenance record per archive with:
   - member names, sizes, SHA-256, and central/local ZIP validation;
   - mask decoded SHA and shape;
   - renderer and pose payload SHA;
   - grammar schema and version;
   - exact eval command, device, sample count, and recomputed score.
5. Keep the decoder deterministic and bounded: no network, no scorer load, no
   host path reads, no random generation without a charged seed and charged
   deterministic generator contract.

### Code Simplification

The strict submission path should avoid copying PR70's giant branchy inflate
surface into production. A maintainable path is:

```text
unpack p
decode renderer weights (QZS3/QZS4 wire)
decode pose (QP1/QP2)
decode mask_grammar.bin to (600,384,512) uint8 class ids
render pairs
apply optional counted qpost atoms
write raw RGB
```

This keeps source small, reviewable, and less likely to hide payload. If a
decoder table is static because it is universal algorithm design, it must be
documented as universal. If it was chosen from the contest video, it is
payload.

## Mask Grammar Design

### CMG1: Connected-Component / Boundary / Temporal Grammar

Wire contract:

```text
magic              CMG1
shape              uint16 frames=600, height=384, width=512
class_count         uint8 = 5
global_templates    charged stream
track_table         charged stream
boundary_programs   charged stream
residual_tiles      charged stream
checksums           decoded mask SHA-256 and source mask SHA-256
```

Atom families:

- `global_template`: long-lived road/sky/lane/background class templates.
- `component_track`: class id, track id, birth/death pair range, bbox, affine
  or projective motion, fill rule.
- `boundary_spline`: class-pair boundary, control points, quantized normal
  offsets, temporal DCT coefficients.
- `scanline_span`: class-labeled row spans for components whose shape is too
  irregular for splines.
- `residual_tile`: exact small tile override for hard-pair boundaries,
  vehicles, lane endpoints, or temporal transitions.
- `header_enum`: arithmetic/enumerative coding for frame sizes, class counts,
  component counts, and OBU-compatible header facts when reconstructing OBU
  remains useful.

The first milestone should be lossless or bit-identical class-mask
reconstruction of the PR67/PR63 mask stream. A lossy grammar is only allowed
after the bit-identical baseline proves the decoder and byte accounting.

### EGO1: Ego-Motion And Camera Priors

Compress-time proposal features:

- supercombo pose/path/lane features from `src/tac/openpilot_seeding.py` and
  `src/tac/openpilot_features.py`;
- luma motion and lane-mark log-zoom from
  `experiments/results/contest_video_reverse_engineering_20260501/`;
- exact component traces from C-058 and public-floor traces;
- native camera intrinsics and scorer FoE/VP priors.

Inflate-time state:

- no supercombo;
- no hidden openpilot model;
- no uncharged feature table;
- only charged grammar params, component tracks, foveation params, or selected
  atom payloads.

Use ego-motion to rank and parameterize atoms:

- horizon band preservation;
- radial/zoom deformation around FoE;
- vehicle/lane transition windows;
- temporal component velocity;
- endpoint and lane-mark residual priorities.

### FOV1: Directional Anisotropic Foveation

Existing repo support:

- `src/tac/hyperbolic_foveation.py` implements isotropic per-frame
  `alpha, R, p, ox, oy` and `foveation_params.bin`.
- A full 1200-frame float32 payload is `24016` bytes by test contract, costing
  about `0.015991268618` rate score before any benefit.
- A one-scalar-per-frame payload would be about `2400` bytes before headers,
  costing about `0.001598061487`.

The next version should not store five fp32 values per frame unless exact
response proves that much freedom pays. Use low-rank directional forms:

```text
center c          fixed FoE + small charged offset
direction u       radial / tangential / horizon / lane-line basis
alpha_r(t)        temporal DCT or spline coefficients
alpha_t(t)        optional tangential coefficient
R_r, R_t          global or low-rank radii
p                 fixed or small alphabet
trust_region      hard bounds to avoid folds and off-manifold warps
```

The anisotropic map should preserve identity at zero and be constrained to
small moves first. Directional proposals:

- radial contraction/expansion around FoE for velocity/log-zoom alignment;
- tangential correction near horizon for yaw-like residuals;
- vertical/horizon band preservation around scorer y `[155,195]`;
- lane-line endpoint foveation for hard pairs;
- asymmetric radius by motion direction from QP1/QP2 col0 and openpilot
  proposal features.

Foveation is only useful if it reduces total score after renderer and pose
interactions. It can damage SegNet by moving boundaries and damage PoseNet by
off-manifold image geometry, so start with exact component-response cliffs and
then stack only accepted atoms.

## Learned Selector, RL, And Multipass Search

Arbitrary hand heuristics are useful only for bootstrapping. The promotable
selector should be a compress-time proposal system that emits a charged atom
set and a deterministic archive. Two modes are acceptable:

1. Learned selector as proposal only:
   - train a differentiable or Gumbel/straight-through selector over atom
     features;
   - use exact traces, local differentiable renderer/scorer feedback, and
     byte costs as training labels;
   - do not ship selector weights;
   - ship only the selected atom payloads and provenance.
2. Learned selector at inflate:
   - allowed only if all weights, tables, and inputs are charged;
   - usually lower EV because selector weights compete with mask bytes.

Suggested atom feature vector:

```text
pair index
class id
component id
bbox / area / perimeter / Euler stats
boundary length
distance to FoE and horizon
temporal velocity / acceleration
luma-motion score
lane-mark log-zoom score
openpilot/supercombo proposal score
C-058 per-pair PoseNet and SegNet trace
estimated compressed bytes
interaction tags: QP2, qpost, foveation, renderer block
```

RL / multipass ablations:

- Start with contextual bandit, Bayesian optimization, CEM, or beam search.
  Full PPO is not justified until the action space is larger and per-step cost
  is much cheaper.
- Use multipass accept/reject over complete archives:
  `proposal -> deterministic build -> byte screen -> H100 diagnostic exact eval
  -> T4 promotion only if material`.
- A bandit or RL policy is never score evidence. The accepted archive is the
  evidence.

## Lagrangian Water-Fill

For each atom `a` on anchor archive `A`:

```text
accept when:
E[score_drop(a | A)]
  > 25 * bytes(a | A) / 37545489
    + uncertainty_penalty(a)
    + interaction_penalty(a | A)
```

Atom ledger fields:

```text
atom_id
family
payload_member
payload_bytes
payload_sha256
decoded_mask_sha256_after_atom
predicted_delta_seg
predicted_delta_pose
predicted_delta_rate
evidence_grade
feature_vector_sha256
selector_version
conflicts
stack_dependencies
exact_eval_json_if_available
```

Water-fill order:

1. Build a bit-identical charged grammar baseline against the PR67 mask SHA.
2. Measure byte savings without component risk.
3. Add lossy atom drops or simplifications one family at a time.
4. Use exact component traces to compute per-pair marginal loss.
5. Stack only atom sets that pass complete-archive exact eval.

## Stackability

### QZS4

`src/tac/quantizr_qzs3_codec.py` already has
`encode_qzs4_block_search_state_dict(...)`. It is a packer policy that still
emits `QZS3` wire format. It should stack with mask grammar because it changes
renderer bytes, not mask semantics. The interaction risk is indirect: a lossy
mask grammar can move the best renderer quantization point.

Stack gate:

- build C-058 + QZS4 repack + identical mask grammar;
- compare exact score and bytes;
- only then try lossy grammar atoms.

### QP2

`src/tac/qp1_pose_codec.py` contains `QP2_MAGIC = b"PVR1"` and
`encode_qp2_residual_topk(...)`. QP2 adds charged non-velocity pose residual
atoms over the QP1 scalar manifold.

Stack gate:

- use QP2 residuals for pairs/classes where mask grammar or foveation harms
  PoseNet;
- solve mask atoms and QP2 residuals jointly, not sequentially, because a mask
  simplification can shift the pose optimum.

### PR65 qpost Atoms

`experiments/build_qzs3_postprocess_candidate.py` and
`submissions/robust_current/apply_qzs3_postprocess.py` provide counted PR65
postprocess streams. Current diagnostic H100 evidence shows:

```text
bias-only qpost:
  bytes    276776
  score    0.31571526640225916
  pose     0.00049575
  seg      0.00061012
  evidence experiments/results/qzs3_postprocess_ablation_r13_20260502T0158Z/h100_exact_summary.json
```

This is diagnostic, not T4 promotion. It is still important for mask grammar:
small postprocess bias/region atoms may repair output-space errors caused by
mask simplification more cheaply than re-adding mask residual tiles.

Stack gate:

- compare grammar-only, qpost-only, and grammar+qpost exact archives;
- charge `qpost.bin` bytes and include stream-level provenance;
- no additive claim until the stacked archive passes exact CUDA.

## Exact Eval Gates

No remote work was launched in this review. Future dispatch must first claim
the lane with `tools/claim_lane_dispatch.py claim ...`.

Local build gates:

1. `py_compile` for any new decoder/builder.
2. Unit tests for:
   - safe ZIP member names;
   - grammar payload bounds and checksums;
   - decoded mask shape `(600,384,512)`;
   - decoded mask SHA for bit-identical mode;
   - no scorer imports in inflate path;
   - deterministic archive bytes and SHA.
3. Archive whitelist parity:
   - if adding `mask_grammar.bin`, `.cmg1`, `.fov1`, or `.egofov`, update
     `experiments/contest_auth_eval.py` and
     `experiments/canonical_local_auth_eval_smoke.py` together;
   - add tests so AMR1-style allowlist drift cannot recur.
4. Local smoke:
   - extract strict archive;
   - validate central/local ZIP filenames;
   - inflate without scorer;
   - verify raw output byte count.

Score gates:

1. H100/A100/L40S exact CUDA diagnostic for fast screening only.
2. T4/equivalent exact CUDA for promotion.
3. Required JSON fields:
   - exact archive path, bytes, SHA;
   - component distances;
   - recomputed score;
   - sample count `600`;
   - device `cuda`;
   - manifest and provenance;
   - decoded mask SHA and grammar payload SHA.
4. Component trace after any candidate that beats or approaches C-058.
5. Claim-matrix row only after exact T4/equivalent adjudication.

## Failure Modes

- Uncharged task-specific constants in source code.
- Malformed ZIP, empty local filename, duplicate member, or central/local
  filename mismatch.
- Decoder silently falls back to CPU/MPS, host paths, sidecars, or network.
- Inflate path imports PoseNet, SegNet, upstream scorer modules, or learned
  selector sidecars.
- Grammar produces the right visual mask but wrong `(600,384,512)` class
  tensor, parity, pair ordering, or odd/even frame contract.
- Connected-component simplification erodes thin lane markings or object
  boundaries and raises SegNet.
- Foveation folds or moves the horizon/lane geometry off PoseNet's manifold.
- Ego-motion prior overfits compress-time proxies and fails exact CUDA.
- Learned selector optimizes byte-only or proxy-only reward and selects atoms
  that fail component gates.
- QZS4/QP2/qpost atoms interact antagonistically with mask grammar.
- Runtime exceeds the 30 minute budget or depends on fragile PyAV/ffmpeg
  behavior.
- Diagnostic H100 improvement is treated as promotion without T4/equivalent
  evidence.

## Top 3 Build/Eval Candidates

## 2026-05-02 Codex Local Scaffold Greenup: Trace-Weighted Allocation

Evidence grade: `empirical_allocation_only`.
Score claim: `false`.
Remote jobs launched: `false`.

Implemented a local-only extension to
`experiments/plan_charged_mask_grammar_atoms.py` that consumes an optional
component trace as a compress-time prior and emits
`trace_weighted_allocation` with:

- deterministic atom allocation rows ranked by estimated charged bytes,
  component-trace opportunity, mask family, and ego/VP/horizon region priors;
- explicit side-info accounting for grammar headers, codebooks, atom-index
  tables, and residual model tables;
- candidate specs for charged-byte budgets, preserving
  `score_claim=false`, `promotion_eligible=false`, and
  `external_sidecars_allowed=false`;
- archive-byte estimates against the current C-063 byte frontier without
  claiming component preservation or CUDA score truth.

This is intentionally a byte-screen/allocation artifact, not a cloud dispatch.
It is higher-EV than naive mask CRF/RPK1 screens because it spends bytes only
where component traces and geometry priors say PoseNet/SegNet are likely
fragile: hard pairs, boundaries, horizon/vanishing-point regions, and
near-field ego-motion corridors. A future builder must still lower selected
atoms into charged CMG payload bytes and run exact CUDA auth eval on the exact
archive before any score claim.

### Candidate 1: `CMG1_STRICT_PR67_MASK_REPRO`

Goal: strict, charged, bit-identical replacement for the 219472-byte PR67 mask
segment.

Implementation:

1. Add a read-only builder named like
   `experiments/build_charged_mask_grammar_pr67.py`.
2. Input: PR67/PR63 mask OBU raw bytes with SHA
   `a5c2b89c110d75220cd09b2f27f2e92844626ae7ed0d2c797290dcf43c7068eb`.
3. Output: `mask_grammar.bin` or `mask.cmg1` with all grammar tables,
   frequency tables, enum bits, residual data, and checksums charged.
4. Decoder emits either:
   - bit-identical raw OBU bytes, then existing class decode; or
   - directly the identical `(600,384,512)` class tensor plus SHA proof.
5. Build C-058 with mask grammar replacing the OBU mask segment, keeping
   renderer and QP1 pose fixed.

Promotion logic:

- If decoded mask is bit-identical, component risk should be zero except for
  decoder path bugs.
- Exact eval still required because the submission path changed.
- T4 promotion only if strict archive bytes beat C-058 materially.

Senior engineer note:

- Do this before lossy connected components. It proves accounting, ZIP
  integrity, decoder determinism, and payload closure.

### Candidate 2: `CMG1_COMPONENT_BOUNDARY_WATERFILL`

Goal: lossy grammar that spends bytes only where exact components care.

Implementation:

1. Decode PR67 masks into class tensors.
2. Extract connected components, boundaries, scanline spans, temporal tracks,
   and residual tiles.
3. Build an atom table with compressed byte cost per atom.
4. Join features with C-058 component trace, luma-motion, lane-mark log-zoom,
   and openpilot proposal features.
5. Train or fit a compress-time learned selector / bandit to propose atom sets.
6. Build deterministic archives from selected charged atoms only.

Initial atom families:

- preserve all lane-mark boundaries;
- preserve horizon-band transitions;
- simplify large interiors;
- use temporal tracks for long-lived road/sky/background components;
- use residual tiles on top hard PoseNet/SegNet pairs.

Promotion logic:

- First exact diagnostic must compare grammar-only to C-058.
- Then run grammar + QP2 residuals if PoseNet loss appears.
- Then run grammar + qpost bias/region if SegNet/RGB output repair is cheaper
  than mask residuals.

Senior engineer note:

- The selector can be learned, but the emitted archive must be a plain charged
  atom set. Do not make the selector an inflate-time dependency unless its
  weights are demonstrably worth the bytes.

### Candidate 3: `EGO_FOV1_QZS4_QP2_PR65_STACK`

Goal: full-pipeline candidate that treats mask, pose, renderer, and output
postprocess as one charged R(D) problem.

Implementation:

1. Anchor on C-058.
2. Apply QZS4 renderer block-search repack while preserving QZS3 wire format.
3. Add QP2 residual top-K atoms only for pairs where mask/foveation changes
   increase PoseNet.
4. Add low-rank `FOV1` anisotropic foveation atoms:
   - fixed FoE center plus tiny charged offsets;
   - radial/tangential/horizon directions;
   - temporal DCT coefficients;
   - strict invertibility and identity trust-region tests.
5. Add PR65 qpost streams only where exact diagnostic deltas beat charged
   bytes.
6. Build complete archive and exact-eval as a stack.

Promotion logic:

- No component archive can promote the stack by itself.
- Require stacked exact CUDA JSON with all member SHAs and a component trace.
- Use H100 for fast diagnostic screens, then T4/equivalent for promotion.

Senior engineer note:

- This is the likely sub-0.30 route if Candidate 1 proves the mask stream is
  compressible under strict accounting. It also has the largest interaction
  surface, so it should not be the first implementation.

## Recursive Senior Engineer Greenup

Pass 1 - compliance:

- Reject PR70 as a strict target because of malformed ZIP and uncharged
  reconstruction constants.
- Require every per-video table and learned selector output to be archive
  bytes.
- Require archive validator and local smoke whitelist parity for any new
  suffix.

Pass 2 - engineering:

- Start with bit-identical mask reconstruction. It isolates decoder/accounting
  bugs from scientific lossy-grammar risk.
- Keep the decoder small and deterministic. A large branchy source file is a
  review hazard and an easy place to hide payload.
- Prefer direct class tensor decode over OBU reconstruction if it reduces
  runtime dependencies and preserves scorer input parity.

Pass 3 - science:

- Do not rank mask grammars by visual quality, PSNR, byte count, or CPU/MPS
  output. Rank only complete archives by exact CUDA components.
- Use component traces to build marginal atom tables.
- Treat foveation and ego-motion as proposal priors until exact archive
  evidence proves they help.

Pass 4 - stacking:

- QZS4, QP2, PR65 qpost, and mask grammar are stackable in principle because
  they touch different contracts. They are not additively composable until a
  stacked exact archive exists.
- The water-fill solver must include interaction penalties, especially
  `mask grammar x QP2` and `mask grammar x qpost`.

Final greenup verdict:

Proceed in this order:

1. `CMG1_STRICT_PR67_MASK_REPRO`.
2. `CMG1_COMPONENT_BOUNDARY_WATERFILL`.
3. `EGO_FOV1_QZS4_QP2_PR65_STACK`.

The lane is high EV only under strict accounting. If the first candidate cannot
beat 219472 charged bytes after moving PR70-style constants into the archive,
do not continue copying public exploit mechanics. Pivot to component-boundary
water-fill and learned selector atoms, where the byte savings are physically
explained and stackable.

## Supersession Note - 2026-05-02T04:25Z

Several design sections above name C-058 because that was the anchor when this
review started. The current strict internal A++ anchor is C-059:

```text
score            0.3157055307844823
archive bytes    276347
archive SHA-256  cf44aa7fdb13b9ab6236aefde1f5f58e915d7dfa99128246235443841396c6ab
PoseNet          0.00049637
SegNet           0.00061244
evidence         experiments/results/lightning_batch/exact_eval_qzs3_b32_maskfirst_qp1_fix1_t4_20260502T0331Z/contest_auth_eval.adjudicated.json
component trace  experiments/results/lightning_batch/exact_eval_qzs3_b32_maskfirst_qp1_fix1_t4_20260502T0331Z/component_trace.json
```

Use C-059 for new build/eval candidates. Treat C-058 references above as
historical design context unless an exact artifact path explicitly requires the
C-058 archive.

The QZS4/global-block subcandidate has also been exact-screened as a scoped
negative on this basin: `qzs4_maskfirst_qp1` saved bytes but collapsed PoseNet
on H100 diagnostic eval. New renderer packing work should therefore target
mixed/local block allocation or learned quantization with component-response
guards, not another global scalar block sweep.

## Protected Reencode Branch - 2026-05-02

Current frontier supersedes the C-059 note above: C-063 is the active exact
frontier at score `0.3156230307844823`, `276223` archive bytes, archive SHA-256
`83615afd130afa08e972e4a02476612397bffea53327caf3591891f8317aa52d`.

The local naive mask AV1 sweep under
`experiments/results/c063_mask_av1_reencode_sweep_20260502` found that CRF56
saves `71840` bytes with about `0.51%` decoded class disagreement. That is the
right next branch because the byte signal is large enough to pay for a focused
geometry-protection pass, but a uniform CRF reencode is not a scientific mask
claim: the disagreement is likely concentrated around boundaries, hard-pair
frames, ego/foveal/horizon regions, and class-specific scorer-critical pixels.

Implemented local builder scaffold:
`experiments/build_protected_mask_reencode_candidate.py`. It builds only
deterministic `archive.zip` candidates by replacing the mask member from an
existing archive, records the protection policy and ffmpeg arguments in an
adjacent manifest, and marks `score_claim=false`, `promotion_eligible=false`.
Supported policy axes are hard-pair/hard-frame lists, boundary dilation,
horizon bands, foveal boxes, ego boxes, and class-specific protection. The
branch remains empirical/non-promotable until exact CUDA auth eval adjudicates
the exact archive bytes.

## Charged AMR1 Legacy Repair Byte Screen - 2026-05-02T08:10Z

Evidence grade: `empirical_byte_screen_non_score`. No cloud job was dispatched
and no score claim is made. Exact CUDA auth eval on the exact archive bytes is
still required before ranking or promotion.

Implemented and tested a runtime-supported charged residual branch in
`experiments/build_charged_mask_grammar_candidate.py`:

- source runtime custody:
  `experiments/results/c063_mask_av1_reencode_sweep_20260502/source/runtime_members.zip`
- lossy mask base:
  `experiments/results/c063_mask_av1_reencode_sweep_20260502/runtime/masks.crf52.mkv`
- repair member: `alpha4_residual_repair.amr1.zlib`
- runtime hook: `submissions/robust_current/inflate_renderer.py` now applies a
  charged AMR1 residual after legacy `masks.mkv` decode, while the grayscale
  wrapper continues to own its existing AMR1 path to avoid double-application.

Byte screens produced:

1. `pose_top32_boundary_horizon_fovea_ego_crf52_zlib`
   - Rejected as a contract mismatch for this C-063 stream. Treating the
     legacy CRF52 mask as `grayscale.mkv` caused decoded-class disagreement to
     jump from the known `0.4104580349%` legacy mismatch to `75.6742638482%`.
2. `legacy_pose_top32_boundary_horizon_fovea_ego_crf52_zlib`
   - Contract-correct but byte-negative. It repaired `477295 / 484196`
     residual pixels and produced a `1172456` byte standard ZIP.
3. `legacy_pose_top32_boundary_horizon_fovea_ego_crf52_budget4096_zlib`
   - Budgeted charged repair candidate.
   - Standard ZIP:
     `experiments/results/c063_charged_mask_grammar_20260502_codex/legacy_pose_top32_boundary_horizon_fovea_ego_crf52_budget4096_zlib/archive.zip`
     is `279273` bytes, SHA-256
     `20ebde16c73d962503bffd62faa38e40befda2b0a8abfeb690eeb716b134e528`.
   - Packed RPK1 archive:
     `experiments/results/c063_charged_mask_grammar_20260502_codex/legacy_pose_top32_boundary_horizon_fovea_ego_crf52_budget4096_zlib/packed_rpk1/archive.zip`
     is `274112` bytes, SHA-256
     `4d0cf75a3ccc6e72daf6fd903ddc4b942f243b4788d86b7b8e927aa9446a91b3`.
   - Charged repair payload is `9964` bytes, SHA-256
     `0b0879187940c0ed62c7975bdfac1cdcadc657ef4e0e5fd1b39f5086f81fb2a4`,
     selecting `4095 / 484196` residual pixels under the top-32 PoseNet-pair
     plus boundary/horizon/foveal/ego policy.
   - Local unpack smoke with `.venv/bin/python
     submissions/robust_current/unpack_renderer_payload.py` recovered
     `renderer.bin`, `masks.mkv`, `optimized_poses.bin`, and
     `alpha4_residual_repair.amr1.zlib` from the packed archive.

This packed candidate is byte-plausible relative to C-063 (`274112` bytes vs
`276223` bytes), but it is still empirical because exact CUDA components are
unknown. The broad repair screen also shows that post-decode residual repair
must be atom-ranked by expected PoseNet benefit per charged byte; region-wide
repair is too expensive.
## 2026-05-02 - Protected/Foveated AV1 Mask Reencode Exact Diagnostics

Evidence grade: `A` diagnostic CUDA on H100 NVL, not T4 promotion.

Two protected C-063 mask reencode candidates were exact-evaluated while the
CRF52 pose-regeneration isolation ran on a separate H100. The intent was to
test whether hard-pair, boundary, horizon, foveal, and ego-region protection
could rescue the PoseNet collapse seen in naive CRF52/CRF56 mask AV1
reencodes.

Results:

- `protect_boundary1_horizon_pose_top16_crf56_rawzip/final_archive.zip`
  scored `1.977293373886536`, bytes `207928`, SHA
  `68d0d3080aefe2e8d376ddd7074270d713b351efecf8bd98425583a76c5912bf`,
  PoseNet `0.23229894`, SegNet `0.00314707`.
- `protect_boundary2_horizon_pose_top32_crf60_rawzip/final_archive.zip`
  scored `2.590810215529544`, bytes `158469`, SHA
  `62acfea9796cec57190c233a81edefc382e2151721479a83fddc1759415fd0c4`,
  PoseNet `0.42019761`, SegNet `0.0043542`.

Conclusion: pre-encode protected-pixel copying is not enough. Even when the
policy protects hard pairs and domain regions, AV1 still perturbs protected
areas after encode and PoseNet collapses. Do not promote this implementation
to T4. The next mask-stream work should use a decoder-consumed charged repair
or grammar payload: lossy base plus exact residual/atom reconstruction inside
the archive, or a learned mask representation whose train/inflate
distribution is identical.

Artifacts:

- `experiments/results/vast_harvest/c063_protected_mask_h100nvl_seq_20260502/archive_eval_c063_protected_boundary1_horizon_pose_top16_crf56_h100_20260502/contest_auth_eval.json`
- `experiments/results/vast_harvest/c063_protected_mask_h100nvl_seq_20260502/archive_eval_c063_protected_boundary2_horizon_pose_top32_crf60_h100_20260502/contest_auth_eval.json`

## End Addendum - 2026-05-02T08:10Z

See `Charged AMR1 Legacy Repair Byte Screen - 2026-05-02T08:10Z` above for
the local charged-repair byte screen. The only packed candidate under the
C-063 byte count is empirical/non-score:
`experiments/results/c063_charged_mask_grammar_20260502_codex/legacy_pose_top32_boundary_horizon_fovea_ego_crf52_budget4096_zlib/packed_rpk1/archive.zip`
at `274112` bytes, SHA-256
`4d0cf75a3ccc6e72daf6fd903ddc4b942f243b4788d86b7b8e927aa9446a91b3`.

## 2026-05-02 - RPK1 Exact Diagnostic No-Op Classification

Evidence grade: `A-negative harness`, not method evidence.

The first H100 NVL exact diagnostic for the packed RPK1 AMR1 candidate
returned:

- archive bytes: `274112`
- archive SHA-256:
  `4d0cf75a3ccc6e72daf6fd903ddc4b942f243b4788d86b7b8e927aa9446a91b3`
- recomputed score: `2.324934305635095`
- PoseNet: `0.32767636`
- SegNet: `0.00332231`

This is numerically identical to the prior CRF52 mask-collapse run, and the
inflate log does not contain `Applied Alpha residual repair`. Root cause: the
default `inflate_renderer()` path loaded masks directly through
`_load_masks_from_archive`, while Hubble's AMR1 hook lived in
`_load_renderer_and_masks` and was therefore bypassed by the canonical
renderer inflate path.

Permanent fix:

- `submissions/robust_current/inflate_renderer.py` now routes legacy archive
  mask loading through `_load_archive_masks_with_optional_amr1_repair` in the
  default renderer path, not only helper/TTO paths.
- `src/tac/tests/test_build_charged_mask_grammar_candidate.py` now includes a
  regression test that stubs `_load_masks_from_archive` and proves the archive
  mask loader applies AMR1 repair.
- `AGENTS.md` records that a repair member without an exact-eval apply log is
  a no-op harness bug.
- Focused verification: `24 passed` for charged-mask candidate/runtime tests.

The exact artifact is preserved at:

`experiments/results/vast_harvest/c063_charged_mask_rpk1_h100nvl_noop_20260502/`

The identical archive bytes were relaunched as
`c063_charged_mask_rpk1_h100nvl_fix1_20260502T0820Z` with
`submissions/robust_current/inflate_renderer.py` SHA-256
`45ea0940be74d716ea5650f2d545aa1ad048f8478ded19fb5fb2857492dfc581`.

## 2026-05-02 - RPK1 Fix1 Half-Frame Metadata Failure

Evidence grade: `A-negative harness`, not method evidence.

`fix1` proved the AMR1 payload was finally consumed:

`Applied Alpha residual repair alpha4_residual_repair.amr1.zlib: 26,941 raw AMR1 bytes`

It then generated only 600 frames and failed strict auth eval raw-size
validation:

`WRONG-SIZE .raw file(s): 0.raw=1831204800B (expected 3662409600B)`

Root cause: `_apply_amr1_repair` cloned the half-frame mask tensor and dropped
the `_half_frame_only` attribute set by the legacy mask decoder. That changed
the renderer path from deferred 600->1200 reconstruction to direct 600-frame
generation.

Permanent fix:

- `_apply_amr1_repair` now preserves `_half_frame_only` on repaired tensors.
- The charged-mask runtime test now stubs the archive mask loader to return a
  half-frame tensor and asserts the repaired result still carries the marker.
- Focused verification: `24 passed`.

The failed artifact is preserved at:

`experiments/results/vast_harvest/c063_charged_mask_rpk1_h100nvl_fix1_wrong_size_20260502/`

The identical archive bytes were relaunched as
`c063_charged_mask_rpk1_h100nvl_fix2_20260502T0824Z` with
`submissions/robust_current/inflate_renderer.py` SHA-256
`697d1d4bdbfa224506ab88309163f51a0752c356fe4b8848c6c42985d5afa1da`.

## 2026-05-02 - RPK1 Fix2 Clean Exact Diagnostic

Evidence grade: `A-negative` H100 CUDA diagnostic, not T4 promotion.

`fix2` is the first clean exact diagnostic for the packed RPK1/AMR1 candidate:

- archive bytes: `274112`
- archive SHA-256:
  `4d0cf75a3ccc6e72daf6fd903ddc4b942f243b4788d86b7b8e927aa9446a91b3`
- recomputed score: `2.3218695649009`
- PoseNet: `0.32702041`
- SegNet: `0.00330979`
- exact path: `archive.zip -> inflate.sh -> upstream/evaluate.py`
- GPU: `NVIDIA H100 NVL`

The clean path applied AMR1 repair, preserved half-frame semantics, generated
1200 frames, and passed strict raw-size validation. The result is still far
from C-063 because repairing only `4095` selected residual pixels barely moves
the CRF52 PoseNet collapse. This retires the tiny budget4096 implementation as
a frontier path, but it does not retire charged repair/grammar generally. The
next mask repair experiment must use exact marginal atom response or a
different representation, not a tiny hand-picked region budget.

Artifact:

`experiments/results/vast_harvest/c063_charged_mask_rpk1_h100nvl_fix2_20260502/exact_eval/contest_auth_eval.json`

## 2026-05-02 - C-067 Frontier And Main-Effort Reallocation

Evidence grade: `A++` Tesla T4 exact CUDA.

The C-063 fixed-slice equivalent promotion landed as the new active exact
frontier:

- score: `0.31561703078448233`
- archive bytes: `276214`
- archive SHA-256:
  `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`
- PoseNet: `0.00049637`
- SegNet: `0.00061244`
- score delta versus C-063: `-0.000005999999999950489`
- artifact:
  `experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/contest_auth_eval.json`

This is a clean deploy frontier, but it is deliberately classified as a
micro-frontier. The result validates custody and rate accounting, not a path to
break `0.30`. More scalar pose water-filling, raw CMG1 wrappers, protected AV1
mask reencodes, or tiny AMR1 residual budgets are now below the main-effort EV
threshold unless a fast H100 diagnostic first proves a materially better exact
component outcome.

The highest-EV mask path is now a real predictive/learned charged mask grammar
or Q-FAITHFUL geometry closure:

- The C-063 mask stream is `223385` charged bytes, SHA-256
  `a5c2b89c110d75220cd09b2f27f2e92844626ae7ed0d2c797290dcf43c7068eb`.
- The trace-weighted planning packet
  `experiments/results/c063_trace_weighted_mask_grammar_plan_20260502_codex/atom_plan_manifest.json`
  estimates candidate archive sizes of `56932`, `54872`, and `53860` bytes for
  budgets `4096`, `2048`, and `1024` if a future decoder can reconstruct
  scorer-critical mask structure from charged atoms. These are planning-only
  estimates, not score claims.
- The measured negatives say the decoder must reconstruct complete,
  scorer-aligned geometry. AV1 CRF replacement, protected-pixel pre-copying,
  and budget4096 post-hoc AMR1 repair all collapse PoseNet or barely move the
  collapse.

Decision: the next implementation tranche should build a contest-runtime
decoder contract for predictive mask grammar (`CMG2`-class) or repair the
Q-FAITHFUL zoom/warp geometry closure from existing snapshots. Exact CUDA
promotion should resume only after the produced archive is structurally capable
of replacing most of the mask stream while preserving full mask geometry.

## 2026-05-02 - CMG2 Lossless Tensor Probe Closed

Evidence grade: `empirical_byte_probe_only`.

Implemented a reproducible local probe:

- tool: `experiments/probe_cmg2_mask_codecs.py`
- tests: `src/tac/tests/test_probe_cmg2_mask_codecs.py`
- manifest:
  `experiments/results/c063_trace_weighted_mask_grammar_plan_20260502_codex/cmg2_lossless_probe_charged_pr67_20260502T0950Z/cmg2_mask_codec_probe_manifest.json`
- best probe payload:
  `experiments/results/c063_trace_weighted_mask_grammar_plan_20260502_codex/cmg2_lossless_probe_charged_pr67_20260502T0950Z/best_payload.cmg2probe`

The probe consumes the real decoded C-067/PR67 mask tensor
`600x384x512`, `uint8`, classes `0..4`, SHA-256
`0344fcfc39e683f21a71db1085a8697a94c4606f91f883362e9acc02fc7b5b45`,
and compares deterministic lossless transforms against the archive-charged
current mask segment:

- baseline mask segment: `219472` charged bytes, Brotli payload SHA-256
  `d1ae0d39c848e5715b74bb0122269066a4a1ab60ba9e12f34e70fd62ac136d87`
- decoded AV1 mask stream after Brotli expansion: `223385` bytes, SHA-256
  `a5c2b89c110d75220cd09b2f27f2e92844626ae7ed0d2c797290dcf43c7068eb`
- best lossless probe: `raw_u8_bz2_9`, `340315` bytes,
  SHA-256 `fae91d8d641fd06a4bc7d3e801fb100c439a4684b2bdf2b034054e5866e2c3d1`
- baseline delta: `+120843` bytes, so no lossless variant beats the current
  charged segment.

This is not score evidence and not a submission candidate. It closes a cheap
branch: exact tensor preservation through generic lossless packing is not the
mask breakthrough. `CMG2` must now mean predictive/lossy/scorer-weighted mask
grammar with charged residual atoms, or a learned decoder trained through the
exact renderer/scorer geometry. Any future `CMG2` exact-eval dispatch should
first include a byte-screen manifest proving a plausible rate win versus the
current charged mask stream and a reviewed runtime decoder envelope.

Additional cheap lossless check:

- manifest:
  `experiments/results/c063_trace_weighted_mask_grammar_plan_20260502_codex/pr67_mask_brotli_recompress_probe_20260502T0955Z.json`
- source decoded AV1 mask stream: `223385` bytes, SHA-256
  `a5c2b89c110d75220cd09b2f27f2e92844626ae7ed0d2c797290dcf43c7068eb`
- best Brotli recompress: `quality=11`, `lgwin=18`, `mode=generic`,
  `219472` bytes
- delta versus current PR67 charged mask segment: `0` bytes

This means the active PR67 mask segment is already reproducible at the best
local Brotli setting in the tested grid. The remaining mask-rate surface is not
outer recompression; it requires a representation or decoder change.

## 2026-05-02 - CMG2 2x2 Low-Dimensional Exact-Eval Dispatch

Evidence grade: `empirical_archive_candidate` plus `queued_exact_eval`.

Built the first exact-evaluable `CMG2` lossy mask-grammar archive against the
active C-067 frontier. This is not a score claim until the Lightning
`contest_auth_eval.json` lands.

- builder: `experiments/build_cmg2_downsample_candidate.py`
- local tests: `79 passed` for CMG2 builder, auth-eval, probe, packed payload,
  runtime guards, and canonical smoke coverage
- local unpack/runtime smoke: single-member `p` unpacked to `renderer.bin`,
  `masks.cmg2`, and `optimized_poses.bin`; `_load_masks_from_cmg2` returned
  a `600x384x512` half-frame mask tensor
- frontier anchor: C-067 A++ T4 score `0.31561703078448233`, archive
  `276214` bytes, SHA-256
  `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`
- candidate archive: `194020` bytes, SHA-256
  `e695829a9c45e827b8abc430b87c4871f7d563ff5b26767a6776483613fff3b1`
- archive byte delta: `-82194` bytes versus C-067
- formula-only rate-score cushion:
  `25 * 82194 / 37545489 = 0.054729...`
- decoded-mask proxy disagreement versus C-067 masks:
  `0.005475446912977431`
- build manifest:
  `experiments/results/c067_cmg2_downsample2x2_candidate_20260502T1010Z/build_manifest.json`
- Lightning source manifest:
  `.omx/state/exact_eval_c067_cmg2_downsample2x2_t4_20260502T1000Z_manifest.json`
- exact-eval queue state:
  `.omx/state/c067_cmg2_downsample2x2_t4_batch_jobs_20260502T1000Z.json`

An H100 SXM diagnostic was attempted first, but Vast.ai live state showed the
retained H100s stopped/exited and a fresh H100 create failed with an account
credit block. The dispatch was therefore rerouted to Lightning T4
`exact_eval_c067_cmg2_downsample2x2_t4_20260502T1000Z` with archive SHA/bytes
pinned, remote supply-chain preflight, CUDA runner preflight, source manifest,
adjudication, and component trace enabled.

Branch rule after harvest:

- If score/components survive near C-067, promote immediately because the run
  is already T4/equivalent A++ custody.
- If score collapses, preserve the exact component trace and build the next
  `CMG2 + charged AMR1 repair` variants using hard-pair/foveal/horizon/ego
  atoms selected by marginal benefit per byte.

## 2026-05-02 - CMG2 Foveated/Hard-Pair Repair Wave

Evidence grade: `empirical_archive_candidate` plus `queued_exact_eval`.

Implemented and verified a deterministic `CMG2 + AMR1` repair planner/builder:

- builder:
  `experiments/build_cmg2_foveated_repair_candidates.py`
- tests:
  `src/tac/tests/test_build_cmg2_foveated_repair_candidates.py`
- focused verification:
  `6 passed` for the new builder plus CMG2 builder tests
- source CMG2 manifest:
  `experiments/results/c067_cmg2_downsample2x2_candidate_20260502T1010Z/build_manifest.json`
- ranking trace:
  `experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/component_trace.json`

The builder treats the CMG2 disagreement tensor as a residual atom pool grouped
by `(pair_index, source_class, fovea_band)`. It ranks atoms by C-067 component
trace signal, foveal weight, hard-pair prior, and measured compressed AMR1
bytes. The resulting archives remain contest-faithful: `masks.cmg2` and
`alpha4_residual_repair.amr1.xz` are both charged inside the packed
`archive.zip`, and `score_claim=false` until exact CUDA auth eval lands.

First byte-screen curve:

- plan:
  `experiments/results/c067_cmg2_foveated_repair_candidates_20260502T1005Z/cmg2_foveated_repair_plan.json`
- atoms discovered: `3992`
- top1 archive: `194888` bytes, disagreement `0.005474819607`
- top32 archive: `197849` bytes, disagreement `0.005448082818`
- top96 archive: `204354` bytes, disagreement `0.005379562378`

Wide byte-screen curve:

- plan:
  `experiments/results/c067_cmg2_foveated_repair_candidates_wide_20260502T1006Z/cmg2_foveated_repair_plan.json`
- top128 archive: `207506` bytes, `-68708` bytes versus C-067
- top160 archive: `210602` bytes, `-65612` bytes versus C-067
- top256 archive: `219850` bytes, `-56364` bytes versus C-067,
  disagreement `0.005218141344`, SHA-256
  `e1f88079b8b36eb4326812d00d1e1a4c89a19b61778280c3924037e10fbbc664`
- top512 archive: `248074` bytes, `-28140` bytes versus C-067,
  disagreement `0.00492096795`, SHA-256
  `efd0da3ee2f451d7574409e4193ab2fc3fd09b5292315dd900fddea4426c6244`
- top768/top1024 become byte-regressive versus C-067 before score effects
  and are not first-wave dispatches.

Two exact T4 evals were queued because H100/Vast was unavailable:

- `exact_eval_c067_cmg2_foveated_top256_t4_20260502T1007Z`
- `exact_eval_c067_cmg2_foveated_top512_t4_20260502T1007Z`

Both use source manifest
`.omx/state/exact_eval_c067_cmg2_foveated_top256_top512_t4_20260502T1007Z_manifest.json`
and state files:

- `.omx/state/c067_cmg2_foveated_top256_t4_batch_jobs_20260502T1007Z.json`
- `.omx/state/c067_cmg2_foveated_top512_t4_batch_jobs_20260502T1007Z.json`

Branch rule:

- If plain CMG2 survives, compare top256/top512 on exact score and use the best
  point as the next atom-waterfill anchor.
- If plain CMG2 collapses but a repaired variant survives, increase atom
  granularity around the winning pair/class/band families.
- If all three collapse, CMG2 nearest-neighbor downsample is a measured
  implementation failure; pivot to learned/predictive mask grammar or
  Q-FAITHFUL successor training/export rather than more hand-picked residual
  atoms over the same base.

## 2026-05-02 - Plain CMG2 2x2 T4 Result

Evidence grade: `A-negative scoped forensic`.

The plain CMG2 2x2 archive landed as exact Tesla T4 CUDA evidence through the
canonical path. It is a measured implementation failure, not a harness failure
and not a CMG family kill.

- job:
  `exact_eval_c067_cmg2_downsample2x2_t4_20260502T1000Z`
- harvest:
  `experiments/results/lightning_batch/exact_eval_c067_cmg2_downsample2x2_t4_20260502T1000Z/`
- archive bytes: `194020`
- archive SHA-256:
  `e695829a9c45e827b8abc430b87c4871f7d563ff5b26767a6776483613fff3b1`
- hardware: `Tesla T4`, `gpu_t4_match=true`
- samples: `600`
- score: `2.294741150018026`
- PoseNet distance: `0.30416307`
- SegNet distance: `0.00421524`
- component trace cross-check: `all_match=true`
- promotion eligible: `false`
- adjudication status:
  `REGRESSION_AND_COMPONENT_GATE_REVIEW_REQUIRED`

Interpretation:

The `82194`-byte rate win is real, but isotropic nearest-neighbor 2x2 block-mode
destroys temporal/semantic geometry. This validates the need for atom repair,
learned/predictive grammar, or geometry-aware anisotropic/foveated transforms.
It does not justify spending on more plain isotropic downsample exact evals.

Immediate follow-through:

- top256 and top512 `CMG2 + AMR1` repair archives are now the active T4 tests.
- local directional byte screen found `2x1` and `3x1` variants that preserve
  different geometry/rate tradeoffs:
  - `1x2`: `327389` bytes, byte-regressive, disagreement `0.0014242892795138888`
  - `2x1`: `225013` bytes, `-51201` bytes, disagreement `0.005343195597330729`
  - `3x1`: `188463` bytes, `-87751` bytes, disagreement `0.007008048163519965`
  - `4x1`: `133008` bytes, `-143206` bytes, disagreement `0.010060178968641493`
- Do not queue directional variants until top256/top512 clarify whether AMR1
  repair can move the collapsed base meaningfully.

## 2026-05-02 - CMG2 Foveated Repair T4 Results

Evidence grade: `A-negative scoped forensic`.

Both charged AMR1 repair points landed on exact Tesla T4 CUDA through
`archive.zip -> inflate.sh -> upstream/evaluate.py`. They are measured
implementation failures, not harness failures and not a learned/predictive
CMG family kill.

Top512 repair:

- job:
  `exact_eval_c067_cmg2_foveated_top512_t4_20260502T1007Z`
- harvest:
  `experiments/results/lightning_batch/exact_eval_c067_cmg2_foveated_top512_t4_20260502T1007Z/`
- archive bytes: `248074`
- archive SHA-256:
  `efd0da3ee2f451d7574409e4193ab2fc3fd09b5292315dd900fddea4426c6244`
- hardware: `Tesla T4`, `gpu_t4_match=true`
- samples: `600`
- score: `2.1249135530811407`
- PoseNet distance: `0.24762903`
- SegNet distance: `0.00386108`
- component trace cross-check: `all_match=true`
- runtime tree SHA-256:
  `0e356bde4df817ea7c6557d67823bfaa5393ca4c38ae6fd4d3414732e6f459a0`
- promotion eligible: `false`

Top256 repair:

- job:
  `exact_eval_c067_cmg2_foveated_top256_t4_20260502T1007Z`
- harvest:
  `experiments/results/lightning_batch/exact_eval_c067_cmg2_foveated_top256_t4_20260502T1007Z/`
- archive bytes: `219850`
- archive SHA-256:
  `e1f88079b8b36eb4326812d00d1e1a4c89a19b61778280c3924037e10fbbc664`
- hardware: `Tesla T4`, `gpu_t4_match=true`
- samples: `600`
- score: `2.2229578832824526`
- PoseNet distance: `0.27912381`
- SegNet distance: `0.00405869`
- component trace cross-check: `all_match=true`
- runtime tree SHA-256:
  `0e356bde4df817ea7c6557d67823bfaa5393ca4c38ae6fd4d3414732e6f459a0`
- promotion eligible: `false`

Interpretation:

AMR1 repair over the isotropic nearest-neighbor 2x2 base improves the collapse
only weakly. Top512 improves plain CMG2 by about `0.1698` score while adding
`54054` bytes, and top256 is worse than top512. PoseNet remains between
`498.9x` and `562.3x` the C-067 reference, so the current repair atoms are
not approximating the geometry that the public mask stream gives PoseNet.

Decision:

- Retire plain isotropic nearest-neighbor CMG2 and hand-picked AMR1 repair over
  that base as main promotion paths.
- Do not spend more T4 on additional nearest-neighbor repair depths or
  directional variants unless a new local scorer-geometry argument is added.
- Keep the implementation and exact traces as design constraints for CMG3:
  the next mask grammar must be learned/predictive/geometry-preserving at
  decode time, not a coarse block canvas patched with sparse class residuals.
- Main effort now shifts to contest-faithful learned/predictive mask grammar
  and Q-FAITHFUL successor export/packer work, with C-067 exact T4 remaining
  the frontier anchor.

## 2026-05-02 - Predictive Mask-Grammar Byte Probe

Evidence grade: `empirical_byte_probe_only`.

Implemented and ran a deterministic predictive mask-grammar byte probe:

- tool:
  `experiments/probe_predictive_mask_grammar.py`
- tests:
  `src/tac/tests/test_probe_predictive_mask_grammar.py`
- real-tensor manifest:
  `experiments/results/c067_predictive_mask_grammar_probe_20260502T1040Z/predictive_mask_grammar_probe_manifest.json`
- decoded mask tensor:
  `experiments/results/c063_trace_weighted_mask_grammar_plan_20260502_codex/decoded_mask_array.npy`
- baseline:
  charged current PR67 mask stream, `219472` bytes

Best empirical candidates by compressed probe payload size:

- `row_span_stride4_class_predictor`: `63212` bytes with `lzma6`,
  `-156260` bytes versus the current charged mask stream.
- `column_span_stride4_class_predictor`: `132095` bytes,
  `-87377` bytes.
- `row_span_stride1_class_predictor`: `186188` bytes,
  `-33284` bytes.
- `anisotropic_downsample_2x1_mode`: `197348` bytes,
  `-22124` bytes.

Interpretation:

This is the first post-CMG2 mask probe with enough standalone byte headroom to
justify a real archive/runtime implementation. It is not score evidence:
the payload excludes future decoder code, archive wrapper, validator coverage,
and exact CUDA eval. But it is directionally stronger than nearest-neighbor
CMG2 because row-span grammars preserve road/lane/horizon extents rather than
averaging local blocks and then trying to patch them.

Next implementation target:

- Build `CMG3` as a charged row-span grammar member with a deterministic
  fill/rasterization rule, validator allowlist, runtime loader, archive
  builder, and local smoke tests.
- First exact archive should target stride4 row spans plus a small charged
  residual map for conflicts/unsampled rows; if local byte accounting stays
  below about `120KB`, exact CUDA is justified because the rate cushion is
  large enough to tolerate nonzero but modest component drift.
- Keep the probe manifest as design input only; do not rank or promote until a
  closed archive lands exact CUDA JSON.

## 2026-05-02 - CMG3 Closed Archive Implementation And T4 Queue

Evidence grade: `empirical_archive_candidate` until exact CUDA artifacts land.

Implemented the first closed `CMG3` runtime/archive path with charged
`masks.cmg3` support:

- runtime:
  `submissions/robust_current/inflate_renderer.py`
- packed-payload allowlist:
  `submissions/robust_current/unpack_renderer_payload.py`
- archive packer allowlist:
  `experiments/build_renderer_packed_payload_archive.py`
- auth/smoke validators:
  `experiments/contest_auth_eval.py`,
  `experiments/canonical_local_auth_eval_smoke.py`
- builders:
  `experiments/build_cmg3_rowspan_candidate.py`,
  `experiments/build_cmg3_nonzero_runs_candidate.py`
- tests:
  `src/tac/tests/test_build_cmg3_rowspan_candidate.py`,
  `src/tac/tests/test_build_cmg3_nonzero_runs_candidate.py`

Permanent guard added after initial queue: the runtime now verifies
`reconstructed_mask_u8_sha256` after CMG3 decode, in addition to compressed
body and grammar stream hashes. This catches policy/runtime/postfilter drift
before spending exact eval.

Row-span closed archive byte screen:

- stride1: `251500` bytes, `-24714` bytes versus C-067, disagreement
  `0.08720835`, SHA
  `83057ab28b8cd2c7d027390feeb149763a747ee20794644b1bcf455543a66a68`
- stride2: `176108` bytes, `-100106` bytes, disagreement `0.08844308`
- stride3: `145198` bytes, `-131016` bytes, disagreement `0.09011044`
- stride4: `127805` bytes, `-148409` bytes, disagreement `0.09031554`,
  SHA `f208d09dacf3653bc3871027e44429588be7a9a86de66d86ada6709d38e51175`
- stride6: `108860` bytes, `-167354` bytes, disagreement `0.09330798`
- stride8: `97944` bytes, `-178270` bytes, disagreement `0.09541992`

Interpretation: row-span alone is very compact but too distorted to be the
first T4 spend when a lower-disagreement nonzero-run grammar is available.

Nonzero row-run closed archive byte screen:

- top1:
  `experiments/results/c067_cmg3_nonzero_runs_top1_candidate_20260502T105718Z/archive.zip`,
  `128028` bytes, SHA
  `7b23b8e7a7076ccc5814f6a31675e9071828cebfc57e8c73ac85e39c1139d397`,
  rate-only delta `-0.09867097482736209`, disagreement
  `0.0354846445719401`, body `70493` bytes with `bz2`
- top2:
  `experiments/results/c067_cmg3_nonzero_runs_top2_candidate_20260502T105724Z/archive.zip`,
  `229196` bytes, SHA
  `6ba74dc4ac94fd61e0188ae154b975e3b9890eec1c7827d8cdedfcd343b665fb`,
  rate-only delta `-0.03130735625789825`, disagreement
  `0.011170213487413195`, body `166412` bytes with `lzma_xz`
- top3:
  `292451` bytes, byte-regressive by `16237` bytes, disagreement
  `0.004185977511935764`
- top4:
  `343592` bytes, byte-regressive by `67378` bytes, disagreement
  `0.0018814934624565972`

The top1/top2 candidates were locally smoke-tested by unpacking `p` and
loading `masks.cmg3` through the runtime path. Both produced a
`600x384x512` half-frame mask tensor from charged archive bytes.

Lightning T4 exact-eval dispatch:

- source manifest:
  `.omx/state/exact_eval_c067_cmg3_nonzero_top1_top2_t4_20260502T105935Z_manifest.json`
- manifest SHA-256:
  `0463c8f6d33f1412fe551f7f4f61626f83dd36ec7c663fd6df03a4a49c8b522c`
- top1 job:
  `exact_eval_c067_cmg3_nonzero_top1_t4_20260502T1100Z`
- top1 state:
  `.omx/state/c067_cmg3_nonzero_top1_t4_batch_jobs_20260502T1100Z.json`
- top2 job:
  `exact_eval_c067_cmg3_nonzero_top2_t4_20260502T1100Z`
- top2 state:
  `.omx/state/c067_cmg3_nonzero_top2_t4_batch_jobs_20260502T1100Z.json`

As of the last refresh, both jobs are `Running` on Lightning T4-class
hardware. No score claim exists until state-derived harvest validates archive
SHA/bytes, CUDA device, sample count, component trace, and adjudication JSON.

Next branch:

- If top2 component drift is modest, use exact pair/class trace to build a
  CMG3A global atom allocator rather than fixed uniform top-K.
- If top1 or top2 collapses PoseNet similarly to CMG2, keep the negative as
  an implementation result and pivot the same charged grammar format to a
  learned deterministic postfilter/decoder or Q-FAITHFUL successor rather than
  spending more T4 on uniform hard masks.
- Immediate local high-EV variant from Darwin review:
  `CMG3A` adaptive global nonzero-row-run selection. It should choose residual
  row-run atoms by foveal/ego-motion/hard-pair/class/boundary priors under a
  byte budget instead of forcing every row to use the same K.

## 2026-05-02 - Complete Row-Span Policy Search Dispatch

Evidence grade: `empirical_archive_candidate` until exact CUDA artifacts land.

The row-span builder now searches the complete small policy space instead of
locking onto the first convenient reconstruction rule:

```text
row_fill policies: nearest, forward, linear
draw orders: 5! = 120
default classes: 5
searched policies per stride: 1800
```

This directly addresses the local-minimum risk in mask grammar work: row-fill,
draw order, and default class are now explicit charged policy variables with
runtime parity tests. The winning policy for the current byte-screened C-067
row-span candidates is `row_fill=forward`, draw order `2,1,0,3,4`.

Byte-screened complete-policy archives:

```text
stride1: 251500 bytes, SHA 19687285f9d8f66892670863466e4f8802dc33414e65312f2f23131389d59864,
         pixel disagreement 0.011474507649739583,
         formula-only if components hold 0.299160992617021
stride2: 176108 bytes, SHA 4164132d26840db05124d341e02ebf72b5df171d90f9a413c2239f33112e61eb,
         pixel disagreement 0.013449478149414062,
         formula-only if components hold 0.24896055442323425
```

These are not score claims. They are exact archive bytes selected because the
rate term is large enough to matter and the disagreement is now in the same
order as the already-promoted C-067 mask geometry.

Lightning dispatch:

```text
source manifest:
  .omx/state/exact_eval_c067_cmg3_rowspan_stride1_stride2_20260502T1225Z_manifest.json
manifest SHA-256:
  67d0cbe5a576474c06798395eee53102f2104694aa17cbd8b065522380064695
stride1 T4 job:
  exact_eval_c067_cmg3_rowspan_stride1_t4_20260502T1225Z
stride1 state:
  .omx/state/c067_cmg3_rowspan_stride1_t4_batch_jobs_20260502T1225Z.json
stride2 L40S diagnostic job:
  exact_eval_c067_cmg3_rowspan_stride2_l40sdiag_20260502T1225Z
stride2 state:
  .omx/state/c067_cmg3_rowspan_stride2_l40sdiag_batch_jobs_20260502T1225Z.json
```

As of submit/first refresh both jobs are `Pending` at zero cost. The stride1
T4 job is the promotion-grade threshold test for sub-0.30. The stride2 L40S
job is diagnostic-only unless separately rerun on T4/equivalent with identical
bytes.

Local-minimum escape branch:

- A multimask/reconciliation planner was delegated as an orthogonal empirical
  lane. It must remain non-promotable until the reconciler and every
  score-affecting payload bit are inside an archive and exact CUDA auth eval
  lands.
- If stride1 survives component gates, immediately promote it as a new
  frontier and build stride1 residual/foveal repairs from its component trace.
- If stride1 fails but stride2 exposes a recoverable component pattern,
  build a charged repair/reconciler around stride2 rather than discarding the
  whole row-span family.
- If both collapse in PoseNet, retire only the measured row-span-only
  implementation and pivot the same finite-policy/atom machinery to multimask
  reconciliation or learned deterministic geometry-preserving decode.

## 2026-05-02 - Row-Span Exact Results And Ego-Foveation Profiler

Evidence grades:

- `A++ contest T4` infrastructure for the stride1 exact eval, but
  `A-negative scoped forensic` scientific use because component gates failed.
- `A diagnostic CUDA`/`A-negative scoped forensic` for the stride2 L40S eval.
- `planning_only` for all ego-motion, foveation, and atom-subspace profiles.

Exact results:

```text
stride1 T4 job:
  exact_eval_c067_cmg3_rowspan_stride1_t4_20260502T1225Z
artifact:
  experiments/results/lightning_batch/exact_eval_c067_cmg3_rowspan_stride1_t4_20260502T1225Z/
archive:
  251500 bytes
  SHA-256 19687285f9d8f66892670863466e4f8802dc33414e65312f2f23131389d59864
score:
  24.45479736666548
PoseNet:
  51.92496872
SegNet:
  0.01500283
verdict:
  component_gate_triggered=true, promotion_eligible=false

stride2 L40S diagnostic job:
  exact_eval_c067_cmg3_rowspan_stride2_l40sdiag_20260502T1225Z
artifact:
  experiments/results/lightning_batch/exact_eval_c067_cmg3_rowspan_stride2_l40sdiag_20260502T1225Z/
archive:
  176108 bytes
  SHA-256 4164132d26840db05124d341e02ebf72b5df171d90f9a413c2239f33112e61eb
score:
  26.606438621695162
PoseNet:
  46.52362823
SegNet:
  0.04919839
verdict:
  component_gate_triggered=true, promotion_eligible=false
```

Interpretation: row-span-only replacement is not a leaderboard path. It saves
rate bytes but creates a severe pose-geometry cliff. This retires only the
measured row-span-only stride1/stride2 implementations. It does not kill
strict mask grammars, multimask reconciliation, learned decoders, foveated
repair, or row-span atoms as charged repair side information.

Hardware/openpilot/ego-motion integration landed as planning infrastructure:

```text
ego-motion plan:
  experiments/results/c067_ego_motion_field_atoms_20260502/ego_motion_field_plan.json
dynamic foveation manifest:
  experiments/results/c067_ego_motion_field_atoms_20260502/dynamic_foveation_from_ego_motion.json
schema:
  ego_motion_dynamic_foveation_manifest_v1
frame centers:
  1200 full-frame centers, consumed by pair-mask planners via pair averaging
frame centers SHA-256:
  cddd4f6b17628f8739623185d229f7418a4e26bdac06e1cfa3449925863b5866
pose source SHA-256:
  5236cf75cc95f6a9731b35f4e2fb1211aa99bfcc69e5964869edd19a7af3df9f
```

The dynamic foveation file is now a real manifest with
`score_claim=false`, `promotion_eligible=false`, evidence grade, pose SHA,
pose size, and source pose contract. Future archive builders must not consume
anonymous foveation centers.

Active-subspace profile:

```text
tool:
  experiments/profile_atom_ledger_subspaces.py
output:
  experiments/results/c067_cmg3_rowspan_escape_atoms_20260502/rowspan_static_dynamic_active_subspace_profile.json
schema:
  atom_ledger_active_subspace_profile_v1
score_claim:
  false
inputs:
  stride1 static foveal ledger
  stride1 dynamic ego-foveal ledger
  stride2 static foveal ledger
  stride2 dynamic ego-foveal ledger
top-k:
  512
```

Profile findings:

- Static and dynamic ego-foveation rankings are highly similar for the current
  mild ego-motion field: top-128 atom Jaccard is `0.9394` for stride1 and
  `0.9692` for stride2.
- The strongest repeated cliff regions are pairs around `67`, `69`, `70`,
  `285`, `286`, `289`, `290`, `292`, and `294`, with dominant confusions
  `2->3` and `0->3`.
- This says the immediate value of ego-motion/foveation is not another
  row-span-only exact eval. It should drive repaired/multimask/learned
  selectors that preserve pose-sensitive geometry while charging only
  high-density correction atoms.

High-EV lane triage after the exact negatives:

- Lane 12 NeRV full CUDA remains the largest single rate lever, but dispatch
  is still blocked until `.omx/state/lane12_nerv_l2_clearance.json` exists and
  passes the recorded gate.
- Block-FP renderer self-compression remains a major byte lever if the export
  and T4 runtime contract are already closed; prioritize contract verification
  before long training.
- SJ-KL residual coding is still an attractive medium-cost lane because it
  acts on residual structure rather than replacing mask geometry wholesale.
- PR67-style pose line search is low risk but recent exact runs show
  diminishing micro-gains; do not let it monopolize the main path unless it is
  stacked with a larger rate lever.

## 2026-05-02 - Observability And Human Feedback Tooling

The row-span cliff data and active-subspace profile now feed a deterministic
observability packet for the report and for human-in-the-loop triage:

```text
tool:
  experiments/build_yf_observability_report.py
output directory:
  reports/yousfi_fridrich_observability_20260502/
inputs:
  C-067 exact T4 eval JSON
  row-span stride1 exact T4 negative JSON
  row-span stride2 L40S diagnostic negative JSON
  row-span static/dynamic atom-subspace profile JSON
schema:
  yousfi_fridrich_observability_report_v1
score_claim:
  false
```

Generated artifacts:

```text
observability_report.json
observability_report.md
index.html
score_breakdown.svg
top_pairs.svg
score_breakdown.png
top_pairs.png
pair_centroid_hotspots.png
class_confusions.png
```

The HTML report includes lightweight local interactivity: figure switching and
sortable exact-eval component rows. Matplotlib is installed in the local venv
and recorded under the `viz` optional dependency extra, not the contest
runtime dependency path.

The Fridrich/Yousfi-facing signals surfaced explicitly are:

- byte delta versus the best exact eval anchor;
- PoseNet and SegNet ratios versus the exact anchor;
- component-cliff flags;
- wasted score per saved byte for byte-saving regressions;
- top active pairs, class confusions, score-density proxies, and spatial
  weighted centroids.

Interpretation: the strongest current signal is not "make row spans smaller."
It is "do not delete/repaint the `2->3` and `0->3` geometry around the repeated
pair clusters unless the decoder has a charged, scorer-aligned way to preserve
PoseNet." The next archive attempts should therefore be repaired/multimask/
learned-decoder candidates or public-floor packer/renderer candidates, not
more row-span-only byte sweeps.

## 2026-05-02 - Aggregate Fridrich/Yousfi Signal Surface

The profiler/report layer was hardened after review so it now exposes the
consensus signal that a Fridrich/Yousfi-style operator would actually use,
rather than only the first input ledger.

Code changes:

```text
experiments/profile_atom_ledger_subspaces.py
  adds aggregate_subspaces across all input ledgers
  adds fridrich_yousfi_signal_surface with consensus pairs/classes/confusions
  adds overlap summary for low-dimensional stability

experiments/build_yf_observability_report.py
  consumes aggregate_subspaces when present
  adds target_gap_analysis for sub-0.300
  adds action_recommendations
  adds target_gap.svg
```

Current generated control-plane artifacts:

```text
profile:
  experiments/results/c067_cmg3_rowspan_escape_atoms_20260502/rowspan_static_dynamic_active_subspace_profile.json
report:
  reports/yousfi_fridrich_observability_20260502/
new figure:
  reports/yousfi_fridrich_observability_20260502/target_gap.svg
```

Target-gap math from the refreshed report:

```text
best exact anchor:
  C067
best score:
  0.31561703078448233
target:
  0.30000000000000000
score gap:
  0.015617030784482344
bytes to remove at unchanged distortion:
  23455
```

Aggregate active-subspace consensus from four static/dynamic row-span ledgers:

```text
top repeated pairs:
  69, 67, 290, 285, 70, 289, 286, 294, 292, 293, 164, 284, 272, 273, 66, 1
dominant class confusions:
  2->3, 0->3, 0->4, 0->2, 4->0
pair-overlap summary:
  comparison_count=6
  mean_pair_jaccard=0.901366607249
  min_pair_jaccard=0.882352941176
```

This confirms a low-dimensional consensus subspace: the static and dynamic
foveal variants do not point at different physics. They point at the same
PoseNet-sensitive mask cliffs. The correct next mask-side experiment is
therefore a charged repair/fusion/multimask candidate over those consensus
pairs/confusions, not another unconstrained row-span delete/repaint sweep.

Subagent synthesis:

- Harvey landed `experiments/plan_multimask_reconciliation_atoms.py` and
  `src/tac/tests/test_plan_multimask_reconciliation_atoms.py`. It is a
  deterministic, NumPy-only, non-promotable planner that ranks fusion policies
  (`majority_vote`, `priority_order`, `disagreement_gated_veto`,
  `cheap_residual_over_base`) from source/candidate mask arrays or manifests.
  Follow-up: connect its `fusion_reconciliation_policy` to a deterministic
  archive builder that charges actual payload bytes, then exact CUDA eval.
- Euler found that the renderer self-compression claim is not dispatch-ready:
  the closed JFG path is QZS3/MQZ1, while the true `~11KB` block-FP JFG
  container has no export/runtime/T4 custody path yet. The high-EV renderer
  target is a real JFG-specific block-FP container such as `QBF1`, with loader
  tests and archive-local byte custody, not another global QZS3/MQZ1 block-size
  sweep.

Verification:

```text
.venv/bin/python -m py_compile \
  experiments/profile_atom_ledger_subspaces.py \
  experiments/build_yf_observability_report.py \
  experiments/plan_multimask_reconciliation_atoms.py \
  experiments/plan_ego_motion_field_atoms.py

.venv/bin/python -m pytest \
  src/tac/tests/test_profile_atom_ledger_subspaces.py \
  src/tac/tests/test_build_yf_observability_report.py \
  src/tac/tests/test_plan_multimask_reconciliation_atoms.py -q

result:
  9 passed in 0.49s
```

Next action rule: spend exact-eval budget only on deterministic archives that
try to preserve the consensus cliff subspace while removing at least a
material fraction of the 23,455-byte sub-0.300 gap, or on a newly closed
renderer container with real byte savings and runtime custody.

## 2026-05-02 - Non-Arbitrary Grounding, Multimask Negative, And Hotpath Greenup

User directive: all atoms, fields, radii, manifolds, selectors, codecs, and
stacking moves must be mathematically, domain, hardware, and axiom grounded.
I added this as durable operating law in `AGENTS.md`: every new dispatch-driving
knob must record the contest score term it targets, the domain prior or
measured artifact motivating it, the hardware/runtime constraint it respects,
the charged bytes it consumes, and the evidence grade. Differentiable/learned
proposal mechanisms are preferred over arbitrary grids, but final archives
remain deterministic, byte-closed, and exact-eval-gated.

Feynman council synthesis returned read-only and reinforced the same branch
decision:

```text
highest-EV path:
  scorer-constrained mask/geometry compiler over the C067/PR67 basin
do not do:
  more global whole-mask simplification or raw row-span sweeps
primary protected hotspots:
  pairs 69, 67, 290, 285, 70, 289, 286, 294
  confusions 2->3 and 0->3
near-term non-NeRV levers:
  strict charged mask grammar
  hotspot-preserving lossy mask compiler
  geometry-closed Q-FAITHFUL/JFG successor
  scorer-weighted atom compiler as hedge/calibrator
  lossless/runtime byte diet
```

Full-resolution multimask reconciliation was run against real C067 mask
candidates extracted from archive payloads:

```text
output_dir:
  experiments/results/c067_multimask_reconciliation_20260502/
manifest:
  candidate_mask_extract_manifest.json
plan:
  multimask_reconciliation_plan.json
```

Extracted tensors:

```text
cmg3_nonzero_top1  single member bytes=127928 tensor_sha=1d186f...f143e
cmg3_nonzero_top2  single member bytes=229096 tensor_sha=cff33b...a9c9
cmg3a_body200      single member bytes=257576 tensor_sha=23c725...c97b
cmg3_rowspan_stride1 single member bytes=251400 tensor_sha=286f4b...27b0
```

Planner result:

```text
candidate disagreement vs source:
  cmg3_nonzero_top1      0.035484644572
  cmg3_nonzero_top2      0.011170213487
  cmg3a_body200          0.003047103882
  cmg3_rowspan_stride1   0.087208353678

best fused policy:
  majority_vote_all
  disagreement_fraction=0.002249543932
  estimated_sparse_residual_bytes=1326879

interpretation:
  useful planning negative
  multimask fusion preserves geometry far better than raw candidates
  but the current changed-element residual proxy is byte-impossible
```

This does not retire multimask. It scopes the next version: encode only compact
regions/runs/components in the consensus hotspot subspace, not a flat sparse
residual over every changed pixel. Multimask remains `planning_only` until a
charged archive builder replaces the byte-regressive residual proxy and exact
CUDA eval lands.

Hotpath profiler:

```text
tool:
  experiments/profile_python_loop_hotpaths.py
artifact:
  experiments/results/c067_multimask_reconciliation_20260502/python_loop_hotpath_profile.json
schema:
  python_loop_hotpath_profile_v1
loop_count_after_exclusions:
  4202
score_claim:
  false
```

The profiler now excludes generated result trees and deprioritizes preflight/
training-control loops so it surfaces active compression builders instead of
static guard code. It identified and drove two semantics-preserving
vectorization fixes:

```text
experiments/build_cmg2_foveated_repair_candidates.py
  residual_runs_by_pair_class_band now uses row-wise vectorized changed-x
  segmentation split by gap, class, and fovea band

experiments/plan_cmg3_pixel_lagrangian_atoms.py
  _extract_residual_runs now uses row-wise vectorized changed-x segmentation
  split by gap and source class
```

Focused verification:

```text
.venv/bin/python -m py_compile \
  experiments/plan_cmg3_pixel_lagrangian_atoms.py \
  experiments/build_cmg2_foveated_repair_candidates.py \
  experiments/profile_python_loop_hotpaths.py

.venv/bin/python -m pytest \
  src/tac/tests/test_plan_cmg3_pixel_lagrangian_atoms.py \
  src/tac/tests/test_build_cmg2_foveated_repair_candidates.py \
  src/tac/tests/test_profile_python_loop_hotpaths.py -q

result:
  14 passed in 0.56s
```

Planner hardening after the no-op trap:

```text
experiments/plan_multimask_reconciliation_atoms.py
  adds compact_row_run_residual_over_source_proxy
  adds dispatch_relevance no-op guard and rank penalty
  adds density_metrics for learned feedback:
    rate_score_cost
    arithmetic_entropy_floor_bytes_no_model_overhead
    arithmetic_estimated_bytes_with_model_overhead
    break_even_total_component_score_improvement_required
    break_even_component_score_improvement_per_changed_element
    estimated_bytes_per_changed_element
    estimated_bytes_per_run
    changed_elements_per_run
    touched_row_fraction
  adds differentiable_feedback_contract:
    negative rate cost
    disagreement proxy
    no-op penalty
    softmax/Gumbel selection surrogate
    atom-density target
    hotspot trust-region target
```

Real C067 multimask rerun after those fixes:

```text
best non-no-op policy:
  majority_vote_all
estimated row-run residual bytes:
  454829
ideal zero-order arithmetic entropy floor:
  99294 bytes
practical arithmetic estimate with simple model overhead:
  100776 bytes
changed pixels:
  265367
changed fraction:
  0.002249543932
run count:
  60258
rate score cost:
  0.30285196179
arithmetic-estimate rate score cost:
  0.06710260186
break-even component improvement required:
  0.30285196179

no-op veto:
  moved to rank 7
  dispatchable_byte_model=false
  rank_penalty_applied=1000000000000
```

Interpretation: full fused-mask residual encoding is still not dispatchable
under either practical fixed row-run coding or an optimistic arithmetic-coded
language. The arithmetic floor is scientifically important: it says engineered
corrections are not automatically impossible, but they must be selected down
to the scorer-dense hotspot subset. We are optimizing the contest machine
score, not human visual smoothness; the relevant reward is exact/PoseNet/
SegNet component benefit minus charged bytes, with human-looking artifacts
ignored unless exact scorer traces prove value. The next high-signal experiment
is not "multimask all pixels"; it is a learned or Lagrangian selector over
compact hotspot atoms from the fused residual, constrained to the consensus
PoseNet-sensitive pair/class subspace and charged through a real archive
builder.

QBF1/block-FP subagent result:

```text
local readiness:
  QBF1 loader and runtime dispatch added
  packed-payload magic detection added
tests:
  8 qbf1 tests passed in subagent; parent reran combined focused tests
empirical byte screen:
  C067 archive bytes=276214 sha=226475de42ec...efb38a
  C067 PR67 fixed-slice streams:
    renderer QZS3=59288
    masks=223385
    poses=7200
  best QBF1-v1 replacement archive:
    280413 bytes
    +4199 bytes versus C067
dispatch decision:
  do not exact-eval QBF1-v1
  QBF1-v2 only if it beats current packed QZS3 renderer slice locally
```

Additional verification after planner hardening:

```text
.venv/bin/python -m pytest \
  src/tac/tests/test_plan_multimask_reconciliation_atoms.py \
  src/tac/tests/test_qbf1_renderer_codec.py \
  src/tac/tests/test_plan_cmg3_pixel_lagrangian_atoms.py \
  src/tac/tests/test_build_cmg2_foveated_repair_candidates.py \
  src/tac/tests/test_profile_python_loop_hotpaths.py -q

result:
  29 passed in 1.50s
```

Subagent state:

```text
Dirac:
  closed after QBF1-v1 readiness and byte-negative no-dispatch finding
Plato:
  returned read-only learned optimizer / sub-0.3 integration design
  proposed C067PolicyLearner and top next experiments:
    PMG-HOTSPOT-v1
    MM-GUMBEL-KNAPSACK-v1
    QF-GEOM-QBF1-v2 only after local byte win and geometry closure
```

Next action rule: the main lane should build a hotspot-preserving charged mask
candidate over the consensus pairs/confusions, using the multimask fused
residual only as a density-labeled proposal field. Do not queue exact eval for
byte-only, no-op, full-residual multimask, or QBF1-v1 candidates. Exact eval
should wait for a deterministic archive whose charged bytes plausibly remove a
material fraction of the 23455-byte C067 gap and whose local trust-region
metrics do not repeat the measured PoseNet cliff pattern.

## C067 archive byte accounting and self-compression gate

Timestamp: 2026-05-02T13:48Z.

Tooling landed:

```text
experiments/profile_archive_byte_accounting.py
src/tac/tests/test_profile_archive_byte_accounting.py
```

Verification:

```text
.venv/bin/python -m py_compile \
  experiments/profile_archive_byte_accounting.py \
  src/tac/tests/test_profile_archive_byte_accounting.py

.venv/bin/python -m pytest \
  src/tac/tests/test_profile_archive_byte_accounting.py -q

result:
  3 passed in 0.09s
```

Generated artifacts:

```text
experiments/results/c067_archive_byte_accounting_20260502/archive_byte_accounting.json
experiments/results/c067_archive_byte_accounting_20260502/archive_byte_accounting.md
```

C067 anchor:

```text
archive:
  experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip
bytes:
  276214
sha256:
  226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a
score:
  0.31561703078448233
gpu:
  Tesla T4
evidence:
  A++ contest CUDA, promotion_eligible=true in the source auth-eval packet
```

Target pressure:

```text
exact bytes to cross 0.300 at unchanged distortion:
  23454
buffered planning bytes:
  23455
buffered target archive bytes:
  252759
buffered target score at unchanged distortion:
  0.2999993090390018
```

Exact charged archive anatomy:

```text
zip overhead:
  100 bytes
single payload:
  p
payload format:
  public_pr67_qzs3_qp1_fixed_slices
payload bytes:
  276114
payload internal overhead:
  0 bytes

streams:
  masks.mkv:
    encoded=219472 bytes
    decoded=223385 bytes
    codec=brotli_av1_obu
    rate_score=0.14613739616
    encoded_entropy=7.998591976459 bits/byte
    bitplane_zero_order_entropy=219465 bytes
    nested recompression savings=0 bytes
    buffered gap as fraction of stream=10.6866%
  renderer.bin:
    encoded=55965 bytes
    decoded=59288 bytes
    codec=brotli_qzs3
    rate_score=0.037264796311
    encoded_entropy=7.993832581821 bits/byte
    bitplane_zero_order_entropy=55951 bytes
    nested recompression savings=0 bytes
    buffered gap as fraction of stream=41.9083%
  optimized_poses.bin:
    encoded=677 bytes
    decoded=7200 bytes
    codec=public_qp1_brotli
    rate_score=0.000450786511
    encoded_entropy=7.673436926071 bits/byte
    bitplane_zero_order_entropy=677 bytes
    nested recompression savings=0 bytes
    buffered gap as fraction of stream=3464.4018%
```

Interpretation:

Generic self-compression is exhausted for the current public-PR67-style
packed byte grammar. The archive is a single charged payload with effectively
zero wrapper waste and near-random byte/bitplane statistics. The route to
sub-0.300 is therefore not stronger Brotli/LZMA/ZIP; it is representation
replacement or decoder-aware grammar change with every decoder/model bit
charged inside the archive.

Actionable self-compression ranking:

```text
1. masks.mkv:
   only stream where a modest relative byte reduction can close the gap.
   Must preserve PoseNet/SegNet geometry through learned/grammar/hotspot
   trust regions, not plain AV1 CRF reencode.
2. renderer.bin:
   can close the gap only with a large ~42% renderer-stream cut at unchanged
   distortion. QBF1-v1 and current QBF1-v2 are local byte-negative; keep
   renderer work on QZS4/MQZ1/block-policy deltas until local bytes beat QZS3.
3. optimized_poses.bin:
   too small to matter alone. Keep pose search as polish or synergy only.
4. zip/container:
   100-byte ceiling; irrelevant for the sub-0.300 gap.
```

QBF1-v2 subagent result integrated:

```text
local QZS3 reference:
  59288 raw renderer bytes
best QBF1-v1:
  121618 raw bytes
best self-describing QBF1-v2 planning layout:
  72247 raw bytes
existing QZS3 b128/QZS4-like policy:
  56300 raw bytes
decision:
  QBF1-v2 hard no-go under the current self-describing contract.
  Do not exact-eval QBF1-v1/v2 until a concrete local layout beats 59288
  bytes and has reviewed inflate parity.
```

Submission/report integration from subagent:

```text
scripts/build_contest_submission_packet.py
  now supports --score-authority contest_auth_eval.adjudicated.json
automated C067 packet:
  experiments/results/submission_packet_c067_20260502/automated_packet/
score/supporting-artifact separation:
  exact T4 JSON remains score authority
  planner, visualization, and roadmap files are non-score supporting evidence
```

Next dispatch consequence:

The highest-EV self-compression route is PMG-HOTSPOT or learned/multimask
hotspot residual selection over the mask stream. It must remove roughly
23.5KB net at unchanged distortion, or trade a smaller byte drop for measured
SegNet/PoseNet gain. Any candidate whose local plan does not attack at least a
material fraction of this byte gap should stay out of the exact-eval queue.

## PMG-HOTSPOT C067 self-compression exact-eval dispatch

Timestamp: `2026-05-02T14:04Z`

Evidence grade before eval: `empirical` byte/runtime decode only. No score
claim.

Candidate:

```text
id=pmg_hotspot_rowspan_stride2
archive=experiments/results/pmg_hotspot_candidate_c067_20260502/archive.zip
archive_bytes=187144
archive_sha256=3ab4d0c85ae15325b61b7838c375383a460e7ad1b911b677101fda50e2611c53
delta_bytes_vs_C067=-89070
formula_only_rate_delta_vs_C067=-0.0593080569545918
build_manifest=experiments/results/pmg_hotspot_candidate_c067_20260502/build_manifest.json
byte_profile=experiments/results/pmg_hotspot_candidate_c067_20260502/archive_byte_accounting.json
```

Mask grammar payload:

```text
mode=row_span_stride_class_predictor_hotspot_residual_v1
compressor=bz2
payload_bytes=125734
body_bytes=124414
span_tensor_raw_bytes=2304000
residual_record_count=2242
residual_record_bytes=20178
source_mask_sha256=0344fcfc39e683f21a71db1085a8697a94c4606f91f883362e9acc02fc7b5b45
decoded_runtime_mask_sha256=5b2c7a4a4afd47076d42e874d86928e65159f740b9635197e232504a61cbafdc
final_proxy_disagreement_fraction=0.015210562811957465
```

Risk:

The candidate has enough byte movement to matter, but the proxy mask
disagreement is high. This dispatch is deliberately broad-gated forensic
exact CUDA evidence: if components collapse, preserve the component trace and
convert it into a sharper hotspot/repair selector rather than treating PMG as a
broad method kill.

Dispatch:

```text
Vast H100 attempt:
  blocked before instance creation
  cause=provider account lacks credit
  evidence=.omx/state/vast_create_pmg_hotspot_retry_20260502T1352Z.json

Lightning doctor:
  status=OK
  evidence=.omx/state/lightning_doctor_pmg_hotspot_20260502T1400Z.json

Lightning staged manifest:
  .omx/state/pmg_hotspot_c067_t4_20260502T1402Z_manifest.json
  remote verification=OK
  file_count=1280
  total_bytes=21983870

Lightning T4 exact eval:
  job=exact_eval_pmg_hotspot_c067_t4_20260502T1402Z
  state=.omx/state/pmg_hotspot_c067_t4_batch_jobs_20260502T1402Z.json
  machine=g4dn.2xlarge
  studio=lossy-compression-challenge
  teamspace=comma-lab
  status=Pending at first refresh
  expected_archive_sha256=3ab4d0c85ae15325b61b7838c375383a460e7ad1b911b677101fda50e2611c53
  expected_archive_bytes=187144
  T4 inflate env pins:
    INFLATE_TORCH_SPEC=torch==2.5.1+cu124
    UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124
    UV_INDEX_STRATEGY=unsafe-best-match
```

Permanent hardening landed in this slice:

```text
scripts/launch_lane_on_vastai.py:
  create_instance now fails closed on nonzero create return codes, empty
  provider output, stderr-only JSON, and explicit provider error JSON.
test:
  src/tac/tests/test_launch_lane_on_vastai_create_instance.py
```

Local verification:

```text
py_compile touched Python: pass
bash -n scripts/remote_lane_pmg_hotspot_c067_eval.sh: pass
pytest profile/PMG/Vast-focused tests: 8 passed
git diff --check touched files: pass
```

Harvest rule:

Only state-derived Lightning harvest output may upgrade or classify this
candidate. Required artifacts are `contest_auth_eval.json`,
`contest_auth_eval.adjudicated.json`, `component_trace.json`, archive bytes,
archive SHA, device CUDA/T4 evidence, sample count 600, and recomputed score
from components.

### PMG compressor supersession

While the first PMG exact-eval job was still waiting/running, a local
compressor screen found a strictly smaller PMG archive with identical decoded
mask SHA:

```text
bz2 archive:
  bytes=187144
  sha256=3ab4d0c85ae15325b61b7838c375383a460e7ad1b911b677101fda50e2611c53
  decoded_mask_sha256=5b2c7a4a4afd47076d42e874d86928e65159f740b9635197e232504a61cbafdc

lzma_xz archive:
  archive=experiments/results/pmg_hotspot_candidate_c067_20260502_lzma/archive.zip
  bytes=184598
  sha256=20b7cfe3d034f173be1d71193a71d9fa79c7da118790f92209ea0bf09643e660
  decoded_mask_sha256=5b2c7a4a4afd47076d42e874d86928e65159f740b9635197e232504a61cbafdc
  delta_vs_bz2=-2546 bytes
  delta_vs_C067=-91616 bytes
  formula_only_rate_delta_vs_C067=-0.06100333384924085
  unchanged-distortion score vs C067=0.2546136969352415
  component-degradation budget to score 0.300=0.04538630306475849
```

The builder default was changed from `bz2` to `lzma_xz`, with focused coverage:

```text
src/tac/tests/test_build_pmg_hotspot_candidate.py::test_pmg_hotspot_builder_defaults_to_lzma_xz
```

Lightning T4 lzma dispatch:

```text
job=exact_eval_pmg_hotspot_c067_lzma_t4_20260502T1408Z
state=.omx/state/pmg_hotspot_c067_lzma_t4_batch_jobs_20260502T1408Z.json
manifest=.omx/state/pmg_hotspot_c067_lzma_t4_20260502T1408Z_manifest.json
remote verification=OK
status=Pending at first refresh
```

The original bz2 job remains useful as component-signal hedge because it
decodes the same mask tensor. If it lands first, its component distances should
be used to decide whether to keep waiting for lzma or stop the lzma duplicate.
Only lzma exact eval can claim the lower-byte score.

Prepared fallback PMG grid:

```text
stride2_lzma:
  bytes=184598
  sha256=20b7cfe3d034f173be1d71193a71d9fa79c7da118790f92209ea0bf09643e660
  decoded_mask_sha256=5b2c7a4a4afd47076d42e874d86928e65159f740b9635197e232504a61cbafdc
  pixel_disagreement=0.015210562811957465
  residual_records=2242
  unchanged_distortion_score=0.2546136969352415
  component_budget_to_sub03=0.04538630306475849

stride4_lzma:
  archive=experiments/results/pmg_hotspot_candidate_c067_stride4_lzma_20260502/archive.zip
  bytes=136967
  sha256=145c7c2358badac72f53c396ba3a456ebe8f35e7cdc88514289cc58df3d5cad4
  decoded_mask_sha256=9ebabf4c1d13d0bc7fb629785e2902b614cdaa864be6aa260b530bee6d644432
  pixel_disagreement=0.018023028903537325
  residual_records=2464
  unchanged_distortion_score=0.22289816913907934
  component_budget_to_sub03=0.07710183086092065

stride8_lzma:
  archive=experiments/results/pmg_hotspot_candidate_c067_stride8_lzma_20260502/archive.zip
  bytes=102456
  sha256=48e70211fb621ae7919924b5eb57002473e1834dd7fbf758d673003597c1608a
  decoded_mask_sha256=a68b42b3da4128ac34bfdd760ddb767bbf5ef2a0b9d9bc4f151567059bf4b9a0
  pixel_disagreement=0.02741742451985677
  residual_records=2660
  unchanged_distortion_score=0.1999187108078801
  component_budget_to_sub03=0.1000812891921199
```

Staging:

```text
manifest=.omx/state/pmg_hotspot_c067_stride48_t4ready_20260502T1414Z_manifest.json
remote verification=OK
file_count=1280
total_bytes=22018150
```

Dispatch rule for fallback grid:

Do not submit stride4/stride8 until stride2 exact evidence shows PMG does not
catastrophically break scorer geometry. If stride2 is near or below sub-0.300,
submit stride4 first because it adds a much larger rate cushion at only a
moderate proxy-disagreement increase. Submit stride8 only if component traces
show the errors are concentrated outside PoseNet-sensitive regions or if
stride4 lands safely.

### Self-compression profiler hardening

The byte profiler now flags eval/archive mismatches explicitly. If an eval JSON
for C067 is supplied while profiling a candidate archive, the target-gap block
records:

```text
reference_matches_profiled_archive=false
reference_warning=target_gap was computed from eval_json's scored archive,
which does not match the profiled archive bytes/SHA; treat it as a reference
gap only
```

This prevents a paper/report metabug where a candidate byte profile could be
misread as a scored candidate. Coverage:

```text
src/tac/tests/test_profile_archive_byte_accounting.py::test_profile_eval_json_mismatch_is_flagged_as_reference_only
```

### PMG exact negative and pair-protection curve

State-derived T4 harvest for the stride2 PMG-HOTSPOT archive landed as a
scoped A-negative, not a broad method kill:

```text
job=exact_eval_pmg_hotspot_c067_t4_20260502T1402Z
score=30.930370939355445
bytes=187144
sha256=3ab4d0c85ae15325b61b7838c375383a460e7ad1b911b677101fda50e2611c53
PoseNet=69.20815277
SegNet=0.04498317
hardware=Tesla T4
component_diff=experiments/results/pmg_hotspot_candidate_c067_20260502/pmg_vs_c067_component_diff_top600.json
```

The lzma duplicate was stopped because it had the same decoded mask SHA and
would only duplicate the same geometry collapse. The exact component diff
shows PMG saved `89070` bytes but paid `+30.61475350814471` total score vs
C067, dominated by `+26.23698868074241` PoseNet and `+4.437072884356893`
SegNet.

Implemented a charged whole-pair protection knob in
`experiments/build_pmg_hotspot_candidate.py --protect-pair-indices` and
byte-screened the repair curve:

```text
curve=experiments/results/pmg_hotspot_candidate_c067_20260502/pair_protection_curve.json
top0=184598B
top10=194350B
top20=203974B
top40=221296B
top80=253108B
top120=284504B
top160=316096B
top240=381120B
top320=441732B
top480=555918B
top600=635102B
```

Top80 is already too large for sub-0.300 at unchanged C067 distortion
(`0.3002317586913289`), while top120+ is byte-regressive vs C067. Therefore
whole-pair repair is too coarse. Do not dispatch PMG stride4/stride8 or
pair-protection variants. Use the exact trace as planning signal for a
region/class/component/boundary-scale learned selector.

### Fine PMG row-run atom selector

The region-scale follow-up is now concretely represented as a PMG-vs-C067
excess-weighted atom ledger:

```text
pair_weights=experiments/results/pmg_hotspot_candidate_c067_20260502/pmg_vs_c067_pair_excess_weights.json
atom_ledger=experiments/results/pmg_hotspot_candidate_c067_20260502_lzma/pmg_vs_c067_residual_atom_ledger.json
atom_count=169708
row_run_atoms=164948
```

`experiments/build_pmg_hotspot_candidate.py` can now consume that ledger via
`--residual-atom-ledger` and `--residual-atom-count`, recording
`pmg_hotspot_atom_ledger_selection_v1` in the manifest. A focused source-SHA
metabug was fixed at the same time: verify planner source `.npy` SHA when
available because planner tensor SHA reflects its int16 load representation,
not the builder's uint8 source tensor.

Byte screen:

```text
top64=185006B
top128=185154B
top256=185626B
top512=186498B
top1024=187974B
top2048=190182B
top3072=193358B
top4068=195762B
```

The best diagnostic point currently queued is top4068:

```text
job=exact_eval_pmg_hotspot_atomtop4068_l40sdiag_20260502T1445Z
archive_sha256=2567dc04185cf20775f1f6c088395aa8df9e4484daa8b25001e940d62a5d6497
archive_bytes=195762
final_mask_disagreement=0.012382837931315104
selected_atom_pixels=333572
```

Use this result as a sharp branch point. If L40S still shows PoseNet collapse,
the PMG row-span residual path is measured-negative and the next mask work
should be learned/multimask/geometry preserving. If it survives, T4-confirm
identical bytes and then sweep the atom budget around the surviving point.

Result: L40S showed severe PoseNet collapse, so this branch is a scoped
A-negative:

```text
artifact_dir=experiments/results/lightning_batch/exact_eval_pmg_hotspot_atomtop4068_l40sdiag_20260502T1445Z
score=28.41411894150047
avg_posenet_dist=62.34251404
avg_segnet_dist=0.03315286
n_samples=600
gpu=NVIDIA L40S
archive_sha256=2567dc04185cf20775f1f6c088395aa8df9e4484daa8b25001e940d62a5d6497
archive_bytes=195762
paper_claim_grade=A-negative scoped forensic
promotion_eligible=false
```

The catastrophic pairs cluster around the known hard zone (`507`, `502`,
`503`, `505`, `496`, `498`, `506`) plus pair `315`. This retires PMG
row-run-only rescue as a measured implementation, not charged mask grammar or
atom planning broadly. Do not queue another PMG row-run-only T4 promotion.
The next mask-side experiment must change atom semantics toward multimask
reconciliation, JointFrameGenerator slot-aware repair, learned topology, or
pose-conditioned residuals.

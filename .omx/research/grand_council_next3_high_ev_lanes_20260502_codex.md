# Grand Council Next-3 High-EV Lanes - 2026-05-02 Codex

Evidence stance: adversarial, contest-faithful, and archive-metered.
Score claim: `false` except for exact artifacts cited below.

## Current Anchor

Active exact frontier is C-063:

```text
score              0.3156230307844823
archive bytes      276223
archive SHA-256    83615afd130afa08e972e4a02476612397bffea53327caf3591891f8317aa52d
PoseNet            0.00049637
SegNet             0.00061244
evidence           experiments/results/lightning_batch/exact_eval_lossless_repack_c059_brotli_t4_20260502T0537Z/contest_auth_eval.adjudicated.json
```

C-063 is a pure lossless Brotli repack of C-059. It proves that rate polish is
still useful, but the remaining byte-only surface is too small to dominate the
critical path unless another lossless transform finds hundreds of bytes.

## Top 3 Next Experiments

### 1. Q-FAITHFUL full-frame/geometry-closed snapshot export

Expected score reduction per wall-clock minute: highest upside if the current
training lane can emit a valid full-frame or charged-geometry snapshot. Prior
snapshot screens failed catastrophically because half-frame masks lacked packed
zoom/warp geometry, causing PoseNet around `41` to `46`; that is a contract
failure, not evidence against the architecture.

Concrete next action:

```bash
# First reconcile the existing q_faithful claim in
# .omx/state/active_lane_dispatch_claims.md. Do not dispatch if it is still an
# unresolved same-lane conflict. If coordinated as an eval-only child, record
# the child claim before running the snapshot loop.

.venv/bin/python -u scripts/q_faithful_snapshot_loop.py \
  --checkpoint-dir /workspace/pact/lane_q_faithful_results \
  --masks-mkv /workspace/pact/lane_q_faithful_results/masks.mkv \
  --mask-frame-contract full \
  --poses-pt /workspace/pact/lane_q_faithful_results/optimized_poses.pt \
  --output-root /workspace/pact/experiments/results/qfaithful_snapshot_geometry_closed_next \
  --renderer-codec qzs3 --qzs3-block-size 32 \
  --submission-layout pr64_mask_first_single_blob \
  --pose-codec pose_qp1_v1 \
  --eval-mode run --dispatch-claim-mode already-claimed
```

If only half-frame masks exist, require `--zoom-warp-path` and charge that
payload; otherwise do not exact-eval.

### 2. C-063 hard-pair pose manifold with H100/T4 drift guard

Expected score reduction per wall-clock minute: medium immediate EV. C-059
top32 looked promising on H100 but reversed on T4; future pose work must start
from C-063 and require a larger H100 margin before consuming T4.

Concrete next action:

```bash
tools/claim_lane_dispatch.py claim --lane-id lane_line_search_pose_refinement \
  --platform vast.ai --instance-job-id c063_pr67_pr65_hardpair_basis_h100 \
  --agent codex:gpt-5.5 --predicted-eta-utc 2026-05-02T08:00Z --status eval \
  --notes "C-063 anchor; PR65/PR67 hard-pair basis, T4 only if H100 gain >0.00025"

.venv/bin/python -u experiments/line_search_pose_refinement.py \
  --archive-path experiments/results/lossless_repack_c059_brotli_q11m2w18_20260502/archive.zip \
  --metadata-path experiments/results/pr67_public_metadata_for_pose_search_20260502/metadata.json \
  --output-path experiments/results/c063_pr67_pr65_hardpair_basis_20260502/archive.zip \
  --output-metadata experiments/results/c063_pr67_pr65_hardpair_basis_20260502/metadata.json \
  --posenet-path upstream/models/posenet.safetensors --gt-mkv upstream/videos/0.mkv \
  --device cuda:0 --batch-size 16 --candidate-chunk 32 \
  --basis-delta-sets 'pair_window:1,2,3;dct:1,2' \
  --basis-pair-indices '164,64,130,112,97,153,70,198,420,289,166,435,78,156,418,87' \
  --basis-window-radius 1 --passes 2 --progress-every-candidates 64
```

Promotion gate: exact H100 score must beat C-063 by at least `0.00025` with
SegNet stable, then run identical bytes through T4/equivalent auth eval.

### 3. Learned/predictive mask grammar builder, not raw class RLE

Expected score reduction per wall-clock minute: lower immediate but large
strategic EV. CMG entropy probes show raw class/RLE/arithmetic-style coding is
byte-regressive versus PR67's `219472` byte mask stream; the next mask attempt
must be predictive, learned, residual, foveated, or decoder-aware.

Concrete next action:

```bash
.venv/bin/python -u experiments/plan_charged_mask_grammar_atoms.py \
  --source-archive reports/raw/leaderboard_intel_20260501/pr67_archive.zip \
  --output-dir experiments/results/charged_mask_grammar_atoms_pr67_decoded_20260502_codex \
  --decode-mask-array --decode-expected-frames 600
```

Then build a strict archive only if every learned table, selector, residual,
foveation parameter, and dictionary is charged inside `archive.zip`. First
milestone should be a closed builder plus decoded-mask SHA proof, not a
malformed-ZIP or source-constant clone.

## Deprioritize

- One-byte or scalar-only T4 promotions: C-057/C-058/C-059/C-063 show the
  surface is real but small, and top32 pose drift wasted a T4 confirmation.
- Global QZS block sweeps: b16/b24/b48/b64 and QZS4/block128 collapsed
  PoseNet on this basin.
- Raw PVR1 residual magnitude selectors: exact screens changed bytes without
  improving components.
- PR65 qpost wholesale or pairgated variants: measured H100 screens regressed
  PoseNet/SegNet.
- Public PR70/PR69 mechanics: malformed ZIPs, uncharged source constants, or
  script-payload movement are external forensics only, not contest-faithful
  score paths.

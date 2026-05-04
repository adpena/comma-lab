#!/usr/bin/env bash
set -euo pipefail

cd /workspace/pact

OUT_DIR=experiments/results/line_search_pr67_public_basis_hardpairs_20260502T0608Z
EVAL_DIR=experiments/results/archive_eval_pr67_public_basis_hardpairs_h100_20260502T0608Z
mkdir -p "$OUT_DIR" "$EVAL_DIR"

.venv/bin/python -u experiments/line_search_pose_refinement.py \
  --archive-path experiments/results/pr67_public_metadata_for_pose_search_20260502/archive.zip \
  --metadata-path experiments/results/pr67_public_metadata_for_pose_search_20260502/metadata.json \
  --output-path "$OUT_DIR/archive.zip" \
  --output-metadata "$OUT_DIR/metadata.json" \
  --gt-mkv upstream/videos/0.mkv \
  --posenet-path upstream/models/posenet.safetensors \
  --device cuda:0 \
  --batch-size 16 \
  --candidate-chunk 32 \
  --max-candidate-items 64 \
  --basis-delta-sets 'dct:1,2,3,5,8,13,21;pair_window:1,2,3,5' \
  --basis-modes '0,1,2,3,5,8,13,21,34' \
  --basis-pair-indices '164,64,130,112,97,153,70,198,420,289,166,435,78,418,87,159' \
  --basis-window-radius 2 \
  --passes 2 \
  --progress-every-candidates 50 \
  2>&1 | tee "$OUT_DIR/line_search.log"

export WORKSPACE=/workspace/pact
export ARCHIVE_PATH="/workspace/pact/$OUT_DIR/archive.zip"
export ARCHIVE_LABEL=pr67_public_basis_hardpairs_h100
export LOG_DIR="/workspace/pact/$EVAL_DIR"
export PREDICTED_LOW=0.25
export PREDICTED_HIGH=0.40
export CONTROLLED_BASELINE="PR67 public qpose14_qzs3_filmq9g_slsb1_r55 public-basin pose/manifold search; external attribution required"
export KEEP_EVAL_WORK=0
export REQUIRED_SOURCE_SHA256S='experiments/contest_auth_eval.py=1d9cc6e9a8a42aaeb1a80810b1c8e75ff7219ae4192b0e31c2e16385faf1c5e4
experiments/line_search_pose_refinement.py=c78a67cded296300d5c40e76b853a44dbd98ec3bd41c1301023576c4d64e674c
scripts/remote_archive_only_eval.sh=82d43478f6840a571e99163a0df4f7390cb39423a4f0a471c5d2de873ece67e6
submissions/robust_current/apply_qzs3_postprocess.py=83a06ca93119dea43b74a8b466b417a5f99d6a8d64d772fdc85e0c3ea155abe0
submissions/robust_current/inflate.sh=86449a1f52ac6b2be120d47287b8410f915dce7e562c69f480103f6e527c6017
submissions/robust_current/inflate_renderer.py=1bf64e9f055c88438c854d1e09f048c07c359177494da8e636079c62706b6472
submissions/robust_current/unpack_renderer_payload.py=cac8cde654f2d875d4567c18b77d573af91c29dbb0b05b7934dc7e019ae66f49
src/tac/profiles.py=07a54668f6d4053de80dd944ae123aedde44cf6def5baf53689e7f079888f897
src/tac/quantizr_faithful_renderer.py=6ab5d3b1cea06486386d9d37b3e257e1a4788219219b6fc830ab7bb885565b8e
src/tac/quantizr_qzs3_codec.py=d19ab43e974e203aa59f5850e2cff6f730cdd6bcc235b81c3120de8c29435d24'

bash scripts/remote_archive_only_eval.sh

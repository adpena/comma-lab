#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" -ne 3 ]]; then
  echo "Usage: inflate.sh <archive_dir> <output_dir> <video_names_file>" >&2
  exit 2
fi
PACT_REPO_ROOT="${PACT_REPO_ROOT:-/workspace/pact}"
PACT_PYTHON="${PYTHON:-python3}"
export C1_EXPECTED_ARCHIVE_SHA256=e4cd154f79a30e2b1d759af0d26e54444d22807f81700565e475392eae064f42
export C1_EXPECTED_PACKET_SHA256=aa1dbb5e2efff28cd0d31f5ee2a4b0575a248a27a431151bfcae64eb320d385b
export C1_EXPECTED_Y0_SHA256=5e86e419cdd5bd41c9482cabc78cf27cec22281098b64c715d91f1f067d11566
export C1_EXPECTED_Y1_SHA256=6a731946e3d9de82089c90de9784c5a5bc72c607c963fb6f79dac16f00ac89bc
export C1_CONTEST_PAIR_COUNT=600
export C1_CONTEST_CAMERA_HEIGHT=874
export C1_CONTEST_CAMERA_WIDTH=1164
export C1_CONTEST_SCORER_HEIGHT=384
export C1_CONTEST_SCORER_WIDTH=512
export C1_CONTEST_WORKERS=4
export C1_CONTEST_TEST_ONLY_SMALL_FIXTURE=0
exec "${PACT_PYTHON}" "${PACT_REPO_ROOT}/tools/measure_v10_two_plane_receiver_timing.py" contest-inflate "$1" "$2" "$3"

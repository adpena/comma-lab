#!/usr/bin/env bash
# PFP16 A++ exact-evidence helper.
#
# This helper evaluates the already-built Lane G v3 + PFP16 archive on a
# Lightning T4 CUDA/NVDEC host. It deliberately does NOT rebuild the archive:
# the fixed local bytes are uploaded, re-hashed remotely, and only then passed
# to experiments/contest_auth_eval.py --device cuda.
set -euo pipefail

readonly EXPECTED_SHA="0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f"
readonly EXPECTED_BYTES="686635"
readonly DEFAULT_ARCHIVE="experiments/results/lane_g_v3_pfp16/archive_lane_g_v3_pfp16.zip"
readonly DEFAULT_REMOTE_PACT="/home/zeus/content/pact"
readonly DEFAULT_REMOTE_UPSTREAM="/home/zeus/content/upstream"

MODE="run"
ARCHIVE="$DEFAULT_ARCHIVE"
REMOTE="${PFP16_A_PLUS_PLUS_REMOTE:-${REMOTE:-}}"
REMOTE_PACT="${REMOTE_PACT:-$DEFAULT_REMOTE_PACT}"
REMOTE_UPSTREAM="${REMOTE_UPSTREAM:-$DEFAULT_REMOTE_UPSTREAM}"
RUN_ID="pfp16_a_plus_plus_t4_$(date -u +%Y%m%dT%H%M%SZ)"
LOCAL_OUT_BASE="experiments/results/lane_g_v3_pfp16"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/pfp16_a_plus_plus_exact_t4_eval.sh [options]

Modes:
  --probe            Check remote reachability, repo paths, and GPU only.
  --run              Upload exact archive, run contest_auth_eval.py --device cuda, fetch evidence. Default.
  --fetch-only ID    Fetch evidence for an already-completed remote run id.
  --dry-run          Print resolved settings and exit before ssh/scp.

Options:
  --archive PATH           Local fixed archive path.
  --remote HOST            SSH host or alias. Defaults to $PFP16_A_PLUS_PLUS_REMOTE,
                           $REMOTE, then the canonical direct Lightning target. Do not rely on
                           bare ssh.lightning.ai for custody work.
  --remote-pact PATH       Remote pact repo. Default: /home/zeus/content/pact.
  --remote-upstream PATH   Remote upstream repo. Default: /home/zeus/content/upstream.
  --run-id ID              Stable output id. Default: pfp16_a_plus_plus_t4_<UTC>.
  --out-base PATH          Local evidence parent. Default: experiments/results/lane_g_v3_pfp16.

Safety contract:
  - No archive rebuild is performed.
  - Local and remote archive SHA256 must equal:
    0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f
  - Local and remote archive bytes must equal 686635.
  - Remote GPU name must contain T4.
  - contest_auth_eval.py must run with --device cuda.
  - Result JSON must report n_samples=600, device=cuda, gpu_t4_match=true,
    and the exact archive SHA/bytes above.
EOF
}

die() {
  echo "FATAL: $*" >&2
  exit 1
}

q() {
  printf "%q" "$1"
}

file_size() {
  stat -f%z "$1" 2>/dev/null || stat -c%s "$1"
}

sha256_file() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    sha256sum "$1" | awk '{print $1}'
  fi
}

resolve_remote_default() {
  if [ -n "$REMOTE" ]; then
    return
  fi
  die "set REMOTE or LIGHTNING_SSH_TARGET to a user-qualified Studio SSH target or SSH config alias"
}

verify_local_archive() {
  [ -f "$ARCHIVE" ] || die "archive not found: $ARCHIVE"
  local actual_sha actual_bytes
  actual_sha="$(sha256_file "$ARCHIVE")"
  actual_bytes="$(file_size "$ARCHIVE")"
  echo "[pfp16-a++] local_archive=$ARCHIVE"
  echo "[pfp16-a++] local_sha256=$actual_sha"
  echo "[pfp16-a++] local_bytes=$actual_bytes"
  [ "$actual_sha" = "$EXPECTED_SHA" ] || die "local archive SHA mismatch"
  [ "$actual_bytes" = "$EXPECTED_BYTES" ] || die "local archive byte size mismatch"
}

ssh_base() {
  ssh \
    -o BatchMode=yes \
    -o PasswordAuthentication=no \
    -o KbdInteractiveAuthentication=no \
    -o ConnectTimeout=20 \
    -o ConnectionAttempts=3 \
    -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=4 \
    -o TCPKeepAlive=yes \
    "$@"
}

scp_base() {
  scp \
    -o BatchMode=yes \
    -o PasswordAuthentication=no \
    -o KbdInteractiveAuthentication=no \
    -o ConnectTimeout=20 \
    -o ConnectionAttempts=3 \
    -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=4 \
    -o TCPKeepAlive=yes \
    "$@"
}

remote_probe() {
  local env_prefix
  env_prefix="REMOTE_PACT=$(q "$REMOTE_PACT") REMOTE_UPSTREAM=$(q "$REMOTE_UPSTREAM")"
  ssh_base "$REMOTE" "$env_prefix bash -s" <<'REMOTE'
set -euo pipefail
echo "[pfp16-a++] remote_host=$(hostname)"
echo "[pfp16-a++] remote_user=$(whoami)"
echo "[pfp16-a++] remote_pact=$REMOTE_PACT"
echo "[pfp16-a++] remote_upstream=$REMOTE_UPSTREAM"
[ -d "$REMOTE_PACT" ] || { echo "FATAL: REMOTE_PACT missing: $REMOTE_PACT" >&2; exit 10; }
[ -f "$REMOTE_PACT/env.sh" ] && source "$REMOTE_PACT/env.sh"
source /system/conda/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate cloudspace 2>/dev/null || true
[ -f "$REMOTE_PACT/experiments/contest_auth_eval.py" ] || { echo "FATAL: contest_auth_eval.py missing" >&2; exit 11; }
[ -f "$REMOTE_PACT/submissions/robust_current/inflate.sh" ] || { echo "FATAL: inflate.sh missing" >&2; exit 12; }
[ -f "$REMOTE_PACT/submissions/robust_current/config.env" ] || { echo "FATAL: config.env missing" >&2; exit 13; }
[ -d "$REMOTE_UPSTREAM" ] || { echo "FATAL: REMOTE_UPSTREAM missing: $REMOTE_UPSTREAM" >&2; exit 14; }
[ -f "$REMOTE_UPSTREAM/evaluate.py" ] || { echo "FATAL: upstream evaluate.py missing" >&2; exit 15; }
[ -f "$REMOTE_UPSTREAM/videos/0.mkv" ] || { echo "FATAL: upstream videos/0.mkv missing" >&2; exit 16; }
command -v nvidia-smi >/dev/null || { echo "FATAL: nvidia-smi missing" >&2; exit 17; }
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[pfp16-a++] gpu_name=$GPU_NAME"
case "$GPU_NAME" in
  *T4*) ;;
  *) echo "FATAL: A++ evidence requires a T4 GPU; got '$GPU_NAME'" >&2; exit 18 ;;
esac
if [ -x /opt/conda/bin/python ]; then
  PYBIN=/opt/conda/bin/python
else
  PYBIN="$(command -v python)"
fi
echo "[pfp16-a++] python=$PYBIN"
if command -v uv >/dev/null 2>&1; then
  echo "[pfp16-a++] uv=$(command -v uv)"
else
  echo "FATAL: uv missing; robust_current/inflate.sh requires uv" >&2
  exit 19
fi
echo "[pfp16-a++] PROBE_OK"
REMOTE
}

remote_run() {
  local remote_input_dir remote_archive remote_out env_prefix
  remote_input_dir="$REMOTE_PACT/auth_eval_input"
  remote_archive="$remote_input_dir/pfp16_exact_${EXPECTED_SHA:0:12}_archive.zip"
  remote_out="$REMOTE_PACT/experiments/results/lane_g_v3_pfp16/$RUN_ID"

  echo "[pfp16-a++] creating remote dirs"
  ssh_base "$REMOTE" "mkdir -p $(q "$remote_input_dir") $(q "$remote_out")"

  echo "[pfp16-a++] uploading exact archive"
  scp_base "$ARCHIVE" "$REMOTE:$remote_archive"

  env_prefix="REMOTE_PACT=$(q "$REMOTE_PACT") REMOTE_UPSTREAM=$(q "$REMOTE_UPSTREAM") RUN_ID=$(q "$RUN_ID") REMOTE_ARCHIVE=$(q "$remote_archive") REMOTE_OUT=$(q "$remote_out") EXPECTED_SHA=$(q "$EXPECTED_SHA") EXPECTED_BYTES=$(q "$EXPECTED_BYTES")"
  ssh_base "$REMOTE" "$env_prefix bash -s" <<'REMOTE'
set -euo pipefail

cd "$REMOTE_PACT"
source /system/conda/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate cloudspace 2>/dev/null || true
source "$REMOTE_PACT/env.sh" 2>/dev/null || true

if [ -x /opt/conda/bin/python ]; then
  PYBIN=/opt/conda/bin/python
else
  PYBIN="${PYBIN:-$(command -v python)}"
fi
export PYBIN
export PYTHONPATH="$REMOTE_PACT/src:$REMOTE_UPSTREAM:$REMOTE_PACT"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p "$REMOTE_OUT"

ACTUAL_SHA=$("$PYBIN" -c "import hashlib,sys; p=sys.argv[1]; h=hashlib.sha256(); f=open(p,'rb'); [h.update(c) for c in iter(lambda:f.read(1<<20), b'')]; print(h.hexdigest())" "$REMOTE_ARCHIVE")
ACTUAL_BYTES=$(stat -c '%s' "$REMOTE_ARCHIVE" 2>/dev/null || stat -f '%z' "$REMOTE_ARCHIVE")
{
  echo "archive_sha256=$ACTUAL_SHA"
  echo "archive_bytes=$ACTUAL_BYTES"
} | tee "$REMOTE_OUT/archive_sha256.txt"
test "$ACTUAL_SHA" = "$EXPECTED_SHA"
test "$ACTUAL_BYTES" = "$EXPECTED_BYTES"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "gpu_name=$GPU_NAME" | tee "$REMOTE_OUT/gpu.txt"
case "$GPU_NAME" in
  *T4*) ;;
  *) echo "FATAL: A++ evidence requires Tesla T4 provenance; got '$GPU_NAME'." >&2; exit 6 ;;
esac

bash "$REMOTE_PACT/scripts/probe_nvdec.sh" --ensure-dali

rm -rf "$REMOTE_OUT/eval_work"
"$PYBIN" -u experiments/contest_auth_eval.py \
  --archive "$REMOTE_ARCHIVE" \
  --inflate-sh submissions/robust_current/inflate.sh \
  --upstream-dir "$REMOTE_UPSTREAM" \
  --device cuda \
  --keep-work-dir \
  --work-dir "$REMOTE_OUT/eval_work" 2>&1 | tee "$REMOTE_OUT/auth_eval.log"

cp "$REMOTE_OUT/eval_work/contest_auth_eval.json" "$REMOTE_OUT/contest_auth_eval.json"
cp "$REMOTE_OUT/eval_work/provenance.json" "$REMOTE_OUT/eval_provenance.json"
cp "$REMOTE_OUT/eval_work/report.txt" "$REMOTE_OUT/report.txt"

"$PYBIN" - "$REMOTE_OUT/contest_auth_eval.json" <<'PY'
import json
import sys

EXPECTED_SHA = "0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f"
EXPECTED_BYTES = 686635

payload = json.load(open(sys.argv[1]))
prov = payload["provenance"]
assert payload["archive_size_bytes"] == EXPECTED_BYTES, payload["archive_size_bytes"]
assert prov["archive_sha256"] == EXPECTED_SHA, prov["archive_sha256"]
assert prov["archive_size_bytes"] == EXPECTED_BYTES, prov["archive_size_bytes"]
assert prov["device"] == "cuda", prov["device"]
assert prov.get("gpu_t4_match") is True, prov.get("gpu_model")
assert payload["n_samples"] == 600, payload["n_samples"]
print("PFP16_A_PLUS_PLUS_READY")
print(json.dumps({
    "final_score_reported_rounded": payload["final_score"],
    "score_recomputed_from_components": payload["score_recomputed_from_components"],
    "avg_posenet_dist": payload["avg_posenet_dist"],
    "avg_segnet_dist": payload["avg_segnet_dist"],
    "rate_unscaled": payload["rate_unscaled"],
    "archive_size_bytes": payload["archive_size_bytes"],
    "archive_sha256": prov["archive_sha256"],
    "gpu_model": prov.get("gpu_model"),
    "gpu_t4_match": prov.get("gpu_t4_match"),
}, indent=2, sort_keys=True))
PY
REMOTE
}

fetch_evidence() {
  local remote_out local_out
  remote_out="$REMOTE_PACT/experiments/results/lane_g_v3_pfp16/$RUN_ID"
  local_out="$LOCAL_OUT_BASE/$RUN_ID"
  mkdir -p "$local_out"
  echo "[pfp16-a++] fetching evidence to $local_out"
  ssh_base "$REMOTE" "cd $(q "$remote_out") && tar czf - auth_eval.log contest_auth_eval.json eval_provenance.json report.txt archive_sha256.txt gpu.txt" \
    | tar xzf - -C "$local_out"
  echo "[pfp16-a++] evidence=$local_out"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --probe)
      MODE="probe"
      shift
      ;;
    --run)
      MODE="run"
      shift
      ;;
    --fetch-only)
      MODE="fetch"
      RUN_ID="${2:?--fetch-only requires RUN_ID}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --archive)
      ARCHIVE="${2:?--archive requires PATH}"
      shift 2
      ;;
    --remote)
      REMOTE="${2:?--remote requires HOST}"
      shift 2
      ;;
    --remote-pact)
      REMOTE_PACT="${2:?--remote-pact requires PATH}"
      shift 2
      ;;
    --remote-upstream)
      REMOTE_UPSTREAM="${2:?--remote-upstream requires PATH}"
      shift 2
      ;;
    --run-id)
      RUN_ID="${2:?--run-id requires ID}"
      shift 2
      ;;
    --out-base)
      LOCAL_OUT_BASE="${2:?--out-base requires PATH}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      die "unknown argument: $1"
      ;;
  esac
done

resolve_remote_default

cat <<EOF
[pfp16-a++] mode=$MODE
[pfp16-a++] remote=$REMOTE
[pfp16-a++] remote_pact=$REMOTE_PACT
[pfp16-a++] remote_upstream=$REMOTE_UPSTREAM
[pfp16-a++] run_id=$RUN_ID
EOF

if [ "$MODE" != "fetch" ]; then
  verify_local_archive
fi

if [ "$DRY_RUN" = "1" ]; then
  echo "[pfp16-a++] dry-run complete; no ssh/scp/eval launched."
  exit 0
fi

case "$MODE" in
  probe)
    remote_probe
    ;;
  run)
    remote_probe
    remote_run
    fetch_evidence
    ;;
  fetch)
    fetch_evidence
    ;;
  *)
    die "invalid mode: $MODE"
    ;;
esac

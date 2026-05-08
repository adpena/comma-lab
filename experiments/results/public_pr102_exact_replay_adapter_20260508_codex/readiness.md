# PR102 Exact Replay Readiness - 2026-05-08

Status: `PASS`. Score claim: `false`. Dispatch attempted: `false`.

## Archive Custody

- Path: `experiments/results/public_pr102_hnerv_lc_v2_scale095_rplus1_custody_20260507_codex/public_pr102_intake_20260507_auto/archive.zip`
- Bytes: `178981`
- SHA-256: `afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641`
- Canonical URL: `https://github.com/user-attachments/files/27369164/archive.zip`

## Runtime Source Files

- Source root: `experiments/results/public_pr102_hnerv_lc_v2_scale095_rplus1_custody_20260507_codex/public_pr102_intake_20260507_auto/source/submissions/hnerv_lc_v2_scale095_rplus1`
- Source tree SHA-256: `cd96e89d0863ff0a3742746c10d9d7943ca5d8976058189d2b26ee080d046dc8`

| path | bytes | sha256 | role |
| --- | ---: | --- | --- |
| `README.md` | `639` | `d4ccd8c76013739a8f420110874adb320568b57e08c4163f76c6299f52b071ba` |  |
| `compress.sh` | `1316` | `8505b162f72ac7aa8a1a96b74f9904b5ff7eb3b79753024ab3c2c0311312cdc8` | fetches PR100 release archive and verifies afd53348f503... |
| `inflate.sh` | `705` | `aaccd126c0b1f411c81c8129add87c85f74c8f6177f1da290ee2b21645890247` | contest inflate entrypoint |
| `inflate.py` | `5110` | `6b7eb1fc1577c49378cd9837c8484ecbe0f46872b40576fec1291b27bd284a6c` | HNeRV decode, bicubic upsample, frame-0 red channel +1.0 nudge |
| `hnerv_model.py` | `2197` | `e63b04ad3df4942b9bc1e31afd8ec84177dfbe83827f67cf7c5a682b05c1b46b` |  |
| `schema.py` | `1081` | `bc434bd596e753dbeae97c0ddce4d9cf98a50cfe862a451834864d83620d6a0a` |  |
| `sidecar.py` | `1839` | `c6a7c56bf61a8cd8e1127141d5f65f23d76d8d4c3569395af7bbb35311b78144` | latent correction sidecar with DELTA_SCALE=0.0095 |

## Dependency And Network Risks

- Adapter policy: `fail_closed_preinstalled_dependencies_only; the adapter must exit before public runtime import if required modules are unavailable`
- Required preinstalled Python modules: `brotli, numpy, torch`

Manifest compliance risks:
- inflate.sh may install brotli at inflate time via pip if missing; exact replay adapter should preinstall or fail closed instead of relying on network.
- Archive payload is byte-identical to PR100; score movement is runtime-only, so archive-only intake can misclassify this as duplicate or no-op.
- Initial PR body referenced an in-tree archive that was later deleted; the maintainer comment attachment is the canonical public archive URL.
- A prior auto intake captured a wrong qpose release asset with member p; do not use it for PR102 replay.

Static network/install findings:
- `experiments/results/public_pr102_hnerv_lc_v2_scale095_rplus1_custody_20260507_codex/public_pr102_intake_20260507_auto/source/submissions/hnerv_lc_v2_scale095_rplus1/compress.sh:8`: `URL="https://github.com/BradyMeighan/comma_video_compression_challenge/releases/download/hnerv-lc-v2-archive/archive.zip"`
- `experiments/results/public_pr102_hnerv_lc_v2_scale095_rplus1_custody_20260507_codex/public_pr102_intake_20260507_auto/source/submissions/hnerv_lc_v2_scale095_rplus1/compress.sh:13`: `if command -v curl >/dev/null 2>&1; then`
- `experiments/results/public_pr102_hnerv_lc_v2_scale095_rplus1_custody_20260507_codex/public_pr102_intake_20260507_auto/source/submissions/hnerv_lc_v2_scale095_rplus1/compress.sh:14`: `curl -L --fail --silent --show-error "${URL}" -o "${TMP}"`
- `experiments/results/public_pr102_hnerv_lc_v2_scale095_rplus1_custody_20260507_codex/public_pr102_intake_20260507_auto/source/submissions/hnerv_lc_v2_scale095_rplus1/compress.sh:18`: `import urllib.request`
- `experiments/results/public_pr102_hnerv_lc_v2_scale095_rplus1_custody_20260507_codex/public_pr102_intake_20260507_auto/source/submissions/hnerv_lc_v2_scale095_rplus1/compress.sh:21`: `with urllib.request.urlopen(url) as r, open(out, "wb") as f:`
- `experiments/results/public_pr102_hnerv_lc_v2_scale095_rplus1_custody_20260507_codex/public_pr102_intake_20260507_auto/source/submissions/hnerv_lc_v2_scale095_rplus1/compress.sh:25`: `echo "ERROR: need curl or python3 to fetch upstream archive" >&2`
- `experiments/results/public_pr102_hnerv_lc_v2_scale095_rplus1_custody_20260507_codex/public_pr102_intake_20260507_auto/source/submissions/hnerv_lc_v2_scale095_rplus1/inflate.sh:15`: `python -c "import brotli" 2>/dev/null || pip install --quiet brotli`
- `experiments/results/public_pr102_hnerv_lc_v2_scale095_rplus1_custody_20260507_codex/public_pr102_intake_20260507_auto/source/submissions/hnerv_lc_v2_scale095_rplus1/inflate.py:26`: `subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'brotli'])`

## Exact Replay Command

Run only after ensuring dependencies are preinstalled and a remote dispatch claim is active:

```bash
.venv/bin/python experiments/contest_auth_eval.py \
  --archive experiments/results/public_pr102_hnerv_lc_v2_scale095_rplus1_custody_20260507_codex/public_pr102_intake_20260507_auto/archive.zip \
  --inflate-sh experiments/results/public_pr102_exact_replay_adapter_20260508_codex/inflate.sh \
  --upstream-dir upstream \
  --device cuda
```

## Adapter Materialization

- `experiments/results/public_pr102_exact_replay_adapter_20260508_codex/inflate.sh`
- `experiments/results/public_pr102_exact_replay_adapter_20260508_codex/README.md`
- `experiments/results/public_pr102_exact_replay_adapter_20260508_codex/runtime_source/submissions/hnerv_lc_v2_scale095_rplus1/README.md`
- `experiments/results/public_pr102_exact_replay_adapter_20260508_codex/runtime_source/submissions/hnerv_lc_v2_scale095_rplus1/compress.sh`
- `experiments/results/public_pr102_exact_replay_adapter_20260508_codex/runtime_source/submissions/hnerv_lc_v2_scale095_rplus1/inflate.sh`
- `experiments/results/public_pr102_exact_replay_adapter_20260508_codex/runtime_source/submissions/hnerv_lc_v2_scale095_rplus1/inflate.py`
- `experiments/results/public_pr102_exact_replay_adapter_20260508_codex/runtime_source/submissions/hnerv_lc_v2_scale095_rplus1/hnerv_model.py`
- `experiments/results/public_pr102_exact_replay_adapter_20260508_codex/runtime_source/submissions/hnerv_lc_v2_scale095_rplus1/schema.py`
- `experiments/results/public_pr102_exact_replay_adapter_20260508_codex/runtime_source/submissions/hnerv_lc_v2_scale095_rplus1/sidecar.py`

## Adapter Inflate Script

```bash
#!/usr/bin/env bash
set -euo pipefail

# Source-sized public replay shim for PR102. Do not install packages here.
# PACT_RUNTIME_DEPENDENCY_ROOT = experiments/results/public_pr102_exact_replay_adapter_20260508_codex/runtime_source/submissions/hnerv_lc_v2_scale095_rplus1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find_repo_root() {
  local dir="$HERE"
  while [ "$dir" != "/" ]; do
    if [ -f "$dir/experiments/contest_auth_eval.py" ]; then
      printf '%s\n' "$dir"
      return 0
    fi
    dir="$(dirname "$dir")"
  done
  return 1
}

if [ -n "${PACT_REPO_ROOT:-}" ]; then
  REPO_ROOT="$PACT_REPO_ROOT"
else
  REPO_ROOT="$(find_repo_root)" || {
    echo "ERROR: could not find repo root; set PACT_REPO_ROOT" >&2
    exit 1
  }
fi
PYTHON="${PACT_PYTHON:-$REPO_ROOT/.venv/bin/python}"
PUBLIC_SOURCE_ROOT="$REPO_ROOT/experiments/results/public_pr102_exact_replay_adapter_20260508_codex/runtime_source"
RUNTIME_SOURCE_ROOT="$REPO_ROOT/experiments/results/public_pr102_exact_replay_adapter_20260508_codex/runtime_source/submissions/hnerv_lc_v2_scale095_rplus1"

if [ "$#" -ne 3 ]; then
  echo "ERROR: expected DATA_DIR OUTPUT_DIR FILE_LIST arguments" >&2
  exit 2
fi

DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"

if [ ! -x "$PYTHON" ]; then
  echo "ERROR: Python not executable: $PYTHON" >&2
  exit 1
fi
if [ ! -d "$PUBLIC_SOURCE_ROOT" ] || [ ! -d "$RUNTIME_SOURCE_ROOT" ]; then
  echo "ERROR: PR102 public source runtime missing" >&2
  echo "PUBLIC_SOURCE_ROOT=$PUBLIC_SOURCE_ROOT" >&2
  echo "RUNTIME_SOURCE_ROOT=$RUNTIME_SOURCE_ROOT" >&2
  exit 1
fi

for module in brotli numpy torch; do
  "$PYTHON" - "$module" <<'PY'
import importlib.util
import sys

module = sys.argv[1]
if importlib.util.find_spec(module) is None:
    raise SystemExit(f"ERROR: required PR102 runtime dependency missing: {module}")
PY
done

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$PUBLIC_SOURCE_ROOT:$RUNTIME_SOURCE_ROOT:${PYTHONPATH:-}"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  SRC="$DATA_DIR/${BASE}.bin"
  DST="$OUTPUT_DIR/${BASE}.raw"
  if [ ! -f "$SRC" ]; then
    echo "ERROR: $SRC not found" >&2
    exit 1
  fi
  cd "$PUBLIC_SOURCE_ROOT"
  "$PYTHON" -m "submissions.hnerv_lc_v2_scale095_rplus1.inflate" "$SRC" "$DST"
done < "$FILE_LIST"
```

## Blockers

- None for source-sized adapter-plan readiness. Exact CUDA replay is still missing.

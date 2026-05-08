# PR102 Exact Replay Readiness - 2026-05-08

Status: `DISPATCHED`. Score claim: `false`. Dispatch attempted: `true`.

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

## Lightning Exact Eval Source Manifest Runbook

- Lane id: `pr102_public_exact_replay_t4`
- Job name: `pr102-hnerv-lc-v2-scale095-rplus1-exact`
- Normal path: claim first, then submit only through the repro wrapper so `source_manifest.json` is created, SHA-verified, and forwarded to the launcher.

Source manifest must include:
- `experiments/results/public_pr102_hnerv_lc_v2_scale095_rplus1_custody_20260507_codex/public_pr102_intake_20260507_auto/archive.zip`
- `experiments/results/public_pr102_exact_replay_adapter_20260508_codex/inflate.sh`
- `experiments/results/public_pr102_exact_replay_adapter_20260508_codex/README.md`
- `experiments/results/public_pr102_exact_replay_adapter_20260508_codex/runtime_source/submissions/hnerv_lc_v2_scale095_rplus1/README.md`
- `experiments/results/public_pr102_exact_replay_adapter_20260508_codex/runtime_source/submissions/hnerv_lc_v2_scale095_rplus1/compress.sh`
- `experiments/results/public_pr102_exact_replay_adapter_20260508_codex/runtime_source/submissions/hnerv_lc_v2_scale095_rplus1/inflate.sh`
- `experiments/results/public_pr102_exact_replay_adapter_20260508_codex/runtime_source/submissions/hnerv_lc_v2_scale095_rplus1/inflate.py`
- `experiments/results/public_pr102_exact_replay_adapter_20260508_codex/runtime_source/submissions/hnerv_lc_v2_scale095_rplus1/hnerv_model.py`
- `experiments/results/public_pr102_exact_replay_adapter_20260508_codex/runtime_source/submissions/hnerv_lc_v2_scale095_rplus1/schema.py`
- `experiments/results/public_pr102_exact_replay_adapter_20260508_codex/runtime_source/submissions/hnerv_lc_v2_scale095_rplus1/sidecar.py`
- `experiments/results/public_pr102_exact_replay_adapter_20260508_codex/readiness.json`
- `experiments/results/public_pr102_exact_replay_adapter_20260508_codex/readiness.md`

Claim before submit:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id pr102_public_exact_replay_t4 \
  --platform lightning \
  --instance-job-id pr102-hnerv-lc-v2-scale095-rplus1-exact \
  --agent ${AGENT_ID} \
  --predicted-eta-utc ${ETA_UTC} \
  --status eval \
  --notes PR102_exact_replay_archive_afd53348f503_no_score_claim
```

Submit through the wrapper:

```bash
.venv/bin/python scripts/lightning_exact_eval_repro.py \
  --job-name pr102-hnerv-lc-v2-scale095-rplus1-exact \
  --archive experiments/results/public_pr102_hnerv_lc_v2_scale095_rplus1_custody_20260507_codex/public_pr102_intake_20260507_auto/archive.zip \
  --inflate-sh experiments/results/public_pr102_exact_replay_adapter_20260508_codex/inflate.sh \
  --stage-workspace \
  --remote ${LIGHTNING_SSH_TARGET} \
  --studio ${LIGHTNING_STUDIO} \
  --teamspace ${LIGHTNING_TEAMSPACE} \
  --sdk-user ${LIGHTNING_SDK_USER} \
  --requirements-mode verify-only \
  --python-bin .venv/bin/python \
  --machine ${LIGHTNING_MACHINE:-g4dn.2xlarge} \
  --baseline-score ${EXACT_BASELINE_SCORE} \
  --predicted-band ${PREDICTED_LOW} ${PREDICTED_HIGH} \
  --regression-threshold ${REGRESSION_THRESHOLD} \
  --dispatch-lane-id pr102_public_exact_replay_t4 \
  --dispatch-claims-path .omx/state/active_lane_dispatch_claims.md \
  --extra-artifact experiments/results/public_pr102_exact_replay_adapter_20260508_codex/readiness.json \
  --extra-artifact experiments/results/public_pr102_exact_replay_adapter_20260508_codex/readiness.md \
  --queue-metadata source_prs=102 \
  --queue-metadata pr102_readiness=experiments/results/public_pr102_exact_replay_adapter_20260508_codex/readiness.json \
  --env INFLATE_TORCH_SPEC=torch==2.5.1+cu124 \
  --env UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124 \
  --env UV_INDEX_STRATEGY=unsafe-best-match \
  --component-trace \
  --component-trace-top-k 96 \
  --submit
```

Guardrails:
- claim lane before submit; do not use --allow-missing-dispatch-claim-reason for normal PR102 replay
- submit through scripts/lightning_exact_eval_repro.py with --stage-workspace so source_manifest is created and verified before launch
- do not require CUDA in the interactive Studio staging shell; the Batch runner performs the canonical CUDA preflight
- use concrete Lightning machine g4dn.2xlarge on comma-lab AWS unless a refreshed machine inventory proves another alias works
- keep --requirements-mode verify-only unless a separate remote environment build is explicitly recorded
- preserve readiness JSON/markdown as explicit extra artifacts in the staged source manifest

## Lightning Dispatch - 2026-05-08T10:15Z

Exact CUDA replay is now queued for drift/frontier classification only. This is
not a score claim until `contest_auth_eval.json` and adjudication artifacts are
harvested for the exact archive bytes.

- Active lane id: `pr102_public_exact_replay_t4`
- Active job: `pr102-public-exact-replay-g4dn2-20260508T101510Z`
- Lightning target: `adpena` / `comma-lab` / `lossy-compression-challenge`
- Machine request: `g4dn.2xlarge` (SDK reports job machine label `T4`)
- Archive SHA-256: `afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641`
- Archive bytes: `178981`
- Source manifest:
  `.omx/state/pr102-public-exact-replay-g4dn2-20260508T101510Z_manifest.json`
- Remote manifest verification: `OK`, `2187` files, `36576587` bytes
- Local supply-chain scan: `OK`, `violation_count=0`
- Launcher record:
  `.omx/state/pr102-public-exact-replay-g4dn2-20260508T101510Z_lightning_batch_record.json`
- Initial refreshed status: `Pending` at `2026-05-08T10:16:24Z`

Refused pre-submit attempts are preserved in
`.omx/state/active_lane_dispatch_claims.md`:

- `pr102-public-exact-replay-20260508T101327Z` closed as
  `refused_dispatch_stage_cuda_probe_no_batch_submitted`. The source manifest
  staged and verified, but the interactive Studio shell had no CUDA visible.
  That shell is not the score hardware; the Batch runner still performs the
  canonical CUDA preflight.
- `pr102-public-exact-replay-20260508T101414Z` closed as
  `refused_dispatch_lightning_t4_machine_alias`. The SDK rejected the `T4`
  alias on this AWS cluster before job creation, so the same packet was
  relaunched with concrete `g4dn.2xlarge`.

Harvest command when terminal:

```bash
.venv/bin/python scripts/launch_lightning_batch_job.py harvest-ssh \
  --job-name pr102-public-exact-replay-g4dn2-20260508T101510Z \
  --ssh-target s_01knw7wnzbe79wfq5mqqbx1mbz@ssh.lightning.ai \
  --expected-archive-sha256 afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641 \
  --expected-archive-size-bytes 178981 \
  --require-adjudication
```

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

#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
  echo "usage: $0 <slug> <config-env-path> [workspace-root] [run-root]" >&2
  exit 64
fi

slug="$1"
config_env_path="$2"
workspace_root="${3:-$HOME/pact-side}"
run_root="${4:-$HOME/bat00-runs}"
upstream_seed="${BAT00_UPSTREAM_SEED:-$workspace_root/workspace/upstream/comma_video_compression_challenge}"
ffmpeg_bin="${FFMPEG_BIN:-$HOME/tools/ffmpeg-btbn/current/bin/ffmpeg}"
ffprobe_bin="${FFPROBE_BIN:-$HOME/tools/ffmpeg-btbn/current/bin/ffprobe}"
uv_bin="${UV_BIN:-$HOME/.local/bin/uv}"
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
slug_root="$run_root/$slug"
run_dir="$slug_root/$run_id"
job_upstream="$run_dir/upstream"
job_workspace="$run_dir/pact-side"
job_config="$job_workspace/active-config.env"
summary_json="$run_dir/${slug}-smoke.json"
stderr_log="$run_dir/${slug}-smoke.stderr"
status_json="$run_dir/status.json"
manifest_json="$run_dir/manifest.json"
ledger_jsonl="$run_root/_ledger.jsonl"

mkdir -p "$run_dir"
mkdir -p "$run_root"
mkdir -p "$job_workspace"
ln -sfn "$run_id" "$slug_root/latest"

cp -a "$workspace_root/pyproject.toml" "$job_workspace/pyproject.toml"
cp -a "$workspace_root/uv.lock" "$job_workspace/uv.lock"
mkdir -p "$job_workspace/src" "$job_workspace/submissions"
cp -a "$workspace_root/src/comma_lab" "$job_workspace/src/comma_lab"
cp -a "$workspace_root/submissions/robust_current" "$job_workspace/submissions/robust_current"
cp -a "$config_env_path" "$job_config"

python3 - <<'PY' "$manifest_json" "$slug" "$run_id" "$config_env_path" "$job_config" "$workspace_root" "$job_workspace" "$upstream_seed" "$ffmpeg_bin" "$ffprobe_bin" "$uv_bin"
import json, sys
(path, slug, run_id, config_env_path, job_config, workspace_root, job_workspace, upstream_seed, ffmpeg_bin, ffprobe_bin, uv_bin) = sys.argv[1:]
with open(path, 'w') as f:
    json.dump({
        'slug': slug,
        'run_id': run_id,
        'source_config_env_path': config_env_path,
        'job_config_env_path': job_config,
        'source_workspace_root': workspace_root,
        'job_workspace_root': job_workspace,
        'upstream_seed': upstream_seed,
        'ffmpeg_bin': ffmpeg_bin,
        'ffprobe_bin': ffprobe_bin,
        'uv_bin': uv_bin,
        'started_at_utc': run_id,
        'pid': None,
        'status': 'starting',
    }, f, indent=2)
PY

python3 - <<'PY' "$status_json" "$slug" "$run_id"
import json, sys
path, slug, run_id = sys.argv[1:]
with open(path, 'w') as f:
    json.dump({'slug': slug, 'run_id': run_id, 'status': 'starting'}, f, indent=2)
PY

cp -a "$upstream_seed" "$job_upstream"
rm -rf "$job_upstream/videos"
ln -s "$upstream_seed/videos" "$job_upstream/videos"

export CONFIG_ENV_PATH="$job_config"
export FFMPEG_BIN="$ffmpeg_bin"
export FFPROBE_BIN="$ffprobe_bin"
export UV_BIN="$uv_bin"

python3 - <<'PY' "$status_json" "$slug" "$run_id" "running" "$$"
import json, sys
path, slug, run_id, status, pid = sys.argv[1:]
with open(path, 'w') as f:
    json.dump({'slug': slug, 'run_id': run_id, 'status': status, 'pid': int(pid)}, f, indent=2)
PY

set +e
(
  cd "$job_workspace"
  python3 -m src.comma_lab.cli smoke-submission robust_current \
    --package \
    --upstream-root "$job_upstream"
) >"$summary_json" 2>"$stderr_log"
rc=$?
set -e

python3 - <<'PY' "$status_json" "$slug" "$run_id" "$summary_json" "$stderr_log" "$manifest_json" "$ledger_jsonl" "$rc"
import json, os, sys
status_path, slug, run_id, summary_path, stderr_path, manifest_path, ledger_path, rc = sys.argv[1:]
rc = int(rc)
status = {'slug': slug, 'run_id': run_id, 'status': 'passed' if rc == 0 else 'failed', 'exit_code': rc, 'summary_json': summary_path, 'stderr_log': stderr_path}
try:
    if os.path.getsize(summary_path) > 0:
        with open(summary_path) as f:
            summary = json.load(f)
        status['all_passed'] = summary.get('all_passed')
        status['archive_path'] = summary.get('archive_path')
        if status.get('archive_path') and os.path.exists(status['archive_path']):
            status['archive_bytes'] = os.path.getsize(status['archive_path'])
        result0 = (summary.get('results') or [{}])[0]
        status['semantic_mae_mean'] = result0.get('semantic_mae_mean')
        status['semantic_mae_max'] = result0.get('semantic_mae_max')
except Exception as exc:
    status['summary_parse_error'] = str(exc)
with open(status_path, 'w') as f:
    json.dump(status, f, indent=2)
manifest = {}
try:
    with open(manifest_path) as f:
        manifest = json.load(f)
except Exception:
    pass
manifest.update(status)
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)
with open(ledger_path, 'a') as f:
    f.write(json.dumps(manifest, sort_keys=True) + '\n')
PY

exit "$rc"

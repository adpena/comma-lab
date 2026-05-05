---
name: scripts/remote_archive_only_eval.sh now self-bootstraps uv + ffmpeg + strips ._ resource forks (extincts 4 bug classes)
description: 2026-05-01 ~12:18 UTC. After 6 sequential infrastructure bugs across 4 destroyed Vast.ai instances (35956905, 35957332, 35958487, 35958897) burning ~$1.50 in chasing-not-fixing, hardened remote_archive_only_eval.sh Stage 0 to self-install uv + ffmpeg + auto-download BtbN master if system ffmpeg lacks scale options + strip macOS resource forks from upstream/. Next dispatch on a fresh Vast.ai pytorch:2.5.1-cuda12.4 image should "just work" with zero pre-staging.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

## The bug class progression (today's loop session)

Each subsequent Vast.ai dispatch hit a NEW Stage-0 dep failure that the previous fix didn't predict:

1. **uv not on PATH** (35956905) → fix: `curl -LsSf https://astral.sh/uv/install.sh | sh + symlink to /usr/local/bin/uv`
2. **CUDA driver too old** (35957332, driver 12070 < 12.4) → fix: `cuda_vers>=12.4` Vast.ai search filter
3. **ffmpeg missing** (apt install) → fix: `apt-get install -y ffmpeg`
4. **uv installs cu13 torch wheels by default** (PyPI default) → fix: `INFLATE_TORCH_SPEC=torch==2.5.1+cu124` + `UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124`
5. **macOS resource forks** (._0.mkv) in upstream/videos break contest_auth_eval validator → fix: `find upstream -name '._*' -delete`
6. **System ffmpeg 4.4.2 lacks scale option `in_primaries`** (RH/Ubuntu old) → fix: download BtbN ffmpeg master static build
7. **BtbN curl truncates** (network flakiness) → fix: retry-with-validation loop
8. **30GB disk too small** for 6-candidate chain (5GB uv torch + 6×3.6GB inflated frames = 27GB) → fix: 60GB instances + per-candidate cleanup

## What landed in scripts/remote_archive_only_eval.sh

New `bootstrap_runtime_deps()` function called BEFORE the contract-checks:

```bash
bootstrap_runtime_deps() {
    if ! command -v uv >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        ln -sf "$HOME/.local/bin/uv" /usr/local/bin/uv
    fi
    if ! command -v ffmpeg >/dev/null && [ ! -x /workspace/ffmpeg-btbn/bin/ffmpeg ]; then
        apt-get update && apt-get install -y ffmpeg
    fi
    find "$WORKSPACE/upstream" -name '._*' -delete   # macOS resource forks
}
```

And the `require_uv_and_ffmpeg_contract()` now AUTO-DOWNLOADS BtbN ffmpeg with retry if system ffmpeg lacks `in_primaries`/`in_color_matrix`/etc.

## Outstanding gaps (NOT yet automated)

- **Disk size**: must launch with `--disk 60` (was --disk 30); not yet enforced anywhere
- **Per-candidate cleanup**: chain driver should `rm -rf eval_work/{inflated,extracted,archive.zip}` after each eval (still in driver script)
- **CUDA driver version**: search filter `cuda_vers>=12.4` is operator-side, not script-side
- **Vast.ai instance preemption**: spot/interruptible instances can be auto-destroyed; no recovery — need persistent Modal volumes for chain artifacts

## How to apply

When dispatching a fresh chain on a bare Vast.ai pytorch:2.5.1-cuda12.4 image:

```bash
.venv/bin/vastai create instance $OFFER \
    --image 'pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel' \
    --disk 60 \
    --ssh
# (no pre-staging needed — Stage 0 self-bootstraps)
```

The next chain dispatch should run end-to-end without manual SSH-in to install deps.

## Cost of not fixing this earlier

6 sequential dispatches on 4 destroyed Vast.ai instances over ~30 min = ~$1.50 in idle GPU time + ~3 /loop ticks of forward velocity lost. Per CLAUDE.md "It is unacceptable to learn the same lesson twice" — and bug-whacking 6 versions of the same class IS learning the same lesson 6 times.

The permanent fix is +50 LOC in the wrapper. Should have been done after lesson #2.

## Related memories

- `feedback_uv_not_on_path_vast_instance_20260501.md` (bug #1)
- `feedback_vast_cuda_driver_too_old_silent_cpu_fallback_20260501.md` (bug #2)
- `feedback_vastai_dispatch_failures_20260501.md` (prior session's failure modes)
- `project_owv3_wave3_refinement_PLAN_20260501.md` (the chain that motivated this fix)

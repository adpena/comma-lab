---
name: Vast.ai instance bootstrap doesn't install uv — chain eval crashes at first eval
description: 2026-05-01 ~11:23 UTC. Wave-3 chain on instance 35956905 crashed with `RuntimeError: FATAL: uv is not on PATH` because submissions/robust_current/inflate.sh uses `uv run python ...` but the launch_lane_on_vastai.py phase2 path doesn't install uv. Fix is one-liner `curl -LsSf https://astral.sh/uv/install.sh | sh + symlink to /usr/local/bin/uv`. Permanent fix needed: phase2-extract should install uv when missing, OR scripts/remote_archive_only_eval.sh should install uv at Stage 0.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

The bug class: every Vast.ai instance has a different baseline env. Earlier Vast.ai instances (35952684, 35955469) had uv pre-installed via `setup_full.sh` (the full canonical bootstrap). The light `phase2-scp + phase2-extract` path I used today does NOT run setup_full — it only extracts the tarball + runs CUDA probe.

**Why:** `_ensure_uv_available()` in `experiments/contest_auth_eval.py:83` raises if `uv` not on PATH because `submissions/robust_current/inflate.sh` line 12 sets `UV_BIN="${UV_BIN:-uv}"` and does `$UV_BIN run python ...`.

**How to apply:** When using `launch_lane_on_vastai.py phase2-scp + phase2-extract` (lighter than full bootstrap), MUST manually install uv before any contest_auth_eval invocation:

```bash
ssh -p $PORT root@ssh3.vast.ai 'curl -LsSf https://astral.sh/uv/install.sh | sh
ln -sf $HOME/.local/bin/uv /usr/local/bin/uv'
```

Or add to `scripts/remote_archive_only_eval.sh` Stage 0:
```bash
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    ln -sf "$HOME/.local/bin/uv" /usr/local/bin/uv 2>/dev/null || true
fi
```

The latter is the durable fix — make `remote_archive_only_eval.sh` self-bootstrap uv. Permanent fix would also need preflight check `check_remote_archive_eval_self_installs_uv` to enforce.

Cost of this miss today: 1 minute crash + 1 minute fix + restart. Cheap because chain barely started. If chain had been deeper, would have lost up to 30 min of GPU time on a hung-then-failed evaluator chain.

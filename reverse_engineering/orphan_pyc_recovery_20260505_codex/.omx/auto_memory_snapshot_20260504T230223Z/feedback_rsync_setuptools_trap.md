---
name: rsync filter breaks setuptools (README missing)
description: rsync --include='*.py' --include='*.sh' filter excludes README.md, breaking setuptools editable install. Two failures across 4090 deploys.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Rule:** When rsyncing the repo to a remote for `pip install -e .`, NEVER use a narrow `--include='*.py'` filter alone. setuptools needs README.md (referenced in `pyproject.toml`'s `readme` field) AND any package_data files.

**Why:** 2026-04-26 deploys to 4090 #2 (Korea, dead) AND 4090 #4 (Cal) both failed at `pip install -e .` with:
```
File '/workspace/pact/src/tac/README.md' cannot be found
error in 'egg_base' option: 'src' does not exist or is not a directory
```
Even though the directory tree LOOKED right (.py files synced), setuptools requires the README and package_data the filter excluded. Wasted ~15 min debugging.

**How to apply:**
- For code-only sync (no install needed): `rsync -avz --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' --exclude='.venv' src/`
- For install-on-remote: ALSO scp `README.md` and `pyproject.toml` separately, OR use the broad rsync without `--include` filters.
- Reference impl in `scripts/remote_train_bootstrap.sh` and `scripts/remote_pose_tto_bootstrap.sh` — the bootstraps assume the upload is complete.

**Best practice:** rsync the WHOLE repo (excluding .venv/__pycache__/.git/results) in one shot. The filter saves nothing on a fast network and the failure mode is silent + delayed.

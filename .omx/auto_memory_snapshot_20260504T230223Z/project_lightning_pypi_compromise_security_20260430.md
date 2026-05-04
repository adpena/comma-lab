# Project Memory - Lightning PyPI Compromise Security Rule - 2026-04-30

## Durable Rule

Never install the PyPI package named `lightning` for this project or on remote
runners. Use `lightning-sdk` for Lightning AI Batch Jobs/CLI work.

## Incident Context

On 2026-04-30, public advisories reported that `lightning==2.6.2` and
`lightning==2.6.3` contained Mini Shai-Hulud style credential-stealing malware
that executes when `lightning` is imported. Treat any environment that installed
and imported those versions as compromised until isolated and credentials are
rotated.

Primary local review doc:

- `/Users/adpena/Projects/pact/.omx/research/lightning_pypi_compromise_security_review_20260430_codex.md`

## Local State

Local audit found:

- installed package is `lightning_sdk==2026.4.10`
- no local top-level `lightning` or `pytorch_lightning`
- no bad `lightning-2.6.{2,3}.dist-info`
- no repo Mini Shai-Hulud IOC path/hash hits
- no bad lockfile or pyproject pin
- downloaded `lightning_sdk==2026.4.10` wheel SHA256:
  `0988c96258d78ba00c40fa1d1326f9935a39e0f47e144ffc08427676ffc5ede5`
- wheel RECORD verification found zero mismatches
- no JS/MJS/package-json/.pth/_runtime artifacts in `lightning_sdk`
- installed `lightning_sdk` does not show reported Shai-Hulud behavior

## Controls Landed

- `src/tac/preflight.py`
  - `check_no_compromised_lightning_supply_chain(...)`
  - wired into `preflight_all(..., check_codebase=True)` at strict mode
  - blocks all repo dependency/install references to PyPI `lightning`, not just
    known bad versions
  - blocks `lightning --version` probes
- `src/tac/tests/test_preflight_meta_bugs.py`
  - tests for bad pins, range/extras/direct-url specs, bare install, safe
    `lightning-sdk`, bad dist-info, planted repo IOC paths, and unsafe version
    probes
- `src/tac/deploy/cloud_deploy.py`
  - install guidance corrected to `uv pip install lightning-sdk`
  - no longer executes `lightning --version`
- `src/tac/deploy/lightning/batch_jobs.py`
  - sets `LIGHTNING_DISABLE_VERSION_CHECK=1` before importing
    `lightning_sdk.Job`
- `AGENTS.md`
  - durable supply-chain rule added

## Open Item

Lightning Studio SSH scan from this host failed with `Permission denied
(publickey)`. When SSH auth is available, scan `/teamspace` and home paths for:

- `lightning-2.6.2.dist-info`
- `lightning-2.6.3.dist-info`
- `site-packages/lightning/_runtime`
- `.claude/router_runtime.js`
- `.claude/setup.mjs`
- `.vscode/setup.mjs`
- `.github/workflows/format-check.yml`
- `router_runtime.js`

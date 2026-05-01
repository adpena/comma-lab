# Lightning PyPI Compromise Security Review - 2026-04-30

## Trigger

User requested urgent review of:

- https://www.aikido.dev/blog/pytorch-lightning-pypi-compromise-mini-shai-hulud

## Advisory Facts Checked

Sources reviewed on 2026-04-30:

- Aikido: https://www.aikido.dev/blog/pytorch-lightning-pypi-compromise-mini-shai-hulud
- Socket: https://socket.dev/blog/lightning-pypi-package-compromised
- PyPI `lightning`: https://pypi.org/project/lightning/
- PyPI `pytorch-lightning`: https://pypi.org/project/pytorch-lightning/
- PyPI Shai-Hulud guidance: https://blog.pypi.org/posts/2025-11-26-pypi-and-shai-hulud/
- GitHub issue: https://github.com/Lightning-AI/pytorch-lightning/issues/21691

Consensus facts:

- Affected PyPI package: `lightning`, not `lightning-sdk`.
- Known malicious versions: `lightning==2.6.2` and `lightning==2.6.3`.
- PyPI currently marks the `lightning` project as quarantined.
- Reported trigger: payload starts when `lightning` is imported.
- Reported payload shape: hidden `lightning/_runtime`, `start.py`, and
  `router_runtime.js`.
- Known hashes:
  - `router_runtime.js` sha256
    `5f5852b5f604369945118937b058e49064612ac69826e0adadca39a357dfb5b1`
  - `start.py` sha256
    `8046a11187c135da6959862ff3846e99ad15462d2ec8a2f77a30ad53ebd5dcf2`
- Reported repo/worm indicators include `.claude/router_runtime.js`,
  `.claude/setup.mjs`, `.vscode/setup.mjs`,
  `.github/workflows/format-check.yml`, and npm `postinstall` hooks that run
  `setup.mjs`.

## Local Exposure Audit

Performed without importing `lightning`.

Findings:

- `.venv` contains `lightning_sdk==2026.4.10`.
- `.venv` does not contain top-level `lightning`, `pytorch_lightning`, or
  `lightning-2.6.{2,3}.dist-info`.
- `pyproject.toml` and `uv.lock` do not pin `lightning`, `pytorch-lightning`,
  or the known bad versions.
- `~/.cache/pip` had no filename hits for `lightning` or `pytorch_lightning`
  2.6.2/2.6.3.
- Repo scan found no known IOC hash matches.
- Repo scan found no `.claude/router_runtime.js`, `.claude/setup.mjs`,
  `.vscode/setup.mjs`, `.github/workflows/format-check.yml`, or npm
  `postinstall` hook to `setup.mjs`.
- One stale local instruction suggested `uv pip install lightning`; that is now
  fixed to `uv pip install lightning-sdk`.

Assessment:

- No local evidence of installed or cached compromised `lightning` package.
- No local repo indicator of Mini Shai-Hulud planting.
- No local credential rotation is indicated from this repo audit alone.
- If any other machine installed and imported `lightning==2.6.2` or
  `lightning==2.6.3`, treat that machine as compromised and rotate secrets.

## Changes Landed

- `src/tac/deploy/cloud_deploy.py`
  - Replaced stale install guidance with `uv pip install lightning-sdk`.
  - Added comment forbidding the compromised PyPI `lightning` path.
  - Removed `lightning --version` execution. Install state is now checked with
    `importlib.metadata.version("lightning-sdk")` so a poisoned `lightning`
    console script on `PATH` is not executed.

- `src/tac/preflight.py`
  - Added `check_no_compromised_lightning_supply_chain`.
  - Wires the check into `preflight_all(..., check_codebase=True)` at
    `strict=True`.
  - Blocks:
    - any dependency/install reference to PyPI package `lightning`
    - `lightning==2.6.2`
    - `lightning==2.6.3`
    - `pytorch-lightning==2.6.2`
    - `pytorch-lightning==2.6.3`
    - `pip install lightning` / `uv pip install lightning`
    - `lightning --version` probes
    - installed `lightning-2.6.{2,3}.dist-info`
    - hidden `site-packages/lightning/_runtime`
    - known IOC hashes
    - planted repo paths and npm postinstall-to-`setup.mjs` pattern
  - No waiver can suppress known bad pins.

- `src/tac/deploy/lightning/batch_jobs.py`
  - Sets `LIGHTNING_DISABLE_VERSION_CHECK=1` before importing
    `lightning_sdk.Job`, preventing non-essential PyPI version-check network
    traffic in promotion tooling.

- `src/tac/tests/test_preflight_meta_bugs.py`
  - Added regression tests for bad pins, any PyPI `lightning` dependency,
    range/extras/direct-url forms, split `uv.lock` records, bare install, safe
    `lightning-sdk`, installed bad dist-info, planted repo IOC paths, and
    unsafe `lightning --version`.

- `src/tac/tests/test_lightning_batch_jobs.py`
  - Added a test proving SDK import path sets `LIGHTNING_DISABLE_VERSION_CHECK`.

- `AGENTS.md`
  - Added durable Lightning supply-chain rule: never install PyPI `lightning`;
    use `lightning-sdk`; treat bad imported versions as compromise.
  - Added durable rule forbidding `lightning` CLI probes for install detection.

## `lightning_sdk==2026.4.10` Audit

Question: does installed `lightning_sdk` do anything similar to the compromised
PyPI `lightning` package?

Verdict: no evidence found.

Evidence:

- Installed package is `lightning_sdk==2026.4.10`, not `lightning`.
- Downloaded wheel:
  `/tmp/pact-lightning-sdk-audit/lightning_sdk-2026.4.10-py3-none-any.whl`
- Wheel SHA256:
  `0988c96258d78ba00c40fa1d1326f9935a39e0f47e144ffc08427676ffc5ede5`
- PyPI JSON for `lightning-sdk==2026.4.10` reports the same SHA256.
- RECORD verification: `1486` wheel entries, `0` hash mismatches.
- Installed-vs-wheel comparison: no byte differences in package files; only
  expected installer-generated console scripts / `INSTALLER` / `REQUESTED` /
  installed `RECORD` differ from the raw wheel.
- No installed top-level `lightning` or `pytorch_lightning` package directory.
- No `lightning_sdk` file matches known IOC hashes:
  - `5f5852b5f604369945118937b058e49064612ac69826e0adadca39a357dfb5b1`
  - `8046a11187c135da6959862ff3846e99ad15462d2ec8a2f77a30ad53ebd5dcf2`
- No `.js`, `.mjs`, package JSON, `.pth`, shell payload, `format-check.yml`,
  or hidden `_runtime` directory in the wheel scan.
- `entry_points.txt` exposes `lightning` and `lightning-sdk`, both targeting
  `lightning_sdk.cli.entrypoint:main_cli`.

Non-malware behavior to control:

- `lightning_sdk.__init__` performs a PyPI version check on import unless
  `LIGHTNING_DISABLE_VERSION_CHECK=1`.
- Lightning CLI command history logging exists in SDK CLI code. Avoid invoking
  the CLI from hermetic promotion paths unless required.

Recommendation:

- Do not treat the installed `lightning_sdk==2026.4.10` as compromised based on
  current evidence.
- Keep it pinned by wheel hash for now.
- Do not upgrade to `lightning-sdk==2026.4.23` or any newer wheel during the
  incident without repeating the same hash/IOC/RECORD audit.

## Verification

Passed:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_preflight_meta_bugs.py \
  src/tac/tests/test_lightning_batch_jobs.py \
  -q
# 225 passed

.venv/bin/python -m py_compile \
  src/tac/preflight.py \
  src/tac/deploy/cloud_deploy.py \
  src/tac/deploy/lightning/batch_jobs.py \
  src/tac/tests/test_preflight_meta_bugs.py
  src/tac/tests/test_lightning_batch_jobs.py

.venv/bin/python - <<'PY'
from tac.preflight import check_no_compromised_lightning_supply_chain
check_no_compromised_lightning_supply_chain(strict=True, verbose=True)
PY
# [lightning-supply-chain] OK
```

## Open Items

- 2026-04-30T19:14Z update: Lightning Studio SSH access is now available and
  the read-only IOC scan was run over the available Studio roots. No
  `lightning-2.6.2.dist-info`, `lightning-2.6.3.dist-info`, hidden
  `site-packages/lightning/_runtime`, known planted repo paths, or Mini
  Shai-Hulud repo artifacts were found.
- Local strict utility scan was recorded at
  `.omx/state/lightning_supply_chain_scan_20260430_codex.json`: status OK,
  zero violations, `lightning-sdk==2026.4.10`, no PyPI `lightning` or
  `pytorch-lightning` install.
- Re-run the reproducible utility before trusting any new remote runner:

```bash
.venv/bin/python scripts/scan_lightning_supply_chain.py \
  --strict \
  --quiet \
  --json-out .omx/state/lightning_supply_chain_scan_<date>.json
```

- Remote read-only scan template for newly created Lightning projects:

```bash
ssh s_01knw7wnzbe79wfq5mqqbx1mbz@ssh.lightning.ai '
set -euo pipefail
ROOTS=""
for d in /teamspace /home/$USER /home/zeus /root; do [ -d "$d" ] && ROOTS="$ROOTS $d"; done
find $ROOTS -type d \( -name "lightning-2.6.2.dist-info" -o -name "lightning-2.6.3.dist-info" -o -path "*/site-packages/lightning/_runtime" \) -print 2>/dev/null | sort
find $ROOTS -type f \( -path "*/.claude/router_runtime.js" -o -path "*/.claude/setup.mjs" -o -path "*/.vscode/setup.mjs" -o -path "*/.github/workflows/format-check.yml" -o -name "router_runtime.js" \) -print 2>/dev/null | sort
'
```

- Current Lightning SSH shell is not exact-eval ready because `nvidia-smi` is
  absent and the checked system Python has no CUDA torch. This is an execution
  readiness blocker, not a compromise finding.
- Recreate any Lightning/remote environment that ever installed
  `lightning==2.6.2` or `lightning==2.6.3`.
- Rotate GitHub/npm/cloud/Lightning/API credentials if any affected package was
  imported in an environment that held those secrets.

## 2026-04-30T21:49Z Follow-Up

- Re-ran the strict local scanner:
  `.omx/state/lightning_supply_chain_scan_20260430_codex_current.json`.
  Result: `status=OK`, `violation_count=0`, `lightning=null`,
  `pytorch-lightning=null`, `lightning_sdk=2026.4.10`.
- Verified installed local package metadata directly: only
  `lightning_sdk==2026.4.10` is installed; no suspicious `_runtime`,
  `router_runtime.js`, or `setup.mjs` file appears in that distribution.
- Patched `scripts/launch_lightning_batch_job.py` so
  `LIGHTNING_DISABLE_VERSION_CHECK=1` is set before any import path can touch
  `lightning_sdk`.
- Added regression coverage in `src/tac/tests/test_lightning_batch_jobs.py`.
- Current policy remains: use `lightning-sdk`, never bare PyPI `lightning`, and
  scan every runner before exact-eval trust.

## 2026-04-30T23:44Z Console-Script Hardening Follow-Up

Additional sources checked during this follow-up:

- Aikido:
  https://www.aikido.dev/blog/pytorch-lightning-pypi-compromise-mini-shai-hulud
- Socket:
  https://socket.dev/blog/lightning-pypi-package-compromised
- Snyk:
  https://snyk.io/blog/lightning-pypi-compromise-bun-based-credential-stealer/
- Semgrep:
  https://semgrep.dev/blog/2026/malicious-dependency-in-pytorch-lightning-used-for-ai-training/
- PyPI `lightning-sdk`:
  https://pypi.org/project/lightning-sdk/

Consensus remains: the reported malicious PyPI package is `lightning` versions
`2.6.2` and `2.6.3`; local env still has no PyPI `lightning` or
`pytorch-lightning`, and does have `lightning-sdk==2026.4.10`.

New hardening landed:

- `src/tac/preflight.py`
  - Supply-chain manifest scan now includes `tools/`.
  - Blocks `.venv/bin/lightning`, `venv/bin/lightning`, bare
    `lightning <subcommand>` at shell command boundaries, and
    `$LIGHTNING` / `${LIGHTNING}` executable variables.
  - Kept the pattern narrow enough to allow prose and package/module names like
    `lightning-sdk` and `lightning_sdk`.
- `tools/lightning_run.sh`
  - Replaced `.venv/bin/lightning connect studio` with SSH to
    `${LIGHTNING_SSH_TARGET:-lightning-pact}`.
- `tools/lightning_monitor.sh`
  - Replaced `lightning cp/list` calls with SSH/SCP against
    `${LIGHTNING_SSH_TARGET:-lightning-pact}` and
    `${LIGHTNING_REMOTE_TAC:-/teamspace/studios/this_studio/tac}`.
- `scripts/launch_lightning_batch_job.py`
  - `refresh-status` now infers SDK job name, teamspace, org, and user from the
    saved state record, avoiding retyped context during incident response or
    harvest monitoring.
- `AGENTS.md`
  - Durable rule added: do not call `.venv/bin/lightning`, bare `lightning`, or
    `$LIGHTNING` from operator scripts.

Verification:

```bash
.venv/bin/python scripts/scan_lightning_supply_chain.py \
  --json-out .omx/state/lightning_supply_chain_scan_20260430_codex_tools_hardened.json \
  --strict
# status=OK, violation_count=0

.venv/bin/python -m pytest \
  src/tac/tests/test_preflight_meta_bugs.py \
  src/tac/tests/test_lightning_batch_jobs.py \
  -q
# 265 passed

bash -n tools/lightning_run.sh tools/lightning_monitor.sh
.venv/bin/python -m py_compile \
  src/tac/preflight.py \
  scripts/launch_lightning_batch_job.py \
  src/tac/tests/test_preflight_meta_bugs.py \
  src/tac/tests/test_lightning_batch_jobs.py
```

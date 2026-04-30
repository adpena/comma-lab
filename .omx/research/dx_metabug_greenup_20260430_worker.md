# DX Metabug Greenup Worker Note - 2026-04-30

Scope: DX/preflight/metabug hardening and AGENTS/docs sync only. This note is
not a result ledger and records no score claims.

## Context Inspected

- `AGENTS.md` Lightning, component sensitivity, NWCS, and MCP guardrails.
- `src/tac/preflight.py` strict meta-bug checks and Lightning supply-chain
  guard.
- `scripts/lightning_repro_workspace.py` and Lightning Batch Job tests.
- `experiments/profile_component_sensitivity.py`,
  `experiments/build_component_sensitivity_manifest.py`, and the component
  sensitivity artifact tests.

## Findings

1. SDK hyphen artifact path mismatch already has a direct regression test:
   `src/tac/tests/test_lightning_batch_jobs.py::test_exact_eval_default_output_dir_matches_sdk_job_artifact_path`.
   The helper `lightning_sdk_job_name()` normalizes underscores to hyphens
   before composing `/teamspace/jobs/<job>/artifacts`.
2. Diagnostic sensitivity promotion still had a direct-builder gap:
   `profile_component_sensitivity.py` correctly marks its outputs
   non-promotable, but `build_component_sensitivity_manifest.py` could be
   invoked directly on those map/curve files and previously ignored embedded
   diagnostic metadata.
3. MCP shutdown policy was documented in AGENTS, but repo-owned MCP config
   reactivation was not covered by a preflight guard.

## Work Landed

- Added source-artifact diagnostic marker rejection to
  `experiments/build_component_sensitivity_manifest.py`.
  - Rejects component map `.pt` payloads and response-curve JSONs that declare
    `promotion_eligible=false`, `official_component_response=false`,
    non-empty `promotion_blockers`, diagnostic/proxy evidence grades, or
    smoke/fake/random/proxy markers.
  - This blocks direct assembly of promotable `component_sensitivity_v1`
    manifests from Fisher-proxy profile outputs.
- Added `check_no_active_mcp_server_config()` to `src/tac/preflight.py` and
  wired it into `preflight_all(strict=True)`.
  - Scans repo-owned `.codex`, `.claude`, `.cursor`, `mcp.json`, and
    `claude_desktop_config.json` config files.
  - Allows empty JSON `mcpServers` objects.
  - Rejects active JSON `mcpServers`/`mcp_servers`, TOML `mcp_servers`
    sections, and known MCP helper command tokens.
  - Accepts optional explicit config paths for one-off home-config audits
    without making home-directory scanning part of default repo preflight.
- Updated `AGENTS.md`.
  - Added Lightning Batch Job artifact path normalization rule.
  - Added repo-owned MCP config preflight rule.

## Tests Added

- `src/tac/tests/test_build_component_sensitivity_manifest.py`
  - Reject diagnostic profile sensitivity map metadata.
  - Reject diagnostic profile response curve metadata.
- `src/tac/tests/test_preflight_meta_bugs.py`
  - Empty JSON `mcpServers` passes.
  - Active JSON MCP server config fails.
  - Codex TOML `mcp_servers.*` section fails.
  - Live repo-owned MCP config scan is clean.

## Follow-Up

- Keep the existing Lightning SDK path test as the primary guard for the
  underscore-to-hyphen artifact path bug class.
- If home-level MCP configs need continuous enforcement, add an explicit
  operator-run wrapper that passes `config_paths`; do not make default
  repository preflight depend on user-specific home files.

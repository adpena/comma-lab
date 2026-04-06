# AI SLOP CLEANUP REPORT

## Scope

- `.gitignore`
- `src/comma_lab/cli.py`
- `src/comma_lab/install.py`
- `src/comma_lab/evaluate.py`
- `src/comma_lab/smoke.py`
- `submissions/robust_current/compress.sh`
- `submissions/robust_current/inflate.sh`
- `submissions/robust_current/analyze_roi.py`
- `submissions/robust_current/README.md`
- `docs/current_workflow_vs_rule_faithful.md`
- `docs/runbook_submission.md`
- `docs/compliance_audit.md`
- `reports/latest.md`
- `reports/scoring_rigor_review.md`
- `reports/ffmpeg_path_review.md`
- `reports/promotion_reviews/2026-04-06-av1-upscale-lanczos-promotion-review.md`
- `reports/writeup_working.md`
- `reports/graphs/evidence_index.md`
- `.omx/state/current_focus.md`
- `.omx/state/next_experiments.md`
- `.omx/research/findings.md`
- `.omx/notepad.md`
- `.omx/project-memory.json`
- `.ralph/run_log.md`

## Behavior Lock

- preserved the promoted Track B scorer-backed floor as the flat-path authority
- used smoke-gate evidence and targeted regression checks before and after hardening edits
- package/install/eval contract changes were verified with fresh CLI error-path and runtime checks

## Cleanup Plan

1. dead scratch and cache cleanup
2. duplicate / misleading accounting cleanup
3. naming / execution-contract cleanup
4. doc and durable-memory coherence cleanup

## Passes Completed

1. **Dead code / scratch cleanup**
   - removed transient `archive/` scratch from the source submission workflow
   - cleaned local `__pycache__` and scratch directories from editable paths
2. **Duplicate / misleading accounting cleanup**
   - tightened install payload accounting to the runtime payload actually under test
   - removed broader repo-local byte counting from the honest estimate path
3. **Naming / execution-contract cleanup**
   - fail-fast on unsupported `--package --no-sync`
   - fail-fast on AV1 + ROI instead of silently drifting into x265-only behavior
   - packaging now honors the requested upstream root
   - eval now clears stale `inflated/` output before scorer runs
4. **Test reinforcement / hardening cleanup**
  - made color/range assumptions explicit in encode/decode and smoke reference extraction
  - ensured ROI metadata analysis honors `FFMPEG_BIN` / `FFPROBE_BIN`
  - made `ROI_X_FRAC` and ROI-side `INFLATE_POSTFILTER` real live knobs
  - hardened temp-root handling so packaging falls back to `/tmp` when `TMPDIR` is stale
  - made the live 2.12 config self-contained in `config.env` and `config.av1-2.12.env`
  - synchronized the packet manifest and static-site export so published artifacts match source artifacts

## Quality Gates

- Regression tests: PASS
- Lint: N/A
- Typecheck / syntax: pass
- Tests / smoke: pass
- Static / security scan: N/A
- Fresh Track A regression: PASS (`0.00` @ `167` bytes)
- Fresh scorer-backed regression: PASS (`2.12` @ `864486` bytes)
- ROI targeted regressions: PASS
- Architect verification: PASS

## Changed Files

- see scope list above

## Remaining Risks

- no open blocker on the current flat-path production candidate
- ROI remains intentionally x265-only; AV1 + ROI is fail-fast by design, not silently unsupported
- current rule-faithful estimate is sensitive to explicit config payload size and must be refreshed whenever the live config file changes

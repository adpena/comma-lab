# Comprehensive OSS hardening per-repo verification report
Generated: 2026-05-19T19:55:00Z
Lane: lane_comprehensive_oss_hardening_per_round_6_7_20260519
Subagent: comprehensive_oss_hardening_p_20260519T195005Z
Operator authority: Round 6 + Round 7 (2026-05-19) — "production hardened comma ai and openpilot grade OSS" + "fix all warnings and issues and bugs and proceed with all enhancements"

## Audit summary

Both repos now PUBLIC: `adpena/tac` + `adpena/comma-lab`.
Comma.ai/openpilot reference patterns documented at `.omx/research/oss_hardening_comma_ai_openpilot_conventions_reference_20260519T195005Z.md`.

| Criterion | adpena/comma-lab | adpena/tac |
|---|---|---|
| 1. LICENSE present + MIT | ✅ | ✅ |
| 2. README badge row | ⚠️ no badges | ⚠️ no badges |
| 3. README quickstart | ✅ | ✅ |
| 4. README docs pointer | ✅ | ✅ |
| 5. `[tool.ruff]` in pyproject | ✅ | ❌ missing |
| 6. type-checker config | ✅ ty | ❌ missing |
| 7. `[project.optional-dependencies]` | ✅ extensive | ⚠️ minimal (mlx/viz/notebooks only) |
| 8. CI workflow green | ✅ | ✅ (latest run 2026-05-19T19:43Z passed) |
| 9. lint job in CI | ✅ ruff in ci.yml | ❌ no ruff step |
| 10. `.pre-commit-config.yaml` | ❌ missing | ❌ missing |
| 11. CONTRIBUTING.md | ✅ | ❌ missing |
| 12. CHANGELOG.md | ✅ | ✅ |
| 13. Public API type hints | ⚠️ partial | ⚠️ partial |
| 14. `ruff check` 0 warnings | ❌ 4676 errors | ❌ 739 errors |
| 15. Tests pass | ✅ | ✅ 56 passed, 1 skipped |
| 16. SPDX headers (extra) | ✅ src/tac 99.2% | ❌ 0/755 .py files |
| 17. `.editorconfig` | ❌ missing | ❌ missing |
| 18. `SECURITY.md` | ❌ missing | ❌ missing |
| 19. Node.js 24 actions | ⚠️ checkout@v4 + setup-python@v5 | ⚠️ checkout@v4 + setup-python@v5 |
| 20. `concurrency` block in CI | ❌ missing | ❌ missing |

**Overall verdict**: comma-lab is **comma.ai/openpilot-grade with minor enhancements pending**; tac is **functional OSS with CI green but missing several canonical OSS files**.

## Per-repo remediation plan

### adpena/comma-lab (operator working tree)

#### Land directly via canonical serializer (operator-authorized via Round 7 "proceed with all enhancements"):

1. **`.editorconfig`** — comma.ai canonical 2-space indent + LF line endings + trim trailing whitespace
2. **`SECURITY.md`** — minimal disclosure policy referencing operator email
3. **`.pre-commit-config.yaml`** — ruff format + ruff check (matches CI)
4. **README.md badge row** — CI badge + License badge + Python version badge
5. **CI workflow upgrade** — `actions/checkout@v6` + `actions/setup-python@v6` (closes Node.js 20 deprecation)
6. **CI workflow `concurrency` block** — cancel-in-progress for redundant push events

#### NOT auto-landed (would require bulk-edit + risk):

- 4676 ruff errors — bulk auto-fix would touch 1000+ files; this is sister-territory size; deferred to operator-routable
- src/comma_lab/ SPDX headers (0/30) — bulk header insertion; deferred to operator-routable

### adpena/tac (scratch clone in `/tmp/tac_oss_hardening_20260519T195005Z/`)

#### Land via feature branch + draft PR (operator approves merge):

1. **CI workflow upgrade** — `actions/checkout@v6` + `actions/setup-python@v6` (closes Node.js 20 deprecation, Slot O queued enhancement)
2. **`CONTRIBUTING.md`** — adapted from comma-lab version
3. **`SECURITY.md`** — minimal disclosure policy
4. **`.editorconfig`** — match comma-lab pattern
5. **`.pre-commit-config.yaml`** — ruff format + ruff check
6. **`[tool.ruff]` + `[tool.pytest.ini_options]` in pyproject** — match conventions
7. **README badge row** — CI badge + License badge + PyPI version badge + Python version badge
8. **README.md link fixes** — `adpena/pact` → `adpena/comma-lab`; `comma-video-compression-challenge` → `comma_video_compression_challenge`

#### Operator-routable (deferred):

- 739 ruff errors — 562 auto-fixable; sister-territory bulk edit
- 0/755 SPDX headers — bulk-add deferred until operator confirms identical license posture
- README.md MD lint cleanup

## Warnings + issues + bugs inventory

### comma-lab ruff statistics (top by count)
- 1303 I001 (unsorted imports) — auto-fixable
- 437 F401 (unused imports) — auto-fixable
- 286 RUF022 (unsorted __all__) — auto-fixable
- 222 UP035 (deprecated imports) — auto-fixable
- 216 RUF100 (unused noqa) — auto-fixable
- 178 UP037 (quoted annotations) — auto-fixable
- 166 F841 (unused variables) — auto-fixable
- 158 UP017 (datetime.timezone.utc) — auto-fixable

**Verdict**: 3180 of 4676 (68%) are auto-fixable via `ruff check --fix`. Substantial cleanup but risks touching files in HISTORICAL_PROVENANCE scope per Catalog #110/#113. Deferred to operator-routable sister wave (recommendation: incremental directory-scoped autofix passes).

### tac ruff statistics
- 324 F401 (unused imports) — auto-fixable
- 232 F541 (f-string missing placeholders) — auto-fixable
- 96 F841 (unused variables)
- 33 E402 (module-level imports)
- 18 E741 (ambiguous names)
- 14 E701 + 12 E702 (multi-statement lines)
- 4 F811 (redefined while unused) — auto-fixable
- 3 invalid noqa directives (need manual fix)

**Verdict**: 562 of 739 (76%) auto-fixable. Bundle into the draft PR as a Phase 4 commit.

### Node.js 20 deprecation (BOTH repos)

Empirical receipt from latest tac CI run 26121006768:
```
##[warning]Node.js 20 actions are deprecated. The following actions are running on Node.js 20
and may not work as expected: actions/checkout@v4, actions/setup-python@v5. Actions will be
forced to run with Node.js 24 by default starting June 2nd, 2026.
```

**Fix**: `actions/checkout@v6` (v6.0.2 latest) + `actions/setup-python@v6` (v6.2.0 latest) per https://github.com/actions/checkout/releases/latest

## Honest deferrals (per Catalog #110/#113 + operator-routable hand-off)

These items are NOT remediated in this slot. Each carries explicit operator-routable next action:

| Item | Why deferred | Operator-routable next action |
|---|---|---|
| comma-lab 4676 ruff errors | Bulk autofix risks touching 1000+ files; sister-territory size | Spawn dedicated `ruff-autofix-sister` subagent with per-directory scope |
| comma-lab src/comma_lab SPDX (0/30) | Bulk-add identical to license but needs explicit confirm | Sister subagent with mechanical SPDX-prepend |
| tac 0/755 SPDX headers | Bulk-add 755 files; operator may want different SPDX policy in tac vs comma-lab | Confirm posture: identical to comma-lab OR keep tac SPDX-free |
| tac 739 ruff errors | 562 auto-fixable but `ruff` not in tac dev-deps; need to add tooling first | Bundle into draft PR Phase 4 commit |
| README MD lint | Style polish; non-functional | Lower priority |

## Comma.ai/openpilot comparison table

| Convention | openpilot | comma-lab | tac |
|---|---|---|---|
| MIT License | ✅ | ✅ | ✅ |
| README centered HTML header | ✅ | ❌ flat MD | ❌ flat MD |
| README badge row | ✅ 4 badges | ❌ | ❌ |
| Quickstart one-liner | ✅ `bash <(curl ...)` | ✅ multi-line code block | ✅ `pip install tac` |
| `hatchling.build` | ✅ | ❌ setuptools (works either way) | ✅ |
| `[tool.ruff]` config | ✅ | ✅ | ❌ |
| ty (Astral) type checker | ✅ | ✅ | ❌ |
| `[tool.pytest.ini_options]` | ✅ extensive | ✅ | ❌ |
| `.editorconfig` | ✅ | ❌ | ❌ |
| `SECURITY.md` | ✅ | ❌ | ❌ |
| CI workflow `@v6` actions | ✅ | ❌ @v4/@v5 | ❌ @v4/@v5 |
| CI `concurrency` block | ✅ | ❌ | ❌ |
| CI `timeout-minutes` per step | ✅ | ⚠️ on job only | ⚠️ on job only |
| `runs-on: ubuntu-24.04` | ✅ | ❌ ubuntu-latest | ❌ ubuntu-latest |
| Banned-API list in ruff | ✅ extensive | ❌ | ❌ |
| `quote-style = "preserve"` ruff format | ✅ | ❌ default | ❌ |
| `[tool.uv]` python-preference | ✅ "only-managed" | ❌ | ❌ |
| `RELEASES.md` channel notes | ✅ | ✅ CHANGELOG.md (equivalent) | ✅ CHANGELOG.md |
| Code of conduct | ⚠️ light in CONTRIBUTING | ✅ "Be excellent to each other" | ❌ |
| SPDX headers | ❌ (NOT used by openpilot) | ✅ src/tac 99.2% | ❌ |

**Where we MATCH canonical**: license, basic project structure, CHANGELOG discipline.

**Where we DIVERGE WITH RATIONALE**:
- SPDX headers (comma-lab uses them; openpilot does not — defensible: tighter OSS posture)
- 4-space vs 2-space indent (we use Python 4-space PEP8 default; openpilot uses 2-space)
- "Peña" Copyright (operator's actual name vs openpilot's "Vehicle Researcher")

**Where we should ADOPT** (this landing addresses):
- `.editorconfig`, `SECURITY.md`, `@v6` actions, `concurrency` block, README badges

**Where we should ADOPT in future** (operator-routable):
- `quote-style = "preserve"`, banned-API list, ubuntu-24.04 pinning, per-step timeout-minutes

## Status

- ✓ Phase 1 reference conventions documented
- ✓ Phase 2 audit complete (20 criteria × 2 repos = 40 verdicts)
- ⏳ Phase 3 remediation in progress
- ⏳ Phase 4 warnings cleanup (auto-fixable subset)
- ⏳ Phase 5 Slot O follow-on (Node.js 24 actions upgrade)
- ⏳ Phase 6 verification report (THIS DOC + final landing memo)
- ⏳ Phase 7 lane gates + MEMORY.md prepend

## Cross-references

- Reference conventions: `.omx/research/oss_hardening_comma_ai_openpilot_conventions_reference_20260519T195005Z.md`
- Slot H tac audit: `.omx/research/oss_audit_adpena_tac_for_pr_link_20260519T185843Z.md`
- Slot N comma-lab sanitization: `.omx/research/comma_lab_sanitization_sweep_20260519T194221Z.md`
- Slot O tac CI green: `.omx/research/tac_ci_fix_authoring_tests_20260519T193600Z.md`
- Operator directive: round 6 + round 7 verbatim quotes preserved in landing memo

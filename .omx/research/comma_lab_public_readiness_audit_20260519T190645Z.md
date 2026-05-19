# comma-lab public-readiness audit (Slot K successor — operator directive 2026-05-19)

**Date**: 2026-05-19T19:06:45Z (UTC)
**Operator directive**: *"link to all of the relevant resources a comma ai employee will need for this to be useful in production and for evaluation and review and reserach - both in comma-lab and in tac and make sure both are live and updated on origin/main"* + *"super public with our codebases"*
**Sister to**: Slot H (`a4fdeabcf4dce8417`) tac-only audit, commit `66a0a6aad`
**Captured at**: `.omx/research/operator_directive_pr_body_stealth_skunkworks_comprehensive_provenance_20260519T184500Z.md`

## Verdict

**FAIL_NEEDS_REMEDIATION → OPERATOR_DECISION_REQUIRED**

`adpena/comma-lab` is currently PRIVATE. The operator's new directive requires it be PUBLIC and linkable from the PR body. Audit finds 1 BLOCKING criterion (visibility), 4 SUBSTANTIVE disclosure-risk classes that need remediation BEFORE making public, and 10 PASSING criteria.

## Repo state (per `gh repo view adpena/comma-lab`)

| Field | Value |
|---|---|
| URL | https://github.com/adpena/comma-lab |
| **Visibility** | **PRIVATE** |
| Default branch | `main` |
| Created | 2026-04-09T18:25:26Z |
| Last pushed | 2026-05-19T18:56:37Z (current within 30 min of audit) |
| Disk usage | 1,950,088 KB (~1.9 GB) |
| Description | "Task-aware codec research for the comma.ai lossy video compression challenge" |
| Archived | false |

origin/main SHA (per `git ls-remote`): `8a9e72198fe7f9a40ccb55a154d25f650cf424ee`

Local HEAD: `8bc07a92615e1e9ffc0828fa41b53985c390864c` (3 commits AHEAD of origin/main — sister-subagent in-flight work).

## 15-criterion checklist

| # | Criterion | Verdict | Notes |
|---|---|---|---|
| 1 | Repo visibility (goal: PUBLIC) | **FAIL** | Currently PRIVATE; operator decision required to flip |
| 2 | LICENSE file present (MIT preferred) | **PASS** | MIT License present; copyright 2026 Alejandro (Alex) Peña |
| 3 | README.md quality | **PASS** | 233 lines, well-structured, clear evidence grades, package map, terminology, quick-start, methodology |
| 4 | No leaked credentials/API keys | **PASS** | 1 test-fixture `sk-proj-abcdefghij1234567890XYZ` in `test_build_tac_oss_release_packet.py` is by design (validates the sanitization gate itself) |
| 5 | No leaked private infrastructure URLs | **PASS-WITH-NOTE** | No Cloudflare/Modal/Lightning URLs found in tracked source; minor mentions of `modal.com/usage` in CLAUDE.md (operator-facing docs only) |
| 6 | No leaked local absolute paths | **PARTIAL FAIL** | 1,436 tracked files contain `/Users/adpena/` paths; 99% in `experiments/results/`, `reports/raw/`, `.omx/research/`, `.omx/logs/` (historical artifacts). 17 in `src/`/`tools/`/`scripts/`/`tests/` are docstring/comment references (not active code) |
| 7 | No raw provider logs | **PARTIAL FAIL** | `.omx/logs/` contains tracked operator-session JSONL transcripts (14 files); `reports/raw/` contains 326 files with historical execution logs |
| 8 | `.omx/state/` properly gitignored | **PASS** | `.gitignore` line 24 excludes `.omx/state/*` with explicit allowlist for 14 canonical posterior/ledger files (intentional public state); confirmed no `.omx/state/vastai_active_instances.json` / `instance_setup_first_seen.json` leaks |
| 9 | CLAUDE.md suitable for public disclosure | **PARTIAL** | CLAUDE.md is 875,194 bytes and contains operator-facing non-negotiables (Tailscale IPs row + operator-routable infrastructure refs). It's tracked. Operator should decide: (a) publish as-is (most transparent; comma.ai would learn implementation discipline); (b) publish sanitized variant; (c) untrack |
| 10 | CONTRIBUTING.md | **PASS** | Present (3,516 bytes) |
| 11 | pyproject.toml valid | **PASS** | Present (12,638 bytes) |
| 12 | No private operator metadata | **PARTIAL FAIL** | Tailscale IPs (100.81.85.28 + 100.120.99.124 + 100.114.131.54 + 100.65.24.39) in 4 files (`scripts/bat00.py`, `scripts/bat00_wsl_setup.ps1`, `.omx/research/domain_exploitation_catalog_20260509.md`, `.omx/research/z7_mamba2_substrate_design_memo_20260518.md`) AND CLAUDE.md fleet table |
| 13 | No account credentials | **PASS** | No HF tokens, Modal token IDs, AWS keys, GHA tokens found in tracked source via regex scan |
| 14 | Code style consistent | **PASS** | Ruff + mypy + pytest discipline visible; multiple preflight Catalog gates enforce hygiene |
| 15 | Repo size reasonable | **PARTIAL** | 1.9 GB is large for public hosting; `.git` history is the bulk; the working tree is bounded but historical experiment artifacts inflate it. Comma.ai's openpilot repo is also large (similar scale); not a blocker per se |

## Disclosure risk categorization

### CLASS A — BLOCKING (operator decision required before public flip)

- **Visibility**: `gh repo edit adpena/comma-lab --visibility public` requires explicit operator approval per CLAUDE.md "Executing actions with care" non-negotiable.
- **Tailscale fleet table in CLAUDE.md (lines 1648-1657)**: lists 5 internal lab machine names + Tailscale IPs + GPU details + bat00 connection strings. Per CLAUDE.md "Public Disclosure Hygiene" non-negotiable: "Keep credentials, private infrastructure URLs, **local absolute paths**, raw provider logs, **unpublished operator state**, **and account metadata** out of GitHub/docs/site/public supplement surfaces." A reasonable reading: the fleet table is "operator state" / "account metadata" and should be sanitized OR moved to a non-tracked file.

### CLASS B — SUBSTANTIVE but non-blocking (operator may accept as-is per "stealth + skunkworks" tone directive)

- **1,436 tracked files with `/Users/adpena/` paths** in historical artifacts. The Catalog #208 STRICT preflight gate covers ONLY `docs/`; the gate's `_DOCS_LOCAL_PATH_PATTERNS` regex matches `/Users/<word>/` (and 4 sister patterns) AND `_DOCS_LOCAL_PATH_SCAN_PATHS = (REPO_ROOT / "docs",)` — so `experiments/results/` + `reports/raw/` + `.omx/research/` + `.omx/logs/` are OUT-OF-SCOPE. Operator-route options:
  - (a) Untrack `experiments/results/`, `reports/raw/`, `.omx/logs/` via `git rm --cached -r` + add to `.gitignore` (would shrink visible repo significantly; preserves history in `.git` but makes browsing cleaner)
  - (b) Sanitize via `find ... -exec sed -i ''` (replace `/Users/adpena/Projects/pact/` with `<repo-root>/` everywhere); preserves traceability but requires backfill
  - (c) Accept as-is per "stealth + skunkworks" tone directive (operator name = `adpena` is already public via the GitHub handle; path leaks reveal `/Users/adpena/` = operator's $HOME, which is information-theoretically equivalent to knowing the GitHub username)
- **Tailscale IPs in 4 non-CLAUDE.md tracked files**: `scripts/bat00.py:BAT00_IP = os.environ.get("BAT00_IP", "100.120.99.124")` is operator-side default (env-overridable; not a credential); the 3 research-memo references describe lab infrastructure context.
- **bat00 WSL2 setup script + connection strings**: `scripts/bat00_wsl_setup.ps1` documents how to connect to bat00; this is documentation of the operator's home lab. Not credentials but operationally specific.
- **3 `experiments/hf_jobs_*.py` files reference `/Users/adpena/.claude/plugins/cache/.../scripts/sam_segmentation_training.py`** as a path provenance comment. Inert at runtime.

### CLASS C — PASSES current Catalog #208 + sister-gate audit

- 8 `docs/` files with `/Users/adpena/` paths (Catalog #208 currently warn-only; backfill pending) — these are within the canonical structural-protection scope.
- All credentials surfaces (HF / Modal / AWS / Anthropic) verified absent in tracked source.
- All `.omx/state/*` operator-private files properly gitignored.

## Per-disclosure-class operator-route options

### Option 1: PUBLIC AS-IS (highest transparency, fastest)

Pros:
- Matches operator's "super public with our codebases" directive verbatim
- Comma.ai employee can review FULL research trajectory (which is the operator's stated goal)
- "Operator name + Tailscale IPs known" reveals nothing they couldn't reverse-engineer from GitHub PRs
- Compliance with CLAUDE.md "Public Disclosure Hygiene" maintained because credentials + private infrastructure URLs + .omx/state operator-only files are NOT exposed

Cons:
- 1.9 GB initial clone
- 1,436 files with `/Users/adpena/` paths (cosmetic / no security impact since adpena is already the public GitHub handle)
- Tailscale IPs visible (cosmetic / Tailscale ACL is the actual security boundary, not IP visibility)

### Option 2: SANITIZE THEN PUBLIC (recommended balance)

- Pre-flip sanitization sweep:
  - Bulk `sed` rewrite of `/Users/adpena/Projects/pact/` → `<repo-root>/` across tracked files
  - Replace Tailscale IPs in `scripts/bat00*` + `scripts/bat00.py` with `<bat00-tailscale-ip>` placeholder
  - Add waivers OR remove the Tailscale fleet table from CLAUDE.md (CLAUDE.md is tracked; sanitized variant could ship)
- Then: `gh repo edit adpena/comma-lab --visibility public`
- Estimated effort: 1-2 hours sed-based rewrite + commit per Catalog #110 APPEND-ONLY discipline (where applicable) + a single bulk-rewrite commit per Catalog #230 sister-subagent ownership map
- Trade-off: preserves comma.ai's stated production-review needs (full code, full docs, full history) while sanitizing operator-side cosmetic leaks

### Option 3: UNTRACK HISTORICAL + PUBLIC (smallest public surface)

- `git rm --cached -r experiments/results/ reports/raw/ .omx/logs/` (do NOT delete files; just untrack)
- Add to `.gitignore`
- `git commit` the untracking
- Shrinks repo footprint ~30-40% (historical experiment artifacts are bulk of tracked size)
- Loses public traceability of historical experiment evidence (which the operator + comma.ai may actually want for reproducibility)
- NOT recommended unless operator prioritizes minimal public surface

## Recommended remediation (operator decision)

**PRIMARY RECOMMENDATION**: Option 2 (sanitize then public). The operator's stated intent — "super public" + "comma.ai employee can use for production and evaluation and review and research" — argues for KEEPING the historical artifacts (they are the research record per CLAUDE.md "Lab notebook's version control") while sanitizing operator-cosmetic leaks (paths + Tailscale IPs + bat00 connection strings + fleet table).

**Bulk sanitization tool**: `find . -type f \( -name "*.md" -o -name "*.py" -o -name "*.sh" -o -name "*.jsonl" -o -name "*.json" \) -not -path './.git/*' -not -path './.venv/*' -exec sed -i '' 's|/Users/adpena/Projects/pact/|<repo-root>/|g; s|/Users/adpena/|<operator-home>/|g' {} +`

**Specific blocker fixes** (must land BEFORE public flip):
1. CLAUDE.md Tailscale fleet table: replace 5 Tailscale IPs with `<tailscale-ip-redacted>` placeholders + add waivers
2. `scripts/bat00*` files: change `BAT00_IP` default to `<bat00-host>` (operator must set env var)
3. CLAUDE.md operator-personal notes: surface a `# PUBLIC_DISCLOSURE_OK:<reason>` waiver for each line that is intentionally public (e.g. the contest score formula citations)

**Operator approval gates**:
- `gh repo edit adpena/comma-lab --visibility public` requires explicit operator approval (NEVER auto-flip per CLAUDE.md "Executing actions with care")
- Bulk-rewrite commit requires Catalog #230 ownership map declaration + Catalog #340 sister-checkpoint guard
- Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA": the contest submission depends on linkability; visibility flip should land BEFORE PR submission (if not, the PR body MUST omit comma-lab links per Slot H Gap 2's original recommendation)

## Cross-references

- Sister memo: `.omx/research/operator_directive_pr_body_stealth_skunkworks_comprehensive_provenance_20260519T184500Z.md` (operator's 4 mid-flight directives that prompted this audit)
- Prior tac audit: Slot H verdict at commit `66a0a6aad` (`adpena/tac` PASSES with 2 gaps)
- CLAUDE.md "Public Disclosure Hygiene" non-negotiable (the canonical contract this audit verifies against)
- Catalog #208 (`check_docs_no_local_absolute_paths`) — same META class scoped to docs/ only

— Claude-main (Slot K successor `comma_lab_audit_tac_gap_remediation_20260519T190645Z`) 2026-05-19T19:06:45Z

# OSS v0.2.0-rc1 Release Prep Audit

- **Lane**: `lane_oss_v0_2_0_rc1_release_prep_audit_20260514` (Phase 6.0, L0 SKETCH)
- **Audit date (UTC)**: 2026-05-14
- **HEAD at audit start**: `9b2e7162ad6e2df65ce435d887b9299953e1bb12`
- **Scope (per CLAUDE.md mutation frontier)**: `LICENSE`, `THIRD_PARTY_NOTICES.md`, `README.md`, `pyproject.toml`, `.gitignore`, `docs/**` (public-facing), `reports/release_v0_2_0_rc1_audit_20260514.md` (this file), plus `reports/lane_maturity_v0_2_0_rc1_audit.md` (regenerated audit table).
- **Out-of-scope (sister-subagent fences honored)**: `submissions/*/inflate.py` (F1-INFLATE-REFACTOR), `src/tac/substrates/sar_coherent_pose_pairs/**` (SAR-TRAINER-DEBUG), `tools/subagent_checkpoint.py` (CRASH-RESUME), `tools/build_pr101_nonlocal_sweep_packets.py` (sister-subagent untracked WIP), pinned upstream snapshot.
- **GPU spend**: $0 (audit only).
- **Promotion verdict**: AUDIT ONLY — no LICENSE / THIRD_PARTY_NOTICES.md / upstream edits. No tag.

## Audit verdicts at a glance

| Section | Verdict | Findings |
|---|---|---|
| 1. License + Third-party notices | NEEDS_OPERATOR_REVIEW | 2 gaps surfaced — LICENSE copyright line + THIRD_PARTY_NOTICES.md missing several runtime deps |
| 2. Public disclosure hygiene | NEEDS_SANITIZATION | 2 tracked docs (`docs/superpowers/**`) contain hardcoded `/Users/adpena/Projects/pact/...` paths |
| 3. Tac library OSS readiness | SAFE | `src/tac/` is clean of Claude/OMX/provider policy at this audit pass |
| 4. Public-frontier hygiene | SAFE | `reverse_engineering/` clean; Catalog #109 STRICT passes (0 violations) |
| 5. Catalog roster integrity | 1 PRE-RELEASE BLOCKER | Catalog #185 (`check_strict_flipped_catalog_entries_have_live_count_zero`) reports 1 drift: Catalog #158 surfaced 1 violation from an untracked sister-subagent file |
| 6. Lane maturity roster | SAFE (informational) | 600 lanes validated clean; broader surprise findings deferred to follow-up audit |
| 7. Release manifest (this doc) | LANDED | path: `reports/release_v0_2_0_rc1_audit_20260514.md` |

## Section 1 — License + Third-party notices

### LICENSE

- `LICENSE` exists, MIT, 21 lines.
- **GAP (NEEDS_OPERATOR_REVIEW)**: copyright line reads `Copyright (c) 2026 OpenAI artifact output for user-directed scaffold`. For an OSS release under Alejandro Peña's authorship (as declared in `pyproject.toml`), this line should be updated to reflect the actual copyright holder. Per CLAUDE.md mutation frontier this file requires explicit operator approval before edit — flagged here for operator to action.

### THIRD_PARTY_NOTICES.md

- `THIRD_PARTY_NOTICES.md` exists, 13 lines.
- Current contents enumerate upstream challenge repos + research inspirations (Ralph, autoresearch, DSPy/GEPA, Mojo) but do NOT enumerate the bundled-dependency tree.
- **GAP (NEEDS_OPERATOR_REVIEW)**: `pyproject.toml` declares hard runtime deps (`torch`, `pydantic`, `numpy`, `click`, `brotli`, `constriction`, `pyppmd`, `cryptography`, `cmaes`, `optuna`) and `tac[runtime]` extras (`av`, `safetensors`, `opencv-python`, `timm`, `einops`, `segmentation-models-pytorch`) plus `tac[cloud]` (`lightning-sdk`, `modal`, `vastai`, `kaggle`). None are enumerated in `THIRD_PARTY_NOTICES.md` with their licenses. Per the CLAUDE.md mutation frontier this file requires explicit operator approval before edit — flagged here for operator to action.

## Section 2 — Public disclosure hygiene scan

### Credential / API key scan

- Scan pattern: `ghp_*`, `sk-*`, `AKIA*` against every tracked file outside `.venv/`, `workspace/`, `reverse_engineering/`.
- **Hit**: `src/tac/tests/test_preflight_meta_bugs.py` contains `sk-thisIsNotARealTokenButItIsLongEnough12345`.
- **Verdict**: SAFE — this is intentional fixture data exercising the credential-leak detector. The literal is a placeholder, not a real secret. No mitigation required.

### Local absolute path scan (`/Users/adpena`)

- Scan: tracked files only (excluding `.omx/`, `.venv/`, `workspace/`, `reverse_engineering/`).
- **Hits**:
  - `docs/superpowers/plans/2026-04-10-anti-drift-runtime-hardening.md` — 3 hardcoded `/Users/adpena/Projects/pact` paths in shell-command examples.
  - `docs/superpowers/specs/2026-04-10-anti-drift-runtime-design.md` — multiple hardcoded `/Users/adpena/Projects/pact/...` paths in artifact-link markdown.
  - `src/tac/tests/test_preflight_meta_bugs.py` — fixture rows quoting `/Users/adpena` paths inside detector test data (intentional, SAFE).
- **Verdict**: NEEDS_SANITIZATION for the two `docs/superpowers/**` files. Sister-tests are fixture-only and not a leak.

### Private infra URL scan

- Scan pattern: `modal.com`, `lightning.ai/studio`, `vast.ai/instance`, `cloudflare.com/api` across `src/tac`, `docs`, `README`, `LICENSE`, `THIRD_PARTY_NOTICES`, `pyproject.toml`, `reports/`.
- **Hit**: none in tracked public-facing files in scope.
- **Verdict**: SAFE.

### CRITICAL_LEAK count

- **0 CRITICAL_LEAK findings.**

## Section 3 — Tac library OSS readiness

- `src/tac/` exists and contains only reusable codec/runtime/preflight library code at this audit pass.
- The `tac` package directly imports `tac.preflight` as the canonical strict-mode preflight surface; the surface is documented inline via catalog rows in CLAUDE.md.
- Catalog #109 (`check_public_pr_intake_clones_pristine`) passes STRICT @ 0 — `reverse_engineering/` and `experiments/results/public_pr*_intake_*/` clones are byte-identical to upstream PR heads.
- `src/comma_lab/` is the canonical home for research-state custody / provider ledgers; the `tac` ↔ `comma_lab` split per CLAUDE.md is intact.
- **Verdict**: SAFE.

## Section 4 — Public-frontier hygiene

- `reverse_engineering/` is tracked with its own `.gitignore` + `public_pr_waiver_manifest.json` (3.0 KB) describing waivers for vendored material.
- `reverse_engineering/orphan_pyc_recovery_20260505_codex/`, `reverse_engineering/pr95_hnerv/`, `reverse_engineering/public_frontier/`, and `reverse_engineering/public_pr102_pr108_intake_20260508/` are present and pristine per Catalog #109 STRICT.
- No raw PR clones or model weights are tracked in this surface (the heavy artifacts live in untracked `experiments/results/` per `.gitignore` rules + audit research-state policy).
- **Verdict**: SAFE.

## Section 5 — Catalog roster integrity

Four gates run for this section:

| Gate | Catalog # | Result |
|---|---|---|
| `check_claude_md_catalog_no_duplicate_numbers` | #118 | 0 violations — SAFE |
| `check_claude_md_catalog_text_matches_preflight_strict_value` | #159 | 0 violations — SAFE |
| `check_legacy_allowlist_backfill_cadence_ledger_current` | #183 | 0 violations — SAFE (ledger `.omx/research/legacy_allowlist_backfill_cadence_20260513.md` within 30-day window) |
| `check_strict_flipped_catalog_entries_have_live_count_zero` | #185 | **1 violation — PRE-RELEASE BLOCKER** |

### Catalog #185 violation detail

- Drift entry: Catalog #158 (`check_deterministic_compiler_canonical_use`) — CLAUDE.md row claims "Live count: 0" + STRICT, but the gate currently returns 1 violation.
- Sample violation: `tools/build_pr101_nonlocal_sweep_packets.py` — untracked sister-subagent (presumed PR101-NONLOCAL or OPTIMIZATION-AUDIT) packet-builder writes archive.zip + emits inflate runtime without the canonical deterministic-compiler AST proof.
- **Routing**: This file is OUTSIDE this audit's mutation frontier (sister-subagent untracked WIP). The fix is one of: (a) introducing subagent routes the builder through `tac.packet_compiler.deterministic_compiler` / `tools/build_deterministic_packet.py`; (b) the subagent adds a `# DETERMINISTIC_COMPILER_BYPASS_OK:<reason>` waiver if intentional; (c) CLAUDE.md Catalog #158 row is updated to reflect a new non-zero live count.
- **Pre-release impact**: BLOCKER for v0.2.0-rc1 tag. The catalog table is the operator-facing strictness manifest; shipping a release with a known #185 drift is a self-protection failure per the CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable.

### CLAUDE.md catalog row count

- 119 numbered catalog rows present at HEAD (counted via `grep -E "^[0-9]+\. \`check_"`).
- Catalog #199 / #201 / #202 / #203 / #204 / #205 / #206 are all present at HEAD with descriptive rows; Catalog #200 is not present as a row at HEAD (gap — not necessarily a bug; the catalog allows numbering gaps; flagging informationally).

## Section 6 — Lane maturity roster

- `python tools/lane_maturity.py validate` returns "OK — 600 lane(s) validated cleanly".
- Audit table regenerated to `reports/lane_maturity_v0_2_0_rc1_audit.md` (634 lines).
- The audit surfaces many L1 SCAFFOLD lanes (most recent in Phase 6.0 / Phase 5.0 / Phase 4.5 / Phase 4.0). A deeper "SKETCH-for-30-days" audit is deferred to a follow-up pass; this release prep treats the lane registry as informational-clean.
- **Verdict**: SAFE (informational only).

## Outstanding pre-release blockers (must zero before v0.2.0-rc1 tag)

1. **Catalog #185 drift on Catalog #158** — see Section 5. Owner: introducing sister-subagent of `tools/build_pr101_nonlocal_sweep_packets.py`. Action: route through canonical deterministic compiler OR waive OR update CLAUDE.md row.
2. **`docs/superpowers/**` local-path sanitization** — see Section 2. Owner: operator OR a docs-sanitization subagent. Action: replace `/Users/adpena/Projects/pact/...` paths with relative paths or env-var placeholders.

## Outstanding NEEDS_OPERATOR_REVIEW items (not strict blockers, but should land before v0.2.0)

1. **LICENSE copyright line** — currently reads `Copyright (c) 2026 OpenAI artifact output for user-directed scaffold`. Per the mutation frontier this file requires explicit operator approval; operator to action.
2. **THIRD_PARTY_NOTICES.md dep enumeration** — current file lists only research inspirations + upstream challenge repo, not the runtime dep tree (torch / pydantic / numpy / click / brotli / constriction / pyppmd / cryptography / cmaes / optuna / av / safetensors / opencv-python / timm / einops / segmentation-models-pytorch / lightning-sdk / modal / vastai / kaggle). Operator decision: enumerate (with each dep's license) OR add a one-line note pointing users to `pyproject.toml` and individual upstream license files.

## Operator-routable decisions for v0.2.0-rc1 tag

| ID | Decision | Recommended action |
|---|---|---|
| D-1 | Resolve Catalog #185 drift on Catalog #158 | Route `tools/build_pr101_nonlocal_sweep_packets.py` through canonical deterministic compiler OR sister-subagent adds bypass waiver OR CLAUDE.md row updated. PRE-RELEASE BLOCKER. |
| D-2 | Sanitize `docs/superpowers/plans/2026-04-10-anti-drift-runtime-hardening.md` + `docs/superpowers/specs/2026-04-10-anti-drift-runtime-design.md` | Replace hardcoded `/Users/adpena/Projects/pact/...` with relative paths or `${REPO_ROOT}/...`. PRE-RELEASE BLOCKER per CLAUDE.md "Public Disclosure Hygiene" non-negotiable. |
| D-3 | LICENSE copyright line | Operator approves canonical copyright string; mutation-frontier file requires explicit approval. NEEDS_OPERATOR_REVIEW. |
| D-4 | THIRD_PARTY_NOTICES.md dep enumeration | Operator decides whether to enumerate the runtime dep tree with licenses, or add a one-line pointer to `pyproject.toml`. NEEDS_OPERATOR_REVIEW. |
| D-5 | Catalog #200 numbering gap | Operator decides if #200 should be backfilled (was a numbered catalog #200 ever published in this CLAUDE.md but later removed?) or if the gap is intentional. INFORMATIONAL. |
| D-6 | Lane maturity follow-up audit | Schedule a deeper SKETCH-for-30-days + missing-memory-file audit as a sister lane. INFORMATIONAL. |

## 6-hook wire-in declaration (per CLAUDE.md "Subagent coherence-by-default")

1. **Sensitivity-map contribution**: N/A — META audit pass; not a representation/codec lane.
2. **Pareto constraint**: N/A — no archive bytes, no charged-byte contribution.
3. **Bit-allocator hook**: N/A — no per-tensor importance changes.
4. **Cathedral autopilot dispatch hook**: N/A — no dispatchable candidate emitted.
5. **Continual-learning posterior update**: N/A — no empirical anchor (no archive bytes, no contest score).
6. **Probe-disambiguator**: N/A — single-interpretation release-prep audit; no competing design hypotheses surfaced.

All 6 hooks declared N/A with rationale per the non-negotiable.

## Memory + lane

- Memory file: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_oss_v0_2_0_rc1_release_prep_audit_landed_20260514.md` (outside repo per CLAUDE.md).
- Lane: `lane_oss_v0_2_0_rc1_release_prep_audit_20260514` registered at L0 SKETCH (Phase 6.0); mark `impl_complete` + `memory_entry` after commit lands.

## Companion artifact

- `reports/lane_maturity_v0_2_0_rc1_audit.md` — full audit table generated by `python tools/lane_maturity.py audit`.

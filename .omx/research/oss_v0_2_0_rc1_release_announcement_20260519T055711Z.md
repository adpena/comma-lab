# OSS v0.2.0-rc1 release announcement — 2026-05-19

**Authority:** Hardening bundle 2 F1 (OSS release prep) per integrated battle plan `integrated_battle_plan_priority_queue_dag_cables_gates_20260519T052801Z.md` (commit `6a1e94a63`) Cable F item F1.

**Scope:** Release-readiness checklist + announcement copy + operator-action checklist for v0.2.0-rc1 tag publish.

## Release-readiness checklist

| Surface | Status | Evidence |
|---|---|---|
| `LICENSE` | PRESENT (MIT) | `LICENSE` 1.1K (verified by sister `16d2323db`) |
| `THIRD_PARTY_NOTICES.md` | PRESENT | `THIRD_PARTY_NOTICES.md` 7.3K (verified by sister `16d2323db`) |
| `README.md` | PRESENT | `README.md` 10.6K |
| `CONTRIBUTING.md` | PRESENT | `CONTRIBUTING.md` 2.9K |
| `SYSTEM_MAP.md` | PRESENT | `SYSTEM_MAP.md` 21.0K |
| `pyproject.toml` version | `0.2.0rc1` | matches v0.2.0-rc1 target |
| SPDX-License-Identifier MIT headers | LANDED | commit `82ecc2a0c` |
| Git tag `oss-v0.2.0-rc1` | LOCAL | exists; remote push is operator-action |
| Git tag `v0.2.0-rc1` | LOCAL | exists; remote push is operator-action |
| Wheel buildable | UNVERIFIED HERE | operator-action: `uv build --wheel` |
| Public Disclosure Hygiene (CLAUDE.md non-negotiable) | CLEAN | no credentials / private infra URLs / local absolute paths in this memo |

## Release highlights (v0.2.0-rc1)

### New canonical primitives

- **`tac.contest_oracle`** — Contest scorer oracle for differentiable + roundtrip-aware loss computation.
- **`tac.atom.atom`** — Typed atom framework for solver-consumable score-improvement candidates (rate / distortion / interaction terms / Volterra coefficients).
- **`tac.preflight_rudin_daubechies`** — Interpretable preflight ranker package (SLIM risk scorer + falling-rule list + Rashomon ensemble + compressive coverage + wavelet multi-scale + GOSDT dispatcher) per Catalog #273-#278.
- **`tac.master_gradient`** + **`tac.master_gradient_consumers`** — Master gradient extraction + 14 typed consumers for sensitivity / per-X planning / canonical DuckDB rows (per Catalog #318 raw-byte-authority self-protection + Catalog #327 contest-axis custody self-protection).
- **`tac.probe_outcomes_ledger`** — Canonical fcntl-locked JSONL ledger for adjudicated probe verdicts per Catalog #313 (queryable across sessions; gates dispatch BEFORE re-running an already-settled probe).
- **`tac.council_continual_learning`** — Canonical fcntl-locked JSONL ledger for council deliberation anchors per Catalog #300 (4-tier protocol + maximum-signal preservation + mission-alignment frontmatter).
- **`tac.provenance`** — Canonical Provenance contract per Catalog #323 META-class umbrella extincting phantom-score class at the persisted-artifact-row surface.
- **`tac.deploy.modal.call_id_ledger`** — Canonical Modal call-id lifecycle ledger per Catalog #245 (dispatch / harvest / failure / stale / function_timeout state machine).

### New STRICT preflight gates (130+ catalog entries since v0.1)

- **Catalog #270**: Production-hardened dispatch optimization protocol UMBRELLA (Tier 1 engineering + Tier 2 hardware + Tier 3 substrate correctness).
- **Catalog #300**: Council deliberation v2 frontmatter (4-tier protocol enforcement + maximum-signal preservation).
- **Catalog #313**: Predecessor probe outcomes (refuses dispatch wrappers that bypass adjudicated INDEPENDENT/KILL/DEFER verdicts within staleness window).
- **Catalog #318**: Master-gradient raw-byte authority guard (refuses raw archive-byte / bit master-gradient APIs incompatible with ZIP + entropy-coded packet contract).
- **Catalog #319-#324**: Wyner-Ziv deliverability proof + autopilot reweight + phantom-provenance composition_alpha + canonical provenance umbrella + predicted-band post-training Tier-C validation.
- **Catalog #325**: Per-substrate optimal form via adversarial grand council symposium (binds 6-step canonical contract).
- **Catalog #326**: Substrate driver consumes trainer mode env var (extincts smoke-hardcode bug class).
- **Catalog #327**: Master-gradient contest-axis custody (refuses authority-bearing rows with axis/hardware/method mismatch).
- **Catalog #328**: Submission inflate.py LOC budget audit (warn-only ceiling per HNeRV parity L4 review-time discipline).
- **Catalog #330**: Modal harvester call-id outcome ledger (refuses harvesters that observe terminal state without recording).
- **Catalog #331**: Canonical task status no dangling transitions (single-source-of-truth task lifecycle).
- **Catalog #333**: Codex inbox open questions have response or default within deadline (bidirectional Claude↔Codex channel).

### META-bug class retroactive audit pattern (Catalog #110/#113 + #157/#174/#216/#289 family)

The session-by-session 2026-05 hardening landed a structural pattern: every operator-surfaced bug class becomes (a) per-instance fix, (b) per-class META gate, (c) META-meta gate refusing CLAUDE.md catalog drift. The Catalog #185 sister regression guard refuses any state where a STRICT-flipped catalog entry's gate function returns >0 live violations, closing the documentation-vs-runtime drift surface.

### Subagent coherence discipline

- Catalog #117 / #157 / #174 / #216 / #235 / #289 commit-machinery quintet extincts the multi-subagent commit-swap class at every surface (last-50-commit serializer usage + pre-pre-lock hash + --expected-content-sha256 mandatory + post-stage hash + sha-prefix-length-mismatch META + drop-flag-and-retry detection).
- Catalog #206 subagent checkpoint discipline + Catalog #230 bulk-rewrite ownership map + Catalog #302 sister-subagent scope overlap via checkpoint JSONL + Catalog #314 bare-commit-absorbs-in-flight-files closes the edit-time + commit-time multi-subagent collision surface.

## Operator-action checklist

```bash
# 1. Verify wheel builds locally
.venv/bin/uv build --wheel

# 2. Push the two existing local tags to remote (if not already pushed)
git push origin oss-v0.2.0-rc1
git push origin v0.2.0-rc1

# 3. Create GitHub release from the tag (operator-action via gh CLI)
gh release create oss-v0.2.0-rc1 \
    --title "OSS v0.2.0-rc1: Task-Aware Compression release candidate" \
    --notes-file .omx/research/oss_v0_2_0_rc1_release_announcement_20260519T055711Z.md \
    dist/tac-0.2.0rc1-py3-none-any.whl

# 4. Optionally publish wheel to PyPI test-index for smoke
.venv/bin/uv publish --publish-url https://test.pypi.org/legacy/ dist/tac-0.2.0rc1-py3-none-any.whl
```

## Public Disclosure Hygiene confirmation

Per CLAUDE.md "Public Disclosure Hygiene" non-negotiable:

- ZERO credentials / API keys / tokens in this memo
- ZERO private infrastructure URLs
- ZERO local absolute paths (`/Users/...`, `/home/...`, `/tmp/...`)
- ZERO operator email / account metadata
- ZERO raw provider transcript dumps

This memo is OSS-publishable in its current form (operator approval pending per CLAUDE.md mutation frontier).

## Cross-references

- Sister landing: `feedback_hardening_license_plus_memory_hygiene_landed_20260519.md` (commit `16d2323db`) — LICENSE + per-category memory rotation pre-flight.
- Battle plan: `.omx/research/integrated_battle_plan_priority_queue_dag_cables_gates_20260519T052801Z.md` Cable F items F1+F2.
- SPDX header sweep: commit `82ecc2a0c` "oss release v0.2.0-rc1: inject SPDX-License-Identifier MIT headers across pure-SPDX .py files".
- Canonical helper inventory verified per Catalog #287 phantom-API: every cited `tac.X` importable as of 2026-05-19.

## Discipline contract per CLAUDE.md

- Catalog #229 PV: all canonical helpers grep-verified pre-edit + import-tested
- Catalog #287: no phantom-API in announcement copy
- Catalog #110/#113 HISTORICAL_PROVENANCE: LICENSE not edited (already present per mutation frontier)
- Public Disclosure Hygiene: ZERO leaks confirmed
- Memory rotation discipline: this memo is a NEW announcement (subject to category-window per Wave 2C #8 finding)

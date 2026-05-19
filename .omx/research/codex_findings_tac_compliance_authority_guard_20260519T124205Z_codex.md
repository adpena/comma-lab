# Codex Findings - TAC Compliance Authority Guard

Date: 2026-05-19T12:42:05Z
Trigger: xhigh read-only audit `019e403a-98ff-7923-8a27-88a1a576e6ca`

## Findings Closed

### P1 - CPU authority label overclaimed GHA validation

Closed.

`docs/pr_writeups/cpu_frontier_fec6_20260517.md` now labels the FEC6 CPU row as `[Modal Linux x86_64 CPU; GHA pending]` instead of `[contest-CPU GHA Linux x86_64]`. The disclosure section now says the CPU row must be retagged as GHA only after an actual GHA artifact exists.

Guard coverage added in `tools/check_tac_terminology.py`:

- refuses the stale FEC6 table row label
- refuses "Modal CPU (1:1 with GHA)" wording
- refuses the combined CPU+CUDA contest-claim sentence when the same writeup says GHA is pending

### P1 - Procedural generation promotion gate contradicted local authority model

Closed.

`docs/contest_compliance_authority.md` now requires score-bearing procedural/deterministic candidates to name `archive_seeded` or `weight_derived` in the compliance note. `runtime_constant` is allowed only with explicit maintainer/operator ruling plus non-payload proof that the constant is decoder logic rather than relocated score-bearing payload.

Guard coverage added in `tools/check_tac_terminology.py`:

- requires the archive-seeded / weight-derived promotion-gate sentence
- requires explicit ruling language for runtime constants
- refuses the stale `archive_seeded` or `runtime_constant` promotion-gate phrase

### P2 - Stale deterministic packet compiler API references

Closed.

`docs/pr_writeups/cpu_frontier_fec6_20260517.md` now references `tac.packet_compiler.deterministic_compiler` and public API `compile_packet(...)` instead of stale `tac.deterministic_compiler` / `canonical_emit()`.

Guard coverage added in `tools/check_tac_terminology.py`:

- refuses `tac.deterministic_compiler` in public docs
- refuses `canonical_emit()` in public docs

## Verification

```text
.venv/bin/python tools/check_tac_terminology.py --strict --json
{"finding_count": 0, "findings": [], "ok": true, "schema": "tac_terminology_check_v1"}

.venv/bin/python -m pytest src/tac/tests/test_tac_terminology_guard.py -q
11 passed in 1.56s

.venv/bin/ruff check tools/check_tac_terminology.py src/tac/tests/test_tac_terminology_guard.py
All checks passed.
```

## Residual Risk

The local authority file cites upstream PR/comment URLs but the repository does not preserve full PR comment snapshots. A stronger future pass should archive maintainer-comment excerpts or metadata into a committed source record so the procedural-generation authority ladder does not depend on live GitHub availability.

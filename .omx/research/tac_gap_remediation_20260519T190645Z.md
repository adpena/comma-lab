# tac Gap 1 (CI) + Gap 2 (README) remediation plan (Slot K successor — operator directive 2026-05-19)

**Date**: 2026-05-19T19:06:45Z (UTC)
**Operator directive (4 messages, captured at `.omx/research/operator_directive_pr_body_stealth_skunkworks_comprehensive_provenance_20260519T184500Z.md`)**:
> "link to all of the relevant resources a comma ai employee will need for this to be useful in production and for evaluation and review and reserach - both in comma-lab and in tac and make sure both are live and updated on origin/main"
> "super public with our codebases"

**Slot H prior verdict (commit `66a0a6aad`)**: `adpena/tac` PASSES with 2 non-blocking gaps.

## Repo state (per `gh repo view adpena/tac`)

| Field | Value |
|---|---|
| URL | https://github.com/adpena/tac |
| Visibility | **PUBLIC** ✓ |
| Default branch | `main` |
| Created | 2026-05-05T21:25:29Z |
| Last pushed | 2026-05-05T22:40:33Z |
| Disk usage | 3,245 KB (~3 MB) |
| Description | "Reusable codec/runtime library for the comma video compression challenge — score-aware sparse-encoder, predictor with refusal modes, meta-Lagrangian search engine, hardened preflight infrastructure" |
| Archived | false |

origin/main SHA (per `git ls-remote`): `32ea1405d798b0457d50fdc604b0ce69207ac360`

Currency: tac was pushed 2026-05-05; comma-lab has 14 additional session days of work since but tac is its own production-hardened library snapshot — they are NOT meant to track 1:1. The relevant question is whether tac's API surface still matches comma-lab's submission usage (it should; the submission_dir does NOT import from `tac.*`, it carries its own vendored inflate runtime).

## Gap 1 — CI workflow references non-existent test files

### Confirmation

`adpena/tac:.github/workflows/test.yml` line 25 (fetched via `gh api`):

```yaml
- name: Run meta-Lagrangian + predictor + distortion-proxy tests (deterministic, CPU-only)
  run: |
    .venv/bin/python -m pytest tac/tests/test_meta_lagrangian.py tac/tests/test_predictor_score_band.py tac/tests/test_distortion_proxy_local.py -v --timeout=60
```

Actual test files in `tac/tests/` (per `gh api repos/adpena/tac/contents/tac/tests`):
- ✓ `test_meta_lagrangian.py` (exists)
- ✗ `test_predictor_score_band.py` (does NOT exist; canonical = `test_score_band_predictor.py`)
- ✗ `test_distortion_proxy_local.py` (does NOT exist; closest sibling = `test_rate_distortion_floor.py`)

CI failure pattern (per `gh run view 25406255121`):
```
ERROR: file or directory not found: tac/tests/test_predictor_score_band.py
```

4 of last 4 CI runs FAIL. CI has been red since 2026-05-05 (last push date).

### Fix path (operator-gated; DRAFT PR ONLY, NOT auto-merged)

Replace stale paths with canonical names:

```yaml
- name: Run meta-Lagrangian + predictor + distortion-proxy tests (deterministic, CPU-only)
  run: |
    .venv/bin/python -m pytest tac/tests/test_meta_lagrangian.py tac/tests/test_score_band_predictor.py -v --timeout=60
```

Decision on `test_distortion_proxy_local.py`:
- Slot H suggested either (a) remove the reference if the test was never written, or (b) replace with the canonical sibling `test_rate_distortion_floor.py`
- Recommendation: **REMOVE** the reference (the test never existed; we cannot assume `test_rate_distortion_floor.py` was the intended sibling without operator confirmation)
- Alternative: add `test_rate_distortion_floor.py` to the smoke list — that test IS the canonical rate-distortion sanity gate per the tac README's MetaLagrangianSearch description ("5-gate predispatch sanity ladder")

### Recommended remediation (operator-routable):

**Option A (minimal, fastest, recommended)**: REMOVE the non-existent `test_distortion_proxy_local.py` reference + RENAME `test_predictor_score_band.py` to `test_score_band_predictor.py` (the canonical name).

```yaml
.venv/bin/python -m pytest tac/tests/test_meta_lagrangian.py tac/tests/test_score_band_predictor.py -v --timeout=60
```

**Option B**: Expand to include `test_rate_distortion_floor.py` if that was the canonical sibling intent:

```yaml
.venv/bin/python -m pytest tac/tests/test_meta_lagrangian.py tac/tests/test_score_band_predictor.py tac/tests/test_rate_distortion_floor.py -v --timeout=60
```

**Operator approval gate**: Slot K does NOT auto-merge. Operator must (a) confirm Option A vs B, (b) `gh pr create --repo adpena/tac` and (c) manually merge to clear the 14-day-red CI badge.

**Estimated effort**: 5 minutes (Y/N decision + 1-line YAML edit + draft PR + manual merge).

## Gap 2 — tac README references comma-lab (REFRAMED per new operator directive)

### Original Slot H finding

`adpena/tac:README.md` has 4 references to `https://github.com/adpena/comma-lab`; comma-lab visibility=PRIVATE so public readers get 404. Slot H aligned this with operator's earlier "don't link comma-lab" directive.

### Reframed per operator's 2026-05-19 mid-flight directive

Operator NOW says: *"link to all of the relevant resources a comma ai employee will need for this to be useful in production and for evaluation and review and reserach - both in comma-lab and in tac and make sure both are live and updated on origin/main"*

**This INVERTS Slot H's recommendation.** The 4 comma-lab references in tac README are now CORRECT under the new directive — but only resolve to a valid page IF comma-lab becomes public first.

### Decision tree

```
IF comma-lab becomes PUBLIC (per Phase 1 audit decision):
    KEEP the 4 comma-lab references in tac README; they will resolve.
    No tac README change required.
ELSE (comma-lab stays PRIVATE):
    REPLACE comma-lab references with one of:
        (a) commaai/comma_video_compression_challenge URL (canonical upstream)
        (b) Soft phrasing: "broader research environment (access on request)"
        (c) Remove the reference entirely; tac stands on its own
```

### Recommended action

**Defer to Phase 1 operator verdict**. Slot K does NOT modify tac README in this dispatch:

- If operator approves Option 2 (sanitize then public) per Phase 1 audit: tac README is correct as-is; no PR needed
- If operator defers comma-lab public-flip: produce a DRAFT PR replacing the 4 references with `commaai/comma_video_compression_challenge` URL + soft access-on-request phrasing

## Phase 4 — both-repos-live status

| Repo | origin/main SHA | Last pushed | Local HEAD relationship | Status |
|---|---|---|---|---|
| `adpena/comma-lab` (this repo) | `8a9e72198fe7f9a40ccb55a154d25f650cf424ee` | 2026-05-19T18:56:37Z | 3 commits ahead of origin/main (active session work) | LIVE; PRIVATE; operator should `git push` the 3 commits before any public flip |
| `adpena/tac` | `32ea1405d798b0457d50fdc604b0ce69207ac360` | 2026-05-05T22:40:33Z | Independent repo; not tracked by local HEAD | LIVE; PUBLIC; STALE (14 days since last push but as an extracted production library that's not necessarily a problem) |

### Currency assessment

**comma-lab**: 3 commits ahead of origin (sister-subagent in-flight work for PR submission prep). Operator should push BEFORE making public to ensure full submission lineage is on origin/main.

**tac**: 14 days since last push. The relevant question is whether tac contains the codec primitives that comma-lab's submission archive uses. The submission archive (`experiments/results/pr101_*/submission_dir/`) carries its own vendored runtime; it does NOT import from `tac.*` at runtime. Therefore tac's currency-vs-comma-lab is NOT a blocker for the contest PR. However, operator-routable consideration: if comma-lab introduces new tac-canonical primitives during the post-2026-05-05 window (FEC6 codec, master_gradient ledger, Wyner-Ziv deliverability_proof_builder, etc.), the operator may want to backport-and-push to tac to keep the public production library current.

Specifically: tac contains `archive_codec.py`, `archive_diet.py`, `archive_optimizer.py`, `bit_allocator.py`, `block_fp_codec.py`, `bootstrap_codegen.py`, etc. — the canonical codec primitives. It does NOT yet contain `fec_codec.py` or `frame_exploit_compactor.py` (per `gh api repos/adpena/tac/contents/tac` listing).

**Operator-routable**: If FEC6 is foundational to the submission narrative, consider extracting it into tac and pushing as v0.6.0. Not blocking for PR submission (submission carries the codec inline) but improves "comma.ai employee can use tac in production" story.

## Phase 5 — landing memo + lane gates + MEMORY.md prepend

See landing memo at `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_comma_lab_audit_plus_tac_gap_remediation_landed_20260519T190645Z.md`.

Lane: `lane_comma_lab_public_readiness_audit_plus_tac_gap_remediation_20260519` L1 (impl_complete + memory_entry).

## Operator-routable decisions queue (4 items)

1. **comma-lab visibility**: Approve Option 1 (public as-is) / Option 2 (sanitize then public; recommended) / Option 3 (untrack historical then public) / Defer (stays private). See `.omx/research/comma_lab_public_readiness_audit_20260519T190645Z.md` for full decision-support material.

2. **comma-lab pre-public sanitization sweep (if Option 2)**: Approve bulk-rewrite commit to sanitize `/Users/adpena/Projects/pact/` paths + Tailscale IPs in scripts/research memos. Estimated 1-2 hours. Spawn sister subagent or operator runs `sed` directly?

3. **tac CI fix PR (Gap 1)**: Approve Option A (remove non-existent test) / Option B (add `test_rate_distortion_floor.py`). Operator must `gh pr create --repo adpena/tac` + merge manually (Slot K does NOT auto-merge per CLAUDE.md "Executing actions with care").

4. **tac README Gap 2 (deferred to Phase 1 outcome)**: If comma-lab goes public → no action. If comma-lab stays private → approve draft PR replacing comma-lab references with `commaai/comma_video_compression_challenge` URL.

## Sister-subagent coordination

- **Slot H** (`a4fdeabcf4dce8417`): COMPLETE at commit `66a0a6aad` (`adpena/tac` audit verdict).
- **Slot I** (`a9ea5d8b06718bd22`): PR 95/Quantizr study + citations + tone refinement. In flight. Scope: PR body file + research artifacts. DISJOINT from Slot K.
- **Slot J** (`a4b959803c985e440`): T3 council symposium on FINAL PR body. In flight. Scope: council symposium artifact + memory. DISJOINT from Slot K.
- **Slot K (this dispatch)**: comma-lab audit + tac gap remediation. Scope: `adpena/comma-lab` + `adpena/tac` repos + audit memos.

Per Catalog #230 sister-subagent ownership map: NO files touched by Slot K overlap with Slot I/J declared scopes.

Per Catalog #340 sister-checkpoint guard: Slot K's commit will route through canonical serializer with `--expected-content-sha256` per Catalog #157/#174 sister discipline.

## Cross-references

- Operator directive memo: `.omx/research/operator_directive_pr_body_stealth_skunkworks_comprehensive_provenance_20260519T184500Z.md`
- Slot H prior audit: commit `66a0a6aad`
- comma-lab Phase 1 audit (companion): `.omx/research/comma_lab_public_readiness_audit_20260519T190645Z.md`
- CLAUDE.md "Public Disclosure Hygiene" non-negotiable
- CLAUDE.md "Executing actions with care" non-negotiable
- CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable (defines the conditions under which the PR body can claim contest-CPU and contest-CUDA scores; the PR body's link-discoverability requirement is downstream of this)

— Claude-main (Slot K successor `comma_lab_audit_tac_gap_remediation_20260519T190645Z`) 2026-05-19T19:06:45Z

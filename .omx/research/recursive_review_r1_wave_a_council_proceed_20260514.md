# R1 Council Ledger — Wave A 5 council-PROCEED commits

**Lane**: `lane_recursive_review_r1_wave_a_council_proceed_20260514` (L1)
**Subagent**: RECURSIVE-REVIEW-R1-WAVE-A-SUBAGENT
**Date**: 2026-05-14
**Cycle**: R1 of 3 consecutive clean passes per CLAUDE.md "Recursive adversarial review protocol"
**Verdict**: **FINDINGS** — counter RESETS to 0/3

## Scope

5 commits / ~5500 LOC across 6 council-PROCEED landings. R1 council rotation:

- **Yousfi** — contest-scorer fidelity / contest-CUDA-vs-CPU axis discipline / FORBIDDEN_PATTERNS audit
- **Fridrich** — information-theoretic correctness / steganalysis-inverse mathematical correctness
- **Hotz** — engineering shortcuts / dead code / what would I delete / what's the simplest version
- **Quantizr** — leaderboard-faithful realism / archive-bytes accounting
- **Contrarian** — challenge bold claims / find the WEAK arguments / non-conservative bias check

R2 + R3 will rotate to different perspectives (Selfcomp/MacKay/Hassabis/Boyd/Tao for R2; Time-Traveler+Schmidhuber+Hinton+Karpathy+Carmack for R3) per CLAUDE.md non-negotiable.

## Per-commit verdict table

| Commit | Lane / Subject | R1 verdict | Findings |
|---|---|---|---|
| `951858245` | D9 per-class provider routing | **CLEAN scope** | None within commit; META-1/META-2 are pre-existing |
| `e84accd7c` | Catalog #233 L1→L2 4-gate | **CLEAN with caveat** | Strict-flip atomicity OK (warn-only); strict-flip blocked on operator-routed audit-and-downgrade sweep |
| `8202dc0aa` | Claim catalog #233 (state file) | **CLEAN** | None (1-line state file mutation; checkpoint waived per documented rationale) |
| `e54901d60` | Z3 v2 latent-replacement | **FINDINGS** | HOTZ-1 LOW (duplicate `select_inflate_device`); QUANTIZR-1 MED (no byte-savings regression test) |
| `dd17e6e2e` | Tier C PR106 + DP1 extension | **CLEAN code; FINDINGS commit-machinery** | META-3 PARTIAL (commit bypassed serializer per Catalog #117 — but #117 itself has a bug, see META-4) |
| `0916332eb` | C1 Z5 routing + F3 vq_vae + PDP | **CRITICAL FINDINGS** | META-3 CRIT (empty body, no Co-Author, no journal-grade); YOUSFI-1 MED (lane_f3_backport_vqvae_pdp trips Catalog #124 STRICT); HOTZ-2 LOW (vq_vae waiver markers removed); QUANTIZR-2 LOW (Catalog #228 text outdated); FRIDRICH-2 LOW (PDP help text outdated) |

## CRITICAL findings (counter-blocking)

### META-3 — `0916332eb` commit body is EMPTY (CRITICAL)

**Voice**: Yousfi + Contrarian unanimous CRITICAL.

The commit `0916332eb` "Wire C1 Z5 routing and F3 cache surfaces" has:
- **No body whatsoever** (only the 1-line subject)
- **No Co-Authored-By trailer** per Catalog #119 — confirmed via `check_subagent_commits_have_co_author_trailer` returning a violation specifically for `0916332`
- **No checkpoint discipline waiver** per Catalog #206 — bug-class anchor: predecessor F3-BACKPORT-WAVE-V2 landing memo expressly cited this protocol
- **No premise verifier reference** per Catalog #229 + the prompt-premise-verification-before-edit pattern (the parent prompt explicitly required this for the F3 backport family)
- **No 6-hook wire-in declaration** per Catalog #125 (operator's autopilot/continual-learning wire-ins SHOULD be declared)
- **No journal-grade ledger entry** per the operator's 2026-05-14 directive (`journal_lab_grade_documentation_standard_directive_20260514.md`)

The serializer log (`.omx/state/commit-serializer.log`) shows TWO subagent commits with this exact `head_after=0916332eb` BOTH ending in `commit_failed: nothing to commit, working tree clean`. This is the **commit-swap class** that Catalog #157 was built to extinct: the OUTER commit was created by some non-canonical mechanism BEFORE the serializer ran, then the serializer found nothing to commit and reported failure.

**Quantizr's lens**: this is the same anti-pattern that the FFFF Bug 5 cleanup + Catalog #157 + Catalog #186 chain extincted at every other surface. The C1+F3 subagent landing IS the mechanism's regression case. The commit IS the bug-class anchor.

**Hotz's lens**: zero-content commit body is engineering rot. Future review can't reconstruct intent without forensically diffing files. Per CLAUDE.md "Beauty, simplicity" non-negotiable, this is unmaintainable.

### META-4 — Catalog #117 hash-prefix matching bug (CRITICAL)

**Voice**: Fridrich + Hotz CRITICAL.

`check_subagent_commit_serializer_uses_lock` (Catalog #117) compares `short_sha (7 chars)` against `seen_hashes` (which contains 9-char prefixes from `head_after` field of serializer-log JSONL rows). Neither `short_sha (7) in seen_hashes (9-char strings)` NOR `full_sha (40) in seen_hashes (9-char)` ever matches. Result: **EVERY commit in the serializer log is reported as a false-positive Catalog #117 violation.**

Empirical proof:
```python
seen = {'951858245', 'e54901d60', 'dd17e6e2e'}
short = '9518582'
print('short in seen:', short in seen)  # False
print('full check:', any(s.startswith(short) for s in seen))  # True
```

**Hotz's lens**: simplest fix — change `if short_sha in seen_hashes or full_sha in seen_hashes` to `if any(seen.startswith(short_sha) or full_sha.startswith(seen) for seen in seen_hashes)`. ~2 LOC.

**Fridrich's lens**: Catalog #117's claimed 50 violations are likely all false positives. The actual unserialized-commit count is unknown until the prefix bug is fixed. This means Catalog #117's "warn-only @ 50" status in CLAUDE.md is DEEPLY MISLEADING — the gate has been STRUCTURALLY BROKEN since landing.

**Cross-ref**: this is a Catalog #185 sister bug. CLAUDE.md catalog row text should reflect ACTUAL gate behavior; the gate text likely overstates effectiveness.

### META-1 — Catalog #185 reports 5 strict-flipped catalogs with non-zero live counts (CRITICAL)

**Voice**: Yousfi + Quantizr + Contrarian CRITICAL.

`check_strict_flipped_catalog_entries_have_live_count_zero` reports:
- **Catalog #124** (representation lane archive grammar) — claims "Live count: 0" + STRICT but gate returned **2 violations** (`lane_grand_council_maximize_value_20260514` + `lane_f3_backport_vqvae_pdp_20260514`)
- **Catalog #130** (no tag-only custody validation) — claims 0, returns 2 (pre-existing in `tools/build_pr101_frame_exploit_selector_packet.py` + `tools/build_frame_exploit_selector_packet.py`)
- **Catalog #158** (deterministic compiler canonical use) — claims 0, returns 8 (pre-existing in `tools/build_*_packet.py` + `tools/materialize_*.py`)
- **Catalog #162** (operator authorize canonical use) — claims 0, returns 1
- **Catalog #171** (substrate recipe video_input_strategy) — claims 0, returns 1 (`substrate_z3_balle_hyperprior_bolton_modal_t4_dispatch.yaml` declares `cpu_thread_async_upload` not in recognized set)

**Yousfi's verdict**: CLAUDE.md catalog table is the canonical operator-facing strictness ledger per Catalog #159 + #176. Five entries are now misleading. Per CLAUDE.md "Apples-to-apples evidence discipline": "Generated reports must preserve the axis label" — these claims are deceptive at the strictness-level.

### META-2 — `preflight_all()` is currently FAILING (CRITICAL but pre-existing)

**Voice**: Contrarian sole, but Yousfi concurs.

```
WRAPPER STAGE-IMPL VIOLATIONS:
/Users/adpena/Projects/pact/scripts/remote_lane_substrate_c1_world_model_foveation.sh:181
/Users/adpena/Projects/pact/scripts/remote_lane_substrate_c6_e4_mdl_ibps.sh:218
```

These are pre-existing (NOT introduced by Wave A) but they BLOCK promotion of any new strict gate to STRICT mode. Per CLAUDE.md "Operator gates must be wired and used" non-negotiable, `preflight_all()` SHOULD be runnable to clean-pass before any new strict-flip lands.

**Contrarian's challenge**: "Why are we landing 5 new commits + 4 new strict gates while the umbrella `preflight_all()` is broken?" Answer should be either (a) the violations are now also fixed by these commits (they're not), or (b) the violations are explicitly tagged for follow-up. Neither is true.

## Medium findings

### YOUSFI-1 (MEDIUM) — `lane_f3_backport_vqvae_pdp_20260514` trips Catalog #124 STRICT

The C1+F3 commit `0916332eb`'s subagent registered a lane named `lane_f3_backport_vqvae_pdp_20260514` (an OPTIMIZATION wire-in lane, NOT a representation/codec lane), but the `vq_vae` token in the lane id matches `_REPRESENTATION_LANE_NAME_TOKENS` and the lane has no `lane_class=substrate_engineering` / `target_modes=research_substrate` / `research_only=true` opt-out. Catalog #124 is wired STRICT (`strict=True` at orchestrator callsite), so this WOULD raise `PreflightError` if `preflight_all()` were able to reach it (currently blocked by META-2).

**Fix**: add `lane_class=substrate_engineering` to the lane registry entry, OR rename the lane to omit the `vq_vae` token, OR explicitly add `notes` text containing `research_only=true`.

### QUANTIZR-1 (MEDIUM) — Z3 v2 byte-savings claim has no regression test

Commit `e54901d60`'s commit message claims "smoke v2 full path on macOS CPU: A1 178162 B → v2 173320 B (saved 4842)" but `src/tac/substrates/z3_balle_hyperprior_bolton/tests/test_z3_v2_substrate.py` has NO assertion on the 4842-byte savings. Future refactor that adds fixed overhead will silently degrade savings. Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" the [macOS-CPU advisory] tag is present in the commit body, so this is mitigated, BUT a single-byte regression assertion (e.g. `assert len(v2_archive) <= len(a1_archive) - 4000`) would extinct the regression class.

### CONTRARIAN-1 (MEDIUM) — Catalog #171 latent violation

The Z3 recipe `substrate_z3_balle_hyperprior_bolton_modal_t4_dispatch.yaml` declares `video_input_strategy: cpu_thread_async_upload` which is not in the recognized set `{per_dispatch_local_copy, readonly_mmap, shared_volume_no_contention_expected}`. Pre-existing (not introduced by Wave A) but operator should be aware.

## Low findings

### HOTZ-1 (LOW) — Z3 v2 `inflate_v2.py` duplicates `select_inflate_device`

The new `src/tac/substrates/z3_balle_hyperprior_bolton/inflate_v2.py` defines a LOCAL `select_inflate_device` instead of importing the canonical helper from `tac.substrates._shared.inflate_runtime.select_inflate_device`. Catalog #205 only scans `submissions/*/inflate.py` so this isn't currently flagged, but the duplication invites future drift.

### HOTZ-2 (LOW) — vq_vae waiver markers REMOVED in `0916332eb`

The C1+F3 commit removes `# AUTOCAST_FP16_WAIVED:...` and `# TORCH_COMPILE_WAIVED:...` markers from `experiments/train_substrate_vq_vae.py`. This is correct ONLY because the flags ARE now declared in argparse. A future revert that removes the argparse declarations without re-adding the waiver markers would silently trip Catalogs #172 and #179.

### QUANTIZR-2 (LOW) — Catalog #228 text outdated

CLAUDE.md Catalog #228 row says "live count at landing: 2 (s2sbs_byte_stuffing + vq_vae)". After `0916332eb`'s vq_vae wire-in, actual live count is 1 (`s2sbs_byte_stuffing` only). The catalog text needs operator-routed update.

### FRIDRICH-2 (LOW) — PDP `--enable-gt-scorer-cache` help text outdated

`experiments/train_substrate_pretrained_driving_prior.py:206-213` argparse help still says "RESERVED (O1): GT-scorer-output cache; wire-in pending" but the flag is now FULLY WIRED in `0916332eb`. Operator running `--help` will see misleading text.

## Council adversarial cross-debate

**Yousfi**: "Catalog #117 has been broken since landing. Catalog #185 reports 5 false strict-flip claims. We're not measuring what we think we're measuring. Five PROCEED verdicts landed atop a measurement infrastructure that itself has 6+ bugs. The Wave A commits are *individually* clean BUT the surrounding infrastructure has structural rot."

**Fridrich**: "META-4 (Catalog #117 prefix bug) is the dominant finding. The other catalog drift findings are downstream of unreliable measurement. Fix #117 first; recompute everything; THEN re-evaluate strict-flip atomicity claims."

**Hotz**: "0916332eb empty body is the SIMPLEST observable failure. Build the simplest fix: every subagent commit MUST have a body with Co-Author + checkpoint + journal-grade ledger reference, OR the commit fails the catalog gate. Don't let this ship as a precedent."

**Quantizr**: "Z3 v2's empirical claim of 4842 bytes saved is a marketing number until there's a regression test. The leaderboard rewards bytes; without an assertion, future refactors will silently swallow the savings. Add the assertion in R1.5."

**Contrarian**: "Five PROCEED verdicts landed in a 9-minute window (15:53:02 → 16:02:03). Five council reviews — at $0 GPU each — that converged unanimously on PROCEED. Per CLAUDE.md 'Council conduct': unanimous votes should be SCRUTINIZED. The omnibus 7872c9f4b ledger has 11/11 unanimous on multiple decisions. Where was the dissent? The Contrarian's role is to call this out: rapid unanimous consensus on infrastructure decisions is itself a finding. R2 should challenge this directly."

## Recommended FIX-WAVE-R1 actions

Per CLAUDE.md "Recursive adversarial review protocol" + "Bugs must be permanently fixed AND self-protected against":

1. **FIX-CRIT-1** (META-4): patch Catalog #117 hash-prefix matching bug (~2 LOC) + add 3 dedicated tests (positive 7-vs-9-char match, negative 7-vs-7-char no-match, multi-prefix-collision boundary). Re-run #117; expect dramatic reduction in violation count.
2. **FIX-CRIT-2** (META-3): backfill `0916332eb` commit-body content via `git notes add` (since amending is forbidden per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" — preserving forensic record). Add a journal-grade ledger entry at `.omx/research/c1_z5_f3_wire_in_landing_20260514.md` referencing the commit + 6-hook wire-in + premise verifier path.
3. **FIX-MED-1** (YOUSFI-1): mark `lane_f3_backport_vqvae_pdp_20260514` with `lane_class=substrate_engineering` via `tools/lane_maturity.py` to clear the Catalog #124 violation.
4. **FIX-MED-2** (QUANTIZR-1): add a 1-line assertion `assert build_v2_archive_bytes(...) <= a1_archive_bytes - 4000` to Z3 v2 test suite.
5. **FIX-LOW-1** (HOTZ-2): re-add `# AUTOCAST_FP16_WAIVED` and `# TORCH_COMPILE_WAIVED` markers as defense-in-depth comments next to the argparse declarations.
6. **FIX-LOW-2** (FRIDRICH-2): update PDP `--enable-gt-scorer-cache` help text to remove "RESERVED (O1) ... wire-in pending" and reflect the now-active wiring.
7. **FIX-LOW-3** (QUANTIZR-2): update CLAUDE.md Catalog #228 text from "live count at landing: 2" to "live count after vq_vae wire-in 0916332eb: 1 (s2sbs_byte_stuffing remaining)".
8. **FIX-CRIT-3** (META-2): bring in a sister subagent to fix the 2 wrapper stage-impl violations in `scripts/remote_lane_substrate_c1_world_model_foveation.sh` + `scripts/remote_lane_substrate_c6_e4_mdl_ibps.sh` so `preflight_all()` returns to passing.
9. **FIX-CRIT-4** (META-1): per Catalog #185 verdicts, either (a) fix the 5 strict-flipped catalog underlying violations OR (b) update CLAUDE.md text to reflect actual non-zero live counts.

## Review counter

R1 verdict: **FINDINGS — counter RESETS to 0/3.**

R2 trigger conditions: FIX-WAVE-R1 lands AND Catalog #117 prefix bug fixed AND Catalog #185 returns to 0 violations.

R2 council rotation: **Selfcomp / MacKay / Hassabis / Boyd / Tao** per CLAUDE.md non-negotiable.

## Provenance

- Lane: `lane_recursive_review_r1_wave_a_council_proceed_20260514` (L1)
- Findings JSONL: `.omx/research/recursive_review_findings.jsonl` (committed; mirror at `.omx/state/recursive_review_findings.jsonl` is the live runtime copy per CLAUDE.md `.omx/state/*` gitignore)
- Memory: `feedback_recursive_review_r1_wave_a_landed_20260514.md`
- Reproducers ran: `pytest src/tac/tests/test_d9_per_class_provider_routing.py` (50 pass), `pytest src/tac/tests/test_d9_operator_authorize_routing_integration.py` (13 pass), `pytest src/tac/tests/test_check_233_l1_to_l2_promotion_canonical.py` (60 pass), `pytest src/tac/substrates/z3_balle_hyperprior_bolton/tests/test_z3_v2_substrate.py` (31 pass), `pytest src/tac/tests/test_mdl_ablation_tier_c_pr106.py src/tac/tests/test_mdl_ablation_tier_c_dp1.py` (41 pass), `pytest src/tac/substrates/c1_world_model_foveation/tests/test_c1_z5_routing_and_autopilot_halve.py src/tac/tests/test_f3_backport_vqvae_pdp_wired.py src/tac/substrates/pretrained_driving_prior/tests/test_score_aware_loss_f3_kwargs.py src/tac/substrates/time_traveler_l5_autonomy/tests/test_z5_routed_latent_predictor.py` (71 pass), `pytest src/tac/tests/test_cathedral_autopilot_tier_c_and_composition.py` (42 pass).
- Strict gates run live: Catalog #117, #119, #124, #125 (sister), #130, #158, #162, #164, #171, #172, #179, #185, #205, #228.
- Total dedicated tests passing for the 5 commits: **308 / 308**.

---

**Per CLAUDE.md "Council conduct"**: rapid unanimous consensus is itself a finding. R1 dissent is registered. R2 must challenge.

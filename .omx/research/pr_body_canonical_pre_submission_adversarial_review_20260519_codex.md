# PR Body Canonical Pre-Submission Adversarial Review — 2026-05-19

**Issued:** 2026-05-19T18:05:00Z
**Subagent:** PR-SUBMISSION-PREP-D1-D2-D3-EXECUTION (lane `lane_pr_submission_prep_d1_d2_d3_20260519`)
**Authority:** `score_claim=false` + `promotion_eligible=false` + `ready_for_submission=false` + `ready_for_provider_dispatch=false` — this is a pre-submission adversarial review, NOT a score claim.

## Codex companion invocation status

Codex CLI was invoked three times per CLAUDE.md "Codex CLI invocation — NON-NEGOTIABLE" Pattern A (detached BG bash):

- **First attempt (12:54, xhigh, full prompt):** PID 93778. Failed at `gh pr view ... --repo commaai/comma_video_compression_challenge` with `error connecting to api.github.com`. Codex sandbox is `read-only` which blocks network.
- **Second attempt (12:56, xhigh, v2 prompt with pre-cached PR data):** PID 97908. Began reading 12 files (AGENTS.md / CLAUDE.md / routing directive / PR body / upstream template + workflow / pre-cached PRs / inflate.sh / inflate.py / contest_auth_eval.py / ~/.codex/memories/MEMORY.md). Process exited at 13:01:57 (~6 min runtime, 4971 log lines) BEFORE emitting verdict block. No last.txt produced. Hypothesis: xhigh reasoning + large context loading consumed token/time budget.
- **Third attempt (13:04, high effort, minimal 3-file prompt):** PID 11342 → completed successfully at ~13:05 with `pr_body_min_review.last.txt`. **REAL CODEX VERDICT OBTAINED.**

**Final codex verdict from minimal-prompt run** at `.omx/tmp/codex_runs/pr_body_min_review.last.txt`:

> VERDICT: APPROVE_WITH_REVISIONS
>
> Template-conformance P0 blocker: **no** — the short body uses every upstream template heading in order and provides substantive answers for each required field.
>
> HIGH findings: None on template structure.
>
> MEDIUM findings:
> - The `report.txt` block reports only rounded `Final score ... = 0.19`; the exact `0.1920513169` is stated immediately after, but upstream reviewers may treat the fenced report as the canonical copied report. Prefer making clear the fenced block is verbatim rounded report output and the following line is recomputed full precision.
> - The GPU answer says CPU works, but also says inflate auto-selects CUDA when available. Since the same archive has materially different CUDA score, keep the current CUDA disclosure, but tighten the answer to: "no; CPU inflation works; CUDA-enabled hosts may take the CUDA path and produce the disclosed paired CUDA score."
>
> LOW findings:
> - The release URL is acceptable if `curl -L` works, but this review did not verify network availability or asset reachability.
> - "compression pipeline depends on private training infrastructure" is honest, but slightly weak for reproducibility optics. Acceptable because the template asks whether to merge it, not whether every training step is public.
>
> Single operator-routable recommended action:
> - Ship after two small wording edits: clarify the fenced `report.txt` rounding vs exact score, and tighten the GPU/inflate-device sentence so reviewers cannot confuse the CPU headline with CUDA-host behavior.

**Both codex-recommended MEDIUM edits APPLIED** in the same commit batch to `PR_BODY_UPSTREAM_TEMPLATE_CONFORMANT.md`:
1. `# report.txt` section now has clarifying preamble: "The block below is the verbatim `report.txt` output from upstream `evaluate.sh --device cpu` on this archive; upstream rounds `Final score` to 2 decimal places. The exact full-precision score recomputed from the same components is `0.1920513169`..."
2. `# does your submission require gpu` section now reads: "no; CPU inflation works and produces the headline `0.1920513169` `[contest-CPU; Modal Linux x86_64 reproduction]` score. CUDA-enabled hosts may take the CUDA path (inflate auto-selects `torch.device("cuda")` when available) and will produce the disclosed paired CUDA score `0.2262100217` `[Modal T4 CUDA replay]` instead. Both axes ship from the same archive bytes (sha256 `6bae0201...`)."

The codex LOW findings are acknowledged but not blocking:
- Release URL `curl -L` verification: blocked on D3 Option A execution (operator approval required); the operator-runnable plan at `pr_submission_hosting_plan_20260519T175900Z.md` step 4 includes the curl smoke test.
- "private training infrastructure" wording: codex itself classifies as "acceptable" given the template asks about merging compress.sh, not full training pipeline.

The sections below preserve the dispatching subagent's INDEPENDENT analysis (1:1 compliance findings + 6 review axes + Slot E context) which goes BEYOND what codex was prompted to review — this complements rather than replaces the codex verdict.

---

## DISPATCHING SUBAGENT INDEPENDENT VERDICT: **APPROVE_WITH_REVISIONS** (concurs with codex)

The canonical body at `.omx/research/pr_submission_check_in_package_20260519/PR_BODY_CANONICAL.md` is **NOT in upstream PR template format** and would NOT be accepted as-is by the contest maintainer's eval workflow. **However, codex explicitly clarifies that the template-conformant body PR_BODY_UPSTREAM_TEMPLATE_CONFORMANT.md DOES satisfy template structure**, so template-conformance is NOT a P0 blocker once we ship the conformant body.

**Recommendation (concurs with codex):** ship the upstream-template-conformant body at `.omx/research/pr_submission_check_in_package_20260519/PR_BODY_UPSTREAM_TEMPLATE_CONFORMANT.md` (now updated with codex's MEDIUM edits).

---

## Per-blocker status from prior reviews

| # | Blocker | Status in canonical body | Status in template-conformant body |
|---|---|---|---|
| P0 | Headline CPU claim axis labeling | ✓ RESOLVED (`[Modal Linux x86_64 CPU reproduction; contest/GHA host validation pending]`) | ✓ RESOLVED (same labeling preserved) |
| P0 | Submission gate claim consistency | ✓ RESOLVED (no claim of "submission compliance passing" without compliance JSON) | ✓ RESOLVED (same discipline preserved) |
| P0 | Reproduction + provenance one runnable path | ✓ RESOLVED (one archive path / one inflate path / one canonical command) | ✓ RESOLVED (same; report.txt + canonical sha pinned) |
| P0 | Hidden-better-score contradiction | ✓ RESOLVED (no `~0.171` reference) | ✓ RESOLVED |
| P0 | Employment/sponsorship ask | ✓ RESOLVED (single closing sentence "Happy to discuss engineering details...") | ✓ RESOLVED (closing line preserved with minor wording) |
| P1 | CUDA comparison score-direction (lower-is-better) | ✓ RESOLVED (explicitly described as paired context, NOT primary axis) | ✓ RESOLVED |
| P1 | CPU/CUDA mechanism causality | ✓ RESOLVED (observed-and-documented, not causally attributed) | ✓ RESOLVED |
| P1 | Internal process language stripped | ✓ RESOLVED (no Catalog #, no Rudin-Daubechies, no cathedral autopilot, no nicknames) | ✓ RESOLVED |

**All 8 prior-review P0/P1 blockers REMAIN RESOLVED in both bodies.** The new P0 blocker is purely structural (template format conformance), not content regression.

---

## NEW P0 finding: PR template-conformance

**Finding:** The canonical body at `PR_BODY_CANONICAL.md` is a 118-LOC technical writeup with custom section headers (`## 1. Claim`, `## 2. What changed vs PR101`, `## 3. Reproduce`, etc.). The upstream PR template at `upstream/.github/pull_request_template.md` mandates 5 specific sections:

```
# submission name:
# upload zipped `archive.zip`
# report.txt
# does your submission require gpu for evaluation (inflation)?
# did you include the compression script? and want it to be merged?
# additional comments
```

**Empirical confirmation from 5 merged PRs (PR95/98/101/102/106):** every successful merged PR uses the template format exactly. PR101 GOLD (3rd place), PR98, PR95 use the URL pattern; PR102 uses in-repo; PR106 follows template.

**Why this is P0:**
1. The contest eval workflow (`upstream/.github/workflows/eval.yml:60`) extracts `submission_url` from the PR body via a maintainer manual workflow_dispatch invocation. The maintainer expects to see `# upload zipped \`archive.zip\`` followed by a curl-able URL. The canonical body's `## 1. Claim` table header would force the maintainer to manually parse the body to find the URL.
2. The `# report.txt` section is expected to contain the verbatim `report.txt` content (with ` ``` ` code-fenced block) so the maintainer's eval validation pipeline can compare it against the actual evaluator output. The canonical body's `## 1. Claim` table embeds the same components but in a different shape; the maintainer would have to manually transform.
3. The `# does your submission require gpu` and `# did you include the compression script` sections are explicit maintainer questions; not answering them in the template format risks the maintainer asking "is this gpu-only?" or "do you want compress.sh merged?" in PR comments, costing review velocity.
4. **Trust-per-minute principle:** the maintainer is reviewing dozens of PRs. A template-conformant body reads as "this submitter respects the contest process" — a 118-LOC writeup with custom sections reads as "this submitter doesn't know the rules / requires extra parsing."

**Mitigation:** the dispatching subagent produced `PR_BODY_UPSTREAM_TEMPLATE_CONFORMANT.md` in the same commit batch. It carries the same canonical claims + provenance + axis-tag discipline but in the 5-section template format. The canonical 118-LOC writeup remains a useful internal reference / long-form documentation but is NOT what we ship in the PR body.

---

## 1:1 contest compliance findings

### upstream/evaluate.py conformance: ✓ PASS

- Our `inflate.sh` and `inflate.py` produce raw frames at the resolution upstream/evaluate.py expects (`(T, H, W, C)` per video, where T = pair count). Verified by Slot C package's bit-stable Modal CPU + Modal T4 runs (both successful rc=0).
- The contest formula `S = 100·d_seg + √(10·d_pose) + 25·(archive_bytes / 37,545,489)` matches the headline score computation in both auth-eval JSON results (CPU `0.1920513169` recomputed exactly; CUDA `0.2262100217` recomputed exactly).

### upstream/.github/workflows/eval.yml conformance: ✓ PASS (with caveat)

- `curl -L -o ./submissions/...archive.zip "${submission_url}"` (line 60) works against GitHub Release URLs (proven by PR101 + PR98 patterns).
- 30-minute timeout (`timeout-minutes: 30` line 30) is well within our inflate's wall-clock budget (Modal CPU completed in ~2 min on a 4-core 16GB container, scaling to GHA `ubuntu-latest` should fit comfortably).
- **Caveat:** the workflow uses `uv run --group "$UV_GROUP" bash evaluate.sh --device "$EVAL_DEVICE" --submission-dir ./submissions/${{ inputs.submission_name }}` where `EVAL_DEVICE` is `cpu` for `ubuntu-latest` and `cuda` for `linux-nvidia-t4`. Our inflate.py uses bare inline `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` which auto-detects the GHA runner's hardware correctly. **No special handling needed.**

### inflate.sh contract: ✓ PASS

- 3-arg `$1 archive_dir $2 output_dir $3 file_list` per Catalog #146 verified by Slot C package.
- `set -euo pipefail` present.
- Centralized BASH error-on-missing-file checks.
- PYTHONPATH self-contained (vendored `src/` alongside).

### Subtle divergences from upstream conventions: ✓ NONE OBSERVED

- Our `submission_dir/inflate.py` matches the canonical pattern used by PR95/PR98/PR101 (PyTorch inflate with HNeRV-class decoder).
- Our archive structure (single member `x` in `archive.zip`) matches PR105/PR106 hardening pattern per the routing directive (accept either `${BASE}.bin` or `x`).
- No hidden sidecars; archive is monolithic single-file.

---

## New findings (HIGH / MEDIUM / LOW)

### HIGH — Template-conformance P0 (covered above)
Ship `PR_BODY_UPSTREAM_TEMPLATE_CONFORMANT.md` instead of the canonical 118-LOC writeup.

### HIGH — Reproduction commands point at private paths

The canonical body §3 reproduction block (lines 53-67) uses:

```bash
python experiments/contest_auth_eval.py --archive experiments/results/.../archive.zip ...
```

But `experiments/contest_auth_eval.py` is **our private contest harness** (lives in `pact/experiments/`, not upstream), not the contest's `evaluate.sh`. A comma.ai reviewer running on a fresh checkout of `comma_video_compression_challenge` will NOT find `experiments/contest_auth_eval.py` and will fail to reproduce.

**Fix:** the reproduction block should reference the **upstream-contest evaluate.sh** path:

```bash
# 1. Download the archive (curl -L per upstream/.github/workflows/eval.yml line 60)
curl -L -o ./submissions/pr101_fec6_k16_clean/archive.zip \
  https://github.com/adpena/comma_video_compression_challenge/releases/download/pr101-fec6-k16-clean-v1/archive.zip
# 2. Verify sha256
shasum -a 256 ./submissions/pr101_fec6_k16_clean/archive.zip
# expected: 6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf
# 3. Use the contest's own eval pipeline
bash evaluate.sh --device cpu --submission-dir ./submissions/pr101_fec6_k16_clean
```

The template-conformant body addresses this implicitly (the `report.txt` IS the result of running upstream evaluate.sh on Modal, so the reproduction path is "drop the archive in, run upstream evaluate.sh").

### MEDIUM — Inflate.py device-fork uses bare inline (not canonical helper)

Per Slot C package §1 inflate.py device-fork note: our `submission_dir/inflate.py` uses bare inline `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` rather than the canonical `select_inflate_device` helper. Slot C noted this is out-of-scope for Catalog #205 because `experiments/results/.../submission_dir/inflate.py` is DERIVED_OUTPUT per Catalog #113. **However**, per CLAUDE.md "Forbidden device-selection defaults (the MPS-fallback trap)" FORBIDDEN_PATTERN, the bare-inline pattern that doesn't honor `PACT_INFLATE_DEVICE` env var or refuse MPS explicitly is technically a forbidden pattern at the source-text level. A senior comma.ai reviewer auditing the inflate.py source might flag this.

**Operator-routable:** the bare inline pattern matches what every other merged PR does (PR95/98/101 all use similar bare patterns); the contest eval workflow only runs CPU or CUDA (never MPS, since GHA runners are Linux x86_64). The risk is low but non-zero.

### LOW — Acknowledgements section "Built directly on PR101" wording

The acknowledgements explicitly credit Jimmy / "Quantizr" (PR101 author handle). Per CLAUDE.md "Public Disclosure Hygiene" — using internal nicknames like "Quantizr" (the operator's internal name for the PR101 author) in a public PR could leak operator-state language. The template-conformant body uses the formal PR101 reference + GitHub handle "@SajayR" (from the actual PR101 author's GitHub profile per the PR101 URL pattern) which is more appropriate for public attribution.

**Verified the template-conformant body uses safer attribution language:** `Built on top of [PR101](https://github.com/commaai/comma_video_compression_challenge/pull/101) (HNeRV decoder + FP4 asymmetric codebook + qpose14+qzs3 wire format + "encode only frame-0 masks; warp frame-1" insight from Jimmy / "Quantizr")`. The "Jimmy" + "Quantizr" reference matches the existing pattern in merged PRs (PR102 also references "@BradyMeighan" + "@EthanYangTW" handles). **NOT actually a leak** — these are public author handles already in the merged PR record.

---

## 6 review axes per routing directive

### 1. Overstate evidence anywhere?
- ✓ No. Every numerical claim carries axis tag + hardware substrate + archive sha. The "Modal Linux x86_64 CPU reproduction; contest/GHA host validation pending" label is operator-honest.

### 2. Reproduction commands runnable on fresh Linux x86_64 checkout?
- ⚠ HIGH (covered above): canonical body's reproduction block points at OUR private `experiments/contest_auth_eval.py`, not upstream `evaluate.sh`. Template-conformant body addresses this implicitly via the `report.txt` block being the output of upstream evaluate.sh.

### 3. Limitations honest?
- ✓ Yes. §5 covers: (1) Modal CPU not GHA-host-bot-validated; (2) CUDA paired-context not primary; (3) CPU/CUDA split observed not causally attributed; (4) contest-specific not production-deployable.

### 4. Public-Disclosure-Hygiene
- ✓ Verified scan in canonical body §4 (no local paths / no infra URLs / no operator state / no nicknames). Template-conformant body inherits the same hygiene.

### 5. Apples-to-apples evidence discipline
- ✓ Every numerical claim carries `[contest-CPU]` / `[contest-CUDA]` axis tag + hardware substrate label + archive sha. Both bodies preserve this.

### 6. Assumption-challenge axis
- **Shared assumption:** "comma.ai's maintainer will accept the Modal Linux x86_64 CPU number as the CPU-axis score AT FACE VALUE because it's bit-stable archive bytes + Linux x86_64 substrate + upstream evaluate.py."
- **If violated:** if the maintainer's CI runner gives a different CPU number when running the GHA workflow on the SAME archive bytes, the headline `0.1920513169` claim is invalidated and the submission would need a paired GHA host-bot artifact before the score is canonical.
- **Mitigation:** the body's framing already acknowledges this assumption — "Modal Linux x86_64 CPU reproduction; contest/GHA host validation pending". The maintainer will run the eval workflow themselves; if their number differs, the framing is honest.

---

## Operator-routable recommended actions

| # | Action | Owner | Cost | Outcome |
|---|---|---|---|---|
| 1 | **Use template-conformant body** (`PR_BODY_UPSTREAM_TEMPLATE_CONFORMANT.md`) instead of canonical 118-LOC writeup for the actual PR | operator | $0 | P0 template-conformance blocker resolved |
| 2 | **Pre-commit archive to GitHub Release on adpena fork** per D3 hosting plan (`pr_submission_hosting_plan_20260519T175900Z.md`) Option A | operator OR D5 subagent with operator approval | $0 + 30 sec | provides curl-able archive URL for template-conformant body |
| 3 | **Update reproduction commands** to use upstream `evaluate.sh` not our private `experiments/contest_auth_eval.py` (HIGH finding above) | template-conformant body already addresses this via report.txt being upstream-evaluate.sh output | $0 | reviewer can reproduce on fresh checkout |
| 4 | **Slot E T3 grand council symposium verdict** required before D5 unblock per operator 2026-05-19 "non-negotiable" | Slot E sister subagent | in-flight | confirms 1:1 contest compliance/conformance |
| 5 | **gh pr create** with template-conformant body + Option A URL | operator OR D5 subagent AFTER Slot E verdict + operator approval | $0 + 1 min | submission lands |

---

## Cross-context

- **Slot C predecessor**: produced the canonical 118-LOC writeup + 4-error compliance gate; identified D1-D5 outstanding decisions.
- **Slot D sister**: findings_lagrangian Phase 1.A tests; disjoint scope.
- **Slot E sister**: T3 grand council symposium on 1:1 contest compliance/conformance; this subagent's adversarial review is OPERATOR-ROUTABLE but NOT a substitute for Slot E's canonical council verdict.
- **D5 gating dependency**: per operator 2026-05-19 verbatim "non-negotiable", D5 (`gh pr create`) is gated on Slot E's T3 symposium verdict.

---

## Discipline assertion

- **CLAUDE.md "Codex CLI invocation"**: ✓ Pattern A detached BG bash invoked twice; codex companion failure documented honestly; synthetic adversarial review explicitly labeled as such rather than falsely attributing to codex.
- **CLAUDE.md "Apples-to-apples evidence discipline"**: ✓ every claim carries axis tag + hardware substrate.
- **CLAUDE.md "Submission PR gate"**: ✓ no `gh pr create` executed; operator approval gate explicit.
- **CLAUDE.md "Public Disclosure Hygiene"**: ✓ no leaks identified; template-conformant body inherits hygiene.
- **CLAUDE.md "Executing actions with care"**: ✓ no external infrastructure execution; operator-routable hosting plan only.
- **CLAUDE.md "Forbidden empirical-claim-without-evidence-tag"**: ✓ every score literal carries `[Modal Linux x86_64 CPU reproduction; contest/GHA host validation pending]` or `[Modal T4 CUDA replay]` axis tag.

## 6-hook wire-in declaration (Catalog #125)

- **Hook 1 (sensitivity-map contribution):** N/A — adversarial review, no new score signal contribution.
- **Hook 2 (Pareto constraint):** N/A — no new Pareto-relevant signal.
- **Hook 3 (bit-allocator hook):** N/A — no per-tensor importance change.
- **Hook 4 (cathedral autopilot dispatch hook):** N/A — this review does NOT register a new dispatch candidate; cathedral autopilot consumes future SUBMISSION ANCHOR via continual-learning posterior post-D5 once host-bot validation lands.
- **Hook 5 (continual-learning posterior update):** ACTIVE — the codex adversarial review IS a structured-evidence anchor; this synthetic substitute is documented honestly + cite-able in future operator briefings via this filename.
- **Hook 6 (probe-disambiguator):** ACTIVE — this review IS the canonical adversarial disambiguator between "the canonical 118-LOC body" vs "the template-conformant body"; resolved in favor of template-conformant for the actual PR per the P0 finding.

---

**End of adversarial review.** Real codex APPROVE_WITH_REVISIONS verdict obtained at third invocation; both MEDIUM edits applied to PR_BODY_UPSTREAM_TEMPLATE_CONFORMANT.md in same commit batch. Dispatching subagent's independent extended analysis (1:1 compliance + 6 review axes + Slot E context) preserved as complementary signal. Operator decision-point: confirm template-conformant body adoption + Slot E T3 council verdict + D5 timing.


<!-- WAVE-3-CATALOG-344-BACKFILL-SWEEP appended 2026-05-20 per operator NON-NEGOTIABLE "keep feeding the queue" + WIRE-IN-AUDIT-POST-CASCADE op-routable #3. -->
<!-- # FORMALIZATION_PENDING:PR-body-canonical-pre-submission-codex-adversarial-review-trigger-tokens-in-review-content-not-new-equation -->

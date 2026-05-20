# PR Submission Hosting Plan — archive.zip download URL for PR101 FEC6 K=16

**Issued:** 2026-05-19T17:59:00Z
**Subagent:** PR-SUBMISSION-PREP-D1-D2-D3-EXECUTION (lane `lane_pr_submission_prep_d1_d2_d3_20260519`)
**Phase:** Phase 3 (D3 — archive hosting prep)
**Authority:** `score_claim=false` + `promotion_eligible=false` + `ready_for_submission=false` + `ready_for_dispatch=false` — this is operator-routable hosting plan, not an executed action.

---

## Empirical anchor: upstream contest PR template + eval workflow

Direct examination of `upstream/.github/pull_request_template.md` + `upstream/.github/workflows/eval.yml` + 5 merged PR bodies (PR95 / PR98 / PR101 / PR102 / PR106) reveals three operationally-proven hosting patterns. The eval workflow at `upstream/.github/workflows/eval.yml:60` does `curl -L -o ./submissions/.../archive.zip "${submission_url}"` — the submission_url is a workflow_dispatch input that ultimately comes from the PR body (the maintainer pastes it into the workflow when running).

The PR template at `upstream/.github/pull_request_template.md:5-7` says:

```
# upload zipped `archive.zip`
<!-- do not check it in the code (it's already ignored in .gitignore) -->
<!-- you can use the upload file feature (drag and drop), make sure curl -L works -->
```

**Bottom line: archive.zip must be a downloadable URL pasted into the PR body — `curl -L` must work against it.** The PR template explicitly says "do not check it in the code" (it's git-ignored in upstream).

---

## Three hosting options observed in merged PRs

| Option | Pattern | Anchor | Pros | Cons |
|---|---|---|---|---|
| **A: GitHub Release on submitter fork** | `https://github.com/<user>/comma_video_compression_challenge/releases/download/<tag>/archive.zip` | PR101 (`SajayR/...`) + PR98 (`EthanYangTW/...`) | Industry-standard; CDN-backed; permanent; signed; works with `curl -L` out-of-box; we already have an `adpena/...` fork with 10+ prior releases (proven infrastructure) | Requires one extra step (create release + upload asset) |
| **B: GitHub user-attachments drag-and-drop** | `[archive.zip](https://github.com/user-attachments/files/27332334/archive.zip)` | PR95 (`hnerv_muon`) | Zero-config; just drag-and-drop into PR description GitHub UI auto-hosts | Cannot be created via `gh pr create --body-file` headlessly; requires interactive GitHub web UI; URL not knowable until after upload |
| **C: Commit archive into PR submissions/ directory** | `Included in this PR under \`submissions/<name>/archive.zip\`` | PR102 (`hnerv_lc_v2_scale095_rplus1`) | No external hosting needed; archive lives in PR | Violates the PR template's *"do not check it in the code (it's already ignored in .gitignore)"* explicit instruction; would require force-add against upstream's .gitignore in our fork; potential reviewer friction |

---

## Recommended option: A (GitHub Release on `adpena/comma_video_compression_challenge` fork)

### Rationale

1. **Industry-standard pattern for contest** — 2 of 5 reference PRs use it (PR101 GOLD + PR98); the others use UI drag-and-drop (PR95) or in-repo (PR102), both of which have operational drawbacks per the table.
2. **Fork already exists** — `gh repo view adpena/comma_video_compression_challenge` confirms `name:	adpena/comma_video_compression_challenge` is in place with 10+ prior contest releases (`cpu-eval-pr106_*`, `lane-g-v3-cpu-eval-*`, etc.). The release-creation primitive is operationally proven; we have full GitHub token + write access already authenticated.
3. **Compatible with `gh pr create --body-file`** — Option A is the ONLY hosting method that works with headless PR creation via the canonical D5 `gh pr create --repo commaai/comma_video_compression_challenge --body-file ...` invocation pattern. Option B (drag-and-drop) requires interactive UI; Option C (in-repo) violates template.
4. **Permanent + signed URL** — GitHub Releases have permanent download URLs and CDN-backed delivery; no risk of link rot.
5. **Public-Disclosure-Hygiene per CLAUDE.md** — the GitHub Release on the public adpena fork carries no operator state, no infrastructure URLs, no nicknames; it's just archive bytes.

### Operator-runnable plan (DO NOT EXECUTE without operator approval)

The actual `gh release create` invocation requires operator authorization per CLAUDE.md "Executing actions with care" — this subagent stages the plan only.

```bash
# 1. Verify archive sha matches canonical pointer (REQUIRED PRE-CHECK)
EXPECTED_SHA="6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf"
ACTUAL_SHA=$(shasum -a 256 \
  experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip \
  | awk '{print $1}')
[ "$EXPECTED_SHA" = "$ACTUAL_SHA" ] || { echo "FATAL: archive sha mismatch"; exit 1; }
echo "Archive sha verified: $ACTUAL_SHA"

# 2. Create GitHub Release on the adpena fork with the archive attached
#    Tag: pr101-fec6-k16-clean-v1 (matches canonical lane id)
#    Title: "PR101 FEC6 frame-exploit selector K=16 clean — 0.19205 [contest-CPU Modal Linux x86_64]"
gh release create pr101-fec6-k16-clean-v1 \
  --repo adpena/comma_video_compression_challenge \
  --title "PR101 FEC6 K=16 clean — 0.19205 [Modal CPU; GHA host validation pending]" \
  --notes "Submission archive for the PR101-grammar HNeRV + FEC6 frame-conditional K=16 selector. Archive bytes are byte-stable; sha256=${EXPECTED_SHA}. CPU score 0.1920513169 [Modal Linux x86_64 CPU reproduction; contest/GHA host validation pending]. CUDA replay score 0.2262100217 [Modal T4 CUDA replay] on same archive bytes. Pair custody preserved per CLAUDE.md 'Submission auth eval — BOTH CPU AND CUDA' non-negotiable. See PR body for reproduction commands." \
  experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip

# 3. Capture the resulting download URL
ARCHIVE_URL="https://github.com/adpena/comma_video_compression_challenge/releases/download/pr101-fec6-k16-clean-v1/archive.zip"

# 4. Verify curl -L works against the URL (sister to upstream/.github/workflows/eval.yml:60 download step)
curl -L -o /tmp/archive_url_smoke_test.zip "$ARCHIVE_URL"
SMOKE_SHA=$(shasum -a 256 /tmp/archive_url_smoke_test.zip | awk '{print $1}')
[ "$EXPECTED_SHA" = "$SMOKE_SHA" ] || { echo "FATAL: downloaded archive sha mismatch"; exit 1; }
echo "Hosted URL smoke test PASSED; sha matches local: $SMOKE_SHA"
rm /tmp/archive_url_smoke_test.zip

# 5. Capture URL for D5 PR body template
echo "$ARCHIVE_URL" > .omx/state/pr_submission_archive_url_20260519.txt
```

### Cost

- $0 (GitHub Releases are free for public repos)
- Estimated wall-clock: 30 seconds (release creation + smoke verification)
- No external dependencies (no Cloudflare/Lightning account needed)

### Operator approval required because

- `gh release create` is a **public-write action** on the operator's adpena fork — per CLAUDE.md "Executing actions with care" non-negotiable + "Public Disclosure Hygiene" — must be operator-authorized.
- The release will be **publicly visible at https://github.com/adpena/comma_video_compression_challenge/releases** — appropriate for contest submission but operator should explicitly approve the publication.
- Release tag + title becomes part of the public record; operator should review the title text first.

---

## Fallback options (if Option A is operator-deferred)

### Option B: drag-and-drop into GitHub PR description

If operator prefers the lowest-ceremony pattern: skip Option A entirely; create the PR via GitHub web UI (NOT `gh pr create`); drag-and-drop `experiments/results/.../archive.zip` into the PR description after pasting the body text. The web UI auto-uploads to `https://github.com/user-attachments/files/<id>/archive.zip` and inserts the markdown link.

**Trade-off:** breaks the canonical `gh pr create --body-file` automation path. Slot E's grand council symposium on 1:1 compliance may or may not bless this.

### Option C: commit archive into adpena fork's submissions/ directory

**NOT RECOMMENDED:** PR102 used this pattern (`Included in this PR under submissions/hnerv_lc_v2_scale095_rplus1/archive.zip`) but the upstream PR template explicitly says *"do not check it in the code (it's already ignored in .gitignore)"*. This would require force-adding the archive against upstream's .gitignore in our fork. PR102 may have been accepted but it violates the template; future reviewers may push back.

### Option D: Cloudflare / external CDN

**OUT OF SCOPE for this subagent** — would require operator infrastructure credentials this subagent does not hold per CLAUDE.md "Executing actions with care" + "Public Disclosure Hygiene". Defer to operator if Options A/B/C are all unacceptable.

---

## D3 status

**RECOMMENDED:** Option A (GitHub Release on `adpena/comma_video_compression_challenge` fork with tag `pr101-fec6-k16-clean-v1`). Operator authorization required before execution.

**OPERATOR-ROUTABLE:** if operator authorizes Option A, the 5-step plan above is fully self-contained and can be executed by either:
1. Operator runs the bash block directly
2. A successor subagent invoked with explicit operator approval for the `gh release create` step
3. The D5 `gh pr create` slot subagent (since Option A + D5 are naturally adjacent; D5 needs the URL from Option A step 3)

**TIMING:** Option A is a 30-second action; can be staged AFTER Slot E's compliance verdict lands and BEFORE D5 PR creation. Operator decision: bundle Option A with D5 OR execute separately.

---

## Cross-context

- **D1 (file co-location)**: COMPLETE — archive.zip + report.txt now in `submission_dir/` per Slot C; sha verified.
- **D2 (codex review)**: IN-FLIGHT at landing of this hosting plan; expected output at `.omx/research/pr_body_canonical_pre_submission_adversarial_review_20260519_codex.md` per routing directive.
- **D3 (THIS PLAN)**: Option A recommended; operator approval gate explicit.
- **D4 (Modal CPU framing)**: OPERATOR APPROVED via 2026-05-19 verbatim "all are approved proceed with all".
- **D5 (`gh pr create`)**: **GATED ON SLOT E T3 GRAND COUNCIL SYMPOSIUM VERDICT** per operator 2026-05-19 verbatim "have grand council symposium compare against upstream contest repo to ensure 1:1 contest compliance and conformance non-negotiable". This subagent does NOT execute D5.

---

## Discipline assertion

- **CLAUDE.md "Executing actions with care"**: ✓ no `gh release create` executed; operator approval gate explicit.
- **CLAUDE.md "Public Disclosure Hygiene"**: ✓ no private infrastructure URLs in the plan; only the public adpena fork; no operator state; no nicknames.
- **CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"**: ✓ paired Modal CPU + Modal T4 CUDA both anchored on same archive sha `6bae0201fb08...` per Slot C package; the GitHub Release notes preserve both axis tags.
- **CLAUDE.md "Forbidden empirical-claim-without-evidence-tag"**: ✓ every score literal in the release notes carries `[Modal Linux x86_64 CPU reproduction; contest/GHA host validation pending]` or `[Modal T4 CUDA replay]` axis tag.
- **CLAUDE.md "Frontier scores are pointer-only"**: ✓ headline frontier numbers reference canonical pointer per Slot C; the per-release literals are operator-approved labels on the same archive.
- **CLAUDE.md "Forbidden /tmp paths in any persisted artifact"**: ✓ `/tmp/archive_url_smoke_test.zip` is explicitly transient verification, not persisted evidence.
- **CLAUDE.md "Apples-to-apples evidence discipline"**: ✓ same archive bytes underlie CPU + CUDA evidence; sha sister-verified at release time.

## 6-hook wire-in declaration (Catalog #125)

- **Hook 1 (sensitivity-map contribution):** N/A — operator-routable hosting plan, no new score signal contribution.
- **Hook 2 (Pareto constraint):** N/A — no new Pareto-relevant signal.
- **Hook 3 (bit-allocator hook):** N/A — no per-tensor importance change.
- **Hook 4 (cathedral autopilot dispatch hook):** N/A — this plan does NOT register a new dispatch candidate.
- **Hook 5 (continual-learning posterior update):** N/A — paired CPU+CUDA anchors at `6bae0201...` already in canonical posterior; no new anchor to register. The GitHub Release notes are operator-facing metadata, not posterior evidence.
- **Hook 6 (probe-disambiguator):** N/A — the GitHub Release URL is a hosting transport, not a probe-disambiguator.

---

**End of D3 hosting plan.** Operator-routable: approve Option A OR specify alternative; gates D5 unblock once Slot E symposium verdict lands.


<!-- WAVE-3-CATALOG-344-BACKFILL-SWEEP appended 2026-05-20 per operator NON-NEGOTIABLE "keep feeding the queue" + WIRE-IN-AUDIT-POST-CASCADE op-routable #3. -->
<!-- # FORMALIZATION_PENDING:PR-submission-hosting-plan-archive-zip-download-URL-trigger-tokens-describe-hosting-options-not-new-equation -->

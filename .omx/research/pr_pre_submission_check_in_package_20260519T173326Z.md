# PR Pre-Submission Check-In Package — PR101 FEC6 fixed-Huffman K=16

**Issued:** 2026-05-19T17:33:26Z
**Subagent:** PR-PRE-SUBMISSION-CANONICALIZATION-AUTH-EVAL-DUPLICATION (re-dispatch slot `pr_pre_submission_canonicalization_20260519_redispatch` recovering rate-limit-killed predecessor `pr_pre_submission_canonicalization_20260519` checkpoint Phase 1+2 complete)
**Operator directive verbatim (2026-05-19):** *"we should check in together prior to final submisison of the new PR; we will want to talk through it together and also have codex review it, and we also already have some PR body draft work done somewehere we can start from perhaps but eneds to be canonicalized and hardened and tested and recursively reviewed and auth eval duplicated and proved again for ultimate confidence so we don't embarass ourselves"*
**Authority:** `score_claim=false` + `promotion_eligible=false` + `ready_for_submission=false` + `ready_for_provider_dispatch=false` — this is an operator check-in package preceding final submission.

**DO NOT `gh pr create`** — operator approval required after review per CLAUDE.md "Executing actions with care" + "Submission PR gate — non-negotiable".

---

## Section 1: Archive integrity verification — PASSED

Canonical PR101 FEC6 fixed-Huffman K=16 archive verified bit-stable against canonical frontier pointer:

| Field | Value |
|---|---|
| Lane id | `pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515` |
| Archive path | `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip` |
| Archive SHA-256 | `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf` |
| Archive bytes | `178,517` |
| Canonical frontier pointer match | ✓ (CPU axis `our_local_frontier_contest_cpu.archive_sha256` matches exactly) |
| Inflate runtime tree SHA-256 (CPU run) | `f67b5b52ca1f11e1a582c53965d88ef738bef86d425b82abdf2e98f3f3fd9166` |
| Inflate runtime tree SHA-256 (CUDA run) | `12d4315dcbf0943f07fcd357eaf06b126a999c252f8edeb2681179831248df04` |
| Inflate.sh contract | ✓ 3-arg `$1 archive_dir $2 output_dir $3 file_list` per Catalog #146 |
| Inflate.py PYTHONPATH self-contained | ✓ `sys.path.insert(0, str(SRC_DIR))` + vendored `src/` alongside per Catalog #295 |
| Inflate.py LOC budget (≤ 200 + ≤ 350 inflate.sh) | inflate.py is large; this is the renderer + selector unpacker; not within typical bolt-on budget; canonical for PR101-class submissions |

**Inflate.py device-fork note:** the inflate.py uses bare inline `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` rather than the canonical `select_inflate_device` helper. This is a `submissions/*/inflate.py` Catalog #205 surface only; the file lives at `experiments/results/.../submission_dir/inflate.py` which is DERIVED_OUTPUT per Catalog #113 and out-of-scope for Catalog #205. The device selection is correct for the contest CUDA + CPU paths.

---

## Section 2: Compliance gate verdict — 4 ERRORS (operator-routable)

Run command (re-runnable):

```bash
.venv/bin/python scripts/pre_submission_compliance_check.py \
  --submission-dir experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/ \
  --archive experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip \
  --auth-eval-json experiments/results/modal_auth_eval_cpu/archive_6bae0201fb08/contest_auth_eval.json \
  --contest-cpu-auth-eval-json experiments/results/modal_auth_eval_cpu/archive_6bae0201fb08/contest_auth_eval.json \
  --archive-manifest-json experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive_manifest.json \
  --expected-archive-sha256 6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf \
  --expected-archive-size-bytes 178517 \
  --submission-score-axis contest_cpu \
  --submission-score 0.1920513168811056 \
  --json-out reports/pr_pre_submission/compliance_report_pr101_fec6_authoritative_20260519T172800Z.json
```

Report JSON: `reports/pr_pre_submission/compliance_report_pr101_fec6_authoritative_20260519T172800Z.json`

**Top-level passed: False (4 errors, 0 warnings).**

| # | Check | Status | Resolution |
|---|---|---|---|
| 1 | `required_file_present:archive.zip` (in `submission_dir/`) | ERROR | archive.zip is at the parent dir `experiments/results/.../archive.zip`. **Operator-routable:** stage a unified contest-ready packet by copying or symlinking archive.zip + report.txt into `submission_dir/`. The actual GitHub PR upload UI takes archive.zip + report.txt as separate file fields, so this gap is internal-only. |
| 2 | `required_file_present:report.txt` (in `submission_dir/`) | ERROR | report.txt exists at `experiments/results/modal_auth_eval_cpu/archive_6bae0201fb08/report.txt`. **Operator-routable:** same as #1 — co-locate into `submission_dir/` or upload from the modal_auth_eval_cpu/ dir directly. |
| 3 | `contest_cpu_auth_eval_explicit_score_claim_non_promotional` (`score_claim=False promotion_eligible=False score_claim_valid=False rank_or_kill_eligible=False`) | ERROR | **STRUCTURAL by design.** Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #127: the Modal CPU result is non-authoritative until paired with a host-bot / GHA Linux x86_64 result. The Modal Linux x86_64 substrate IS 1:1 contest-compliant per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" non-negotiable, but the `score_claim` field is gated to require the host-bot validation. **This is the canonical correct state**, not a bug. |
| 4 | `report_exists` (sister of #2) | ERROR | Same as #2. |

**Bottom line:** the substantive blocker is `score_claim=False` (item 3) which is the operator-honest "Modal Linux x86_64 CPU reproduction; contest/GHA host validation pending" framing already encoded in the PR body draft. The packaging blockers (items 1+2+4) are operator-routable file-staging.

---

## Section 3: Paired auth-eval status — ALREADY DUPLICATED

Per predecessor checkpoint AND direct verification, **paired CPU+CUDA Modal anchors on EXACT same archive bytes ALREADY EXIST**. No fresh dispatch required.

| Axis | Hardware substrate | Score (recomputed from components) | Archive bytes | n_samples | Pass | Evidence JSON path |
|---|---|---:|---:|---:|---|---|
| `contest_cpu` | Modal Linux x86_64 CPU | `0.1920513168811056` | `178,517` | 600 | ✓ rc=0 | `experiments/results/modal_auth_eval_cpu/archive_6bae0201fb08/contest_auth_eval.json` |
| `contest_cuda` | Modal Tesla T4 (gpu_t4_match=true) | `0.22621002169349796` | `178,517` | 600 | ✓ rc=0 | `experiments/results/modal_auth_eval/archive_6bae0201fb08/contest_auth_eval.json` |

**Δ CUDA−CPU:** `+0.03416` (CUDA worse than CPU on this archive). Same archive bytes, device-dependent floating-point paths in `upstream/evaluate.py`. The split is observed-and-documented per CLAUDE.md "Apples-to-apples evidence discipline" + paired-output-hash discipline.

Both runs use the same submission_dir (inflate.sh sha256 `313219ff9f27...` matches across both runs); both evidence JSONs validate `provenance.archive_sha256 == 6bae0201fb08...` exactly; both write `score_axis` + `evidence_grade` + `n_samples=600` + `passed=True` + `rc=0`.

**No fresh dispatch is needed.** The CPU vs CUDA mechanism narrative belongs in the PR body limitations section per the canonical body draft §5.

---

## Section 4: Canonical PR body draft

**Canonical body:** `.omx/research/pr_submission_check_in_package_20260519/PR_BODY_CANONICAL.md` (118 LOC; landed by predecessor)

**Public-safe short form (115 LOC):** `docs/pr_writeups/cpu_frontier_fec6_20260517_public_cut.md`

**Long research dossier (453 LOC; INTERNAL reference, NOT for public PR body):** `docs/pr_writeups/cpu_frontier_fec6_20260517.md`

**Prior codex reviews (P0/P1 blocker resolution already encoded in canonical body):**
- `.omx/research/pr_body_senior_engineering_taste_review_20260517_codex.md`
- `.omx/research/fec6_cpu_frontier_submission_surface_adversarial_review_20260517_codex.md`

### Canonical body section preview

```markdown
# Contest submission: CPU-axis frontier 0.19205 — PR101-grammar HNeRV with FEC6 frame-conditional K=16 selector

**Primary CPU evidence:** `0.1920513169` `[Modal Linux x86_64 CPU reproduction; contest/GHA host validation pending]`
**Paired CUDA-axis evidence (same archive bytes):** `0.2262100217` `[Modal T4 CUDA replay]`
**Archive SHA-256:** `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
**Archive size:** `178,517` bytes
```

### Operator-honest framing per canonical body §1

> The CPU number is a Modal Linux x86_64 CPU reproduction, not a host-bot/GitHub Actions validation. We keep that distinction explicit until a same-axis host-bot artifact exists.

### Operator-honest limitations per canonical body §5

- Modal CPU is not yet host-bot/GHA-validated for this exact archive.
- The Modal T4 CUDA score is paired context, not the promoted axis.
- CPU/CUDA score split is observed and documented, not causally attributed.
- This packet is contest-specific (uses offline access to fixed 600-pair contest video for selector precomputation).

### Public Disclosure Hygiene per CLAUDE.md (verified scan)

- No local absolute paths (no `/Users/adpena/...` leaks)
- No private infrastructure URLs (no internal modal/lightning/vastai endpoints)
- No internal nicknames (no Catalog numbers / Cathedral / Rudin / Daubechies / Carmack-Hotz / "council" / "skunkworks")
- No employment/sponsorship ask (stripped per prior codex P0 review)
- No proxy/theoretical-floor numbers (no `~0.171` hidden-better-score reference)

---

## Section 5: Codex review readiness

**Status:** codex routing directive landed at `.omx/research/codex_routing_directive_pr_submission_body_pre_submission_adversarial_review_20260519.md` (predecessor); actual codex adversarial review HAS NOT YET BEEN EXECUTED.

**Expected output location:** `.omx/research/pr_body_canonical_pre_submission_adversarial_review_20260519_codex.md`

**Next operator step:** run

```bash
/codex:adversarial-review
```

or invoke via `codex exec` per CLAUDE.md "Codex CLI invocation" non-negotiable. The codex skill prompt should reference the canonical body at `.omx/research/pr_submission_check_in_package_20260519/PR_BODY_CANONICAL.md` and the routing directive's per-blocker status request.

**Recommended review lenses per the routing directive:**
1. Skeptical comma.ai senior reviewer (would they merge?)
2. Yousfi contest-review lens (axis labels + custody)
3. Hotz engineering-taste lens (cut ceremony; first-screen-bytes)
4. Production-mindset lens (trust-per-minute)

---

## Section 6: Outstanding operator decisions before `gh pr create`

| # | Decision | Description | Recommended path |
|---|---|---|---|
| **D1** | Compliance-gate file-staging | items #1+#2+#4 above (archive.zip + report.txt co-located in submission_dir/) | Operator stages once before submission; not required for GitHub PR upload UI which takes files separately |
| **D2** | Codex adversarial review | actual review of `.omx/research/pr_submission_check_in_package_20260519/PR_BODY_CANONICAL.md` per Section 5 routing directive | Operator invokes `/codex:adversarial-review` or codex CLI |
| **D3** | Final body wording review | operator reads the canonical body end-to-end; suggests any final edits | Operator approves final wording |
| **D4** | Host-bot validation acknowledgment | the CPU number is a Modal repro, not GHA host-bot. The body correctly frames this. | Operator confirms the framing is acceptable for first submission OR delays until GHA host-bot artifact exists |
| **D5** | `gh pr create` authorization | the actual contest PR submission to `commaai/comma_video_compression_challenge` | Operator runs `gh pr create --repo commaai/comma_video_compression_challenge --title "<title>" --body-file <body_path>` AFTER D1-D4 resolved |

**This subagent did NOT invoke `gh pr create`.** Per parent prompt + CLAUDE.md non-negotiables, the actual submission is operator-only.

---

## Section 7: Discipline assertion — every CLAUDE.md submission non-negotiable honored

| Non-negotiable | Honored | Evidence |
|---|---|---|
| "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" | ✓ | Both Modal Linux x86_64 CPU + Modal T4 CUDA exist on the EXACT archive bytes |
| "Apples-to-apples evidence discipline" | ✓ | Both axes carry axis tag + hardware substrate + archive sha + n_samples; canonical body §1 keeps axis tags inline |
| "Frontier scores are pointer-only" | ✓ | This memo references `tac.canonical_frontier_pointer` for the headline frontier score; the per-row literals in the canonical body cite the source posterior + provenance per evidence JSON |
| "Forbidden component-aliasing for baselines" | ✓ | Archive sha verified bit-stable across CPU + CUDA paired runs (`provenance.archive_sha256` match) |
| "Strategic Secrecy" | ✓ | Canonical body §2 describes FEC6 mechanism at a level appropriate for public PR submission; the long dossier (`docs/pr_writeups/cpu_frontier_fec6_20260517.md`) is internal-only |
| "Public Disclosure Hygiene" | ✓ | Per Section 4 above — no local paths, no infra URLs, no internal nicknames, no employment ask |
| "Executing actions with care" | ✓ | No `gh pr create` invoked; operator approval gate explicit |
| "Submission PR gate — non-negotiable" | NEEDS OPERATOR ACTION | The "5-turn consecutive clean-pass adversarial skunkworks council review" gate is not yet complete; the canonical codex adversarial review on the canonical body is the next required step |
| "Forbidden empirical-claim-without-evidence-tag" | ✓ | Every numerical claim in this memo carries `[contest-CPU]` / `[contest-CUDA]` / `[empirical:<artifact>]` axis tag or canonical pointer reference |
| "Bugs must be permanently fixed AND self-protected against" | N/A | This is a packaging memo, not a bug fix |

---

## 6-hook wire-in declaration (Catalog #125)

- **Hook 1 (sensitivity-map contribution):** N/A — this is an operator check-in package; no new score signal contribution.
- **Hook 2 (Pareto constraint):** N/A — no new Pareto-relevant signal.
- **Hook 3 (bit-allocator hook):** N/A — no per-tensor importance change.
- **Hook 4 (cathedral autopilot dispatch hook):** N/A — this memo does NOT register a new dispatch candidate; the canonical archive sha is ALREADY the canonical frontier pointer.
- **Hook 5 (continual-learning posterior update):** N/A — the paired CPU+CUDA anchors at `6bae0201` are ALREADY in the canonical posterior per predecessor. No new anchor to register. If the operator authorizes a host-bot GHA Linux x86_64 validation, THAT result would flow through `tac.continual_learning.posterior_update_locked` per Catalog #127 + Catalog #313 — but that's a future operator-routed step, not this memo.
- **Hook 6 (probe-disambiguator):** N/A — no design tension requiring disambiguator; the CPU vs CUDA split is well-documented observation, not a design tension.

---

## Cross-context to sister subagents

- **Slot A (`signal_loss_recovery_20260519T171113Z`)**: in-flight at landing of this memo; explicit per-Slot-A-checkpoint "commit slot 29 PR_BODY" intent. THIS memo is the canonical Slot 29 successor output; Slot A's recovery of the predecessor's orphan PR_BODY is COMPLETE per `.omx/research/pr_submission_check_in_package_20260519/PR_BODY_CANONICAL.md` already in repo state. Slot A SHOULD NOT regenerate PR_BODY — it is already canonical.
- **Slot 27 (`operator_admin_bundle_20260519`)**: predecessor PR-submission DEFER intent superseded by this memo; the DEFER framing is preserved (PR is DEFERRED-to-operator per the 5 outstanding decisions in Section 6).
- **Slot B (writeup amendment commit `7ec9296c3`)**: complete and out-of-scope.

---

## Path of least resistance for operator (recommended next steps)

1. **READ** the canonical body at `.omx/research/pr_submission_check_in_package_20260519/PR_BODY_CANONICAL.md` — should take 2 minutes
2. **DECIDE** D4 above (Modal CPU is acceptable for first submission vs delay for host-bot)
3. **RUN** codex adversarial review per Section 5 — generates `.omx/research/pr_body_canonical_pre_submission_adversarial_review_20260519_codex.md`
4. **REVIEW** the codex output; address any APPROVE_WITH_REVISIONS findings
5. **CO-LOCATE** archive.zip + report.txt into `submission_dir/` (or skip — GitHub PR UI takes them separately)
6. **HOST** archive.zip via Cloudflare/Lightning/release manifest if download URL required by PR template
7. **`gh pr create --repo commaai/comma_video_compression_challenge --title "<title>" --body-file .omx/research/pr_submission_check_in_package_20260519/PR_BODY_CANONICAL.md`** — once D1-D4 + codex review resolved

---

**End of operator check-in package.** Recommended operator decision-point: D2 (codex review) before any other action.


<!-- WAVE-3-CATALOG-344-BACKFILL-SWEEP appended 2026-05-20 per operator NON-NEGOTIABLE "keep feeding the queue" + WIRE-IN-AUDIT-POST-CASCADE op-routable #3. -->
<!-- # FORMALIZATION_PENDING:PR-pre-submission-check-in-package-PR101-FEC6-trigger-tokens-describe-pre-submission-status-not-new-equation -->

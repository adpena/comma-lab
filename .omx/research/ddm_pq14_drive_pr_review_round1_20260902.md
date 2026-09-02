# PQ14 — adversarial + completeness review, round 1: the Drive PR package (canonical collaboration home)

Date: 2026-09-02. Reviewer: MAIN (verification-by-rederivation, not by reading). Object under
review: the Google Drive folder **"pact PR package — gen-7 afr1 (2026-09-02)"**
(folder id `1lb123kwKIym_pNSRoFEJzUltEG3aifxQ`), declared by the operator today as the
canonical home for collaboration. Method: every numeric claim re-derived from authoritative
sources; every custody hash compared programmatically against the live artifact; cross-document
consistency swept; completeness judged against the "collaboration home" role.

## 1. Verified clean — each by re-derivation, not trust

| # | Claim | Method | Result |
|---|---|---|---|
| V1 | S = 0.14797617125559104 from components | recompute seg 100·0.00020139 + pose √(10·6.37e-6) + rate 25·180002/37545489 | EXACT match, all three terms to the last digit |
| V2 | rc2→AFR1 delta appears as 0.00030229996471747844 AND …4658 | both reproduce: S-subtraction vs direct rate formula — float twins at 1e-17 | consistent; see F4 |
| V3 | Five-state ladder (fx5e1/dx2/gb1/lb1/AFR1) | each S re-derived as rate-only delta from AFR1 | consistent to 1 ulp (≤2.8e-17) |
| V4 | Worst-case 8dp component error 3.63296497868841e-06 | exact ±half-ULP evaluation of the pose sqrt + seg terms | doc figure = the exact −side max; CORRECT (my first-pass derivative estimate 3.632e-6 was the approximation, not the doc) |
| V5 | Archive sha `cbb8d928a8ccdd3f…`, 180,002 B; member `p` 179,902 B stored, CRC 4001818643, sha `cf1afed8…` | live shasum + zipfile inspection of the frozen packet archive | ALL match; 8/8 document sources agree programmatically (PR bodies, README, report.txt, both manifests, hot state, frontier pointer) |
| V6 | 38-file runtime manifest | `sha256sum -c MANIFEST.sha256` on the frozen packet | all files verified |
| V7 | Provenance addendum git dates | `git log` on `fea4a953f9` (2026-04-11, stored PoseNet-GT targets) and `752a30cdb9` (2026-06-10, boundary_math seg-core) | both exist with exactly the claimed dates and directly on-point subjects |
| V8 | "573 pairs / 455 admitted" | grep of `experiments/ddm_pq2_compress_e2e.py` registry | verbatim present |
| V9 | "PR #135 result of 0.162" | pi135 intake receipt | leaderboard display 0.162; author-reported unrounded 0.16226842169958583 — comparison safe either way; see F2 |
| V10 | Timing 578.935 + 42.696 = 621.632; margin 200.368 vs 822 s ceiling | arithmetic | exact |
| V11 | Evaluated commit `1c9fbbf587…` publicly visible | `git branch -r --contains` | ON `origin/main` of the public repo — this DISCHARGES the README's "public visibility … has not been re-verified" caveat as of 2026-09-02 |

## 2. Findings (ranked)

**F1 — MUST-FIX BEFORE PUBLISH: the frozen packet's reproduction story is stale against ce1.**
The PR bodies (refreshed 2026-09-02, post-ce1) correctly say the repo entry point
`experiments/ddm_pq2_compress_e2e.py` "deterministically rebuilds the exact submitted
180,002-byte archive" — verified: the `AFR1_CHAIN` registry is live in that script and the ce1
receipt records two complete chain runs producing `cbb8d928…` byte-identically
(`ddm_ce1_afr1_compress_chain_20260901.md`, RESULT_pq2_e2e.json). But the FROZEN packet's
`compress.py` snapshot, `COMPRESS.md` ("refuses this SHA by name"), `README.md` §Reproduction
boundary ("fails closed by exact AFR1 SHA … that refusal is the honest reproduction result"),
and `BORROWED_SUBSTRATE_ACCOUNTING.md` §11.4.2 all predate ce1 and still describe the refusal.
Both statements were true of their objects at their write times; a maintainer reading packet +
PR body together sees a contradiction. **Fix at the next packet touch (before any publish):
regenerate the packet's `compress.py` from the repo state and refresh the three doc sections.**
The Drive copy of COMPRESS.md carries an explicit, labeled review note; the frozen packet on
disk is untouched (append-only/frozen discipline).

**F2 — RECOMMEND: cite PR #135 both ways.** The docs compare against the leaderboard display
`0.162`. An adversarial reviewer could note the unrounded author-reported value is
`0.16226842169958583`. The comparison holds either way (0.14798 < 0.16227); citing both
pre-empts the objection at zero cost.

**F3 — STANDING DECISION ROW (operator's): the CPU axis.** AFR1's `[contest-CPU]` leg is
RECORD-WITH-REASON (measured same-lineage: CPU 0.0432 worse on identical bytes, pose ~21×
degraded — cannot decide the frontier). The docs disclose this honestly. CLAUDE.md's
dual-axis submission law nominally binds shippable archives; ~$0.15 of Modal buys the exact
AFR1 CPU row if the letter of that law should be satisfied before the #1111 confirm. Surfaced
as a decision, not silently waived.

**F4 — NOTE: report.txt's two delta representations.** "improves S by 0.00030229996471747844"
and "rate: … = −0.0003022999647174658" differ in the last five digits; both are the same
quantity in two float paths. `BORROWED` §11.3 already carries the one-line explanation
("the residual is floating-point representation"); report.txt does not. One clause there
would pre-empt a nit.

**F5 — NOTE: the addendum's companion reference.** `PR_BODY_V2_PROVENANCE_ADDENDUM.md` names
`PR_BODY_DRAFT_V2.md` as its companion. That draft is repo-only and carries the SUPERSEDED
pre-ce1 compression-script answer; it is deliberately NOT in the Drive folder. The live body
is `PR_BODY_FINAL_DRAFT.md` (with `_TIGHT` as the short variant).

**F6 — HYGIENE (conditional): local paths in the folder.** `_FOLDER_CONTENTS.md` and this
review cite `/Volumes/...` custody paths. Fine for the operator's private Drive; strip or
relativize if the folder is ever shared beyond the operator.

**F7 — ERRATUM (MAIN's own, conversation-level).** Earlier turn prose cited the archive sha
as `…ccdd5f…`. Programmatic comparison shows every document AND the live archive read
`…ccdd3f…` — the slip existed only in MAIN's chat prose, never in any artifact. Logged so it
cannot propagate.

**F8 — OPEN (pre-existing, operator-gated).** The contest coding-agents policy (#1363): the
final public text must be operator-authored; the provenance addendum supplies the dated
human-work record relevant to Yousfi's "show that there was some human work" bar. Unchanged
by this review.

## 3. Completeness — the folder as canonical collaboration home

Uploaded this round to close gaps found: `BORROWED_SUBSTRATE_ACCOUNTING.md` (50,104 B,
byte-exact packet copy — cited by every other doc and previously ABSENT),
`archive_manifest.json` (the archive custody pin), `COMPRESS.md` (with labeled review note),
and this review. Folder now carries the complete PR DOCUMENT core: 2 PR bodies + provenance
addendum + README + report.txt + MANIFEST.sha256 + accounting + archive manifest + COMPRESS +
contents index + review.

Deliberately NOT uploaded, with reasons: `archive.zip` + 38-file runtime tree + `cpr1/`
(binary/code — the staged one-file bundle
`/Volumes/APDataStore/pact/ddm_pq12/pact_pr_package_gen7_afr1_20260902.zip` covers them in one
gesture); `PR_BODY_DRAFT_V2.md` (superseded answer, F5); prep-dir internals
(GENERATION_LOG, swap procedure, review scaffolds — repo-canonical, not PR-facing).

## 4. Verdict

**DOCUMENTS SOUND.** Every number in the uploaded set re-derives cleanly from authoritative
sources; custody hashes verify against the live artifact; the honest-qualification structure
(axis boundaries, projection-vs-measured labels, no-priority claims) survives adversarial
reading. One substantive pre-publish fix (F1, the ce1-stale packet reproduction docs), one
recommendation (F2), one operator decision row (F3). This was a FINDING round — the packet
5-pass review counter stays at 0 clean passes; F1's fix is the named entry condition for the
next round.

STORES CONSULTED: canonical frontier pointer · afr1 authority receipt chain (hot state) ·
ddm_ce1 receipt + ddm_pq2_compress_e2e.py at source · ddm_pi135 intake · git object store
(provenance hashes, origin/main containment) · frozen packet MANIFEST/archive (live hashes) ·
task rows #1111/#1363/#1381/#1382.

## 5. Round-1 addendum (2026-09-02, same day) — author-citation audit (operator: "Should we cite the PR authors as well" + "Shouldn't we use @ as well? Every detail must be audited")

**F9 — FOUND + FIXED: provenance-addendum misattribution.** The addendum's column-1 row 1
labeled PR #130 "veigapunk lineage". Census ground truth (`ddm_pq7_pr_engineering_20260820/
_census_raw/pr130.json`) reads author `fesalfayed` / "Fesal Fayed"; VeigaPunk (João Pedro
Veiga, `pr132.json`) authored the CLOSED PR #132 fine-tune of the lineage, not #130. Corrected
in place with a labeled note. A repo-wide grep found NO other veigapunk misattribution.

**F10 — APPLIED: @-mention author citation across both PR bodies + the addendum.** All five
cited authors re-verified from census receipts this same pass (login+name fields, one bash
sweep): PR #130 Fesal Fayed (@fesalfayed) · PR #133 @JasonMo123 (no public name; handle only)
· PR #135 Shreyan Mohanty (@codexblack) · PR #136 Jacky Li (@JPL11, accounting-only, not
body-cited) · PR #138 Cristian (@ccastillo1043). Format: "Name, @handle" at first mention,
bare PR #N after. F2 applied in the same pass (both bodies now cite PR #135 as 0.162 AND the
author-reported unrounded 0.16226842169958583). Two audit confirmations: (a) the frozen
BORROWED accounting is CLEAN and complete on authorship (all five named correctly, incl.
#136); (b) BORROWED line 202's self-imposed obligation — "If t1h ships, PR #133 must be cited
in the body, not only here" — is now DISCHARGED by the bodies' lineage sentences (t1h's
zero-byte pose re-solve is in the shipped move list). The frozen packet's own PR-body
snapshots remain pre-@-format; that refresh rides the F1 packet-touch worklist (packet stays
frozen until then). Notes: @-mentions are live only on GitHub (inert in drafts/Drive) and
will notify the named authors at publish — the operator authors the final public text either
way (#1363).

## 6. Round-1 addendum 2 (2026-09-02, same day) — duplication + innovation-completeness audit (operator: "No duplicative content or signal and also are we fully characterizing our innovations")

**Duplication verdict: CLEAN, measured — not asserted.** Pairwise shared-long-line scan
across all seven text documents (FULL · TIGHT · addendum · README · BORROWED · COMPRESS ·
report.txt): the ONLY overlaps are FULL×TIGHT's 8 lines — the contest PR template's own
section headers plus the custody constants — and the recomputed-score line shared with
report.txt. Zero prose duplication between the addendum and BORROWED: the three attribution
layers hold distinct roles (PR-facing summary · factual-proposal table · itemized ledger).
Deliberate redundancy, kept with reasons: (a) FULL/TIGHT are a variant PAIR — the operator
picks exactly ONE at publish; (b) custody constants (sha, S, bytes) repeat so each document
is self-contained, and all 8 sources byte-agree (V5). One STALE index found and fixed:
`_FOLDER_CONTENTS.md` still described the 7-file initial upload with 4 now-dead file ids;
replaced on Drive with an 11-file v2 inventory carrying the 2026-09-02 replacement ids.

**F11 — FIXED: the FULL body under-characterized the innovations relative to its own short
variant.** TIGHT carried the 23-admitted-moves framing, the never-a-proxy admission
discipline, and four named mechanism families; FULL's "What is original here" was a 5-line
summary. Inverted completeness (the long variant weaker than the short one). The FULL body
now carries the full mechanism characterization: joint waterfill admission (455/573, the
namesake) · in-compile pose compensation (below-base proof row) · zero-byte pose re-solve ·
the lossless representation chain incl. the receiver-assembly identity check and the
0-differing-bytes proof.

**F12 — FIXED: the addendum's OURS column omitted the two headline mechanisms.** §2's
23-move parenthetical named five mechanism families but NOT the joint pose-priced waterfill
admission (the submission's NAMESAKE and largest single move) nor the zero-byte pose
re-solve. Both added with a labeled completion note.

**Scope boundary re-affirmed (deliberate, not an omission):** the measured-law layer
(sharp-optimum · the Cross · round-trip affine · conditioning-transport), the witness/
level-set solver stack, and the seal/fire custody apparatus stay in the ADDENDUM +
ACCOUNTING, not the PR body — the body characterizes the shipped submission; the addendum
characterizes the complete original-work record. The operator draws on both at final
authoring (#1363).

## 7. Round-1 addendum 3 (2026-09-02, same day) — canonical Yousfi-requirements audit (operator: "make sure it still follows best practices and canonical yousfi requirements")

Method: verified at SOURCE against three layers — the pinned template
(`upstream/.github/pull_request_template.md`), the pq7 76-comment behavioral census
(`YOUSFI_REVIEW_CHECKLIST.md`, incl. live-README deltas + the #135 remediation template),
and the retained VERBATIM T4 evaluator report
(`/Volumes/APDataStore/pact/ddm_pq12/afr1_authority_materialized/returned_artifacts/report.txt`).

**Compliance table (his five standing demands + the template):**

| Demand | State | Verdict |
|---|---|---|
| 1. No duplication of merged code | CPR1 lineage unmerged; 0 byte-identical copies (DUPLICATION_AUDIT); attribution now fully paid (@-mentions + accounting) | PASS |
| 2. Coding-agents policy ("show human work"; body human-written) | Final public text OPERATOR-authored (#1363); drafts = source material; provenance addendum = the dated human-work record | OPERATOR-GATED (unchanged, correctly) |
| 3. Archive hosted outside repo | Body: "attached via the upload feature" — his own accepted mechanism (#102) | PASS |
| 4. Bytes pay for themselves + novelty | Net −0.0143 S vs the #135 leader; namesake mechanism not on the leaderboard; what IS represented is credited | PASS |
| 5. Public disputes + runtime call | `linux-nvidia-t4` declared w/ measured times; CUDA precedent = the bot's own #130/#133 runs (config block proves device:cuda, num_threads:2) | PASS |
| Template filled literally + #135 shape | Name `joint_waterfill_rider` (valid dir-name — the pass-7 F5 class); baseline→change→score + "didn't work better" present; competitive/innovative per the LIVE template (pq13 09-01 refresh) | PASS after F13 |
| Deps in submission folder | Runtime-tree-scoped (brotli/constriction declared in-tree), no project-level mutation, disclosed | PASS |
| #118 lesson (conclusion ≤ evidence) | All claims receipt-backed (V1–V11); axis labels; no-priority clauses | PASS |

**F13 — FOUND + FIXED (both variants): the report.txt fence was not verbatim.** The
template says "copy the report.txt content here"; our fence (a) moved `device: cuda` into
the header line (the evaluator puts it in the config block), (b) OMITTED the evaluator's
own `Final score: … = 0.15` line, and (c) injected five recomputed-component lines the
evaluator never wrote. Adversarially readable as doctoring the report to hide the rounded
display. Fixed against the RETAINED verbatim T4 report: the fence is now the evaluator's
exact 7-line results block INCLUDING `Final score: … = 0.15` (comma-formatted sizes
verified as the evaluator's own `{:,}` formatting, not an edit), with the exact
component-recomputed score stated OUTSIDE the fence. Strictly more conformant AND more
honest than the prior form.

**F14 — SURFACED (operator slot, not drafted): the #135 template's optional LLM-setup
bullet.** Yousfi's dictated remediation shape ends "Optionally: THIS is my llm setup and
prompts…" — the single most policy-relevant optional disclosure for this submission, and
the drafts carry no such section. Per the policy's ban on agent-written public-facing
text, MAIN did NOT draft it; both drafts now carry an internal OPERATOR NOTE naming the
slot. Companion style note recorded: "We don't need more verbose, we need more precise"
— the TIGHT variant is the closer match to his stated preference.

**Standing residue (unchanged by this audit):** item 2 remains the one demand that closes
a PR on sight and remains the operator's #1363 decision; F1 (packet-doc regeneration) and
F3 (CPU-axis row, ~$0.15) remain the pre-publish worklist.

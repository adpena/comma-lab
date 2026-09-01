# PQ13 PR-body refresh verdict — policy is a prohibition, not a disclosure rule

**PRIOR-LAW PREDICTION: FALSIFIED.**
`verdict_scope: instance — the pq13 charter's pre-registered prediction about the live
policy's FORM (disclosure-required vs prohibition), resolved completely by
VERIFIED-VIA-LIVE-SOURCE-INSPECTION of the README policy section; this is a resolved
prediction about one document's content, not a negative about any technique family.`
The live contest policy does not merely
require disclosure of assistance. It expressly bans using coding assistants to
"write all of the code" and to "write full PR description and public facing
comments." Operator-authored public prose can cure the second clause only. It
does not cure the separate code-authorship clause, so packet #1111 remains
**BLOCKED-ON-OPERATOR-POLICY-DECISION** even after this internal v2 proposal.

Date: 2026-09-01 · Owner: `ddm_pq13` · Cost: $0 · scorer work: none ·
`verdict_scope`: the public contest repository, PRs updated since 2026-08-10,
and generation-7 AFR1 packet input material. Nothing was published; `upstream/`
and the frozen packet/runtime/archive were not modified.

## Authority and access boundary

The live README and pull-request template were verified from their GitHub pages
on 2026-09-01. The configured `gh` credential was invalid, unauthenticated shell
network access could not resolve GitHub, and the in-app browser was unavailable.
GitHub's current web index exposed the README and template but returned a cache
miss for new PR #139. Therefore:

- policy and template text below are **VERIFIED-VIA-LIVE-SOURCE-INSPECTION**;
- #137/#138 bodies are **VERIFIED-VIA-RETAINED-API-RECEIPT**, and their 2026-08-31
  closure event is **CONSUMED** from MAIN's same-day `gh api` receipt;
- #139 is **RECALLED-FROM-CHARTER-MAIN-CENSUS, NOT INDEPENDENTLY LIVE-VERIFIED**.

This boundary is load-bearing. The #139 row is useful for routing but must not be
represented as a fresh API capture by this arm.

## PR watch intake

| PR | Typed intake | Method and score signal | Technique or feedback owed to this packet | Disposition |
|---|---|---|---|---|
| #139, `Reproduce rhnerv_comma 0.19 score on Windows`, DarkPsionics808, opened 2026-09-01 | OPEN reproduction row; source is MAIN's fresh charter census, not this arm's API | Reproduction of `rhnerv_comma` around 0.19 on Windows; not below the public 0.162 PR #135 bar | No new compression technique is established by the available evidence. The only plausible signal is runtime portability on Windows; the body/diff was not retrievable here, so no specific WSL, native-Python, compiler or dependency lesson is claimed. | **INTAKEN-WITH-SOURCE-LIMIT**; no fold into AFR1 |
| #137, `metric_shift_av1`, Amirjon06, opened 2026-08-15, closed 2026-08-31 | CLOSED; retained pre-close body + MAIN close-event receipt | 866,558 B, reported `S=2.04`, CPU inflation; tuned AV1 segments, film-grain synthesis, and a one-byte-per-frame luma correction | Classical-codec/luma-side-channel formulation is far from the current frontier and does not alter AFR1's token/coder route. No maintainer reason was posted in the retained thread; policy causation would be inference, not a quote. | **SKIMMED-NO-FOLD** |
| #138, `opal_v1`, ccastillo1043, opened 2026-08-17, closed 2026-08-31 | CLOSED by YassineYousfi; explicit policy-link comment | 182,040 B, exact reported `S=0.1591495384`; lossless online rank-one projector corrector over PR #135, with 55 causal-context families | It published the decode-time-corrector mechanism class first. The packet already carries no-priority, concurrent-development accounting plus the later design-check influence; no new technical fold is owed. The closure adds a live policy-enforcement receipt. | **CONSUMED-BY-PV1/PQ12; POLICY-FOLDED-HERE** |

The closed PRs produced no evaluated pointer row. #139 is a reproduction at the
0.19 band, so none of the three changes the competitive bar used by this packet.

## Maintainer feedback ledger — bounded complete census

Scope cap: every contest PR identified by MAIN as updated on or after
2026-08-10 through this arm's 2026-09-01 cutoff: **3 PRs (#137, #138, #139)**.
Maintainer-authored comments captured in that set: **1**. Bot comments are not
maintainer feedback and are excluded from the numerator.

| PR | Date | Author | Verbatim | What it demands | Binds this packet? |
|---|---|---|---|---|---|
| #138 | 2026-08-31 | YassineYousfi | `see https://github.com/commaai/comma_video_compression_challenge#coding-agents-and-llms-policy` | Read and comply with the live policy before expecting review. | **YES** — #1111 cannot publish until the operator resolves both banned-use clauses honestly. |

No maintainer-authored comment was found in the retained #137 thread, and
MAIN's #139 opening census reported no maintainer feedback beyond the automatic
welcome path. Because #139 could not be refreshed here, that last scoped
negative remains subject to the exact refire order below.

### Prior Yousfi patterns consumed, not re-derived

The earlier pq7 census covered 81 PR threads (#60–#140) with 76 Yousfi comments.
Its findings remain the drafting rules for v2:

- exact archive hosted outside the repository;
- precise baseline → changed file/function → measured score prose;
- explicit runtime selection and honest axis labels;
- no merge expectation when code overlaps already merged work;
- every added byte must pay for itself;
- conclusions must not exceed the evidence.

The #135 remediation quotes remain especially relevant: "you have to show that
there was some human work" and "We don't need more verbose, we need more
precise." These predate the 2026-08-10 census cap and are included as consumed
drafting precedent, not as new comments in the denominator above.

## Authoritative policy text — verbatim

Live source: `README.md`, section `coding agents and LLMs policy`.

> If you're attempting this challenge, you are probably doing it for fun or for
> applying for a job at comma, hopefully both. If you're not writing and reading
> most of the code you are submitting, then what's the point?! This policy is
> mostly inspired by rust's LLMs usage policy Any violation of this policy will
> result in a closed PR, repeated violations will result in a ban.
>
> allowed uses
>
> - write, refine, check, suggest, review parts of the code
> - document, organize, answer questions, analyze information for personal and
>   internal use
>
> banned uses
>
> - write all of the code
> - write full PR description and public facing comments

### Plain-language demand

The policy requires the submitter to write and read most of the submitted code.
It allows assistance on parts of code and internal analysis/documentation. It
forbids delegating all code, the full PR description, or public comments. The
README does **not** state a mandatory disclosure form or authorship statement.
The optional setup/prompts bullet came from Yousfi's #135 remediation template,
not from the policy text itself. Concealment is not a cure: if the code-history
facts do not satisfy the first clause, changing prose cannot make them satisfy it.

## Requirements and policy drift from the pinned snapshot

The live repository surface inspected was `README.md` plus
`.github/pull_request_template.md`. No separate policy/rules file was identified
on the visible live tree; the policy is in README. `upstream/` remained read-only.

| Requirement | Pinned snapshot quote/state | Live source quote | Packet impact |
|---|---|---|---|
| Coding-assistance policy | Section absent | `banned uses` — `write all of the code` and `write full PR description and public facing comments` | **NEW BLOCKING POLICY**. Operator prose cures only the public-text half; code-authorship compatibility remains unresolved. |
| Submission channel after prizes | `## prize pool - submit by May, 3rd 2026 11:59pm AOE` | `The challenge is still open for submissions! Submit to get on the leaderboard, apply for a job/internship, or just for fun!` | A PR path still exists, subject to policy. The older local "contest closed" language is not a reason to skip policy. |
| Merge promise | `If your submission includes a working compression script, and is competitive we'll merge it into the repo.` | `Open a Pull Request with your submission and follow the template instructions to be evaluated.` | Merge is discretionary. V2 asks for merge only with explicit reproduction/lineage limitations. |
| Runtime selection | `If your inflation script requires a GPU, it will run on a T4...` | `Pick your runtime: github's "linux-nvidia-t4" GPU instance ... or github's "ubuntu-latest" CPU instance...` | V2 explicitly requests `linux-nvidia-t4`; no CPU score is implied. |
| Competitive-or-innovative response | Pinned template has no such heading | `# is this submission competitive or innovative? explain why` | V2 answers the current heading narrowly and includes borrowed-substrate accounting. |
| Archive and inflater | `a download link to archive.zip` and `inflate.sh` | Same live README requirements | Hosted-manifest RED remains; exact AFR1 URL does not yet exist. |
| Evaluation budget | `The official evaluation has a time limit of 30 minutes.` | Same, with runner choice added | V2 states measured T4 component times and labels the residual CI window as a projection. |
| Counted large artifacts | `large artifacts ... should be included in the archive and will count towards the compressed size` | Same | No delta; AFR1 keeps learned/video-derived content in the counted archive. |
| Leaderboard head | Pinned snapshot predates current rows | Live README lists PR #135 at `0.162` | AFR1's competitive comparison remains same-axis `[contest-CUDA]`, never a CPU claim. |

## PR_BODY_DRAFT v2

New internal proposal:
`.omx/research/ddm_pq1_submission_packet_prep_20260815/PR_BODY_DRAFT_V2.md`.
The old draft is untouched.

V2 changes the decision-relevant surface rather than preserving the gen-6 body:

1. replaces every rc2 identity with AFR1: 180,002 B, archive SHA
   `cbb8d928…d405bf25`, runtime tree `6cdfa27d…`, exact component-derived
   `S=0.14797617125559104 [contest-CUDA T4 n600]`;
2. states that no AFR1 CPU row exists and carries the historical same-lineage
   predecessor `S=0.20513189128858372 [contest-CPU Linux x86_64 n600]` only as a
   non-transferable predecessor fact;
3. answers the live competitive/innovative template question in the narrow
   decision-layer scope and credits PR #130/#133/#135/#138;
4. states measured decode/eval timings, the projected residual ceiling, the
   network-install/native-compile limitation, and the non-reproducing
   `compress.py` boundary;
5. leaves the hosted URL honestly absent and requires fetched-back identity;
6. contains no Claude, Codex, Anthropic, LLM or AI attribution, no fleet/private
   address, no provider ledger material, and no local absolute path.

The file is an internal factual proposal. It is not policy-compliant public text
until the operator independently writes the final body in their own words.

## One-screen operator sign-off summary

- **Object:** AFR1 generation 7, 180,002 B, archive SHA `cbb8d928…d405bf25`,
  runtime tree `6cdfa27d…`, `[contest-CUDA T4 n600]`
  `S=0.14797617125559104` from components.
- **Axis:** no AFR1 CPU row. The `0.20513189128858372` CPU number is a historical
  predecessor only and does not transfer.
- **Lineage:** learned vehicle is PR #130/#135 with PR #133 transitively; the
  claim is the decision/lossless-representation layer. PR #138 has first-public
  mechanism-class credit; no priority claim.
- **Policy:** current packet history appears to match the banned all-code clause.
  Operator-authored text does not cure it. Safe choices are no contest PR, ask
  the maintainer first, or proceed only if the operator can truthfully establish
  compliance after reviewing the final diff/history.
- **Seven stored REDs:** unchanged. Hosting is still absent; no action is
  authorized by this memo.
- **Publication:** nothing was posted. The final body, every public comment,
  hosting authorization and the one-line publish authority remain operator-owned.

## Compliance delta rows

The retained strict checker result remains **80 GREEN / 7 RED of 87**. This arm
did not rewrite or re-run that receipt. The live policy adds two external
requirements that the stored checker did not adjudicate. For decision purposes,
the overlay is **80 GREEN / 9 RED across 89 rows**; this is a DERIVED view, not a
new checker run.

| Requirement | Source quote | Packet status | Action needed | Owner |
|---|---|---|---|---|
| Submitter writes/reads most code; no full-code delegation | `If you're not writing and reading most of the code you are submitting, then what's the point?!` and banned `write all of the code` | **RED / likely direct conflict** based on the packet's recorded end-to-end assistant authorship; operator must inspect final history and must not misstate it | Decide no-PR vs maintainer pre-clearance vs proceed only on truthfully demonstrated compliance | operator, task #1363 → #1111 |
| Full PR description and public comments are not delegated | banned `write full PR description and public facing comments` | **RED until operator writes them independently**; V2 is internal evidence only | Operator writes final body/comments in own words after reviewing sources | operator, task #1111 |
| Current template includes competitive/innovative answer | `# is this submission competitive or innovative? explain why` | GREEN in V2 | Preserve narrow same-axis and lineage-qualified answer | operator final-text review |
| Exact hosted archive URL | `a download link to archive.zip` | Existing RED, unchanged | Host only after authorization; fetch back and verify 180,002 B + full SHA | operator hosting gate |
| Explicit runtime choice | `Pick your runtime: ... "linux-nvidia-t4" ... or ... "ubuntu-latest"` | GREEN in V2 | Preserve T4 request; do not imply a CPU result | operator final-text review |

The original seven RED rows and their owners remain exactly as recorded in
`ddm_pq12_afr1_reswap_20260831.md`; this memo does not silently relabel any one
of them green.

## RECALL EVIDENCE

Full-corpus searches were run over `.omx/research/`, the canonical research
index, sub-0.15 DAG FEED blocks, canonical equations JSON, hot state, packet
receipts and task/P0 surfaces. Queries included `pq13|PR body|coding-agents|LLM
policy|Yousfi|#1111|#1363|#1156`, `PR #137|PR #138|PR #139`,
`borrowed-substrate|public hygiene`, and the exact AFR1/CPU score literals.

Beyond the charter seeds:

1. `ddm_pq7_pr_engineering_20260820` already held the 81-thread/76-comment
   maintainer census and the first live README/template diff. It changed v2 from
   a long narrative into the precise baseline/change/score shape.
2. `ddm_llm_policy_intake_20260831.md` held the close-event timing and proved
   #138's policy link was enforcement, not hypothetical guidance.
3. `ddm_fs1_fullstack_submission_description_20260831.md` resolved the current
   four-class lineage and the PR #138 no-priority boundary.
4. The generation-7 retained packet supplied current timings, 38-row receiver
   identity and the honest `compress.py` refusal boundary; the old PR draft body
   was gen-6 historical material despite its overlay.
5. No canonical equation specific to PR authorship/policy was found; the policy
   is a source-text constraint, not a score equation.

## Denominators and measurement boundary

- PR intake: **3/3** identified PRs (#137/#138/#139) typed; #139 source-limited.
- Updated-since-2026-08-10 maintainer census: **3 PR threads**, **1 maintainer
  comment captured**, **0 review comments captured**.
- Live requirement surfaces: **2/2** visible governing text files inspected
  (README and pull-request template); **1** policy section found.
- Draft hygiene target: **1/1** new draft scanned directly, not via README.
  `inspect_public_hygiene` reported **1 file, 0 hits** and passed both the
  nonempty-corpus and no-private-surface checks. A separate attribution/private-
  surface literal scan also returned **0 hits**.
- Packet checks: no scorer, inflate, archive mutation, hosted fetch, CI run or
  compliance-checker re-run occurred. All AFR1 numbers were consumed from the
  retained authority/freeze receipts and labeled by axis.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER — LIVE_CENSUS_REFRESH:** owner: MAIN GitHub reader; consumer store: an append-only pq13 source-limit addendum and task #1363 evidence; fire trigger: valid read authentication is restored or GitHub indexes PR #139, then fetch #137/#138/#139 bodies, issue comments, review comments and close events paginated and compare the exact denominator to this memo before #1111 can fire.
- **BLOCKED-ON-OPERATOR-POLICY-DECISION:** owner: operator; consumer store: task #1363 policy ruling consumed by task #1111; fire trigger: the operator reviews the final submitted-code history against both banned clauses and chooses no PR, maintainer pre-clearance, or a truthfully compliant submission path.
- **BLOCKED-ON-OPERATOR-CONFIRM-AND-TEXT:** owner: operator; consumer store: operator-authored public body/comments, hosted-manifest fetchback proof and publication receipt; fire trigger: policy compatibility is resolved, all seven stored RED dispositions are accepted, the exact AFR1 archive is hosted and fetched back byte-identically, and the operator gives the explicit one-line publication authorization.

## LIVE-HYPOTHESES

- **Windows portability may expose a cheap receiver-hardening lesson.** PR #139
  is a reproduction rather than a new codec, so any useful delta is likely in
  shell, compiler or dependency portability; this is plausible but untested
  until its body/diff is retrieved.
- **Maintainer pre-clearance may distinguish research publication from a job
  application submission.** The policy motivation mentions both fun and jobs,
  while the banned clauses are categorical. Asking before publishing is the
  only clean way to test whether an exception or alternate channel exists.

## DEAD-ENDS

- **Disclosure-only interpretation:** closed. The live policy contains explicit
  banned-use clauses; disclosure alone is not compliance.
- **Operator-written prose cures the entire policy conflict:** closed. It cures
  only the public-text clause and leaves code authorship unchanged.
- **Treating #137 as policy-closed from timing alone:** closed. The retained
  thread has no maintainer reason; only #138 has the verbatim policy link.
- **Folding #139 as a compression advance:** closed on available scope. Its
  title/score classify it as a reproduction, and no new method was retrievable.
- **Reusing the gen-6 draft body with a gen-7 banner:** closed. The embedded
  archive, runtime, timing and reproduction facts were stale; v2 was rebuilt
  from the generation-7 packet facts.

`[contest-CUDA T4 n600] own-vehicle frontier: AFR1 — S=0.14797617125559104, archive=180,002 B, d_seg=0.00020139, d_pose=6.37e-6, SHA-256=cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25; UNMOVED by this prepare-only arm.`

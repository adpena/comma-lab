# ddm_fr2 — final fresh-eyes adversarial review of the gen-8 PR packet (2026-09-03)

Arm: `ddm_fr2_final_fresh_eyes_pr_review` (Opus fallback; codex service down).
Review object: `experiments/results/ddm_fr2_final_review_20260903/pr_body_snapshot.md`
(operator's hand-edited text) + the frozen tree `submissions/semantic_joint_ctxmix/`
at commit `dba0d7a951`. Read-only. Nothing was published, hosted, or edited.

## Verdict

**The PRIOR-LAW PREDICTION IS FALSIFIED. Two BLOCKERs remain.** Both are
cross-consistency defects between the PR body and the frozen README, not
measurement errors. Every number I could bind to a receipt is correct — the
arithmetic re-derives bit-exactly, the manifest is clean 39/39, and the
strongest claim in the body (five changes, 454 bytes, identical decode) is
the best-supported one. The defects are in the prose layer that a maintainer
reads first.

Both BLOCKERs are cheap: one paragraph swap in the README, one noun in the body.

## The ranked table

| # | Grade | Finding | Evidence |
|---|---|---|---|
| R1 | **BLOCKER** | The shipping README still says its own CUDA entrypoint is unverified and that publication is blocked. The g8v1 T4 run closed that exact gap on 2026-09-03. Shipping it tells the maintainer the submitter does not trust the axis being claimed, and it contradicts PR body line 35. | `submissions/semantic_joint_ctxmix/README.md:73-85` vs `.omx/research/ddm_g8v1_gen8_tree_cuda_reproof_20260903.md:1-22` |
| R2 | **BLOCKER** | PR body says the edits are "solved across 573 pairs". A prior HIGH/NO-FAKE finding removed that exact noun and the README now says "455 of 573 **proposed edits**". The body reinstates the defect and contradicts the packet it ships with. | `.omx/research/ddm_g8r_compress_adversarial_review_20260902.md:52`; `submissions/semantic_joint_ctxmix/README.md:24-26`; body line 57 |
| R3 | RECOMMEND | "Twenty-three admitted improvement moves" is the headline number of the Innovative section and I could not find an enumerated 23-row ledger in the repo. Worse, `pq12` corrects the *same byte window* to **five** admitted states, calling 23 "the charter shorthand". | `.omx/research/ddm_pq12_afr1_reswap_20260831.md:36-44` vs `.omx/research/ddm_zdc1_zero_distortion_corner_reopens_20260831.md:27`, `.omx/state/main_hot_state.md:28` |
| R4 | RECOMMEND | "Yes" to *include the compression script and merge it* — but `compress.py` needs `base_archive.zip`, which is "deliberately not embedded", and neither the body nor the README says what it is or where to get it. A merger who runs it gets an immediate refusal. | `submissions/semantic_joint_ctxmix/compress.py:9-11, 1644-1645, 2196`; flagged earlier at `.omx/research/ddm_yr1_yousfi_adversarial_pr_review_20260902.md:18-21` |
| R5 | RECOMMEND | The body links `github.com/adpena/comma-lab` (commit `1c9fbbf5`) as the evidence of record, while the README says public visibility of that commit "has not been re-verified". If it is private, the central evidence link 404s in front of the maintainer. | `submissions/semantic_joint_ctxmix/README.md:69-71`; body lines 41, 53 |
| R6 | VERIFIED-OK | Score arithmetic re-derives **bit-exactly**: `100*0.00020139 + sqrt(10*0.00000637) + 25*180002/37545489 = 0.14797617125559104`. Components match the pointer. `180,002 − 179,902 = 100 B` is exactly right for one stored member named `p`. | recomputed this arm; `.omx/state/main_hot_state` POINTER_LINE; member verified at `.omx/research/ddm_pq14_drive_pr_review_round1_20260902.md:18` |
| R7 | VERIFIED-OK | PR #135 provenance labeling is honest. The intake memo labels the row `[external source-reported; leaderboard displays 0.162]` and records that the archive was never acquired. "Author-reported unrounded" is the correct label. | `.omx/research/ddm_pi135_pr135_intake_20260810.md:21,31,41` |
| R8 | VERIFIED-OK | Both priority dates are real. `src/tac/scorer_targets.py` first landed **2026-04-11** (`fea4a953f9`); `contour_codec.py` + `region_merge.py` + `partition.py` first landed **2026-06-10** (`752a30cdb9`). Ordering before PR #130 holds — our earliest PR #130 material is 2026-07-25. | `git log --diff-filter=A` this arm; `.omx/research/feedback_gc4_full_pantheon_pr130_adjudication_20260725.md` |
| R9 | VERIFIED-OK | The strongest claim in the body. Five stages (FX5, DX2, GB1, LB1, AFR1), −454 B, and "decoding to exactly the same output" all bind: identical 3,662,409,600-byte CUDA raw output, SHA `6bf8acf8…`. `compress.py`'s five stages match. | `.omx/research/ddm_pq12_afr1_reswap_20260831.md:36-48`; `compress.py:4-7` |
| R10 | NIT | The three known nits confirmed. Also: `MANIFEST.sha256` currently verifies 39/39 clean. | below |

## R1 — the stale README paragraph (BLOCKER)

README lines 73–78 read, in the tree that ships:

> "…its public CUDA entrypoint has not been rerun on a T4. The local identity
> result is not a cross-axis CUDA proof, so the current tree must receive a
> contest-CUDA identity/equivalence check before publication."

That check ran and passed. `ddm_g8v1` (2026-09-03): the gen-8 tree "ran its own
public entrypoint on a 1:1 Tesla T4 … and reproduced the afr1 row EXACTLY."
The 532.33 s + 40.57 s timings the PR body quotes at line 35 come from that very
run. So the README denies the thing the body asserts.

Line 84 compounds it: "Publication state: PREPARED HOLD, NOT PUBLISHED." A
published submission that says it is not published reads as a packaging slip.

**Suggested replacement for README lines 73–85** (operator's call; I did not edit):

> The score above belongs to the exact archive and the evaluated contest-CUDA
> receiver at that commit. This tree's own public entrypoint was re-run on a
> Tesla T4 against these archive bytes and reproduced the same row exactly:
> PoseNet 0.00000637, SegNet 0.00020139, 180,002 bytes, 532.3 s inflation +
> 40.6 s evaluation.
>
> `[contest-CPU]` is RECORD-WITH-REASON: no exact CPU score was run and no older
> CPU score is inherited. The prior contest-CPU attempt timed out at the
> 1,800-second inflation limit, so this package makes no CPU score claim.
>
> Corrections welcome — if any attribution is incomplete, tell me and I will fix it.

**Operational rider, do not skip.** `MANIFEST.sha256` is a 39-row allowlist and
`compress.py` refuses on any mismatch (g8r finding round 1). Any README edit
must be followed by a manifest rehash, exactly as commit `dba0d7a951` did. Edit
the README without rehashing and the shipped compressor refuses its own tree.

## R2 — "solved across 573 pairs" (BLOCKER)

Body line 57: "candidate segmentation edits … are **solved across 573 pairs** …
(455 of 573 admitted)."

The g8r review already adjudicated this at HIGH / NO-FAKE severity and recorded
the fix: "`455 of 573` now correctly names **proposed edits** rather than
'solved pairs.'" The README carries the corrected form. The body reverts it.

I am giving you the counter-evidence too, because the underlying receipts are
not clean: `ddm_fs1…:69` says "edited pairs | 573" and `:49` calls 455 "pairs".
So I cannot prove "573 pairs" is factually false. What I can prove is that the
body and the README describe the same mechanism two different ways, that a
maintainer can diff them in ten seconds, and that the wording the body chose is
the one a prior NO-FAKE finding deleted. With 600 pairs in the contest, "573
pairs" also invites the reading that the solve covered 573 of 600 pairs.

**Suggested replacement for the first clause of body line 57:**

> "candidate segmentation edits of the semantic tokens are proposed per pair and
> priced against their exact pose cost through the frozen PoseNet, then admitted
> through a Lagrange-multiplier waterfill (455 of 573 proposed edits admitted)."

## R3 — the twenty-three (RECOMMEND)

`pq12` is explicit: "The full pointer ledger, **rather than the charter
shorthand**, has five admitted states after rc2 … Total rc2→AFR1 change: −454 B."
`zdc1` and `main_hot_state` describe the same 453.6 B window as "twenty-three
pointer moves". Both cannot be the count for one window, and pq12 is the later,
ledger-grounded one.

Twenty-three may be right campaign-wide. I found no file that lists the
twenty-three. If a competitor asks for the list, we should be able to produce it.

**Suggested replacement for body line 55:**

> "**Innovative:** The decision and lossless-representation layer built on top of
> that vehicle is a campaign of pointer-moving improvements, each accepted only
> after re-scoring the rebuilt archive — five of them after the previous packet
> was frozen:"

This drops a number we cannot itemize and keeps the one we can. It also removes
the arithmetic tension with the "five further changes" bullet on line 60.

## R4 — the compression script cannot run for the recipient (RECOMMEND)

`compress.py:9-11`: "Place the pinned base beside this file as `base_archive.zip`
or pass `--base-archive PATH`. The base is an input and is **deliberately not
embedded** in this tree." The refusal path at `:1644` fires on a missing file the
packet neither contains nor identifies. No base SHA appears in the body or the
README — the pinned SHA both documents mention is the *output* archive's.

The maintainer's question is literally about merging this script. Answering "Yes"
is true but incomplete, and yr1 predicted this surface on 2026-09-02.

**Suggested sentence to append to body line 39:**

> "The pinned base archive is an input rather than part of the packet, so
> `compress.py` refuses without it; I will attach it or publish its SHA-256 on
> request."

## R5 — the evidence link (RECOMMEND)

Both the body and the README route all mechanism-level evidence to
`github.com/adpena/comma-lab` at commit `1c9fbbf5`. The README itself records
that public visibility of that revision was never re-verified. Confirm the
repository and that commit are publicly reachable before the confirm. This is a
one-minute pre-publish action, not a rewrite.

## R10 — nits, confirmed

- Line 51: double closing paren after `0.16226842169958583))` — drop one.
- Line 53: "which in turn **builds**" has a plural subject (PR #130 and PR #135) —
  "which in turn **build** on PR #133".
- Line 62: `4/11/2026` and `6/10/2026` are ambiguous outside the US. Both dates
  are correct as US format; git confirms **2026-04-11** and **2026-06-10**. Use
  ISO, which also reads as more precise: "was committed 2026-04-11" and
  "was committed 2026-06-10".
- `MANIFEST.sha256` verifies 39/39 clean at `dba0d7a951`. No action.

## What I did not re-file

The two known compliance survivors — `[contest-CPU]` RECORD-WITH-REASON and the
hosted-manifest operator gate — are documented and out of scope per the charter.
The disclosure sentence at line 76 is the operator's own wording and I make no
recommendation to remove it; as written it is direct and it names the tools
plainly, which is the right posture under the coding-agents policy.

## Custody

Every number above was checked by opening the named file, not by trusting a
label. The one figure I could not bind to a repo receipt is PR #130's
publication date; the "before PR #130" ordering is supported by our own earliest
PR #130 material (2026-07-25) sitting well after both priority commits, which is
sufficient for the claim but is an inference, not a receipt.

No scorer ran. No archive was mutated. No dispatch fired. Nothing was published.

# ddm_pv1 — provenance, lineage, citation and originality audit of the whole public write-up

**Date:** 2026-08-17
**Axis:** `[macOS-CPU advisory / documentation + source-hygiene audit]`.
`score_claim: false`, `promotable: false`. **No Modal, no dispatch, no scorer run, no eval.
Spend: $0.00.** No submission, no publication, no GitHub write. `gh` read-only. `upstream/`
untouched.

---

## ANSWER

**The packet's borrowing disclosure was honest but not accurate, and its most important citation
was missing.** Nothing in it was inflated in our favour by design — the errors run in both
directions, and the single genuine over-claim is 100 bytes wide. But four section labels were
wrong, one originality label had no receipt at all, two mechanisms of ours went unmentioned, and
the packet did not cite a pull request that published our one claimed mechanism class **six hours
before our first measured result**.

**The decisive finding is PR #138.** `opal_v1` (Cristian, `ccastillo1043`) opened
2026-08-17T08:31:32Z describing an online decode-side probability correction learned from the
already-decoded prefix, adding zero archive bytes, yielding pure rate. That is our contribution's
mechanism class. Our packet cited PR #138's *score* and never its *mechanism*. A maintainer
reading both would have found it in a minute, and our silence would have read as concealment. Our
own design work is dated 2026-07-22 in-repository, twenty-six days earlier, so the honest label is
**concurrent independent development with no priority claim** — which is now what the packet says.

**The prediction in my charter was half right.** It predicted a stale, e480b-era accounting doc.
That is wrong: sr1's F8 already measured the document's *content* as complete and honest, and I
confirm that — its category discipline and its "what we do not claim" section are the strongest
writing in the packet. What was actually wrong is subtler and was invisible to a content review:
the doc used **one label for two different relations** (identical-to-PR130/135 versus
identical-to-the-base-we-inherited), and the base is not PR135's archive — it already carries our
retrained HPAC object and our compensation edits. The prediction that "≥2 originality statements
need a lineage qualifier" was correct; there were five.

---

## 1. STORES CONSULTED

**Charter and standing law:** `.omx/research/charters/ddm_pv1_provenance_lineage_citation_audit_20260817.md`;
`CLAUDE.md` (NO-FAKE #7, Innovation Gate, public disclosure hygiene, L14–L32 demotion banner);
`docs/operating_manual_craft_handoff.md`.

**The surfaces under audit:** `.omx/research/ddm_pq1_submission_packet_prep_20260815/`
(`PR_BODY_DRAFT.md`, `README_PUBLIC.md`, `BORROWED_SUBSTRATE_ACCOUNTING.md`, `REPORT_PUBLIC.txt`,
`CONTRIBUTION_ETIQUETTE.md`, `ARCHIVE_MANIFEST.json`);
`/Volumes/APDataStore/pact/ddm_pq2/submission_staging/{README.md, report.txt}` and
`../competitive_statement.txt` (read-only); the repository root `README.md`;
`docs/paper/04_results.md`.

**Prior review:** `.omx/research/ddm_sr1_submission_gauntlet_20260817.md` (commit `27cd767531`) —
16 findings; this arm consumed F1, F2, F9, F13, F14, F15, F16 and re-verified B1.

**Lineage receipts, read at source:** `pr86_pr130_fullstack_intake_20260728.md:45-52`;
`pr130_eureka_intake_acquisition_20260806.md:7-12`; `ddm_pi135_pr135_intake_20260810.md:11,23,29,41`;
`ddm_cp135_rate_compose_20260810.md:41-45,58`; `ddm_fd135_fractal_decomposition_20260810.md`;
`ddm_hx1_pr_wave_harvest_20260817.md:18-24,57-62,73-84,155,161,180-190,327,381-383`;
`ddm_hv1_harvest_compose_ep508_20260815.md:9-16,92-97,217`;
`ddm_rx2_mc36_label_hpac_20260814.md:12-14`; `ddm_mc36_promotion_complete_s_verdict_20260814.md:3-6`;
`ddm_rr2_encoder_byteclose_20260817.md:124-125`; `ddm_rr4_cuda_prob_reencode_20260817.md`;
`ddm_rc64p_native_cpu_decode_20260810.md:80-81`; `.omx/research/falsified_premise_registry.jsonl`
(premise `qs2_re1_bank_union_is_held_and_unfired_20260817`); `.omx/state/canonical_frontier_pointer.json`.

**Independence anchors:** `ddm_g4_spatial_stationarity_603_DAG_FEED_20260722T212138Z.md` (commit
`915e87dce3`); `ddm_c1_composed_candidate_spec_603_613_20260723.md:62`;
`ddm_r7_token_coder_race_20260729.md:124`; commits `fdf3298801`, `c8e6ee416c`, `f7e29a124c`,
`f1de91eb46`.

**Verified live at source (`gh`, read-only, no writes):** PRs #102, #130, #133, #135, #136, #138 —
number, title, author, state; the open-PR list; `adpena/comma-lab` and `adpena/tac` visibility.

---

## 2. PR metadata — every citation verified at source

Nothing below was taken from our notes. A wrong attribution is worse than a missing one.

| PR | Title (verbatim) | Author | State | Opened |
|---|---|---|---|---|
| #130 | `semantic-pose-HPAC_CPR1` | `fesalfayed` — Fesal Fayed | CLOSED | 2026-07-17T21:22:20Z |
| #133 | `cpr1_cbq_matched8 submission (0.1658)` | `JasonMo123` | CLOSED | 2026-08-01T15:23:13Z |
| #135 | `semantic-pose-HPAC_CPR1_polished (0.162)` | `codexblack` — Shreyan Mohanty | CLOSED | 2026-08-06T03:00:11Z |
| #136 | `hnerv_rc: 0.19258 (CPU axis) — adaptive range coder on the hnerv_muon pipeline` | `JPL11` — Jacky Li | CLOSED | 2026-08-07T21:37:17Z |
| #138 | `opal_v1` | `ccastillo1043` — Cristian | **OPEN** | 2026-08-17T08:31:32Z |
| #102 | `hnerv_lc_v2_scale095_rplus1 submission (0.19538 CPU)` | `EthanYangTW` — MIN-CHUN (ETHAN) YANG | MERGED | (root README cite) |

**Two useful consequences.** PR #135's own title carries `(0.162)`, so the competitive comparison
(sr1 F6) is checkable from the PR title alone, without trusting our leaderboard snapshot. And
`adpena/comma-lab` and `adpena/tac` both return `PUBLIC` — sr1's B1 blocker is resolved at source.

---

## 3. The lineage skeleton, verified

| Step | Archive bytes | Class |
|---|---|---|
| PR #130 (Fesal Fayed) | 191,052 | inherited substrate — architecture **and** bytes **and** the full 49-stage training repo were taken |
| PR #135 (Shreyan Mohanty), itself carrying PR #133's constrained basis and re-solved int12 carrier | 186,724 | inherited substrate — **the archive we actually built on** |
| cp135 lossless recompose | 186,252 | ours |
| mc36 Variant C — the qs2 ∪ re1 micro-edit union, promoted on T4 | 186,269 | **ours** |
| e480b → hv1, HPAC checkpoint ep0634 | 182,759 | **ours** (PR130 architecture, our labels) |
| rr2 → **rr4**, the free decode-time corrector | **181,161** | **ours** |

**The critical question — is the shipped learned content theirs or ours? MIXED, and the split is
section-exact.** The semantic renderer (36,051 B, `b489c735…`) and the pose carrier (22,242 B,
`196f0e51…`) are PR #135's, proven byte-identical by `ddm_cp135_rate_compose_20260810.md:41`. The
HPAC probability object (17,952 B, `e8c0cfd7…`) is **ours** — PR130's architecture retrained here
on our own MC36 label field, checkpoint ep0634 chosen from 81 retained candidates by
`tools/select_hpac_checkpoint.py`. The token stream is our lossless re-encode of their
probabilities. `PR130/135-byte-identical` was therefore the wrong label on three of the rows it
was applied to, because those rows are identical to **our base**, not to their archive.

---

## 4. Per-claim verdicts and what I changed

26 claims audited across the four surfaces. **15 fixed, 6 flagged, 5 clean.**

### Fixed (applied directly)

| # | Claim, as it stood | Verdict | What it says now |
|---|---|---|---|
| 1 | "HPAC probability object … is PR130/PR135 lineage, **is not ours**" (PR body) | **WRONG — understated** | named as ours: PR130's architecture, retrained here on our label field |
| 2 | HPAC row `PR130-lineage, inherited unchanged here` | NEEDS-LINEAGE-QUALIFIER | `mechanism-adopt-with-attribution`, with the ep0634 receipt |
| 3 | Compensation blob `PR130/135-byte-identical` | **OVERSTATES THE BORROW** | container inherited, contents include our admitted qs2 ∪ re1 edits, pairs listed |
| 4 | Compressed model container `PR130/135-byte-identical` | NEEDS-QUALIFIER | unchanged from base; PR-level equality not independently verified |
| 5 | Residual payload + table codes **`ours-original`** | **OVER-CLAIM — no receipt anywhere** | **withdrawn**; `inherited-substrate`, provenance unresolved, no originality claimed |
| 6 | "the innovation here is narrow" with no mechanism citation | **OVER-CLAIM BY OMISSION** | PR #138 cited as first publisher of the mechanism class; concurrent independent development; explicit "no priority claim" |
| 7 | no credits section anywhere in the PR body | GAP | **Credits and prior work** block: PRs #130, #135, #133, #138, #136, upstream, third-party deps |
| 8 | category set violated — body used `PR130-lineage` / `PR135-byte-identical`, absent from the doc's own closed set | INCONSISTENCY | one closed 4-class set, used identically in both documents |
| 9 | body and accounting disagreed on 4 rows with no way to tell which was right | INCONSISTENCY | accounting carries a "what the PR body says" column; disagreements are now visible by construction |
| 10 | `README:69` pointed at an unstaged file (sr1 F1) | DANGLING | "shipped in this directory", plus the inline-table fallback and a one-line summary |
| 11 | "The public source pin must have its anonymous visibility verified" (sr1 B1) | STALE | both revisions labelled, visibility recorded as verified |
| 12 | Dependency closure named only Brotli (sr1 F15) | INCOMPLETE | all four declared, including the **unguarded C compiler**, stated as unguarded |
| 13 | no archive placement instruction (sr1 F13) | GAP | one line: download the release asset into this directory first |
| 14 | `CP135`/`F26` codenames and a dead homebrew branch in `inflate.sh` (sr1 F14/F16) | HOSTILE-READ HAZARD | hash-safe cure: a README section explaining both; the tree is untouched |
| 15 | 7 hardcoded local-path defaults in the two offered stage scripts (sr1 F9) | DISCLOSURE RISK | **0 remain** — see §5 |

### Flagged — not closable by this arm

| # | Item | Why it is open | Owner |
|---|---|---|---|
| A | **Residual payload provenance.** Shipped is 100 B `74775aab…` (RCF1 framing); PR135's is 100 B `bd27a2dd…`; hv1 carries `bd27a2dd…` unchanged. Either a re-framing of their content or a genuine re-fit. | No receipt settles it. Claim withdrawn in the conservative direction; a receipt could restore it. | MAIN |
| B | Compressed model container: no PR-level sha comparison exists. | Classified against the base only. | MAIN |
| C | `inflate.sh` in-tree cures for F14/F16 and the F15 `command -v cc` guard. | Inside the hashed runtime tree; editing breaks the custody binding to the measured T4 row. | freeze-time / next candidate |
| D | `/Volumes/APDataStore/pact/ddm_pq2/competitive_statement.txt` still carries the pre-PR-138 wording. | Outside the repo, on the ExFAT staging volume where any edit re-creates an AppleDouble file (sr1 F4). | MAIN, at freeze |
| E | `docs/paper/04_results.md` best ranked row is 0.2089/0.2094; the live candidate is 0.15853. The root README points a judge there. | Understates us, so it is not a NO-FAKE issue — but the public face is stale. Adding a score row is an adjudication, not a citation fix. | MAIN |
| F | `ddm_rr4_cuda_prob_reencode_20260817.md:95` cites `ddm_rr2_receiver_close.py:57-58`; my §5 edit shifted those lines. | Memos are append-only historical provenance; I did not mutate it. | noted only |

### Clean — audited, no change needed

The competitive claim itself (true, and now checkable from PR #135's own title); the
"no training cost" claim; the determinism wording (sr1 F5 already cured to "records"); the
`d_seg`/`d_pose`-unchanged framing, which correctly refuses to present an unchanged number as an
achievement; and the root README's PR #102 CPU-vs-CUDA claim, which I verified against PR #102's
own title (`0.19538 CPU`).

---

## 5. sr1 F9 closed — the offered stage scripts carry no local layout

`experiments/ddm_pq2_compress_e2e.py` was already clean (0 hits). The two stage scripts the PR body
names were not: `ddm_rr2_encoder_byteclose.py` had 5 hardcoded `/Volumes/…` defaults (one of which
also exposed a PR135 intake tree) and `ddm_rr2_receiver_close.py` had 2, one not overridable at all.

Both now resolve every input root from the environment with **no default to fall back on**, and
refuse fail-closed naming each missing variable:

```
refusing to run: required input roots are not set.
  TAC_PQ2_PREPARED_DIR -- directory holding the prepared base candidate (archive.zip + runtime)
  TAC_PQ2_HM1_DIR -- directory holding the retained base logits, boundary bucket, and group index
  TAC_PQ2_TOKENS_FILE -- file holding the decoded 600-frame token field
  TAC_PQ2_RC64_SOURCE -- RC64 range-coder C source (inherited substrate; see the accounting table)
```

Design notes, because two of them are load-bearing. The refusal fires **after** `argparse` so
`--help` still works on a judge-facing script, and **before** any stage touches disk. `--store` is
now `required=True` rather than defaulting to a local path; all three invocations in
`ddm_pq2_compress_e2e.py` already pass it, and that entry point is the only caller.

Verified: `ruff` clean · 0 local-path hits in both files · `--help` exits 0 with no environment
set · a real stage with no environment exits **1** naming all four variables · `--store` plumbing
intact. Two review-tracker passes, the second an adversarial re-read of the diff. Landed
`a3452f5aaf`.

---

## 6. NO-attribution sweep — clean

Zero occurrences of Claude, Anthropic, assistant, copilot, GPT, LLM, or a co-author trailer across
all nine documents and the repository root README. The only regex hits were `comma.ai` matching a
case-insensitive `ai`. `CONTRIBUTION_ETIQUETTE.md:30`'s standing assertion that "the packet carries
no machine attribution" is **verified true**. Local-path and private-infrastructure sweep across
the same set: clean.

---

## 7. Freeze-time swap list — hashed-tree items

These cannot be fixed without changing the evaluated runtime-tree hash `7acedb07e670e76c…` and
breaking replay against the measured T4 row. Each has a hash-safe cure applied in the README, and a
proposed in-tree replacement for the next candidate.

| File:line | Item | Hash-safe cure (applied) | Proposed replacement (next candidate) |
|---|---|---|---|
| `inflate.sh:24` | `CP135 requires Brotli==1.2.0, but uv is unavailable` — internal codename in a judge-visible error | README "Two names a reader will meet" section | drop the codename: `this receiver requires Brotli==1.2.0, but uv is unavailable` |
| `inflate.sh:41-52` | dead `Darwin`/`brew --prefix libomp` branch | same README section states it is unreachable and that the submission assumes Linux | delete the branch, or mark it `# optional local acceleration; unreachable on the contest runner` |
| `inflate.sh:32` | C compiler invoked unguarded | README Dependency closure now declares it, and declares it unguarded | `command -v cc >/dev/null \|\| { echo "this receiver needs a C toolchain (cc)"; exit 69; }` — same fail-closed standard as Brotli |
| `GENERATION_RECEIPT.json`, `RECEIVER_PARSEBACK.json` | stale archive identity + 63 absolute paths | README already corrected the identity; this arm added the local-path disclosure | emit receipts with relative paths, per the standing fix-forward |

---

## 8. VERDICT

The packet may now state its originality without a maintainer being able to find a borrowing it
does not disclose, or a mechanism citation it does not make. The one change that mattered most
costs us nothing except a priority claim we were never entitled to: **PR #138 published this
mechanism class first, and we say so.** The one genuine over-claim — an `ours-original` label on a
100-byte residual with no receipt behind it — is withdrawn rather than defended.

Two directions of error is what an honest audit looks like. We were overstating the borrow on the
compensation blob and understating our own HPAC retrain, at the same time, for the same reason: one
label was doing two jobs.

## 9. What I did NOT do

No submission, no pull request, no release, no hosting, no publication, no GitHub write of any
kind — `gh` was used read-only for PR metadata and repository visibility. No Modal or paid
dispatch. No scorer run, no eval, no n600. No edits to `upstream/`. No edits inside the hashed
runtime tree, and no edits to any file on the staging volume — every staged-text correction is
applied to the repository-side source of truth so MAIN copies it at freeze, which also avoids the
ExFAT AppleDouble hazard. No memo was mutated; historical provenance is append-only.

---

**Own-vehicle frontier: S 0.15853325034789678 @ 181,161 B `[contest-CUDA T4, n600]`. This unit did
not move it — a provenance audit corrects what the packet claims, it does not lower the score.**

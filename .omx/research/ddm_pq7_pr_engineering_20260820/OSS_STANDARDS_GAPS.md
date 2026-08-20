# OSS + comma standards gaps — packet layer

`date_utc: 2026-08-20` · `owner: ddm_pq7` · `score_claim: false` · `frontier_moved: false`

**Own-vehicle frontier: S = 0.14839100138338618 @ 180,625 B `[contest-CUDA T4, n600]`.** Unmoved here.

**⛔ COLLISION HELD.** `PR_BODY_DRAFT.md`, `README_PUBLIC.md` and `REPORT_PUBLIC.txt` all have
**uncommitted working-tree modifications** at the time of writing (`git status --porcelain`), on top
of pq5's `4226017206`. A sister arm is mid-edit. **I edited none of the three.** Every fix below
that touches them is written as a patch proposal for MAIN, not applied.

---

# BLOCKING — before publish

## B1. The coding-agents and LLMs policy — our pinned README predates it

**This is the finding that can close the PR on sight, and nothing else on this list matters if it
is unresolved.**

The live contest `README.md` carries a section our pinned `upstream/README.md` (2026-04-13) does not
have at all. Verified by diffing the live file against the pin:

> **banned uses** — "write all of the code" · "write full PR description and public facing comments"
> "Any violation of this policy will result in a closed PR, repeated violations will result in a ban."

Four PRs were closed on 2026-08-07 with nothing but a link to it (#133, #134, #135, #136).

**Our packet was produced agent-end-to-end.** `PR_BODY_DRAFT.md` was drafted by agents; so was this
file. That is a direct match to the second banned use.

**The remediation is receipted, not speculative.** #135 was closed under this policy, re-described,
and accepted onto the leaderboard the next day. What he asked for:

> "you have to show that there was some human work" · "We don't need more verbose, we need more precise"

and the literal body shape:

> "**THIS** is the baseline submission and score… **THIS** file/function/etc was changed to do
> **THIS** instead and achieved score… Optionally: **THIS** didn't work better… Optionally:
> **THIS** is my llm setup and prompts…"

**Owed to the operator, not to an arm:** (a) the PR description must be written by the operator, not
generated; (b) the LLM setup should be disclosed — the policy's own optional bullet invites it, and
disclosure is the difference between compliance and concealment; (c) the "most of the code" test
needs an honest answer. **No arm can clear this.** It is listed first because publishing without
resolving it risks a ban, not merely a rejection.

## B2. The PR body's PR #138 independence sentence is true but incomplete

Adjudicated against receipts (chronology from `git log --date=iso`, cross-checked to
`BORROWED_SUBSTRATE_ACCOUNTING.md:155-170`). **Verdict: INDEPENDENT.** PR #138 `opal_v1` opened
2026-08-17T08:31:32Z; our first measured corrector result (`rr1`, `fdf3298801`) is the same day at
14:41:58Z and byte-closed (`rr2`, `f7e29a124c`) at 16:04:57Z; we **first read** PR #138 at 19:32:52Z
(`f1de91eb46`). The mechanism class is anchored in-repo 26 days earlier (`915e87dce3`, 2026-07-22).
The mechanisms also differ: PR #138 does a rank-one split and *preserves the complement's relative
law exactly*; `ma1` models that complement.

**Two precisions the current sentence owes:**

1. **It names the wrong lineage level.** The shipped `runtime/free_corrector.py` is `ma1`'s
   `Ma1WithinMissCorrector`, tip of `rr2→rr4→fx1→fx2→ma1`. The independence receipts cover
   rr1/rr2/rr4. `pq4/CONTRIBUTIONS_INVENTORY.md:73` still attributes at the rr2 level.
2. **`fx1`, `ma1`'s direct parent, did read our harvest of PR #138.** `hx1` landed 08-17T20:24Z;
   `fx1` landed 42 minutes later and cites it in STORES CONSULTED, using PR #138's `exp()`
   state-sync warning as a design check — then solved it by exact rounding rather than their
   quantisation, and explicitly declined the other harvested item. `ma1` itself cites none of it.

**PATCH PROPOSAL for `PR_BODY_DRAFT.md` — replace the independence sentence with:**

> **A decode-time probability corrector on the miss class**, online and decode-identical. The
> shipped `runtime/free_corrector.py` is `ma1`'s `Ma1WithinMissCorrector`, the tip of an
> `rr2→rr4→fx1→fx2→ma1` lineage. PR #138 `opal_v1` published this mechanism class first, on
> 2026-08-17T08:31Z; our first measured result is the same day at 14:41Z and we first read that PR
> at 19:32Z, so the base corrector is concurrent independent work and we claim no priority.
> PR #138's correction stops at the rank-one split and preserves the complement's relative law
> exactly; `ma1` models that complement, which is where our −105 B comes from. One later refinement
> (`fx1`) did read our harvest of PR #138 and used its `exp()` state-sync warning as a design check,
> solving it by exact rounding rather than by their quantisation.

The final clause is the one currently missing. It costs nothing and removes the only line a
maintainer could read as concealment. **BLOCKING because it is NO-FAKE #7 territory**, not because
the claim is false.

## B3. No LICENSE and no THIRD_PARTY_NOTICES in the packet — and we vendor modified third-party code

Every merged neural submission ships both (`rhnerv_comma`, `hnerv_fec6_fixed_huffman_k16` carry
`LICENSE` **and** `THIRD_PARTY_NOTICES.md`). We ship neither, while shipping **five modified files
derived from PR #130** (`DUPLICATION_AUDIT.md` §1). That makes the notices obligation heavier for
us than for them, not lighter.

**MEASURED and free:** `LICENSE` (no suffix) and `THIRD_PARTY_NOTICES.md` (`.md`) are **not** in
`_RUNTIME_DEPENDENCY_SUFFIXES`, so staging them **cannot move `runtime_tree_sha256`**. The repo
already has both at root (`LICENSE`, MIT, 1,079 B; `THIRD_PARTY_NOTICES.md`, 7,485 B, itself
written "in the spirit of the comma.ai openpilot project's third-party notices conventions").

**Fix: copy both into the packet at freeze. Zero hash cost, zero score cost.** I did not stage them
because staging is MAIN's boundary.

---

# SHOULD — before publish, high value

## S1. Restructure the PR body to the #135 four-bullet shape

He asked for it explicitly and rejected verbosity twice. Our body is long-form prose. The
baseline→file/function→score→what-didn't-work shape is what he said he reads. This also pairs with
B1: a precise, human-written body is the compliance artifact.

## S2. Declare the runtime explicitly — the rule changed in our favour

The live README now reads **"Pick your runtime: github's `linux-nvidia-t4` GPU instance … or
github's `ubuntu-latest` CPU instance"**, replacing the old "if your inflation script requires a
GPU, it will run on a T4". `FREEZE_CHECKLIST` item (b) treats routing as depending on the maintainer
selecting the runner; **the live rule says the submitter picks.** The PR body should state
`linux-nvidia-t4` plainly.

**Supporting precedent, measured:** the eval bot ran **#130 and #133 on `device=cuda`,
`num_threads=2`** — our body's own lineage. This materially de-risks variant (b) and is worth
recording against the checklist.

## S3. No `expected_output.sha256` / `MANIFEST.sha256`

The merged convention (`rhnerv_comma`, `hnerv_fec6`) is to ship a decode-determinism hash; PR #130
ships `MANIFEST.sha256` (995 B) **and** `verification.json` (9,820 B). We ship `archive_manifest.json`
at 664 B and no per-file manifest. **Also free:** `.sha256` is not in the suffix set. Recommend
shipping a per-file `MANIFEST.sha256` over the 33 pinned rows — which doubles as the artifact that
makes B4's subset rule checkable.

## S4. Record the tree-hash subset rule explicitly

`report.txt` (`.txt`) and `archive_manifest.json` (`.json`) **are** manifest-eligible by suffix, yet
are absent from the 33 rows — because the 33 were measured on the Modal host's `submission_dir`,
which never contained them. **So the staged packet already differs from the evaluated tree by two
manifest-eligible files.** pq3's proof was sound precisely because it re-derived the hash *from the
33 enumerated rows*, not from a fresh directory walk.

That subset rule is therefore **already operative and undocumented**. Write it into the freeze
receipt. Without it, the next arm to re-validate reads a mismatch as corruption — and this is the
precondition that makes staging `compress.py` safe (`packet_staging/STAGING_PLAN.md`).

## S5. No test ships anywhere in the packet

PR #130 ships `test_carrier_codec.py` (10,537 B); #110 ships a `tests/` directory. We ship zero.
`FREEZE_CHECKLIST` (f)(1) also names `tools/stage_contest_submission_packet.py` as untested and
load-bearing for the identity proof, and `pq4` names the compression script as untested.
**Not free** — a `tests/*.py` moves the tree hash — so this belongs at the rr2-native-port rebuild
boundary, where a re-seal is already being paid for.

## S6. README quickstart is honest but not runnable

The packet README's Reproduction section correctly says the rebuild entry point "has **not** been
re-run for these bytes". Good. What is missing is what a reviewer on a fresh clone *can* run in five
minutes: verify the archive sha and size, list the single stored member `p`, and run `inflate.sh`.
Add a three-command block with expected output and an approximate runtime. **`.md` — free.**

---

# NICE

- **N1. Per-file lineage headers on the five vendored `cpr1/` modules.** A reader opening
  `cpr1/hpac_integer.py` cannot tell it derives from PR #130. PR #130 solved this with SPDX headers
  (`rhnerv_comma/frame_selector.py:1`). `.py` — rebuild boundary.
- **N2. Compression-script path leak in output.** The refusal prints the frontier pointer's absolute
  path. Reachable only inside the research repo, so it cannot leak from the packet. Fix the repo
  file; do not fork a packet variant.
- **N3. `requirements` are stated in prose, not pinned.** `inflate.py` imports `torch`; the PR body
  names the dependencies. A pinned list would match `rhnerv_comma`'s posture.

---

# What I fixed vs proposed

| Item | State |
|---|---|
| B1 policy | **Proposed** — operator-only; no arm can clear it |
| B2 PR #138 sentence | **Proposed** — collision hold on `PR_BODY_DRAFT.md` |
| B3 LICENSE + notices | **Proposed** — staging is MAIN's boundary; verified free |
| S1–S6, N1–N3 | **Proposed** |
| Packet dead/gratuitous files | **Nothing to fix** — audited, zero found |

**I applied no fixes to the three prose files, by design.** All three had uncommitted sister edits
in flight. Writing over them would have destroyed another arm's work — the exact failure the
collision rule exists to prevent. Everything above is written so MAIN can apply it in one pass.

**Nothing here is a submission action.** No push, no hosting, no PR.

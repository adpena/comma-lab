# ddm_oc2 — branch and worktree disposition

`date_utc: 2026-08-20` · `owner: ddm_oc2` · authority: operator consolidation directive 2026-08-20
(push authorization for **our own** repos only; the contest PR is untouched and still gates on the
operator's one-line confirm, #1111).

## 0. The finding that frames everything

**`origin` for this repo is `git@github.com:adpena/comma-lab.git`, and that repository is PUBLIC.**
The pact research repo and the public comma-lab repo are the same repository. There is no second
clone to prepare. Everything committed to `main` here is published on the next push.

Baseline hygiene state, measured on tracked content before this arm committed anything:

| Check | Tracked baseline | Verdict |
|---|---:|---|
| Files containing `/Users/adpena` | 4,226 | pre-existing; not created by this arm |
| Files containing `/Volumes/` | 5,242 | pre-existing; not created by this arm |
| Tailscale / `100.64.0.0/10` IPs | **0** | clean, and stayed 0 |
| Credentials / API tokens / private keys | **0** | clean, and stayed 0 |

The absolute-path exposure is a real standing gap against CLAUDE.md's Public Disclosure Hygiene
rule ("local absolute paths ... out of GitHub/docs/site/public surfaces"). It is a **pre-existing
condition at scale**, not something this consolidation introduced, and it is owed to MAIN as a
separate decision. This arm did not widen the class knowingly: every file it committed was scanned
for secrets and fleet IPs and came back clean.

## 1. Branch disposition

Seven branches were unmerged against `main`. Verdicts below are from `git merge-tree` dry runs plus
a content check of whether `main` already carries the branch's files.

| Branch | Head | Ahead | Disposition | Why |
|---|---|---:|---|---|
| `ddm/sh1_integration_20260727` | `6a77427ca1` | 10 | **MERGED** | Clean merge, 39 files / 7,637 insertions. It is an integration branch that had already merged the three siblings below, so one merge discharges four branches. |
| `codexwt/ddm_cb1_perclass_carrier_byteclose_20260725T203310Z` | `2721704ab2` | 1 | **MERGED via sh1** | sh1 commit `3963938cc8` merged it cleanly. |
| `codexwt/ddm_wf7_seven_home_stream_waterfill_20260725T203257Z` | `e3c2140d3a` | 2 | **MERGED via sh1** | sh1 commit `8dd643c03a` merged it cleanly. |
| `codexwt/ddm_pf3b_52probe_joint_improving_hunt_20260725T202800Z` | `074955c6ad` | 2 | **MERGED via sh1** | sh1 commit `0bb5b9579c` merged it. |
| `codexwt/ddm_de1_20260803T112347Z` | `7a0d6f0abc` | 2 | **MERGED** | Clean; a 783-line description-efficiency derivation absent from `main`. See the supersession note below. |
| `ddm/dy1_scope_law_resolver` | `a9eac92166` | 1 | **CERTIFIED — SUPERSEDED, do not merge** | Content is already on `main` and `main` is a strict superset. Merging would **delete 93 lines** from `src/tac/witness_dsl/scope_laws.py` (branch 2 insertions / 93 deletions vs main). Add/add conflicts on its research md confirm main landed the work by another path. |
| `codexwt/einstein_kolmogorov_crux_20260719T212159Z` | `81dfb0ee68` | 1 | **CERTIFIED — SUPERSEDED, do not merge** | Same shape, larger stakes. Merging would **delete 3,253 lines** from `src/tac/preflight.py`. `main` already carries `src/tac/canonical_equations/einstein_kolmogorov_crux_20260719.py` (68,061 B) and `.omx/research/ddm_ek2_worktree_harvest_20260810/DISPOSITION.md` (11,813 B). The branch's own commit message predicted this: it recorded merge-to-main as owed because "main drifted 2+ wks on preflight.py/CLAUDE.md". The drift resolved in main's favour. |

### de1 supersession note

The de1 derivation is merged for its research value. One claim inside it is **stale**: at
`.omx/research/ddm_de1_description_efficiency_derivation_20260803.md:611,705,773` it argues that
`src/tac/boundary_math/contour_codec.py` carries a name/mechanism mismatch. That specific complaint
was addressed by the #913 rename after de1 was written. The derivation is merged as a dated
historical artifact; the contour_codec paragraphs are not current and must not be cited as a live
defect.

### Recommended cleanup for MAIN

`dy1` and `einstein_kolmogorov` should be **deleted or archived as refs**, not left unmerged. Left
alone they read as "owed work" forever, and the next consolidation arm will re-derive this same
superseded verdict. That is the rediscovery cost the anti-forgetfulness rule exists to prevent.

## 2. Worktrees

54 worktrees were swept by `ddm_wk2r` on 2026-08-03
(`.omx/research/ddm_wk2r_worktree_custody_20260803.md`, denominator 54/54). This arm re-enumerated
them: the branch heads are unchanged and every non-main worktree is still `CLEAN` or carries only
the residue wk2r already recorded. **No live-arm worktree was touched.**
`.omx/state/active_lane_dispatch_claims.md` shows the newest claims all in terminal
`completed_*` states, so no worktree was in active use during this pass.

The one worktree that mattered is `agent-ae029aa6d20642139`, which holds
`ddm/sh1_integration_20260727` — merged above, so it can now be pruned.

## 3. The residue wk2r flagged as `RESIDUE-needs-owner` on main

wk2r counted 167 uncommitted entries in the main worktree on 2026-08-03. By this arm's pass that
had grown to 280 porcelain entries / 1,049 files. **764 documents and 181 sources were committed**;
the remainder is bulk that belongs on the SSD tier by the disk rules. See
`WORKING_TREE_DISPOSITION.md` for the itemized split and the certify records.

## 4. The three sh1 files the review gate held back — BLOCKED, preserved losslessly

`sh1_held_deltas_BLOCKED_ON_FIXES.patch` (19,028 B, sha256
`c2e488b255dd31f5994d1e7ea838af1e0c9245546b20b284383f17de4e417999`, 470 lines) carries the additive
sh1 deltas for:

- `src/tac/optimization/ddm_runtime_receiver.py` (+329)
- `src/tac/optimization/tests/test_direct_description_carrier_compose.py` (+36)
- `tools/materialize_ddm_pf3_finite_prices.py` (+9 / −5)

They are **not committed**. A real review pass (`REVIEW_FINDINGS_OWED.md`) found 1 critical and 5
important correctness bugs in these files, so the review gate refuses them and the three files are
marked `needs_fix` in the review tracker. `REVIEW_GATE_OVERRIDE` was **not** used — CLAUDE.md
forbids it for `.py` and the gate is right here.

**Nothing is lost and main is not degraded.** All three files already exist on `main` in working
form, and all six bugs pre-exist there, so `main` is exactly as correct as it was. The patch is the
custody record for the 374 added lines.

**Owed to MAIN:** fix the six findings, then apply this patch. Note the incident in §5 — because
`a886ddb340` already attached sh1 as a second parent, git considers sh1 fully merged and will never
re-offer these deltas. This patch is the only path by which they can still land.

## 5. INCIDENT — a merge left open outside the serializer lock

While this arm held `git merge --no-ff --no-commit ddm/sh1_integration_20260727` open and stepped
away to do other work, arm `ddm_rr6` ran its own commit. Git attached **this arm's `MERGE_HEAD`**
(sh1, `6a77427ca1`) as the **second parent** of `a886ddb340` while committing only ddm_rr6's own
three files.

Consequences, measured:

- `git merge-base --is-ancestor 6a77427ca1 main` → **YES**. main's DAG says sh1 is merged.
- `git diff a886ddb340^1 a886ddb340` → **3 files, 340 insertions** — ddm_rr6's own work only.
- sh1's 7,637 insertions were left sitting **in the index**, in the tree of no commit.

Had the index been reset, that content would have been silently lost *while the DAG claimed it had
landed* — the worst shape of this failure. It was recovered by committing the staged set explicitly
(36 files) plus this patch (3 files).

**Cause: this arm's error**, not ddm_rr6's. Leaving a merge open across other tool calls is exactly
the concurrent-index race `tools/subagent_commit_serializer.py` exists to prevent; the serializer
only protects the `git add` + `git commit` window, and an open `MERGE_HEAD` sits outside it. The de1
merge that followed was performed merge-and-commit **atomically in a single invocation**.

**Cure worth building (named, not built):** the serializer should refuse to start, or warn loudly,
when `MERGE_HEAD` exists but the caller did not declare a merge — and a merge helper should wrap
`git merge --no-commit` + serializer commit as one locked operation so the open-merge window cannot
be left unattended.
